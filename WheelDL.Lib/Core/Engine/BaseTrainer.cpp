#include "pch.h"
#include "BaseTrainer.h"
#include "../../Optimizer/Scheduler/CosineAnnealingLR.h"
#include "../../Optimizer/Scheduler/LinearLR.h"
#include "../../Utils/Common/Constants.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Memory/GPUMemoryGuard.h"

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            BaseTrainer::BaseTrainer(const std::shared_ptr<Config::Configuration> config)
                : _config(config)
                , _logger(Utils::Logger::getInstance())
                , _profiler(Utils::PerformanceProfiler::getInstance())
                , _memoryManager(Utils::MemoryManager::getInstance())
                , _device(torch::kCPU)
                , _useAMP(false)
                , _currentEpoch(0)
                , _totalEpochs(0)
                , _bestFitness(-1.0f)
                , _shouldStop(false)
            {
                _logger->info("BaseTrainer", "Initializing trainer with configuration");

                // Initialize timers
                _epochTimer = std::make_unique<Utils::Timer>();
                _totalTimer = std::make_unique<Utils::Timer>();

                // Initialize Random with seed from config
                _random = std::make_unique<Utils::Random>(config->getSeed());

                // Initialize callback throttler (100ms default interval)
                _callbackThrottler = std::make_unique<Callback::CallbackThrottler>(100);
				_currentMetrics = MetricsData();

            }

            BaseTrainer::~BaseTrainer()
            {
                // Stop async callback queue if running
                if (_asyncCallbackQueue) {
                    _asyncCallbackQueue->stop();
                }

                _logger->info("BaseTrainer", "Trainer destroyed");
            }

            void BaseTrainer::setProgressCallback(ProgressCallback callback)
            {
                _progressCallback = callback;

                if (callback) {
                    // Create async callback queue
                    _asyncCallbackQueue = std::make_unique<Callback::AsyncCallbackQueue>(callback);
                    _asyncCallbackQueue->start();
                    _logger->info("BaseTrainer", "Async callback queue started");
                }
            }

            void BaseTrainer::stop()
            {
                _shouldStop = true;
                _logger->warn("BaseTrainer", "Training stop requested");
            }

            MetricsData BaseTrainer::train()
            {
                _profiler.start("total_training");
                _totalTimer->reset();

                _logger->info("BaseTrainer", "========== Starting Training ==========");
                _logger->info("BaseTrainer", "Epochs: " + std::to_string(_totalEpochs));
                _logger->info("BaseTrainer", "Batch Size: " + std::to_string(_config->getBatchSize()));
                _logger->info("BaseTrainer", "Device: " + _config->getDevice());

                try 
                {
                    // ========== Setup Phase ==========
                    _profiler.start("setup");

                    initializeSeeds();
                    setupDevice();
                    setupSaveDirectory();
                    setupModel();           // Hook
                    setupDataLoaders();     // Hook
                    setupValidator();       // Hook
                    setupOptimizer();
                    setupScheduler();
                    setupAMP();
                    initializeEMA();
                    initializeEarlyStopping();
                    freezeLayersIfNeeded();
                    _totalEpochs = _config->getEpochs();

                    _profiler.stop("setup");
                    _logger->info("BaseTrainer", "Setup completed in " +
                                 std::to_string(_profiler.getStatistics("setup").total) + "ms");

                    // ========== Training Loop ==========
                    for (int epoch = 0; epoch < _totalEpochs; ++epoch) 
                    {
                        if (_shouldStop) 
                        {
                            _logger->warn("BaseTrainer", "Training stopped by user at epoch " +
                                         std::to_string(epoch));
                            break;
                        }

                        _currentEpoch = epoch;
                        _epochTimer->reset();

                        // Train epoch
                        _logger->info("BaseTrainer", "---------- Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(_totalEpochs) + " ----------");
                        trainEpoch(epoch);

                        // Validate epoch
                        if (_validator)
                        {
                            // Use EMA model for validation if available
                            if (_ema && _model)
                            {
                                auto originalSeq = _model->swapModelSequential(_ema->getEMAModel());
                                _currentMetrics = _validator->validate(*_model, epoch);
                                _model->swapModelSequential(originalSeq);
                            }
                            else
                            {
                                // Validate with regular model
                                _currentMetrics = _validator->validate(*_model, epoch);
                            }
                        }
                        else
                        {
                            _logger->warn("BaseTrainer", "Validator not initialized, skipping validation");
                        }

                        schedulerStep();
                        float currentFitness = calculateFitness(_currentMetrics);
                        saveCheckpoint(epoch, currentFitness, false);

                        // Save best checkpoint
                        bool isBest = currentFitness > _bestFitness;
                        if (isBest) {
                            _bestFitness = currentFitness;
                            saveCheckpoint(epoch, currentFitness, true);
                        }

                        // Early stopping check
                        if (_earlyStopping && _earlyStopping->shouldStop(currentFitness)) {
                            _logger->info("BaseTrainer", "Early stopping triggered at epoch " +
                                         std::to_string(epoch + 1));
                            break;
                        }

                        // Epoch callback
                        double epochTime = _epochTimer->elapsedSeconds();
                        double totalTime = _totalTimer->elapsedSeconds();
                        double eta = (totalTime / (epoch + 1)) * (_totalEpochs - epoch - 1);

                        ProgressData progressData;
                        progressData.stage = ProgressStage::TRAIN_EPOCH;
                        progressData.currentEpoch = epoch + 1;
                        progressData.totalEpochs = _totalEpochs;
                        progressData.loss = _currentMetrics.loss;
                        progressData.metrics = _currentMetrics;
                        progressData.learningRate = _scheduler ? _scheduler->getCurrentLR() : _config->getLearningRate();
                        progressData.gpuMemoryUsage = _memoryManager.getGPUMemoryUsagePercent();
                        progressData.elapsedTime = totalTime;
                        progressData.eta = eta;

                        invokeProgressCallback(progressData);

                        _logger->info("BaseTrainer", "Epoch completed in " +
                                     std::to_string(epochTime) + "s | ETA: " +
                                     std::to_string(eta / 60.0) + "m");
                    }

                    // ========== Training Completed ==========

					// ========== Last Validation ==========
					_profiler.start("final_validation");

                    if (_validator)
                    {
                        _logger->info("BaseTrainer", "Performing final validation with best model");
                        if (_ema && _model)
                        {
                            auto originalSeq = _model->swapModelSequential(_ema->getEMAModel());
                            _currentMetrics = _validator->validate(*_model, _currentEpoch);
                            _model->swapModelSequential(originalSeq);
                        }
                        else
                        {
                            _currentMetrics = _validator->validate(*_model, _currentEpoch);
                        }
                        _logger->info("BaseTrainer", "Final validation metrics - Loss: " +
                                     std::to_string(_currentMetrics.loss));

                        // Update best fitness and save checkpoint if this is the best so far
                        float finalFitness = calculateFitness(_currentMetrics);
                        if (finalFitness > _bestFitness) {
                            _bestFitness = finalFitness;
                            saveCheckpoint(_currentEpoch, finalFitness, true);
                            _logger->info("BaseTrainer", "Final validation improved best fitness: " +
                                         std::to_string(_bestFitness));
                        }
						saveCheckpoint(_currentEpoch, finalFitness, false);
                    }
                    else
                    {
                        _logger->warn("BaseTrainer", "Validator not initialized, skipping final validation");
					}

                    _profiler.stop("total_training");
                    double totalTime = _totalTimer->elapsedSeconds();

                    _logger->info("BaseTrainer", "========== Training Completed ==========");
                    _logger->info("BaseTrainer", "Total time: " + std::to_string(totalTime / 60.0) + " minutes");
                    _logger->info("BaseTrainer", "Best fitness: " + std::to_string(_bestFitness));

					_profiler.stop("final_validation");
                    // Export profiler report
                    if (_workspace) 
                    {
                        std::string profilerPath = _workspace->getProfilerDir() + "/profiler_report.html";
                        _profiler.exportToHTML(profilerPath);
                        _logger->info("BaseTrainer", "Profiler report saved: " + profilerPath);
                    }

                    return _currentMetrics;
                }
                catch (const std::exception& e) {
                    _logger->error("BaseTrainer", "Training failed: " + std::string(e.what()));
                    throw;
                }
            }

            void BaseTrainer::initializeSeeds()
            {
                _profiler.start("initialize_seeds");

                int seed = _config->getSeed();
                _logger->info("BaseTrainer", "Setting seed: " + std::to_string(seed));

                // Set torch seed
                torch::manual_seed(seed);
                if (torch::cuda::is_available()) 
                {
                    torch::cuda::manual_seed_all(seed);
                }

                // Set deterministic if requested
                if (_config->isDeterministic()) {
                    // LibTorch deterministic settings
                    torch::globalContext().setDeterministicCuDNN(true);
                    torch::globalContext().setBenchmarkCuDNN(false);
                    _logger->info("BaseTrainer", "Deterministic mode enabled (CuDNN)");
                }

                _profiler.stop("initialize_seeds");
            }

            void BaseTrainer::setupDevice()
            {
                _profiler.start("setup_device");

                std::string deviceStr = _config->getDevice();

                if (deviceStr.empty()) 
                {
                    // Auto-detect
                    _device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
                    _logger->info("BaseTrainer", std::string("Auto-detected device: ") + (_device.is_cuda() ? "CUDA" : "CPU"));
                } else if (deviceStr == "cpu") 
                {
                    _device = torch::kCPU;
                    _logger->info("BaseTrainer", "Using CPU");
                } else 
                {
                    // CUDA device
                    _device = torch::Device(torch::kCUDA, 0); // Default to GPU 0
                    _logger->info("BaseTrainer", "Using CUDA device: " + deviceStr);
                }

                // Log GPU memory if CUDA
                if (_device.is_cuda()) {
                    float totalMem = _memoryManager.getGPUMemoryTotal();
                    float usedMem = _memoryManager.getGPUMemoryUsed();
                    _logger->info("BaseTrainer", "GPU Memory: " +
                                 std::to_string(usedMem) + "MB / " +
                                 std::to_string(totalMem) + "MB");
                }

                _profiler.stop("setup_device");
            }

            void BaseTrainer::setupSaveDirectory()
            {
                _profiler.start("setup_save_directory");

                // Create workspace using Workspace utility
                _workspace = std::make_unique<Utils::Workspace>(
                    Constants::DefaultPaths::RUNS_DIR,
                    "train"
                );

                _logger->info("BaseTrainer", "Workspace created: " + _workspace->getRoot());
                _logger->info("BaseTrainer", "Weights directory: " + _workspace->getWeightsDir());

                _profiler.stop("setup_save_directory");
            }

            void BaseTrainer::freezeLayersIfNeeded()
            {
                // TODO: Implement layer freezing based on config
                // This would require additional config parameters
            }

            void BaseTrainer::setupAMP()
            {
                _profiler.start("setup_amp");

                _useAMP = _config->useAMP();

                if (_useAMP && _device.is_cuda())
                {
                    // TODO: Implement AMP support (LibTorch C++ does not have native AMP like Python PyTorch)
                    // - Custom autocast implementation
                    // - Custom GradScaler implementation
                    _logger->info("BaseTrainer", "AMP requested (implementation pending)");
                }
                else if (_useAMP && !_device.is_cuda())
                {
                    _logger->warn("BaseTrainer", "AMP requested but CUDA not available, disabling AMP");
                    _useAMP = false;
                }

                _profiler.stop("setup_amp");
            }

            void BaseTrainer::setupOptimizer()
            {
                _profiler.start("setup_optimizer");

                if (!_model) {
                    throw Utils::WheelLibException(Utils::ErrorCode::TRAINING_FAILED,
                        "Model must be initialized before optimizer");
                }

                auto parameters = _model->parameters();
                _optimizer = Optimizer::OptimizerFactory::createFromConfig(parameters, *_config);

                _logger->info("BaseTrainer", "Optimizer created: " + _config->getOptimizer());
                _logger->info("BaseTrainer", "Initial LR: " + std::to_string(_config->getLearningRate()));

                _profiler.stop("setup_optimizer");
            }

            void BaseTrainer::setupScheduler()
            {
                _profiler.start("setup_scheduler");

                if (!_optimizer) {
                    throw Utils::WheelLibException(Utils::ErrorCode::TRAINING_FAILED,
                        "Optimizer must be initialized before scheduler");
                }

                if (_config->useCosineLR()) {
                    _scheduler = std::make_unique<Optimizer::Scheduler::CosineAnnealingLR>(
                        _optimizer.get(), *_config);
                    _logger->info("BaseTrainer", "Using CosineAnnealingLR scheduler");
                } else {
                    _scheduler = std::make_unique<Optimizer::Scheduler::LinearLR>(
                        _optimizer.get(), *_config);
                    _logger->info("BaseTrainer", "Using LinearLR scheduler");
                }

                _profiler.stop("setup_scheduler");
            }

            void BaseTrainer::initializeEMA()
            {
                if (_config->getTaskType() == TaskType::ANOMALY)
                {
                    _logger->info("BaseTrainer", "EMA is only supported for detection tasks, skipping EMA initialization");
                    return;
                }

                if (!_model)
                {
                    throw Utils::WheelLibException(Utils::ErrorCode::TRAINING_FAILED,
                        "Model must be initialized before EMA");
                }
                
                _profiler.start("initialize_ema");

                _ema = std::make_unique<Optimizer::EMA::ModelEMA>(
                    *_model,
                    Constants::DEFAULT_EMA_DECAY
                );

                _logger->info("BaseTrainer", "EMA initialized with decay: " +
                             std::to_string(Constants::DEFAULT_EMA_DECAY));

                _profiler.stop("initialize_ema");
            }

            void BaseTrainer::initializeEarlyStopping()
            {
                _profiler.start("initialize_early_stopping");

                int patience = _config->getPatience();
                _earlyStopping = std::make_unique<Optimizer::EarlyStopping::EarlyStopping>(patience);

                _logger->info("BaseTrainer", "Early stopping initialized with patience: " +
                             std::to_string(patience));

                _profiler.stop("initialize_early_stopping");
            }


            void BaseTrainer::trainBatch(const WheelDL::Data::Dataset::DataExample& batch, int batchIdx)
            {
                // Preprocess batch (hook)
                auto processedBatch = preprocessBatch(batch);

                // Move to device
                processedBatch.toDevice(_device);

                _optimizer->zero_grad();

                // Forward pass
                // TODO: Add AMP autocast here if _useAMP is enabled
                auto lossDict = _model->forward(processedBatch);
                auto totalLoss = lossDict["total"];

                // Backward
                // TODO: Add GradScaler backward/step if _useAMP is enabled
                totalLoss.backward();
                optimizerStep();

                // Update EMA after each batch
                if (_ema && _model)
                {
                    _ema->update(*_model);
                }
            }

            void BaseTrainer::trainEpoch(int epoch)
            {
                // RAII guard: automatically clears GPU cache when epoch ends
                Utils::GPUMemoryGuard gpuGuard;

                _profiler.start("train_epoch");

                if (!_model) {
                    throw Utils::WheelLibException(Utils::ErrorCode::TRAINING_FAILED,
                        "Model not initialized");
                }

                if (!_trainBatchIterator) {
                    throw Utils::WheelLibException(Utils::ErrorCode::TRAINING_FAILED,
                        "Train data loader not initialized. Call setupDataLoaders() first.");
                }

                // Set model to training mode
                _model->train();

                float epochLoss = 0.0f;
                int batchCount = 0;

                // Iterate over training batches using the type-erased iterator
                _trainBatchIterator([this, &epochLoss, &batchCount](const WheelDL::Data::Dataset::DataExample& batch, int batchIdx) {
                    trainBatch(batch, batchIdx);
                    batchCount++;

                    // Accumulate loss for epoch average
                    // Note: Loss is stored in model's internal state or we need to track it separately
                    // For now, we'll compute metrics at the end
                });

                _logger->info("BaseTrainer", "Training epoch " + std::to_string(epoch + 1) +
                             " completed with " + std::to_string(batchCount) + " batches");

                _profiler.stop("train_epoch");
            }

            void BaseTrainer::saveCheckpoint(int epoch, float fitness, bool isBest)
            {
                _profiler.start("save_checkpoint");

                if (!_model || !_optimizer || !_workspace) 
                {
                    _logger->warn("BaseTrainer", "Cannot save checkpoint - model, optimizer, or workspace not initialized");
                    _profiler.stop("save_checkpoint");
                    return;
                }

                // Create metadata from configuration
                auto metadata = CheckpointMetadata::fromConfiguration(*_config, epoch, fitness);

                // Determine filename: best.pt or last.pt
                std::string filename = isBest ? "best.pt" : "last.pt";
                std::string path = _workspace->getWeightsDir() + "/" + filename;

                // Save with EMA model if available
                if (_ema && _model)
                {
                    // Swap to EMA Sequential
                    auto originalSeq = _model->swapModelSequential(_ema->getEMAModel());
                    // Save checkpoint with EMA weights
                    Checkpoint::save(path, *_model, *_optimizer, metadata);
                    // Restore original Sequential
                    _model->swapModelSequential(originalSeq);

                    _logger->info("BaseTrainer", "Checkpoint saved with EMA weights: " + filename);
                }
                else
                {
                    // Save regular model
                    Checkpoint::save(path, *_model, *_optimizer, metadata);
                    _logger->info("BaseTrainer", "Checkpoint saved: " + filename);
                }

                _profiler.stop("save_checkpoint");
            }

            void BaseTrainer::optimizerStep()
            {
                // Gradient clipping could be added here
                _optimizer->step();
            }

            void BaseTrainer::schedulerStep()
            {
                if (_scheduler) 
                {
                    _scheduler->step(_currentEpoch);
                }
            }

            void BaseTrainer::invokeProgressCallback(const ProgressData& data)
            {
                if (!_asyncCallbackQueue) return;

                // Check throttler
                if (_callbackThrottler->shouldInvoke()) 
                {
                    _asyncCallbackQueue->enqueue(data);
                    _callbackThrottler->recordInvocation();
                }
            }

            std::string BaseTrainer::getSaveDirectory() const
            {
                return _workspace ? _workspace->getRoot() : "";
            }

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
