#include "pch.h"
#include "BaseTrainer.h"
#include "../../Optimizer/Scheduler/CosineAnnealingLR.h"
#include "../../Optimizer/Scheduler/LinearLR.h"
#include "../../Utils/Common/Constants.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Memory/GPUMemoryGuard.h"
#include "../../Utils/Export/MetricsExporter.h"
#include "../Engine/BasePredictor.h"

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            BaseTrainer::BaseTrainer(
                std::shared_ptr<Config::Configuration> config,
                WheelDL::Utils::Logger* logger,
                WheelDL::Utils::Workspace* workspace,
                WheelDL::Utils::PerformanceProfiler* profiler,
                ProgressCallback progressCallback,
                std::atomic<bool>* stopFlag)
                : _config(config)
                , _logger(logger)
                , _workspace(workspace)
                , _profiler(profiler)
                , _memoryManager(WheelDL::Utils::MemoryManager::getInstance())
                , _device(torch::kCPU)
                , _useAMP(false)
                , _currentEpoch(0)
                , _totalEpochs(0)
                , _bestFitness(-1.0f)
                , _stopFlag(stopFlag)
                , _checkpointPath("")
                , _progressCallback(progressCallback)
            {
                _logger->info("BaseTrainer", "Initializing trainer with configuration");

                // Initialize timers
                _epochTimer = std::make_unique<WheelDL::Utils::Timer>();
                _totalTimer = std::make_unique<WheelDL::Utils::Timer>();

                // Initialize Random with seed from config
                _random = std::make_unique<WheelDL::Utils::Random>(config->getSeed());

                // Initialize callback throttler (100ms default interval)
                _callbackThrottler = std::make_unique<Callback::CallbackThrottler>(100);
				_currentMetrics = MetricsData();

                // Setup async callback queue if callback provided
                if (_progressCallback) {
                    _asyncCallbackQueue = std::make_unique<Callback::AsyncCallbackQueue>(_progressCallback, _logger);
                    _asyncCallbackQueue->start();
                    _logger->info("BaseTrainer", "Async callback queue started");
                }
            }

            BaseTrainer::~BaseTrainer()
            {
                // Stop async callback queue if running
                if (_asyncCallbackQueue) {
                    _asyncCallbackQueue->stop();
                }

                _logger->info("BaseTrainer", "Trainer destroyed");
            }

            MetricsData BaseTrainer::train()
            {
                _profiler->start("total_training");
                _totalTimer->reset();

                _logger->info("BaseTrainer", "========== Starting Training ==========");
                _logger->info("BaseTrainer", "Epochs: " + std::to_string(_config->getEpochs()));
                _logger->info("BaseTrainer", "Device: " + _config->getDevice());

                try
                {
                    // ========== Setup Phase ==========
                    _profiler->start("setup");

                    initializeSeeds();
                    setupDevice();
                    setupSaveDirectory();
                    setupModel();
                    if (!_checkpointPath.empty())
                    {
                        loadCheckpoint(_checkpointPath);
                    }
                    setupDataLoaders();
                    setupValidator();
                    setupOptimizer();
                    setupScheduler();
                    setupAMP();
                    initializeEMA();
                    initializeEarlyStopping();
                    freezeLayersIfNeeded();

                    _totalEpochs = _config->getEpochs();

                    _profiler->stop("setup");
                    _logger->info("BaseTrainer", "Setup completed in " +
                        std::to_string(_profiler->getStatistics("setup").total) + "ms");

                    // ========== Training Loop ==========
                    for (int epoch = _currentEpoch; epoch < _totalEpochs; ++epoch)
                    {
                        if (isStopRequested())
                        {
                            _logger->warn("BaseTrainer", "Training stopped by user at epoch " + std::to_string(epoch));
                            throw WheelDL::Utils::StopRequestedException();
                        }

                        _currentEpoch = epoch;
                        _epochTimer->reset();

                        _logger->info("BaseTrainer", "---------- Epoch " + std::to_string(epoch + 1) +
                            "/" + std::to_string(_totalEpochs) + " ----------");

                        trainEpoch(epoch);

                        if (_validator)
                        {
                            _currentMetrics = runValidation(epoch, true);
                        }
                        else 
                        {
                            _logger->warn("BaseTrainer", "Skipping validation (No validator)");
                        }

                        schedulerStep();
                        float currentFitness = calculateFitness(_currentMetrics);
						_currentMetrics.fitness = currentFitness;
                        _epochMetrics.push_back(_currentMetrics);

                        // 4. Save Checkpoints
                        saveCheckpoint(epoch, false); // Save Last

                        if (currentFitness > _bestFitness) 
                        {
                            _bestFitness = currentFitness;
                            saveCheckpoint(epoch, true); // Save Best
                        }

                        // 5. Check Early Stopping
                        if (_earlyStopping && _earlyStopping->shouldStop(currentFitness)) {
                            _logger->info("BaseTrainer", "Early stopping triggered at epoch " + std::to_string(epoch + 1));
                            break;
                        }

                        // 6. Report Progress
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

                        _logger->info("BaseTrainer", "Epoch finished: " + std::to_string(epochTime) +
                            "s | ETA: " + std::to_string(eta / 60.0) + "m");
                    }


                    _profiler->stop("total_training");

                    saveTrainingResults();
                    finalValidation();      
                    finalPrediction();      
                    
                    _logger->info("BaseTrainer", "========== Training Completed ==========");
                    _logger->info("BaseTrainer", "Total time: " + std::to_string(_totalTimer->elapsedSeconds() / 60.0) + "m");
                    _logger->info("BaseTrainer", "Best fitness: " + std::to_string(_bestFitness));

                    return _currentMetrics;
                }
                catch (const WheelDL::Utils::StopRequestedException&) {
                    // Stop requested - don't save results, just log and re-throw
                    _profiler->stop("total_training");
                    _logger->warn("BaseTrainer", "Training stopped by user request - results not saved");
                    throw;
                }
                catch (const std::exception& e) {
                    _logger->error("BaseTrainer", "Training failed: " + std::string(e.what()));
                    throw;
                }
            }

            void BaseTrainer::initializeSeeds()
            {
                _profiler->start("initialize_seeds");

                int seed = _config->getSeed();
                _logger->info("BaseTrainer", "Setting seed: " + std::to_string(seed));

                // Set torch seed
                torch::manual_seed(seed);
                if (torch::cuda::is_available()) 
                {
                    torch::cuda::manual_seed_all(seed);
                }

                // Set deterministic if requested
                if (_config->isDeterministic()) 
                {
                    // LibTorch deterministic settings
                    torch::globalContext().setDeterministicCuDNN(true);
                    torch::globalContext().setBenchmarkCuDNN(false);
                    _logger->info("BaseTrainer", "Deterministic mode enabled (CuDNN)");
                }

                _profiler->stop("initialize_seeds");
            }

            void BaseTrainer::setupDevice()
            {
                _profiler->start("setup_device");

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

                _profiler->stop("setup_device");
            }

            void BaseTrainer::setupSaveDirectory()
            {
                _profiler->start("setup_save_directory");

                // Workspace is injected via constructor (owned by Launcher)
                // Just log the paths for verification
                _logger->info("BaseTrainer", "Workspace root: " + _workspace->getRoot());
                _logger->info("BaseTrainer", "Weights directory: " + _workspace->getWeightsDir());

                _profiler->stop("setup_save_directory");
            }

            void BaseTrainer::freezeLayersIfNeeded()
            {
                int freezeLayers = _config->getFreezeLayers();
                if (freezeLayers <= 0) {
                    return;
                }

                _profiler->start("freeze_layers");
                _logger->info("BaseTrainer", "Freezing first " + std::to_string(freezeLayers) + " layers");

                auto params = _model->getModel()->named_parameters();
                int count = 0;
                int frozenCount = 0;

                for (auto& param : params) {
                    if (count < freezeLayers) {
                        param.value().set_requires_grad(false);
                        frozenCount++;
                        _logger->debug("BaseTrainer", "Frozen: " + param.key());
                    }
                    count++;
                }

                _logger->info("BaseTrainer", "Frozen " + std::to_string(frozenCount) + " parameters out of " + std::to_string(count));
                _profiler->stop("freeze_layers");
            }

            void BaseTrainer::setupAMP()
            {
                _profiler->start("setup_amp");

                bool ampRequested = _config->useAMP();

                if (ampRequested) {
                    // LibTorch C++ does not have native AMP (torch.cuda.amp) like Python PyTorch.
                    // AMP in Python uses autocast context manager and GradScaler, which are not
                    // available in the C++ API. Manual half-precision implementation would require:
                    // - Converting model to half precision (model->to(torch::kHalf))
                    // - Custom gradient scaling logic
                    // - Careful handling of loss scaling to avoid underflow
                    //
                    // For now, we disable AMP and log a warning to avoid silent failures.
                    _logger->warn("BaseTrainer",
                        "AMP (Automatic Mixed Precision) is not supported in LibTorch C++. "
                        "Disabling AMP and using full precision (FP32) training.");
                    _useAMP = false;
                } else {
                    _useAMP = false;
                }

                _profiler->stop("setup_amp");
            }

            void BaseTrainer::setupOptimizer()
            {
                if (_config->IsPatchCore())
                {
                    _logger->info("BaseTrainer", "No optimizer configured, skipping optimizer setup");
                    return;
                }

                _profiler->start("setup_optimizer");

                if (!_model) {
                    throw WheelDL::Utils::WheelLibException(WheelDL::Utils::ErrorCode::TRAINING_FAILED,
                        "Model must be initialized before optimizer");
                }

                // Create optimizer with parameter grouping (Python YOLO style)
                _optimizer = Optimizer::OptimizerFactory::createFromConfig(*_model, *_config);
				_logger->info("BaseTrainer", "Optimizer created: " + _config->getOptimizer());
                _profiler->stop("setup_optimizer");
            }

            void BaseTrainer::setupScheduler()
            {
                _profiler->start("setup_scheduler");

                if (!_optimizer) 
                {
					return;
                }

                if (_config->useCosineLR()) {
                    _scheduler = std::make_unique<Optimizer::Scheduler::CosineAnnealingLR>(
                        _optimizer.get(), *_config);
                    _logger->info("BaseTrainer", "Using CosineAnnealingLR scheduler");
                } 
				else if (_config->useLinearLR())
                {
                    _scheduler = std::make_unique<Optimizer::Scheduler::LinearLR>(
                        _optimizer.get(), *_config);
                    _logger->info("BaseTrainer", "Using LinearLR scheduler");
                }
                else
                {
					_logger->info("BaseTrainer", "No learning rate scheduler configured");
                }

                _profiler->stop("setup_scheduler");
            }

            void BaseTrainer::initializeEMA()
            {
                // Skip EMA for anomaly detection (PatchCore doesn't need EMA)
                if (_config->IsPatchCore())
                {
                    _logger->info("BaseTrainer", "Skipping EMA initialization for PatchCore");
                    return;
                }

                if (!_config->isEmaEnabled())
                {
                    _logger->info("BaseTrainer", "EMA not enabled in configuration");
                    return;
				}

                if (!_model)
                {
                    throw WheelDL::Utils::WheelLibException(WheelDL::Utils::ErrorCode::TRAINING_FAILED,
                        "Model must be initialized before EMA");
                }

                _profiler->start("initialize_ema");

                _ema = std::make_unique<Optimizer::EMA::ModelEMA>(
                    *_model,
                    Constants::DEFAULT_EMA_DECAY
                );

                _logger->info("BaseTrainer", "EMA initialized with decay: " +
                             std::to_string(Constants::DEFAULT_EMA_DECAY));

                _profiler->stop("initialize_ema");
            }

            void BaseTrainer::initializeEarlyStopping()
            {
                _profiler->start("initialize_early_stopping");

                int patience = _config->getPatience();
                _earlyStopping = std::make_unique<Optimizer::EarlyStopping::EarlyStopping>(patience);

                _logger->info("BaseTrainer", "Early stopping initialized with patience: " +
                             std::to_string(patience));

                _profiler->stop("initialize_early_stopping");
            }


            void BaseTrainer::trainBatch(const WheelDL::Data::Dataset::DataExample& batch, int batchIdx)
            {
                // Preprocess batch (hook)
                auto processedBatch = preprocessBatch(batch);

                // Move to device
                processedBatch.toDevice(_device);

                // Forward pass
                // TODO: Add AMP autocast here if _useAMP is enabled
                auto lossDict = _model->forward(processedBatch);
                auto totalLoss = lossDict["total"];

                // Backward
                // TODO: Add GradScaler backward/step if _useAMP is enabled
                totalLoss.backward();

                // Optimizer step (includes gradient clipping and zero_grad)
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
                WheelDL::Utils::GPUMemoryGuard gpuGuard;

                _profiler->start("train_epoch");

                if (!_model) {
                    throw WheelDL::Utils::WheelLibException(WheelDL::Utils::ErrorCode::TRAINING_FAILED,
                        "Model not initialized");
                }

                if (!_trainBatchIterator) {
                    throw WheelDL::Utils::WheelLibException(WheelDL::Utils::ErrorCode::TRAINING_FAILED,
                        "Train data loader not initialized. Call setupDataLoaders() first.");
                }

                // Set model to training mode
                _model->train();

                float epochLoss = 0.0f;
                int batchCount = 0;

                // Iterate over training batches using the type-erased iterator
                try {
                    _trainBatchIterator([this, &epochLoss, &batchCount](const WheelDL::Data::Dataset::DataExample& batch, int batchIdx) {
                        // Check stop flag at iteration level
                        if (isStopRequested()) {
                            _logger->info("BaseTrainer", "Stop requested, aborting training epoch");
                            throw WheelDL::Utils::StopRequestedException();
                        }

                        trainBatch(batch, batchIdx);
                        batchCount++;

                        // Accumulate loss for epoch average
                        // Note: Loss is stored in model's internal state or we need to track it separately
                        // For now, we'll compute metrics at the end
                    });
                }
                catch (const WheelDL::Utils::StopRequestedException&) {
                    // Stop requested - re-throw to let train() handle
                    _profiler->stop("train_epoch");
                    throw;
                }

                _logger->info("BaseTrainer", "Training epoch " + std::to_string(epoch + 1) +
                             " completed with " + std::to_string(batchCount) + " batches");

                _profiler->stop("train_epoch");
            }

            void BaseTrainer::saveCheckpoint(int epoch, bool isBest)
            {
                _profiler->start("save_checkpoint");

                if (isBest)
                {
                    if (!_model || !_workspace)
                    {
                        _logger->warn("BaseTrainer", "Cannot save checkpoint - model or workspace not initialized");
                        _profiler->stop("save_checkpoint");
                        return;
                    }
                }
                else
                {
                    if (!_model || !_optimizer || !_workspace)
                    {
                        _logger->warn("BaseTrainer", "Cannot save checkpoint - model, optimizer, or workspace not initialized");
                        _profiler->stop("save_checkpoint");
                        return;
                    }
                }

                // Create metadata from configuration
                auto metadata = Utils::CheckpointMetadata::fromConfiguration(*_config, epoch, _currentMetrics);

                // Determine filename: best.pt or last.pt
                std::string filename = isBest ? "best.pt" : "last.pt";
                std::string path = _workspace->getWeightsDir() + "/" + filename;

                // Save with EMA parameters if available (RAII guard for exception safety)
                {
                    EMAGuard emaGuard(_ema.get(), _model.get());

                    if (isBest) {
                        Utils::Checkpoint::saveModelOnly(path, *_model, metadata);
                    } else {
                        Utils::Checkpoint::save(path, *_model, *_optimizer, metadata);
                    }

                    _logger->info("BaseTrainer", emaGuard.isApplied()
                        ? "Checkpoint saved with EMA weights: " + filename
                        : "Checkpoint saved: " + filename);
                }

                _profiler->stop("save_checkpoint");
            }

            void BaseTrainer::saveTrainingResults()
            {
                if (_epochMetrics.empty())
                {
                    _logger->warn("BaseTrainer", "No epoch metrics to save");
                    return;
                }

                _profiler->start("save_training_results");

                // Get results directory from workspace
                std::string resultsDir = _workspace ? _workspace->getLogsDir() : "./results";

                // Export metrics using MetricsExporter utility
                WheelDL::Utils::Export::MetricsExporter::exportAll(
                    _epochMetrics,
                    resultsDir,
                    "training_metrics.json",
                    "training_metrics.csv",
                    _bestFitness
                );

                _profiler->stop("save_training_results");
			}

            MetricsData BaseTrainer::runValidation(int epoch, bool useEmaIfAvailable)
            {
                if (!_validator)
                {
                    return MetricsData();
                }

                // Use RAII guard for EMA if requested (automatically restores on scope exit)
                EMAGuard emaGuard(useEmaIfAvailable ? _ema.get() : nullptr, _model.get());

                return _validator->validate(*_model, epoch);
            }

            void BaseTrainer::finalValidation()
            {
                _profiler->start("final_validation");
                _logger->info("BaseTrainer", "========== Final Validation ==========");

                if (!_validator) {
                    _logger->warn("BaseTrainer", "Validator not initialized, skipping final validation");
                    _profiler->stop("final_validation");
                    return;
                }

                try
                {
                    namespace fs = std::filesystem;
                    fs::path weightsDir = _workspace->getWeightsDir();
                    fs::path bestPath = weightsDir / "best.pt";

                    bool loadedBest = false;

                    if (fs::exists(bestPath))
                    {
                        _logger->info("BaseTrainer", "Loading best checkpoint: " + bestPath.string());

                        auto metadata = Utils::Checkpoint::loadModelOnly(bestPath.string(), *_model, _logger);  // Core::Utils::Checkpoint
						_currentEpoch = metadata.epoch;
                        _model->to(_device);
                        _model->eval();
                        loadedBest = true;
                        _logger->info("BaseTrainer", "Loaded best model (epoch=" + std::to_string(metadata.epoch) +
                            ", fitness=" + std::to_string(metadata.bestFitness) + ")");
                    }
                    else
                    {
                        _logger->warn("BaseTrainer", "best.pt not found, utilizing current model state.");
                        _model->eval();
                    }

                    _currentMetrics = runValidation(_currentEpoch, !loadedBest);

                    if (!loadedBest)
                    {
                        _logger->info("BaseTrainer", "Saving current model as best checkpoint.");
                        saveCheckpoint(_currentEpoch, true);
                    }

                    fs::path logsDir = _workspace->getLogsDir();
                    std::string jsonPath = (logsDir / "best_metrics.json").string();
                    _bestFitness = calculateFitness(_currentMetrics);
					_currentMetrics.fitness = _bestFitness;
                    WheelDL::Utils::Export::MetricsExporter::exportToJSON(
                        { _currentMetrics },
                        jsonPath,
                        _bestFitness
                    );

                    _logger->info("BaseTrainer", "Final Validation Completed");
                    _logger->info("BaseTrainer", "  Loss: " + std::to_string(_currentMetrics.loss));
                    _logger->info("BaseTrainer", "  Fitness: " + std::to_string(_bestFitness));
                }
                catch (const std::exception& e)
                {
                    _logger->error("BaseTrainer", "Final validation failed: " + std::string(e.what()));
                }

                _profiler->stop("final_validation");
            }

            void BaseTrainer::finalPrediction()
            {
                _profiler->start("final_prediction");
                _logger->info("BaseTrainer", "========== Final Prediction ==========");

                try
                {
                    std::string bestPath = _workspace->getWeightsDir() + "/best.pt";
                    if (!std::filesystem::exists(bestPath)) {
                        _logger->warn("BaseTrainer", "best.pt not found, skipping prediction");
                        _profiler->stop("final_prediction");
                        return;
                    }

                    auto predictor = setupPredictor(bestPath);
                    if (!predictor)
                    {
                        throw WheelDL::Utils::WheelLibException(WheelDL::Utils::ErrorCode::PREDICTION_FAILED, "Failed to create predictor");
                    }

                    std::string outputDir = _workspace->getResultDir();
                    predictor->predictAndExport(outputDir);

                    _logger->info("BaseTrainer", "Predictions exported to: " + outputDir);
                }
                catch (const std::exception& e)
                {
                    _logger->error("BaseTrainer", "Final prediction failed: " + std::string(e.what()));
                }

                _profiler->stop("final_prediction");
            }

            void BaseTrainer::optimizerStep()
            {
                torch::nn::utils::clip_grad_norm_(_model->parameters(), 10.0);
                // Optimizer step
                _optimizer->step();
                // Zero gradients for next iteration
                _optimizer->zero_grad();
            }

            void BaseTrainer::schedulerStep()
            {
                if (_scheduler) 
                {
                    _scheduler->step(_currentEpoch);
                    _logger->info("BaseTrainer", "Learning Rate updated to: " +
						std::to_string(_scheduler->getCurrentLR()));
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

            Utils::CheckpointMetadata BaseTrainer::loadCheckpoint(const std::string& checkpointPath, bool resumeTraining)
            {
                _profiler->start("load_checkpoint");
                _logger->info("BaseTrainer", "Loading checkpoint: " + checkpointPath);

                if (!_model)
                {
                    throw WheelDL::Utils::WheelLibException(
                        WheelDL::Utils::ErrorCode::TRAINING_FAILED,
                        "Model must be initialized before loading checkpoint. Call setupModel() first."
                    );
                }

                try
                {
                    Utils::CheckpointMetadata metadata;  // Core::Utils::CheckpointMetadata

                    if (resumeTraining)
                    {
                        if (!_optimizer)
                        {
                            throw WheelDL::Utils::WheelLibException(
                                WheelDL::Utils::ErrorCode::TRAINING_FAILED,
                                "Optimizer must be initialized to resume training. Call setupOptimizer() first."
                            );
                        }

                        metadata = Utils::Checkpoint::load(checkpointPath, *_model, *_optimizer);  // Core::Utils::Checkpoint
                        _logger->info("BaseTrainer", "Loaded checkpoint with optimizer state for resume training");
                    }
                    else
                    {
                        metadata = Utils::Checkpoint::loadModelOnly(checkpointPath, *_model, _logger);  // Core::Utils::Checkpoint
                        _logger->info("BaseTrainer", "Loaded model weights only");
                    }

                    // Move model to configured device
                    _model->to(_device);

                    if (!_usePretrained)
                    {
                        _currentEpoch = metadata.epoch;
                        _bestFitness = metadata.bestFitness;
                    }

                    _logger->info("BaseTrainer", "Checkpoint loaded successfully (epoch=" +
                        std::to_string(metadata.epoch) + ", fitness=" +
                        std::to_string(metadata.bestFitness) + ")");

                    _profiler->stop("load_checkpoint");
                    return metadata;
                }
                catch (const WheelDL::Utils::WheelLibException&)
                {
                    _profiler->stop("load_checkpoint");
                    throw;
                }
                catch (const std::exception& e)
                {
                    _profiler->stop("load_checkpoint");
                    throw WheelDL::Utils::WheelLibException(
                        WheelDL::Utils::ErrorCode::FILE_IO_ERROR,
                        "Failed to load checkpoint: " + std::string(e.what())
                    );
                }
            }

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
