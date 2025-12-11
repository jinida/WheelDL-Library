#include "pch.h"
#include "BasePredictor.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Data/Dataset/PredDataset.h"

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            BasePredictor::BasePredictor(
                std::shared_ptr<Config::Configuration> config,
                const std::string& checkpointPath,
                WheelDL::Utils::Logger* logger,
                WheelDL::Utils::PerformanceProfiler* profiler,
                std::atomic<bool>* stopFlag)
                : _config(config)
                , _logger(logger)
                , _profiler(profiler)
                , _memoryManager(WheelDL::Utils::MemoryManager::getInstance())
                , _stopFlag(stopFlag)
                , _device(torch::kCPU)
                , _isModelLoaded(false)
                , _isWarmedUp(false)
                , _checkpointPath(checkpointPath)
            {
                _logger->info("BasePredictor", "Initializing predictor");

                // Setup device
                setupDevice();

                // NOTE: setupModel() is called by derived class after construction
                // Cannot call pure virtual function from base constructor

                _logger->info("BasePredictor", "Base predictor initialized");
            }

            BasePredictor::~BasePredictor()
            {
                _logger->info("BasePredictor", "Predictor destroyed");
            }

            void BasePredictor::setupDevice()
            {
                _profiler->start("setup_device");

                std::string deviceStr = _config->getDevice();

                if (deviceStr.empty()) {
                    // Auto-detect
                    _device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
                    _logger->info("BasePredictor", std::string("Auto-detected device: ") + (_device.is_cuda() ? "CUDA" : "CPU"));
                } else if (deviceStr == "cpu") {
                    _device = torch::kCPU;
                    _logger->info("BasePredictor", "Using CPU");
                } else {
                    // CUDA device
                    _device = torch::Device(torch::kCUDA, 0); // Default to GPU 0
                    _logger->info("BasePredictor", "Using CUDA device: " + deviceStr);
                }

                // Log GPU memory if CUDA
                if (_device.is_cuda()) {
                    float totalMem = _memoryManager.getGPUMemoryTotal();
                    float usedMem = _memoryManager.getGPUMemoryUsed();
                    _logger->info("BasePredictor", "GPU Memory: " +
                                 std::to_string(usedMem) + "MB / " +
                                 std::to_string(totalMem) + "MB");
                }

                _profiler->stop("setup_device");
            }

            void BasePredictor::setDevice(const torch::Device& device)
            {
                if (_device.str() == device.str()) {
                    return; // Already on this device
                }

                _device = device;
                _logger->info("BasePredictor", "Device changed to: " + device.str());

                // Move model to new device if loaded
                if (_model && _isModelLoaded) {
                    _model->to(_device);
                    _logger->info("BasePredictor", "Model moved to new device");
                }
            }

            void BasePredictor::loadCheckpoint(const std::string& checkpointPath)
            {
                _profiler->start("load_checkpoint");

                _logger->info("BasePredictor", "Loading checkpoint: " + checkpointPath);

                if (!_model) {
                    throw WheelDL::Utils::WheelLibException(WheelDL::Utils::ErrorCode::PREDICTION_FAILED,
                        "Model must be initialized before loading checkpoint");
                }

                try
                {
                    // Load checkpoint using Checkpoint utility (model only, no optimizer)
                    auto metadata = Utils::Checkpoint::loadModelOnly(checkpointPath, *_model, _logger);
                    _config->setImageSize(std::stoi(metadata.hyperParams.at("image_size")));
                    _threshold = metadata.threshold;

                    _model->to(_device);
                    _model->eval();
                    _checkpointPath = checkpointPath;

                    _isModelLoaded = true;

                    _logger->info("BasePredictor", "Checkpoint loaded successfully (epoch=" +
                                 std::to_string(metadata.epoch) + ", fitness=" +
                                 std::to_string(metadata.bestFitness) + ")");
                }
                catch (const std::exception& e) {
                    throw WheelDL::Utils::WheelLibException(WheelDL::Utils::ErrorCode::PREDICTION_FAILED,
                        "Failed to load checkpoint: " + std::string(e.what()));
                }

                _profiler->stop("load_checkpoint");
            }

            void BasePredictor::predictAndExport(const std::string& outputDir)
            {
                _profiler->start("predict_and_export");
                _logger->info("BasePredictor", "========== Batch Prediction ==========");

                if (!_isModelLoaded) {
                    throw WheelDL::Utils::WheelLibException(WheelDL::Utils::ErrorCode::PREDICTION_FAILED,
                        "Model not loaded. Call loadCheckpoint() first.");
                }

                // Clear previous results
                clearResults();

                // Create PredDataset (handles image loading and preprocessing)
                auto dataset = Data::Dataset::PredDataset(*_config);
                auto dataSize = dataset.size();

                if (!dataSize.has_value() || dataSize.value() == 0) {
                    _logger->warn("BasePredictor", "No images found for prediction");
                    _profiler->stop("predict_and_export");
                    return;
                }

                _logger->info("BasePredictor", "Found " + std::to_string(dataSize.value()) + " images");

                // Create DataLoader
                auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                    std::move(dataset),
                    torch::data::DataLoaderOptions()
                        .batch_size(1)
                        .workers(_config->getWorkers())
                );

                // Warmup if needed
                if (!_isWarmedUp && _device.is_cuda()) {
                    warmup();
                }

                // Batch inference
                int processedCount = 0;
                for (auto& batch : *dataLoader) {
                    // Check stop flag at iteration level
                    if (isStopRequested()) {
                        _logger->info("BasePredictor", "Stop requested, aborting prediction");
                        clearResults();
                        _profiler->stop("predict_and_export");
                        return;
                    }

                    if (batch.empty()) continue;

                    const auto& sample = batch.front();
                    auto input = sample.data.to(_device).unsqueeze(0);

                    // Inference with timing
                    _profiler->start("inference");
                    auto output = inference(input);
                    _profiler->stop("inference");

                    // Postprocess and store in internal container
                    postprocess(output, sample.originalShape[0], sample.imagePath[0]);

                    processedCount++;

                    if (processedCount % 100 == 0) {
                        _logger->info("BasePredictor", "Processed " +
                            std::to_string(processedCount) + "/" +
                            std::to_string(dataSize.value()) + " images");
                    }
                }

                _logger->info("BasePredictor", "Inference completed: " + std::to_string(processedCount) + " images");

                // Export results
                exportResults(outputDir);

                _profiler->stop("predict_and_export");
            }

            std::vector<torch::Tensor> BasePredictor::inference(const torch::Tensor& input)
            {
                torch::NoGradGuard noGrad;  // Disable gradient computation

                // Forward pass
                auto outputs = _model->predict(input);

                return outputs;
            }

            void BasePredictor::warmup()
            {
                _profiler->start("warmup");

                _logger->info("BasePredictor", "Warming up model...");

                try {
                    // Get input size from config
                    int batchSize = 1;
                    int channels = 3;
                    int height = _config->getImageSize();
                    int width = _config->getImageSize();

                    // Create dummy input
                    torch::Tensor dummyInput = torch::randn({batchSize, channels, height, width});
                    dummyInput = dummyInput.to(_device);

                    // Run a few dummy inferences
                    const int warmupRuns = 3;
                    for (int i = 0; i < warmupRuns; ++i) {
                        torch::NoGradGuard noGrad;
                        _model->predict(dummyInput);
                    }

                    // Synchronize CUDA
                    if (_device.is_cuda()) {
                        torch::cuda::synchronize();
                    }

                    _isWarmedUp = true;
                    _logger->info("BasePredictor", "Warmup completed");
                }
                catch (const std::exception& e) {
                    _logger->warn("BasePredictor", "Warmup failed: " + std::string(e.what()));
                }

                _profiler->stop("warmup");
            }

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
