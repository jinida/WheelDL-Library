#include "pch.h"
#include "BasePredictor.h"
#include "../../Utils/Error/WheelLibException.h"

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            BasePredictor::BasePredictor(
                std::shared_ptr<Config::Configuration> config,
                const std::string& checkpointPath)
                : _config(config)
                , _logger(Utils::Logger::getInstance())
                , _profiler(Utils::PerformanceProfiler::getInstance())
                , _memoryManager(Utils::MemoryManager::getInstance())
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
                _profiler.start("setup_device");

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

                _profiler.stop("setup_device");
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
                _profiler.start("load_checkpoint");

                _logger->info("BasePredictor", "Loading checkpoint: " + checkpointPath);

                if (!_model) {
                    throw Utils::WheelLibException(Utils::ErrorCode::PREDICTION_FAILED,
                        "Model must be initialized before loading checkpoint");
                }

                try {
                    // Load checkpoint using Checkpoint utility (model only, no optimizer)
                    auto metadata = Trainer::Checkpoint::loadModelOnly(checkpointPath, *_model);

                    // Move model to device
                    _model->to(_device);

                    // Set to eval mode
                    _model->eval();

                    _isModelLoaded = true;
                    _checkpointPath = checkpointPath;

                    _logger->info("BasePredictor", "Checkpoint loaded successfully (epoch=" +
                                 std::to_string(metadata.epoch) + ", fitness=" +
                                 std::to_string(metadata.bestFitness) + ")");
                }
                catch (const std::exception& e) {
                    throw Utils::WheelLibException(Utils::ErrorCode::PREDICTION_FAILED,
                        "Failed to load checkpoint: " + std::string(e.what()));
                }

                _profiler.stop("load_checkpoint");
            }



            PredictionResult BasePredictor::predict(const std::string& filePath)
            {
                if (filePath.empty())
                {
                    throw Utils::WheelLibException(Utils::ErrorCode::INVALID_ARGUMENT,
                        "Input file path is empty");
				}
				auto image = WheelDL::Data::Utils::ImageIO::loadImage(filePath);
				return predict(image);
            }

            PredictionResult BasePredictor::predict(const cv::Mat& input)
            {
                if (input.empty())
                {
                    throw Utils::WheelLibException(Utils::ErrorCode::INVALID_ARGUMENT,
                        "Input image is empty");
                }

				// Convert to float
				cv::Mat floatImage;
				input.convertTo(floatImage, CV_32FC3);

				// Ensure continuous memory for safe memcpy
				if (!floatImage.isContinuous())
				{
					floatImage = floatImage.clone();
				}

				// Create tensor and copy data
				torch::Tensor tensor = torch::empty(
					{ floatImage.rows, floatImage.cols, 3 },
					torch::kFloat32
				);
				std::memcpy(
					tensor.data_ptr<float>(),
					floatImage.data,
					floatImage.total() * floatImage.elemSize()
				);

				// Permute from [H, W, C] to [C, H, W]
				tensor = tensor.permute({ 2, 0, 1 });

				// Delegate to tensor version of predict
				return predict(tensor);
            }

            PredictionResult BasePredictor::predict(const torch::Tensor& input)
            {
                // RAII guard: automatically clears GPU cache when prediction ends
                Utils::GPUMemoryGuard gpuGuard;

                _profiler.start("predict");

                if (!_model || !_isModelLoaded) {
                    throw Utils::WheelLibException(Utils::ErrorCode::PREDICTION_FAILED,
                        "Model not loaded. Call loadCheckpoint() first.");
                }

                // Warmup if not done yet
                if (!_isWarmedUp && _device.is_cuda()) {
                    warmup();
                }

                // Preprocess
                _profiler.start("preprocess");
                torch::Tensor preprocessedInput = preprocess(input);
                _profiler.stop("preprocess");

                // Inference
                _profiler.start("inference");
                torch::Tensor output = inference(preprocessedInput);
                _profiler.stop("inference");

                // Postprocess
                _profiler.start("postprocess");
                PredictionResult result = postprocess(output, input);
                _profiler.stop("postprocess");

                _profiler.stop("predict");

                return result;
            }

            std::vector<PredictionResult> BasePredictor::predictBatch(const std::vector<torch::Tensor>& inputs)
            {
                // RAII guard: automatically clears GPU cache when prediction ends
                Utils::GPUMemoryGuard gpuGuard;

                _profiler.start("predict_batch");

                if (!_model || !_isModelLoaded) {
                    throw Utils::WheelLibException(Utils::ErrorCode::PREDICTION_FAILED,
                        "Model not loaded. Call loadCheckpoint() first.");
                }

                // Warmup if not done yet
                if (!_isWarmedUp && _device.is_cuda()) {
                    warmup();
                }

                std::vector<PredictionResult> results;
                results.reserve(inputs.size());

                // Preprocess all inputs
                _profiler.start("preprocess_batch");
                std::vector<torch::Tensor> preprocessedInputs;
                preprocessedInputs.reserve(inputs.size());
                for (const auto& input : inputs) {
                    preprocessedInputs.push_back(preprocess(input));
                }
                _profiler.stop("preprocess_batch");

                // Stack into batch
                torch::Tensor batchInput = torch::stack(preprocessedInputs, 0);

                // Inference on batch
                _profiler.start("inference_batch");
                torch::Tensor batchOutput = inference(batchInput);
                _profiler.stop("inference_batch");

                // Postprocess each output
                _profiler.start("postprocess_batch");
                for (size_t i = 0; i < inputs.size(); ++i) {
                    // Extract output for this sample
                    torch::Tensor output = batchOutput[i];
                    PredictionResult result = postprocess(output, inputs[i]);
                    results.push_back(result);
                }
                _profiler.stop("postprocess_batch");

                _profiler.stop("predict_batch");

                return results;
            }

            torch::Tensor BasePredictor::inference(const torch::Tensor& preprocessedInput)
            {
                torch::NoGradGuard noGrad;  // Disable gradient computation

                // Move input to device
                torch::Tensor deviceInput = preprocessedInput.to(_device);

                // Forward pass
                auto outputs = _model->predict(deviceInput);

                // Return first output (most models have single output)
                // Multi-scale models can override this behavior
                return outputs[0];
            }

            void BasePredictor::warmup()
            {
                _profiler.start("warmup");

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

                _profiler.stop("warmup");
            }

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
