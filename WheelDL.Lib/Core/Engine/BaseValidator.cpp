#include "pch.h"
#include "BaseValidator.h"
#include "../../Utils/Error/WheelLibException.h"

namespace WheelDL {
    namespace Core {
        namespace Validator {

            BaseValidator::BaseValidator(
                const std::shared_ptr<Config::Configuration> config,
                const torch::Device& device)
                : _config(config)
                , _logger(Utils::Logger::getInstance())
                , _profiler(Utils::PerformanceProfiler::getInstance())
                , _device(device)
            {
                _logger->info("BaseValidator", "Validator initialized");
                setupDevice();
            }

            BaseValidator::~BaseValidator()
            {
                _logger->info("BaseValidator", "Validator destroyed");
            }

            MetricsData BaseValidator::validate(Model::BaseModel& model, int epoch)
            {
                // RAII guard: automatically clears GPU cache when validation ends
                Utils::GPUMemoryGuard gpuGuard;

                _profiler.start("validation");

                if (!_batchIterator) 
                {
                    // Try to setup data loader if not injected
                    setupDataLoader();

                    if (!_batchIterator) {
                        throw Utils::WheelLibException(Utils::ErrorCode::VALIDATION_FAILED,
                            "Validation data loader not initialized. Call setDataLoader() or implement setupDataLoader().");
                    }
                }

				model.to(_device);
                model.eval();

                torch::NoGradGuard noGrad;  // Disable gradient computation

                std::vector<torch::Tensor> allPreds;
                std::vector<torch::Tensor> allTargets;
                float totalLoss = 0.0f;
                int batchCount = 0;

                // Iterate over validation batches
                _batchIterator([this, &model, &allPreds, &allTargets, &totalLoss, &batchCount](
                    const WheelDL::Data::Dataset::DataExample& batch, int batchIdx) {

                    // Preprocess batch
                    auto processedBatch = preprocessBatch(batch);
                    processedBatch.toDevice(_device);

                    auto inputData = processedBatch.data;
                    auto prediction = model.forward(inputData);
					auto lossDict = model.loss(processedBatch, prediction);
                    auto loss = lossDict["total"];

                    totalLoss += loss.item<float>();
                    batchCount++;

                    // Store predictions and targets for metrics computation
                    allPreds.push_back(postprocessBatch(prediction));
                    allTargets.push_back(processedBatch.targets.cpu());
                });

                // Compute average loss
                float avgLoss = batchCount > 0 ? totalLoss / batchCount : 0.0f;

                // Concatenate all predictions and targets
                torch::Tensor allPredsTensor = torch::cat(allPreds, 0);
                torch::Tensor allTargetsTensor = torch::cat(allTargets, 0);

                // Compute metrics using derived class implementation
                MetricsData metrics = computeMetrics(allPredsTensor, allTargetsTensor);
                metrics.loss = avgLoss;

                _logger->info("BaseValidator", "Validation epoch " + std::to_string(epoch + 1) +
                             " completed | Loss: " + std::to_string(avgLoss));

                _profiler.stop("validation");

                return metrics;
            }

            void BaseValidator::setupDevice()
            {
                _profiler.start("setup_device");

                std::string deviceStr = _config->getDevice();

                if (deviceStr.empty())
                {
                    // Auto-detect
                    _device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
                    _logger->info("BaseValidator", std::string("Auto-detected device: ") + (_device.is_cuda() ? "CUDA" : "CPU"));
                }
                else if (deviceStr == "cpu")
                {
                    _device = torch::kCPU;
                    _logger->info("BaseValidator", "Using CPU");
                }
                else
                {
                    // CUDA device
                    _device = torch::Device(torch::kCUDA, 0); // Default to GPU 0
                    _logger->info("BaseValidator", "Using CUDA device: " + deviceStr);
                }

                _profiler.stop("setup_device");
			}

        } // namespace Validator
    } // namespace Core
} // namespace WheelDL
