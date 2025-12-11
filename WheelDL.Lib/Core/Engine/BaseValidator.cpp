#include "pch.h"
#include "BaseValidator.h"
#include "../../Utils/Error/WheelLibException.h"

namespace WheelDL {
    namespace Core {
        namespace Validator {

            BaseValidator::BaseValidator(
                std::shared_ptr<Config::Configuration> config,
                WheelDL::Utils::Logger* logger,
                WheelDL::Utils::PerformanceProfiler* profiler,
                std::atomic<bool>* stopFlag)
                : _config(config)
                , _logger(logger)
                , _profiler(profiler)
                , _stopFlag(stopFlag)
                , _device(torch::kCPU)
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
                WheelDL::Utils::GPUMemoryGuard gpuGuard;

                _profiler->start("validation");

                if (!_batchIterator) 
                {
                    // Try to setup data loader if not injected
                    setupDataLoader();

                    if (!_batchIterator) {
                        throw WheelDL::Utils::WheelLibException(WheelDL::Utils::ErrorCode::VALIDATION_FAILED,
                            "Validation data loader not initialized. Call setDataLoader() or implement setupDataLoader().");
                    }
                }

				model.to(_device);
                model.eval();

                torch::NoGradGuard noGrad;  // Disable gradient computation

                std::vector<torch::Tensor> allPreds;
                std::vector<torch::Tensor> allTargets;
                int batchCount = 0;
				std::unordered_map<std::string, torch::Tensor> lossInfo;
                
                try {
                    _batchIterator([this, &model, &allPreds, &allTargets, &lossInfo, &batchCount](
                        const WheelDL::Data::Dataset::DataExample& batch, int batchIdx) {

                        // Check stop flag at iteration level
                        if (isStopRequested()) {
                            _logger->info("BaseValidator", "Stop requested, aborting validation");
                            throw WheelDL::Utils::StopRequestedException();
                        }

                        // Preprocess batch
                        auto processedBatch = preprocessBatch(batch);
                        processedBatch.toDevice(_device);

                        auto inputData = processedBatch.data;
                        auto prediction = model.forward(inputData);
                        auto lossDict = model.loss(processedBatch, prediction);
                        for (const auto& [key, value] : lossDict)
                        {
                            if (lossInfo.find(key) == lossInfo.end())
                            {
                                lossInfo[key] = value.detach().cpu();
                            }
                            else
                            {
                                lossInfo[key] += value.detach().cpu();
                            }
                        }

                        // Store predictions and targets for metrics computation
                        allPreds.push_back(postprocessBatch(prediction));
                        if (this->_config->getTaskType() == TaskType::DETECTION || this->_config->getTaskType() == TaskType::OBB)
                        {
                            processedBatch.targets.select(1, 0).add_(batchCount * processedBatch.data.size(0));
                        }

                        allTargets.push_back(processedBatch.targets.cpu());
                        batchCount++;
                    });
                }
                catch (const WheelDL::Utils::StopRequestedException&) {
                    // Stop requested - return empty metrics without saving results
                    _profiler->stop("validation");
                    throw; // Re-throw to let caller handle
                }

                float totalLoss = lossInfo["total"].item<float>();
                float avgLoss = batchCount > 0 ? totalLoss / batchCount : 0.0f;
                avgLoss /= static_cast<float>(_config->getBatchSize());
                // Concatenate all predictions and targets
                torch::Tensor allPredsTensor = torch::cat(allPreds, 0);
                torch::Tensor allTargetsTensor = torch::cat(allTargets, 0);

                // Compute metrics using derived class implementation
                MetricsData metrics = computeMetrics(allPredsTensor, allTargetsTensor);
                metrics.loss = avgLoss;

                _logger->info("BaseValidator", "Validation epoch " + std::to_string(epoch + 1) + " completed");
                for (const auto& [key, value] : lossInfo)
                {
                    if (key == "total")
                        continue;
                    float lossValue = value.item<float>() / static_cast<float>(batchCount);
                    _logger->info("BaseValidator", "  " + key + ": " + std::to_string(lossValue));
                }

                _profiler->stop("validation");

                return metrics;
            }

            MetricsData BaseValidator::validate(const std::string& checkpointPath)
            {
                _logger->info("BaseValidator", "Validating from checkpoint: " + checkpointPath);

                // Create model using derived class implementation
                auto model = setupModel();
                if (!model) {
                    throw WheelDL::Utils::WheelLibException(WheelDL::Utils::ErrorCode::VALIDATION_FAILED,
                        "Failed to create model in setupModel()");
                }

                // Load checkpoint
                Utils::Checkpoint::loadModelOnly(checkpointPath, *model, _logger);
                _logger->info("BaseValidator", "Checkpoint loaded successfully");

                // Run validation with loaded model
                return validate(*model, 0);
            }

            void BaseValidator::setupDevice()
            {
                _profiler->start("setup_device");

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

                _profiler->stop("setup_device");
			}

        } // namespace Validator
    } // namespace Core
} // namespace WheelDL
