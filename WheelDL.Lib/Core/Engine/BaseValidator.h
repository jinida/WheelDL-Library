#pragma once

#include <torch/torch.h>
#include <memory>
#include <functional>
#include "../../Config/Configuration.h"
#include "../../Model/Task/BaseModel.h"
#include "../../Data/Dataset/BaseDataset.h"
#include "../../Utils/Logger/Logger.h"
#include "../../Utils/Profiler/PerformanceProfiler.h"
#include "../../Utils/Memory/GPUMemoryGuard.h"
#include "../../Utils/Common/Types.h"

namespace WheelDL {
    namespace Core {
        namespace Validator {

            /**
             * @class BaseValidator
             * @brief Base class for model validation
             *
             * Provides validation loop with hooks for task-specific customization.
             * Can be used with injected DataLoader or create its own.
             *
             * Derived classes must implement:
             * - setupDataLoader() (if not using injected DataLoader)
             * - preprocessBatch()
             * - computeMetrics()
             */
            class BaseValidator {
            public:
                /**
                 * @brief Constructor
                 * @param config Configuration object
                 * @param device Device to run validation on
                 */
                explicit BaseValidator(
                    const std::shared_ptr<Config::Configuration> config,
                    const torch::Device& device
                );

                /**
                 * @brief Virtual destructor
                 */
                virtual ~BaseValidator();

                // Disable copy and move
                BaseValidator(const BaseValidator&) = delete;
                BaseValidator& operator=(const BaseValidator&) = delete;
                BaseValidator(BaseValidator&&) = delete;
                BaseValidator& operator=(BaseValidator&&) = delete;

                /**
                 * @brief Set data loader from outside (dependency injection)
                 * @tparam DataLoaderType Type of data loader
                 * @param dataLoader Data loader to use for validation
                 */
                template<typename DataLoaderType>
                void setDataLoader(DataLoaderType&& dataLoader) {
                    auto dataLoaderPtr = std::make_shared<DataLoaderType>(std::forward<DataLoaderType>(dataLoader));

                    _batchIterator = [dataLoaderPtr](std::function<void(const WheelDL::Data::Dataset::DataExample&, int)> callback) {
                        int batchIdx = 0;
                        for (auto& batch : *(*dataLoaderPtr)) {
                            callback(batch, batchIdx++);
                        }
                    };
                }

                /**
                 * @brief Run validation
                 * @param model Model to validate
                 * @param epoch Current epoch number
                 * @return Validation metrics
                 */
                MetricsData validate(Model::BaseModel& model, int epoch = 0);

                void setupDevice();
                
            protected:
                // ========== Hook Methods (Pure Virtual) ==========

                /**
                 * @brief Setup data loader (must be implemented by derived class)
                 *
                 * Two implementation patterns:
                 * 1. If using dependency injection (setDataLoader):
                 *    - Provide empty implementation: void setupDataLoader() override {}
                 * 2. If validator creates its own DataLoader:
                 *    - Create DataLoader and call setDataLoader(std::move(loader))
                 *
                 * This method is called by validate() if no DataLoader was injected.
                 */
                virtual void setupDataLoader() = 0;

                /**
                 * @brief Preprocess batch before forward pass
                 * @param batch Input batch from dataloader
                 * @return Preprocessed batch ready for model
                 */
                virtual WheelDL::Data::Dataset::DataExample preprocessBatch(
                    const WheelDL::Data::Dataset::DataExample& batch) = 0;

                virtual torch::Tensor postprocessBatch(
					const std::vector<torch::Tensor>& prediction) = 0;

                /**
                 * @brief Compute metrics from predictions and targets
                 * @param pred Model predictions
                 * @param target Ground truth targets
                 * @return Computed metrics
                 */
                virtual MetricsData computeMetrics(
                    const torch::Tensor& pred,
                    const torch::Tensor& target) = 0;

            protected:
                // ========== Configuration & Core Components ==========
                std::shared_ptr<Config::Configuration> _config;
                std::shared_ptr<Utils::Logger> _logger;
                Utils::PerformanceProfiler& _profiler;

                // ========== Device ==========
                torch::Device _device;

                // ========== Data Loader ==========
                using BatchIteratorFunc = std::function<void(std::function<void(const WheelDL::Data::Dataset::DataExample&, int)>)>;
                BatchIteratorFunc _batchIterator;
            };

        } // namespace Validator
    } // namespace Core
} // namespace WheelDL
