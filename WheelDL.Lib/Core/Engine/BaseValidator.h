#pragma once

#include <torch/torch.h>
#include <memory>
#include <functional>
#include <atomic>
#include "../../Config/Configuration.h"
#include "../../Model/Task/BaseModel.h"
#include "../../Data/Dataset/BaseDataset.h"
#include "../Utils/Checkpoint.h"
#include "../Utils/CheckpointMetadata.h"
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
             * - setupModel() - create task-specific model
             * - setupDataLoader() (if not using injected DataLoader)
             * - preprocessBatch()
             * - computeMetrics()
             */
            class BaseValidator {
            public:
                /**
                 * @brief Constructor with dependency injection
                 * @param config Configuration object
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit BaseValidator(
                    std::shared_ptr<Config::Configuration> config,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    std::atomic<bool>* stopFlag = nullptr);

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
                 * @brief Run validation with existing model
                 * @param model Model to validate
                 * @param epoch Current epoch number
                 * @return Validation metrics
                 */
                MetricsData validate(Model::BaseModel& model, int epoch = 0);

                /**
                 * @brief Run validation from checkpoint (standalone mode)
                 * @param checkpointPath Path to model checkpoint
                 * @return Validation metrics
                 */
                MetricsData validate(const std::string& checkpointPath);

                void setupDevice();
                
            protected:
                // ========== Hook Methods (Pure Virtual) ==========

                /**
                 * @brief Setup model (must be implemented by derived class)
                 * @return Task-specific model instance
                 */
                virtual std::unique_ptr<Model::BaseModel> setupModel() = 0;

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
                // ========== Stop Flag ==========

                /**
                 * @brief Check if stop has been requested
                 * @return true if stop was requested, false otherwise
                 */
                bool isStopRequested() const {
                    return _stopFlag && _stopFlag->load(std::memory_order_acquire);
                }

                // ========== Configuration & Core Components ==========
                std::shared_ptr<Config::Configuration> _config;
                WheelDL::Utils::Logger* _logger;                      // Injected (owned by Context)
                WheelDL::Utils::PerformanceProfiler* _profiler;       // Injected (owned by Context)
                std::atomic<bool>* _stopFlag;                         // Injected (owned by Context)

                // ========== Device ==========
                torch::Device _device;

                // ========== Data Loader ==========
                using BatchIteratorFunc = std::function<void(std::function<void(const WheelDL::Data::Dataset::DataExample&, int)>)>;
                BatchIteratorFunc _batchIterator;
            };

        } // namespace Validator
    } // namespace Core
} // namespace WheelDL
