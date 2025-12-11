#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include "../../Config/Configuration.h"
#include "../../Model/Task/BaseModel.h"
#include "../../Utils/Logger/Logger.h"
#include "../../Utils/Profiler/PerformanceProfiler.h"
#include "../../Utils/Memory/MemoryManager.h"
#include "../../Utils/Memory/GPUMemoryGuard.h"
#include "../../Utils/Common/Types.h"
#include "../Utils/Checkpoint.h"

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            /**
             * @class BasePredictor
             * @brief Base class for model inference/prediction
             *
             * Provides batch inference pipeline with hooks for task-specific customization.
             * Uses PredDataset for data loading and preprocessing.
             *
             * Key features:
             * - Configuration-driven inference
             * - Checkpoint loading with EMA support
             * - Comprehensive logging and profiling
             * - Memory-efficient batch inference
             * - Template Method Pattern for customization
             *
             * Derived classes must implement:
             * - setupModel()
             * - postprocess()
             * - exportResults()
             * - clearResults()
             */
            class BasePredictor {
            public:
                /**
                 * @brief Constructor with dependency injection
                 * @param config Configuration object
                 * @param checkpointPath Path to checkpoint file (optional)
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit BasePredictor(
                    std::shared_ptr<Config::Configuration> config,
                    const std::string& checkpointPath,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    std::atomic<bool>* stopFlag = nullptr);

                /**
                 * @brief Virtual destructor
                 */
                virtual ~BasePredictor();

                // Disable copy and move
                BasePredictor(const BasePredictor&) = delete;
                BasePredictor& operator=(const BasePredictor&) = delete;
                BasePredictor(BasePredictor&&) = delete;
                BasePredictor& operator=(BasePredictor&&) = delete;

                /**
                 * @brief Run batch prediction and export results
                 * @param outputDir Output directory for results
                 *
                 * Creates PredDataset from config, runs inference on all images,
                 * and exports results using task-specific exportResults().
                 */
                void predictAndExport(const std::string& outputDir);

                /**
                 * @brief Load model from checkpoint
                 * @param checkpointPath Path to checkpoint file
                 */
                void loadCheckpoint(const std::string& checkpointPath);

                /**
                 * @brief Set device for inference
                 * @param device Target device (CPU/CUDA)
                 */
                void setDevice(const torch::Device& device);

                /**
                 * @brief Get current device
                 */
                torch::Device getDevice() const { return _device; }

            protected:
                // ========== Hook Methods (Pure Virtual) ==========

                /**
                 * @brief Setup model architecture
                 *
                 * Create and initialize the model based on configuration.
                 * Should set _model member.
                 */
                virtual void setupModel() = 0;

                /**
                 * @brief Postprocess model output and store in internal container
                 * @param output Raw model output
                 * @param originalShape Original image shape (height, width)
                 * @param imagePath Image file path for export
                 *
                 * Derived classes should process output and store results
                 * in their internal container for later export.
                 */
                virtual void postprocess(
                    const std::vector<torch::Tensor>& output,
                    const std::tuple<int, int>& originalShape,
                    const std::string& imagePath) = 0;

                /**
                 * @brief Export accumulated results to files
                 * @param outputDir Output directory
                 *
                 * Derived classes should export their internal results
                 * in task-specific format (e.g., JSON, images).
                 */
                virtual void exportResults(const std::string& outputDir) = 0;

                /**
                 * @brief Clear internal results container
                 *
                 * Called at the start of predictAndExport() to reset state.
                 */
                virtual void clearResults() = 0;

                // ========== Common Methods ==========

                /**
                 * @brief Setup device (CPU/CUDA)
                 *
                 * Configures torch device based on Configuration.
                 */
                void setupDevice();

                /**
                 * @brief Run inference on input tensor
                 * @param input Input tensor (already preprocessed by PredDataset)
                 * @return Raw model output
                 */
                std::vector<torch::Tensor> inference(const torch::Tensor& input);

                /**
                 * @brief Warmup model (run dummy inference)
                 *
                 * Runs a few dummy inferences to warm up CUDA kernels.
                 * Improves performance of first real inference.
                 */
                void warmup();

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
                WheelDL::Utils::MemoryManager& _memoryManager;
                std::atomic<bool>* _stopFlag;                         // Injected (owned by Context)

                // ========== Model ==========
                std::unique_ptr<Model::BaseModel> _model;

                // ========== Device ==========
                torch::Device _device;

                // ========== State ==========
                bool _isModelLoaded;
                bool _isWarmedUp;

                // ========== Checkpoint Path ==========
                std::string _checkpointPath;
                float _threshold = 0.5f;
            };

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
