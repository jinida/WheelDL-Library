#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
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
             * Provides inference pipeline with hooks for task-specific customization.
             * Supports single image, batch, and streaming inference.
             *
             * Key features:
             * - Configuration-driven inference
             * - Checkpoint loading with EMA support
             * - Comprehensive logging and profiling
             * - Memory-efficient inference
             * - Template Method Pattern for customization
             *
             * Derived classes must implement:
             * - setupModel()
             * - preprocess()
             * - postprocess()
             */
            class BasePredictor {
            public:
                /**
                 * @brief Constructor
                 * @param config Configuration object
                 * @param checkpointPath Path to checkpoint file (optional)
                 */
                explicit BasePredictor(
                    std::shared_ptr<Config::Configuration> config,
                    const std::string& checkpointPath = ""
                );

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
                 * @brief Predict on single input
                 * @param input Input tensor (e.g., single image)
                 * @return Prediction results
                 */
                PredictionResult predict(const std::string& filePath);
                PredictionResult predict(const cv::Mat& input);
                PredictionResult predict(const torch::Tensor& input);

                /**
                 * @brief Predict on batch of inputs
                 * @param inputs Vector of input tensors
                 * @return Vector of prediction results
                 */
                std::vector<PredictionResult> predictBatch(const std::vector<torch::Tensor>& inputs);

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
                 * @brief Preprocess input tensor
                 * @param input Raw input tensor
                 * @return Preprocessed tensor ready for model
                 */
                virtual torch::Tensor preprocess(const torch::Tensor& input) = 0;

                /**
                 * @brief Postprocess model output
                 * @param output Raw model output
                 * @param originalInput Original input tensor (for dimensions, etc.)
                 * @return Processed prediction result
                 */
                virtual PredictionResult postprocess(
                    const torch::Tensor& output,
                    const torch::Tensor& originalInput) = 0;

                // ========== Common Methods ==========

                /**
                 * @brief Setup device (CPU/CUDA)
                 *
                 * Configures torch device based on Configuration.
                 */
                void setupDevice();

                /**
                 * @brief Run inference on preprocessed input
                 * @param preprocessedInput Preprocessed input tensor
                 * @return Raw model output
                 */
                torch::Tensor inference(const torch::Tensor& preprocessedInput);

                /**
                 * @brief Warmup model (run dummy inference)
                 *
                 * Runs a few dummy inferences to warm up CUDA kernels.
                 * Improves performance of first real inference.
                 */
                void warmup();

            protected:
                // ========== Configuration & Core Components ==========
                std::shared_ptr<Config::Configuration> _config;
                std::shared_ptr<Utils::Logger> _logger;
                Utils::PerformanceProfiler& _profiler;
                Utils::MemoryManager& _memoryManager;

                // ========== Model ==========
                std::unique_ptr<Model::BaseModel> _model;

                // ========== Device ==========
                torch::Device _device;

                // ========== State ==========
                bool _isModelLoaded;
                bool _isWarmedUp;
                std::string _checkpointPath;
            };

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
