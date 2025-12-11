#pragma once

#include "../Engine/BaseTrainer.h"
#include "../../Model/Task/ClassificationModel.h"
#include "../../Data/Dataset/ClassificationDataset.h"

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            /**
             * @class ClassificationTrainer
             * @brief Trainer for image classification tasks
             *
             * Implements training for classification models:
             * - Multi-class classification (softmax + cross-entropy)
             * - Multi-label classification (sigmoid + BCE)
             * - Support for various backbone architectures (ResNet, EfficientNet, etc.)
             *
             * Key features:
             * - Standard supervised learning pipeline
             * - Top-1 and Top-5 accuracy metrics
             * - Support for class imbalance handling (focal loss, class weights)
             * - Label smoothing support
             */
            class ClassificationTrainer : public BaseTrainer {
            public:
                /**
                 * @brief Constructor with dependency injection
                 * @param config Configuration object with classification settings
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param workspace Workspace instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param progressCallback Optional progress callback
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit ClassificationTrainer(
                    std::shared_ptr<Config::Configuration> config,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::Workspace* workspace,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    ProgressCallback progressCallback = nullptr,
                    std::atomic<bool>* stopFlag = nullptr);

                /**
                 * @brief Destructor
                 */
                ~ClassificationTrainer() override = default;

            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup classification model
                 *
                 * Creates ClassificationModel based on configuration model path.
                 */
                void setupModel() override;

                /**
                 * @brief Setup data loaders for classification
                 *
                 * Creates ClassificationDataset and DataLoader for training.
                 */
                void setupDataLoaders() override;

                /**
                 * @brief Setup validator for classification
                 *
                 * Creates ClassificationValidator and injects validation dataloader.
                 */
                void setupValidator() override;

                /**
                 * @brief Preprocess batch before forward pass
                 * @param batch Input batch from dataloader
                 * @return Preprocessed batch with data moved to device
                 */
                WheelDL::Data::Dataset::DataExample preprocessBatch(
                    const WheelDL::Data::Dataset::DataExample& batch) override;

                /**
                 * @brief Calculate fitness metric for checkpoint selection
                 *
                 * For classification, fitness is typically based on:
                 * - Top-1 Accuracy
                 * - F1 Score (for imbalanced datasets)
                 *
                 * @param metrics Validation metrics
                 * @return Fitness value (higher is better)
                 */
                float calculateFitness(const MetricsData& metrics) override;

                /**
                 * @brief Setup classification predictor for inference
                 *
                 * Creates ClassificationPredictor with checkpoint.
                 *
                 * @param checkpointPath Path to checkpoint file
                 * @return Unique pointer to ClassificationPredictor
                 */
                std::unique_ptr<Predictor::BasePredictor> setupPredictor(
                    const std::string& checkpointPath) override;
            };

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
