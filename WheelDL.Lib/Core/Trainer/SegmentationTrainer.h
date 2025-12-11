#pragma once

#include "../Engine/BaseTrainer.h"
#include "../../Model/Task/SegmentationModel.h"
#include "../../Data/Dataset/SegmentationDataset.h"

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            /**
             * @class SegmentationTrainer
             * @brief Trainer for semantic/instance segmentation tasks
             *
             * Implements training for segmentation models:
             * - Pixel-wise classification
             * - Multi-class segmentation with BCE + Dice loss
             * - Support for various backbone architectures
             *
             * Key features:
             * - Segmentation loss (BCE + Dice)
             * - Mean IoU (mIoU) metric
             * - Dice coefficient metric
             * - Per-class IoU metrics
             */
            class SegmentationTrainer : public BaseTrainer {
            public:
                /**
                 * @brief Constructor with dependency injection
                 * @param config Configuration object with segmentation settings
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param workspace Workspace instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param progressCallback Optional progress callback
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit SegmentationTrainer(
                    std::shared_ptr<Config::Configuration> config,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::Workspace* workspace,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    ProgressCallback progressCallback = nullptr,
                    std::atomic<bool>* stopFlag = nullptr);

                /**
                 * @brief Destructor
                 */
                ~SegmentationTrainer() override = default;

            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup segmentation model
                 *
                 * Creates SegmentationModel based on configuration model path.
                 */
                void setupModel() override;

                /**
                 * @brief Setup data loaders for segmentation
                 *
                 * Creates SegmentationDataset and DataLoader for training.
                 */
                void setupDataLoaders() override;

                /**
                 * @brief Setup validator for segmentation
                 *
                 * Creates SegmentationValidator and injects validation dataloader.
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
                 * For segmentation, fitness is based on:
                 * - Mean IoU (mIoU) as primary metric
                 * - Weighted combination of mIoU and Dice coefficient
                 *
                 * @param metrics Validation metrics
                 * @return Fitness value (higher is better)
                 */
                float calculateFitness(const MetricsData& metrics) override;

                /**
                 * @brief Setup segmentation predictor for inference
                 *
                 * Creates SegmentationPredictor with checkpoint.
                 *
                 * @param checkpointPath Path to checkpoint file
                 * @return Unique pointer to SegmentationPredictor
                 */
                std::unique_ptr<Predictor::BasePredictor> setupPredictor(
                    const std::string& checkpointPath) override;
            };

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
