#pragma once

#include "../Engine/BaseTrainer.h"
#include "../../Model/Task/DetectionModel.h"
#include "../../Data/Dataset/DetectionDataset.h"

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            /**
             * @class DetectionTrainer
             * @brief Trainer for object detection tasks
             *
             * Implements training for YOLO-style detection models:
             * - Anchor-free detection with DFL
             * - Multi-scale predictions (P3, P4, P5)
             * - Task-aligned assignment for training
             *
             * Key features:
             * - Detection loss (box + classification + DFL)
             * - mAP@0.5 and mAP@0.5:0.95 metrics
             * - Non-Maximum Suppression (NMS)
             * - Support for various backbone architectures
             */
            class DetectionTrainer : public BaseTrainer {
            public:
                /**
                 * @brief Constructor with dependency injection
                 * @param config Configuration object with detection settings
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param workspace Workspace instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param progressCallback Optional progress callback
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit DetectionTrainer(
                    std::shared_ptr<Config::Configuration> config,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::Workspace* workspace,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    ProgressCallback progressCallback = nullptr,
                    std::atomic<bool>* stopFlag = nullptr);

                /**
                 * @brief Destructor
                 */
                ~DetectionTrainer() override = default;
            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup detection model
                 *
                 * Creates DetectionModel based on configuration model path.
                 */
                void setupModel() override;

                /**
                 * @brief Setup data loaders for detection
                 *
                 * Creates DetectionDataset and DataLoader for training.
                 */
                void setupDataLoaders() override;

                /**
                 * @brief Setup validator for detection
                 *
                 * Creates DetectionValidator and injects validation dataloader.
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
                 * For detection, fitness is based on:
                 * - mAP@0.5 (primary metric)
                 * - Weighted combination of mAP@0.5 and mAP@0.5:0.95
                 *
                 * @param metrics Validation metrics
                 * @return Fitness value (higher is better)
                 */
                float calculateFitness(const MetricsData& metrics) override;

                /**
                 * @brief Setup detection predictor for inference
                 *
                 * Creates DetectionPredictor with checkpoint.
                 *
                 * @param checkpointPath Path to checkpoint file
                 * @return Unique pointer to DetectionPredictor
                 */
                std::unique_ptr<Predictor::BasePredictor> setupPredictor(
                    const std::string& checkpointPath) override;
            };
        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
