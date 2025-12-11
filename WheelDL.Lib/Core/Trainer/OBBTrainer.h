#pragma once

#include "../Engine/BaseTrainer.h"
#include "../../Model/Task/OBBModel.h"
#include "../../Data/Dataset/OBBDataset.h"

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            /**
             * @class OBBTrainer
             * @brief Trainer for oriented bounding box (OBB) detection tasks
             *
             * Implements training for YOLO-OBB style rotated object detection models:
             * - Anchor-free detection with DFL
             * - Multi-scale predictions (P3, P4, P5)
             * - Task-aligned assignment for training
             * - Probiou-based loss for rotated boxes
             *
             * Key features:
             * - OBB loss (box + classification + DFL + angle)
             * - mAP@0.5 and mAP@0.5:0.95 metrics using Probiou
             * - Non-Maximum Suppression for OBBs using Probiou
             * - Support for various backbone architectures
             */
            class OBBTrainer : public BaseTrainer {
            public:
                /**
                 * @brief Constructor with dependency injection
                 * @param config Configuration object with OBB detection settings
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param workspace Workspace instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param progressCallback Optional progress callback
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit OBBTrainer(
                    std::shared_ptr<Config::Configuration> config,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::Workspace* workspace,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    ProgressCallback progressCallback = nullptr,
                    std::atomic<bool>* stopFlag = nullptr);

                /**
                 * @brief Destructor
                 */
                ~OBBTrainer() override = default;

            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup OBB detection model
                 *
                 * Creates OBBModel based on configuration model path.
                 */
                void setupModel() override;

                /**
                 * @brief Setup data loaders for OBB detection
                 *
                 * Creates OBBDataset and DataLoader for training.
                 */
                void setupDataLoaders() override;

                /**
                 * @brief Setup validator for OBB detection
                 *
                 * Creates OBBValidator and injects validation dataloader.
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
                 * For OBB detection, fitness is based on:
                 * - mAP@0.5 (primary metric) using Probiou
                 * - Weighted combination of mAP@0.5 and mAP@0.5:0.95
                 *
                 * @param metrics Validation metrics
                 * @return Fitness value (higher is better)
                 */
                float calculateFitness(const MetricsData& metrics) override;

                /**
                 * @brief Setup OBB predictor for inference
                 *
                 * Creates OBBPredictor with checkpoint.
                 *
                 * @param checkpointPath Path to checkpoint file
                 * @return Unique pointer to OBBPredictor
                 */
                std::unique_ptr<Predictor::BasePredictor> setupPredictor(
                    const std::string& checkpointPath) override;
            };

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
