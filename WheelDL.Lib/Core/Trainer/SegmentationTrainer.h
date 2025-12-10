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
                 * @brief Constructor
                 * @param config Configuration object with segmentation settings
                 */
                explicit SegmentationTrainer(const std::shared_ptr<Config::Configuration> config);

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

                /**
                 * @brief Export segmentation prediction results
                 *
                 * Saves results as JSON file with contour data.
                 * Format includes:
                 * - Image path
                 * - Segmentation contours per class
                 * - Class IDs
                 * - Inference time
                 *
                 * @param results Vector of prediction results
                 * @param imagePaths Vector of image paths
                 */
                void exportPredictionResults(
                    const std::vector<PredictionResult>& results,
                    const std::vector<std::string>& imagePaths) override;

            private:
                std::string _modelYamlPath;  ///< Path to model YAML configuration
            };

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
