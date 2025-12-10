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
                 * @brief Constructor
                 * @param config Configuration object with detection settings
                 */
                explicit DetectionTrainer(const std::shared_ptr<Config::Configuration> config);

                /**
                 * @brief Destructor
                 */
                ~DetectionTrainer() override = default;

                void drawBBoxesOnBatch(const torch::Tensor& batchData, const torch::Tensor& batchIndices, const torch::Tensor& bboxes, const torch::Tensor& classes);
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

                /**
                 * @brief Export detection prediction results
                 *
                 * Saves results as JSON file and optionally visualization images.
                 * Format includes:
                 * - Image path
                 * - Bounding boxes (x1, y1, x2, y2)
                 * - Class IDs and names
                 * - Confidence scores
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
