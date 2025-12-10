#pragma once

#include "../Engine/BaseValidator.h"
#include "../../Model/Task/SegmentationModel.h"

namespace WheelDL {
    namespace Core {
        namespace Validator {

            /**
             * @class SegmentationValidator
             * @brief Validator for semantic/instance segmentation tasks
             *
             * Computes metrics for segmentation:
             * - Mean IoU (mIoU) - Intersection over Union averaged across classes
             * - F1 Score - Harmonic mean of Precision and Recall
             * - Pixel Accuracy
             * - Precision / Recall
             * - Optimal threshold search (maximizes F1 score)
             *
             * Supports:
             * - Multi-class semantic segmentation
             * - Binary segmentation
             */
            class SegmentationValidator : public BaseValidator {
            public:
                /**
                 * @brief Constructor
                 * @param config Configuration object
                 */
                explicit SegmentationValidator(const std::shared_ptr<Config::Configuration> config);

                /**
                 * @brief Destructor
                 */
                ~SegmentationValidator() override = default;

            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup data loader (empty - uses dependency injection)
                 *
                 * SegmentationValidator uses DataLoader injected by SegmentationTrainer.
                 */
                void setupDataLoader() override;

                /**
                 * @brief Preprocess batch before forward pass
                 * @param batch Input batch from dataloader
                 * @return Preprocessed batch with data moved to device
                 */
                WheelDL::Data::Dataset::DataExample preprocessBatch(
                    const WheelDL::Data::Dataset::DataExample& batch) override;

                /**
                 * @brief Postprocess model output
                 *
                 * Applies sigmoid to logits for binary classification per pixel.
                 *
                 * @param prediction Model output tensors (logits)
                 * @return Processed prediction tensor (probabilities) [N, C, H, W]
                 */
                torch::Tensor postprocessBatch(
                    const std::vector<torch::Tensor>& prediction) override;

                /**
                 * @brief Compute segmentation metrics with optimal threshold search
                 *
                 * Evaluates multiple thresholds (0.1 ~ 0.9) to find optimal F1 score.
                 * Processes in chunks to reduce memory usage.
                 *
                 * @param pred Model predictions (probabilities) [N, C, H, W]
                 * @param target Ground truth masks (one-hot) [N, C, H, W]
                 * @return MetricsData with:
                 *         - mAP: Mean IoU (mIoU)
                 *         - f1Score: F1 score at optimal threshold
                 *         - accuracy: Pixel accuracy
                 *         - precision: Mean precision across classes
                 *         - recall: Mean recall across classes
                 *         - threshold: Optimal threshold value
                 *         - fitness: Same as mIoU
                 */
                MetricsData computeMetrics(
                    const torch::Tensor& pred,
                    const torch::Tensor& target) override;

            private:
                int _numClasses;
            };

        } // namespace Validator
    } // namespace Core
} // namespace WheelDL
