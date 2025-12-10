#pragma once

#include "../Engine/BaseValidator.h"
#include "../../Model/Utils/IoU.h"

namespace WheelDL {
    namespace Core {
        namespace Validator {

            /**
             * @class OBBValidator
             * @brief Validator for oriented bounding box (OBB) detection tasks
             *
             * Computes mAP metrics using Probiou for rotated box IoU computation.
             * Follows the same protocol as DetectionValidator but adapted for OBBs.
             *
             * Metrics:
             * - mAP@0.5 (using Probiou threshold 0.5)
             * - mAP@0.5:0.95 (average over thresholds)
             * - Precision and Recall at IoU=0.5
             */
            class OBBValidator : public BaseValidator {
            public:
                /**
                 * @brief Constructor
                 * @param config Configuration object containing validation settings
                 */
                explicit OBBValidator(const std::shared_ptr<Config::Configuration> config);

                /**
                 * @brief Destructor
                 */
                ~OBBValidator() override = default;

            protected:
                /**
                 * @brief Setup validation DataLoader
                 *
                 * Creates OBBDataset for validation if not injected.
                 */
                void setupDataLoader() override;

                /**
                 * @brief Preprocess batch before evaluation
                 * @param batch Input batch from dataloader
                 * @return Preprocessed batch
                 */
                WheelDL::Data::Dataset::DataExample preprocessBatch(
                    const WheelDL::Data::Dataset::DataExample& batch) override;

                /**
                 * @brief Postprocess model output
                 * @param prediction Raw model predictions
                 * @return Processed predictions
                 */
                torch::Tensor postprocessBatch(
                    const std::vector<torch::Tensor>& prediction) override;

                /**
                 * @brief Compute OBB detection metrics
                 *
                 * Calculates mAP using Probiou for IoU computation between OBBs.
                 *
                 * @param pred Predicted OBBs after NMS
                 * @param target Ground truth OBBs
                 * @return MetricsData with mAP, precision, recall
                 */
                MetricsData computeMetrics(
                    const torch::Tensor& pred,
                    const torch::Tensor& target) override;

            private:
                int _numClasses;    ///< Number of classes
                float _confThresh;  ///< Confidence threshold for filtering
                float _iouThresh;   ///< IoU threshold for NMS
                int _maxDet;        ///< Maximum detections per image
            };

        } // namespace Validator
    } // namespace Core
} // namespace WheelDL
