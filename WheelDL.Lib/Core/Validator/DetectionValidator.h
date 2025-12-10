#pragma once

#include "../Engine/BaseValidator.h"
#include "../../Model/Task/DetectionModel.h"
#include "../../Model/Utils/IoU.h"

namespace WheelDL {
    namespace Core {
        namespace Validator {

            /**
             * @class DetectionValidator
             * @brief Validator for object detection tasks
             *
             * Computes metrics for detection:
             * - mAP@0.5 (PASCAL VOC style)
             * - mAP@0.5:0.95 (COCO style)
             * - Precision, Recall at various IoU thresholds
             *
             * Supports:
             * - Non-Maximum Suppression (NMS) via Model::Utils::nonMaxSuppression
             * - Multi-class detection
             * - Anchor-free detection
             */
            class DetectionValidator : public BaseValidator {
            public:
                /**
                 * @brief Constructor
                 * @param config Configuration object
                 */
                explicit DetectionValidator(const std::shared_ptr<Config::Configuration> config);

                /**
                 * @brief Destructor
                 */
                ~DetectionValidator() override = default;

            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup data loader (empty - uses dependency injection)
                 *
                 * DetectionValidator uses DataLoader injected by DetectionTrainer.
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
                 * Applies NMS using Model::Utils::nonMaxSuppression.
                 *
                 * @param prediction Model output tensors (multi-scale predictions)
                 * @return Processed prediction tensor with decoded boxes [N, 6] (x1, y1, x2, y2, conf, class)
                 */
                torch::Tensor postprocessBatch(
                    const std::vector<torch::Tensor>& prediction) override;

                /**
                 * @brief Compute detection metrics
                 *
                 * @param pred Model predictions [N, 6] (x1, y1, x2, y2, conf, class)
                 * @param target Ground truth [M, 4] (x, y, w, h) with separate classes tensor
                 * @return MetricsData with:
                 *         - mAP: mAP@0.5:0.95
                 *         - precision: Precision at IoU=0.5
                 *         - recall: Recall at IoU=0.5
                 *         - fitness: Same as mAP
                 */
                MetricsData computeMetrics(
                    const torch::Tensor& pred,
                    const torch::Tensor& target) override;

            private:
                int _numClasses;        ///< Number of classes
                float _confThresh;      ///< Confidence threshold
                float _iouThresh;       ///< IoU threshold for NMS
                int _maxDet;            ///< Maximum detections per image
            };

        } // namespace Validator
    } // namespace Core
} // namespace WheelDL
