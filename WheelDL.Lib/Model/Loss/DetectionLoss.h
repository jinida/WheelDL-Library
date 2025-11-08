#pragma once

#include "BaseLoss.h"
#include "../Utils/TaskAlignedAssigner.h"
#include <torch/torch.h>
#include <memory>

namespace WheelDL {
    namespace Model {
        namespace Loss {

            // Forward declarations for internal classes
            class DFLoss;
            class BboxLoss;

            /**
             * @brief Complete Detection Loss for YOLO-style object detection
             *
             * Integrates:
             * 1. Task-Aligned Assignment for GT matching
             * 2. BboxLoss with CIoU and DFL
             * 3. Classification loss (BCE)
             * 4. Automatic anchor generation and bbox decoding
             *
             * Key improvements over Python version:
             * - Zero-copy tensor operations where possible
             * - Efficient batch processing
             * - Smart memory management with move semantics
             * - Pre-allocated tensors for repeated operations
             * - Template-based polymorphism for different tasks
             *
             * Usage:
             * @code
             * DetectionLoss loss(model);
             * loss.setLossWeights(7.5f, 0.5f, 1.5f);  // box, cls, dfl
             * auto [totalLoss, detachedLoss] = loss(predictions, batch);
             * @endcode
             */
            class DetectionLoss : public BaseLoss {
            public:
                /**
                 * @brief Construct detection loss
                 *
                 * @param numClasses Number of object classes
                 * @param stride Stride values for each detection layer
                 * @param boxGain Box loss weight (default: 7.5)
                 * @param clsGain Classification loss weight (default: 0.5)
                 * @param dflGain DFL loss weight (default: 1.5)
                 * @param talTopk Top-k for task-aligned assignment (default: 10)
                 */
                explicit DetectionLoss(
                    int64_t numClasses,
                    const torch::Tensor& stride,
                    float boxGain = 7.5f,
                    float clsGain = 0.5f,
                    float dflGain = 1.5f,
                    int64_t talTopk = 10
                );

                /**
                 * @brief Destructor
                 */
                ~DetectionLoss() = default;

                /**
                 * @brief Compute detection loss with concatenated predictions (inference format)
                 *
                 * @param prediction Model predictions [batch, 144, 8400] (concatenated from all scales)
                 * @param target DataExample containing targets, classes, and batchIndices
                 * @return Map containing {"box": box_loss, "cls": cls_loss, "dfl": dfl_loss, "total": total_loss}
                 */
                [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
                    const torch::Tensor& prediction,
                    const Data::Dataset::DataExample& target) override;

                /**
                 * @brief Compute detection loss with multi-scale predictions (training format)
                 *
                 * @param predictions Vector of predictions for each scale:
                 *                    - [batch, 144, 80, 80] for P3
                 *                    - [batch, 144, 40, 40] for P4
                 *                    - [batch, 144, 20, 20] for P5
                 *                    where 144 = 4*regMax + numClasses (typically 4*16 + 80)
                 * @param target DataExample containing targets, classes, and batchIndices
                 * @return Map containing {"box": box_loss, "cls": cls_loss, "dfl": dfl_loss, "total": total_loss}
                 */
                [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
                    const std::vector<torch::Tensor>& predictions,
                    const Data::Dataset::DataExample& target) override;

                // Note: compute(Tensor, Tensor) is not implemented.
                // Calling it will throw runtime_error from BaseLoss default implementation.

                /**
                 * @brief Get loss name
                 */
                [[nodiscard]] std::string name() const override {
                    return "DetectionLoss";
                }

                /**
                 * @brief Set loss weights
                 *
                 * @param box Box loss weight (default: 7.5)
                 * @param cls Classification loss weight (default: 0.5)
                 * @param dfl DFL loss weight (default: 1.5)
                 */
                void setLossWeights(float box, float cls, float dfl) {
                    _boxGain = box;
                    _clsGain = cls;
                    _dflGain = dfl;
                }

                /**
                 * @brief Get loss weights
                 *
                 * @return Tuple of (box_gain, cls_gain, dfl_gain)
                 */
                [[nodiscard]] std::tuple<float, float, float> getLossWeights() const {
                    return std::make_tuple(_boxGain, _clsGain, _dflGain);
                }

                /**
                 * @brief Set device for loss computation
                 *
                 * @param device Target device (CPU or CUDA)
                 */
                void to(const torch::Device& device);

            private:
                /**
                 * @brief Preprocess targets from batch format to model format
                 *
                 * @param targets Raw targets [num_gt, 6] as [batch_idx, class, x, y, w, h]
                 * @param batchSize Batch size
                 * @param scaleTensor Scale tensor for denormalizing coordinates
                 * @return Preprocessed targets [batch_size, max_num_gt, 5]
                 */
                [[nodiscard]] torch::Tensor preprocess(
                    const torch::Tensor& targets,
                    int64_t batchSize,
                    const torch::Tensor& scaleTensor
                );

                /**
                 * @brief Decode distance predictions to bounding boxes
                 *
                 * @param anchorPoints Anchor points [num_anchors, 2]
                 * @param predDist Distance distributions [batch, num_anchors, 4*regMax]
                 * @return Decoded boxes [batch, num_anchors, 4] in xyxy format
                 */
                [[nodiscard]] torch::Tensor decodeBbox(
                    const torch::Tensor& anchorPoints,
                    const torch::Tensor& predDist
                );

            private:
                // Loss components
                std::unique_ptr<BCEWithLogitsLoss> _bce;  ///< BCE loss for classification
                std::unique_ptr<BboxLoss> _bboxLoss;      ///< Bbox loss with CIoU + DFL
                std::unique_ptr<Utils::TaskAlignedAssigner> _assigner;  ///< Task-aligned assigner

                // Model parameters
                int64_t _numClasses;   ///< Number of classes
                int64_t _regMax;       ///< DFL regression max
                int64_t _numOutputs;   ///< Outputs per anchor
                torch::Tensor _stride; ///< Stride for each layer
                torch::Tensor _proj;   ///< Projection tensor for DFL decoding [regMax]

                // Loss weights
                float _boxGain;  ///< Box loss weight
                float _clsGain;  ///< Classification loss weight
                float _dflGain;  ///< DFL loss weight

                // Cached values for performance optimization
                float _totalStrideFactor;  ///< Pre-computed stride factor (sum of 1/(stride^2))

                // Flags
                bool _useDfl;    ///< Whether to use DFL

                // Device
                torch::Device _device;
            };

        } // namespace Loss
    } // namespace Model
} // namespace WheelDL
