#pragma once

#include "BaseLoss.h"
#include <torch/torch.h>
#include <memory>

namespace WheelDL {
    namespace Model {
        namespace Loss {

            /**
             * @brief Classification Loss with multiple loss types
             *
             * Supports various loss functions for image classification:
             * - Cross-Entropy Loss (standard classification loss)
             * - Focal Loss (for imbalanced datasets)
             * - Label Smoothing Cross-Entropy
             *
             * Key improvements over Python version:
             * - Efficient LibTorch implementations
             * - Configurable label smoothing
             * - Focal loss with optimized computation
             * - Support for both hard and soft labels
             */
            class ClassificationLoss : public BaseLoss {
            public:
                /**
                 * @brief Loss type enum
                 */
                enum class LossType {
                    CROSS_ENTROPY,      ///< Standard cross-entropy loss
                    FOCAL,              ///< Focal loss for imbalanced classes
                    LABEL_SMOOTHING     ///< Cross-entropy with label smoothing
                };

                /**
                 * @brief Construct a classification loss
                 *
                 * @param lossType Type of loss function (default: CROSS_ENTROPY)
                 * @param numClasses Number of classes (required for some loss types)
                 * @param labelSmoothing Label smoothing factor (0.0 = no smoothing, default: 0.0)
                 * @param focalAlpha Focal loss alpha parameter (default: 0.25)
                 * @param focalGamma Focal loss gamma parameter (default: 2.0)
                 * @param reduction Reduction method: "none", "mean", "sum" (default: "mean")
                 */
                explicit ClassificationLoss(
                    LossType lossType = LossType::CROSS_ENTROPY,
                    int64_t numClasses = 1000,
                    float labelSmoothing = 0.0f,
                    float focalAlpha = 0.25f,
                    float focalGamma = 2.0f,
                    const std::string& reduction = "mean"
                );

                /**
                 * @brief Destructor
                 */
                ~ClassificationLoss() override = default;

                /**
                 * @brief Compute classification loss with DataExample
                 *
                 * @param prediction Model predictions [N, num_classes] (logits)
                 * @param target DataExample containing classes field [N] (class indices)
                 * @return Map containing {"total": loss_value}
                 */
                [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
                    const torch::Tensor& prediction,
                    const Data::Dataset::DataExample& target) override;

                /**
                 * @brief Compute classification loss with Tensor target
                 *
                 * @param prediction Model predictions [N, num_classes] (logits)
                 * @param target Class indices [N] or soft labels [N, num_classes]
                 * @return Map containing {"total": loss_value}
                 */
                [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
                    const torch::Tensor& prediction,
                    const torch::Tensor& target) override;

                /**
                 * @brief Get loss name
                 *
                 * @return Loss function name
                 */
                [[nodiscard]] std::string name() const override;

                /**
                 * @brief Set label smoothing factor
                 *
                 * @param smoothing Smoothing factor in [0, 1]
                 */
                void setLabelSmoothing(float smoothing) {
                    _labelSmoothing = std::clamp(smoothing, 0.0f, 1.0f);
                }

                /**
                 * @brief Set focal loss parameters
                 *
                 * @param alpha Alpha parameter (class weighting)
                 * @param gamma Gamma parameter (focusing parameter, must be in [0.0, 5.0])
                 * @throws std::invalid_argument if gamma is out of valid range
                 */
                void setFocalParams(float alpha, float gamma) {
                    if (gamma < 0.0f || gamma > 5.0f) {
                        throw std::invalid_argument("Focal gamma must be in range [0.0, 5.0] to prevent overflow");
                    }
                    _focalAlpha = alpha;
                    _focalGamma = gamma;
                }

                /**
                 * @brief Get current loss type
                 *
                 * @return Loss type
                 */
                [[nodiscard]] LossType getLossType() const {
                    return _lossType;
                }

            private:
                /**
                 * @brief Compute standard cross-entropy loss
                 */
                [[nodiscard]] torch::Tensor computeCrossEntropy(
                    const torch::Tensor& prediction,
                    const torch::Tensor& target
                );

                /**
                 * @brief Compute focal loss
                 *
                 * Focal loss down-weights easy examples and focuses on hard examples.
                 * Formula: FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
                 */
                [[nodiscard]] torch::Tensor computeFocalLoss(
                    const torch::Tensor& prediction,
                    const torch::Tensor& target
                );

                /**
                 * @brief Compute cross-entropy with label smoothing
                 *
                 * Label smoothing: y_smooth = y * (1 - smoothing) + smoothing / num_classes
                 */
                [[nodiscard]] torch::Tensor computeLabelSmoothingLoss(
                    const torch::Tensor& prediction,
                    const torch::Tensor& target
                );

                /**
                 * @brief Apply reduction to loss tensor
                 */
                [[nodiscard]] torch::Tensor applyReduction(const torch::Tensor& loss);

            private:
                LossType _lossType;           ///< Type of loss function
                int64_t _numClasses;          ///< Number of classes
                float _labelSmoothing;        ///< Label smoothing factor
                float _focalAlpha;            ///< Focal loss alpha parameter
                float _focalGamma;            ///< Focal loss gamma parameter
                std::string _reduction;       ///< Reduction method
            };

        } // namespace Loss
    } // namespace Model
} // namespace WheelDL
