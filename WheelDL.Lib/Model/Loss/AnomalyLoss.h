#pragma once

#include "BaseLoss.h"
#include <torch/torch.h>

namespace WheelDL {
    namespace Model {
        namespace Loss {

            /**
             * @brief Anomaly Detection Loss
             *
             * Supports multiple loss types for anomaly detection:
             * - MSE (Mean Squared Error): Simple reconstruction error
             * - SSIM (Structural Similarity Index): Perceptual reconstruction quality
             * - Perceptual Loss: Feature-level reconstruction (using pre-trained network)
             *
             * For autoencoder-based anomaly detection, the model learns to
             * reconstruct normal samples. Anomalies produce high reconstruction error.
             *
             * For embedding-based methods (e.g., PatchCore), contrastive or
             * triplet loss can be used instead.
             */
            class AnomalyLoss : public BaseLoss {
            public:
                /**
                 * @brief Loss type for anomaly detection
                 */
                enum class LossType {
                    SimpleNet,
					EfficientAD,
                    PatchCore
                };

                /**
                 * @brief Construct an anomaly detection loss
                 *
                 * @param lossType Type of loss to use (default: MSE)
                 */
                explicit AnomalyLoss(LossType lossType = LossType::SimpleNet);

                /**
                 * @brief Destructor
                 */
                ~AnomalyLoss() override = default;

                /**
                 * @brief Compute anomaly detection loss with DataExample
                 *
                 * @param prediction Reconstructed images [N, C, H, W]
                 * @param target DataExample containing data field with original images [N, C, H, W]
                 * @return Map containing {"total": loss_value}
                 */
                [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
                    const torch::Tensor& prediction,
                    const Data::Dataset::DataExample& target) override;

                /**
                 * @brief Compute anomaly detection loss with Tensor target
                 *
                 * @param prediction Reconstructed images [N, C, H, W]
                 * @param target Original images [N, C, H, W]
                 * @return Map containing {"total": loss_value}
                 */
                [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
                    const torch::Tensor& prediction,
                    const torch::Tensor& target) override;

                [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
                    const std::vector<torch::Tensor>& predictions,
					const Data::Dataset::DataExample& target) override;

                /**
                 * @brief Get loss name
                 *
                 * @return Loss name string
                 */
                [[nodiscard]] std::string name() const override;

                /**
                 * @brief Set loss type
                 *
                 * @param lossType New loss type
                 */
                void setLossType(LossType lossType) {
                    _lossType = lossType;
                }

                /**
                 * @brief Get current loss type
                 *
                 * @return Current loss type
                 */
                [[nodiscard]] LossType getLossType() const {
                    return _lossType;
                }

            private:
                LossType _lossType;

                /**
                 * @brief Private helper methods for specific loss types
                 */
                [[nodiscard]] std::unordered_map<std::string, torch::Tensor> computeEfficientAD(
                    const std::vector<torch::Tensor>& predictions);

                [[nodiscard]] std::unordered_map<std::string, torch::Tensor> computePatchCore(
                    const std::vector<torch::Tensor>& predictions);

                [[nodiscard]] std::unordered_map<std::string, torch::Tensor> computeSimpleNet(
                    const std::vector<torch::Tensor>& predictions);
            };
        } // namespace Loss
    } // namespace Model
} // namespace WheelDL
