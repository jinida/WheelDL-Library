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
                    MSE,        ///< Mean Squared Error (simple, fast)
                    SSIM,       ///< Structural Similarity Index (perceptual)
                    COMBINED    ///< MSE + SSIM combination
                };

                /**
                 * @brief Construct an anomaly detection loss
                 *
                 * @param lossType Type of loss to use (default: MSE)
                 * @param ssimWeight Weight for SSIM in combined mode (default: 0.5)
                 */
                explicit AnomalyLoss(LossType lossType = LossType::MSE,
                    float ssimWeight = 0.5f);

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

                /**
                 * @brief Set SSIM weight for combined mode
                 *
                 * @param weight SSIM weight in [0, 1]
                 */
                void setSSIMWeight(float weight) {
                    _ssimWeight = std::clamp(weight, 0.0f, 1.0f);
                }

            private:
                LossType _lossType;               ///< Type of loss
                float _ssimWeight;                ///< Weight for SSIM in combined mode
                torch::nn::MSELoss _mseLoss;      ///< MSE loss

                /**
                 * @brief Compute SSIM (Structural Similarity Index)
                 *
                 * SSIM measures perceptual similarity between two images.
                 * Range: [-1, 1], where 1 means identical images.
                 *
                 * @param img1 First image [N, C, H, W]
                 * @param img2 Second image [N, C, H, W]
                 * @return SSIM value (scalar)
                 */
                [[nodiscard]] torch::Tensor computeSSIM(const torch::Tensor& img1,
                    const torch::Tensor& img2);

                /**
                 * @brief Compute SSIM loss (1 - SSIM)
                 *
                 * @param img1 First image [N, C, H, W]
                 * @param img2 Second image [N, C, H, W]
                 * @return SSIM loss (scalar)
                 */
                [[nodiscard]] torch::Tensor computeSSIMLoss(const torch::Tensor& img1,
                    const torch::Tensor& img2) {
                    return 1.0f - computeSSIM(img1, img2);
                }
            };

        } // namespace Loss
    } // namespace Model
} // namespace WheelDL
