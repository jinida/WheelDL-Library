#pragma once

#include "BaseLoss.h"
#include <torch/torch.h>

namespace WheelDL {
namespace Model {
namespace Loss {

/**
 * @brief Segmentation Loss for semantic/instance segmentation
 *
 * Combines multiple loss components:
 * - BCE Loss: Binary cross-entropy with logits (numerically stable)
 * - Dice Loss: Better boundary handling and class imbalance
 *
 * This implementation follows the BCE + Dice combination which is
 * particularly effective for segmentation tasks with multi-label
 * binary segmentation or class imbalance. The model output should be
 * logits (before sigmoid) for numerical stability.
 *
 * Formula: Loss = w_bce * BCE(logits, targets) + w_dice * Dice(sigmoid(logits), targets)
 */
class SegmentationLoss : public BaseLoss {
public:
    /**
     * @brief Construct a segmentation loss
     *
     * @param numClasses Number of segmentation classes
     * @param bceWeight Weight for BCE loss (default: 1.0)
     * @param diceWeight Weight for Dice loss (default: 1.0)
     * @param smooth Smoothing factor for Dice loss to prevent division by zero (default: 1e-6)
     */
    explicit SegmentationLoss(int64_t numClasses,
                             float bceWeight = 1.0f,
                             float diceWeight = 1.0f,
                             float smooth = 1e-6f);

    /**
     * @brief Destructor
     */
    ~SegmentationLoss() override = default;

    /**
     * @brief Compute segmentation loss with DataExample
     *
     * @param prediction Model predictions (logits) [N, num_classes, H, W]
     * @param target DataExample containing targets field with segmentation masks [N, num_classes, H, W]
     * @return Map containing {"bce": bce_loss, "dice": dice_loss, "total": total_loss}
     */
    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const Data::Dataset::DataExample& target) override;

    /**
     * @brief Compute segmentation loss with Tensor target
     *
     * @param prediction Model predictions (logits) [N, num_classes, H, W]
     * @param target Segmentation masks [N, num_classes, H, W] (binary masks)
     * @return Map containing {"bce": bce_loss, "dice": dice_loss, "total": total_loss}
     */
    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const torch::Tensor& target) override;

    /**
     * @brief Get loss name
     *
     * @return "SegmentationLoss"
     */
    [[nodiscard]] std::string name() const override {
        return "SegmentationLoss";
    }

    /**
     * @brief Set loss weights
     *
     * @param bceWeight Weight for BCE loss
     * @param diceWeight Weight for Dice loss
     */
    void setLossWeights(float bceWeight, float diceWeight) {
        _bceWeight = std::max(0.0f, bceWeight);
        _diceWeight = std::max(0.0f, diceWeight);
    }

    /**
     * @brief Get current loss weights
     *
     * @return Tuple of (bce_weight, dice_weight)
     */
    [[nodiscard]] std::tuple<float, float> getLossWeights() const {
        return std::make_tuple(_bceWeight, _diceWeight);
    }

    /**
     * @brief Set smoothing factor for Dice loss
     *
     * @param smooth Smoothing factor (must be positive)
     */
    void setSmooth(float smooth) {
        if (smooth <= 0.0f) {
            throw std::invalid_argument("Smooth factor must be positive");
        }
        _smooth = smooth;
    }

private:
    int64_t _numClasses;                      ///< Number of classes
    float _bceWeight;                         ///< Weight for BCE loss
    float _diceWeight;                        ///< Weight for Dice loss
    float _smooth;                            ///< Smoothing factor for Dice loss
    torch::nn::BCEWithLogitsLoss _bceLoss = nullptr;  ///< BCE with logits loss

    /**
     * @brief Compute Dice loss for better boundary handling
     *
     * Dice coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
     * Dice loss = 1 - Dice coefficient
     *
     * @param probs Predicted probabilities (after sigmoid) [N, C, H, W]
     * @param targets Target binary masks [N, C, H, W]
     * @return Dice loss (scalar)
     */
    [[nodiscard]] torch::Tensor computeDiceLoss(
        const torch::Tensor& probs,
        const torch::Tensor& targets);

    /**
     * @brief Compute combined loss (BCE + Dice)
     *
     * @param logits Predicted logits (before sigmoid) [N, C, H, W]
     * @param targets Target binary masks [N, C, H, W]
     * @return Tuple of (bce_loss, dice_loss, total_loss)
     */
    [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> computeCombinedLoss(
        const torch::Tensor& logits,
        const torch::Tensor& targets);
};

} // namespace Loss
} // namespace Model
} // namespace WheelDL
