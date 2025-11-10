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
 * - Cross-Entropy Loss: Pixel-wise classification
 * - Dice Loss: Better boundary handling and class imbalance
 * - Focal Loss: Hard example mining (optional)
 *
 * The combination of Cross-Entropy and Dice Loss is particularly
 * effective for segmentation tasks with imbalanced classes or
 * objects with complex boundaries.
 */
class SegmentationLoss : public BaseLoss {
public:
    /**
     * @brief Construct a segmentation loss
     *
     * @param numClasses Number of segmentation classes
     * @param diceWeight Weight for Dice loss (default: 0.5, range: [0, 1])
     *                   Set to 0 for pure cross-entropy, 1 for pure Dice
     * @param classWeights Optional class weights for handling imbalanced datasets
     */
    explicit SegmentationLoss(int64_t numClasses,
                             float diceWeight = 0.5f,
                             const torch::Tensor& classWeights = torch::Tensor());

    /**
     * @brief Destructor
     */
    ~SegmentationLoss() override = default;

    /**
     * @brief Compute segmentation loss with DataExample
     *
     * @param prediction Model predictions (logits) [N, num_classes, H, W]
     * @param target DataExample containing targets field with segmentation masks [N, H, W]
     * @return Map containing {"total": loss_value}
     */
    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const Data::Dataset::DataExample& target) override;

    /**
     * @brief Compute segmentation loss with Tensor target
     *
     * @param prediction Model predictions (logits) [N, num_classes, H, W]
     * @param target Segmentation masks [N, H, W]
     * @return Map containing {"total": loss_value}
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
     * @brief Set Dice loss weight
     *
     * @param weight Dice loss weight in [0, 1]
     */
    void setDiceWeight(float weight) {
        _diceWeight = std::clamp(weight, 0.0f, 1.0f);
    }

    /**
     * @brief Get current Dice loss weight
     *
     * @return Dice loss weight
     */
    [[nodiscard]] float getDiceWeight() const {
        return _diceWeight;
    }

private:
    int64_t _numClasses;                      ///< Number of classes
    float _diceWeight;                        ///< Weight for Dice loss
    torch::nn::CrossEntropyLoss _ceLoss = nullptr;      ///< Cross-entropy loss

    /**
     * @brief Compute Dice loss for better boundary handling
     *
     * Dice coefficient = 2 * |X ∩ Y| / (|X| + |Y|)
     * Dice loss = 1 - Dice coefficient
     *
     * @param prediction Predicted probabilities [N, num_classes, H, W]
     * @param target Target masks [N, H, W]
     * @return Dice loss (scalar)
     */
    [[nodiscard]] torch::Tensor computeDiceLoss(
        const torch::Tensor& prediction,
        const torch::Tensor& target);

    /**
     * @brief Compute combined loss (CE + Dice)
     *
     * @param prediction Predicted logits [N, num_classes, H, W]
     * @param target Target masks [N, H, W]
     * @return Combined loss (scalar)
     */
    [[nodiscard]] torch::Tensor computeCombinedLoss(
        const torch::Tensor& prediction,
        const torch::Tensor& target);
};

} // namespace Loss
} // namespace Model
} // namespace WheelDL
