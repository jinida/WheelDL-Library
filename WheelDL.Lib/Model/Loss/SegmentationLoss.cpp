#include "pch.h"
#include "SegmentationLoss.h"
#include "../Constants.h"
#include <algorithm>

namespace WheelDL {
    namespace Model {
        namespace Loss {

            // Helper function to validate tensor dimensions
            namespace {
                inline void validateSegmentationTensors(const torch::Tensor& prediction, const torch::Tensor& target) {
                    if (prediction.dim() != 4) {
                        throw std::invalid_argument("Prediction must be 4D tensor [N, num_classes, H, W]");
                    }
                    if (target.dim() != 4) {
                        throw std::invalid_argument("Target masks must be 4D tensor [N, num_classes, H, W]");
                    }
                    if (prediction.size(0) != target.size(0)) {
                        throw std::invalid_argument("Prediction and target batch sizes must match");
                    }
                    if (prediction.size(1) != target.size(1)) {
                        throw std::invalid_argument("Prediction and target channel sizes must match");
                    }
                    if (prediction.size(2) != target.size(2) || prediction.size(3) != target.size(3)) {
                        throw std::invalid_argument("Prediction and target spatial dimensions must match");
                    }
                }
            }

            SegmentationLoss::SegmentationLoss(int64_t numClasses,
                float bceWeight,
                float diceWeight,
                float smooth)
                : _numClasses(numClasses)
                , _bceWeight(std::max(0.0f, bceWeight))
                , _diceWeight(std::max(0.0f, diceWeight))
                , _smooth(smooth) {

                if (_numClasses <= 0) {
                    throw std::invalid_argument("Number of classes must be positive");
                }

                if (_smooth <= 0.0f) {
                    throw std::invalid_argument("Smooth factor must be positive");
                }

                // Configure BCE with logits loss for numerical stability
                _bceLoss = torch::nn::BCEWithLogitsLoss();
            }

            std::unordered_map<std::string, torch::Tensor> SegmentationLoss::compute(
                const torch::Tensor& prediction,
                const Data::Dataset::DataExample& target) {
                // Delegate to tensor overload
                return compute(prediction, target.targets);
            }

            std::unordered_map<std::string, torch::Tensor> SegmentationLoss::compute(
                const torch::Tensor& prediction,
                const torch::Tensor& target) {
                // Validate inputs
                validateSegmentationTensors(prediction, target);

                // Compute combined loss
                auto [bceLoss, diceLoss, totalLoss] = computeCombinedLoss(prediction, target);

                // Validate loss for NaN/Inf values
                validateLoss(totalLoss, "SegmentationLoss");

                return {
                    {"bce", bceLoss.detach()},
                    {"dice", diceLoss.detach()},
                    {"total", totalLoss}
                };
            }

            std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SegmentationLoss::computeCombinedLoss(
                const torch::Tensor& logits,
                const torch::Tensor& targets) {
                // Compute BCE loss (BCEWithLogitsLoss handles sigmoid internally)
                auto bceLoss = _bceLoss->forward(logits, targets);

                // Compute Dice loss (apply sigmoid for probability)
                torch::Tensor diceLoss;
                if (_diceWeight > std::numeric_limits<float>::epsilon()) {
                    auto probs = torch::sigmoid(logits);
                    diceLoss = computeDiceLoss(probs, targets);
                }
                else {
                    // If dice weight is zero, create a zero tensor for consistency
                    diceLoss = torch::zeros({}, logits.options());
                }

                // Compute total loss
                auto totalLoss = _bceWeight * bceLoss + _diceWeight * diceLoss;

                return std::make_tuple(bceLoss, diceLoss, totalLoss);
            }

            torch::Tensor SegmentationLoss::computeDiceLoss(
                const torch::Tensor& probs,
                const torch::Tensor& targets) {
                /**
                 * Dice Loss implementation
                 *
                 * Args:
                 *   probs: Predicted probabilities [N, C, H, W] (after sigmoid)
                 *   targets: Target binary masks [N, C, H, W]
                 *
                 * Formula:
                 *   intersection = sum(probs * targets, dim=[H, W])
                 *   union = sum(probs, dim=[H, W]) + sum(targets, dim=[H, W])
                 *   dice = (2 * intersection + smooth) / (union + smooth)
                 *   loss = 1 - mean(dice)
                 */

                // Intersection: sum over H and W dimensions -> [N, C]
                auto intersection = (probs * targets).sum({ 2, 3 });

                // Union: sum of prediction and target over H and W -> [N, C]
                auto predSum = probs.sum({ 2, 3 });
                auto targetSum = targets.sum({ 2, 3 });

                // Dice coefficient = 2 * intersection / (pred + target)
                auto dice = (2.0f * intersection + _smooth) / (predSum + targetSum + _smooth);

                // Dice loss = 1 - mean(dice across classes and batch)
                return 1.0f - dice.mean();
            }

        } // namespace Loss
    } // namespace Model
} // namespace WheelDL
