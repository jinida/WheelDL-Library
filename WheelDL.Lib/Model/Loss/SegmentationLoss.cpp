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
                    if (target.dim() != 3) {
                        throw std::invalid_argument("Target masks must be 3D tensor [N, H, W]");
                    }
                }
            }

            SegmentationLoss::SegmentationLoss(int64_t numClasses,
                float diceWeight,
                const torch::Tensor& classWeights)
                : _numClasses(numClasses)
                , _diceWeight(std::clamp(diceWeight, 0.0f, 1.0f)) {

                // Configure cross-entropy loss with optional class weights
                auto options = torch::nn::CrossEntropyLossOptions();

                if (classWeights.defined() && classWeights.numel() > 0) {
                    options.weight(classWeights);
                }

                _ceLoss = torch::nn::CrossEntropyLoss(options);
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
                auto loss = computeCombinedLoss(prediction, target);

                // Validate loss for NaN/Inf values
                validateLoss(loss, "SegmentationLoss");
                return {{"total", loss}};
            }

            torch::Tensor SegmentationLoss::computeCombinedLoss(
                const torch::Tensor& prediction,
                const torch::Tensor& target) {
                // Compute cross-entropy loss
                auto ceLoss = _ceLoss->forward(prediction, target);

                // Only compute Dice loss if weight is non-zero (performance optimization)
                if (_diceWeight > std::numeric_limits<float>::epsilon()) {
                    auto diceLoss = computeDiceLoss(prediction, target);
                    return ceLoss * (1.0f - _diceWeight) + diceLoss * _diceWeight;
                }

                return ceLoss;
            }

            torch::Tensor SegmentationLoss::computeDiceLoss(
                const torch::Tensor& prediction,
                const torch::Tensor& target) {

                // Convert logits to probabilities
                auto probs = torch::softmax(prediction, /*dim=*/1);

                // Convert target to one-hot encoding [N, H, W] -> [N, num_classes, H, W]
                // Optimized: Use contiguous() to minimize memory copies during permute
                auto targetOneHot = torch::one_hot(target, _numClasses);

                // Permute from [N, H, W, num_classes] to [N, num_classes, H, W]
                // Use contiguous() to ensure efficient memory layout for subsequent operations
                targetOneHot = targetOneHot.permute({ 0, 3, 1, 2 }).contiguous().to(probs.dtype());

                // Compute Dice coefficient for each class
                // Fixed: Use unified constants from Constants.h
                using namespace Constants;
                constexpr float EPSILON = DICE_EPSILON;

                // Intersection: sum over H and W dimensions
                auto intersection = (probs * targetOneHot).sum({ 2, 3 });

                // Union: sum of prediction and target over H and W
                auto predSum = probs.sum({ 2, 3 });
                auto targetSum = targetOneHot.sum({ 2, 3 });

                // Dice coefficient = 2 * intersection / (pred + target)
                auto dice = (2.0f * intersection + EPSILON) / (predSum + targetSum + EPSILON);

                // Dice loss = 1 - mean(dice across classes and batch)
                return 1.0f - dice.mean();
            }

        } // namespace Loss
    } // namespace Model
} // namespace WheelDL
