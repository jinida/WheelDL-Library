#include "ClassificationLoss.h"
#include <stdexcept>
#include <algorithm>
#include <array>
#include <string_view>

namespace WheelDL {
    namespace Model {
        namespace Loss {

            // Valid reduction types
            namespace {
                constexpr std::array<std::string_view, 3> VALID_REDUCTIONS = {"none", "mean", "sum"};
            }

            ClassificationLoss::ClassificationLoss(
                LossType lossType,
                int64_t numClasses,
                float labelSmoothing,
                float focalAlpha,
                float focalGamma,
                const std::string& reduction
            )
                : _lossType(lossType)
                , _numClasses(numClasses)
                , _labelSmoothing(std::clamp(labelSmoothing, 0.0f, 1.0f))
                , _focalAlpha(focalAlpha)
                , _focalGamma(focalGamma)
                , _reduction(reduction) {

                // Validate parameters
                if (_numClasses <= 0) {
                    throw std::invalid_argument("Number of classes must be positive");
                }

                // Validate focal gamma to prevent numerical overflow in pow operation
                if (_focalGamma < 0.0f || _focalGamma > 5.0f) {
                    throw std::invalid_argument("Focal gamma must be in range [0.0, 5.0] to prevent overflow");
                }

                // Use std::array and std::string_view for efficient validation
                if (std::find(VALID_REDUCTIONS.begin(), VALID_REDUCTIONS.end(), std::string_view(_reduction))
                    == VALID_REDUCTIONS.end()) {
                    throw std::invalid_argument("Reduction must be 'none', 'mean', or 'sum'");
                }
            }

            std::unordered_map<std::string, torch::Tensor> ClassificationLoss::compute(
                const torch::Tensor& prediction,
                const Data::Dataset::DataExample& target) {
                // Fixed: Delegate to tensor overload to avoid code duplication
                return compute(prediction, target.classes);
            }

            std::unordered_map<std::string, torch::Tensor> ClassificationLoss::compute(
                const torch::Tensor& prediction,
                const torch::Tensor& target) {
                // Validate inputs
                if (prediction.dim() != 2) {
                    throw std::invalid_argument("Prediction must be 2D tensor [N, num_classes]");
                }

                if (target.dim() != 1 && target.dim() != 2) {
                    throw std::invalid_argument("Target must be 1D [N] (hard labels) or 2D [N, num_classes] (soft labels)");
                }

                if (prediction.size(0) != target.size(0)) {
                    throw std::invalid_argument("Prediction and target batch sizes must match");
                }

                torch::Tensor loss;

                // Compute loss based on type
                switch (_lossType) {
                case LossType::CROSS_ENTROPY:
                    loss = computeCrossEntropy(prediction, target);
                    break;

                case LossType::FOCAL:
                    loss = computeFocalLoss(prediction, target);
                    break;

                case LossType::LABEL_SMOOTHING:
                    loss = computeLabelSmoothingLoss(prediction, target);
                    break;

                default:
                    throw std::runtime_error("Unknown loss type");
                }

                // Validate loss for NaN/Inf values
                validateLoss(loss, name());
                return { {"total", loss} };
            }

            std::string ClassificationLoss::name() const {
                switch (_lossType) {
                case LossType::CROSS_ENTROPY:
                    return "ClassificationLoss(CrossEntropy)";
                case LossType::FOCAL:
                    return "ClassificationLoss(Focal)";
                case LossType::LABEL_SMOOTHING:
                    return "ClassificationLoss(LabelSmoothing)";
                default:
                    return "ClassificationLoss(Unknown)";
                }
            }

            torch::Tensor ClassificationLoss::computeCrossEntropy(
                const torch::Tensor& prediction,
                const torch::Tensor& target) {

                // Handle both hard labels (1D) and soft labels (2D)
                if (target.dim() == 1) {
                    // Hard labels: use standard cross_entropy
                    auto options = torch::nn::functional::CrossEntropyFuncOptions();
                    if (_reduction == "none") {
                        options.reduction(torch::kNone);
                    }
                    else if (_reduction == "mean") {
                        options.reduction(torch::kMean);
                    }
                    else {
                        options.reduction(torch::kSum);
                    }
                    return torch::nn::functional::cross_entropy(prediction, target, options);
                }
                else {
                    // Soft labels: compute manually
                    auto logProbs = torch::log_softmax(prediction, /*dim=*/1);
                    auto loss = -(target * logProbs).sum(/*dim=*/1);
                    return applyReduction(loss);
                }
            }

            torch::Tensor ClassificationLoss::computeFocalLoss(
                const torch::Tensor& prediction,
                const torch::Tensor& target) {

                // Focal Loss: FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
                // where pt is the probability of the correct class

                // Get probabilities
                auto probs = torch::softmax(prediction, /*dim=*/1);

                // Get target probabilities
                torch::Tensor targetProbs;
                if (target.dim() == 1) {
                    // Hard labels: gather probabilities for target classes
                    targetProbs = probs.gather(1, target.unsqueeze(1)).squeeze(1);
                }
                else {
                    // Soft labels: compute weighted sum
                    targetProbs = (probs * target).sum(/*dim=*/1);
                }

                // Compute focal loss with numerical stability
                // Clamp probabilities to avoid log(0) and ensure numerical stability
                constexpr float PROB_EPSILON = 1e-7f;
                auto clampedProbs = torch::clamp(targetProbs, PROB_EPSILON, 1.0f - PROB_EPSILON);
                auto focusingFactor = torch::pow(1.0f - clampedProbs, _focalGamma);
                auto logProbs = torch::log(clampedProbs);
                auto loss = -_focalAlpha * focusingFactor * logProbs;

                return applyReduction(loss);
            }

            torch::Tensor ClassificationLoss::computeLabelSmoothingLoss(
                const torch::Tensor& prediction,
                const torch::Tensor& target) {

                // Label smoothing: y_smooth = y * (1 - smoothing) + smoothing / num_classes

                if (target.dim() == 2) {
                    // Already soft labels, apply smoothing
                    auto smoothedTarget = target * (1.0f - _labelSmoothing) +
                        _labelSmoothing / static_cast<float>(_numClasses);
                    auto logProbs = torch::log_softmax(prediction, /*dim=*/1);
                    auto loss = -(smoothedTarget * logProbs).sum(/*dim=*/1);
                    return applyReduction(loss);
                }
                else {
                    // Hard labels: convert to one-hot then smooth
                    auto oneHot = torch::one_hot(target, _numClasses)
                        .to(prediction.dtype());

                    auto smoothedTarget = oneHot * (1.0f - _labelSmoothing) +
                        _labelSmoothing / static_cast<float>(_numClasses);

                    auto logProbs = torch::log_softmax(prediction, /*dim=*/1);
                    auto loss = -(smoothedTarget * logProbs).sum(/*dim=*/1);
                    return applyReduction(loss);
                }
            }

            torch::Tensor ClassificationLoss::applyReduction(const torch::Tensor& loss) {
                if (_reduction == "mean") {
                    return loss.mean();
                }
                else if (_reduction == "sum") {
                    return loss.sum();
                }
                else {
                    return loss;
                }
            }

        } // namespace Loss
    } // namespace Model
} // namespace WheelDL
