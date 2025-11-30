#include "pch.h"
#include "ClassificationLoss.h"
#include <stdexcept>
#include <algorithm>
#include <array>
#include <string_view>

namespace WheelDL {
    namespace Model {
        namespace Loss {

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
                , _reduction(parseReduction(reduction))
            {
                if (numClasses <= 0) throw std::invalid_argument("numClasses must be positive");
                if (_focalGamma < 0.f || _focalGamma > 5.f)
                    throw std::invalid_argument("Focal gamma must be in range [0, 5]");
            }

            std::unordered_map<std::string, torch::Tensor>
                ClassificationLoss::compute(const torch::Tensor& prediction,
                    const Data::Dataset::DataExample& target)
            {
                return compute(prediction, target.classes);
            }

            std::unordered_map<std::string, torch::Tensor>
                ClassificationLoss::compute(const torch::Tensor& prediction,
                    const torch::Tensor& target)
            {
                if (prediction.dim() != 2)
                    throw std::invalid_argument("Prediction must be [N, C]");
                if (!(target.dim() == 1 || target.dim() == 2))
                    throw std::invalid_argument("Target must be [N] or [N, C]");
                if (prediction.size(0) != target.size(0))
                    throw std::invalid_argument("Batch sizes must match");

                auto logProb = torch::log_softmax(prediction, /*dim=*/1);

                torch::Tensor loss;
                switch (_lossType) {
                case LossType::CROSS_ENTROPY:
                    loss = computeCrossEntropy(logProb, target);
                    break;
                case LossType::FOCAL:
					loss = computeFocalLoss(logProb, target);
                    break;
                case LossType::LABEL_SMOOTHING:
                    loss = computeLabelSmoothingLoss(logProb, target);
                    break;
                default:
                    throw std::runtime_error("Unknown loss type");
                }

                validateLoss(loss, name());
                return { {"total", loss} };
            }

            torch::Tensor ClassificationLoss::getSoftTarget(const torch::Tensor& target) const
            {
                if (target.dim() == 2) return target;
                return torch::one_hot(target, _numClasses).to(target.dtype());
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
                const torch::Tensor& logProb,
                const torch::Tensor& target)
            {
                if (target.dim() == 1) {
                    auto nll = -logProb.gather(1, target.unsqueeze(1)).squeeze(1);
                    return applyReduction(nll);
                }
                else {
                    auto loss = -(target * logProb).sum(1);
                    return applyReduction(loss);
                }
            }

            torch::Tensor ClassificationLoss::computeLabelSmoothingLoss(
                const torch::Tensor& logProb,
                const torch::Tensor& target)
            {
                if (target.dim() == 1) {
                    auto nll = -logProb.gather(1, target.unsqueeze(1)).squeeze(1);
                    auto uniform = -logProb.mean(1);
                    auto loss = (1.0f - _labelSmoothing) * nll + _labelSmoothing * uniform;
                    return applyReduction(loss);
                }
                else {
                    auto nll_soft = -(target * logProb).sum(1);
                    auto uniform = -logProb.mean(1);
                    auto loss = (1.0f - _labelSmoothing) * nll_soft + _labelSmoothing * uniform;
                    return applyReduction(loss);
                }
            }

            torch::Tensor ClassificationLoss::computeFocalLoss(
                const torch::Tensor& logProb,
                const torch::Tensor& target)
            {
                constexpr double PROB_EPS = 1e-7;
                auto probs = logProb.exp();

                torch::Tensor pt;
                if (target.dim() == 1) {
                    pt = probs.gather(1, target.unsqueeze(1)).squeeze(1);
                }
                else {
                    pt = (probs * target).sum(1);
                }

                auto clamped = pt.clamp(PROB_EPS, 1.0 - PROB_EPS);

                torch::Tensor focusing;
                if (_focalGamma == 0.0f) {
                    focusing = torch::ones_like(clamped);
                }
                else if (_focalGamma == 1.0f) {
                    focusing = (1.0 - clamped);
                }
                else if (_focalGamma == 2.0f) {
                    auto t = (1.0 - clamped);
                    focusing = t * t;
                }
                else {
                    focusing = torch::pow(1.0 - clamped, _focalGamma);
                }

                auto log_p = torch::log(clamped);
                auto loss = -_focalAlpha * focusing * log_p;
                return applyReduction(loss);
            }

            torch::Tensor ClassificationLoss::applyReduction(const torch::Tensor& loss)
            {
                switch (_reduction) {
                case Reduction::Mean: return loss.mean();
                case Reduction::Sum:  return loss.sum();
                default: return loss;
                }
            }

        } // namespace Loss
    } // namespace Model
} // namespace WheelDL
