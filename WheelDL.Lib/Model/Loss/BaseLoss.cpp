#include "BaseLoss.h"
#include <stdexcept>
#include <string_view>

namespace WheelDL {
    namespace Model {
        namespace Loss {

            // Helper function to convert reduction string to torch reduction enum
            // Uses std::string_view for better performance (C++17)
            namespace {
                inline torch::Reduction::Reduction getReductionType(std::string_view reduction) {
                    if (reduction == "none") return torch::kNone;
                    if (reduction == "mean") return torch::kMean;
                    if (reduction == "sum") return torch::kSum;
                    // Fixed: Optimize string concatenation to avoid multiple allocations
                    throw std::invalid_argument(
                        "Invalid reduction: " + std::string(reduction) +
                        ". Must be 'none', 'mean', or 'sum'"
                    );
                }
            }

            // ============================================================================
            // BCEWithLogitsLoss Implementation
            // ============================================================================
            BCEWithLogitsLoss::BCEWithLogitsLoss(const std::string& reduction)
                : _criterion(torch::nn::BCEWithLogitsLossOptions())
            {
                _criterion->options.reduction(getReductionType(reduction));
            }

            std::unordered_map<std::string, torch::Tensor> BCEWithLogitsLoss::compute(
                const torch::Tensor& prediction,
                const Data::Dataset::DataExample& target) {
                // For BCEWithLogitsLoss, use classes field from DataExample
                auto loss = _criterion->forward(prediction, target.classes);
                return { {"total", loss} };
            }

            std::unordered_map<std::string, torch::Tensor> BCEWithLogitsLoss::compute(
                const torch::Tensor& prediction,
                const torch::Tensor& target) {
                auto loss = _criterion->forward(prediction, target);
                return { {"total", loss} };
            }

            // ============================================================================
            // MSELoss Implementation
            // ============================================================================
            MSELoss::MSELoss(const std::string& reduction)
                : _criterion(torch::nn::MSELossOptions())
            {
                _criterion->options.reduction(getReductionType(reduction));
            }

            std::unordered_map<std::string, torch::Tensor> MSELoss::compute(
                const torch::Tensor& prediction,
                const Data::Dataset::DataExample& target) {
                auto loss = _criterion->forward(prediction, target.targets);
                return { {"total", loss} };
            }

            std::unordered_map<std::string, torch::Tensor> MSELoss::compute(
                const torch::Tensor& prediction,
                const torch::Tensor& target) {
                auto loss = _criterion->forward(prediction, target);
                return { {"total", loss} };
            }

            // ============================================================================
            // MAELoss Implementation
            // ============================================================================
            MAELoss::MAELoss(const std::string& reduction)
                : _criterion(torch::nn::L1LossOptions())
            {
                _criterion->options.reduction(getReductionType(reduction));
            }

            std::unordered_map<std::string, torch::Tensor> MAELoss::compute(
                const torch::Tensor& prediction,
                const Data::Dataset::DataExample& target) {
                auto loss = _criterion->forward(prediction, target.targets);
                return { {"total", loss} };
            }

            std::unordered_map<std::string, torch::Tensor> MAELoss::compute(
                const torch::Tensor& prediction,
                const torch::Tensor& target) {
                auto loss = _criterion->forward(prediction, target);
                return { {"total", loss} };
            }

            // ============================================================================
            // SmoothL1Loss Implementation
            // ============================================================================
            SmoothL1Loss::SmoothL1Loss(float beta, const std::string& reduction)
                : _criterion(torch::nn::SmoothL1LossOptions().beta(beta))
            {
                _criterion->options.reduction(getReductionType(reduction));
            }

            std::unordered_map<std::string, torch::Tensor> SmoothL1Loss::compute(
                const torch::Tensor& prediction,
                const Data::Dataset::DataExample& target) {
                auto loss = _criterion->forward(prediction, target.targets);
                return { {"total", loss} };
            }

            std::unordered_map<std::string, torch::Tensor> SmoothL1Loss::compute(
                const torch::Tensor& prediction,
                const torch::Tensor& target) {
                auto loss = _criterion->forward(prediction, target);
                return { {"total", loss} };
            }

        } // namespace Loss
    } // namespace Model
} // namespace WheelDL
