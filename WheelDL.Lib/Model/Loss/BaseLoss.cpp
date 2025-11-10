#include "pch.h"
#include "BaseLoss.h"
#include <stdexcept>
#include <string_view>

namespace WheelDL {
    namespace Model {
        namespace Loss {

            // Helper function to convert reduction string to torch reduction enum
            // Template function to handle proper type deduction for reduction options
            namespace {
                template<typename T>
                inline T getReductionOptions(const std::string& reduction) {
                    T options;
                    if (reduction == "none") {
                        options.reduction(torch::kNone);
                    } else if (reduction == "mean") {
                        options.reduction(torch::kMean);
                    } else if (reduction == "sum") {
                        options.reduction(torch::kSum);
                    } else {
                        throw std::invalid_argument(
                            "Invalid reduction: " + reduction +
                            ". Must be 'none', 'mean', or 'sum'"
                        );
                    }
                    return options;
                }
            }

            // ============================================================================
            // BCEWithLogitsLoss Implementation
            // ============================================================================
            BCEWithLogitsLoss::BCEWithLogitsLoss(const std::string& reduction)
                : _criterion(getReductionOptions<torch::nn::BCEWithLogitsLossOptions>(reduction))
            {
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
                : _criterion(getReductionOptions<torch::nn::MSELossOptions>(reduction))
            {
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
                : _criterion(getReductionOptions<torch::nn::L1LossOptions>(reduction))
            {
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
                : _criterion([&]() {
                    auto options = torch::nn::SmoothL1LossOptions().beta(beta);
                    if (reduction == "none") {
                        options.reduction(torch::kNone);
                    } else if (reduction == "mean") {
                        options.reduction(torch::kMean);
                    } else {
                        options.reduction(torch::kSum);
                    }
                    return options;
                }())
            {
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
