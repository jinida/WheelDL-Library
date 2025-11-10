#include "pch.h"
#include "AnomalyLoss.h"
#include "../Constants.h"
#include <algorithm>
#include <stdexcept>

namespace WheelDL {
    namespace Model {
        namespace Loss {

            AnomalyLoss::AnomalyLoss(LossType lossType, float ssimWeight)
                : _lossType(lossType)
                , _ssimWeight(std::clamp(ssimWeight, 0.0f, 1.0f))
                , _mseLoss(torch::nn::MSELoss()) {
            }

            std::unordered_map<std::string, torch::Tensor> AnomalyLoss::compute(
                const torch::Tensor& prediction,
                const Data::Dataset::DataExample& target) {
                // Delegate to tensor overload
                return compute(prediction, target.data);
            }

            std::unordered_map<std::string, torch::Tensor> AnomalyLoss::compute(
                const torch::Tensor& prediction,
                const torch::Tensor& target) {
                // Validate inputs
                if (prediction.sizes() != target.sizes()) {
                    throw std::invalid_argument("Prediction and target must have same shape");
                }

                if (prediction.dim() != 4) {
                    throw std::invalid_argument("Inputs must be 4D tensors [N, C, H, W]");
                }

                torch::Tensor loss;

                switch (_lossType) {
                case LossType::MSE:
                    // Simple MSE reconstruction loss
                    loss = _mseLoss->forward(prediction, target);
                    break;

                case LossType::SSIM:
                    // SSIM-based loss for perceptual quality
                    loss = computeSSIMLoss(prediction, target);
                    break;

                case LossType::COMBINED:
                    // Combine MSE and SSIM
                {
                    auto mseLoss = _mseLoss->forward(prediction, target);
                    auto ssimLoss = computeSSIMLoss(prediction, target);
                    loss = mseLoss * (1.0f - _ssimWeight) + ssimLoss * _ssimWeight;
                    break;
                }

                default:
                    throw std::runtime_error("Unknown loss type");
                }

                // Validate loss for NaN/Inf values
                validateLoss(loss, name());
                return {{"total", loss}};
            }

            std::string AnomalyLoss::name() const {
                switch (_lossType) {
                case LossType::MSE:
                    return "AnomalyLoss(MSE)";
                case LossType::SSIM:
                    return "AnomalyLoss(SSIM)";
                case LossType::COMBINED:
                    return "AnomalyLoss(MSE+SSIM)";
                default:
                    return "AnomalyLoss(Unknown)";
                }
            }

            torch::Tensor AnomalyLoss::computeSSIM(const torch::Tensor& img1,
                const torch::Tensor& img2) {
                // Fixed: Add input validation to prevent runtime errors
                if (img1.sizes() != img2.sizes()) {
                    throw std::invalid_argument(
                        "SSIM: Images must have same shape, got img1=" +
                        std::to_string(img1.dim()) + "D and img2=" +
                        std::to_string(img2.dim()) + "D"
                    );
                }
                if (img1.dim() != 4) {
                    throw std::invalid_argument(
                        "SSIM: Images must be 4D tensors [N, C, H, W], got " +
                        std::to_string(img1.dim()) + "D"
                    );
                }

                // Fixed: Use unified constants from Constants.h
                // SSIM (Structural Similarity Index Measure) constants for numerical stability
                // Reference: Wang et al. "Image Quality Assessment: From Error Visibility to
                // Structural Similarity" IEEE TIP 2004
                using namespace Constants;
                constexpr float C1 = SSIM_C1;  // = 0.0001
                constexpr float C2 = SSIM_C2;  // = 0.0009

                // Compute mean (mu)
                auto mu1 = img1.mean({ 2, 3 }, /*keepdim=*/true);
                auto mu2 = img2.mean({ 2, 3 }, /*keepdim=*/true);

                // Compute variance and covariance
                auto mu1_sq = mu1.pow(2);
                auto mu2_sq = mu2.pow(2);
                auto mu1_mu2 = mu1 * mu2;

                // Clamp to prevent negative variance due to floating point errors
                auto sigma1_sq = (img1.pow(2).mean({ 2, 3 }, /*keepdim=*/true) - mu1_sq).clamp_min(0.0f);
                auto sigma2_sq = (img2.pow(2).mean({ 2, 3 }, /*keepdim=*/true) - mu2_sq).clamp_min(0.0f);
                auto sigma12 = (img1 * img2).mean({ 2, 3 }, /*keepdim=*/true) - mu1_mu2;

                // SSIM formula:
                // SSIM = (2*mu1*mu2 + C1) * (2*sigma12 + C2) / ((mu1^2 + mu2^2 + C1) * (sigma1^2 + sigma2^2 + C2))
                auto numerator = (2.0f * mu1_mu2 + C1) * (2.0f * sigma12 + C2);
                auto denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2);

                // Add epsilon to prevent division by zero for numerical stability
                auto ssim = numerator / torch::clamp_min(denominator, 1e-8f);

                // Return mean SSIM across batch and channels
                return ssim.mean();
            }

        } // namespace Loss
    } // namespace Model
} // namespace WheelDL
