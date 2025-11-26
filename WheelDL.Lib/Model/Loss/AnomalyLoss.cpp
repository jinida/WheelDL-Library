#include "pch.h"
#include "AnomalyLoss.h"
#include "../Constants.h"
#include <algorithm>
#include <stdexcept>

namespace WheelDL {
    namespace Model {
        namespace Loss {

            AnomalyLoss::AnomalyLoss(LossType lossType)
                : _lossType(lossType) {}

            std::unordered_map<std::string, torch::Tensor> AnomalyLoss::compute(
                const torch::Tensor& prediction,
                const Data::Dataset::DataExample& target) 
            {
                return compute(prediction, target.data);
            }

            std::unordered_map<std::string, torch::Tensor> AnomalyLoss::compute(
                const torch::Tensor& prediction,
                const torch::Tensor& target)
            {
                // This method is for simple reconstruction losses (not used for EfficientAD/PatchCore/SimpleNet)
                throw std::runtime_error("Use compute(vector, DataExample) for " + name());
            }

            std::unordered_map<std::string, torch::Tensor> AnomalyLoss::compute(
                const std::vector<torch::Tensor>& predictions,
                const Data::Dataset::DataExample& target)
            {
                switch (_lossType) {
                case LossType::EfficientAD:
                    return computeEfficientAD(predictions);

                case LossType::PatchCore:
                    return computePatchCore(predictions);

                case LossType::SimpleNet:
                    return computeSimpleNet(predictions);

                default:
                    throw std::runtime_error("Unknown loss type");
                }
			}

            std::unordered_map<std::string, torch::Tensor> AnomalyLoss::computeEfficientAD(
                const std::vector<torch::Tensor>& predictions)
            {
                // Validate input size
                if (predictions.size() < 5) {
                    throw std::invalid_argument(
						"EfficientAD loss requires exactly more than 5 prediction tensors: "
                        "[teacher_out, student_out, ae_teacher_out, ae_student_out, ae_out], got " +
                        std::to_string(predictions.size())
                    );
                }

                auto teacherOut = predictions[0];      // [N, C, H, W]
                auto studentOut = predictions[1];      // [N, C, H, W]
                auto aeTeacherOut = predictions[2];    // [N, C, H, W]
                auto aeStudentOut = predictions[3];    // [N, C, H, W]
                auto aeOut = predictions[4];           // [N, C, H, W]

                // 1. Hard Loss: Focus on difficult samples (top 0.1%)
                // Distance between teacher and student outputs
                auto distanceOut = torch::pow(teacherOut - studentOut, 2);

                // Get 99.9th percentile threshold (top 0.1% hardest samples)
                auto dHard = torch::quantile(distanceOut.flatten(), 0.999f);

                // Create mask for hard samples
                auto hardMask = distanceOut >= dHard;

                // Compute hard loss only on difficult samples
                auto lossHard = hardMask.any().item<bool>() ?
                    torch::mean(distanceOut.masked_select(hardMask)) :
                    torch::zeros({}, distanceOut.options());

                // 2. AE Loss: AutoEncoder reconstruction of teacher features
                // This ensures the autoencoder can reconstruct normal (teacher) features
                auto distanceAe = torch::pow(aeTeacherOut - aeOut, 2);
                auto lossAe = torch::mean(distanceAe);

                // 3. STAE Loss: Student-Teacher-AutoEncoder alignment
                // This ensures student features align with autoencoder output
                auto distanceStae = torch::pow(aeStudentOut - aeOut, 2);
                auto lossStae = torch::mean(distanceStae);

                // Total loss (all components equally weighted)
                auto totalLoss = lossHard + lossAe + lossStae;

                validateLoss(totalLoss, name());

                return {
                    {"hard", lossHard.detach()},
                    {"ae", lossAe.detach()},
                    {"stae", lossStae.detach()},
                    {"total", totalLoss}
                };
            }

            std::unordered_map<std::string, torch::Tensor> AnomalyLoss::computePatchCore(
                const std::vector<torch::Tensor>& predictions)
            {
				return { {"total", torch::zeros({}, torch::kFloat32)} };
            }

            std::unordered_map<std::string, torch::Tensor> AnomalyLoss::computeSimpleNet(
                const std::vector<torch::Tensor>& predictions)
            {
                // TODO: Implement SimpleNet loss (feature matching)
                // For now, return a mockup loss
                auto mockLoss = torch::zeros({}, torch::kFloat32);

                validateLoss(mockLoss, name());
                return { {"total", mockLoss} };
            }

            std::string AnomalyLoss::name() const 
            {
                switch (_lossType) 
                {
                    case LossType::SimpleNet:
						return "AnomalyLoss::SimpleNet";
                    case LossType::EfficientAD:
                        return "AnomalyLoss::EfficientAD";
                    case LossType::PatchCore:
                        return "AnomalyLoss::PatchCore";
                    default:
						return "AnomalyLoss::Unknown";
                }
            }
        } // namespace Loss
    } // namespace Model
} // namespace WheelDL
