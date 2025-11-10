#include "pch.h"
#include "Activation.h"

namespace WheelDL {
	namespace Model {
		namespace Modules {

			AGLUImpl::AGLUImpl() {
				// Initialize learnable parameters with uniform distribution
				auto lambdaTensor = torch::empty(1);
				torch::nn::init::uniform_(lambdaTensor);
				_lambda = register_parameter("lambda", lambdaTensor);

				auto kappaTensor = torch::empty(1);
				torch::nn::init::uniform_(kappaTensor);
				_kappa = register_parameter("kappa", kappaTensor);

				// Initialize Softplus with negative beta
				_act = register_module("act", torch::nn::Softplus(torch::nn::SoftplusOptions().beta(-1.0)));
			}

			torch::Tensor AGLUImpl::forward(const torch::Tensor& x) {
				// Clamp lambda to avoid division by zero
				auto lam = torch::clamp(_lambda, /*min=*/0.0001);

				// Apply AGLU transformation:
				// exp((1 / lam) * softplus((kappa * x) - log(lam)))
				auto kappaX = _kappa * x;
				auto logLam = torch::log(lam);
				auto softplusResult = _act->forward(kappaX - logLam);
				auto result = torch::exp((1.0 / lam) * softplusResult);

				return result;
			}

			HardswishImpl::HardswishImpl() {
				// No parameters needed
			}

			torch::Tensor HardswishImpl::forward(const torch::Tensor& x) {
				// Hardswish: x * relu6(x + 3) / 6
				// relu6(x) = min(max(x, 0), 6)
				return x * torch::clamp(x + 3.0, 0.0, 6.0) / 6.0;
			}

		} // namespace Modules
	} // namespace Model
} // namespace WheelDL
