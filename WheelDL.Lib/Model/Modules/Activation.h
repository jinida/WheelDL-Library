#pragma once

#include <torch/torch.h>

namespace WheelDL {
namespace Model {
namespace Modules {

/**
 * @brief Adaptive Gated Linear Unit (AGLU) activation function.
 *
 * Implements a parameterized activation function with learnable parameters lambda and kappa.
 * Based on the AGLU approach from https://github.com/kostas1515/AGLU
 *
 * The forward computation is:
 * output = exp((1 / lambda) * softplus((kappa * x) - log(lambda)))
 *
 * where softplus has negative beta (-1.0)
 */
class AGLUImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new AGLU activation module
     */
    AGLUImpl();

    /**
     * @brief Forward pass of the AGLU activation function
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor with AGLU activation applied
     */
    torch::Tensor forward(const torch::Tensor& x);

private:
    /// Learnable lambda parameter
    torch::Tensor _lambda;

    /// Learnable kappa parameter
    torch::Tensor _kappa;

    /// Softplus activation with beta = -1.0
    torch::nn::Softplus _act;
};

TORCH_MODULE(AGLU);

/**
 * @brief Hardswish activation function
 *
 * Implements: x * relu6(x + 3) / 6
 * where relu6(x) = min(max(x, 0), 6)
 *
 * Reference: Searching for MobileNetV3 (https://arxiv.org/abs/1905.02244)
 */
class HardswishImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new Hardswish activation module
     */
    HardswishImpl();

    /**
     * @brief Forward pass of the Hardswish activation function
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor with Hardswish activation applied
     */
    torch::Tensor forward(const torch::Tensor& x);
};

TORCH_MODULE(Hardswish);

} // namespace Modules
} // namespace Model
} // namespace WheelDL
