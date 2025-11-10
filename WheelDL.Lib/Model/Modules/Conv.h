#pragma once

#include <torch/torch.h>
#include "Interfaces.h"

namespace WheelDL {
    namespace Model {
        namespace Modules {

            /**
             * @brief Calculate padding for 'same' shape outputs
             *
             * @param k Kernel size
             * @param p Padding (optional)
             * @param d Dilation
             * @return int64_t Calculated padding
             */
            int64_t autoPad(int64_t k, std::optional<int64_t> p = std::nullopt, int64_t d = 1);

            /**
             * @brief Standard convolution with batch normalization and activation
             *
             * This module implements: Conv2d -> BatchNorm2d -> Activation
             */

            class ConvImpl : public IBlockImpl {
            public:
                /**
                 * @brief Construct a new Conv module
                 *
                 * @param c1 Number of input channels
                 * @param c2 Number of output channels
                 * @param k Kernel size (default: 1)
                 * @param s Stride (default: 1)
                 * @param p Padding (optional, auto-calculated if not provided)
                 * @param g Groups (default: 1)
                 * @param d Dilation (default: 1)
                 * @param act Activation type name (default: "SiLU"), or empty string for no activation
                 */
                ConvImpl(int64_t c1, int64_t c2, int64_t k = 1, int64_t s = 1,
                    std::optional<int64_t> p = std::nullopt, int64_t g = 1,
                    int64_t d = 1, const std::string& act = "SiLU");

                /**
                 * @brief Forward pass: Conv -> BN -> Act
                 *
                 * @param x Input tensor
                 * @return torch::Tensor Output tensor
                 */
                torch::Tensor forward(torch::Tensor x);


            protected:
                torch::nn::Conv2d _conv = nullptr;
                torch::nn::BatchNorm2d _bn = nullptr;
                torch::nn::AnyModule _act;
            };

            TORCH_MODULE(Conv);

            /**
             * @brief Depthwise Convolution
             *
             * Implements depthwise separable convolution by setting groups = gcd(c1, c2)
             */
            class DWConvImpl : public ConvImpl {
            public:
                /**
                 * @brief Construct a new DWConv module
                 *
                 * @param c1 Number of input channels
                 * @param c2 Number of output channels
                 * @param k Kernel size (default: 1)
                 * @param s Stride (default: 1)
                 * @param d Dilation (default: 1)
                 * @param act Activation type name (default: "SiLU")
                 */
                DWConvImpl(int64_t c1, int64_t c2, int64_t k = 1, int64_t s = 1,
                    int64_t d = 1, const std::string& act = "SiLU");
            };

            TORCH_MODULE(DWConv);

            /**
             * @brief Simplified RepConv module with Conv fusing
             *
             * Combines 3x3 and 1x1 convolutions
             */
            class Conv2Impl : public ConvImpl {
            public:
                /**
                 * @brief Construct a new Conv2 module
                 *
                 * @param c1 Number of input channels
                 * @param c2 Number of output channels
                 * @param k Kernel size (default: 3)
                 * @param s Stride (default: 1)
                 * @param p Padding (optional)
                 * @param g Groups (default: 1)
                 * @param d Dilation (default: 1)
                 * @param act Activation type name (default: "SiLU")
                 */
                Conv2Impl(int64_t c1, int64_t c2, int64_t k = 3, int64_t s = 1,
                    std::optional<int64_t> p = std::nullopt, int64_t g = 1,
                    int64_t d = 1, const std::string& act = "SiLU");

                /**
                 * @brief Forward pass with parallel convolutions
                 *
                 * @param x Input tensor
                 * @return torch::Tensor Output tensor
                 */
                torch::Tensor forward(torch::Tensor x);

            private:
                torch::nn::Conv2d _cv2 = nullptr;
            };

            TORCH_MODULE(Conv2);

            /**
             * @brief Light Convolution with 1x1 and depthwise convolutions
             */
            class LightConvImpl : public IBlockImpl {
            public:
                /**
                 * @brief Construct a new LightConv module
                 *
                 * @param c1 Number of input channels
                 * @param c2 Number of output channels
                 * @param k Kernel size for depthwise conv (default: 1)
                 */
                LightConvImpl(int64_t c1, int64_t c2, int64_t k = 1);

                /**
                 * @brief Forward pass: 1x1 Conv -> DWConv
                 *
                 * @param x Input tensor
                 * @return torch::Tensor Output tensor
                 */
                torch::Tensor forward(torch::Tensor x);

            private:
                Conv _conv1 = nullptr;
                DWConv _conv2 = nullptr;
            };

            TORCH_MODULE(LightConv);

            /**
             * @brief Ghost Convolution
             *
             * Generates more features with fewer parameters using cheap operations
             *
             * Reference: https://github.com/huawei-noah/Efficient-AI-Backbones
             */
            class GhostConvImpl : public IBlockImpl {
            public:
                /**
                 * @brief Construct a new GhostConv module
                 *
                 * @param c1 Number of input channels
                 * @param c2 Number of output channels
                 * @param k Kernel size (default: 1)
                 * @param s Stride (default: 1)
                 * @param g Groups (default: 1)
                 * @param act Activation type name (default: "SiLU")
                 */
                GhostConvImpl(int64_t c1, int64_t c2, int64_t k = 1, int64_t s = 1,
                    int64_t g = 1, const std::string& act = "SiLU");

                /**
                 * @brief Forward pass: primary conv + cheap operations
                 *
                 * @param x Input tensor
                 * @return torch::Tensor Concatenated output
                 */
                torch::Tensor forward(torch::Tensor x);

            private:
                Conv _cv1 = nullptr;
                Conv _cv2 = nullptr;
            };

            TORCH_MODULE(GhostConv);

            /**
             * @brief RepConv module with training and deploy modes
             *
             * Can fuse multiple branches during inference for efficiency
             *
             * Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
             */
            class RepConvImpl : public IBlockImpl {
            public:
                /**
                 * @brief Construct a new RepConv module
                 *
                 * @param c1 Number of input channels
                 * @param c2 Number of output channels
                 * @param k Kernel size (must be 3)
                 * @param s Stride (default: 1)
                 * @param p Padding (must be 1)
                 * @param g Groups (default: 1)
                 * @param d Dilation (default: 1)
                 * @param act Activation type name (default: "SiLU")
                 * @param bn Use batch norm for identity branch (default: false)
                 * @param deploy Deploy mode (default: false)
                 */
                RepConvImpl(int64_t c1, int64_t c2, int64_t k = 3, int64_t s = 1,
                    int64_t p = 1, int64_t g = 1, int64_t d = 1,
                    const std::string& act = "SiLU", bool bn = false, bool deploy = false);

                /**
                 * @brief Forward pass (training mode)
                 *
                 * @param x Input tensor
                 * @return torch::Tensor Output tensor
                 */
                torch::Tensor forward(torch::Tensor x);

            private:
                int64_t _g;
                int64_t _c1;
                int64_t _c2;
                torch::nn::AnyModule _act;
                torch::nn::BatchNorm2d _bn = nullptr;
                Conv _conv1 = nullptr;
                Conv _conv2 = nullptr;
            };

            TORCH_MODULE(RepConv);

            /**
             * @brief Depthwise Transpose Convolution module
             *
             * Implements depthwise separable transpose convolution
             */
            class DWConvTranspose2dImpl : public torch::nn::ConvTranspose2dImpl {
            public:
                /**
                 * @brief Construct a new DWConvTranspose2d module
                 *
                 * @param c1 Number of input channels
                 * @param c2 Number of output channels
                 * @param k Kernel size (default: 1)
                 * @param s Stride (default: 1)
                 * @param p1 Padding (default: 0)
                 * @param p2 Output padding (default: 0)
                 */
                DWConvTranspose2dImpl(int64_t c1, int64_t c2, int64_t k = 1, int64_t s = 1,
                    int64_t p1 = 0, int64_t p2 = 0);
            };

            TORCH_MODULE(DWConvTranspose2d);

            /**
             * @brief Transpose Convolution with optional batch normalization and activation
             */
            class ConvTransposeImpl : public IBlockImpl {
            public:
                /**
                 * @brief Construct a new ConvTranspose module
                 *
                 * @param c1 Number of input channels
                 * @param c2 Number of output channels
                 * @param k Kernel size (default: 2)
                 * @param s Stride (default: 2)
                 * @param p Padding (default: 0)
                 * @param bn Use batch normalization (default: true)
                 * @param act Activation type name (default: "SiLU")
                 */
                ConvTransposeImpl(int64_t c1, int64_t c2, int64_t k = 2, int64_t s = 2,
                    int64_t p = 0, bool bn = true, const std::string& act = "SiLU");

                /**
                 * @brief Forward pass: ConvTranspose2d -> BN -> Act
                 *
                 * @param x Input tensor
                 * @return torch::Tensor Output tensor
                 */
                torch::Tensor forward(torch::Tensor x);

            private:
                torch::nn::ConvTranspose2d _convTranspose = nullptr;
                torch::nn::AnyModule _bn;
                torch::nn::AnyModule _act;
            };

            TORCH_MODULE(ConvTranspose);

            /**
             * @brief Focus module for concentrating feature information
             *
             * Slices input tensor into 4 parts and concatenates them in the channel dimension
             */
            class FocusImpl : public IBlockImpl {
            public:
                /**
                 * @brief Construct a new Focus module
                 *
                 * @param c1 Number of input channels
                 * @param c2 Number of output channels
                 * @param k Kernel size (default: 1)
                 * @param s Stride (default: 1)
                 * @param p Padding (optional)
                 * @param g Groups (default: 1)
                 * @param act Activation type name (default: "SiLU")
                 */
                FocusImpl(int64_t c1, int64_t c2, int64_t k = 1, int64_t s = 1,
                    std::optional<int64_t> p = std::nullopt, int64_t g = 1, const std::string& act = "SiLU");

                /**
                 * @brief Apply Focus operation and convolution
                 *
                 * Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2)
                 *
                 * @param x Input tensor
                 * @return torch::Tensor Output tensor
                 */
                torch::Tensor forward(torch::Tensor x);

            private:
                Conv _conv = nullptr;
            };

            TORCH_MODULE(Focus);

            /**
             * @brief Concatenate a list of tensors along specified dimension
             */
            class ConcatImpl : public torch::nn::Module {
            public:
                /**
                 * @brief Construct a new Concat module
                 *
                 * @param dimension Dimension along which to concatenate tensors (default: 1)
                 */
                ConcatImpl(int64_t dimension = 1);

                /**
                 * @brief Concatenate input tensors along specified dimension
                 *
                 * @param x Vector of input tensors
                 * @return torch::Tensor Concatenated tensor
                 */
                torch::Tensor forward(std::vector<torch::Tensor> x);

            private:
                int64_t _d;
            };

            TORCH_MODULE(Concat);

            /**
             * @brief Returns a particular index of the input
             */
            class IndexImpl : public torch::nn::Module {
            public:
                /**
                 * @brief Construct a new Index module
                 *
                 * @param index Index to select from input (default: 0)
                 */
                IndexImpl(int64_t index = 0);

                /**
                 * @brief Select and return a particular index from input
                 *
                 * @param x Vector of input tensors
                 * @return torch::Tensor Selected tensor
                 */
                torch::Tensor forward(std::vector<torch::Tensor> x);

            private:
                int64_t _index;
            };

            TORCH_MODULE(Index);
        } // namespace Modules
    } // namespace Model
} // namespace WheelDL
