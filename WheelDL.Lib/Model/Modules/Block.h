#pragma once

#include <torch/torch.h>
#include "Conv.h"
#include "Transformer.h"
#include "Attention.h"

namespace WheelDL {
namespace Model {
namespace Modules {

// Constants for Block modules
namespace BlockConstants {
    constexpr double DEFAULT_EXPANSION_RATIO = 0.5;
    constexpr double PSA_EXPANSION_RATIO = 0.5;
    constexpr int64_t PSA_NUM_HEADS = 4;
    constexpr int64_t DEFAULT_HIDDEN_CHANNELS = 256;
    constexpr int64_t SECOND_HIDDEN_CHANNELS = 128;
}

/**
 * @brief Distribution Focal Loss (DFL) integral module
 *
 * Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
 */
class DFLImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new DFL module
     *
     * @param c1 Number of input channels (default: 16)
     */
    explicit DFLImpl(int64_t c1 = 16);

    /**
     * @brief Apply the DFL module to input tensor
     *
     * @param x Input tensor
     * @return torch::Tensor Transformed output
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d _conv;
    int64_t _c1;
};

TORCH_MODULE(DFL);

/**
 * @brief Proto module for segmentation models
 */
class ProtoImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new Proto module
     *
     * @param c1 Input channels
     * @param cMid Intermediate channels (default: 256)
     * @param c2 Output channels (number of protos, default: 32)
     */
    ProtoImpl(int64_t c1, int64_t cMid = 256, int64_t c2 = 32);

    /**
     * @brief Forward pass through Proto module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    Conv _cv1;
    torch::nn::ConvTranspose2d _upsample;
    Conv _cv2;
    Conv _cv3;
};

TORCH_MODULE(Proto);

/**
 * @brief Spatial Pyramid Pooling (SPP) layer
 *
 * Reference: https://arxiv.org/abs/1406.4729
 */
class SPPImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new SPP module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param k Kernel sizes for max pooling (default: {5, 9, 13})
     */
    SPPImpl(int64_t c1, int64_t c2, std::vector<int64_t> k = {5, 9, 13});

    /**
     * @brief Forward pass through SPP module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    Conv _cv1;
    Conv _cv2;
    torch::nn::ModuleList _m;
};

TORCH_MODULE(SPP);

/**
 * @brief Spatial Pyramid Pooling - Fast (SPPF) layer
 *
 * Faster version of SPP, equivalent to SPP(k=(5, 9, 13))
 */
class SPPFImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new SPPF module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param k Kernel size (default: 5)
     */
    SPPFImpl(int64_t c1, int64_t c2, int64_t k = 5);

    /**
     * @brief Forward pass through SPPF module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    Conv _cv1;
    Conv _cv2;
    torch::nn::MaxPool2d _m;
};

TORCH_MODULE(SPPF);

/**
 * @brief Standard bottleneck block
 */
class BottleneckImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new Bottleneck module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param shortcut Whether to use shortcut connection (default: true)
     * @param g Groups for convolutions (default: 1)
     * @param k Kernel sizes for convolutions (default: {3, 3})
     * @param e Expansion ratio (default: 0.5)
     */
    BottleneckImpl(int64_t c1, int64_t c2, bool shortcut = true, int64_t g = 1,
                   std::vector<int64_t> k = {3, 3}, double e = 0.5);

    /**
     * @brief Forward pass through Bottleneck module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    Conv _cv1;
    Conv _cv2;
    bool _add;
};

TORCH_MODULE(Bottleneck);

/**
 * @brief CSP Bottleneck with 1 convolution
 */
class C1Impl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new C1 module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param n Number of convolutions (default: 1)
     */
    C1Impl(int64_t c1, int64_t c2, int64_t n = 1);

    /**
     * @brief Forward pass through C1 module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    Conv _cv1;
    torch::nn::Sequential _m;
};

TORCH_MODULE(C1);

/**
 * @brief CSP Bottleneck with 2 convolutions
 */
class C2Impl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new C2 module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param n Number of Bottleneck blocks (default: 1)
     * @param shortcut Whether to use shortcut connections (default: true)
     * @param g Groups for convolutions (default: 1)
     * @param e Expansion ratio (default: 0.5)
     */
    C2Impl(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = true,
           int64_t g = 1, double e = 0.5);

    /**
     * @brief Forward pass through C2 module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

protected:
    int64_t _c;
    Conv _cv1;
    Conv _cv2;
    torch::nn::Sequential _m;
};

TORCH_MODULE(C2);

/**
 * @brief Faster Implementation of CSP Bottleneck with 2 convolutions
 */
class C2fImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new C2f module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param n Number of Bottleneck blocks (default: 1)
     * @param shortcut Whether to use shortcut connections (default: false)
     * @param g Groups for convolutions (default: 1)
     * @param e Expansion ratio (default: 0.5)
     */
    C2fImpl(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = false,
            int64_t g = 1, double e = 0.5);

    /**
     * @brief Forward pass through C2f module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

protected:
    int64_t _c;
    Conv _cv1;
    Conv _cv2;
    torch::nn::ModuleList _m;
};

TORCH_MODULE(C2f);

/**
 * @brief CSP Bottleneck with 3 convolutions
 */
class C3Impl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new C3 module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param n Number of Bottleneck blocks (default: 1)
     * @param shortcut Whether to use shortcut connections (default: true)
     * @param g Groups for convolutions (default: 1)
     * @param e Expansion ratio (default: 0.5)
     */
    C3Impl(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = true,
           int64_t g = 1, double e = 0.5);

    /**
     * @brief Forward pass through C3 module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

protected:
    int64_t _c;
    Conv _cv1;
    Conv _cv2;
    Conv _cv3;
    torch::nn::Sequential _m;
};

TORCH_MODULE(C3);

/**
 * @brief C3 module with Ghost Bottleneck
 */
class GhostBottleneckImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new GhostBottleneck module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param k Kernel size (default: 3)
     * @param s Stride (default: 1)
     */
    GhostBottleneckImpl(int64_t c1, int64_t c2, int64_t k = 3, int64_t s = 1);

    /**
     * @brief Forward pass through GhostBottleneck module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential _conv;
    torch::nn::Sequential _shortcut;
};

TORCH_MODULE(GhostBottleneck);

/**
 * @brief PPHGNetV2 Stem block with 5 convolutions and one maxpool2d
 */
class HGStemImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new HGStem module
     *
     * @param c1 Input channels
     * @param cm Middle channels
     * @param c2 Output channels
     */
    HGStemImpl(int64_t c1, int64_t cm, int64_t c2);

    /**
     * @brief Forward pass through HGStem module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    Conv _stem1;
    Conv _stem2a;
    Conv _stem2b;
    Conv _stem3;
    Conv _stem4;
    torch::nn::MaxPool2d _pool;
};

TORCH_MODULE(HGStem);

/**
 * @brief PPHGNetV2 HG block with 2 convolutions and LightConv
 */
class HGBlockImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new HGBlock module
     *
     * @param c1 Input channels
     * @param cm Middle channels
     * @param c2 Output channels
     * @param k Kernel size (default: 3)
     * @param n Number of blocks (default: 6)
     * @param lightConv Whether to use LightConv (default: false)
     * @param shortcut Whether to use shortcut connection (default: false)
     */
    HGBlockImpl(int64_t c1, int64_t cm, int64_t c2, int64_t k = 3, int64_t n = 6,
                bool lightConv = false, bool shortcut = false);

    /**
     * @brief Forward pass through HGBlock module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::ModuleList _m;
    Conv _sc;
    Conv _ec;
    bool _add;
    bool _lightConv;
};

TORCH_MODULE(HGBlock);

/**
 * @brief RepVGG-style bottleneck using RepConv
 */
class RepBottleneckImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new RepBottleneck module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param shortcut Whether to use shortcut connection (default: true)
     * @param g Groups for convolutions (default: 1)
     * @param k Kernel sizes (default: {3, 3})
     * @param e Expansion ratio (default: 0.5)
     */
    RepBottleneckImpl(int64_t c1, int64_t c2, bool shortcut = true, int64_t g = 1,
                      std::vector<int64_t> k = {3, 3}, double e = 0.5);

    /**
     * @brief Forward pass through RepBottleneck module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    RepConv _cv1;
    Conv _cv2;
    bool _add;
};

TORCH_MODULE(RepBottleneck);

/**
 * @brief C3 module with cross-convolutions
 */
class C3xImpl : public C3Impl {
public:
    /**
     * @brief Construct a new C3x module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param n Number of Bottleneck blocks (default: 1)
     * @param shortcut Whether to use shortcut connections (default: true)
     * @param g Groups for convolutions (default: 1)
     * @param e Expansion ratio (default: 0.5)
     */
    C3xImpl(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = true,
            int64_t g = 1, double e = 0.5);
};

TORCH_MODULE(C3x);

/**
 * @brief RepC3 module with RepConv blocks
 */
class RepC3Impl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new RepC3 module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param n Number of RepConv blocks (default: 3)
     * @param e Expansion ratio (default: 1.0)
     */
    RepC3Impl(int64_t c1, int64_t c2, int64_t n = 3, double e = 1.0);

    /**
     * @brief Forward pass through RepC3 module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    int64_t _c;
    Conv _cv1;
    Conv _cv2;
    torch::nn::Sequential _m;
    torch::nn::AnyModule _cv3;
};

TORCH_MODULE(RepC3);

/**
 * @brief C3 module with GhostBottleneck blocks
 */
class C3GhostImpl : public C3Impl {
public:
    /**
     * @brief Construct a new C3Ghost module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param n Number of Ghost bottleneck blocks (default: 1)
     * @param shortcut Whether to use shortcut connections (default: true)
     * @param g Groups for convolutions (default: 1)
     * @param e Expansion ratio (default: 0.5)
     */
    C3GhostImpl(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = true,
                int64_t g = 1, double e = 0.5);
};

TORCH_MODULE(C3Ghost);

/**
 * @brief CSP Bottleneck module
 */
class BottleneckCSPImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new BottleneckCSP module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param n Number of Bottleneck blocks (default: 1)
     * @param shortcut Whether to use shortcut connections (default: true)
     * @param g Groups for convolutions (default: 1)
     * @param e Expansion ratio (default: 0.5)
     */
    BottleneckCSPImpl(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = true,
                      int64_t g = 1, double e = 0.5);

    /**
     * @brief Forward pass through BottleneckCSP module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    Conv _cv1;
    torch::nn::Conv2d _cv2;
    torch::nn::Conv2d _cv3;
    Conv _cv4;
    torch::nn::BatchNorm2d _bn;
    torch::nn::SiLU _act;
    torch::nn::Sequential _m;
};

TORCH_MODULE(BottleneckCSP);

/**
 * @brief RepCSP module
 */
class RepCSPImpl : public C3Impl {
public:
    /**
     * @brief Construct a new RepCSP module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param n Number of RepBottleneck blocks (default: 1)
     * @param shortcut Whether to use shortcut connections (default: true)
     * @param g Groups for convolutions (default: 1)
     * @param e Expansion ratio (default: 0.5)
     */
    RepCSPImpl(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = true,
               int64_t g = 1, double e = 0.5);
};

TORCH_MODULE(RepCSP);

/**
 * @brief ADown module for downsampling
 */
class ADownImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new ADown module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     */
    ADownImpl(int64_t c1, int64_t c2);

    /**
     * @brief Forward pass through ADown module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    int64_t _c;
    Conv _cv1;
    Conv _cv2;
};

TORCH_MODULE(ADown);

/**
 * @brief AConv module
 */
class AConvImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new AConv module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     */
    AConvImpl(int64_t c1, int64_t c2);

    /**
     * @brief Forward pass through AConv module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    Conv _cv1;
};

TORCH_MODULE(AConv);

/**
 * @brief SPPELAN module
 */
class SPPELANImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new SPPELAN module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param c3 Intermediate channels
     * @param k Kernel size for max pooling (default: 5)
     */
    SPPELANImpl(int64_t c1, int64_t c2, int64_t c3, int64_t k = 5);

    /**
     * @brief Forward pass through SPPELAN module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    int64_t _c;
    Conv _cv1;
    torch::nn::MaxPool2d _cv2;
    torch::nn::MaxPool2d _cv3;
    torch::nn::MaxPool2d _cv4;
    Conv _cv5;
};

TORCH_MODULE(SPPELAN);

/**
 * @brief RepNCSPELAN4 module
 */
class RepNCSPELAN4Impl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new RepNCSPELAN4 module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param c3 Intermediate channels
     * @param c4 Intermediate channels for RepCSP
     * @param n Number of RepCSP blocks (default: 1)
     */
    RepNCSPELAN4Impl(int64_t c1, int64_t c2, int64_t c3, int64_t c4, int64_t n = 1);

    /**
     * @brief Forward pass through RepNCSPELAN4 module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

protected:
    int64_t _c;
    Conv _cv1;
    torch::nn::Sequential _cv2;
    torch::nn::Sequential _cv3;
    Conv _cv4;
};

TORCH_MODULE(RepNCSPELAN4);

/**
 * @brief ELAN1 module
 */
class ELAN1Impl : public RepNCSPELAN4Impl {
public:
    /**
     * @brief Construct a new ELAN1 module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param c3 Intermediate channels
     * @param c4 Intermediate channels for convolutions
     */
    ELAN1Impl(int64_t c1, int64_t c2, int64_t c3, int64_t c4);
};

TORCH_MODULE(ELAN1);

/**
 * @brief CBLinear module
 */
class CBLinearImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new CBLinear module
     *
     * @param c1 Input channels
     * @param c2s List of output channel sizes
     * @param k Kernel size (default: 1)
     * @param s Stride (default: 1)
     * @param p Padding (optional)
     * @param g Groups (default: 1)
     */
    CBLinearImpl(int64_t c1, std::vector<int64_t> c2s, int64_t k = 1, int64_t s = 1,
                 std::optional<int64_t> p = std::nullopt, int64_t g = 1);

    /**
     * @brief Forward pass through CBLinear module
     *
     * @param x Input tensor
     * @return std::vector<torch::Tensor> List of output tensors
     */
    std::vector<torch::Tensor> forward(torch::Tensor x);

private:
    std::vector<int64_t> _c2s;
    torch::nn::Conv2d _conv;
};

TORCH_MODULE(CBLinear);

/**
 * @brief CBFuse module
 */
class CBFuseImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new CBFuse module
     *
     * @param idx Indices for feature selection
     */
    explicit CBFuseImpl(std::vector<int64_t> idx);

    /**
     * @brief Forward pass through CBFuse module
     *
     * @param xs List of input tensors
     * @return torch::Tensor Fused output tensor
     */
    torch::Tensor forward(std::vector<torch::Tensor> xs);

private:
    std::vector<int64_t> _idx;
};

TORCH_MODULE(CBFuse);

/**
 * @brief C3k module
 */
class C3kImpl : public C3Impl {
public:
    /**
     * @brief Construct a new C3k module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param n Number of Bottleneck blocks (default: 1)
     * @param shortcut Whether to use shortcut connections (default: true)
     * @param g Groups for convolutions (default: 1)
     * @param e Expansion ratio (default: 0.5)
     * @param k Kernel size (default: 3)
     */
    C3kImpl(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = true,
            int64_t g = 1, double e = 0.5, int64_t k = 3);
};

TORCH_MODULE(C3k);

/**
 * @brief RepVGGDW module
 */
class RepVGGDWImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new RepVGGDW module
     *
     * @param ed Input and output channels
     */
    explicit RepVGGDWImpl(int64_t ed);

    /**
     * @brief Forward pass through RepVGGDW module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    Conv _conv;
    Conv _conv1;
    int64_t _dim;
    torch::nn::SiLU _act;
};

TORCH_MODULE(RepVGGDW);

/**
 * @brief CIB (Conditional Identity Block) module
 */
class CIBImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new CIB module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param shortcut Whether to use shortcut connection (default: true)
     * @param e Expansion ratio (default: 0.5)
     * @param lk Whether to use RepVGGDW (default: false)
     */
    CIBImpl(int64_t c1, int64_t c2, bool shortcut = true, double e = 0.5, bool lk = false);

    /**
     * @brief Forward pass through CIB module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential _cv1;
    bool _add;
};

TORCH_MODULE(CIB);

/**
 * @brief C2fCIB module
 */
class C2fCIBImpl : public C2fImpl {
public:
    /**
     * @brief Construct a new C2fCIB module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param n Number of CIB modules (default: 1)
     * @param shortcut Whether to use shortcut connection (default: false)
     * @param lk Whether to use local key connection (default: false)
     * @param g Groups for convolutions (default: 1)
     * @param e Expansion ratio (default: 0.5)
     */
    C2fCIBImpl(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = false,
               bool lk = false, int64_t g = 1, double e = 0.5);
};

TORCH_MODULE(C2fCIB);

/**
 * @brief SCDown module for downsampling with separable convolutions
 */
class SCDownImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new SCDown module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param k Kernel size
     * @param s Stride
     */
    SCDownImpl(int64_t c1, int64_t c2, int64_t k, int64_t s);

    /**
     * @brief Forward pass through SCDown module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    Conv _cv1;
    Conv _cv2;
};

TORCH_MODULE(SCDown);

/**
 * @brief C3k2 module
 */
class C3k2Impl : public C2fImpl {
public:
    /**
     * @brief Construct a new C3k2 module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param n Number of blocks (default: 1)
     * @param c3k Whether to use C3k blocks (default: false)
     * @param e Expansion ratio (default: 0.5)
     * @param g Groups for convolutions (default: 1)
     * @param shortcut Whether to use shortcut connections (default: true)
     */
    C3k2Impl(int64_t c1, int64_t c2, int64_t n = 1, bool c3k = false,
             double e = 0.5, int64_t g = 1, bool shortcut = true);
};

TORCH_MODULE(C3k2);

/**
 * @brief C3f module
 */
class C3fImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new C3f module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param n Number of Bottleneck blocks (default: 1)
     * @param shortcut Whether to use shortcut connections (default: false)
     * @param g Groups for convolutions (default: 1)
     * @param e Expansion ratio (default: 0.5)
     */
    C3fImpl(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = false,
            int64_t g = 1, double e = 0.5);

    /**
     * @brief Forward pass through C3f module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    Conv _cv1;
    Conv _cv2;
    Conv _cv3;
    torch::nn::ModuleList _m;
};

TORCH_MODULE(C3f);

/**
 * @brief C3 module with TransformerBlock
 */
class C3TRImpl : public C3Impl {
public:
    C3TRImpl(int64_t c1, int64_t c2, int64_t n = 1, bool shortcut = true,
             int64_t g = 1, double e = 0.5);
};

TORCH_MODULE(C3TR);

/**
 * @brief ResNet block
 */
class ResNetBlockImpl : public torch::nn::Module {
public:
    ResNetBlockImpl(int64_t c1, int64_t c2, int64_t s = 1, double e = 4.0);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential _identity;
    Conv _cv1;
    Conv _cv2;
    Conv _cv3;
};

TORCH_MODULE(ResNetBlock);

/**
 * @brief ResNet layer
 */
class ResNetLayerImpl : public torch::nn::Module {
public:
    ResNetLayerImpl(int64_t c1, int64_t c2, int64_t s = 1, bool isFirst = false,
                    int64_t n = 1, double e = 4.0);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential _blocks;
};

TORCH_MODULE(ResNetLayer);

/**
 * @brief C2PSA module
 */
class C2PSAImpl : public torch::nn::Module {
public:
    C2PSAImpl(int64_t c1, int64_t c2, int64_t n = 1, double e = 0.5);
    torch::Tensor forward(torch::Tensor x);

private:
    int64_t _c;
    Conv _cv1;
    Conv _cv2;
    torch::nn::ModuleList _m;
};

TORCH_MODULE(C2PSA);

/**
 * @brief C2fPSA module
 */
class C2fPSAImpl : public C2fImpl {
public:
    C2fPSAImpl(int64_t c1, int64_t c2, int64_t n = 1, double e = 0.5);
};

TORCH_MODULE(C2fPSA);

/**
 * @brief ContrastiveHead module
 */
class ContrastiveHeadImpl : public torch::nn::Module {
public:
    ContrastiveHeadImpl();
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear _linear;
};

TORCH_MODULE(ContrastiveHead);

/**
 * @brief BN Contrastive Head
 */
class BNContrastiveHeadImpl : public torch::nn::Module {
public:
    explicit BNContrastiveHeadImpl(torch::nn::BatchNorm1d norm);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::BatchNorm1d _norm;
    torch::nn::Linear _linear;
};

TORCH_MODULE(BNContrastiveHead);

} // namespace Modules
} // namespace Model
} // namespace WheelDL
