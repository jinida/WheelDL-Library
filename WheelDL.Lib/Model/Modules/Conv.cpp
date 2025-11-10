#include "pch.h"
#include "Conv.h"
#include "Utils.h"
#include <numeric>
#include <cmath>

namespace WheelDL {
namespace Model {
namespace Modules {

int64_t autoPad(int64_t k, std::optional<int64_t> p, int64_t d) {
    // Early return if padding is explicitly provided
    if (p.has_value()) {
        if (p.value() < 0) {
            throw std::invalid_argument("Padding must be non-negative, got: " + std::to_string(p.value()));
        }
        return p.value();
    }

    // Only validate k and d if we need to auto-calculate padding
    if (k <= 0) {
        throw std::invalid_argument("Kernel size must be positive, got: " + std::to_string(k));
    }
    if (d <= 0) {
        throw std::invalid_argument("Dilation must be positive, got: " + std::to_string(d));
    }

    // Adjust kernel size for dilation
    int64_t effectiveKernel = k;
    if (d > 1) {
        effectiveKernel = d * (k - 1) + 1;
    }

    // Auto-calculate padding for 'same' convolution
    // For 'same' padding: output_size = input_size (when stride=1)
    // Formula: padding = (kernel_size - 1) / 2
    int64_t autoPadding = effectiveKernel / 2;

    return autoPadding;
}

// ============================================================================
// Conv Implementation
// ============================================================================

ConvImpl::ConvImpl(int64_t c1, int64_t c2, int64_t k, int64_t s,
                   std::optional<int64_t> p, int64_t g, int64_t d, const std::string& act) {
    // Validate input parameters
    if (c1 <= 0) {
        throw std::invalid_argument("ConvImpl: c1 must be positive, got: " + std::to_string(c1));
    }
    if (c2 <= 0) {
        throw std::invalid_argument("ConvImpl: c2 must be positive, got: " + std::to_string(c2));
    }
    if (k <= 0) {
        throw std::invalid_argument("ConvImpl: k must be positive, got: " + std::to_string(k));
    }
    if (s <= 0) {
        throw std::invalid_argument("ConvImpl: s must be positive, got: " + std::to_string(s));
    }
    if (g <= 0) {
        throw std::invalid_argument("ConvImpl: g must be positive, got: " + std::to_string(g));
    }

    int64_t padding = autoPad(k, p, d);

    // Create Conv2d layer
    _conv = register_module("conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, k)
            .stride(s)
            .padding(padding)
            .groups(g)
            .dilation(d)
            .bias(false)));

    // Create BatchNorm2d layer
    _bn = register_module("bn", torch::nn::BatchNorm2d(c2));

    // Create activation layer using createActivation helper
    _act = createActivation(act);
}

torch::Tensor ConvImpl::forward(torch::Tensor x) {
    x = _conv->forward(x);
    x = _bn->forward(x);
    return _act.forward<torch::Tensor>(x);
}

// ============================================================================
// DWConv Implementation
// ============================================================================

DWConvImpl::DWConvImpl(int64_t c1, int64_t c2, int64_t k, int64_t s,
                       int64_t d, const std::string& act)
    : ConvImpl(c1, c2, k, s, std::nullopt,
               std::gcd(c1, c2), // groups = gcd(c1, c2) for depthwise
               d, act) {
}

// ============================================================================
// Conv2 Implementation
// ============================================================================

Conv2Impl::Conv2Impl(int64_t c1, int64_t c2, int64_t k, int64_t s,
                     std::optional<int64_t> p, int64_t g, int64_t d, const std::string& act)
    : ConvImpl(c1, c2, k, s, p, g, d, act) {

    int64_t padding = autoPad(1, p, d);

    // Add 1x1 conv
    _cv2 = register_module("cv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, 1)
            .stride(s)
            .padding(padding)
            .groups(g)
            .dilation(d)
            .bias(false)));
}

torch::Tensor Conv2Impl::forward(torch::Tensor x) {
    auto out1 = _conv->forward(x);
    auto out2 = _cv2->forward(x);
    auto combined = out1 + out2;
    combined = _bn->forward(combined);
    return _act.forward<torch::Tensor>(combined);
}

// ============================================================================
// LightConv Implementation
// ============================================================================

LightConvImpl::LightConvImpl(int64_t c1, int64_t c2, int64_t k)
{
    // Validate input parameters
    if (c1 <= 0) {
        throw std::invalid_argument("LightConvImpl: c1 must be positive, got: " + std::to_string(c1));
    }
    if (c2 <= 0) {
        throw std::invalid_argument("LightConvImpl: c2 must be positive, got: " + std::to_string(c2));
    }
    if (k <= 0) {
        throw std::invalid_argument("LightConvImpl: k must be positive, got: " + std::to_string(k));
    }

    _conv1 = register_module("conv1", Conv(c1, c2, 1, 1, std::nullopt, 1, 1, "Identity"));
    _conv2 = register_module("conv2", DWConv(c2, c2, k, 1, 1, "SiLU"));
}

torch::Tensor LightConvImpl::forward(torch::Tensor x) {
    x = _conv1->forward(x);
    x = _conv2->forward(x);
    return x;
}

// ============================================================================
// GhostConv Implementation
// ============================================================================

GhostConvImpl::GhostConvImpl(int64_t c1, int64_t c2, int64_t k, int64_t s,
                             int64_t g, const std::string& act) {
    // Validate input parameters
    if (c1 <= 0) {
        throw std::invalid_argument("GhostConvImpl: c1 must be positive, got: " + std::to_string(c1));
    }
    if (c2 <= 0) {
        throw std::invalid_argument("GhostConvImpl: c2 must be positive, got: " + std::to_string(c2));
    }
    if (c2 % 2 != 0) {
        throw std::invalid_argument("GhostConvImpl: c2 must be even, got: " + std::to_string(c2));
    }

    int64_t hiddenChannels = c2 / 2;

    _cv1 = register_module("cv1", Conv(c1, hiddenChannels, k, s, std::nullopt, g, 1, act));
    _cv2 = register_module("cv2", Conv(hiddenChannels, hiddenChannels, 5, 1, std::nullopt, hiddenChannels, 1, act));
}

torch::Tensor GhostConvImpl::forward(torch::Tensor x) {
    auto y = _cv1->forward(x);
    auto y2 = _cv2->forward(y);
    return torch::cat({y, y2}, 1);
}

// ============================================================================
// RepConv Implementation
// ============================================================================

RepConvImpl::RepConvImpl(int64_t c1, int64_t c2, int64_t k, int64_t s,
                         int64_t p, int64_t g, int64_t d,
                         const std::string& act, bool bn, bool deploy)
    : _g(g), _c1(c1), _c2(c2) {

    // Assert k == 3 and p == 1
    if (k != 3 || p != 1) {
        throw std::invalid_argument("RepConv requires k=3 and p=1");
    }

    // Create activation layer
    _act = createActivation(act);

    // Create optional batch norm for identity branch
    if (bn && c2 == c1 && s == 1) {
        _bn = register_module("bn", torch::nn::BatchNorm2d(c1));
    }

    // Create convolution branches (no activation, will be applied after combining)
    _conv1 = register_module("conv1", Conv(c1, c2, k, s, p, g, d, "Identity"));
    _conv2 = register_module("conv2", Conv(c1, c2, 1, s, p - k / 2, g, d, "Identity"));
}

torch::Tensor RepConvImpl::forward(torch::Tensor x) {
    auto out1 = _conv1->forward(x);
    auto out2 = _conv2->forward(x);

    torch::Tensor result = out1 + out2;

    if (!_bn.is_empty()) {
        result = result + _bn->forward(x);
    }

    return _act.forward<torch::Tensor>(result);
}

// ============================================================================
// DWConvTranspose2d Implementation
// ============================================================================

DWConvTranspose2dImpl::DWConvTranspose2dImpl(int64_t c1, int64_t c2, int64_t k,
                                             int64_t s, int64_t p1, int64_t p2)
    : torch::nn::ConvTranspose2dImpl(
          torch::nn::ConvTranspose2dOptions(c1, c2, k)
              .stride(s)
              .padding(p1)
              .output_padding(p2)
              .groups(std::gcd(c1, c2))) {
}

// ============================================================================
// ConvTranspose Implementation
// ============================================================================

ConvTransposeImpl::ConvTransposeImpl(int64_t c1, int64_t c2, int64_t k,
                                     int64_t s, int64_t p, bool bn, const std::string& act) {
    // Create ConvTranspose2d layer
    _convTranspose = register_module("conv_transpose",
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(c1, c2, k)
            .stride(s)
            .padding(p)
            .bias(!bn)));

    // Create BatchNorm2d or Identity layer
    if (bn) {
        auto batchNorm = torch::nn::BatchNorm2d(c2);
        _bn = register_module("bn", batchNorm);
    } else {
        auto identity = torch::nn::Identity();
        _bn = register_module("bn", identity);
    }

    // Create activation layer using createActivation helper
    _act = createActivation(act);
}

torch::Tensor ConvTransposeImpl::forward(torch::Tensor x) {
    x = _convTranspose->forward(x);
    x = _bn.forward<torch::Tensor>(x);
    return _act.forward<torch::Tensor>(x);
}

// ============================================================================
// Focus Implementation
// ============================================================================

FocusImpl::FocusImpl(int64_t c1, int64_t c2, int64_t k, int64_t s,
                     std::optional<int64_t> p, int64_t g, const std::string& act) {
    // Focus slices input into 4 parts, so we multiply c1 by 4
    _conv = register_module("conv", Conv(c1 * 4, c2, k, s, p, g, 1, act));
}

torch::Tensor FocusImpl::forward(torch::Tensor x) {
    using namespace torch::indexing;

    // Slice input into 4 parts: [::2, ::2], [1::2, ::2], [::2, 1::2], [1::2, 1::2]
    // x shape: (B, C, H, W)
    auto x1 = x.index({Slice(), Slice(), Slice(None, None, 2), Slice(None, None, 2)});
    auto x2 = x.index({Slice(), Slice(), Slice(1, None, 2), Slice(None, None, 2)});
    auto x3 = x.index({Slice(), Slice(), Slice(None, None, 2), Slice(1, None, 2)});
    auto x4 = x.index({Slice(), Slice(), Slice(1, None, 2), Slice(1, None, 2)});

    // Concatenate along channel dimension
    auto concatenated = torch::cat({x1, x2, x3, x4}, 1);

    return _conv->forward(concatenated);
}

// ============================================================================
// Concat Implementation
// ============================================================================

ConcatImpl::ConcatImpl(int64_t dimension)
    : _d(dimension) {
}

torch::Tensor ConcatImpl::forward(std::vector<torch::Tensor> x) {
    return torch::cat(x, _d);
}

// ============================================================================
// Index Implementation
// ============================================================================

IndexImpl::IndexImpl(int64_t index)
    : _index(index) {
}

torch::Tensor IndexImpl::forward(std::vector<torch::Tensor> x) {
    // Validate index bounds
    if (_index < 0 || _index >= static_cast<int64_t>(x.size())) {
        throw std::out_of_range("Index out of bounds in Index forward: index=" +
                               std::to_string(_index) + ", size=" + std::to_string(x.size()));
    }
    return x[_index];
}

} // namespace Modules
} // namespace Model
} // namespace WheelDL
