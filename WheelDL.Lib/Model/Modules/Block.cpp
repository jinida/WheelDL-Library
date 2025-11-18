#include "pch.h"
#include "Block.h"
#include <limits>

namespace WheelDL {
    namespace Model {
        namespace Modules {

            // Helper function to safely multiply channels with expansion ratio
            inline int64_t safeChannelMultiply(int64_t c, double e, const char* context)
            {
                double result = c * e;
                if (result > static_cast<double>(std::numeric_limits<int64_t>::max())) {
                    throw std::overflow_error(std::string(context) +
                        ": Channel calculation would overflow (c=" +
                        std::to_string(c) + ", e=" + std::to_string(e) + ")");
                }

                if (result < 0) {
                    throw std::invalid_argument(std::string(context) +
                        ": Channel calculation resulted in negative value");
                }
                return static_cast<int64_t>(result);
            }

            // ============================================================================
            // DFL Implementation
            // ============================================================================

            DFLImpl::DFLImpl(int64_t c1)
                : _c1(c1) {
                // Create conv layer with no gradient
                _conv = register_module("conv",
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, 1, 1).bias(false)));
                _conv->weight.set_requires_grad(false);

                // Initialize weights with arange
                auto x = torch::arange(c1, torch::kFloat);
                _conv->weight.data().copy_(x.view({ 1, c1, 1, 1 }));
            }

            torch::Tensor DFLImpl::forward(torch::Tensor x)
            {
                auto b = x.size(0);
                auto a = x.size(2);

                // Reshape and apply DFL
                auto reshaped = x.view({ b, 4, _c1, a }).transpose(2, 1).softmax(1);
                return _conv->forward(reshaped).view({ b, 4, a });
            }

            ProtoImpl::ProtoImpl(int64_t c1, int64_t cMid, int64_t c2) {
                _cv1 = register_module("cv1", Conv(c1, cMid, 3));
                _upsample = register_module("upsample",
                    torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(cMid, cMid, 2)
                        .stride(2)
                        .padding(0)
                        .bias(true)));
                _cv2 = register_module("cv2", Conv(cMid, cMid, 3));
                _cv3 = register_module("cv3", Conv(cMid, c2, 1));
            }

            torch::Tensor ProtoImpl::forward(torch::Tensor x) {
                x = _cv1->forward(x);
                x = _upsample->forward(x);
                x = _cv2->forward(x);
                return _cv3->forward(x);
            }

            // ============================================================================
            // SPP Implementation
            // ============================================================================

            SPPImpl::SPPImpl(int64_t c1, int64_t c2, std::vector<int64_t> k) {
                // Validate c1 is even to avoid integer division truncation
                if (c1 % 2 != 0) {
                    throw std::invalid_argument("SPPImpl: c1 must be even, got: " + std::to_string(c1));
                }
                int64_t cMid = c1 / 2;
                _cv1 = register_module("cv1", Conv(c1, cMid, 1, 1));
                _cv2 = register_module("cv2", Conv(cMid * (k.size() + 1), c2, 1, 1));

                _m = register_module("m", torch::nn::ModuleList());
                for (auto kernelSize : k) {
                    _m->push_back(torch::nn::MaxPool2d(
                        torch::nn::MaxPool2dOptions(kernelSize)
                        .stride(1)
                        .padding(kernelSize / 2)));
                }
            }

            torch::Tensor SPPImpl::forward(torch::Tensor x) {
                x = _cv1->forward(x);

                std::vector<torch::Tensor> outputs;
                outputs.push_back(x);

                for (auto& m : *_m) {
                    auto maxpool = m->as<torch::nn::MaxPool2d>();
                    if (!maxpool)
                    {
                        throw std::runtime_error("SPPImpl::forward - MaxPool2d module is null");
                    }
                    outputs.push_back(maxpool->forward(x));
                }

                return _cv2->forward(torch::cat(outputs, 1));
            }

            // ============================================================================
            // SPPF Implementation
            // ============================================================================

            SPPFImpl::SPPFImpl(int64_t c1, int64_t c2, int64_t k)
            {
                // Validate c1 is even to avoid integer division truncation
                if (c1 % 2 != 0) {
                    throw std::invalid_argument("SPPFImpl: c1 must be even, got: " + std::to_string(c1));
                }
                int64_t cMid = c1 / 2;
                _cv1 = register_module("cv1", Conv(c1, cMid, 1, 1));
                _cv2 = register_module("cv2", Conv(cMid * 4, c2, 1, 1));
                _m = register_module("m",
                    torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(k)
                        .stride(1)
                        .padding(k / 2)));
            }

            torch::Tensor SPPFImpl::forward(torch::Tensor x) {
                std::vector<torch::Tensor> y;
                y.push_back(_cv1->forward(x));

                // Apply max pooling 3 times
                for (int i = 0; i < 3; ++i) {
                    y.push_back(_m->forward(y.back()));
                }

                return _cv2->forward(torch::cat(y, 1));
            }

            // ============================================================================
            // Bottleneck Implementation
            // ============================================================================

            BottleneckImpl::BottleneckImpl(int64_t c1, int64_t c2, bool shortcut,
                int64_t g, std::vector<int64_t> k, double e) {
                int64_t cMid = safeChannelMultiply(c2, e, "BottleneckImpl");
                _cv1 = register_module("cv1", Conv(c1, cMid, k[0], 1));
                _cv2 = register_module("cv2", Conv(cMid, c2, k[1], 1, std::nullopt, g));
                _add = shortcut && (c1 == c2);
            }

            torch::Tensor BottleneckImpl::forward(torch::Tensor x) {
                auto out = _cv2->forward(_cv1->forward(x));
                return _add ? (x + out) : out;
            }

            // ============================================================================
            // C1 Implementation
            // ============================================================================

            C1Impl::C1Impl(int64_t c1, int64_t c2, int64_t n) {
                _cv1 = register_module("cv1", Conv(c1, c2, 1, 1));

                torch::nn::Sequential seq;
                for (int64_t i = 0; i < n; ++i) {
                    seq->push_back(Conv(c2, c2, 3));
                }
                _m = register_module("m", seq);
            }

            torch::Tensor C1Impl::forward(torch::Tensor x) {
                auto y = _cv1->forward(x);
                return _m->forward(y) + y;
            }

            // ============================================================================
            // C2 Implementation
            // ============================================================================

            C2Impl::C2Impl(int64_t c1, int64_t c2, int64_t n, bool shortcut,
                int64_t g, double e) {
                _c = safeChannelMultiply(c2, e, "C2Impl");
                _cv1 = register_module("cv1", Conv(c1, 2 * _c, 1, 1));
                _cv2 = register_module("cv2", Conv(2 * _c, c2, 1));

                torch::nn::Sequential seq;
                for (int64_t i = 0; i < n; ++i) {
                    seq->push_back(Bottleneck(_c, _c, shortcut, g, std::vector<int64_t>{3, 3}, 1.0));
                }
                _m = register_module("m", seq);
            }

            torch::Tensor C2Impl::forward(torch::Tensor x) {
                auto y = _cv1->forward(x);
                auto totalSize = y.size(1);
                auto c1 = totalSize / 2;
                auto c2 = totalSize - c1;  // Handle odd sizes correctly

                // Validate that narrow operations will be safe
                if (totalSize <= 0) {
                    throw std::out_of_range("C2Impl::forward - Invalid tensor size: totalSize=" + std::to_string(totalSize));
                }
                if (c1 < 0 || c2 < 0 || c1 > totalSize || c1 + c2 > totalSize) {
                    throw std::out_of_range("C2Impl::forward - Invalid split sizes: c1=" + std::to_string(c1) +
                        ", c2=" + std::to_string(c2) +
                        ", totalSize=" + std::to_string(totalSize));
                }

                auto a = y.narrow(1, 0, c1);
                auto b = y.narrow(1, c1, c2);
                return _cv2->forward(torch::cat({ _m->forward(a), b }, 1));
            }

            // ============================================================================
            // C2f Implementation
            // ============================================================================

            C2fImpl::C2fImpl(int64_t c1, int64_t c2, int64_t n, bool shortcut,
                int64_t g, double e) {
                _c = safeChannelMultiply(c2, e, "C2fImpl");
                _cv1 = register_module("cv1", Conv(c1, 2 * _c, 1, 1));
                _cv2 = register_module("cv2", Conv((2 + n) * _c, c2, 1));

                _m = register_module("m", torch::nn::ModuleList());
                for (int64_t i = 0; i < n; ++i) {
                    _m->push_back(Bottleneck(_c, _c, shortcut, g, std::vector<int64_t>{3, 3}, 1.0));
                }
            }

            torch::Tensor C2fImpl::forward(torch::Tensor x) {
                auto cv1Out = _cv1->forward(x);
                auto totalChannels = cv1Out.size(1);

                // Validate bounds BEFORE calculating split sizes
                if (totalChannels <= 0) {
                    throw std::out_of_range("C2fImpl::forward - Invalid tensor channels: totalChannels=" +
                        std::to_string(totalChannels));
                }

                auto c = totalChannels / 2;

                // Ensure narrow operations will be within bounds
                if (c <= 0 || c > totalChannels || (c + c) > totalChannels) {
                    throw std::out_of_range("C2fImpl::forward - Invalid split size: c=" +
                        std::to_string(c) + ", totalChannels=" + std::to_string(totalChannels));
                }

                std::vector<torch::Tensor> y;
                y.reserve(2 + _m->size());
                y.push_back(cv1Out.narrow(1, 0, c));
                y.push_back(cv1Out.narrow(1, c, c));

                for (auto& m : *_m)
                {
                    auto bottleneck = m->as<IBlockImpl>();
                    if (!bottleneck)
                    {
                        throw std::runtime_error("C2fImpl::forward - Bottleneck module is null");
                    }
                    y.push_back(bottleneck->forward(y.back()));
                }

                return _cv2->forward(torch::cat(y, 1));
            }

            // ============================================================================
            // C3 Implementation
            // ============================================================================

            C3Impl::C3Impl(int64_t c1, int64_t c2, int64_t n, bool shortcut,
                int64_t g, double e) {
                _c = safeChannelMultiply(c2, e, "C3Impl");
                _cv1 = register_module("cv1", Conv(c1, _c, 1, 1));
                _cv2 = register_module("cv2", Conv(c1, _c, 1, 1));
                _cv3 = register_module("cv3", Conv(2 * _c, c2, 1));

                torch::nn::Sequential seq;
                for (int64_t i = 0; i < n; ++i) {
                    seq->push_back(Bottleneck(_c, _c, shortcut, g, std::vector<int64_t>{1, 3}, 1.0));
                }
                _m = register_module("m", seq);
            }

            torch::Tensor C3Impl::forward(torch::Tensor x) {
                auto out1 = _m->forward(_cv1->forward(x));
                auto out2 = _cv2->forward(x);
                return _cv3->forward(torch::cat({ out1, out2 }, 1));
            }

            // ============================================================================
            // GhostBottleneck Implementation
            // ============================================================================

            GhostBottleneckImpl::GhostBottleneckImpl(int64_t c1, int64_t c2, int64_t k, int64_t s) {
                // Validate c2 is even to avoid integer division truncation
                if (c2 % 2 != 0) {
                    throw std::invalid_argument("GhostBottleneckImpl: c2 must be even, got: " + std::to_string(c2));
                }
                int64_t cMid = c2 / 2;

                // Build conv sequence
                torch::nn::Sequential convSeq;
                convSeq->push_back(GhostConv(c1, cMid, 1, 1));

                if (s == 2) {
                    convSeq->push_back(DWConv(cMid, cMid, k, s, 1));
                }
                else {
                    convSeq->push_back(torch::nn::Identity());
                }

                convSeq->push_back(GhostConv(cMid, c2, 1, 1, 1));
                _conv = register_module("conv", convSeq);

                // Build shortcut sequence
                torch::nn::Sequential shortcutSeq;
                if (s == 2) {
                    shortcutSeq->push_back(DWConv(c1, c1, k, s, 1));
                    shortcutSeq->push_back(Conv(c1, c2, 1, 1, std::nullopt, 1, 1));
                }
                else {
                    shortcutSeq->push_back(torch::nn::Identity());
                }
                _shortcut = register_module("shortcut", shortcutSeq);
            }

            torch::Tensor GhostBottleneckImpl::forward(torch::Tensor x) {
                return _conv->forward(x) + _shortcut->forward(x);
            }

            // ============================================================================
            // HGStem Implementation
            // ============================================================================

            HGStemImpl::HGStemImpl(int64_t c1, int64_t cm, int64_t c2) {
                // Validate cm is even to avoid integer division truncation
                if (cm % 2 != 0) {
                    throw std::invalid_argument("HGStemImpl: cm must be even, got: " + std::to_string(cm));
                }
                // Use ReLU activation (Note: Python uses nn.ReLU() as parameter)
                _stem1 = register_module("stem1", Conv(c1, cm, 3, 2));
                _stem2a = register_module("stem2a", Conv(cm, cm / 2, 2, 1, 0));
                _stem2b = register_module("stem2b", Conv(cm / 2, cm, 2, 1, 0));
                _stem3 = register_module("stem3", Conv(cm * 2, cm, 3, 2));
                _stem4 = register_module("stem4", Conv(cm, c2, 1, 1));
                _pool = register_module("pool",
                    torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)
                        .stride(1)
                        .padding(0)
                        .ceil_mode(true)));
            }

            torch::Tensor HGStemImpl::forward(torch::Tensor x) {
                x = _stem1->forward(x);
                x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({ 0, 1, 0, 1 }));
                auto x2 = _stem2a->forward(x);
                x2 = torch::nn::functional::pad(x2, torch::nn::functional::PadFuncOptions({ 0, 1, 0, 1 }));
                x2 = _stem2b->forward(x2);
                auto x1 = _pool->forward(x);
                x = torch::cat({ x1, x2 }, /*dim=*/1);
                x = _stem3->forward(x);
                x = _stem4->forward(x);
                return x;
            }

            // ============================================================================
            // HGBlock Implementation
            // ============================================================================

            HGBlockImpl::HGBlockImpl(int64_t c1, int64_t cm, int64_t c2, int64_t k,
                int64_t n, bool lightConv, bool shortcut)
                : _lightConv(lightConv) {
                _m = register_module("m", torch::nn::ModuleList());

                for (int64_t i = 0; i < n; ++i) {
                    int64_t inputChannels = (i == 0) ? c1 : cm;
                    if (lightConv)
                    {
                        _m->push_back(LightConv(inputChannels, cm, k));
                    }
                    else
                    {
                        _m->push_back(Conv(inputChannels, cm, k));
                    }
                }

                _sc = register_module("sc", Conv(c1 + n * cm, c2 / 2, 1, 1));
                _ec = register_module("ec", Conv(c2 / 2, c2, 1, 1));
                _add = shortcut && (c1 == c2);
            }

            torch::Tensor HGBlockImpl::forward(torch::Tensor x)
            {
                std::vector<torch::Tensor> y;
                y.push_back(x);

                for (auto& module : *_m)
                {
                    auto conv = module->as<IBlockImpl>();
                    y.push_back(conv->forward(y.back()));
                }
                auto output = _ec->forward(_sc->forward(torch::cat(y, 1)));
                return _add ? (output + x) : output;
            }

            // ============================================================================
            // RepBottleneck Implementation
            // ============================================================================

            RepBottleneckImpl::RepBottleneckImpl(int64_t c1, int64_t c2, bool shortcut,
                int64_t g, std::vector<int64_t> k, double e) {
                int64_t cMid = safeChannelMultiply(c2, e, "RepBottleneckImpl");

                _cv1 = register_module("cv1", RepConv(c1, cMid, k[0], 1));
                _cv2 = register_module("cv2", Conv(cMid, c2, k[1], 1, std::nullopt, g));
                _add = shortcut && (c1 == c2);
            }

            torch::Tensor RepBottleneckImpl::forward(torch::Tensor x) {
                auto out = _cv2->forward(_cv1->forward(x));
                return _add ? (x + out) : out;
            }

            // ============================================================================
            // C3x Implementation
            // ============================================================================

            C3xImpl::C3xImpl(int64_t c1, int64_t c2, int64_t n, bool shortcut,
                int64_t g, double e)
                : C3Impl(c1, c2, n, shortcut, g, e) {
                torch::nn::Sequential seq;
                for (int64_t i = 0; i < n; ++i) {
                    seq->push_back(Bottleneck(_c, _c, shortcut, g, std::vector<int64_t>{1, 3}, 1.0));
                }
                _m = seq;
            }

            // ============================================================================
            // RepC3 Implementation
            // ============================================================================

            RepC3Impl::RepC3Impl(int64_t c1, int64_t c2, int64_t n, double e) {
                _c = safeChannelMultiply(c2, e, "RepC3Impl");
                _cv1 = register_module("cv1", Conv(c1, _c, 1, 1));
                _cv2 = register_module("cv2", Conv(c1, _c, 1, 1));

                torch::nn::Sequential seq;
                for (int64_t i = 0; i < n; ++i) {
                    seq->push_back(RepConv(_c, _c));
                }
                _m = register_module("m", seq);

                if (_c != c2) {
                    auto conv = Conv(_c, c2, 1, 1);
                    _cv3 = register_module("cv3", conv);
                }
                else {
                    auto identity = torch::nn::Identity();
                    _cv3 = register_module("cv3", identity);
                }
            }

            torch::Tensor RepC3Impl::forward(torch::Tensor x) {
                auto out1 = _m->forward(_cv1->forward(x));
                auto out2 = _cv2->forward(x);
                return _cv3.forward<torch::Tensor>(out1 + out2);
            }

            // ============================================================================
            // C3Ghost Implementation
            // ============================================================================

            C3GhostImpl::C3GhostImpl(int64_t c1, int64_t c2, int64_t n, bool shortcut,
                int64_t g, double e)
                : C3Impl(c1, c2, n, shortcut, g, e) {
                torch::nn::Sequential seq;
                for (int64_t i = 0; i < n; ++i) {
                    seq->push_back(GhostBottleneck(_c, _c));
                }
                _m = seq;
            }

            // ============================================================================
            // BottleneckCSP Implementation
            // ============================================================================

            BottleneckCSPImpl::BottleneckCSPImpl(int64_t c1, int64_t c2, int64_t n,
                bool shortcut, int64_t g, double e) {
                int64_t cMid = safeChannelMultiply(c2, e, "BottleneckCSPImpl");
                _cv1 = register_module("cv1", Conv(c1, cMid, 1, 1));
                _cv2 = register_module("cv2",
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, cMid, 1)
                        .stride(1)
                        .bias(false)));
                _cv3 = register_module("cv3",
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(cMid, cMid, 1)
                        .stride(1)
                        .bias(false)));
                _cv4 = register_module("cv4", Conv(2 * cMid, c2, 1, 1));
                _bn = register_module("bn", torch::nn::BatchNorm2d(2 * cMid));
                _act = register_module("act", torch::nn::SiLU());

                torch::nn::Sequential seq;
                for (int64_t i = 0; i < n; ++i) {
                    seq->push_back(Bottleneck(cMid, cMid, shortcut, g, std::vector<int64_t>{3, 3}, 1.0));
                }
                _m = register_module("m", seq);
            }

            torch::Tensor BottleneckCSPImpl::forward(torch::Tensor x) {
                auto y1 = _cv3->forward(_m->forward(_cv1->forward(x)));
                auto y2 = _cv2->forward(x);
                return _cv4->forward(_act->forward(_bn->forward(torch::cat({ y1, y2 }, 1))));
            }

            // ============================================================================
            // RepCSP Implementation
            // ============================================================================

            RepCSPImpl::RepCSPImpl(int64_t c1, int64_t c2, int64_t n, bool shortcut,
                int64_t g, double e)
                : C3Impl(c1, c2, n, shortcut, g, e) {
                torch::nn::Sequential seq;
                for (int64_t i = 0; i < n; ++i) {
                    seq->push_back(RepBottleneck(_c, _c, shortcut, g, std::vector<int64_t>{3, 3}, 1.0));
                }
                _m = seq;
            }

            // ============================================================================
            // ADown Implementation
            // ============================================================================

            ADownImpl::ADownImpl(int64_t c1, int64_t c2) {
                // Validate c1 and c2 are even to avoid integer division truncation
                if (c1 % 2 != 0) {
                    throw std::invalid_argument("ADownImpl: c1 must be even, got: " + std::to_string(c1));
                }
                if (c2 % 2 != 0) {
                    throw std::invalid_argument("ADownImpl: c2 must be even, got: " + std::to_string(c2));
                }
                _c = c2 / 2;
                _cv1 = register_module("cv1", Conv(c1 / 2, _c, 3, 2, 1));
                _cv2 = register_module("cv2", Conv(c1 / 2, _c, 1, 1, 0));
            }

            torch::Tensor ADownImpl::forward(torch::Tensor x) {
                x = torch::nn::functional::avg_pool2d(x,
                    torch::nn::functional::AvgPool2dFuncOptions(2).stride(1).padding(0).count_include_pad(true));
                auto totalChannels = x.size(1);
                auto c = totalChannels / 2;

                // Validate narrow bounds
                if (c + c > totalChannels) {
                    throw std::out_of_range("ADownImpl::forward - narrow would exceed bounds: c=" +
                        std::to_string(c) + ", totalChannels=" + std::to_string(totalChannels));
                }

                auto x1 = _cv1->forward(x.narrow(1, 0, c));
                auto x2 = torch::nn::functional::max_pool2d(x.narrow(1, c, c),
                    torch::nn::functional::MaxPool2dFuncOptions(3).stride(2).padding(1));
                x2 = _cv2->forward(x2);
                return torch::cat({ x1, x2 }, 1);
            }

            // ============================================================================
            // AConv Implementation
            // ============================================================================

            AConvImpl::AConvImpl(int64_t c1, int64_t c2) {
                _cv1 = register_module("cv1", Conv(c1, c2, 3, 2, 1));
            }

            torch::Tensor AConvImpl::forward(torch::Tensor x) {
                x = torch::nn::functional::avg_pool2d(x,
                    torch::nn::functional::AvgPool2dFuncOptions(2).stride(1).padding(0).count_include_pad(true));
                return _cv1->forward(x);
            }

            // ============================================================================
            // SPPELAN Implementation
            // ============================================================================

            SPPELANImpl::SPPELANImpl(int64_t c1, int64_t c2, int64_t c3, int64_t k) {
                _c = c3;
                _cv1 = register_module("cv1", Conv(c1, c3, 1, 1));
                _cv2 = register_module("cv2",
                    torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(k)
                        .stride(1)
                        .padding(k / 2)));
                _cv3 = register_module("cv3",
                    torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(k)
                        .stride(1)
                        .padding(k / 2)));
                _cv4 = register_module("cv4",
                    torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(k)
                        .stride(1)
                        .padding(k / 2)));
                _cv5 = register_module("cv5", Conv(4 * c3, c2, 1, 1));
            }

            torch::Tensor SPPELANImpl::forward(torch::Tensor x) {
                std::vector<torch::Tensor> y;
                y.push_back(_cv1->forward(x));

                y.push_back(_cv2->forward(y.back()));
                y.push_back(_cv3->forward(y.back()));
                y.push_back(_cv4->forward(y.back()));

                return _cv5->forward(torch::cat(y, 1));
            }

            // ============================================================================
            // RepNCSPELAN4 Implementation
            // ============================================================================

            RepNCSPELAN4Impl::RepNCSPELAN4Impl(int64_t c1, int64_t c2, int64_t c3,
                int64_t c4, int64_t n) {
                _c = c3 / 2;
                _cv1 = register_module("cv1", Conv(c1, c3, 1, 1));

                torch::nn::Sequential seq2;
                seq2->push_back(RepCSP(c3 / 2, c4, n));
                seq2->push_back(Conv(c4, c4, 3, 1));
                _cv2 = register_module("cv2", seq2);

                torch::nn::Sequential seq3;
                seq3->push_back(RepCSP(c4, c4, n));
                seq3->push_back(Conv(c4, c4, 3, 1));
                _cv3 = register_module("cv3", seq3);

                _cv4 = register_module("cv4", Conv(c3 + (2 * c4), c2, 1, 1));
            }

            torch::Tensor RepNCSPELAN4Impl::forward(torch::Tensor x) {
                auto cv1Out = _cv1->forward(x);
                auto totalChannels = cv1Out.size(1);
                auto c = totalChannels / 2;

                // Validate narrow bounds
                if (c + c > totalChannels) {
                    throw std::out_of_range("RepNCSPELAN4Impl::forward - narrow would exceed bounds: c=" +
                        std::to_string(c) + ", totalChannels=" + std::to_string(totalChannels));
                }

                std::vector<torch::Tensor> y;
                y.reserve(4);
                y.push_back(cv1Out.narrow(1, 0, c));
                y.push_back(cv1Out.narrow(1, c, c));
                y.push_back(_cv2->forward(y.back()));
                y.push_back(_cv3->forward(y.back()));

                return _cv4->forward(torch::cat(y, 1));
            }

            // ============================================================================
            // ELAN1 Implementation
            // ============================================================================

            ELAN1Impl::ELAN1Impl(int64_t c1, int64_t c2, int64_t c3, int64_t c4)
                : RepNCSPELAN4Impl(c1, c2, c3, c4, 1) {
                _c = c3 / 2;
                _cv1 = register_module("cv1", Conv(c1, c3, 1, 1));

                torch::nn::Sequential cv2;
                cv2->push_back(Conv(c3 / 2, c4, 3, 1));
                _cv2 = register_module("cv2", cv2);

                torch::nn::Sequential cv3;
                cv3->push_back(Conv(c4, c4, 3, 1));
                _cv3 = register_module("cv3", cv3);

                _cv4 = register_module("cv4", Conv(c3 + (2 * c4), c2, 1, 1));
            }

            // ============================================================================
            // CBLinear Implementation
            // ============================================================================

            CBLinearImpl::CBLinearImpl(int64_t c1, std::vector<int64_t> c2s, int64_t k,
                int64_t s, std::optional<int64_t> p, int64_t g) {
                _c2s = c2s;
                int64_t totalChannels = std::accumulate(c2s.begin(), c2s.end(), 0LL);
                int64_t padding = p.has_value() ? p.value() : (k / 2);

                _conv = register_module("conv",
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, totalChannels, k)
                        .stride(s)
                        .padding(padding)
                        .groups(g)
                        .bias(true)));
            }

            std::vector<torch::Tensor> CBLinearImpl::forward(torch::Tensor x) {
                auto output = _conv->forward(x);
                return output.split(_c2s, /*dim=*/1);
            }

            // ============================================================================
            // CBFuse Implementation
            // ============================================================================

            CBFuseImpl::CBFuseImpl(std::vector<int64_t> idx)
                : _idx(idx) {
            }

            torch::Tensor CBFuseImpl::forward(std::vector<torch::Tensor> xs) {
                // Validate inputs
                if (xs.empty()) {
                    throw std::invalid_argument("CBFuseImpl::forward received empty xs vector");
                }
                if (_idx.size() < xs.size() - 1) {
                    throw std::invalid_argument("_idx size (" + std::to_string(_idx.size()) +
                        ") must be >= xs.size()-1 (" + std::to_string(xs.size() - 1) + ")");
                }

                auto targetSize = xs.back().sizes().slice(2);
                std::vector<torch::Tensor> res;

                for (size_t i = 0; i < xs.size() - 1; ++i) {
                    auto interpolated = torch::nn::functional::interpolate(
                        xs[i].index({ torch::indexing::Slice(), _idx[i] }),
                        torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>(targetSize.begin(), targetSize.end()))
                        .mode(torch::kNearest));
                    res.push_back(interpolated);
                }

                res.push_back(xs.back());
                return torch::sum(torch::stack(res), /*dim=*/0);
            }

            // ============================================================================
            // C3k Implementation
            // ============================================================================

            C3kImpl::C3kImpl(int64_t c1, int64_t c2, int64_t n, bool shortcut,
                int64_t g, double e, int64_t k)
                : C3Impl(c1, c2, n, shortcut, g, e) {
                torch::nn::Sequential seq;
                for (int64_t i = 0; i < n; ++i) {
                    seq->push_back(Bottleneck(_c, _c, shortcut, g, std::vector<int64_t>{k, k}, 1.0));
                }
                _m = seq;
            }

            // ============================================================================
            // RepVGGDW Implementation
            // ============================================================================

            RepVGGDWImpl::RepVGGDWImpl(int64_t ed) {
                _conv = register_module("conv", Conv(ed, ed, 7, 1, 3, ed, 1));
                _conv1 = register_module("conv1", Conv(ed, ed, 3, 1, 1, ed, 1));
                _dim = ed;
                _act = register_module("act", torch::nn::SiLU());
            }

            torch::Tensor RepVGGDWImpl::forward(torch::Tensor x) {
                return _act->forward(_conv->forward(x) + _conv1->forward(x));
            }

            // ============================================================================
            // CIB Implementation
            // ============================================================================

            CIBImpl::CIBImpl(int64_t c1, int64_t c2, bool shortcut, double e, bool lk) {
                int64_t cMid = safeChannelMultiply(c2, e, "CIBImpl");

                torch::nn::Sequential seq;
                seq->push_back(Conv(c1, c1, 3, 1, std::nullopt, c1));
                seq->push_back(Conv(c1, 2 * cMid, 1));

                if (lk) {
                    seq->push_back(RepVGGDW(2 * cMid));
                }
                else {
                    seq->push_back(Conv(2 * cMid, 2 * cMid, 3, 1, std::nullopt, 2 * cMid));
                }

                seq->push_back(Conv(2 * cMid, c2, 1));
                seq->push_back(Conv(c2, c2, 3, 1, std::nullopt, c2));

                _cv1 = register_module("cv1", seq);
                _add = shortcut && (c1 == c2);
            }

            torch::Tensor CIBImpl::forward(torch::Tensor x) {
                return _add ? (x + _cv1->forward(x)) : _cv1->forward(x);
            }

            // ============================================================================
            // C2fCIB Implementation
            // ============================================================================

            C2fCIBImpl::C2fCIBImpl(int64_t c1, int64_t c2, int64_t n, bool shortcut,
                bool lk, int64_t g, double e)
                : C2fImpl(c1, c2, n, shortcut, g, e) {
                _m = register_module("m", torch::nn::ModuleList());
                for (int64_t i = 0; i < n; ++i) {
                    _m->push_back(CIB(_c, _c, shortcut, 1.0, lk));
                }
            }

            // ============================================================================
            // SCDown Implementation
            // ============================================================================

            SCDownImpl::SCDownImpl(int64_t c1, int64_t c2, int64_t k, int64_t s) {
                _cv1 = register_module("cv1", Conv(c1, c2, 1, 1));
                _cv2 = register_module("cv2", Conv(c2, c2, k, s, std::nullopt, c2, 1));
            }

            torch::Tensor SCDownImpl::forward(torch::Tensor x) {
                return _cv2->forward(_cv1->forward(x));
            }

            // ============================================================================
            // C3k2 Implementation
            // ============================================================================

            C3k2Impl::C3k2Impl(int64_t c1, int64_t c2, int64_t n, bool c3k,
                double e, int64_t g, bool shortcut)
                : C2fImpl(c1, c2, n, shortcut, g, e) {
                auto m = torch::nn::ModuleList();
                for (int64_t i = 0; i < n; ++i) {
                    if (c3k) {
                        m->push_back(C3k(_c, _c, 2, shortcut, g));
                    }
                    else {
                        m->push_back(Bottleneck(_c, _c, shortcut, g, std::vector<int64_t>{3, 3}, 1.0));
                    }
                }
                _m = m;
            }

            // ============================================================================
            // C3f Implementation
            // ============================================================================

            C3fImpl::C3fImpl(int64_t c1, int64_t c2, int64_t n, bool shortcut,
                int64_t g, double e) {
                int64_t cMid = safeChannelMultiply(c2, e, "C3fImpl");
                _cv1 = register_module("cv1", Conv(c1, cMid, 1, 1));
                _cv2 = register_module("cv2", Conv(c1, cMid, 1, 1));
                _cv3 = register_module("cv3", Conv((2 + n) * cMid, c2, 1));

                _m = register_module("m", torch::nn::ModuleList());
                for (int64_t i = 0; i < n; ++i) {
                    _m->push_back(Bottleneck(cMid, cMid, shortcut, g, std::vector<int64_t>{3, 3}, 1.0));
                }
            }

            torch::Tensor C3fImpl::forward(torch::Tensor x) {
                std::vector<torch::Tensor> y;
                y.push_back(_cv2->forward(x));
                y.push_back(_cv1->forward(x));

                for (auto& m : *_m)
                {
                    auto bottleneck = m->as<IBlockImpl>();
                    y.push_back(bottleneck->forward(y.back()));
                }

                return _cv3->forward(torch::cat(y, 1));
            }

            // ============================================================================
            // C3TR Implementation
            // ============================================================================

            C3TRImpl::C3TRImpl(int64_t c1, int64_t c2, int64_t n, bool shortcut,
                int64_t g, double e)
                : C3Impl(c1, c2, n, shortcut, g, e) {
                torch::nn::Sequential m;
                m->push_back(TransformerBlock(_c, _c, 4, n));
                _m = m;
            }

            // ============================================================================
            // ResNetBlock Implementation
            // ============================================================================

            ResNetBlockImpl::ResNetBlockImpl(int64_t c1, int64_t c2, int64_t s, double e) {
                int64_t c3 = safeChannelMultiply(c2, e, "ResNetBlockImpl");

                _cv1 = register_module("cv1", Conv(c1, c2, 1, 1));
                _cv2 = register_module("cv2", Conv(c2, c2, 3, s, std::nullopt, 1, 1));
                _cv3 = register_module("cv3", Conv(c2, c3, 1, 1, std::nullopt, 1, 1));

                torch::nn::Sequential identity;
                if (s != 1 || c1 != c3) {
                    identity->push_back(Conv(c1, c3, 1, s, std::nullopt, 1, 1));
                }
                else {
                    identity->push_back(torch::nn::Identity());
                }
                _identity = register_module("identity", identity);
            }

            torch::Tensor ResNetBlockImpl::forward(torch::Tensor x) {
                return _cv3->forward(_cv2->forward(_cv1->forward(x))) + _identity->forward(x);
            }

            // ============================================================================
            // ResNetLayer Implementation
            // ============================================================================

            ResNetLayerImpl::ResNetLayerImpl(int64_t c1, int64_t c2, int64_t s, bool isFirst, int64_t n, double e)
            {

                torch::nn::Sequential blocks;
                if (isFirst)
                {
                    blocks->push_back(Conv(c1, c2, 7, 2, 3));
                    blocks->push_back(torch::nn::MaxPool2d(
                        torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
                }
                else
                {
                    blocks->push_back(ResNetBlock(c1, c2, s, e));
                    int64_t c3 = safeChannelMultiply(c2, e, "ResNetLayerImpl");
                    for (int64_t i = 1; i < n; ++i)
                    {
                        blocks->push_back(ResNetBlock(c3, c2, 1, e));
                    }
                }

                _blocks = register_module("blocks", blocks);
            }

            torch::Tensor ResNetLayerImpl::forward(torch::Tensor x) {
                return _blocks->forward(x);
            }

            // ============================================================================
            // C2PSA Implementation
            // ============================================================================

            C2PSAImpl::C2PSAImpl(int64_t c1, int64_t c2, int64_t n, double e) {
                _c = safeChannelMultiply(c2, e, "C2PSAImpl");
                _cv1 = register_module("cv1", Conv(c1, 2 * _c, 1, 1));
                _cv2 = register_module("cv2", Conv(2 * _c, c2, 1));

                _m = register_module("m", torch::nn::ModuleList());
                for (int64_t i = 0; i < n; ++i) {
                    _m->push_back(PSABlock(_c, BlockConstants::PSA_EXPANSION_RATIO,
                        BlockConstants::PSA_NUM_HEADS, true));
                }
            }

            torch::Tensor C2PSAImpl::forward(torch::Tensor x)
            {
                auto cv1Out = _cv1->forward(x);
                auto c = cv1Out.size(1) / 2;
                auto a = cv1Out.narrow(1, 0, c);
                auto b = cv1Out.narrow(1, c, c);

                for (auto& m : *_m)
                {
                    auto psaBlock = m->as<IBlockImpl>();
                    b = psaBlock->forward(b);
                }

                return _cv2->forward(torch::cat({ a, b }, 1));
            }

            // ============================================================================
            // C2fPSA Implementation
            // ============================================================================

            C2fPSAImpl::C2fPSAImpl(int64_t c1, int64_t c2, int64_t n, double e)
                : C2fImpl(c1, c2, n, false, 1, e)
            {
                auto m = torch::nn::ModuleList();
                for (int64_t i = 0; i < n; ++i) {
                    m->push_back(PSABlock(_c, 0.5));
                }
				_m = m;
            }

            // ============================================================================
            // ContrastiveHead Implementation
            // ============================================================================

            ContrastiveHeadImpl::ContrastiveHeadImpl() {
                _linear = register_module("linear", torch::nn::Linear(512, 128));
            }

            torch::Tensor ContrastiveHeadImpl::forward(torch::Tensor x) {
                return torch::nn::functional::normalize(_linear->forward(x), torch::nn::functional::NormalizeFuncOptions().dim(1));
            }

            // ============================================================================
            // BNContrastiveHead Implementation
            // ============================================================================

            BNContrastiveHeadImpl::BNContrastiveHeadImpl(torch::nn::BatchNorm1d norm) {
                _norm = register_module("norm", norm);
                _linear = register_module("linear", torch::nn::Linear(512, 128));
            }

            torch::Tensor BNContrastiveHeadImpl::forward(torch::Tensor x) {
                x = _norm->forward(x);
                return torch::nn::functional::normalize(_linear->forward(x), torch::nn::functional::NormalizeFuncOptions().dim(1));
            }

            // ============================================================================
            // DBlock Implementation
            // ============================================================================

            DBlockImpl::DBlockImpl(int64_t c1, int64_t c2, int64_t growthRate, int64_t bottleNeckSize, const std::string& act)
            {
                // Validate parameters
                if (c1 <= 0) {
                    throw std::invalid_argument("DBlockImpl: c1 must be positive, got: " + std::to_string(c1));
                }
                if (c2 <= 0) {
                    throw std::invalid_argument("DBlockImpl: c2 must be positive, got: " + std::to_string(c2));
                }
                if (growthRate <= 0) {
                    throw std::invalid_argument("DBlockImpl: growthRate must be positive, got: " + std::to_string(growthRate));
                }
                if (bottleNeckSize <= 0) {
                    throw std::invalid_argument("DBlockImpl: bn_size must be positive, got: " + std::to_string(bottleNeckSize));
                }

                if ((c2 - c1) % growthRate != 0)
                {
                    throw std::invalid_argument(
                        "DBlockImpl: (c2 - c1) must be divisible by growthRate. " +
                        std::string("Got c1=") + std::to_string(c1) +
                        ", c2=" + std::to_string(c2) +
                        ", growthRate=" + std::to_string(growthRate)
                    );
                }

                _numLayers = (c2 - c1) / growthRate;

                if (_numLayers <= 0) {
                    throw std::invalid_argument(
                        "DBlockImpl: num_layers must be positive. " +
                        std::string("Got c1=") + std::to_string(c1) +
                        ", c2=" + std::to_string(c2) +
                        ", resulting in num_layers=" + std::to_string(_numLayers)
                    );
                }

                // Create layers
                int64_t currentChannels = c1;
                for (int64_t i = 0; i < _numLayers; i++) 
                {
                    int64_t bottleneckChannels = bottleNeckSize * growthRate;
                    auto bottleneck = Conv(currentChannels, bottleneckChannels, 1, 1, std::nullopt, 1, 1, act);
                    _bottlenecks.push_back(register_module("bottleneck" + std::to_string(i), bottleneck));

                    auto conv = Conv(bottleneckChannels, growthRate, 3, 1, std::nullopt, 1, 1, act);
                    _convs.push_back(register_module("conv" + std::to_string(i), conv));

                    currentChannels += growthRate;
                }
            }

            torch::Tensor DBlockImpl::forward(torch::Tensor x) 
            {
                std::vector<torch::Tensor> features = { x };

                for (int64_t i = 0; i < _numLayers; i++) 
                {
                    torch::Tensor concatenated = torch::cat(features, 1);
                    torch::Tensor out = _bottlenecks[i]->forward(concatenated);
                    out = _convs[i]->forward(out);

                    features.push_back(out);
                }

                // Return concatenation of all features (input + all layer outputs)
                return torch::cat(features, 1);
            }

            // ============================================================================
            // CNXBlock Implementation
            // ============================================================================

            CNXBlockImpl::CNXBlockImpl(int64_t dim, double layerScaleInit)
                : _useLayerScale(layerScaleInit > 0)
            {
                // Validate input
                if (dim <= 0) {
                    throw std::invalid_argument("CNXBlockImpl: dim must be positive, got: " + std::to_string(dim));
                }

                // 7x7 Depthwise Conv
                _dwconv = register_module("dwconv",
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, dim, 7)
                        .padding(3)
                        .groups(dim)));

                // LayerNorm2d
                _norm = register_module("norm", LayerNorm2d(dim, 1e-6));

                // 1x1 Conv expansion (dim -> 4*dim)
                _pwconv1 = register_module("pwconv1",
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(dim, 4 * dim, 1)));

                // 1x1 Conv reduction (4*dim -> dim)
                _pwconv2 = register_module("pwconv2",
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(4 * dim, dim, 1)));

                // LayerScale parameter
                if (_useLayerScale) {
                    _gamma = register_parameter("gamma", torch::full({ dim, 1, 1 }, layerScaleInit));
                }
            }

            torch::Tensor CNXBlockImpl::forward(torch::Tensor x)
            {
                torch::Tensor shortcut = x;

                x = _dwconv->forward(x);
                x = _norm->forward(x);
                x = _pwconv1->forward(x);
                x = torch::gelu(x);
                x = _pwconv2->forward(x);

                if (_useLayerScale) 
                {
                    x = x * _gamma;
                }

                x = shortcut + x;

                return x;
            }

        } // namespace Modules
    } // namespace Model
} // namespace WheelDL
