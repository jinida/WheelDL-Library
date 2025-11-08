#include "Head.h"
#include "Utils.h"
#include <cmath>
#include <algorithm>

namespace WheelDL {
    namespace Model {
        namespace Modules {

            // ============================================================================
            // DetectImpl Implementation
            // ============================================================================

            DetectImpl::DetectImpl(int64_t nc, const std::vector<int64_t>& ch)
                : nc(nc), nl(static_cast<int64_t>(ch.size())), regMax(HeadConstants::DEFAULT_REG_MAX) {

                if (ch.empty()) {
                    throw std::invalid_argument("DetectImpl: Channel vector ch cannot be empty");
                }

                // Validate regMax
                if (regMax <= 0) {
                    throw std::invalid_argument("DetectImpl: regMax must be positive, got: " + std::to_string(regMax));
                }

                no = nc + regMax * 4;  // number of outputs per anchor
                stride = torch::zeros(nl);

                // Calculate channel sizes
                int64_t c2 = std::max({ HeadConstants::MIN_CHANNEL_SIZE, ch[0] / 4, regMax * 4 });
                int64_t c3 = std::max(ch[0], std::min(nc, HeadConstants::MAX_CLASS_CHANNELS));

                // Build cv2 (box regression layers)
                for (int64_t i = 0; i < nl; ++i) {
                    torch::nn::Sequential seq;
                    seq->push_back(Conv(ch[i], c2, 3));
                    seq->push_back(Conv(c2, c2, 3));
                    seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(c2, 4 * regMax, 1)));
                    _cv2->push_back(seq);
                }
                register_module("cv2", _cv2);

                // Build cv3 (classification layers)
                for (int64_t i = 0; i < nl; ++i) {
                    torch::nn::Sequential seq;
                    torch::nn::Sequential dwConvSeq;
                    dwConvSeq->push_back(DWConv(ch[i], ch[i], 3));
                    dwConvSeq->push_back(Conv(ch[i], c3, 1));
                    seq->push_back(dwConvSeq);

                    torch::nn::Sequential dwConvSeq2;
                    dwConvSeq2->push_back(DWConv(c3, c3, 3));
                    dwConvSeq2->push_back(Conv(c3, c3, 1));
                    seq->push_back(dwConvSeq2);

                    seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(c3, nc, 1)));
                    _cv3->push_back(seq);
                }
                register_module("cv3", _cv3);

                // Build DFL layer
                if (regMax > 1) {
                    auto dfl = DFL(regMax);
                    _dfl = dfl.ptr();
                    register_module("dfl", dfl);
                }
                else {
                    auto identity = torch::nn::Identity();
                    _dfl = identity.ptr();
                    register_module("dfl", identity);
                }

                anchors = torch::empty(0);
                strides = torch::empty(0);
            }

            torch::autograd::variable_list DetectImpl::forward(std::vector<torch::Tensor> x) {
                // Standard detection mode
                for (int64_t i = 0; i < nl; ++i) {
                    // Validate module list bounds
                    if (i >= _cv2->size() || i >= _cv3->size()) {
                        throw std::out_of_range("Module index " + std::to_string(i) +
                            " out of range (cv2 size: " + std::to_string(_cv2->size()) +
                            ", cv3 size: " + std::to_string(_cv3->size()) + ")");
                    }
                    // Validate input tensor bounds
                    if (i >= static_cast<int64_t>(x.size())) {
                        throw std::out_of_range("Input tensor index " + std::to_string(i) +
                            " out of range (x size: " + std::to_string(x.size()) + ")");
                    }

                    auto cv2Out = _cv2->ptr(i)->as<torch::nn::Sequential>()->forward(x[i]);
                    auto cv3Out = _cv3->ptr(i)->as<torch::nn::Sequential>()->forward(x[i]);
                    x[i] = torch::cat({ cv2Out, cv3Out }, 1);
                }

                if (is_training()) {
                    return x;
                }

                auto y = _inference(x);
                if (export_) {
                    return { y };
                }

                // Return both inference output and raw outputs
                std::vector<torch::Tensor> result = { y };
                result.insert(result.end(), x.begin(), x.end());
                return result;
            }

            torch::Tensor DetectImpl::_inference(const std::vector<torch::Tensor>& x) {
                auto shape = x[0].sizes();  // BCHW

                std::vector<torch::Tensor> xCatList;
                xCatList.reserve(x.size());
                for (auto& xi : x) {
                    xCatList.push_back(xi.view({ shape[0], no, -1 }));
                }
                auto xCat = torch::cat(xCatList, 2);

                // Check if input shapes have changed
                std::vector<std::vector<int64_t>> currentShapes;
                currentShapes.reserve(x.size());
                for (const auto& xi : x) {
                    currentShapes.push_back(xi.sizes().vec());
                }

                // Initialize or regenerate anchors and strides if input shape changed
                bool needsRegeneration = !this->shape.defined() || currentShapes != _lastInputShapes;
                if (needsRegeneration) {
                    auto anchorStrides = makeAnchors(x, stride, 0.5);

                    // Validate makeAnchors results before using them
                    auto& anchorTensor = std::get<0>(anchorStrides);
                    auto& strideTensor = std::get<1>(anchorStrides);

                    if (anchorTensor.numel() == 0) {
                        throw std::runtime_error("DetectImpl::forward - makeAnchors returned empty anchor tensor");
                    }
                    if (strideTensor.numel() == 0) {
                        throw std::runtime_error("DetectImpl::forward - makeAnchors returned empty stride tensor");
                    }

                    anchors = anchorTensor.transpose(0, 1);
                    strides = strideTensor.transpose(0, 1);
                    this->shape = torch::tensor({ shape[0], shape[1], shape[2], shape[3] });
                    _lastInputShapes = currentShapes;
                }

                auto splits = xCat.split({ regMax * 4, nc }, 1);
                auto box = splits[0];
                auto cls = splits[1];

                torch::Tensor dbox;
                if (regMax > 1) {
                    auto dflBox = _dfl.forward<torch::Tensor>(box);
                    dbox = decodeBboxes(dflBox, anchors.unsqueeze(0)) * strides;
                }
                else {
                    dbox = decodeBboxes(box, anchors.unsqueeze(0)) * strides;
                }

                return torch::cat({ dbox, cls.sigmoid() }, 1);
            }

            void DetectImpl::biasInit() {
                for (int64_t i = 0; i < nl; ++i) {
                    // Validate stride tensor bounds
                    if (i >= stride.size(0)) {
                        throw std::out_of_range("DetectImpl::biasInit - Stride index " + std::to_string(i) +
                            " out of bounds, size: " + std::to_string(stride.size(0)));
                    }

                    auto cv2 = _cv2->ptr(i)->as<torch::nn::Sequential>();
                    auto cv3 = _cv3->ptr(i)->as<torch::nn::Sequential>();
                    auto s = stride[i].item<double>();

                    // Get the last conv layer
                    auto cv2Last = (*cv2)[cv2->size() - 1]->as<torch::nn::Conv2d>();
                    auto cv3Last = (*cv3)[cv3->size() - 1]->as<torch::nn::Conv2d>();

                    // Initialize box bias
                    cv2Last->bias.data().fill_(1.0);

                    // Initialize class bias
                    // Use configurable input size instead of hardcoded 640.0
                    auto clsBias = std::log(5.0 / nc / std::pow(inputSize / s, 2));
                    cv3Last->bias.data().slice(0, 0, nc).fill_(clsBias);
                }
            }

            torch::Tensor DetectImpl::decodeBboxes(const torch::Tensor& bboxes, const torch::Tensor& anchors, bool xywh) {
                return dist2bbox(bboxes, anchors, xywh, 1);
            }

            // ============================================================================
            // OBBImpl Implementation
            // ============================================================================

            OBBImpl::OBBImpl(int64_t nc, int64_t ne, const std::vector<int64_t>& ch)
                : DetectImpl(nc, ch), ne(ne) {

                int64_t c4 = std::max(ch[0] / 4, ne);
                for (int64_t i = 0; i < nl; ++i) {
                    torch::nn::Sequential seq;
                    seq->push_back(Conv(ch[i], c4, 3));
                    seq->push_back(Conv(c4, c4, 3));
                    seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(c4, ne, 1)));
                    _cv4->push_back(seq);
                }
                register_module("cv4", _cv4);
            }

            torch::autograd::variable_list OBBImpl::forward(std::vector<torch::Tensor> x) {
                auto bs = x[0].size(0);

                std::vector<torch::Tensor> angleList;
                angleList.reserve(nl);
                for (int64_t i = 0; i < nl; ++i) {
                    auto cv4Out = _cv4->ptr(i)->as<torch::nn::Sequential>()->forward(x[i]);
                    angleList.push_back(cv4Out.view({ bs, ne, -1 }));
                }
                auto angleTensor = torch::cat(angleList, 2);

                // Transform angle: (sigmoid - 0.25) * pi to get range [-pi/4, 3pi/4]
                angle = (angleTensor.sigmoid() - 0.25) * M_PI;

                auto detOut = DetectImpl::forward(x);

                if (is_training()) {
                    std::vector<torch::Tensor> result = detOut;
                    result.push_back(angle);
                    return result;
                }

                if (export_)
                {
                    return { torch::cat({detOut[0], angle}, 1) };
                }

                return { torch::cat({detOut[0], angle}, 1), detOut[1], angle };
            }

            torch::Tensor OBBImpl::decodeBboxes(const torch::Tensor& bboxes, const torch::Tensor& anchors) {
                return dist2rbox(bboxes, angle, anchors, 1);
            }

            // ============================================================================
            // ClassifyImpl Implementation
            // ============================================================================

            ClassifyImpl::ClassifyImpl(int64_t c1, int64_t c2, int64_t k, int64_t s,
                std::optional<int64_t> p, int64_t g) {
                int64_t c_ = HeadConstants::EFFICIENTNET_B0_CHANNELS;

                _conv = Conv(c1, c_, k, s, p, g);
                register_module("conv", _conv);

                _pool = torch::nn::AdaptiveAvgPool2d(1);
                register_module("pool", _pool);

                _drop = torch::nn::Dropout(torch::nn::DropoutOptions().p(0.0).inplace(true));
                register_module("drop", _drop);

                _linear = torch::nn::Linear(c_, c2);
                register_module("linear", _linear);
            }

            torch::Tensor ClassifyImpl::forward(torch::Tensor x) {
                x = _conv->forward(x);
                x = _pool->forward(x);
                x = x.flatten(1);
                x = _drop->forward(x);
                x = _linear->forward(x);

                if (is_training()) {
                    return x;
                }

                auto y = x.softmax(1);
                return export_ ? y : x;
            }

            torch::Tensor ClassifyImpl::forward(const std::vector<torch::Tensor>& x) {
                auto xCat = torch::cat(x, 1);
                return forward(xCat);
            }

        } // namespace Modules
    } // namespace Model
} // namespace WheelDL
