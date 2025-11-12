#include "pch.h"
#include "Transformer.h"
#include "Utils.h"
#include <cmath>

namespace WheelDL {
    namespace Model {
        namespace Modules {

            // ============================================================================
            // LayerNorm2d Implementation
            // ============================================================================

            LayerNorm2dImpl::LayerNorm2dImpl(int64_t numChannels, double eps)
                : _eps(eps) {
                _weight = register_parameter("weight", torch::ones({ numChannels }));
                _bias = register_parameter("bias", torch::zeros({ numChannels }));
            }

            torch::Tensor LayerNorm2dImpl::forward(torch::Tensor x) 
            {
                auto u = x.mean(1, /*keepdim=*/true);
                auto s = (x - u).pow(2).mean(1, /*keepdim=*/true);

                // Add epsilon before sqrt for numerical stability
                // This is the standard approach in layer normalization
                auto denominator = torch::sqrt(s + _eps);

                x = (x - u) / denominator;
                return _weight.unsqueeze(-1).unsqueeze(-1) * x + _bias.unsqueeze(-1).unsqueeze(-1);
            }

            // ============================================================================
            // MLPBlock Implementation
            // ============================================================================

            MLPBlockImpl::MLPBlockImpl(int64_t embeddingDim, int64_t mlpDim) {
                _lin1 = register_module("lin1", torch::nn::Linear(embeddingDim, mlpDim));
                _lin2 = register_module("lin2", torch::nn::Linear(mlpDim, embeddingDim));
                _act = register_module("act", torch::nn::GELU());
            }

            torch::Tensor MLPBlockImpl::forward(torch::Tensor x) {
                return _lin2->forward(_act->forward(_lin1->forward(x)));
            }

            // ============================================================================
            // MLP Implementation
            // ============================================================================

            MLPImpl::MLPImpl(int64_t inputDim, int64_t hiddenDim, int64_t outputDim,
                int64_t numLayers, bool sigmoid)
                : _numLayers(numLayers), _sigmoid(sigmoid) {

                _layers = register_module("layers", torch::nn::ModuleList());
                _act = register_module("act", torch::nn::ReLU());

                std::vector<int64_t> h(numLayers - 1, hiddenDim);
                std::vector<int64_t> inputDims = { inputDim };
                inputDims.insert(inputDims.end(), h.begin(), h.end());

                std::vector<int64_t> outputDims = h;
                outputDims.push_back(outputDim);

                if (inputDims.size() != outputDims.size()) {
                    throw std::invalid_argument("MLPImpl: inputDims and outputDims must have the same size: " +
                                               std::to_string(inputDims.size()) + " vs " +
                                               std::to_string(outputDims.size()));
                }

                for (size_t i = 0; i < inputDims.size(); ++i) {
                    _layers->push_back(torch::nn::Linear(inputDims[i], outputDims[i]));
                }
            }

            torch::Tensor MLPImpl::forward(torch::Tensor x) {
                for (int64_t i = 0; i < _numLayers; ++i) {
                    auto layer = _layers->ptr(i)->as<torch::nn::Linear>();
                    if (i < _numLayers - 1) {
                        x = _act->forward(layer->forward(x));
                    }
                    else {
                        x = layer->forward(x);
                    }
                }
                return _sigmoid ? x.sigmoid() : x;
            }

            // ============================================================================
            // TransformerLayer Implementation
            // ============================================================================

            TransformerLayerImpl::TransformerLayerImpl(int64_t c, int64_t numHeads) {
                _q = register_module("q", torch::nn::Linear(torch::nn::LinearOptions(c, c).bias(false)));
                _k = register_module("k", torch::nn::Linear(torch::nn::LinearOptions(c, c).bias(false)));
                _v = register_module("v", torch::nn::Linear(torch::nn::LinearOptions(c, c).bias(false)));
                _ma = register_module("ma", torch::nn::MultiheadAttention(
                    torch::nn::MultiheadAttentionOptions(c, numHeads)));
                _fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(c, c).bias(false)));
                _fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(c, c).bias(false)));
            }

            torch::Tensor TransformerLayerImpl::forward(torch::Tensor x) {
                auto q = _q->forward(x);
                auto k = _k->forward(x);
                auto v = _v->forward(x);
                x = std::get<0>(_ma->forward(q, k, v)) + x;
                return _fc2->forward(_fc1->forward(x)) + x;
            }

            // ============================================================================
            // TransformerBlock Implementation
            // ============================================================================

            TransformerBlockImpl::TransformerBlockImpl(int64_t c1, int64_t c2,
                int64_t numHeads, int64_t numLayers)
                : _c2(c2) {

                if (c1 != c2) {
                    _conv = register_module("conv", Conv(c1, c2));
                }

                _linear = register_module("linear", torch::nn::Linear(c2, c2));

                torch::nn::Sequential tr;
                for (int64_t i = 0; i < numLayers; ++i) {
                    tr->push_back(TransformerLayer(c2, numHeads));
                }
                _tr = register_module("tr", tr);
            }

            torch::Tensor TransformerBlockImpl::forward(torch::Tensor x) {
                if (!_conv.is_empty()) {
                    x = _conv->forward(x);
                }

                auto b = x.size(0);
                auto h = x.size(2);  // Height
                auto w = x.size(3);  // Width

                auto p = x.flatten(2).permute({ 2, 0, 1 });
                auto output = _tr->forward(p + _linear->forward(p));
                return output.permute({ 1, 2, 0 }).reshape({ b, _c2, h, w });
            }

            // ============================================================================
            // TransformerEncoderLayer Implementation
            // ============================================================================

            TransformerEncoderLayerImpl::TransformerEncoderLayerImpl(int64_t c1, int64_t cm,
                int64_t numHeads, double dropout, bool normalizeBefore)
                : _normalizeBefore(normalizeBefore) {

                _ma = register_module("ma", torch::nn::MultiheadAttention(
                    torch::nn::MultiheadAttentionOptions(c1, numHeads)
                    .dropout(dropout)));

                _fc1 = register_module("fc1", torch::nn::Linear(c1, cm));
                _fc2 = register_module("fc2", torch::nn::Linear(cm, c1));
                _norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ c1 })));
                _norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ c1 })));
                _dropout = register_module("dropout", torch::nn::Dropout(dropout));
                _dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
                _dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
                _act = register_module("act", torch::nn::GELU());
            }

            torch::Tensor TransformerEncoderLayerImpl::withPosEmbed(
                const torch::Tensor& tensor, const torch::Tensor& pos) {
                if (pos.defined()) {
                    return tensor + pos;
                }
                else {
                    return tensor;
                }
            }

            torch::Tensor TransformerEncoderLayerImpl::applyAttention(
                const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                const torch::Tensor& srcMask, const torch::Tensor& srcKeyPaddingMask) {

                if (srcMask.defined() && srcKeyPaddingMask.defined()) {
                    return std::get<0>(_ma->forward(q, k, v, srcKeyPaddingMask, true, srcMask));
                }
                else if (srcMask.defined()) {
                    return std::get<0>(_ma->forward(q, k, v, torch::Tensor(), true, srcMask));
                }
                else if (srcKeyPaddingMask.defined()) {
                    return std::get<0>(_ma->forward(q, k, v, srcKeyPaddingMask));
                }
                else {
                    return std::get<0>(_ma->forward(q, k, v));
                }
            }

            torch::Tensor TransformerEncoderLayerImpl::forwardPost(
                torch::Tensor src, torch::Tensor srcMask,
                torch::Tensor srcKeyPaddingMask, torch::Tensor pos) {

                auto q = withPosEmbed(src, pos);
                auto k = q;
                auto attnOutput = applyAttention(q, k, src, srcMask, srcKeyPaddingMask);

                src = src + _dropout1->forward(attnOutput);
                src = _norm1->forward(src);
                auto src2 = _fc2->forward(_dropout->forward(_act->forward(_fc1->forward(src))));
                src = src + _dropout2->forward(src2);
                return _norm2->forward(src);
            }

            torch::Tensor TransformerEncoderLayerImpl::forwardPre(
                torch::Tensor src, torch::Tensor srcMask,
                torch::Tensor srcKeyPaddingMask, torch::Tensor pos) {

                auto src2 = _norm1->forward(src);
                auto q = withPosEmbed(src2, pos);
                auto k = q;
                auto attnOutput = applyAttention(q, k, src2, srcMask, srcKeyPaddingMask);

                src = src + _dropout1->forward(attnOutput);
                src2 = _norm2->forward(src);
                src2 = _fc2->forward(_dropout->forward(_act->forward(_fc1->forward(src2))));
                return src + _dropout2->forward(src2);
            }

            torch::Tensor TransformerEncoderLayerImpl::forward(
                torch::Tensor src, torch::Tensor srcMask,
                torch::Tensor srcKeyPaddingMask, torch::Tensor pos) {

                if (_normalizeBefore) {
                    return forwardPre(src, srcMask, srcKeyPaddingMask, pos);
                }
                return forwardPost(src, srcMask, srcKeyPaddingMask, pos);
            }

            // ============================================================================
            // AIFI Implementation
            // ============================================================================

            AIFIImpl::AIFIImpl(int64_t c1, int64_t cm, int64_t numHeads,
                double dropout, bool normalizeBefore)
                : TransformerEncoderLayerImpl(c1, cm, numHeads, dropout, normalizeBefore) {
            }

            torch::Tensor AIFIImpl::forward(torch::Tensor x) {
                auto c = x.size(1);
                auto h = x.size(2);
                auto w = x.size(3);

                auto posEmbed = build2dSincosPositionEmbedding(w, h, c);
                posEmbed = posEmbed.to(x.device()).to(x.dtype());

                // Flatten [B, C, H, W] to [B, HxW, C]
                x = x.flatten(2).permute({ 0, 2, 1 });
                x = TransformerEncoderLayerImpl::forward(x, torch::Tensor(), torch::Tensor(), posEmbed);
                return x.permute({ 0, 2, 1 }).view({ -1, c, h, w }).contiguous();
            }

            torch::Tensor AIFIImpl::build2dSincosPositionEmbedding(
                int64_t w, int64_t h, int64_t embedDim, double temperature) {

                // Validate input parameters
                if (embedDim <= 0) {
                    throw std::invalid_argument("Embed dimension must be positive, got: " +
                                                std::to_string(embedDim));
                }
                if (embedDim % 4 != 0) {
                    throw std::invalid_argument("Embed dimension must be divisible by 4, got: " +
                                                std::to_string(embedDim));
                }
                if (w <= 0 || h <= 0) {
                    throw std::invalid_argument("Width and height must be positive, got w=" +
                                                std::to_string(w) + ", h=" + std::to_string(h));
                }

                auto gridW = torch::arange(w, torch::kFloat32);
                auto gridH = torch::arange(h, torch::kFloat32);
                auto meshgrids = torch::meshgrid({ gridW, gridH }, "ij");
                gridW = meshgrids[0];
                gridH = meshgrids[1];

                int64_t posDim = embedDim / 4;
                auto omega = torch::arange(posDim, torch::kFloat32) / posDim;
                omega = 1.0 / torch::pow(temperature, omega);

                auto outW = torch::matmul(gridW.flatten().unsqueeze(-1), omega.unsqueeze(0));
                auto outH = torch::matmul(gridH.flatten().unsqueeze(-1), omega.unsqueeze(0));

                return torch::cat({ torch::sin(outW), torch::cos(outW),
                                   torch::sin(outH), torch::cos(outH) }, 1).unsqueeze(0);
            }

            // ============================================================================
            // MSDeformAttn Implementation
            // ============================================================================

            MSDeformAttnImpl::MSDeformAttnImpl(int64_t dModel, int64_t nLevels,
                int64_t nHeads, int64_t nPoints)
                : _im2colStep(64), _dModel(dModel), _nLevels(nLevels),
                _nHeads(nHeads), _nPoints(nPoints) {

                if (dModel % nHeads != 0) {
                    throw std::invalid_argument("d_model must be divisible by n_heads");
                }

                _samplingOffsets = register_module("sampling_offsets",
                    torch::nn::Linear(dModel, nHeads * nLevels * nPoints * 2));
                _attentionWeights = register_module("attention_weights",
                    torch::nn::Linear(dModel, nHeads * nLevels * nPoints));
                _valueProj = register_module("value_proj", torch::nn::Linear(dModel, dModel));
                _outputProj = register_module("output_proj", torch::nn::Linear(dModel, dModel));

                resetParameters();
            }

            void MSDeformAttnImpl::resetParameters() {
                torch::nn::init::constant_(_samplingOffsets->weight, 0.0);

                auto thetas = torch::arange(_nHeads, torch::kFloat32) * (2.0 * M_PI / _nHeads);
                auto gridInit = torch::stack({ thetas.cos(), thetas.sin() }, -1);
                gridInit = (gridInit / gridInit.abs().amax(-1, /*keepdim=*/true).unsqueeze(-1))
                    .view({ _nHeads, 1, 1, 2 })
                    .repeat({ 1, _nLevels, _nPoints, 1 });

                for (int64_t i = 0; i < _nPoints; ++i) {
                    gridInit.index_put_({ torch::indexing::Slice(), torch::indexing::Slice(), i },
                        gridInit.index({ torch::indexing::Slice(), torch::indexing::Slice(), i }) * (i + 1));
                }

                {
                    torch::NoGradGuard noGrad;
                    _samplingOffsets->bias.copy_(gridInit.view({ -1 }));
                }

                torch::nn::init::constant_(_attentionWeights->weight, 0.0);
                torch::nn::init::constant_(_attentionWeights->bias, 0.0);
                torch::nn::init::xavier_uniform_(_valueProj->weight);
                torch::nn::init::constant_(_valueProj->bias, 0.0);
                torch::nn::init::xavier_uniform_(_outputProj->weight);
                torch::nn::init::constant_(_outputProj->bias, 0.0);
            }

            torch::Tensor MSDeformAttnImpl::forward(
                const torch::Tensor& query, const torch::Tensor& referBbox,
                const torch::Tensor& value, const torch::Tensor& valueShapes,
                const torch::Tensor& valueMask) {

                auto bs = query.size(0);
                auto lenQ = query.size(1);
                auto lenV = value.size(1);

                auto valueProj = _valueProj->forward(value);
                if (valueMask.defined()) {
                    valueProj = valueProj.masked_fill(valueMask.unsqueeze(-1), 0.0);
                }

                valueProj = valueProj.view({ bs, lenV, _nHeads, _dModel / _nHeads });

                auto samplingOffsets = _samplingOffsets->forward(query)
                    .view({ bs, lenQ, _nHeads, _nLevels, _nPoints, 2 });

                auto attentionWeights = _attentionWeights->forward(query)
                    .view({ bs, lenQ, _nHeads, _nLevels * _nPoints });
                attentionWeights = torch::softmax(attentionWeights, -1)
                    .view({ bs, lenQ, _nHeads, _nLevels, _nPoints });

                auto numPoints = referBbox.size(-1);
                torch::Tensor samplingLocations;

                if (numPoints == 2) {
                    auto offsetNormalizer = valueShapes.to(query.dtype()).to(query.device()).flip({ -1 });
                    auto add = samplingOffsets / offsetNormalizer.index({ torch::indexing::None, torch::indexing::None,
                        torch::indexing::None, torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice() });
                    samplingLocations = referBbox.index({ torch::indexing::Slice(), torch::indexing::Slice(),
                        torch::indexing::None, torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice() }) + add;
                }
                else if (numPoints == 4) {
                    auto add = samplingOffsets / _nPoints *
                        referBbox.index({ torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::None,
                            torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(2, torch::indexing::None) }) * 0.5;
                    samplingLocations = referBbox.index({ torch::indexing::Slice(), torch::indexing::Slice(),
                        torch::indexing::None, torch::indexing::Slice(), torch::indexing::None, torch::indexing::Slice(torch::indexing::None, 2) }) + add;
                }
                else {
                    throw std::invalid_argument("Last dim of reference_points must be 2 or 4");
                }

                auto output = multiScaleDeformableAttnPytorch(valueProj, valueShapes, samplingLocations, attentionWeights);
                return _outputProj->forward(output);
            }

            // ============================================================================
            // DeformableTransformerDecoderLayer Implementation
            // ============================================================================

            DeformableTransformerDecoderLayerImpl::DeformableTransformerDecoderLayerImpl(
                int64_t dModel, int64_t nHeads, int64_t dFfn, double dropout,
                int64_t nLevels, int64_t nPoints) {

                _selfAttn = register_module("self_attn",
                    torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(dModel, nHeads).dropout(dropout)));
                _dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
                _norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dModel })));

                _crossAttn = register_module("cross_attn", MSDeformAttn(dModel, nLevels, nHeads, nPoints));
                _dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
                _norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dModel })));

                _linear1 = register_module("linear1", torch::nn::Linear(dModel, dFfn));
                _act = register_module("act", torch::nn::ReLU());
                _dropout3 = register_module("dropout3", torch::nn::Dropout(dropout));
                _linear2 = register_module("linear2", torch::nn::Linear(dFfn, dModel));
                _dropout4 = register_module("dropout4", torch::nn::Dropout(dropout));
                _norm3 = register_module("norm3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ dModel })));
            }

            torch::Tensor DeformableTransformerDecoderLayerImpl::withPosEmbed(
                const torch::Tensor& tensor, const torch::Tensor& pos) {
                if (pos.defined()) {
                    return tensor + pos;
                }
                else {
                    return tensor;
                }
            }

            torch::Tensor DeformableTransformerDecoderLayerImpl::forwardFfn(torch::Tensor tgt) {
                auto tgt2 = _linear2->forward(_dropout3->forward(_act->forward(_linear1->forward(tgt))));
                tgt = tgt + _dropout4->forward(tgt2);
                return _norm3->forward(tgt);
            }

            torch::Tensor DeformableTransformerDecoderLayerImpl::forward(
                const torch::Tensor& embed, const torch::Tensor& referBbox,
                const torch::Tensor& feats, const torch::Tensor& shapes,
                const torch::Tensor& paddingMask, const torch::Tensor& attnMask,
                const torch::Tensor& queryPos) {

                auto q = withPosEmbed(embed, queryPos);
                auto k = q;

                torch::Tensor tgt;
                if (attnMask.defined()) {
                    tgt = std::get<0>(_selfAttn->forward(
                        q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attnMask)).transpose(0, 1);
                }
                else {
                    tgt = std::get<0>(_selfAttn->forward(
                        q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1))).transpose(0, 1);
                }

                auto embedOut = embed + _dropout1->forward(tgt);
                embedOut = _norm1->forward(embedOut);

                tgt = _crossAttn->forward(withPosEmbed(embedOut, queryPos),
                    referBbox.unsqueeze(2), feats, shapes, paddingMask);
                embedOut = embedOut + _dropout2->forward(tgt);
                embedOut = _norm2->forward(embedOut);

                return forwardFfn(embedOut);
            }

            // ============================================================================
            // DeformableTransformerDecoder Implementation
            // ============================================================================

            DeformableTransformerDecoderImpl::DeformableTransformerDecoderImpl(
                int64_t hiddenDim, DeformableTransformerDecoderLayer decoderLayer,
                int64_t numLayers, int64_t evalIdx)
                : _hiddenDim(hiddenDim), _numLayers(numLayers),
                _evalIdx(evalIdx >= 0 ? evalIdx : numLayers + evalIdx),
                _dModel(hiddenDim), _nHeads(8), _dFfn(1024), _dropout(0.0), _nLevels(4), _nPoints(4) {

                _layers = register_module("layers", torch::nn::ModuleList());
                for (int64_t i = 0; i < numLayers; ++i) {
                    auto newLayer = DeformableTransformerDecoderLayer(_dModel, _nHeads, _dFfn, _dropout, _nLevels, _nPoints);
                    _layers->push_back(newLayer);
                }
            }

            std::tuple<torch::Tensor, torch::Tensor> DeformableTransformerDecoderImpl::forward(
                torch::Tensor embed, torch::Tensor referBbox, torch::Tensor feats,
                const torch::Tensor& shapes, torch::nn::ModuleList bboxHead,
                torch::nn::ModuleList scoreHead, torch::nn::Sequential posMlp,
                const torch::Tensor& attnMask, const torch::Tensor& paddingMask) {

                auto output = embed;
                std::vector<torch::Tensor> decBboxes;
                std::vector<torch::Tensor> decCls;
                torch::Tensor lastRefinedBbox;
                referBbox = referBbox.sigmoid();

                for (int64_t i = 0; i < _numLayers; ++i) {
                    auto layer = _layers->ptr(i)->as<DeformableTransformerDecoderLayer>();
                    output = layer->forward(output, referBbox, feats, shapes, paddingMask, attnMask,
                        posMlp->forward(referBbox));

                    auto bboxModule = bboxHead->ptr(i)->as<torch::nn::Linear>();
                    auto bbox = bboxModule->forward(output);
                    auto refinedBbox = torch::sigmoid(bbox + inverseSigmoid(referBbox));

                    if (is_training()) {
                        auto scoreModule = scoreHead->ptr(i)->as<torch::nn::Linear>();
                        decCls.push_back(scoreModule->forward(output));

                        if (i == 0) {
                            decBboxes.push_back(refinedBbox);
                        }
                        else {
                            decBboxes.push_back(torch::sigmoid(bbox + inverseSigmoid(lastRefinedBbox)));
                        }
                    }
                    else if (i == _evalIdx) {
                        auto scoreModule = scoreHead->ptr(i)->as<torch::nn::Linear>();
                        decCls.push_back(scoreModule->forward(output));
                        decBboxes.push_back(refinedBbox);
                        break;
                    }

                    lastRefinedBbox = refinedBbox;
                    referBbox = is_training() ? refinedBbox.detach() : refinedBbox;
                }

                return std::make_tuple(torch::stack(decBboxes), torch::stack(decCls));
            }

        } // namespace Modules
    } // namespace Model
} // namespace WheelDL
