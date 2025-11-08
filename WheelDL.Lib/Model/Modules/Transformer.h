#pragma once

#include <torch/torch.h>
#include "Conv.h"

namespace WheelDL {
namespace Model {
namespace Modules {

/**
 * @brief 2D Layer Normalization module
 *
 * Normalizes across the channel dimension while preserving spatial dimensions
 */
class LayerNorm2dImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new LayerNorm2d module
     *
     * @param numChannels Number of channels in the input
     * @param eps Small constant for numerical stability (default: 1e-6)
     */
    explicit LayerNorm2dImpl(int64_t numChannels, double eps = 1e-6);

    /**
     * @brief Forward pass for 2D layer normalization
     *
     * @param x Input tensor [B, C, H, W]
     * @return torch::Tensor Normalized output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::Tensor _weight;
    torch::Tensor _bias;
    double _eps;
};

TORCH_MODULE(LayerNorm2d);

/**
 * @brief Multi-layer perceptron block
 */
class MLPBlockImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new MLPBlock module
     *
     * @param embeddingDim Input and output dimension
     * @param mlpDim Hidden dimension
     */
    MLPBlockImpl(int64_t embeddingDim, int64_t mlpDim);

    /**
     * @brief Forward pass through MLPBlock
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear _lin1;
    torch::nn::Linear _lin2;
    torch::nn::GELU _act;
};

TORCH_MODULE(MLPBlock);

/**
 * @brief Simple multi-layer perceptron (FFN)
 */
class MLPImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new MLP module
     *
     * @param inputDim Input dimension
     * @param hiddenDim Hidden dimension
     * @param outputDim Output dimension
     * @param numLayers Number of layers
     * @param sigmoid Whether to apply sigmoid to output (default: false)
     */
    MLPImpl(int64_t inputDim, int64_t hiddenDim, int64_t outputDim,
            int64_t numLayers, bool sigmoid = false);

    /**
     * @brief Forward pass through MLP
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    int64_t _numLayers;
    torch::nn::ModuleList _layers;
    bool _sigmoid;
    torch::nn::ReLU _act;
};

TORCH_MODULE(MLP);

/**
 * @brief Transformer layer without LayerNorm
 */
class TransformerLayerImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new TransformerLayer module
     *
     * @param c Input and output channel dimension
     * @param numHeads Number of attention heads
     */
    TransformerLayerImpl(int64_t c, int64_t numHeads);

    /**
     * @brief Forward pass through TransformerLayer
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear _q;
    torch::nn::Linear _k;
    torch::nn::Linear _v;
    torch::nn::MultiheadAttention _ma;
    torch::nn::Linear _fc1;
    torch::nn::Linear _fc2;
};

TORCH_MODULE(TransformerLayer);

/**
 * @brief Vision Transformer block
 */
class TransformerBlockImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new TransformerBlock module
     *
     * @param c1 Input channel dimension
     * @param c2 Output channel dimension
     * @param numHeads Number of attention heads
     * @param numLayers Number of transformer layers
     */
    TransformerBlockImpl(int64_t c1, int64_t c2, int64_t numHeads, int64_t numLayers);

    /**
     * @brief Forward pass through TransformerBlock
     *
     * @param x Input tensor [b, c1, w, h]
     * @return torch::Tensor Output tensor [b, c2, w, h]
     */
    torch::Tensor forward(torch::Tensor x);

private:
    Conv _conv;
    torch::nn::Linear _linear;
    torch::nn::Sequential _tr;
    int64_t _c2;
};

TORCH_MODULE(TransformerBlock);

/**
 * @brief Transformer encoder layer with multi-head attention and FFN
 */
class TransformerEncoderLayerImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new TransformerEncoderLayer module
     *
     * @param c1 Input dimension
     * @param cm Hidden dimension in FFN (default: 2048)
     * @param numHeads Number of attention heads (default: 8)
     * @param dropout Dropout probability (default: 0.0)
     * @param normalizeBefore Whether to apply normalization before attention (default: false)
     */
    TransformerEncoderLayerImpl(int64_t c1, int64_t cm = 2048, int64_t numHeads = 8,
                                double dropout = 0.0, bool normalizeBefore = false);

    /**
     * @brief Forward pass through TransformerEncoderLayer
     *
     * @param src Input tensor
     * @param srcMask Mask for the src sequence (optional)
     * @param srcKeyPaddingMask Mask for the src keys per batch (optional)
     * @param pos Positional encoding (optional)
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor src,
                          torch::Tensor srcMask = {},
                          torch::Tensor srcKeyPaddingMask = {},
                          torch::Tensor pos = {});

    /**
     * @brief Add position embeddings to tensor if provided
     */
    static torch::Tensor withPosEmbed(const torch::Tensor& tensor, const torch::Tensor& pos);

private:
    torch::Tensor forwardPost(torch::Tensor src, torch::Tensor srcMask,
                              torch::Tensor srcKeyPaddingMask, torch::Tensor pos);

    torch::Tensor forwardPre(torch::Tensor src, torch::Tensor srcMask,
                             torch::Tensor srcKeyPaddingMask, torch::Tensor pos);

    torch::Tensor applyAttention(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                  const torch::Tensor& srcMask, const torch::Tensor& srcKeyPaddingMask);

    torch::nn::MultiheadAttention _ma;
    torch::nn::Linear _fc1;
    torch::nn::Linear _fc2;
    torch::nn::LayerNorm _norm1;
    torch::nn::LayerNorm _norm2;
    torch::nn::Dropout _dropout;
    torch::nn::Dropout _dropout1;
    torch::nn::Dropout _dropout2;
    torch::nn::GELU _act;
    bool _normalizeBefore;
};

TORCH_MODULE(TransformerEncoderLayer);

/**
 * @brief AIFI transformer layer for 2D data with positional embeddings
 */
class AIFIImpl : public TransformerEncoderLayerImpl {
public:
    /**
     * @brief Construct a new AIFI module
     *
     * @param c1 Input dimension
     * @param cm Hidden dimension in FFN (default: 2048)
     * @param numHeads Number of attention heads (default: 8)
     * @param dropout Dropout probability (default: 0.0)
     * @param normalizeBefore Whether to apply normalization before attention (default: false)
     */
    AIFIImpl(int64_t c1, int64_t cm = 2048, int64_t numHeads = 8,
             double dropout = 0.0, bool normalizeBefore = false);

    /**
     * @brief Forward pass for AIFI transformer layer
     *
     * @param x Input tensor [B, C, H, W]
     * @return torch::Tensor Output tensor [B, C, H, W]
     */
    torch::Tensor forward(torch::Tensor x);

    /**
     * @brief Build 2D sine-cosine position embedding
     *
     * @param w Width of the feature map
     * @param h Height of the feature map
     * @param embedDim Embedding dimension
     * @param temperature Temperature for sine/cosine functions (default: 10000.0)
     * @return torch::Tensor Position embedding [1, embedDim, h*w]
     */
    static torch::Tensor build2dSincosPositionEmbedding(int64_t w, int64_t h, int64_t embedDim,
                                                        double temperature = 10000.0);
};

TORCH_MODULE(AIFI);

/**
 * @brief Multiscale Deformable Attention Module
 */
class MSDeformAttnImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new MSDeformAttn module
     *
     * @param dModel Model dimension (default: 256)
     * @param nLevels Number of feature levels (default: 4)
     * @param nHeads Number of attention heads (default: 8)
     * @param nPoints Number of sampling points per head per level (default: 4)
     */
    MSDeformAttnImpl(int64_t dModel = 256, int64_t nLevels = 4,
                     int64_t nHeads = 8, int64_t nPoints = 4);

    /**
     * @brief Forward pass for multiscale deformable attention
     *
     * @param query Query tensor [bs, query_length, C]
     * @param referBbox Reference bounding boxes [bs, query_length, n_levels, 2]
     * @param value Value tensor [bs, value_length, C]
     * @param valueShapes List of shapes [(H_0, W_0), ...]
     * @param valueMask Mask tensor (optional)
     * @return torch::Tensor Output tensor [bs, query_length, C]
     */
    torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& referBbox,
                          const torch::Tensor& value, const torch::Tensor& valueShapes,
                          const torch::Tensor& valueMask = {});

private:
    void resetParameters();

    int64_t _im2colStep;
    int64_t _dModel;
    int64_t _nLevels;
    int64_t _nHeads;
    int64_t _nPoints;
    torch::nn::Linear _samplingOffsets;
    torch::nn::Linear _attentionWeights;
    torch::nn::Linear _valueProj;
    torch::nn::Linear _outputProj;
};

TORCH_MODULE(MSDeformAttn);

/**
 * @brief Deformable Transformer Decoder Layer
 */
class DeformableTransformerDecoderLayerImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new DeformableTransformerDecoderLayer module
     *
     * @param dModel Model dimension (default: 256)
     * @param nHeads Number of attention heads (default: 8)
     * @param dFfn Dimension of FFN (default: 1024)
     * @param dropout Dropout probability (default: 0.0)
     * @param nLevels Number of feature levels (default: 4)
     * @param nPoints Number of sampling points (default: 4)
     */
    DeformableTransformerDecoderLayerImpl(int64_t dModel = 256, int64_t nHeads = 8,
                                          int64_t dFfn = 1024, double dropout = 0.0,
                                          int64_t nLevels = 4, int64_t nPoints = 4);

    /**
     * @brief Forward pass through decoder layer
     *
     * @param embed Input embeddings
     * @param referBbox Reference bounding boxes
     * @param feats Feature maps
     * @param shapes Feature shapes
     * @param paddingMask Padding mask (optional)
     * @param attnMask Attention mask (optional)
     * @param queryPos Query position embeddings (optional)
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(const torch::Tensor& embed, const torch::Tensor& referBbox,
                          const torch::Tensor& feats, const torch::Tensor& shapes,
                          const torch::Tensor& paddingMask = {},
                          const torch::Tensor& attnMask = {},
                          const torch::Tensor& queryPos = {});

    /**
     * @brief Add positional embeddings to tensor if provided
     */
    static torch::Tensor withPosEmbed(const torch::Tensor& tensor, const torch::Tensor& pos);

private:
    torch::Tensor forwardFfn(torch::Tensor tgt);

    torch::nn::MultiheadAttention _selfAttn;
    torch::nn::Dropout _dropout1;
    torch::nn::LayerNorm _norm1;
    MSDeformAttn _crossAttn;
    torch::nn::Dropout _dropout2;
    torch::nn::LayerNorm _norm2;
    torch::nn::Linear _linear1;
    torch::nn::ReLU _act;
    torch::nn::Dropout _dropout3;
    torch::nn::Linear _linear2;
    torch::nn::Dropout _dropout4;
    torch::nn::LayerNorm _norm3;
};

TORCH_MODULE(DeformableTransformerDecoderLayer);

/**
 * @brief Deformable Transformer Decoder
 */
class DeformableTransformerDecoderImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new DeformableTransformerDecoder module
     *
     * @param hiddenDim Hidden dimension
     * @param decoderLayer Decoder layer module
     * @param numLayers Number of decoder layers
     * @param evalIdx Index of layer to use during evaluation (default: -1)
     */
    DeformableTransformerDecoderImpl(int64_t hiddenDim, DeformableTransformerDecoderLayer decoderLayer,
                                     int64_t numLayers, int64_t evalIdx = -1);

    /**
     * @brief Forward pass through entire decoder
     *
     * @param embed Decoder embeddings
     * @param referBbox Reference bounding boxes
     * @param feats Image features
     * @param shapes Feature shapes
     * @param bboxHead Bounding box prediction head
     * @param scoreHead Score prediction head
     * @param posMlp Position MLP
     * @param attnMask Attention mask (optional)
     * @param paddingMask Padding mask (optional)
     * @return std::tuple<torch::Tensor, torch::Tensor> Decoded bboxes and classification scores
     */
    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor embed, torch::Tensor referBbox, torch::Tensor feats,
        const torch::Tensor& shapes, torch::nn::ModuleList bboxHead,
        torch::nn::ModuleList scoreHead, torch::nn::Sequential posMlp,
        const torch::Tensor& attnMask = {}, const torch::Tensor& paddingMask = {});

private:
    torch::nn::ModuleList _layers;
    int64_t _numLayers;
    int64_t _hiddenDim;
    int64_t _evalIdx;
    int64_t _dModel;
    int64_t _nHeads;
    int64_t _dFfn;
    double _dropout;
    int64_t _nLevels;
    int64_t _nPoints;
};

TORCH_MODULE(DeformableTransformerDecoder);

} // namespace Modules
} // namespace Model
} // namespace WheelDL
