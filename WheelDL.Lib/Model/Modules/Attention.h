#pragma once

#include <torch/torch.h>
#include "Interfaces.h"
#include "Conv.h"

namespace WheelDL {
namespace Model {
namespace Modules {

/**
 * @brief Channel-attention module for feature recalibration
 *
 * Applies attention weights to channels based on global average pooling.
 *
 * Reference: https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
 */
class ChannelAttentionImpl : public IBlockImpl {
public:
    /**
     * @brief Construct a new ChannelAttention module
     *
     * @param channels Number of input channels
     */
    explicit ChannelAttentionImpl(int64_t channels);

    /**
     * @brief Apply channel attention to input tensor
     *
     * @param x Input tensor
     * @return torch::Tensor Channel-attended output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::AdaptiveAvgPool2d _pool = nullptr;
    torch::nn::Conv2d _fc = nullptr;
    torch::nn::Sigmoid _act = nullptr;
};

TORCH_MODULE(ChannelAttention);

/**
 * @brief Spatial-attention module for feature recalibration
 *
 * Applies attention weights to spatial dimensions based on channel statistics.
 */
class SpatialAttentionImpl : public IBlockImpl {
public:
    /**
     * @brief Construct a new SpatialAttention module
     *
     * @param kernelSize Size of the convolutional kernel (3 or 7, default: 7)
     */
    explicit SpatialAttentionImpl(int64_t kernelSize = 7);

    /**
     * @brief Apply spatial attention to input tensor
     *
     * @param x Input tensor
     * @return torch::Tensor Spatial-attended output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d _cv1 = nullptr;
    torch::nn::Sigmoid _act = nullptr;
};

TORCH_MODULE(SpatialAttention);

/**
 * @brief Convolutional Block Attention Module (CBAM)
 *
 * Combines channel and spatial attention mechanisms for comprehensive feature refinement.
 */
class CBAMImpl : public IBlockImpl {
public:
    /**
     * @brief Construct a new CBAM module
     *
     * @param c1 Number of input channels
     * @param kernelSize Size of the convolutional kernel for spatial attention (default: 7)
     */
    explicit CBAMImpl(int64_t c1, int64_t kernelSize = 7);

    /**
     * @brief Apply channel and spatial attention sequentially
     *
     * @param x Input tensor
     * @return torch::Tensor Attended output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    ChannelAttention _channelAttention = nullptr;
    SpatialAttention _spatialAttention = nullptr;
};

TORCH_MODULE(CBAM);

/**
 * @brief Self-attention module for feature extraction
 *
 * Performs multi-head self-attention on input tensors
 */
class AttentionImpl : public IBlockImpl {
public:
    /**
     * @brief Construct a new Attention module
     *
     * @param dim Input tensor dimension
     * @param numHeads Number of attention heads (default: 8)
     * @param attnRatio Ratio of attention key dimension to head dimension (default: 0.5)
     */
    AttentionImpl(int64_t dim, int64_t numHeads = 8, double attnRatio = 0.5);

    /**
     * @brief Forward pass of Attention module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor after self-attention
     */
    torch::Tensor forward(torch::Tensor x);

private:
    int64_t _numHeads;
    int64_t _headDim;
    int64_t _keyDim;
    double _scale;
    Conv _qkv = nullptr;
    Conv _proj = nullptr;
    Conv _pe = nullptr;
};

TORCH_MODULE(Attention);

/**
 * @brief Position-Sensitive Attention block
 *
 * Applies multi-head attention and feed-forward network with optional shortcut
 */
class PSABlockImpl : public IBlockImpl {
public:
    /**
     * @brief Construct a new PSABlock module
     *
     * @param c Input and output channels
     * @param attnRatio Attention ratio for key dimension (default: 0.5)
     * @param numHeads Number of attention heads (default: 4)
     * @param shortcut Whether to use shortcut connections (default: true)
     */
    PSABlockImpl(int64_t c, double attnRatio = 0.5, int64_t numHeads = 4, bool shortcut = true);

    /**
     * @brief Forward pass through PSABlock
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor after attention and feed-forward processing
     */
    torch::Tensor forward(torch::Tensor x);

private:
    Attention _attn = nullptr;
    Conv _ffn_cv1 = nullptr;
    Conv _ffn_cv2 = nullptr;
    bool _add;
};

TORCH_MODULE(PSABlock);

/**
 * @brief Position-Sensitive Attention module
 *
 * Enhances feature extraction with position-sensitive attention mechanism
 */
class PSAImpl : public IBlockImpl {
public:
    /**
     * @brief Construct a new PSA module
     *
     * @param c1 Input channels
     * @param c2 Output channels
     * @param e Expansion ratio (default: 0.5)
     */
    PSAImpl(int64_t c1, int64_t c2, double e = 0.5);

    /**
     * @brief Forward pass through PSA module
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor after attention and feed-forward processing
     */
    torch::Tensor forward(torch::Tensor x);

private:
    int64_t _c;
    Conv _cv1 = nullptr;
    Conv _cv2 = nullptr;
    Attention _attn = nullptr;
    Conv _ffn_cv1 = nullptr;
    Conv _ffn_cv2 = nullptr;
};

TORCH_MODULE(PSA);

} // namespace Modules
} // namespace Model
} // namespace WheelDL
