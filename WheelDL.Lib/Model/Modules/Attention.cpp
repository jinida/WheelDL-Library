#include "pch.h"
#include "Attention.h"

namespace WheelDL {
	namespace Model {
		namespace Modules {

			// ============================================================================
			// ChannelAttention Implementation
			// ============================================================================

			ChannelAttentionImpl::ChannelAttentionImpl(int64_t channels) {
				_pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(1));

				_fc = register_module("fc",
					torch::nn::Conv2d(torch::nn::Conv2dOptions(channels, channels, 1)
						.stride(1)
						.padding(0)
						.bias(true)));

				_act = register_module("act", torch::nn::Sigmoid());
			}

			torch::Tensor ChannelAttentionImpl::forward(torch::Tensor x) {
				auto pooled = _pool->forward(x);
				auto attention = _fc->forward(pooled);
				attention = _act->forward(attention);
				return x * attention;
			}

			// ============================================================================
			// SpatialAttention Implementation
			// ============================================================================

			SpatialAttentionImpl::SpatialAttentionImpl(int64_t kernelSize) {
				if (kernelSize != 3 && kernelSize != 7) 
				{
					throw std::invalid_argument("kernel size must be 3 or 7");
				}

				int64_t padding = (kernelSize == 7) ? 3 : 1;

				_cv1 = register_module("cv1",
					torch::nn::Conv2d(torch::nn::Conv2dOptions(2, 1, kernelSize)
						.padding(padding)
						.bias(false)));

				_act = register_module("act", torch::nn::Sigmoid());
			}

			torch::Tensor SpatialAttentionImpl::forward(torch::Tensor x) 
			{
				// Compute channel-wise statistics
				auto meanChannel = torch::mean(x, /*dim=*/1, /*keepdim=*/true);
				auto maxChannel = std::get<0>(torch::max(x, /*dim=*/1, /*keepdim=*/true));

				// Concatenate mean and max
				auto combined = torch::cat({ meanChannel, maxChannel }, /*dim=*/1);

				// Apply convolution and activation
				auto attention = _cv1->forward(combined);
				attention = _act->forward(attention);

				return x * attention;
			}

			// ============================================================================
			// CBAM Implementation
			// ============================================================================

			CBAMImpl::CBAMImpl(int64_t c1, int64_t kernelSize) {
				_channelAttention = register_module("channel_attention", ChannelAttention(c1));
				_spatialAttention = register_module("spatial_attention", SpatialAttention(kernelSize));
			}

			torch::Tensor CBAMImpl::forward(torch::Tensor x) {
				x = _channelAttention->forward(x);
				x = _spatialAttention->forward(x);
				return x;
			}

			// ============================================================================
			// Attention Implementation
			// ============================================================================

			AttentionImpl::AttentionImpl(int64_t dim, int64_t numHeads, double attnRatio) {
				if (numHeads <= 0) {
					throw std::invalid_argument("numHeads must be greater than 0");
				}
				if (dim % numHeads != 0) {
					throw std::invalid_argument("dim (" + std::to_string(dim) +
						") must be divisible by numHeads (" +
						std::to_string(numHeads) + ")");
				}
				_numHeads = numHeads;
				_headDim = dim / numHeads;

				// Ensure keyDim is at least 1
				_keyDim = std::max(static_cast<int64_t>(1),
					static_cast<int64_t>(_headDim * attnRatio));

				if (_keyDim <= 0) {
					throw std::invalid_argument("keyDim must be positive (headDim=" +
						std::to_string(_headDim) +
						", attnRatio=" + std::to_string(attnRatio) + ")");
				}
				_scale = std::pow(_keyDim, -0.5);

				int64_t nhKd = _keyDim * numHeads;
				int64_t h = dim + nhKd * 2;

				_qkv = register_module("qkv", Conv(dim, h, 1, 1, std::nullopt, 1, 1));
				_proj = register_module("proj", Conv(dim, dim, 1, 1, std::nullopt, 1, 1));
				_pe = register_module("pe", Conv(dim, dim, 3, 1, std::nullopt, dim, 1));
			}

			torch::Tensor AttentionImpl::forward(torch::Tensor x) {
				auto B = x.size(0);
				auto C = x.size(1);
				auto H = x.size(2);
				auto W = x.size(3);
				auto N = H * W;

				auto qkv = _qkv->forward(x);
				qkv = qkv.view({ B, _numHeads, _keyDim * 2 + _headDim, N });

#if TORCH_VERSION_MAJOR < 2
				auto splits = qkv.split_with_sizes({ _keyDim, _keyDim, _headDim }, /*dim=*/2);
#else
				auto splits = qkv.split({ _keyDim, _keyDim, _headDim }, /*dim=*/2);
#endif
				// Transpose from [B, numHeads, dim, N] to [B, numHeads, N, dim] for attention computation
				auto q = splits[0].transpose(-2, -1);  // [B, numHeads, N, keyDim]
				auto k = splits[1].transpose(-2, -1);  // [B, numHeads, N, keyDim]
				auto v = splits[2].transpose(-2, -1);  // [B, numHeads, N, headDim]

				// Standard attention: (Q @ K.T) * scale
				auto attn = torch::matmul(q, k.transpose(-2, -1)) * _scale;  // [B, numHeads, N, N]
				attn = attn.softmax(-1);

				// Apply attention to values: (attn @ V)
				x = torch::matmul(attn, v);  // [B, numHeads, N, headDim]
				// Transpose back to [B, numHeads, headDim, N] then reshape to [B, C, H, W]
				x = x.transpose(-2, -1).contiguous().view({ B, C, H, W });
				x = x + _pe->forward(v.transpose(-2, -1).contiguous().reshape({ B, C, H, W }));
				x = _proj->forward(x);

				return x;
			}

			// ============================================================================
			// PSABlock Implementation
			// ============================================================================

			PSABlockImpl::PSABlockImpl(int64_t c, double attnRatio, int64_t numHeads, bool shortcut) {
				_attn = register_module("attn", Attention(c, numHeads, attnRatio));

				// Cannot register Sequential directly, so split into individual modules
				_ffn_cv1 = register_module("ffn_cv1", Conv(c, c * 2, 1));
				_ffn_cv2 = register_module("ffn_cv2", Conv(c * 2, c, 1, 1, std::nullopt, 1, 1));

				_add = shortcut;
			}

			torch::Tensor PSABlockImpl::forward(torch::Tensor x) {
				if (_add) {
					x = x + _attn->forward(x);
					auto ffn_out = _ffn_cv1->forward(x);
					x = x + _ffn_cv2->forward(ffn_out);
				}
				else {
					x = _attn->forward(x);
					x = _ffn_cv1->forward(x);
					x = _ffn_cv2->forward(x);
				}
				return x;
			}

			// ============================================================================
			// PSA Implementation
			// ============================================================================

			PSAImpl::PSAImpl(int64_t c1, int64_t c2, double e) {
				if (c1 != c2) {
					throw std::invalid_argument("PSA requires c1 == c2");
				}

				_c = static_cast<int64_t>(c1 * e);

				// Validate _c is positive
				if (_c <= 0) {
					throw std::invalid_argument("PSA: expansion resulted in non-positive channels: " +
						std::to_string(_c));
				}

				_cv1 = register_module("cv1", Conv(c1, 2 * _c, 1, 1));
				_cv2 = register_module("cv2", Conv(2 * _c, c1, 1));

				// Calculate numHeads ensuring divisibility
				int64_t numHeads = std::max(static_cast<int64_t>(1), _c / 64);

				// Adjust numHeads to be a divisor of _c
				while (_c % numHeads != 0 && numHeads > 1) {
					numHeads--;
				}

				// Validate final numHeads
				if (_c % numHeads != 0) {
					throw std::invalid_argument("PSA: Cannot find valid numHeads for _c=" +
						std::to_string(_c));
				}

				_attn = register_module("attn", Attention(_c, numHeads, 0.5));

				// Cannot register Sequential directly, so split into individual modules
				_ffn_cv1 = register_module("ffn_cv1", Conv(_c, _c * 2, 1));
				_ffn_cv2 = register_module("ffn_cv2", Conv(_c * 2, _c, 1, 1, std::nullopt, 1, 1));
			}

			torch::Tensor PSAImpl::forward(torch::Tensor x) {
				auto splits = _cv1->forward(x).split(_c, /*dim=*/1);
				auto& a = splits[0];
				auto& b = splits[1];

				b = b + _attn->forward(b);
				auto ffn_out = _ffn_cv1->forward(b);
				b = b + _ffn_cv2->forward(ffn_out);

				return _cv2->forward(torch::cat({ a, b }, 1));
			}

		} // namespace Modules
	} // namespace Model
} // namespace WheelDL
