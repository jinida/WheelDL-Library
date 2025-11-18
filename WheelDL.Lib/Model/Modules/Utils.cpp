#include "pch.h"
#include "Utils.h"
#include "Activation.h"
#include <cmath>
#include <algorithm>
#include <cctype>

namespace WheelDL {
    namespace Model {
        namespace Modules {

            float biasInitWithProb(float priorProb) 
            {
                // Validate input range
                if (priorProb <= 0.0f || priorProb >= 1.0f) {
                    throw std::invalid_argument("priorProb must be in (0, 1), got: " + std::to_string(priorProb));
                }
                // Calculate: -log((1 - priorProb) / priorProb)
                return static_cast<float>(-std::log((1.0f - priorProb) / priorProb));
            }

            void linearInit(torch::nn::Linear& module) {
                auto& weight = module->weight;

                // Validate weight has non-zero size
                if (weight.size(0) == 0) {
                    throw std::invalid_argument("Weight tensor must have non-zero size in dimension 0");
                }

                float bound = 1.0f / std::sqrt(static_cast<float>(weight.size(0)));

                torch::nn::init::uniform_(weight, -bound, bound);

                if (module->options.bias()) {
                    auto& bias = module->bias;
                    torch::nn::init::uniform_(bias, -bound, bound);
                }
            }

            torch::Tensor inverseSigmoid(const torch::Tensor& x, float eps) {
                // Clamp x to (eps, 1-eps) to avoid log(0) and log(inf)
                // This ensures numerical stability: log(eps/(1-eps)) to log((1-eps)/eps)
                auto xClamped = torch::clamp(x, eps, 1.0f - eps);

                // Compute inverse sigmoid: log(x / (1-x))
                return torch::log(xClamped / (1.0 - xClamped));
            }

            torch::nn::AnyModule createActivation(const std::string& actName) {
                // Convert to lowercase for case-insensitive comparison
                std::string actLower = actName;
                std::transform(actLower.begin(), actLower.end(), actLower.begin(),
                    [](unsigned char c) { return std::tolower(c); });

                // Empty string means no activation (Identity)
                if (actLower.empty() || actLower == "identity" || actLower == "none") {
                    return torch::nn::AnyModule(torch::nn::Identity());
                }

                // Match activation type
                if (actLower == "relu") {
                    return torch::nn::AnyModule(torch::nn::ReLU());
                }
                else if (actLower == "silu" || actLower == "swish") {
                    return torch::nn::AnyModule(torch::nn::SiLU());
                }
                else if (actLower == "leakyrelu") {
                    return torch::nn::AnyModule(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
                }
                else if (actLower == "gelu") {
                    return torch::nn::AnyModule(torch::nn::GELU());
                }
                else if (actLower == "mish") {
                    return torch::nn::AnyModule(torch::nn::Mish());
                }
                else if (actLower == "hardswish") {
                    return torch::nn::AnyModule(Hardswish());
                }
                else if (actLower == "aglu") {
                    return torch::nn::AnyModule(AGLU());
                }
                else {
                    // Default to ReLU for unknown activations
                    return torch::nn::AnyModule(torch::nn::ReLU());
                }
            }

            torch::Tensor multiScaleDeformableAttnPytorch(
                const torch::Tensor& value,
                const torch::Tensor& valueSpatialShapes,
                const torch::Tensor& samplingLocations,
                const torch::Tensor& attentionWeights
            ) {
                using namespace torch::indexing;

                // value: (bs, numKeys, numHeads, embedDims)
                // valueSpatialShapes: (numLevels, 2)
                // samplingLocations: (bs, numQueries, numHeads, numLevels, numPoints, 2)
                // attentionWeights: (bs, numQueries, numHeads, numLevels, numPoints)

                auto bs = value.size(0);
                auto numHeads = value.size(2);
                auto embedDims = value.size(3);

                auto numQueries = samplingLocations.size(1);
                auto numLevels = samplingLocations.size(3);
                auto numPoints = samplingLocations.size(4);

                // Split value by spatial shapes
                std::vector<torch::Tensor> valueList;
                std::vector<int64_t> splitSizes;

                // Validate valueSpatialShapes has at least 2 columns (H, W)
                if (valueSpatialShapes.size(1) < 2) {
                    throw std::invalid_argument("valueSpatialShapes must have at least 2 columns (H, W), got: " +
                                               std::to_string(valueSpatialShapes.size(1)));
                }

                auto shapesAccessor = valueSpatialShapes.accessor<int64_t, 2>();
                for (int64_t i = 0; i < valueSpatialShapes.size(0); ++i) {
                    int64_t H = shapesAccessor[i][0];
                    int64_t W = shapesAccessor[i][1];
                    splitSizes.push_back(H * W);
                }

                valueList = value.split_with_sizes(splitSizes, /*dim=*/1);

                // Convert sampling locations to grid coordinates
                // samplingGrids = 2 * samplingLocations - 1
                auto samplingGrids = 2.0 * samplingLocations - 1.0;

                std::vector<torch::Tensor> samplingValueList;

                for (int64_t level = 0; level < numLevels; ++level) {
                    int64_t H = shapesAccessor[level][0];
                    int64_t W = shapesAccessor[level][1];

                    // valueL: (bs, H*W, numHeads, embedDims)
                    // -> (bs, H*W, numHeads*embedDims)
                    // -> (bs, numHeads*embedDims, H*W)
                    // -> (bs*numHeads, embedDims, H, W)
                    auto valueL = valueList[level]
                        .flatten(2)
                        .transpose(1, 2)
                        .reshape({ bs * numHeads, embedDims, H, W });

                    // samplingGridL: (bs, numQueries, numHeads, numPoints, 2)
                    // -> (bs, numHeads, numQueries, numPoints, 2)
                    // -> (bs*numHeads, numQueries, numPoints, 2)
                    auto samplingGridL = samplingGrids.index({ Slice(), Slice(), Slice(), level })
                        .transpose(1, 2)
                        .flatten(0, 1);

                    // Apply grid_sample
                    // Output: (bs*numHeads, embedDims, numQueries, numPoints)
                    auto samplingValueL = torch::nn::functional::grid_sample(
                        valueL,
                        samplingGridL,
                        torch::nn::functional::GridSampleFuncOptions()
                        .mode(torch::kBilinear)
                        .padding_mode(torch::kZeros)
                        .align_corners(false)
                    );

                    samplingValueList.push_back(samplingValueL);
                }

                // Reshape attention weights
                // (bs, numQueries, numHeads, numLevels, numPoints)
                // -> (bs, numHeads, numQueries, numLevels, numPoints)
                // -> (bs*numHeads, 1, numQueries, numLevels*numPoints)
                auto attnWeights = attentionWeights
                    .transpose(1, 2)
                    .reshape({ bs * numHeads, 1, numQueries, numLevels * numPoints });

                // Stack sampling values and apply attention
                // (bs*numHeads, embedDims, numQueries, numLevels, numPoints)
                auto stacked = torch::stack(samplingValueList, /*dim=*/-2);
                // -> (bs*numHeads, embedDims, numQueries, numLevels*numPoints)
                auto flattened = stacked.flatten(-2);

                // Apply attention weights and sum
                auto output = (flattened * attnWeights)
                    .sum(-1)
                    .view({ bs, numHeads * embedDims, numQueries });

                // Transpose to (bs, numQueries, embedDims)
                return output.transpose(1, 2).contiguous();
            }

            torch::Tensor dist2bbox(
                const torch::Tensor& distance,
                const torch::Tensor& anchors,
                bool xywh,
                int64_t dim
            ) {
                // distance: (bs, 4, h*w) or (bs, h*w, 4)
                // anchors: (h*w, 2) or (1, h*w, 2)

                // Validate distance dimension is even
                auto distDimSize = distance.size(dim);
                if (distDimSize <= 0 || distDimSize % 2 != 0) {
                    throw std::invalid_argument("distance.size(dim) must be positive and even, got: " +
                                               std::to_string(distDimSize));
                }

                auto size = distDimSize / 2;
                auto lt = distance.narrow(dim, 0, size);  // left-top [B, 2, N]
                auto rb = distance.narrow(dim, size, size);  // right-bottom [B, 2, N]

                // Ensure anchors has correct dimensions: [N, 2] -> [1, N, 2] or keep [B, N, 2]
                auto anchorExpanded = anchors;
                if (anchors.dim() == 2) {
                    anchorExpanded = anchors.unsqueeze(0);  // [N, 2] -> [1, N, 2]
                }
                // If already 3D [B, N, 2], use as-is

                // Validate anchor dimension is 2 (x, y)
                auto anchorDimSize = anchorExpanded.size(-1);  // Last dimension should be 2
                if (anchorDimSize != 2) {
                    throw std::invalid_argument("anchor last dimension must be 2 (x, y), got: " +
                                               std::to_string(anchorDimSize));
                }

                // Permute lt/rb from [B, 2, N] to [B, N, 2] to match anchor format
                lt = lt.permute({0, 2, 1});  // [B, 2, N] -> [B, N, 2]
                rb = rb.permute({0, 2, 1});  // [B, 2, N] -> [B, N, 2]

                // Now compute bbox: anchors [B, N, 2] +/- lt/rb [B, N, 2]
                auto x1y1 = anchorExpanded - lt;  // [B, N, 2]
                auto x2y2 = anchorExpanded + rb;  // [B, N, 2]

                // Permute back to [B, 2, N] for concatenation
                x1y1 = x1y1.permute({0, 2, 1});  // [B, N, 2] -> [B, 2, N]
                x2y2 = x2y2.permute({0, 2, 1});  // [B, N, 2] -> [B, 2, N]

                auto bbox = torch::cat({ x1y1, x2y2 }, dim);  // [B, 4, N]

                if (xywh) {
                    // Convert to center x, center y, width, height
                    auto c_xy = (x1y1 + x2y2) / 2.0;
                    auto wh = x2y2 - x1y1;
                    return torch::cat({ c_xy, wh }, dim);
                }

                return bbox;  // Return as x1y1x2y2
            }

            torch::Tensor dist2rbox(
                const torch::Tensor& distance,
                const torch::Tensor& angle,
                const torch::Tensor& anchors,
                int64_t dim
            ) {
                // distance: (bs, 4, h*w)
                // angle: (bs, 1, h*w)
                // anchors: (h*w, 2)

                // Validate distance dimension is even
                auto distDimSize = distance.size(dim);
                if (distDimSize <= 0 || distDimSize % 2 != 0) {
                    throw std::invalid_argument("distance.size(dim) must be positive and even, got: " +
                                               std::to_string(distDimSize));
                }

                auto size = distDimSize / 2;
                auto lt = distance.narrow(dim, 0, size);
                auto rb = distance.narrow(dim, size, size);

                auto cos = torch::cos(angle);
                auto sin = torch::sin(angle);

                // Compute center points
                auto anchorExpanded = anchors.unsqueeze(0);
                if (anchors.dim() == 2) {
                    anchorExpanded = anchors;
                }

                auto xc = anchorExpanded.select(-1, 0);
                auto yc = anchorExpanded.select(-1, 1);

                // Compute rotated box parameters - validate lt size is even
                auto ltDimSize = lt.size(dim);
                if (ltDimSize <= 0 || ltDimSize % 2 != 0) {
                    throw std::invalid_argument("lt.size(dim) must be positive and even, got: " +
                                               std::to_string(ltDimSize));
                }
                auto ltSize = ltDimSize / 2;
                auto w = lt.narrow(dim, 0, ltSize) + rb.narrow(dim, 0, ltSize);
                auto h = lt.narrow(dim, ltSize, ltSize) + rb.narrow(dim, ltSize, ltSize);

                // Return [cx, cy, w, h, angle]
                std::vector<torch::Tensor> outputs;
                outputs.push_back(xc.unsqueeze(dim));
                outputs.push_back(yc.unsqueeze(dim));
                outputs.push_back(w);
                outputs.push_back(h);
                outputs.push_back(angle);

                return torch::cat(outputs, dim);
            }

            std::pair<torch::Tensor, torch::Tensor> makeAnchors(
                const std::vector<torch::Tensor>& feats,
                const torch::Tensor& stride,
                double gridCellOffset
            ) {
                // Validate inputs
                if (feats.empty()) {
                    throw std::invalid_argument("makeAnchors: feats vector cannot be empty");
                }

                if (stride.size(0) != static_cast<int64_t>(feats.size())) {
                    throw std::invalid_argument("makeAnchors: stride size (" +
                                               std::to_string(stride.size(0)) +
                                               ") must match feats size (" +
                                               std::to_string(feats.size()) + ")");
                }

                // Generate anchor points for each feature map
                std::vector<torch::Tensor> anchorPoints;
                std::vector<torch::Tensor> stridePoints;

                auto device = feats[0].device();
                auto dtype = feats[0].dtype();

                for (size_t i = 0; i < feats.size(); ++i) {
                    // Validate tensor is 4D
                    if (feats[i].dim() != 4) {
                        throw std::invalid_argument("makeAnchors: feats[" + std::to_string(i) +
                                                   "] must be 4D (NCHW), got " +
                                                   std::to_string(feats[i].dim()) + "D");
                    }

                    auto h = feats[i].size(2);
                    auto w = feats[i].size(3);

                    // Validate dimensions are positive
                    if (h <= 0 || w <= 0) {
                        throw std::invalid_argument("makeAnchors: feats[" + std::to_string(i) +
                                                   "] has invalid spatial dimensions: " +
                                                   std::to_string(h) + "x" + std::to_string(w));
                    }

                    auto sx = torch::arange(0, w, torch::TensorOptions().dtype(dtype).device(device));
                    auto sy = torch::arange(0, h, torch::TensorOptions().dtype(dtype).device(device));

                    auto meshgridResult = torch::meshgrid({ sy, sx }, "ij");
                    auto& gridY = meshgridResult[0];
                    auto& gridX = meshgridResult[1];

                    // Stack and add offset
                    auto grid = torch::stack({ gridX, gridY }, -1) + gridCellOffset;

                    // Flatten and reshape
                    auto anchorPoint = grid.reshape({ -1, 2 });

                    // Create stride tensor
                    auto strideValue = stride[i].item<double>();
                    auto strideTensor = torch::full({ anchorPoint.size(0), 1 }, strideValue,
                        torch::TensorOptions().dtype(dtype).device(device));

                    anchorPoints.push_back(anchorPoint);
                    stridePoints.push_back(strideTensor);
                }

                // Concatenate all levels
                auto anchors = torch::cat(anchorPoints, 0);
                auto strides = torch::cat(stridePoints, 0);

                return { anchors, strides };
            }

        } // namespace Modules
    } // namespace Model
} // namespace WheelDL
