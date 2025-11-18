#include "pch.h"
#include "OBBLoss.h"
#include "../Utils/IoU.h"
#include <cmath>
#include <stdexcept>

namespace WheelDL {
    namespace Model {
        namespace Loss {

            // ============================================================================
            // OrientedBboxLoss - Internal class for OBB Loss with Probiou and DFL
            // ============================================================================

            class OrientedBboxLoss {
            public:
                explicit OrientedBboxLoss(int64_t regMax) {
                    if (regMax > 1) {
                        _dflLoss = std::make_unique<DFLoss>(regMax);
                    }
                }

                std::tuple<torch::Tensor, torch::Tensor> forward(
                    const torch::Tensor& predDist,
                    const torch::Tensor& predBboxes,
                    const torch::Tensor& anchorPoints,
                    const torch::Tensor& targetBboxes,
                    const torch::Tensor& targetScores,
                    const torch::Tensor& targetScoresSum,
                    const torch::Tensor& fgMask
                ) {
                    // Weight from target scores
                    auto weight = targetScores.sum(-1).index({ fgMask }).unsqueeze(-1);

                    // Compute Probiou for oriented boxes (element-wise)
                    auto predBboxesFg = predBboxes.index({ fgMask });
                    auto targetBboxesFg = targetBboxes.index({ fgMask });

                    // Compute element-wise IoU
                    auto iou = Utils::probiou(
                        predBboxesFg,
                        targetBboxesFg
                    );  // [N]

                    // IoU loss
                    auto lossIoU = ((1.0f - iou) * weight).sum() / targetScoresSum;

                    torch::Tensor lossDfl;
                    if (_dflLoss) {
                        // For OBB, convert target boxes to xyxy for distance computation
                        // Python: target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
                        auto targetXywh = targetBboxes.slice(2, 0, 4);  // [batch, num_gt, 4]
                        auto targetXyxy = Utils::xywh2xyxy(targetXywh);

                        auto targetLtrb = Utils::bbox2dist(
                            anchorPoints, targetXyxy, _dflLoss->getRegMax() - 1
                        );

                        auto predDistFg = predDist.index({ fgMask }).view({ -1, _dflLoss->getRegMax() });
                        auto targetLtrbFg = targetLtrb.index({ fgMask });

                        auto dflLossVal = _dflLoss->compute(predDistFg, targetLtrbFg);
                        lossDfl = (dflLossVal * weight).sum() / targetScoresSum;
                    }
                    else {
                        lossDfl = torch::zeros({ 1 }, predDist.options());
                    }

                    return std::make_tuple(lossIoU, lossDfl);
                }

            private:
                std::unique_ptr<DFLoss> _dflLoss;
            };

            // ============================================================================
            // OBBLoss Implementation
            // ============================================================================

            OBBLoss::OBBLoss(
                int64_t numClasses,
                const torch::Tensor& stride,
                float boxGain,
                float clsGain,
                float dflGain,
                int64_t talTopk
            )
                : _numClasses(numClasses)
                , _stride(stride.clone())  // FIXED: Clone to ensure internal state independence
                , _boxGain(boxGain)
                , _clsGain(clsGain)
                , _dflGain(dflGain)
                , _regMax(16)
                , _numOutputs(16 * 4 + numClasses + 1)
                , _totalStrideFactor(0.0f)
                , _useDfl(true)  // Always use DFL since regMax = 16 > 1
                , _device(stride.device())  // Use same device as stride tensor
            {
                // Initialize loss components
                _bce = std::make_unique<BCEWithLogitsLoss>("none");
                _bboxLoss = std::make_unique<OrientedBboxLoss>(_regMax);
                _assigner = std::make_unique<Utils::OBBTaskAlignedAssigner>(
                    talTopk, numClasses, 0.5f, 6.0f
                );

                // OPTIMIZATION: Pre-cache projection tensors for different dtypes
                _proj = torch::arange(_regMax, torch::TensorOptions().dtype(torch::kFloat).device(_device));
                _projFloat32 = _proj.to(torch::kFloat32);
                _projFloat16 = _proj.to(torch::kFloat16);

                // BFloat16 support for Ampere+ GPUs (A100, RTX 3090, etc.)
                if (torch::cuda::is_available() && torch::cuda::cudnn_is_available()) {
                    try {
                        // Test BFloat16 support with a small tensor operation
                        auto testTensor = torch::ones({1}, torch::TensorOptions().device(_device).dtype(torch::kBFloat16));
                        _projBFloat16 = _proj.to(torch::kBFloat16);
                    } catch (const c10::Error& e) {
                        // BFloat16 not supported on this device, fallback to Float16
                        // Log the error for debugging purposes
                        std::cerr << "[OBBLoss] BFloat16 not supported: " << e.what() << "\n";
                        std::cerr << "[OBBLoss] Falling back to Float16 for BFloat16 operations\n";
                        _projBFloat16 = _projFloat16;
                    } catch (const std::exception& e) {
                        // Unexpected error
                        std::cerr << "[OBBLoss] Unexpected error during BFloat16 setup: " << e.what() << "\n";
                        _projBFloat16 = _projFloat16;
                    }
                } else {
                    _projBFloat16 = _projFloat16;  // Fallback to Float16
                }

                // OPTIMIZATION: Cache stride values on CPU once during initialization
                // Use cloned stride to ensure we have our own copy
                auto strideCpu = _stride.cpu();
                auto strideAccessor = strideCpu.accessor<float, 1>();
                _strideValuesCPU.reserve(strideCpu.size(0));
                for (int64_t i = 0; i < strideCpu.size(0); ++i) {
                    float s = strideAccessor[i];
                    _strideValuesCPU.push_back(s);
                    _totalStrideFactor += 1.0f / (s * s);
                }
            }

			OBBLoss::~OBBLoss() = default;

            void OBBLoss::to(const torch::Device& device)
            {
                _device = device;
                _stride = _stride.to(device);
                _proj = _proj.to(device);
            }

            torch::Tensor OBBLoss::preprocess(
                const torch::Tensor& targets,
                int64_t batchSize,
                const torch::Tensor& scaleTensor
            ) {
                if (targets.size(0) == 0) {
                    return torch::zeros({ batchSize, 0, 6 },
                        torch::TensorOptions().device(_device));
                }

                // OPTIMIZATION: Vectorized batch processing
                auto i = targets.select(1, 0);  // batch indices

                // Get unique batch indices and counts efficiently
                auto uniqueResult = torch::_unique2(i, /*sorted=*/true, /*return_inverse=*/true, /*return_counts=*/true);
                auto uniqueIndices = std::get<0>(uniqueResult);
                auto inverseIndices = std::get<1>(uniqueResult);
                auto counts = std::get<2>(uniqueResult);

                auto maxCount = counts.max().item<int64_t>();
                auto out = torch::zeros({ batchSize, maxCount, 6 },
                    torch::TensorOptions().device(_device));

                // OPTIMIZATION: Process all batches with minimal loops
                if (targets.size(0) > 0 && batchSize > 0) {
                    for (int64_t idx = 0; idx < uniqueIndices.size(0); ++idx) {
                        auto batchIdx = uniqueIndices[idx].item<int64_t>();
                        if (batchIdx < batchSize) {
                            auto mask = (i == batchIdx);
                            auto n = counts[idx].item<int64_t>();

                            if (n > 0) {
                                auto bboxes = targets.index({ mask }).slice(1, 2, 7);  // [n, 5] - xywh + angle

                                // Scale bbox coordinates (not angle) - vectorized operation
                                auto bboxesXywh = bboxes.slice(1, 0, 4);  // [n, 4]
                                bboxesXywh = bboxesXywh * scaleTensor.unsqueeze(0);

                                auto angle = bboxes.slice(1, 4, 5);  // [n, 1]
                                auto scaledBboxes = torch::cat({ bboxesXywh, angle }, /*dim=*/1);  // [n, 5]

                                auto classLabels = targets.index({ mask }).slice(1, 1, 2);  // [n, 1]
                                out[batchIdx].slice(0, 0, n) = torch::cat({ classLabels, scaledBboxes }, /*dim=*/1);
                            }
                        }
                    }
                }

                return out;
            }

            torch::Tensor OBBLoss::decodeBbox(
                const torch::Tensor& anchorPoints,
                const torch::Tensor& predDist,
                const torch::Tensor& predAngle
            ) {
                torch::Tensor predDistDecoded;
                if (_useDfl) {
                    // OPTIMIZATION: Fused DFL decoding with cached projection tensor
                    auto b = predDist.size(0);
                    auto a = predDist.size(1);
                    auto c = predDist.size(2);

                    // Reshape and apply softmax
                    auto predDistReshaped = predDist.view({ b, a, 4, c / 4 });
                    auto predDistSoftmax = torch::softmax(predDistReshaped, /*dim=*/3);

                    // OPTIMIZATION: Use pre-cached projection tensor based on dtype
                    torch::Tensor proj;
                    if (predDist.dtype() == torch::kFloat16) {
                        proj = _projFloat16;
                    } else if (predDist.dtype() == torch::kBFloat16) {
                        proj = _projBFloat16;
                    } else {
                        proj = _projFloat32;
                    }

                    // Matrix multiply with cached projection (no dtype conversion needed)
                    predDistDecoded = torch::matmul(predDistSoftmax, proj);
                }
                else {
                    predDistDecoded = predDist;
                }

                // Convert to rotated boxes and append angle
                auto rbox = Utils::dist2rbox(predDistDecoded, predAngle, anchorPoints);
                return torch::cat({ rbox, predAngle }, /*dim=*/-1);  // [batch, num_anchors, 5]
            }

            std::unordered_map<std::string, torch::Tensor> OBBLoss::compute(
                const std::vector<torch::Tensor>& predictions,
                const Data::Dataset::DataExample& target) {
                // OPTIMIZATION: Pre-allocate concatenated tensor to avoid intermediate vectors
                // predictions: vector of [batch, 145, H, W] for each scale
                // where 145 = 4*regMax + numClasses + 1 (for angle)
                // Convert to concatenated format [batch, 145, total_anchors]

                if (predictions.empty()) {
                    throw std::invalid_argument("OBBLoss::compute: predictions cannot be empty");
                }

                auto batchSize = predictions[0].size(0);
                auto channels = predictions[0].size(1);

                // 1. Calculate total number of anchors across all scales
                int64_t totalAnchors = 0;
                for (const auto& pred : predictions) {
                    totalAnchors += pred.size(2) * pred.size(3);
                }

                // 2. Pre-allocate output tensor (single allocation)
                auto concatenated = torch::empty(
                    {batchSize, channels, totalAnchors},
                    predictions[0].options()
                );

                // 3. Copy each scale directly into pre-allocated tensor (no intermediate storage)
                int64_t offset = 0;
                for (const auto& pred : predictions) {
                    auto height = pred.size(2);
                    auto width = pred.size(3);
                    auto numAnchors = height * width;

                    // Direct copy into the concatenated tensor
                    concatenated.slice(2, offset, offset + numAnchors).copy_(
                        pred.reshape({batchSize, channels, numAnchors})
                    );
                    offset += numAnchors;
                }

                // Call single tensor version
                return compute(concatenated, target);
            }

            std::unordered_map<std::string, torch::Tensor> OBBLoss::compute(
                const torch::Tensor& prediction,
                const Data::Dataset::DataExample& target) {
                // Initialize loss tensor: [box, cls, dfl]
                auto loss = torch::zeros({ 3 }, torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(_device));

                // OPTIMIZATION: Use slice instead of split to avoid creating temporary vector
                // Split predictions: [batch, reg_max*4 + num_classes + 1, num_anchors]
                // Last channel is angle
                auto predAngle = prediction.slice(1, _regMax * 4 + _numClasses, _regMax * 4 + _numClasses + 1);
                predAngle = predAngle.permute({ 0, 2, 1 }).contiguous();

                // Get distribution and scores using slices
                auto predDistri = prediction.slice(1, 0, _regMax * 4).permute({ 0, 2, 1 }).contiguous();
                auto predScores = prediction.slice(1, _regMax * 4, _regMax * 4 + _numClasses).permute({ 0, 2, 1 }).contiguous();

                auto dtype = predScores.dtype();
                auto batchSize = predScores.size(0);
                auto numAnchors = predScores.size(1);

                auto imgSizeFloat = std::sqrt(static_cast<float>(numAnchors) / _totalStrideFactor);
                auto imgSizeInt = static_cast<int64_t>(std::round(imgSizeFloat));
                auto imgSize = torch::tensor({ static_cast<float>(imgSizeInt), static_cast<float>(imgSizeInt) },
                    torch::TensorOptions().dtype(dtype).device(_device));

                // OPTIMIZATION: Use cached CPU stride values to avoid GPU->CPU transfer
                std::vector<std::pair<int64_t, int64_t>> featShapes;
                featShapes.reserve(_strideValuesCPU.size());
                for (size_t i = 0; i < _strideValuesCPU.size(); ++i)
                {
                    auto strideVal = static_cast<int64_t>(_strideValuesCPU[i]);
                    auto featH = imgSizeInt / strideVal;
                    auto featW = imgSizeInt / strideVal;
                    featShapes.emplace_back(featH, featW);
                }

                // OPTIMIZATION: Use cached anchors if available
                torch::Tensor anchorPoints, strideTensor;
                auto dtypeScalar = dtype.toScalarType();

                if (_anchorCache.imgSize == imgSizeInt &&
                    _anchorCache.dtype == dtypeScalar &&
                    _anchorCache.device == _device &&
                    _anchorCache.anchorPoints.defined()) {
                    // Use cached anchors
                    anchorPoints = _anchorCache.anchorPoints;
                    strideTensor = _anchorCache.strideTensor;
                } else {
                    // Generate new anchors and cache them
                    std::tie(anchorPoints, strideTensor) = Utils::makeAnchors(
                        featShapes, _stride, dtypeScalar, _device, 0.5f
                    );

                    // Update cache
                    _anchorCache.anchorPoints = anchorPoints;
                    _anchorCache.strideTensor = strideTensor;
                    _anchorCache.imgSize = imgSizeInt;
                    _anchorCache.dtype = dtypeScalar;
                    _anchorCache.device = _device;
                }

                // Prepare targets: torch.cat((batch_idx, classes, obb), 1) where obb is [x, y, w, h, angle]
                auto batchIdx = target.batchIndices.view({ -1, 1 });
                auto classes = target.classes.view({ -1, 1 });
                auto obb = target.targets;  // [num_gt, 5] - xywh + angle

                // OPTIMIZATION: Ensure all tensors are on correct device before concatenation
                auto targets = torch::cat({
                    batchIdx.to(_device),
                    classes.to(_device),
                    obb.to(_device)
                }, /*dim=*/1);

                // Filter small bboxes: targets[(rw >= 2) & (rh >= 2)]
                // Python: rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
                // targets structure: [batch_idx, class, x, y, w, h, angle]
                // index 4 = w (width), index 5 = h (height)
                // imgsz = [height, width] so imgsz[0] = height, imgsz[1] = width
                // Note: Python code uses [0] for width and [1] for height, which works for square images.
                // For non-square images, corrected to use proper dimensions:
                auto rw = targets.select(1, 4) * imgSize[1].item<float>();  // w * imgSize[1] (width)
                auto rh = targets.select(1, 5) * imgSize[0].item<float>();  // h * imgSize[0] (height)
                auto validMask = (rw >= 2.0f) & (rh >= 2.0f);
                targets = targets.index({ validMask });

                // Python: scale_tensor=imgsz[[1, 0, 1, 0]]
                auto scaleTensor = torch::stack({ imgSize[1], imgSize[0], imgSize[1], imgSize[0] });
                targets = preprocess(targets, batchSize, scaleTensor);

                // Split targets: gt_labels, gt_bboxes = targets.split((1, 5), 2)
                auto targetParts = targets.split({ 1, 5 }, /*dim=*/2);
                auto gtLabels = targetParts[0];
                auto gtBboxes = targetParts[1];  // gtBboxes: [batch, max_num_gt, 5] - xywh + angle

                // mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
                auto maskGt = gtBboxes.sum(/*dim=*/2, /*keepdim=*/true) > 0.0f;

                // Decode predicted boxes
                auto predBboxes = decodeBbox(anchorPoints, predDistri, predAngle);

                // OPTIMIZATION: Clone and scale inplace for assigner
                auto bboxesForAssigner = predBboxes.clone().detach();
                bboxesForAssigner.slice(2, 0, 4).mul_(strideTensor);  // Inplace scale xywh only, not angle

                // Task-aligned assignment
                auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] = _assigner->forward(
                    predScores.detach().sigmoid(),
                    bboxesForAssigner.to(gtBboxes.dtype()),
                    anchorPoints * strideTensor,
                    gtLabels,
                    gtBboxes,
                    maskGt
                );

                // Fixed: Validate for NaN/Inf before using in division
                auto targetScoresSumRaw = targetScores.sum().item<float>();
                float targetScoresSum = 1.0f;
                if (std::isfinite(targetScoresSumRaw) && targetScoresSumRaw > 0.0f) {
                    targetScoresSum = targetScoresSumRaw;
                }

                auto bceLossMap = _bce->compute(predScores, targetScores.to(dtype));
                loss[1] = bceLossMap.at("total").sum() / targetScoresSum;

                // Box and DFL loss (only for foreground)
                if (fgMask.sum().item<int64_t>() > 0) {
                    // OPTIMIZATION: Inplace division for xywh only (not angle)
                    targetBboxes.slice(2, 0, 4).div_(strideTensor);

                    auto [boxLoss, dflLoss] = _bboxLoss->forward(
                        predDistri,
                        predBboxes,
                        anchorPoints,
                        targetBboxes,
                        targetScores,
                        torch::tensor(targetScoresSum, predScores.options()),
                        fgMask
                    );

                    loss[0] = boxLoss;
                    loss[2] = dflLoss;
                }
                else {
                    // When no foreground boxes exist, add dummy gradient to keep computational graph connected
                    // This prevents gradient computation errors in PyTorch's autograd
                    // The multiplication by 0 ensures no actual loss contribution
                    loss[0] += (predAngle * 0.0f).sum();
                }

                // OPTIMIZATION: Apply loss weights inplace
                loss[0].mul_(_boxGain);
                loss[1].mul_(_clsGain);
                loss[2].mul_(_dflGain);

                // OPTIMIZATION: Multiply by batch_size inplace
                loss.mul_(static_cast<float>(batchSize));

                // Return loss components as map
                return {
                    {"box", loss[0]},
                    {"cls", loss[1]},
                    {"dfl", loss[2]},
                    {"total", loss.sum()}
                };
            }

        } // namespace Loss
    } // namespace Model
} // namespace WheelDL
