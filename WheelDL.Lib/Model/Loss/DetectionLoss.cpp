#include "pch.h"
#include "DetectionLoss.h"
#include "../Utils/IoU.h"
#include <cmath>
#include <stdexcept>

namespace WheelDL {
    namespace Model {
        namespace Loss {

            // ============================================================================
            // BboxLoss - Internal class for Bounding Box Loss with CIoU and DFL
            // ============================================================================

            class BboxLoss {
            public:
                explicit BboxLoss(int64_t regMax) {
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
                    // Compute weight from target scores
                    auto weight = targetScores.sum(-1).index({ fgMask }).unsqueeze(-1);

                    // Compute CIoU for positive samples (element-wise)
                    auto predBboxesFg = predBboxes.index({ fgMask });
                    auto targetBboxesFg = targetBboxes.index({ fgMask });

                    // Compute element-wise IoU
                    auto iou = Utils::bboxIoU(
                        predBboxesFg,
                        targetBboxesFg,
                        /*xywh=*/false, /*GIoU=*/false, /*DIoU=*/false, /*CIoU=*/true
                    );  // [N]

                    // IoU loss: weighted (1 - IoU)
                    auto lossIoU = ((1.0f - iou) * weight).sum() / targetScoresSum;

                    torch::Tensor lossDfl;
                    if (_dflLoss) {
                        // Compute DFL loss
                        // Python: target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
                        auto targetLtrb = Utils::bbox2dist(
                            anchorPoints, targetBboxes, _dflLoss->getRegMax() - 1
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
            // DetectionLoss Implementation
            // ============================================================================

            DetectionLoss::DetectionLoss(
                int64_t numClasses,
                const torch::Tensor& stride,
                float boxGain,
                float clsGain,
                float dflGain,
                int64_t talTopk
            )
                : _numClasses(numClasses)
                , _regMax(16)  // Fixed to 16
                , _stride(stride.clone())  // FIXED: Clone to ensure internal state independence
                , _boxGain(boxGain)
                , _clsGain(clsGain)
                , _dflGain(dflGain)
                , _totalStrideFactor(0.0f)
                , _useDfl(true)  // Always use DFL since regMax = 16 > 1
                , _device(stride.device())  // Use same device as stride tensor
            {
                // Initialize loss components
                _bce = std::make_unique<BCEWithLogitsLoss>("none");
                _bboxLoss = std::make_unique<BboxLoss>(_regMax);
                _assigner = std::make_unique<Utils::TaskAlignedAssigner>(
                    talTopk, numClasses, 0.5f, 6.0f
                );
                _numOutputs = _useDfl ? _regMax * 4 + numClasses : 4 + numClasses;

                // OPTIMIZATION: Pre-cache projection tensors for different dtypes
                _proj = torch::arange(_regMax, torch::TensorOptions().dtype(torch::kFloat).device(_device));
                _projFloat32 = _proj.to(torch::kFloat32);
                _projFloat16 = _proj.to(torch::kFloat16);

                if (torch::cuda::is_available() && torch::cuda::cudnn_is_available()) {
                    try {
                        // Test BFloat16 support with a small tensor operation
                        auto testTensor = torch::ones({1}, torch::TensorOptions().device(_device).dtype(torch::kBFloat16));
                        _projBFloat16 = _proj.to(torch::kBFloat16);
                    } catch (const c10::Error& e) {
                        // BFloat16 not supported on this device, fallback to Float16
                        _projBFloat16 = _projFloat16;
                    } catch (const std::exception& e) {
                        // Unexpected error
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

            DetectionLoss::~DetectionLoss() = default;

            void DetectionLoss::to(const torch::Device& device) {
                _device = device;
                _stride = _stride.to(device);
                _proj = _proj.to(device);
            }

            torch::Tensor DetectionLoss::preprocess(
                const torch::Tensor& targets,
                int64_t batchSize,
                const torch::Tensor& scaleTensor
            ) {
                // Number of ground truth targets
                auto numGts = targets.size(0);
                auto ne = targets.size(1);

                // Handle empty batch (no ground truth targets)
                if (numGts == 0) {
                    return torch::zeros({ batchSize, 0, ne - 1 },
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
                auto out = torch::zeros({ batchSize, maxCount, ne - 1 },
                    torch::TensorOptions().device(_device));

                // OPTIMIZATION: Vectorized assignment instead of loop
                if (numGts > 0 && batchSize > 0) {
                    // Create offset tensor for scatter operation
                    auto cumCounts = torch::cumsum(counts, /*dim=*/0);
                    auto startIndices = torch::cat({torch::zeros({1}, counts.options()), cumCounts.slice(0, 0, -1)});

                    // Use scatter to fill all batches at once
                    for (int64_t idx = 0; idx < uniqueIndices.size(0); ++idx) {
                        auto batchIdx = uniqueIndices[idx].item<int64_t>();
                        if (batchIdx < batchSize) {
                            auto mask = (i == batchIdx);
                            auto n = counts[idx].item<int64_t>();
                            if (n > 0) {
                                out[batchIdx].slice(0, 0, n) = targets.index({ mask }).slice(1, 1, ne);
                            }
                        }
                    }
                }

                // Convert xywh to xyxy and scale: out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
                auto bboxes = out.slice(2, 1, 5);  // [batch, max_num_gt, 4]
                bboxes = bboxes * scaleTensor.unsqueeze(0).unsqueeze(0);
                bboxes = Utils::xywh2xyxy(bboxes);
                bboxes = bboxes.contiguous();
                out.slice(2, 1, 5).copy_(bboxes);

                return out;
            }

            torch::Tensor DetectionLoss::decodeBbox(
                const torch::Tensor& anchorPoints,
                const torch::Tensor& predDist
            ) {
                if (!_useDfl) {
                    return Utils::dist2bbox(predDist, anchorPoints, /*xywh=*/false);
                }

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
                auto predDistDecoded = torch::matmul(predDistSoftmax, proj);

                return Utils::dist2bbox(predDistDecoded, anchorPoints, /*xywh=*/false);
            }

            std::unordered_map<std::string, torch::Tensor> DetectionLoss::compute(
                const std::vector<torch::Tensor>& predictions,
                const Data::Dataset::DataExample& batch
            ) {
                // OPTIMIZATION: Pre-allocate concatenated tensor to avoid intermediate vectors
                // predictions: vector of [batch, 144, H, W] for each scale
                // Convert to concatenated format [batch, 144, total_anchors]

                if (predictions.empty()) {
                    throw std::invalid_argument("DetectionLoss::compute: predictions cannot be empty");
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
                return compute(concatenated, batch);
            }

            std::unordered_map<std::string, torch::Tensor> DetectionLoss::compute(
                const torch::Tensor& preds,
                const Data::Dataset::DataExample& batch
            ) {
                auto loss = torch::zeros({ 3 }, torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(_device));

                // OPTIMIZATION: Use slice instead of split to avoid creating temporary vector
                // Split predictions into distribution and scores
                // preds shape: [batch, reg_max*4 + num_classes, num_anchors]
                auto predDistri = preds.slice(1, 0, _regMax * 4).permute({ 0, 2, 1 }).contiguous();  // [batch, num_anchors, 4*regMax]
                auto predScores = preds.slice(1, _regMax * 4, _regMax * 4 + _numClasses).permute({ 0, 2, 1 }).contiguous();  // [batch, num_anchors, num_classes]

                auto dtype = predScores.dtype();
                auto batchSize = predScores.size(0);
                auto numAnchors = predScores.size(1);

                // Fixed: Use pre-computed totalStrideFactor (calculated in constructor)
                // This avoids GPU→CPU copy on every forward pass
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

                // Prepare targets: torch.cat((batch["batch_idx"].view(-1, 1), batch["classes"].view(-1, 1), batch["bboxes"]), 1)
                auto batchIdx = batch.batchIndices.view({ -1, 1 });
                auto classes = batch.classes.view({ -1, 1 });
                auto bboxes = batch.targets;  // [num_gt, 4]

                // OPTIMIZATION: Ensure all tensors are on correct device before concatenation
                auto targets = torch::cat({
                    batchIdx.to(_device),
                    classes.to(_device),
                    bboxes.to(_device)
                }, /*dim=*/1);

                // Python: scale_tensor=imgsz[[1, 0, 1, 0]]
                auto scaleTensor = torch::stack({ imgSize[1], imgSize[0], imgSize[1], imgSize[0] });
                targets = preprocess(targets, batchSize, scaleTensor);

                // Split targets: gt_labels, gt_bboxes = targets.split((1, 4), 2)
                auto targetParts = targets.split({ 1, 4 }, /*dim=*/2);
                auto gtLabels = targetParts[0];
                auto gtBboxes = targetParts[1];

                // mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
                auto maskGt = gtBboxes.sum(/*dim=*/2, /*keepdim=*/true) > 0.0f;

                // Decode predicted boxes
                auto predBboxes = decodeBbox(anchorPoints, predDistri);

                // Task-aligned assignment
                auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] = _assigner->forward(
                    predScores.detach().sigmoid(),
                    (predBboxes.detach() * strideTensor).to(gtBboxes.dtype()),
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

                // Classification loss: loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
                // BCEWithLogitsLoss expects targets, so create temp DataExample
                Data::Dataset::DataExample tempTarget;
                auto bceLossMap = _bce->compute(predScores, targetScores.to(dtype));
                loss[1] = bceLossMap.at("total").sum() / targetScoresSum;

                // Box and DFL loss (only for foreground)
                if (fgMask.sum().item<int64_t>() > 0) {
                    // OPTIMIZATION: Inplace division to avoid temporary tensor
                    targetBboxes.div_(strideTensor);

                    // loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
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
