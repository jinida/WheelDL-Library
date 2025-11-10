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

                    // Compute CIoU for positive samples
                    auto iou = Utils::bboxIoU(
                        predBboxes.index({ fgMask }),
                        targetBboxes.index({ fgMask }),
                        /*xywh=*/false, /*GIoU=*/false, /*DIoU=*/false, /*CIoU=*/true
                    );

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
                , _stride(stride.clone())
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
                // Initialize projection tensor for DFL decoding
                _proj = torch::arange(_regMax, torch::TensorOptions().dtype(torch::kFloat));

                // Fixed: Pre-compute totalStrideFactor once to avoid GPU→CPU copy on every forward
                auto strideCpu = _stride.cpu();
                auto strideAccessor = strideCpu.accessor<float, 1>();
                for (int64_t i = 0; i < strideCpu.size(0); ++i) {
                    auto s = strideAccessor[i];
                    _totalStrideFactor += 1.0f / (s * s);
                }
            }

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

                // Get image indices and count per image
                // Python: _, counts = i.unique(return_counts=True)
                auto i = targets.select(1, 0);
                auto uniqueResult = torch::_unique2(i, /*sorted=*/false, /*return_inverse=*/false, /*return_counts=*/true);
                auto counts = std::get<2>(uniqueResult).to(torch::kInt32);

                auto maxCount = counts.max().item<int64_t>();
                auto out = torch::zeros({ batchSize, maxCount, ne - 1 },
                    torch::TensorOptions().device(_device));

                // Fill targets for each image
                for (int64_t j = 0; j < batchSize; ++j) {
                    auto matches = (i == j);
                    auto n = matches.sum().item<int64_t>();

                    if (n > 0) {
                        out[j].slice(0, 0, n) = targets.index({ matches }).slice(1, 1, ne);
                    }
                }

                // Convert xywh to xyxy and scale: out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
                auto bboxes = out.slice(2, 1, 5);  // [batch, max_num_gt, 4]
                bboxes = bboxes * scaleTensor.unsqueeze(0).unsqueeze(0);
                bboxes = Utils::xywh2xyxy(bboxes);
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

                // DFL decoding
                auto b = predDist.size(0);
                auto a = predDist.size(1);
                auto c = predDist.size(2);

                // Reshape and apply softmax
                auto predDistReshaped = predDist.view({ b, a, 4, c / 4 });
                auto predDistSoftmax = torch::softmax(predDistReshaped, /*dim=*/3);

                // Matrix multiply with projection
                auto predDistDecoded = torch::matmul(predDistSoftmax, _proj.to(predDist.dtype()));

                return Utils::dist2bbox(predDistDecoded, anchorPoints, /*xywh=*/false);
            }

            std::unordered_map<std::string, torch::Tensor> DetectionLoss::compute(
                const std::vector<torch::Tensor>& predictions,
                const Data::Dataset::DataExample& batch
            ) {
                // Multi-scale version for training
                // predictions: vector of [batch, 144, H, W] for each scale
                // Convert to concatenated format [batch, 144, total_anchors]

                std::vector<torch::Tensor> flattenedPreds;
                flattenedPreds.reserve(predictions.size());

                for (const auto& pred : predictions) {
                    // pred shape: [batch, 144, H, W]
                    auto batch_size = pred.size(0);
                    auto channels = pred.size(1);
                    auto height = pred.size(2);
                    auto width = pred.size(3);

                    // Flatten spatial dimensions: [batch, 144, H*W]
                    auto flattened = pred.view({batch_size, channels, height * width});
                    flattenedPreds.push_back(flattened);
                }

                // Concatenate all scales: [batch, 144, total_anchors]
                auto concatenated = torch::cat(flattenedPreds, /*dim=*/2);

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

                // Split predictions into distribution and scores
                // preds shape: [batch, reg_max*4 + num_classes, num_anchors]
                // Use structured bindings with std::tie for cleaner code (C++17)
                auto parts = preds.split({ _regMax * 4, _numClasses }, /*dim=*/1);
                auto predDistri = parts[0].permute({ 0, 2, 1 }).contiguous();  // [batch, num_anchors, 4*regMax]
                auto predScores = parts[1].permute({ 0, 2, 1 }).contiguous();  // [batch, num_anchors, num_classes]

                auto dtype = predScores.dtype();
                auto batchSize = predScores.size(0);
                auto numAnchors = predScores.size(1);

                // Fixed: Use pre-computed totalStrideFactor (calculated in constructor)
                // This avoids GPU→CPU copy on every forward pass
                auto imgSizeFloat = std::sqrt(static_cast<float>(numAnchors) / _totalStrideFactor);
                auto imgSizeInt = static_cast<int64_t>(std::round(imgSizeFloat));

                auto imgSize = torch::tensor({ static_cast<float>(imgSizeInt), static_cast<float>(imgSizeInt) },
                    torch::TensorOptions().dtype(dtype).device(_device));

                // Calculate feature shapes without allocating tensors (memory efficient)
                // Use CPU copy only for stride values (small one-time cost)
                auto strideCpu = _stride.cpu();
                auto strideAccessor = strideCpu.accessor<float, 1>();
                std::vector<std::pair<int64_t, int64_t>> featShapes;
                featShapes.reserve(strideCpu.size(0));
                for (int64_t i = 0; i < strideCpu.size(0); ++i)
                {
                    auto strideVal = static_cast<int64_t>(strideAccessor[i]);
                    auto featH = imgSizeInt / strideVal;
                    auto featW = imgSizeInt / strideVal;
                    featShapes.emplace_back(featH, featW);
                }

                // Generate anchors using efficient shape-based method
                auto [anchorPoints, strideTensor] = Utils::makeAnchors(featShapes, _stride, dtype.toScalarType(), _device, 0.5f);

                // Prepare targets: torch.cat((batch["batch_idx"].view(-1, 1), batch["classes"].view(-1, 1), batch["bboxes"]), 1)
                auto batchIdx = batch.batchIndices.view({ -1, 1 });
                auto classes = batch.classes.view({ -1, 1 });
                auto bboxes = batch.targets;  // [num_gt, 4]

                auto targets = torch::cat({ batchIdx, classes, bboxes }, /*dim=*/1);

                // Python: scale_tensor=imgsz[[1, 0, 1, 0]]
                // Ensure scaleTensor is on the same device as targets
                auto scaleTensor = torch::stack({ imgSize[1], imgSize[0], imgSize[1], imgSize[0] })
                    .to(_device);
                targets = preprocess(targets.to(_device), batchSize, scaleTensor);

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
                tempTarget.targets = targetScores.to(dtype);
                auto bceLossMap = _bce->compute(predScores, tempTarget);
                loss[1] = bceLossMap.at("total").sum() / targetScoresSum;

                // Box and DFL loss (only for foreground)
                if (fgMask.sum().item<int64_t>() > 0) {
                    // target_bboxes /= stride_tensor
                    targetBboxes = targetBboxes / strideTensor;

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

                // Apply loss weights
                loss[0] *= _boxGain;
                loss[1] *= _clsGain;
                loss[2] *= _dflGain;

                // Multiply by batch_size (same as Python: return loss * batch_size, loss.detach())
                loss *= static_cast<float>(batchSize);

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
