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

                    // Compute Probiou for oriented boxes
                    auto iou = Utils::probiou(
                        predBboxes.index({ fgMask }),
                        targetBboxes.index({ fgMask })
                    );

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
                , _stride(stride.clone())
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

                // Get image indices and count per image
                // Python: _, counts = i.unique(return_counts=True)
                auto i = targets.select(1, 0);
                auto uniqueResult = torch::_unique2(i, /*sorted=*/false, /*return_inverse=*/false, /*return_counts=*/true);
                auto counts = std::get<2>(uniqueResult).to(torch::kInt32);

                auto maxCount = counts.max().item<int64_t>();
                auto out = torch::zeros({ batchSize, maxCount, 6 },
                    torch::TensorOptions().device(_device));

                // Fill targets for each image
                for (int64_t j = 0; j < batchSize; ++j) {
                    auto matches = (i == j);
                    auto n = matches.sum().item<int64_t>();

                    if (n > 0) {
                        auto bboxes = targets.index({ matches }).slice(1, 2, 7);  // [n, 5] - xywh + angle

                        // Scale bbox coordinates (not angle)
                        auto bboxesXywh = bboxes.slice(1, 0, 4);  // [n, 4]
                        bboxesXywh = bboxesXywh * scaleTensor.unsqueeze(0);

                        auto angle = bboxes.slice(1, 4, 5);  // [n, 1]
                        auto scaledBboxes = torch::cat({ bboxesXywh, angle }, /*dim=*/1);  // [n, 5]

                        auto classLabels = targets.index({ matches }).slice(1, 1, 2);  // [n, 1]
                        out[j].slice(0, 0, n) = torch::cat({ classLabels, scaledBboxes }, /*dim=*/1);
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
                    auto b = predDist.size(0);
                    auto a = predDist.size(1);
                    auto c = predDist.size(2);

                    // Reshape and apply softmax
                    auto predDistReshaped = predDist.view({ b, a, 4, c / 4 });
                    auto predDistSoftmax = torch::softmax(predDistReshaped, /*dim=*/3);

                    // Matrix multiply with projection
                    predDistDecoded = torch::matmul(predDistSoftmax, _proj.to(predDist.dtype()));
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
                // Multi-scale version for training
                // predictions: vector of [batch, 145, H, W] for each scale
                // where 145 = 4*regMax + numClasses + 1 (for angle)
                // Convert to concatenated format [batch, 145, total_anchors]

                std::vector<torch::Tensor> flattenedPreds;
                flattenedPreds.reserve(predictions.size());

                for (const auto& pred : predictions) {
                    // pred shape: [batch, 145, H, W]
                    auto batch_size = pred.size(0);
                    auto channels = pred.size(1);
                    auto height = pred.size(2);
                    auto width = pred.size(3);

                    // Flatten spatial dimensions: [batch, 145, H*W]
                    auto flattened = pred.view({batch_size, channels, height * width});
                    flattenedPreds.push_back(flattened);
                }

                // Concatenate all scales: [batch, 145, total_anchors]
                auto concatenated = torch::cat(flattenedPreds, /*dim=*/2);

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

                // Split predictions: [batch, reg_max*4 + num_classes + 1, num_anchors]
                // Last channel is angle
                auto predAngle = prediction.slice(1, _regMax * 4 + _numClasses, _regMax * 4 + _numClasses + 1);
                predAngle = predAngle.permute({ 0, 2, 1 }).contiguous();

                // Split predictions: cleaner code without lambda (C++17)
                auto parts = prediction.slice(1, 0, _regMax * 4 + _numClasses).split({ _regMax * 4, _numClasses }, /*dim=*/1);
                auto predDistri = parts[0].permute({ 0, 2, 1 }).contiguous();
                auto predScores = parts[1].permute({ 0, 2, 1 }).contiguous();

                auto dtype = predScores.dtype();
                auto batchSize = predScores.size(0);
                auto numAnchors = predScores.size(1);

                // Calculate image size from number of anchors
                // For strides [8, 16, 32] and square image:
                // num_anchors = (H/8)*(W/8) + (H/16)*(W/16) + (H/32)*(W/32)
                // For square: num_anchors = H^2 * (1/64 + 1/256 + 1/1024)
                // Solve for image size
                // Fixed: Use pre-computed totalStrideFactor (calculated in constructor)
                // This avoids GPU→CPU copy on every forward pass
                auto imgSizeFloat = std::sqrt(static_cast<float>(numAnchors) / _totalStrideFactor);
                auto imgSizeInt = static_cast<int64_t>(std::round(imgSizeFloat));

                // Python: imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
                // This gives [height, width] in pixels
                auto imgSize = torch::tensor({ static_cast<float>(imgSizeInt), static_cast<float>(imgSizeInt) },
                    torch::TensorOptions().dtype(dtype).device(_device));

                // Calculate feature shapes without allocating tensors (memory efficient)
                // Use CPU copy only for stride values (small one-time cost)
                auto strideCpu = _stride.cpu();
                auto strideAccessor = strideCpu.accessor<float, 1>();
                std::vector<std::pair<int64_t, int64_t>> featShapes;
                featShapes.reserve(strideCpu.size(0));
                for (int64_t i = 0; i < strideCpu.size(0); ++i) {
                    auto strideVal = static_cast<int64_t>(strideAccessor[i]);
                    auto featH = imgSizeInt / strideVal;
                    auto featW = imgSizeInt / strideVal;
                    featShapes.emplace_back(featH, featW);
                }

                // Generate anchors using efficient shape-based method
                auto [anchorPoints, strideTensor] = Utils::makeAnchors(featShapes, _stride, dtype.toScalarType(), _device, 0.5f);

                // Prepare targets: torch.cat((batch_idx, classes, obb), 1) where obb is [x, y, w, h, angle]
                auto batchIdx = target.batchIndices.view({ -1, 1 });
                auto classes = target.classes.view({ -1, 1 });
                auto obb = target.targets;  // [num_gt, 5] - xywh + angle

                auto targets = torch::cat({ batchIdx, classes, obb }, /*dim=*/1);

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
                // Ensure scaleTensor is on the same device as targets
                auto scaleTensor = torch::stack({ imgSize[1], imgSize[0], imgSize[1], imgSize[0] })
                    .to(_device);
                targets = preprocess(targets.to(_device), batchSize, scaleTensor);

                // Split targets: gt_labels, gt_bboxes = targets.split((1, 5), 2)
                auto targetParts = targets.split({ 1, 5 }, /*dim=*/2);
                auto gtLabels = targetParts[0];
                auto gtBboxes = targetParts[1];  // gtBboxes: [batch, max_num_gt, 5] - xywh + angle

                // mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
                auto maskGt = gtBboxes.sum(/*dim=*/2, /*keepdim=*/true) > 0.0f;

                // Decode predicted boxes
                auto predBboxes = decodeBbox(anchorPoints, predDistri, predAngle);

                // Clone and scale for assigner
                auto bboxesForAssigner = predBboxes.clone().detach();
                bboxesForAssigner.slice(2, 0, 4) *= strideTensor;  // Only scale xywh, not angle

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

                // Classification loss
                Data::Dataset::DataExample tempTarget;
                tempTarget.targets = targetScores.to(dtype);
                auto bceLossMap = _bce->compute(predScores, tempTarget);
                loss[1] = bceLossMap.at("total").sum() / targetScoresSum;

                // Box and DFL loss (only for foreground)
                if (fgMask.sum().item<int64_t>() > 0) {
                    // target_bboxes[..., :4] /= stride_tensor
                    targetBboxes.slice(2, 0, 4) /= strideTensor;

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

                // Apply loss weights
                loss[0] *= _boxGain;
                loss[1] *= _clsGain;
                loss[2] *= _dflGain;

                // Multiply by batch_size
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
