#include "pch.h"
#include "TaskAlignedAssigner.h"
#include "IoU.h"
#include <stdexcept>
#include <algorithm>

namespace WheelDL {
namespace Model {
namespace Utils {

namespace {
    /**
     * @brief Check if a PyTorch C++ exception is a CUDA out-of-memory error
     *
     * @param e The caught c10::Error exception
     * @return true if the error is a CUDA OOM error
     *
     * @note This function checks multiple indicators:
     *       1. Exception message contains OOM-related keywords
     *       2. Error type is OutOfMemoryError (if available)
     */
    bool isCudaOutOfMemoryError(const c10::Error& e) {
        // Method 1: Check error message for OOM patterns
        std::string errorMsg(e.what());

        // Convert to lowercase for case-insensitive matching
        std::transform(errorMsg.begin(), errorMsg.end(), errorMsg.begin(),
                      [](unsigned char c) { return std::tolower(c); });

        // Check for common OOM patterns
        bool hasOOMKeyword = (errorMsg.find("out of memory") != std::string::npos) ||
                             (errorMsg.find("oom") != std::string::npos) ||
                             (errorMsg.find("cuda") != std::string::npos &&
                              errorMsg.find("memory") != std::string::npos);

        // Method 2: Check exception type name (more robust)
        // PyTorch C++ exceptions include type information in what() string
        bool hasOutOfMemoryType = (errorMsg.find("outofmemoryerror") != std::string::npos);

        return hasOOMKeyword || hasOutOfMemoryType;
    }
} // anonymous namespace

TaskAlignedAssigner::TaskAlignedAssigner(
    int64_t topk,
    int64_t numClasses,
    float alpha,
    float beta,
    float eps
)
    : _topk(topk)
    , _numClasses(numClasses)
    , _bgIdx(numClasses)
    , _alpha(alpha)
    , _beta(beta)
    , _eps(eps)
    , _bs(0)
    , _nMaxBoxes(0) {
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TaskAlignedAssigner::forward(
    const torch::Tensor& pdScores,
    const torch::Tensor& pdBboxes,
    const torch::Tensor& ancPoints,
    const torch::Tensor& gtLabels,
    const torch::Tensor& gtBboxes,
    const torch::Tensor& maskGt
) {
    // Python: self.bs = pd_scores.shape[0]
    _bs = pdScores.size(0);
    // Python: self.n_max_boxes = gt_bboxes.shape[1]
    _nMaxBoxes = gtBboxes.size(1);
    auto device = gtBboxes.device();

    // Python: if self.n_max_boxes == 0: return (...)
    if (_nMaxBoxes == 0) {
        return std::make_tuple(
            torch::full_like(pdScores.select(2, 0), _bgIdx),
            torch::zeros_like(pdBboxes),
            torch::zeros_like(pdScores),
            torch::zeros_like(pdScores.select(2, 0)).to(torch::kBool),
            torch::zeros_like(pdScores.select(2, 0))
        );
    }

    // Python: try: return self._forward(...) except torch.cuda.OutOfMemoryError: ...
    try {
        // Python: mask_pos, align_metric, overlaps = self.get_pos_mask(...)
        auto [maskPos, alignMetric, overlaps] = getPosMask(
            pdScores, pdBboxes, gtLabels, gtBboxes, ancPoints, maskGt
        );

        // Python: target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(...)
        auto [targetGtIdx, fgMask, maskPosUpdated] = selectHighestOverlaps(
            maskPos, overlaps, _nMaxBoxes
        );

        // Python: target_labels, target_bboxes, target_scores = self.get_targets(...)
        auto [targetLabels, targetBboxes, targetScores] = getTargets(
            gtLabels, gtBboxes, targetGtIdx, fgMask
        );

        // OPTIMIZATION: Use inplace operations to reduce memory allocations
        // Python: align_metric *= mask_pos
        alignMetric.mul_(maskPosUpdated);

        // Python: pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        auto posAlignMetrics = alignMetric.amax(/*dim=*/-1, /*keepdim=*/true);

        // Python: pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        auto posOverlaps = overlaps.mul(maskPosUpdated).amax(/*dim=*/-1, /*keepdim=*/true);

        // Python: norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        auto normAlignMetric = alignMetric.mul(posOverlaps)
                                .div_(posAlignMetrics + _eps)
                                .amax(/*dim=*/-2).unsqueeze(-1);

        // Python: target_scores = target_scores * norm_align_metric
        // Note: Cannot use inplace mul_ because targetScores is int64 and normAlignMetric is float
        targetScores = targetScores * normAlignMetric;

        // Python: return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
        return std::make_tuple(
            targetLabels,
            targetBboxes,
            targetScores,
            fgMask.to(torch::kBool),
            targetGtIdx
        );

    } catch (const c10::Error& e) {
        // Handle CUDA out of memory errors by falling back to CPU computation
        if (!isCudaOutOfMemoryError(e)) {
            throw;
        }

        // Python: cpu_tensors = [t.cpu() for t in (...)]
        auto cpuTensors = std::vector<torch::Tensor>{
            pdScores.cpu(), pdBboxes.cpu(), ancPoints.cpu(),
            gtLabels.cpu(), gtBboxes.cpu(), maskGt.cpu()
        };

        // Python: result = self._forward(*cpu_tensors)
        auto result = forward(
            cpuTensors[0], cpuTensors[1], cpuTensors[2],
            cpuTensors[3], cpuTensors[4], cpuTensors[5]
        );

        // Python: return tuple(t.to(device) for t in result)
        return std::make_tuple(
            std::get<0>(result).to(device),
            std::get<1>(result).to(device),
            std::get<2>(result).to(device),
            std::get<3>(result).to(device),
            std::get<4>(result).to(device)
        );
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
TaskAlignedAssigner::getPosMask(
    const torch::Tensor& pdScores,
    const torch::Tensor& pdBboxes,
    const torch::Tensor& gtLabels,
    const torch::Tensor& gtBboxes,
    const torch::Tensor& ancPoints,
    const torch::Tensor& maskGt
) {
    // Python: mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
    auto maskInGts = selectCandidatesInGts(ancPoints, gtBboxes);

    // Python: align_metric, overlaps = self.get_box_metrics(...)
    auto [alignMetric, overlaps] = getBoxMetrics(
        pdScores, pdBboxes, gtLabels, gtBboxes, maskInGts * maskGt
    );

    // Python: mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
    auto maskTopk = selectTopkCandidates(
        alignMetric,
        /*largest=*/true,
        maskGt.expand({-1, -1, _topk}).to(torch::kBool)
    );

    // Python: mask_pos = mask_topk * mask_in_gts * mask_gt
    auto maskPos = maskTopk * maskInGts * maskGt;

    return std::make_tuple(maskPos, alignMetric, overlaps);
}

std::tuple<torch::Tensor, torch::Tensor>
TaskAlignedAssigner::getBoxMetrics(
    const torch::Tensor& pdScores,
    const torch::Tensor& pdBboxes,
    const torch::Tensor& gtLabels,
    const torch::Tensor& gtBboxes,
    const torch::Tensor& maskGt
) {
    // OPTIMIZATION: Memory-efficient implementation processing only valid GTs
    auto na = pdBboxes.size(1);
    auto maskGtBool = maskGt.to(torch::kBool);

    // Early exit if no valid GT boxes
    if (!maskGtBool.any().item<bool>()) {
        auto zeros = torch::zeros(
            {_bs, _nMaxBoxes, na},
            torch::TensorOptions().dtype(pdBboxes.dtype()).device(pdBboxes.device())
        );
        return std::make_tuple(zeros.clone(), zeros);
    }

    // Allocate output tensors
    auto overlaps = torch::zeros(
        {_bs, _nMaxBoxes, na},
        torch::TensorOptions().dtype(pdBboxes.dtype()).device(pdBboxes.device())
    );
    auto bboxScores = torch::zeros(
        {_bs, _nMaxBoxes, na},
        torch::TensorOptions().dtype(pdScores.dtype()).device(pdScores.device())
    );

    // Process each batch separately to reduce memory footprint
    for (int64_t b = 0; b < _bs; ++b) {
        auto batchMask = maskGtBool[b];  // [n_max_boxes, na]
        auto validGtMask = batchMask.any(1);  // [n_max_boxes]

        if (!validGtMask.any().item<bool>()) {
            continue;  // Skip this batch if no valid GTs
        }

        // Get valid GT indices
        auto validGtIndices = torch::nonzero(validGtMask).squeeze(1);
        auto nValid = validGtIndices.size(0);

        // Extract valid GTs for this batch
        auto validGtLabels = gtLabels[b].index({validGtIndices}).squeeze(-1);  // [n_valid]
        auto validGtBboxes = gtBboxes[b].index({validGtIndices});  // [n_valid, bbox_dim]

        // Gather classification scores for valid GTs
        // pd_scores[b]: [na, num_classes], validGtLabels: [n_valid]
        auto pdScoresBatch = pdScores[b];  // [na, num_classes]
        auto labelIndices = validGtLabels.to(torch::kLong);

        // Index into scores: [na, num_classes] -> [na, n_valid]
        auto scoresForGts = pdScoresBatch.index({
            torch::indexing::Slice(),
            labelIndices
        });  // [na, n_valid]

        // Transpose to [n_valid, na] to match output format
        bboxScores[b].index_put_({validGtIndices}, scoresForGts.t());

        // Compute IoU only for valid GTs
        // Get bbox dimension (4 for regular bbox, 5 for OBB)
        auto bboxDim = gtBboxes.size(-1);
        auto pdBboxesBatch = pdBboxes[b];  // [na, bbox_dim]

        // OPTIMIZATION: Broadcasting with expand (lazy, no memory allocation)
        // Compute IoU between each valid GT and all predictions
        auto pdExpanded = pdBboxesBatch.unsqueeze(0).expand({nValid, -1, -1});  // [n_valid, na, bbox_dim]
        auto gtExpanded = validGtBboxes.unsqueeze(1).expand({-1, na, -1});      // [n_valid, na, bbox_dim]

        // OPTIMIZATION: Use contiguous().view() for explicit control
        // expand creates non-contiguous views, so we need contiguous() before reshape
        auto iou = computeIou(
            gtExpanded.contiguous().view({-1, bboxDim}),
            pdExpanded.contiguous().view({-1, bboxDim})
        ).view({nValid, na});

        // Store IoU values
        overlaps[b].index_put_({validGtIndices}, iou);
    }

    // OPTIMIZATION: Compute alignment metric with fully in-place operations
    auto alignMetric = bboxScores.pow(_alpha);
    alignMetric.mul_(overlaps.pow(_beta));

    return std::make_tuple(alignMetric, overlaps);
}

torch::Tensor TaskAlignedAssigner::computeIou(
    const torch::Tensor& gtBboxes,
    const torch::Tensor& pdBboxes
) const {
    // Python: return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
    return bboxIoU(gtBboxes, pdBboxes, /*xywh=*/false, /*GIoU=*/false,
                   /*DIoU=*/false, /*CIoU=*/true).squeeze(-1).clamp(0.0f);
}

torch::Tensor TaskAlignedAssigner::selectTopkCandidates(
    const torch::Tensor& metrics,
    bool largest,
    const torch::Tensor& topkMask
) {
    // Python implementation:
    // topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
    // if topk_mask is None:
    //     topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
    // topk_idxs.masked_fill_(~topk_mask, 0)
    // count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
    // ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
    // for k in range(self.topk):
    //     count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
    // count_tensor.masked_fill_(count_tensor > 1, 0)
    // return count_tensor.to(metrics.dtype)

    auto [topkMetrics, topkIdxs] = torch::topk(metrics, _topk, /*dim=*/-1, largest);

    torch::Tensor mask;
    if (topkMask.defined()) {
        mask = topkMask;
    } else {
        auto [maxValues, maxIndices] = topkMetrics.max(/*dim=*/-1, /*keepdim=*/true);
        mask = (maxValues > _eps).expand_as(topkIdxs);
    }

    // Inplace mask_fill to avoid copy
    topkIdxs.masked_fill_(~mask, 0);

    // Pre-allocate count tensor
    auto countTensor = torch::zeros(
        metrics.sizes(),
        torch::TensorOptions().dtype(torch::kInt8).device(topkIdxs.device())
    );

    // Python uses a loop: for k in range(self.topk):
    //     count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
    auto ones = torch::ones_like(topkIdxs.slice(/*dim=*/2, /*start=*/0, /*end=*/1), torch::kInt8);
    for (int64_t k = 0; k < _topk; ++k) {
        countTensor.scatter_add_(-1, topkIdxs.slice(/*dim=*/2, /*start=*/k, /*end=*/k + 1), ones);
    }

    // Inplace masked_fill to avoid temporary
    countTensor.masked_fill_(countTensor > 1, 0);
    return countTensor.to(metrics.dtype());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
TaskAlignedAssigner::getTargets(
    const torch::Tensor& gtLabels,
    const torch::Tensor& gtBboxes,
    const torch::Tensor& targetGtIdx,
    const torch::Tensor& fgMask
) {
    // Python: batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
    auto batchInd = torch::arange(_bs, torch::TensorOptions()
                                  .dtype(torch::kLong)
                                  .device(gtLabels.device())).unsqueeze(-1);

    // Python: target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
    auto targetGtIdxOffset = targetGtIdx + batchInd * _nMaxBoxes;

    // Python: target_labels = gt_labels.long().flatten()[target_gt_idx]
    auto targetLabels = gtLabels.to(torch::kLong).flatten().index({targetGtIdxOffset});

    // Python: target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
    auto targetBboxes = gtBboxes.view({-1, gtBboxes.size(-1)}).index({targetGtIdxOffset});

    // OPTIMIZATION: Inplace clamp to avoid copy
    // Python: target_labels.clamp_(0)
    targetLabels.clamp_(0);

    // Python: target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes), dtype=torch.int64, device=target_labels.device)
    auto targetScores = torch::zeros(
        {targetLabels.size(0), targetLabels.size(1), _numClasses},
        torch::TensorOptions().dtype(torch::kLong).device(targetLabels.device())
    );

    // Python: target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
    targetScores.scatter_(2, targetLabels.unsqueeze(-1), 1);

    // OPTIMIZATION: Inplace masking to avoid temporary tensor
    // Python: fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
    auto fgScoresMask = fgMask.unsqueeze(-1).expand({-1, -1, _numClasses});

    // Python: target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
    targetScores.masked_fill_(fgScoresMask <= 0, 0);

    return std::make_tuple(targetLabels, targetBboxes, targetScores);
}

torch::Tensor TaskAlignedAssigner::selectCandidatesInGts(
    const torch::Tensor& xyCenters,
    const torch::Tensor& gtBboxes,
    float eps
) {
    // Python: @staticmethod
    // Python: def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    // Python:     n_anchors = xy_centers.shape[0]
    // Python:     bs, n_boxes, _ = gt_bboxes.shape
    auto nAnchors = xyCenters.size(0);
    auto bs = gtBboxes.size(0);
    auto nBoxes = gtBboxes.size(1);

    // Python: lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
    auto gtBboxesReshaped = gtBboxes.view({-1, 1, 4});
    auto chunks = gtBboxesReshaped.chunk(2, /*dim=*/2);
    auto lt = chunks[0];
    auto rb = chunks[1];

    // Python: bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
    auto bboxDeltas = torch::cat(
        {xyCenters.unsqueeze(0) - lt, rb - xyCenters.unsqueeze(0)},
        /*dim=*/2
    ).view({bs, nBoxes, nAnchors, -1});

    // Python: return bbox_deltas.amin(3).gt_(eps)
    return bboxDeltas.amin(/*dim=*/3) > eps;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
TaskAlignedAssigner::selectHighestOverlaps(
    const torch::Tensor& maskPos,
    const torch::Tensor& overlaps,
    int64_t nMaxBoxes
) {
    // Python: @staticmethod
    // Python: def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    // Python:     fg_mask = mask_pos.sum(-2)
    auto fgMask = maskPos.sum(/*dim=*/-2);

    // Python:     if fg_mask.max() > 1:
    if (fgMask.max().item<float>() > 1.0f) {
        // Python:         mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
        auto maskMultiGts = (fgMask.unsqueeze(1) > 1).expand({-1, nMaxBoxes, -1});

        // Python:         max_overlaps_idx = overlaps.argmax(1)
        auto maxOverlapsIdx = overlaps.argmax(/*dim=*/1);

        // Python:         is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        auto isMaxOverlaps = torch::zeros_like(maskPos);

        // Python:         is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
        isMaxOverlaps.scatter_(1, maxOverlapsIdx.unsqueeze(1), 1);

        // Python:         mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
        auto maskPosUpdated = torch::where(maskMultiGts, isMaxOverlaps, maskPos).to(maskPos.dtype());

        // Python:         fg_mask = mask_pos.sum(-2)
        fgMask = maskPosUpdated.sum(/*dim=*/-2);

        // Python:     target_gt_idx = mask_pos.argmax(-2)
        auto targetGtIdx = maskPosUpdated.argmax(/*dim=*/-2);

        return std::make_tuple(targetGtIdx, fgMask, maskPosUpdated);
    }

    // Python:     target_gt_idx = mask_pos.argmax(-2)
    auto targetGtIdx = maskPos.argmax(/*dim=*/-2);

    // Python:     return target_gt_idx, fg_mask, mask_pos
    return std::make_tuple(targetGtIdx, fgMask, maskPos);
}

// ============================================================================
// OBBTaskAlignedAssigner Implementation
// ============================================================================

torch::Tensor OBBTaskAlignedAssigner::computeIou(
    const torch::Tensor& gtBboxes,
    const torch::Tensor& pdBboxes
) const {
    // Python: def iou_calculation(self, gt_bboxes, pd_bboxes):
    // Python:     return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)
    return probiou(gtBboxes, pdBboxes).squeeze(-1).clamp(0.0f);
}

torch::Tensor OBBTaskAlignedAssigner::selectCandidatesInGts(
    const torch::Tensor& xyCenters,
    const torch::Tensor& gtBboxes,
    float eps  // Unused - kept for interface compatibility
) {
    // Python: @staticmethod
    // Python: def select_candidates_in_gts(xy_centers, gt_bboxes):
    // Python:     corners = xywhr2xyxyxyxy(gt_bboxes)
    (void)eps;  // Suppress unused parameter warning
    auto corners = xywhr2xyxyxyxy(gtBboxes);  // [batch, max_num_gt, 4, 2]

    // Python:     a, b, _, d = corners.split(1, dim=-2)
    auto cornersSplit = corners.split(1, /*dim=*/-2);
    auto a = cornersSplit[0];  // [batch, max_num_gt, 1, 2]
    auto b = cornersSplit[1];
    auto d = cornersSplit[3];

    // Python:     ab = b - a
    auto ab = b - a;

    // Python:     ad = d - a
    auto ad = d - a;

    // Python:     ap = xy_centers - a
    // Note: xy_centers is [num_anchors, 2], a is [batch, max_num_gt, 1, 2]
    auto ap = xyCenters.unsqueeze(0).unsqueeze(0) - a;  // [batch, max_num_gt, num_anchors, 2]

    // Python:     norm_ab = (ab * ab).sum(dim=-1)
    auto normAb = (ab * ab).sum(/*dim=*/-1);  // [batch, max_num_gt, 1]

    // Python:     norm_ad = (ad * ad).sum(dim=-1)
    auto normAd = (ad * ad).sum(/*dim=*/-1);  // [batch, max_num_gt, 1]

    // Python:     ap_dot_ab = (ap * ab).sum(dim=-1)
    auto apDotAb = (ap * ab).sum(/*dim=*/-1);  // [batch, max_num_gt, num_anchors]

    // Python:     ap_dot_ad = (ap * ad).sum(dim=-1)
    auto apDotAd = (ap * ad).sum(/*dim=*/-1);  // [batch, max_num_gt, num_anchors]

    // Python:     return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)
    return (apDotAb >= 0) & (apDotAb <= normAb) & (apDotAd >= 0) & (apDotAd <= normAd);
}

} // namespace Utils
} // namespace Model
} // namespace WheelDL
