#pragma once

#include <torch/torch.h>

namespace WheelDL {
namespace Model {
namespace Utils {

// C++17 inline constexpr constants - can be defined in header without ODR violations
inline constexpr float PI = 3.14159265358979323846f;
inline constexpr float HEIGHT_MIN_THRESHOLD = 1e-5f;

/**
 * @brief Compute element-wise Intersection over Union (IoU) for bounding boxes
 *
 * This function computes IoU between corresponding boxes (box1[i] with box2[i]).
 * This is the standard behavior matching Python implementation.
 *
 * Supports multiple IoU variants:
 * - Standard IoU: Intersection / Union
 * - GIoU (Generalized IoU): Considers smallest enclosing box
 * - DIoU (Distance IoU): Adds penalty for center distance
 * - CIoU (Complete IoU): Adds aspect ratio consistency
 *
 * @param box1 First set of boxes [N, 4] in format specified by xywh
 * @param box2 Second set of boxes [N, 4] in format specified by xywh (must match box1 size)
 * @param xywh If true, boxes are [x_center, y_center, width, height]
 *             If false, boxes are [x1, y1, x2, y2]
 * @param GIoU If true, compute Generalized IoU
 * @param DIoU If true, compute Distance IoU
 * @param CIoU If true, compute Complete IoU
 * @param eps Small value for numerical stability
 * @return IoU vector [N] - element-wise IoU values
 */
[[nodiscard]] torch::Tensor bboxIoU(
    const torch::Tensor& box1,
    const torch::Tensor& box2,
    bool xywh = true,
    bool GIoU = false,
    bool DIoU = false,
    bool CIoU = false,
    float eps = 1e-6f
);

/**
 * @brief Compute element-wise Probiou (Probabilistic IoU) for oriented bounding boxes
 *
 * This function computes Probiou between corresponding OBBs (obb1[i] with obb2[i]).
 * This is the standard behavior matching Python implementation.
 *
 * Computes IoU for rotated boxes using Gaussian-based approximation.
 * Much faster than exact polygon intersection methods.
 *
 * Reference: "Gaussian Bounding Boxes for Oriented Object Detection"
 *
 * @param obb1 First set of OBBs [N, 5] where 5 = [cx, cy, w, h, angle]
 * @param obb2 Second set of OBBs [N, 5] (must match obb1 size)
 * @param eps Small value for numerical stability
 * @return Probiou vector [N] - element-wise Probiou values
 */
[[nodiscard]] torch::Tensor probiou(
    const torch::Tensor& obb1,
    const torch::Tensor& obb2,
    double eps = 1e-10f
);

/**
 * @brief Convert bounding boxes from xywh to xyxy format
 *
 * @param boxes Boxes [N, 4] in [x_center, y_center, width, height] format
 * @return Boxes [N, 4] in [x1, y1, x2, y2] format
 */
[[nodiscard]] torch::Tensor xywh2xyxy(const torch::Tensor& boxes);

/**
 * @brief Convert bounding boxes from xyxy to xywh format
 *
 * @param boxes Boxes [N, 4] in [x1, y1, x2, y2] format
 * @return Boxes [N, 4] in [x_center, y_center, width, height] format
 */
[[nodiscard]] torch::Tensor xyxy2xywh(const torch::Tensor& boxes);

/**
 * @brief Convert distance predictions to bounding boxes
 *
 * Used in DFL (Distribution Focal Loss) based detectors.
 *
 * @param distance Distance predictions [N, 4] as [left, top, right, bottom]
 * @param anchorPoints Anchor points [N, 2] as [x, y]
 * @param xywh If true, return xywh format; else xyxy format
 * @return Decoded bounding boxes [N, 4]
 */
[[nodiscard]] torch::Tensor dist2bbox(
    const torch::Tensor& distance,
    const torch::Tensor& anchorPoints,
    bool xywh = true
);

/**
 * @brief Convert bounding boxes to distance predictions
 *
 * Inverse of dist2bbox, used for target generation in DFL.
 *
 * @param anchorPoints Anchor points [N, 2] as [x, y]
 * @param bboxes Bounding boxes [N, 4] in xyxy format
 * @param regMax Maximum regression range (default: 16)
 * @return Distance predictions [N, 4] as [left, top, right, bottom]
 */
[[nodiscard]] torch::Tensor bbox2dist(
    const torch::Tensor& anchorPoints,
    const torch::Tensor& bboxes,
    int64_t regMax = 16
);

/**
 * @brief Convert distance and angle to rotated bounding boxes (xywh only)
 *
 * Note: This function returns [N, 4] or [batch, N, 4] as [cx, cy, w, h] WITHOUT angle.
 * The caller must concatenate the angle separately if a full [N, 5] output is needed.
 * This design provides flexibility in handling angle transformations independently.
 *
 * Supports both 2D and 3D tensors:
 * - 2D: distance [N, 4], angle [N, 1], anchorPoints [N, 2] -> output [N, 4]
 * - 3D: distance [batch, N, 4], angle [batch, N, 1], anchorPoints [N, 2] -> output [batch, N, 4]
 *
 * @param distance Distance predictions [N, 4] or [batch, N, 4] as [left, top, right, bottom]
 * @param angle Rotation angle [N, 1] or [batch, N, 1] (used for center calculation but not returned)
 * @param anchorPoints Anchor points [N, 2] as [x, y]
 * @return Rotated boxes [N, 4] or [batch, N, 4] as [cx, cy, w, h] (angle NOT included)
 */
[[nodiscard]] torch::Tensor dist2rbox(
    const torch::Tensor& distance,
    const torch::Tensor& angle,
    const torch::Tensor& anchorPoints
);

/**
 * @brief Convert rotated box from xywhr to 4 corner points (xyxyxyxy)
 *
 * @param rboxes Rotated boxes [N, 5] as [cx, cy, w, h, angle]
 * @return Corner points [N, 4, 2] where each box has 4 corners (x, y)
 */
[[nodiscard]] torch::Tensor xywhr2xyxyxyxy(const torch::Tensor& rboxes);

/**
 * @brief Generate anchor points from feature maps
 *
 * Creates anchor points (grid centers) for each feature map level.
 * Matches Python implementation in func.py:make_anchors
 *
 * @param feats Vector of feature map tensors [batch, channels, height, width]
 * @param strides Stride values for each feature level
 * @param gridCellOffset Offset for grid cell centers (default: 0.5 for center)
 * @return Tuple of (anchor_points [num_anchors, 2], stride_tensor [num_anchors, 1])
 */
[[nodiscard]] std::tuple<torch::Tensor, torch::Tensor> makeAnchors(
    const std::vector<torch::Tensor>& feats,
    const torch::Tensor& strides,
    float gridCellOffset = 0.5f
);

/**
 * @brief Generate anchor points from feature shapes (efficient version)
 *
 * Creates anchor points without allocating full feature tensors.
 * More memory efficient for cases where only shape is needed.
 *
 * @param featShapes Vector of (height, width) pairs for each feature level
 * @param strides Stride values for each feature level
 * @param dtype Data type for generated tensors
 * @param device Device for generated tensors
 * @param gridCellOffset Offset for grid cell centers (default: 0.5 for center)
 * @return Tuple of (anchor_points [num_anchors, 2], stride_tensor [num_anchors, 1])
 */
[[nodiscard]] std::tuple<torch::Tensor, torch::Tensor> makeAnchors(
    const std::vector<std::pair<int64_t, int64_t>>& featShapes,
    const torch::Tensor& strides,
    torch::Dtype dtype,
    torch::Device device,
    float gridCellOffset = 0.5f
);

} // namespace Utils
} // namespace Model
} // namespace WheelDL
