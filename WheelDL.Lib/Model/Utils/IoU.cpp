#include "pch.h"
#include "IoU.h"
#include <cmath>
#include <algorithm>

namespace WheelDL {
    namespace Model {
        namespace Utils {

            torch::Tensor xywh2xyxy(const torch::Tensor& boxes) {
                int64_t dim = 0;
                if (boxes.dim() == 2)
                {
                    dim = 1;
                }
                else if (boxes.dim() == 3)
                {
                    dim = 2;
                }
				else
                {
                    throw std::invalid_argument(
                        "xywh2xyxy: boxes must be [N, 4] or [N, M, 4] tensor, got " +
                        std::to_string(boxes.dim()) + "D tensor"
					);
                }

                // OPTIMIZATION: Pre-allocate output and write directly to avoid cat overhead
                auto xy = boxes.slice(dim, 0, 2);  // [N, 2] center
                auto wh = boxes.slice(dim, 2, 4);  // [N, 2] width/height
                auto half_wh = wh * 0.5f;

                // Allocate output once and fill in-place
                auto output = torch::empty_like(boxes);
                output.slice(dim, 0, 2).copy_(xy - half_wh);  // xy1 (top-left)
                output.slice(dim, 2, 4).copy_(xy + half_wh);  // xy2 (bottom-right)

                return output;
            }

            torch::Tensor xyxy2xywh(const torch::Tensor& boxes) {
                int64_t dim = 0;
                if (boxes.dim() == 2)
                {
                    dim = 1;
                }
                else if (boxes.dim() == 3)
                {
                    dim = 2;
                }
                else
                {
                    throw std::invalid_argument(
						"xyxy2xywh: boxes must be [N, 4] or [N, M, 4] tensor, got " +
                        std::to_string(boxes.dim()) + "D tensor"
                    );
                }
                // OPTIMIZATION: Reduce memory allocations
                auto xy1 = boxes.slice(dim, 0, 2);
                auto xy2 = boxes.slice(dim, 2, 4);

                auto xy = (xy1 + xy2).mul_(0.5f);  // center - inplace multiply
                auto wh = xy2 - xy1;                // width/height

                return torch::cat({ xy, wh }, dim);
            }

            torch::Tensor bboxIoU(
                const torch::Tensor& box1,
                const torch::Tensor& box2,
                bool xywh,
                bool GIoU,
                bool DIoU,
                bool CIoU,
                float eps
            ) {
                // Validate input dimensions
                if (box1.dim() != 2 || box1.size(1) != 4) {
                    throw std::invalid_argument(
                        "bboxIoU: box1 must be [N, 4] tensor, got shape " +
                        std::to_string(box1.dim()) + "D with size " +
                        std::to_string(box1.size(1)) + " in dim 1"
                    );
                }
                if (box2.dim() != 2 || box2.size(1) != 4) {
                    throw std::invalid_argument(
                        "bboxIoU: box2 must be [N, 4] tensor, got shape " +
                        std::to_string(box2.dim()) + "D with size " +
                        std::to_string(box2.size(1)) + " in dim 1"
                    );
                }
                if (box1.size(0) != box2.size(0)) {
                    throw std::invalid_argument(
                        "bboxIoU: box1 and box2 must have same number of boxes, got " +
                        std::to_string(box1.size(0)) + " and " + std::to_string(box2.size(0))
                    );
                }

                // Convert to xyxy format if needed
                torch::Tensor b1 = xywh ? xywh2xyxy(box1) : box1;
                torch::Tensor b2 = xywh ? xywh2xyxy(box2) : box2;

                // Extract coordinates - element-wise operations
                auto b1_x1 = b1.select(1, 0);  // [N]
                auto b1_y1 = b1.select(1, 1);
                auto b1_x2 = b1.select(1, 2);
                auto b1_y2 = b1.select(1, 3);

                auto b2_x1 = b2.select(1, 0);  // [N]
                auto b2_y1 = b2.select(1, 1);
                auto b2_x2 = b2.select(1, 2);
                auto b2_y2 = b2.select(1, 3);

                // Intersection area [N]
                auto inter_x1 = torch::max(b1_x1, b2_x1);
                auto inter_y1 = torch::max(b1_y1, b2_y1);
                auto inter_x2 = torch::min(b1_x2, b2_x2);
                auto inter_y2 = torch::min(b1_y2, b2_y2);

                auto inter_w = (inter_x2 - inter_x1).clamp_min_(0.0f);
                auto inter_h = (inter_y2 - inter_y1).clamp_min_(0.0f);
                auto inter_area = inter_w.mul_(inter_h);

                // Union area
                auto b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1);  // [N]
                auto b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1);  // [N]
                auto union_area = b1_area + b2_area - inter_area + eps;

                // IoU
                auto iou = inter_area / union_area;

                if (GIoU || DIoU || CIoU) {
                    // Smallest enclosing box
                    auto cw = torch::max(b1_x2, b2_x2) - torch::min(b1_x1, b2_x1);
                    auto ch = torch::max(b1_y2, b2_y2) - torch::min(b1_y1, b2_y1);

                    if (CIoU || DIoU) {
                        auto c2 = (cw * cw).add_(ch * ch).add_(eps);

                        // Center distance
                        auto b1_cx = (b1_x1 + b1_x2).mul_(0.5f);
                        auto b1_cy = (b1_y1 + b1_y2).mul_(0.5f);
                        auto b2_cx = (b2_x1 + b2_x2).mul_(0.5f);
                        auto b2_cy = (b2_y1 + b2_y2).mul_(0.5f);

                        auto dx = b1_cx - b2_cx;
                        auto dy = b1_cy - b2_cy;
                        auto rho2 = (dx * dx).add_(dy * dy);

                        if (CIoU) {
                            // Aspect ratio consistency
                            auto b1_w = b1_x2 - b1_x1;
                            auto b1_h = b1_y2 - b1_y1;
                            auto b2_w = b2_x2 - b2_x1;
                            auto b2_h = b2_y2 - b2_y1;

                            auto b1_h_safe = torch::clamp_min(b1_h, HEIGHT_MIN_THRESHOLD);
                            auto b2_h_safe = torch::clamp_min(b2_h, HEIGHT_MIN_THRESHOLD);

                            auto v = (4.0f / (PI * PI)) *
                                torch::pow(torch::atan(b2_w / b2_h_safe) -
                                    torch::atan(b1_w / b1_h_safe), 2);

                            auto alphaDenom = torch::clamp_min(1.0f - iou + v, eps);
                            auto alpha = v / alphaDenom;

                            return iou - (rho2 / c2 + v * alpha);
                        }
                        else {
                            // DIoU
                            return iou - rho2 / c2;
                        }
                    }
                    else {
                        // GIoU
                        auto c_area = cw * ch + eps;
                        return iou - (c_area - union_area) / c_area;
                    }
                }

                return iou;
            }

            /**
             * @brief Compute Probabilistic IoU for Oriented Bounding Boxes (OBB)
             *
             * Implements the Probabilistic IoU metric for rotated bounding boxes using
             * Gaussian-based Bhattacharyya distance computation.
             *
             * Reference: "Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss"
             *           (https://arxiv.org/abs/2101.11952)
             *
             * Python reference: src/utils/func.py::probiou
             *
             * @param obb1 [N, 5] tensor containing N oriented boxes [cx, cy, w, h, angle]
             * @param obb2 [N, 5] tensor containing N oriented boxes [cx, cy, w, h, angle]
             * @param eps Small constant for numerical stability (default: 1e-7)
             * @return [N] tensor of probabilistic IoU values in range [0, 1]
             */
            torch::Tensor probiou(
                const torch::Tensor& obb1,
                const torch::Tensor& obb2,
                float eps
            ) {
                // Validate input dimensions
                if (obb1.dim() != 2 || obb1.size(1) != 5) {
                    throw std::invalid_argument(
                        "probiou: obb1 must be [N, 5] tensor, got shape " +
                        std::to_string(obb1.dim()) + "D with size " +
                        std::to_string(obb1.size(1)) + " in dim 1"
                    );
                }
                if (obb2.dim() != 2 || obb2.size(1) != 5) {
                    throw std::invalid_argument(
                        "probiou: obb2 must be [N, 5] tensor, got shape " +
                        std::to_string(obb2.dim()) + "D with size " +
                        std::to_string(obb2.size(1)) + " in dim 1"
                    );
                }
                if (obb1.size(0) != obb2.size(0)) {
                    throw std::invalid_argument(
                        "probiou: obb1 and obb2 must have same number of boxes, got " +
                        std::to_string(obb1.size(0)) + " and " + std::to_string(obb2.size(0))
                    );
                }

                // Extract coordinates - element-wise operations
                auto x1 = obb1.select(1, 0);  // [N]
                auto y1 = obb1.select(1, 1);  // [N]
                auto x2 = obb2.select(1, 0);  // [N]
                auto y2 = obb2.select(1, 1);  // [N]

                // Get covariance matrix for obb1
                auto w1 = obb1.select(1, 2);
                auto h1 = obb1.select(1, 3);
                auto angle1 = obb1.select(1, 4);

                auto a1_base = (w1 * w1) / 12.0f;
                auto b1_base = (h1 * h1) / 12.0f;

                auto cos1 = torch::cos(angle1);
                auto sin1 = torch::sin(angle1);
                auto cos1_sq = cos1 * cos1;
                auto sin1_sq = 1.0f - cos1_sq;

                auto a1 = a1_base * cos1_sq + b1_base * sin1_sq;  // [N]
                auto b1 = a1_base * sin1_sq + b1_base * cos1_sq;
                auto c1 = (a1_base - b1_base) * cos1 * sin1;

                // Get covariance matrix for obb2
                auto w2 = obb2.select(1, 2);
                auto h2 = obb2.select(1, 3);
                auto angle2 = obb2.select(1, 4);

                auto a2_base = (w2 * w2) / 12.0f;
                auto b2_base = (h2 * h2) / 12.0f;

                auto cos2 = torch::cos(angle2);
                auto sin2 = torch::sin(angle2);
                auto cos2_sq = cos2 * cos2;
                auto sin2_sq = 1.0f - cos2_sq;

                auto a2 = a2_base * cos2_sq + b2_base * sin2_sq;  // [N]
                auto b2 = a2_base * sin2_sq + b2_base * cos2_sq;
                auto c2 = (a2_base - b2_base) * cos2 * sin2;

                // Compute distances and denominator
                const float DENOM_MIN = eps * 10.0f;

                auto a_sum = a1 + a2;
                auto b_sum = b1 + b2;
                auto c_sum = c1 + c2;
                auto c_sum_sq = c_sum * c_sum;

                auto denom = (a_sum * b_sum - c_sum_sq + eps).clamp_min(DENOM_MIN);

                // Compute deltas and terms
                auto dx = x2 - x1;
                auto dy = y1 - y2;
                auto dx_sq = dx * dx;
                auto dy_sq = dy * dy;

                // OPTIMIZATION: Reuse 1/denom to reduce divisions
                auto inv_denom = 1.0f / denom;

                // t1: distance term - fused operations
                auto t1 = (a_sum * dy_sq + b_sum * dx_sq) * inv_denom * 0.25f;

                // t2: cross term - fused operations
                auto t2 = (c_sum * dx * dy) * inv_denom * 0.5f;

                // t3: determinant term - optimized expression
                auto c1_sq = c1 * c1;
                auto c2_sq = c2 * c2;
                auto det1 = (a1 * b1 - c1_sq).clamp_min(0.0f);
                auto det2 = (a2 * b2 - c2_sq).clamp_min(0.0f);
                auto det_sqrt = (det1 * det2).sqrt() * 4.0f + eps;
                auto t3 = ((denom / det_sqrt) + eps).log() * 0.5f;

                // bd = (t1 + t2 + t3).clamp(eps, 100.0)
                auto bd = (t1 + t2 + t3).clamp(eps, 100.0f);

                // hd = (1.0 - (-bd).exp() + eps).sqrt()
                auto hd = (1.0f - (-bd).exp() + eps).sqrt();

                // iou = 1 - hd
                return 1.0f - hd;
            }

            torch::Tensor dist2bbox(
                const torch::Tensor& distance,
                const torch::Tensor& anchorPoints,
                bool xywh
            ) {
				int64_t dim = distance.dim();
                if (dim != 2 && dim != 3)
                {
                    throw std::invalid_argument(
                        "dist2bbox: distance must be [N, 4] or [N, M, 4] tensor, got " +
                        std::to_string(distance.dim()) + "D tensor"
                    );
                }
                
				dim = dim - 1; // last dimension
                // distance: [N, 4] as [left, top, right, bottom]
                // anchorPoints: [N, 2] as [x, y]

                auto lt = distance.slice(dim, 0, 2);  // left, top
                auto rb = distance.slice(dim, 2, 4);  // right, bottom

                auto x1y1 = anchorPoints - lt;
                auto x2y2 = anchorPoints + rb;

                if (xywh) {
                    auto c_xy = (x1y1 + x2y2) / 2.0f;
                    auto wh = x2y2 - x1y1;
                    return torch::cat({ c_xy, wh }, dim);
                }

                return torch::cat({ x1y1, x2y2 }, dim);
            }

            torch::Tensor bbox2dist(
                const torch::Tensor& anchorPoints,
                const torch::Tensor& bboxes,
                int64_t regMax
            ) {
				int64_t dim = bboxes.dim();
                if (dim != 2 && dim != 3) {
                    throw std::invalid_argument(
						"bbox2dist: bboxes must be [N, 4] or [N, M, 4] tensor, got " +
                        std::to_string(bboxes.dim()) + "D with size " +
                        std::to_string(bboxes.size(1)) + " in dim 1"
                    );
				}
				dim = dim - 1; // last dimension
                auto x1y1 = bboxes.slice(dim, 0, 2);
                auto x2y2 = bboxes.slice(dim, 2, 4);

                auto lt = anchorPoints - x1y1;  // left, top distances
                auto rb = x2y2 - anchorPoints;  // right, bottom distances

                auto dist = torch::cat({ lt, rb }, dim);

                // Clamp to [0, regMax - 0.01] (matches Python exactly)
                return dist.clamp(0.0f, static_cast<float>(regMax) - 0.01f);
            }

            torch::Tensor dist2rbox(
                const torch::Tensor& distance,
                const torch::Tensor& angle,
                const torch::Tensor& anchorPoints
            ) {
                int64_t dim = distance.dim();
                if (dim != 2 && dim != 3)
                {
                    throw std::invalid_argument(
                        "dist2rbox: distance must be [N, 4] or [batch, N, 4] tensor, got " +
                        std::to_string(distance.dim()) + "D tensor"
                    );
                }

                dim = dim - 1; // last dimension (same as dist2bbox)

                auto lt = distance.slice(dim, 0, 2);  // [N, 2] or [batch, N, 2]
                auto rb = distance.slice(dim, 2, 4);  // [N, 2] or [batch, N, 2]

                auto cos_angle = torch::cos(angle);
                auto sin_angle = torch::sin(angle);

                auto offset = (rb - lt) / 2.0f;  // [N, 2] or [batch, N, 2]
                auto xf = offset.select(dim, 0).unsqueeze(-1);  // [N, 1] or [batch, N, 1]
                auto yf = offset.select(dim, 1).unsqueeze(-1);  // [N, 1] or [batch, N, 1]

                auto x = xf * cos_angle - yf * sin_angle;  // [N, 1] or [batch, N, 1]
                auto y = xf * sin_angle + yf * cos_angle;  // [N, 1] or [batch, N, 1]

                auto xy = torch::cat({ x, y }, dim) + anchorPoints;  // [N, 2] or [batch, N, 2]
                auto wh = lt + rb;  // [N, 2] or [batch, N, 2]

                return torch::cat({ xy, wh }, dim);  // [N, 4] or [batch, N, 4] - NOTE: No angle in output!
            }

            torch::Tensor xywhr2xyxyxyxy(const torch::Tensor& rboxes) {
				int64_t dim = rboxes.dim();
				if (dim != 2 && dim != 3)
                {
                    throw std::invalid_argument(
						"xywhr2xyxyxyxy: rboxes must be [N, 5] or [batch, N, 5] tensor, got " +
                        std::to_string(rboxes.dim()) + "D tensor"
                    );
				}
                int64_t lastDim = dim - 1; // last dimension

                auto cx = rboxes.select(lastDim, 0);
                auto cy = rboxes.select(lastDim, 1);
                auto w = rboxes.select(lastDim, 2);
                auto h = rboxes.select(lastDim, 3);
                auto angle = rboxes.select(lastDim, 4);

                auto cos_a = torch::cos(angle);
                auto sin_a = torch::sin(angle);

                auto hw = w / 2.0f;
                auto hh = h / 2.0f;

                // 4 corners in local coordinate system, then rotate
                // Corner 1: (-hw, -hh)
                auto x1 = cx + (-hw * cos_a) - (-hh * sin_a);
                auto y1 = cy + (-hw * sin_a) + (-hh * cos_a);

                // Corner 2: (hw, -hh)
                auto x2 = cx + (hw * cos_a) - (-hh * sin_a);
                auto y2 = cy + (hw * sin_a) + (-hh * cos_a);

                // Corner 3: (hw, hh)
                auto x3 = cx + (hw * cos_a) - (hh * sin_a);
                auto y3 = cy + (hw * sin_a) + (hh * cos_a);

                // Corner 4: (-hw, hh)
                auto x4 = cx + (-hw * cos_a) - (hh * sin_a);
                auto y4 = cy + (-hw * sin_a) + (hh * cos_a);

                // Stack as [N, 4, 2]
                auto corners = torch::stack({
                    torch::stack({x1, y1}, lastDim),
                    torch::stack({x2, y2}, lastDim),
                    torch::stack({x3, y3}, lastDim),
                    torch::stack({x4, y4}, lastDim)
                    }, lastDim);

                return corners;
            }

            std::tuple<torch::Tensor, torch::Tensor> makeAnchors(
                const std::vector<torch::Tensor>& feats,
                const torch::Tensor& strides,
                float gridCellOffset
            ) 
            {
                if (feats.empty()) {
                    throw std::invalid_argument("makeAnchors: feats cannot be empty");
                }

                std::vector<torch::Tensor> anchorPoints;
                std::vector<torch::Tensor> strideTensors;
                anchorPoints.reserve(feats.size());
                strideTensors.reserve(feats.size());

                auto dtype = feats[0].dtype();
                auto device = feats[0].device();

                for (size_t i = 0; i < feats.size(); ++i) {
                    auto h = feats[i].size(2);
                    auto w = feats[i].size(3);

                    // Create grid coordinates with offset
                    auto sx = torch::arange(w, torch::TensorOptions().dtype(dtype).device(device)) + gridCellOffset;
                    auto sy = torch::arange(h, torch::TensorOptions().dtype(dtype).device(device)) + gridCellOffset;

                    // Create meshgrid (indexing="ij")
                    auto meshgrid = torch::meshgrid({ sy, sx }, "ij");
                    auto grid_y = meshgrid[0];
                    auto grid_x = meshgrid[1];

                    // Stack and reshape: [h, w, 2] -> [h*w, 2]
                    auto anchorGrid = torch::stack({ grid_x, grid_y }, -1).view({ -1, 2 });

                    // Get stride value
                    auto strideValue = strides[i].item<float>();
                    auto strideTensor = torch::full(
                        { h * w, 1 },
                        strideValue,
                        torch::TensorOptions().dtype(dtype).device(device)
                    );

                    anchorPoints.push_back(anchorGrid);
                    strideTensors.push_back(strideTensor);
                }

                // Concatenate all anchor points and stride tensors
                auto allAnchors = torch::cat(anchorPoints, 0);
                auto allStrides = torch::cat(strideTensors, 0);

                return std::make_tuple(allAnchors, allStrides);
            }

            std::tuple<torch::Tensor, torch::Tensor> makeAnchors(
                const std::vector<std::pair<int64_t, int64_t>>& featShapes,
                const torch::Tensor& strides,
                torch::Dtype dtype,
                torch::Device device,
                float gridCellOffset
            ) {
                if (featShapes.empty()) {
                    throw std::invalid_argument("makeAnchors: featShapes cannot be empty");
                }

                std::vector<torch::Tensor> anchorPoints;
                std::vector<torch::Tensor> strideTensors;
                anchorPoints.reserve(featShapes.size());
                strideTensors.reserve(featShapes.size());

                for (size_t i = 0; i < featShapes.size(); ++i) {
                    auto [h, w] = featShapes[i];

                    // Create grid coordinates with offset
                    auto sx = torch::arange(w, torch::TensorOptions().dtype(dtype).device(device)) + gridCellOffset;
                    auto sy = torch::arange(h, torch::TensorOptions().dtype(dtype).device(device)) + gridCellOffset;

                    // Create meshgrid (indexing="ij")
                    auto meshgrid = torch::meshgrid({ sy, sx }, "ij");
                    auto grid_y = meshgrid[0];
                    auto grid_x = meshgrid[1];

                    // Stack and reshape: [h, w, 2] -> [h*w, 2]
                    auto anchorGrid = torch::stack({ grid_x, grid_y }, -1).view({ -1, 2 });

                    // Get stride value
                    auto strideValue = strides[i].item<float>();
                    auto strideTensor = torch::full(
                        { h * w, 1 },
                        strideValue,
                        torch::TensorOptions().dtype(dtype).device(device)
                    );

                    anchorPoints.push_back(anchorGrid);
                    strideTensors.push_back(strideTensor);
                }

                // Concatenate all anchor points and stride tensors
                auto allAnchors = torch::cat(anchorPoints, 0);
                auto allStrides = torch::cat(strideTensors, 0);

                return std::make_tuple(allAnchors, allStrides);
            }

        } // namespace Utils
    } // namespace Model
} // namespace WheelDL
