#include "pch.h"
#include "IoU.h"
#include <cmath>
#include <algorithm>

namespace WheelDL {
    namespace Model {
        namespace Utils {

            torch::Tensor xywh2xyxy(const torch::Tensor& boxes) {
                // boxes: [N, 4] as [cx, cy, w, h]
                auto xy = boxes.slice(1, 0, 2);  // [N, 2] center
                auto wh = boxes.slice(1, 2, 4);  // [N, 2] width/height

                auto half_wh = wh / 2.0f;
                auto xy1 = xy - half_wh;  // top-left
                auto xy2 = xy + half_wh;  // bottom-right

                return torch::cat({ xy1, xy2 }, 1);
            }

            torch::Tensor xyxy2xywh(const torch::Tensor& boxes) {
                // boxes: [N, 4] as [x1, y1, x2, y2]
                auto xy1 = boxes.slice(1, 0, 2);
                auto xy2 = boxes.slice(1, 2, 4);

                auto xy = (xy1 + xy2) / 2.0f;  // center
                auto wh = xy2 - xy1;            // width/height

                return torch::cat({ xy, wh }, 1);
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
                        "bboxIoU: box2 must be [M, 4] tensor, got shape " +
                        std::to_string(box2.dim()) + "D with size " +
                        std::to_string(box2.size(1)) + " in dim 1"
                    );
                }

                // Convert to xyxy format if needed
                // Fixed: Use value semantics instead of const reference to avoid dangling reference
                // when xywh2xyxy returns a temporary object
                torch::Tensor b1 = xywh ? xywh2xyxy(box1) : box1;
                torch::Tensor b2 = xywh ? xywh2xyxy(box2) : box2;

                // Get coordinates [N, 4] and [M, 4]
                auto b1_x1 = b1.select(1, 0).unsqueeze(1);  // [N, 1]
                auto b1_y1 = b1.select(1, 1).unsqueeze(1);
                auto b1_x2 = b1.select(1, 2).unsqueeze(1);
                auto b1_y2 = b1.select(1, 3).unsqueeze(1);

                auto b2_x1 = b2.select(1, 0).unsqueeze(0);  // [1, M]
                auto b2_y1 = b2.select(1, 1).unsqueeze(0);
                auto b2_x2 = b2.select(1, 2).unsqueeze(0);
                auto b2_y2 = b2.select(1, 3).unsqueeze(0);

                // Intersection area [N, M]
                auto inter_x1 = torch::max(b1_x1, b2_x1);
                auto inter_y1 = torch::max(b1_y1, b2_y1);
                auto inter_x2 = torch::min(b1_x2, b2_x2);
                auto inter_y2 = torch::min(b1_y2, b2_y2);

                auto inter_w = torch::clamp(inter_x2 - inter_x1, 0.0f);
                auto inter_h = torch::clamp(inter_y2 - inter_y1, 0.0f);
                auto inter_area = inter_w * inter_h;

                // Union area
                auto b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1);  // [N, 1]
                auto b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1);  // [1, M]
                auto union_area = b1_area + b2_area - inter_area + eps;

                // IoU
                auto iou = inter_area / union_area;

                if (GIoU || DIoU || CIoU) {
                    // Smallest enclosing box
                    auto cw = torch::max(b1_x2, b2_x2) - torch::min(b1_x1, b2_x1);
                    auto ch = torch::max(b1_y2, b2_y2) - torch::min(b1_y1, b2_y1);

                    if (CIoU || DIoU) {
                        // Diagonal of smallest enclosing box
                        auto c2 = cw.pow(2) + ch.pow(2) + eps;

                        // Center distance
                        auto b1_cx = (b1_x1 + b1_x2) / 2.0f;
                        auto b1_cy = (b1_y1 + b1_y2) / 2.0f;
                        auto b2_cx = (b2_x1 + b2_x2) / 2.0f;
                        auto b2_cy = (b2_y1 + b2_y2) / 2.0f;

                        auto rho2 = (b1_cx - b2_cx).pow(2) + (b1_cy - b2_cy).pow(2);

                        if (CIoU) {
                            // Aspect ratio consistency
                            auto b1_w = b1_x2 - b1_x1;
                            auto b1_h = b1_y2 - b1_y1;
                            auto b2_w = b2_x2 - b2_x1;
                            auto b2_h = b2_y2 - b2_y1;

                            // Clamp heights to avoid numerical instability in atan
                            auto b1_h_safe = torch::clamp_min(b1_h, HEIGHT_MIN_THRESHOLD);
                            auto b2_h_safe = torch::clamp_min(b2_h, HEIGHT_MIN_THRESHOLD);

                            auto v = (4.0f / (PI * PI)) *
                                torch::pow(torch::atan(b2_w / b2_h_safe) -
                                    torch::atan(b1_w / b1_h_safe), 2);

                            // Clamp alpha denominator to prevent division by very small numbers
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
             * @param obb2 [M, 5] tensor containing M oriented boxes [cx, cy, w, h, angle]
             * @param eps Small constant for numerical stability (default: 1e-7)
             * @return [N, M] tensor of probabilistic IoU values in range [0, 1]
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
                        "probiou: obb2 must be [M, 5] tensor, got shape " +
                        std::to_string(obb2.dim()) + "D with size " +
                        std::to_string(obb2.size(1)) + " in dim 1"
                    );
                }

                // Python: x1, y1 = obb1[..., :2].split(1, dim=-1)
                // Python: x2, y2 = obb2[..., :2].split(1, dim=-1)
                auto x1 = obb1.select(1, 0).unsqueeze(1);  // [N, 1]
                auto y1 = obb1.select(1, 1).unsqueeze(1);
                auto x2 = obb2.select(1, 0).unsqueeze(0);  // [1, M]
                auto y2 = obb2.select(1, 1).unsqueeze(0);

                // Python: def _get_covariance_matrix(boxes):
                //     gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
                //     a, b, c = gbbs.split(1, dim=-1)
                //     cos = c.cos()
                //     sin = c.sin()
                //     cos2 = cos.pow(2)
                //     sin2 = sin.pow(2)
                //     return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin

                // Get covariance matrix for obb1
                auto w1 = obb1.select(1, 2);
                auto h1 = obb1.select(1, 3);
                auto angle1 = obb1.select(1, 4);
                auto a1_base = w1.pow(2) / 12.0f;
                auto b1_base = h1.pow(2) / 12.0f;
                auto cos1 = torch::cos(angle1);
                auto sin1 = torch::sin(angle1);
                auto cos1_sq = cos1.pow(2);
                auto sin1_sq = sin1.pow(2);
                auto a1 = (a1_base * cos1_sq + b1_base * sin1_sq).unsqueeze(1);  // [N, 1]
                auto b1 = (a1_base * sin1_sq + b1_base * cos1_sq).unsqueeze(1);
                auto c1 = ((a1_base - b1_base) * cos1 * sin1).unsqueeze(1);

                // Get covariance matrix for obb2
                auto w2 = obb2.select(1, 2);
                auto h2 = obb2.select(1, 3);
                auto angle2 = obb2.select(1, 4);
                auto a2_base = w2.pow(2) / 12.0f;
                auto b2_base = h2.pow(2) / 12.0f;
                auto cos2 = torch::cos(angle2);
                auto sin2 = torch::sin(angle2);
                auto cos2_sq = cos2.pow(2);
                auto sin2_sq = sin2.pow(2);
                auto a2 = (a2_base * cos2_sq + b2_base * sin2_sq).unsqueeze(0);  // [1, M]
                auto b2 = (a2_base * sin2_sq + b2_base * cos2_sq).unsqueeze(0);
                auto c2 = ((a2_base - b2_base) * cos2 * sin2).unsqueeze(0);

                // Python: t1 = ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps) * 0.25
                // Fixed: Clamp denominator to prevent numerical instability with very small values
                const float DENOM_MIN = eps * 10.0f;
                auto denomRaw = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps;
                auto denom = torch::clamp_min(denomRaw, DENOM_MIN);
                auto t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / denom) * 0.25f;

                // Python: t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
                auto t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5f;

                // Python: t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps) + eps).log() * 0.5
                auto det1 = (a1 * b1 - c1.pow(2)).clamp_min(0.0f);
                auto det2 = (a2 * b2 - c2.pow(2)).clamp_min(0.0f);
                auto t3 = (denom / (4.0f * (det1 * det2).sqrt() + eps) + eps).log() * 0.5f;

                // Python: bd = (t1 + t2 + t3).clamp(eps, 100.0)
                auto bd = (t1 + t2 + t3).clamp(eps, 100.0f);

                // Python: hd = (1.0 - (-bd).exp() + eps).sqrt()
                auto hd = (1.0f - (-bd).exp() + eps).sqrt();

                // Python: iou = 1 - hd
                auto iou = 1.0f - hd;

                return iou;
            }

            torch::Tensor dist2bbox(
                const torch::Tensor& distance,
                const torch::Tensor& anchorPoints,
                bool xywh
            ) {
                // distance: [N, 4] as [left, top, right, bottom]
                // anchorPoints: [N, 2] as [x, y]

                auto lt = distance.slice(1, 0, 2);  // left, top
                auto rb = distance.slice(1, 2, 4);  // right, bottom

                auto x1y1 = anchorPoints - lt;
                auto x2y2 = anchorPoints + rb;

                if (xywh) {
                    auto c_xy = (x1y1 + x2y2) / 2.0f;
                    auto wh = x2y2 - x1y1;
                    return torch::cat({ c_xy, wh }, 1);
                }

                return torch::cat({ x1y1, x2y2 }, 1);
            }

            torch::Tensor bbox2dist(
                const torch::Tensor& anchorPoints,
                const torch::Tensor& bboxes,
                int64_t regMax
            ) {
                // anchorPoints: [N, 2]
                // bboxes: [N, 4] in xyxy format

                // Python implementation:
                // def bbox2dist(anchor_points, bbox, reg_max):
                //     x1y1, x2y2 = bbox.chunk(2, -1)
                //     return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)

                auto x1y1 = bboxes.slice(1, 0, 2);
                auto x2y2 = bboxes.slice(1, 2, 4);

                auto lt = anchorPoints - x1y1;  // left, top distances
                auto rb = x2y2 - anchorPoints;  // right, bottom distances

                auto dist = torch::cat({ lt, rb }, 1);

                // Clamp to [0, regMax - 0.01] (matches Python exactly)
                return dist.clamp(0.0f, static_cast<float>(regMax) - 0.01f);
            }

            torch::Tensor dist2rbox(
                const torch::Tensor& distance,
                const torch::Tensor& angle,
                const torch::Tensor& anchorPoints
            ) {
                auto lt = distance.slice(1, 0, 2);  // [N, 2]
                auto rb = distance.slice(1, 2, 4);  // [N, 2]

                auto cos_angle = torch::cos(angle);
                auto sin_angle = torch::sin(angle);

                auto offset = (rb - lt) / 2.0f;  // [N, 2]
                auto xf = offset.select(1, 0).unsqueeze(1);  // [N, 1]
                auto yf = offset.select(1, 1).unsqueeze(1);  // [N, 1]

                auto x = xf * cos_angle - yf * sin_angle;  // [N, 1]
                auto y = xf * sin_angle + yf * cos_angle;  // [N, 1]

                auto xy = torch::cat({ x, y }, 1) + anchorPoints;  // [N, 2]
                auto wh = lt + rb;  // [N, 2]

                return torch::cat({ xy, wh }, 1);  // [N, 4] - NOTE: No angle in output!
            }

            torch::Tensor xywhr2xyxyxyxy(const torch::Tensor& rboxes) {
                // rboxes: [N, 5] as [cx, cy, w, h, angle]

                auto cx = rboxes.select(1, 0);
                auto cy = rboxes.select(1, 1);
                auto w = rboxes.select(1, 2);
                auto h = rboxes.select(1, 3);
                auto angle = rboxes.select(1, 4);

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
                    torch::stack({x1, y1}, 1),
                    torch::stack({x2, y2}, 1),
                    torch::stack({x3, y3}, 1),
                    torch::stack({x4, y4}, 1)
                    }, 1);

                return corners;
            }

            std::tuple<torch::Tensor, torch::Tensor> makeAnchors(
                const std::vector<torch::Tensor>& feats,
                const torch::Tensor& strides,
                float gridCellOffset
            ) {
                // Python implementation:
                // def make_anchors(feats, strides, grid_cell_offset=0.5):
                //     anchor_points, stride_tensor = [], []
                //     assert feats is not None
                //     dtype, device = feats[0].dtype, feats[0].device
                //     for i, stride in enumerate(strides):
                //         h, w = feats[i].shape[2:]
                //         sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
                //         sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
                //         sy, sx = torch.meshgrid(sy, sx, indexing="ij")
                //         anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
                //         stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
                //     return torch.cat(anchor_points), torch::cat(stride_tensor)

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
