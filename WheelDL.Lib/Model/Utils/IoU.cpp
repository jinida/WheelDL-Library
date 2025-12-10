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
                double eps
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

                auto precObb1 = obb1.to(torch::kFloat64);
                auto precObb2 = obb2.to(torch::kFloat64);

                const double minWH = 1e-7;
                const double quarter = 0.25;
                const double half = 0.5;
                const double twelve = 12.0;
                const double four = 4.0;

                // Extract all at once
                auto x1 = precObb1.select(1, 0);
                auto y1 = precObb1.select(1, 1);
                auto w1 = precObb1.select(1, 2).clamp_min_(minWH);  // in-place
                auto h1 = precObb1.select(1, 3).clamp_min_(minWH);  // in-place
                auto angle1 = precObb1.select(1, 4);

                auto x2 = precObb2.select(1, 0);
                auto y2 = precObb2.select(1, 1);
                auto w2 = precObb2.select(1, 2).clamp_min_(minWH);  // in-place
                auto h2 = precObb2.select(1, 3).clamp_min_(minWH);  // in-place
                auto angle2 = precObb2.select(1, 4);

                auto cos1 = torch::cos(angle1);
                auto cos2 = torch::cos(angle2);

                auto cos1Sq = cos1 * cos1;
                auto sin1Sq = 1.0 - cos1Sq;
                auto cos2Sq = cos2 * cos2;
                auto sin2Sq = 1.0 - cos2Sq;

                auto sin1 = torch::sin(angle1);
                auto sin2 = torch::sin(angle2);

                auto w1SqDiv12 = w1 * w1 / twelve;
                auto h1SqDiv12 = h1 * h1 / twelve;
                auto w2SqDiv12 = w2 * w2 / twelve;
                auto h2SqDiv12 = h2 * h2 / twelve;

                auto a1 = w1SqDiv12 * cos1Sq + h1SqDiv12 * sin1Sq;
                auto b1 = w1SqDiv12 * sin1Sq + h1SqDiv12 * cos1Sq;
                auto sinCos1 = sin1 * cos1;
                auto c1 = (w1SqDiv12 - h1SqDiv12) * sinCos1;

                auto a2 = w2SqDiv12 * cos2Sq + h2SqDiv12 * sin2Sq;
                auto b2 = w2SqDiv12 * sin2Sq + h2SqDiv12 * cos2Sq;
                auto sinCos2 = sin2 * cos2;
                auto c2 = (w2SqDiv12 - h2SqDiv12) * sinCos2;

                auto maxAB = torch::maximum(
                    torch::maximum(a1, b1),
                    torch::maximum(a2, b2)
                );
                auto scale = maxAB.clamp_min_(eps);  // in-place

                a1 = a1 / scale;
                b1 = b1 / scale;
                c1 = c1 / scale;
                a2 = a2 / scale;
                b2 = b2 / scale;
                c2 = c2 / scale;

                auto aSum = a1 + a2;
                auto bSum = b1 + b2;
                auto cSum = c1 + c2;
                auto cSumSq = cSum * cSum;

                auto denom = (aSum * bSum - cSumSq).clamp_min_(eps);
                auto invDenom = 1.0 / denom;

                auto dx = x2 - x1;
                auto dy = y1 - y2;

                auto invSqrtScale = 1.0 / scale.sqrt();
                dx *= invSqrtScale;
                dy *= invSqrtScale;

                auto dxSq = dx * dx;
                auto dySq = dy * dy;

                auto t1 = (bSum * dxSq + aSum * dySq) * (invDenom * quarter);
                auto t2 = cSum * dx * dy * (invDenom * half);

                auto c1Sq = c1 * c1;
                auto c2Sq = c2 * c2;
                auto det1 = (a1 * b1 - c1Sq).clamp_min_(eps);  // in-place
                auto det2 = (a2 * b2 - c2Sq).clamp_min_(eps);  // in-place

                auto detProdSqrt = (det1 * det2).sqrt();
                auto t3 = half * (denom / (four * detProdSqrt + eps)).clamp_min_(eps).log();

                auto bd = (t1 + t2 + t3).clamp_(0.0, 100.0);  // in-place

                auto probIou = 1.0 - (1.0 - (-bd).exp_()).clamp_min_(0.0).sqrt();

                return probIou.to(obb1.dtype());
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

            std::vector<torch::Tensor> nonMaxSuppression(
                const torch::Tensor& prediction,
                float confThresh,
                float iouThresh,
                int maxDet
            ) {
                torch::NoGradGuard no_grad;

                torch::Tensor pred = prediction.dim() == 2 ? prediction.unsqueeze(0) : prediction;
                const int64_t batchSize = pred.size(0);

                std::vector<torch::Tensor> results;
                results.reserve(batchSize);

                int totalDetections = 0;

                for (int64_t b = 0; b < batchSize; ++b) {
                    auto img = pred[b];

                    auto [classConf, classIdx] = img.slice(1, 4).max(1);
                    auto confMask = classConf > confThresh;
                    auto validIndices = confMask.nonzero().squeeze(1);

                    if (validIndices.numel() == 0) {
                        results.push_back(torch::empty({ 0, 6 }, img.options()));
                        continue;
                    }

                    const int64_t topK = std::min(static_cast<int64_t>(maxDet * 5), validIndices.numel());
                    auto validConf = classConf.index_select(0, validIndices);
                    auto [topConf, topLocalIdx] = validConf.topk(topK);
                    auto topIdx = validIndices.index_select(0, topLocalIdx);

                    auto boxes = img.index_select(0, topIdx).slice(1, 0, 4);
                    auto finalClassIdx = classIdx.index_select(0, topIdx);

                    auto x = boxes.select(1, 0);
                    auto y = boxes.select(1, 1);
                    auto halfW = boxes.select(1, 2) * 0.5f;
                    auto halfH = boxes.select(1, 3) * 0.5f;

                    auto x1 = x - halfW;
                    auto y1 = y - halfH;
                    auto x2 = x + halfW;
                    auto y2 = y + halfH;

                    auto x1_cpu = x1.cpu();
                    auto y1_cpu = y1.cpu();
                    auto x2_cpu = x2.cpu();
                    auto y2_cpu = y2.cpu();
                    auto cls_cpu = finalClassIdx.cpu();

                    const float* x1_ptr = x1_cpu.data_ptr<float>();
                    const float* y1_ptr = y1_cpu.data_ptr<float>();
                    const float* x2_ptr = x2_cpu.data_ptr<float>();
                    const float* y2_ptr = y2_cpu.data_ptr<float>();
                    const int64_t* cls_ptr = cls_cpu.data_ptr<int64_t>();

                    std::vector<float> areas(topK);
                    for (int64_t i = 0; i < topK; ++i) {
                        areas[i] = (x2_ptr[i] - x1_ptr[i]) * (y2_ptr[i] - y1_ptr[i]);
                    }

                    std::vector<int64_t> keep;
                    keep.reserve(maxDet);
                    std::vector<bool> suppressed(topK, false);

                    for (int64_t i = 0; i < topK && static_cast<int>(keep.size()) < maxDet; ++i) {
                        if (suppressed[i]) continue;
                        keep.push_back(i);

                        const float ax1 = x1_ptr[i], ay1 = y1_ptr[i];
                        const float ax2 = x2_ptr[i], ay2 = y2_ptr[i];
                        const float area_a = areas[i];
                        const int64_t cls_a = cls_ptr[i];

                        for (int64_t j = i + 1; j < topK; ++j) {
                            if (suppressed[j] || cls_ptr[j] != cls_a) continue;

                            const float inter_x1 = std::max(ax1, x1_ptr[j]);
                            const float inter_y1 = std::max(ay1, y1_ptr[j]);
                            const float inter_x2 = std::min(ax2, x2_ptr[j]);
                            const float inter_y2 = std::min(ay2, y2_ptr[j]);

                            if (inter_x2 <= inter_x1 || inter_y2 <= inter_y1) continue;

                            const float inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
                            const float iou = inter_area / (area_a + areas[j] - inter_area);

                            if (iou > iouThresh) {
                                suppressed[j] = true;
                            }
                        }
                    }

                    if (keep.empty()) {
                        results.push_back(torch::empty({ 0, 6 }, img.options()));
                        continue;
                    }

                    const int64_t numKeep = static_cast<int64_t>(keep.size());
                    auto keepTensor = torch::from_blob(keep.data(), { numKeep }, torch::kLong).to(topIdx.device());

                    auto finalBoxes = torch::stack({
                        x1.index_select(0, keepTensor),
                        y1.index_select(0, keepTensor),
                        x2.index_select(0, keepTensor),
                        y2.index_select(0, keepTensor)
                        }, 1);

                    results.push_back(torch::cat({
                        finalBoxes,
                        topConf.index_select(0, keepTensor).unsqueeze(1),
                        finalClassIdx.index_select(0, keepTensor).to(torch::kFloat32).unsqueeze(1)
                        }, 1));
                }

                return results;
            }

            std::vector<torch::Tensor> nonMaxSuppressionOBB(
                const torch::Tensor& prediction,
                float confThresh,
                float iouThresh,
                int maxDet
            ) {
                torch::NoGradGuard no_grad;

                torch::Tensor pred = prediction.dim() == 2 ? prediction.unsqueeze(0) : prediction;
                const int64_t batchSize = pred.size(0);
                const auto device = pred.device();

                std::vector<torch::Tensor> results;
                results.reserve(batchSize);

                for (int64_t b = 0; b < batchSize; ++b) {
                    auto img = pred[b];

                    if (img.size(0) == 0 || img.size(1) <= 5) {
                        results.push_back(torch::empty({ 0, 7 }, img.options()));
                        continue;
                    }

                    auto classScores = img.slice(1, 5);
                    if (classScores.size(1) == 0) {
                        results.push_back(torch::empty({ 0, 7 }, img.options()));
                        continue;
                    }

                    auto [classConf, classIdx] = classScores.max(1);
                    auto validMask = classConf > confThresh;
                    auto validCount = validMask.sum().item<int64_t>();

                    if (validCount == 0) {
                        results.push_back(torch::empty({ 0, 7 }, img.options()));
                        continue;
                    }

                    auto validIndices = validMask.nonzero().view(-1);

                    const int64_t topK = std::min(static_cast<int64_t>(maxDet * 5), validCount);
                    auto validConf = classConf.index_select(0, validIndices);
                    auto [topConf, topLocalIdx] = validConf.topk(topK);
                    auto topIdx = validIndices.index_select(0, topLocalIdx);

                    auto obbBoxes = img.index_select(0, topIdx).slice(1, 0, 5);
                    auto topClassIdx = classIdx.index_select(0, topIdx);

                    auto uniqueClasses = std::get<0>(torch::_unique(topClassIdx));
                    auto uniqueClassesCPU = uniqueClasses.cpu();
                    const int64_t numClasses = uniqueClassesCPU.size(0);
                    const int64_t* uniqueClassPtr = uniqueClassesCPU.data_ptr<int64_t>();

                    std::vector<int64_t> keepAll;
                    keepAll.reserve(maxDet);

                    for (int64_t c = 0; c < numClasses && static_cast<int64_t>(keepAll.size()) < maxDet; ++c) {
                        int64_t cls = uniqueClassPtr[c];
                        auto classMask = topClassIdx == cls;
                        auto classIndices = classMask.nonzero().view(-1);
                        const int64_t numBoxes = classIndices.size(0);

                        if (numBoxes == 0) continue;

                        if (numBoxes == 1) {
                            keepAll.push_back(classIndices[0].item<int64_t>());
                            continue;
                        }

                        auto classBoxes = obbBoxes.index_select(0, classIndices);

                        auto boxes1 = classBoxes.unsqueeze(1).expand({ numBoxes, numBoxes, 5 }).reshape({ -1, 5 });
                        auto boxes2 = classBoxes.unsqueeze(0).expand({ numBoxes, numBoxes, 5 }).reshape({ -1, 5 });
                        auto iouFlat = probiou(boxes1, boxes2);
                        auto iouMatrix = iouFlat.view({ numBoxes, numBoxes }).cpu();

                        auto iouPtr = iouMatrix.data_ptr<float>();
                        auto classIndicesCPU = classIndices.cpu();
                        auto classIdxPtr = classIndicesCPU.data_ptr<int64_t>();

                        std::vector<bool> suppressed(numBoxes, false);

                        for (int64_t i = 0; i < numBoxes && static_cast<int64_t>(keepAll.size()) < maxDet; ++i) {
                            if (suppressed[i]) continue;

                            keepAll.push_back(classIdxPtr[i]);

                            for (int64_t j = i + 1; j < numBoxes; ++j) {
                                if (!suppressed[j] && iouPtr[i * numBoxes + j] > iouThresh) {
                                    suppressed[j] = true;
                                }
                            }
                        }
                    }

                    if (keepAll.empty()) {
                        results.push_back(torch::empty({ 0, 7 }, img.options()));
                        continue;
                    }

                    auto keepTensor = torch::tensor(keepAll, torch::TensorOptions().dtype(torch::kLong).device(device));

                    results.push_back(torch::cat({
                        obbBoxes.index_select(0, keepTensor),
                        topConf.index_select(0, keepTensor).unsqueeze(1),
                        topClassIdx.index_select(0, keepTensor).to(torch::kFloat32).unsqueeze(1)
                        }, 1));
                }

                return results;
            }
        } // namespace Utils
    } // namespace Model
} // namespace WheelDL
