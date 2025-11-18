#pragma once

#include <torch/torch.h>
#include <tuple>

namespace WheelDL {
    namespace Model {
        namespace Utils {

            /**
             * @brief Task-Aligned Assigner for object detection
             *
             * Assigns ground truth objects to anchors based on task-aligned metric,
             * which combines both classification and localization information.
             *
             * Reference: "TOOD: Task-aligned One-stage Object Detection"
             *
             * Key improvements over Python version:
             * - Efficient tensor operations with minimal allocations
             * - Better memory management with move semantics
             * - CUDA OOM handling with automatic CPU fallback
             * - Optimized topk selection with scatter operations
             * - Cache-friendly data access patterns
             */
            class TaskAlignedAssigner {
            public:
                /**
                 * @brief Construct a task-aligned assigner
                 *
                 * @param topk Number of top candidates to consider per GT box (default: 13)
                 * @param numClasses Number of object classes (default: 80)
                 * @param alpha Weight for classification component (default: 1.0)
                 * @param beta Weight for localization component (default: 6.0)
                 * @param eps Small value to prevent division by zero (default: 1e-9)
                 */
                explicit TaskAlignedAssigner(
                    int64_t topk = 13,
                    int64_t numClasses = 80,
                    float alpha = 1.0f,
                    float beta = 6.0f,
                    float eps = 1e-9f
                );

                /**
                 * @brief Destructor
                 */
                ~TaskAlignedAssigner() = default;

                /**
                 * @brief Assign ground truth to predictions
                 *
                 * @param pdScores Predicted classification scores [batch, num_anchors, num_classes]
                 * @param pdBboxes Predicted bounding boxes [batch, num_anchors, 4]
                 * @param ancPoints Anchor points [num_anchors, 2]
                 * @param gtLabels Ground truth class labels [batch, max_num_gt, 1]
                 * @param gtBboxes Ground truth bounding boxes [batch, max_num_gt, 4]
                 * @param maskGt Mask for valid GT boxes [batch, max_num_gt, 1]
                 * @return Tuple of (target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx)
                 */
                [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
                    forward(const torch::Tensor& pdScores, 
                        const torch::Tensor& pdBboxes, 
                        const torch::Tensor& ancPoints, 
                        const torch::Tensor& gtLabels, 
                        const torch::Tensor& gtBboxes, 
                        const torch::Tensor& maskGt);

            protected:
                /**
                 * @brief Get positive mask based on alignment metric
                 *
                 * @return Tuple of (mask_pos, align_metric, overlaps)
                 */
                [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                    getPosMask(
                        const torch::Tensor& pdScores,
                        const torch::Tensor& pdBboxes,
                        const torch::Tensor& gtLabels,
                        const torch::Tensor& gtBboxes,
                        const torch::Tensor& ancPoints,
                        const torch::Tensor& maskGt
                    );

                /**
                 * @brief Compute alignment metric and IoU overlaps
                 *
                 * @return Tuple of (align_metric, overlaps)
                 */
                [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor>
                    getBoxMetrics(
                        const torch::Tensor& pdScores,
                        const torch::Tensor& pdBboxes,
                        const torch::Tensor& gtLabels,
                        const torch::Tensor& gtBboxes,
                        const torch::Tensor& maskGt
                    );

                /**
                 * @brief Compute IoU between GT and predicted boxes
                 *
                 * Virtual function to allow OBB override with different IoU computation.
                 *
                 * @param gtBboxes Ground truth boxes
                 * @param pdBboxes Predicted boxes
                 * @return IoU values
                 */
                [[nodiscard]] virtual torch::Tensor computeIou(
                    const torch::Tensor& gtBboxes,
                    const torch::Tensor& pdBboxes
                ) const;

                /**
                 * @brief Select top-k candidates based on alignment metric
                 *
                 * Efficient implementation using scatter operations.
                 *
                 * @param metrics Alignment metrics [batch, max_num_gt, num_anchors]
                 * @param largest If true, select largest values; else smallest
                 * @param topkMask Optional mask for valid candidates
                 * @return Mask for selected candidates
                 */
                [[nodiscard]] torch::Tensor selectTopkCandidates(
                    const torch::Tensor& metrics,
                    bool largest = true,
                    const torch::Tensor& topkMask = torch::Tensor()
                );

                /**
                 * @brief Get assigned targets
                 *
                 * @return Tuple of (target_labels, target_bboxes, target_scores)
                 */
                [[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                    getTargets(
                        const torch::Tensor& gtLabels,
                        const torch::Tensor& gtBboxes,
                        const torch::Tensor& targetGtIdx,
                        const torch::Tensor& fgMask
                    );

                /**
                 * @brief Select anchors that fall inside ground truth boxes
                 *
                 * @param xyCenters Anchor center points [num_anchors, 2]
                 * @param gtBboxes Ground truth boxes [batch, max_num_gt, 4]
                 * @param eps Small value for numerical stability
                 * @return Mask indicating which anchors are inside GTs
                 */
                [[nodiscard]] virtual torch::Tensor selectCandidatesInGts(
                    const torch::Tensor& xyCenters,
                    const torch::Tensor& gtBboxes,
                    float eps = 1e-9f
                );

                /**
                 * @brief Select highest overlapping GT when anchor matches multiple GTs
                 *
                 * @param maskPos Position mask [batch, max_num_gt, num_anchors]
                 * @param overlaps IoU overlaps [batch, max_num_gt, num_anchors]
                 * @param nMaxBoxes Maximum number of GT boxes
                 * @return Tuple of (target_gt_idx, fg_mask, mask_pos)
                 */
                [[nodiscard]] static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
                    selectHighestOverlaps(
                        const torch::Tensor& maskPos,
                        const torch::Tensor& overlaps,
                        int64_t nMaxBoxes
                    );

            protected:
                int64_t _topk;        ///< Number of top candidates
                int64_t _numClasses;  ///< Number of classes
                int64_t _bgIdx;       ///< Background class index
                float _alpha;         ///< Classification component weight
                float _beta;          ///< Localization component weight
                float _eps;           ///< Numerical stability epsilon

                // Cached values during forward pass
                int64_t _bs;          ///< Batch size
                int64_t _nMaxBoxes;   ///< Max number of GT boxes
            };

            /**
             * @brief OBB Task-Aligned Assigner
             *
             * Specialized assigner for oriented bounding boxes.
             * Uses Probiou for IoU computation and different inside-GT check.
             */
            class OBBTaskAlignedAssigner : public TaskAlignedAssigner {
            public:
                using TaskAlignedAssigner::TaskAlignedAssigner;

            protected:
                /**
                 * @brief Compute Probiou for oriented boxes
                 */
                [[nodiscard]] torch::Tensor computeIou(
                    const torch::Tensor& gtBboxes,
                    const torch::Tensor& pdBboxes
                ) const override;

                /**
                 * @brief Select anchors inside oriented boxes
                 *
                 * Uses point-in-rotated-box test with corner points.
                 * Note: eps parameter not needed for OBB since we use exact corner-based check
                 */
                [[nodiscard]] torch::Tensor selectCandidatesInGts(
                    const torch::Tensor& xyCenters,
                    const torch::Tensor& gtBboxes,
                    float eps = 1e-9f  // Kept for interface compatibility but unused
                ) override;
            };

        } // namespace Utils
    } // namespace Model
} // namespace WheelDL
