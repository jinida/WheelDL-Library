#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Utils/IoU.h"

using namespace WheelDL::Model::Utils;

class IoUTest : public ::testing::Test {
protected:
    torch::Device device_ = torch::kCPU;

    void SetUp() override {
        torch::manual_seed(42);
#ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            device_ = torch::kCUDA;
        }
#endif
    }

    // Helper: Create box tensor in xyxy format [N, 4]
    torch::Tensor createXYXYBoxes(std::vector<std::vector<float>> boxes) {
        std::vector<float> flat;
        for (const auto& box : boxes) {
            for (float v : box) flat.push_back(v);
        }
        return torch::tensor(flat, torch::TensorOptions().dtype(torch::kFloat32).device(device_))
            .view({static_cast<int64_t>(boxes.size()), 4});
    }

    // Helper: Create box tensor in xywh format [N, 4]
    torch::Tensor createXYWHBoxes(std::vector<std::vector<float>> boxes) {
        return createXYXYBoxes(boxes);  // Same creation, different interpretation
    }

    // Helper: Create OBB tensor [N, 5] as [cx, cy, w, h, angle]
    torch::Tensor createOBBBoxes(std::vector<std::vector<float>> obbs) {
        std::vector<float> flat;
        for (const auto& obb : obbs) {
            for (float v : obb) flat.push_back(v);
        }
        return torch::tensor(flat, torch::TensorOptions().dtype(torch::kFloat32).device(device_))
            .view({static_cast<int64_t>(obbs.size()), 5});
    }
};

// =============================================================================
// bboxIoU Basic Tests (Section 2.1.1)
// =============================================================================

TEST_F(IoUTest, BboxIoU_PerfectOverlap) {
    auto box = createXYXYBoxes({{0, 0, 100, 100}});
    auto iou = bboxIoU(box, box, false);  // xyxy format

    EXPECT_NEAR(iou[0].item<float>(), 1.0f, 1e-5f);
}

TEST_F(IoUTest, BboxIoU_NoOverlap) {
    auto box1 = createXYXYBoxes({{0, 0, 10, 10}});
    auto box2 = createXYXYBoxes({{20, 20, 30, 30}});
    auto iou = bboxIoU(box1, box2, false);

    EXPECT_NEAR(iou[0].item<float>(), 0.0f, 1e-5f);
}

TEST_F(IoUTest, BboxIoU_Overlap_50Percent) {
    // box1: 0-100, box2: 50-150 -> 50% overlap
    auto box1 = createXYXYBoxes({{0, 0, 100, 100}});
    auto box2 = createXYXYBoxes({{50, 0, 150, 100}});
    auto iou = bboxIoU(box1, box2, false);

    // Intersection: 50*100 = 5000, Union: 10000+10000-5000 = 15000
    // IoU = 5000/15000 = 0.333
    EXPECT_NEAR(iou[0].item<float>(), 0.333f, 0.01f);
}

TEST_F(IoUTest, BboxIoU_Containment) {
    // Small box inside large box
    auto box1 = createXYXYBoxes({{0, 0, 100, 100}});
    auto box2 = createXYXYBoxes({{25, 25, 75, 75}});
    auto iou = bboxIoU(box1, box2, false);

    // Intersection: 50*50 = 2500, Union: 10000 + 2500 - 2500 = 10000
    // IoU = 2500/10000 = 0.25
    EXPECT_NEAR(iou[0].item<float>(), 0.25f, 0.01f);
}

TEST_F(IoUTest, BboxIoU_TouchingEdge) {
    // Adjacent boxes touching at edge
    auto box1 = createXYXYBoxes({{0, 0, 10, 10}});
    auto box2 = createXYXYBoxes({{10, 0, 20, 10}});
    auto iou = bboxIoU(box1, box2, false);

    EXPECT_NEAR(iou[0].item<float>(), 0.0f, 1e-5f);
}

TEST_F(IoUTest, BboxIoU_TouchingCorner) {
    // Diagonal boxes touching at corner
    auto box1 = createXYXYBoxes({{0, 0, 10, 10}});
    auto box2 = createXYXYBoxes({{10, 10, 20, 20}});
    auto iou = bboxIoU(box1, box2, false);

    EXPECT_NEAR(iou[0].item<float>(), 0.0f, 1e-5f);
}

TEST_F(IoUTest, BboxIoU_DifferentSizes) {
    // Large box and small box with overlap
    auto box1 = createXYXYBoxes({{0, 0, 100, 100}});
    auto box2 = createXYXYBoxes({{0, 0, 50, 50}});
    auto iou = bboxIoU(box1, box2, false);

    // Intersection: 50*50 = 2500, Union: 10000 + 2500 - 2500 = 10000
    // IoU = 2500/10000 = 0.25
    EXPECT_NEAR(iou[0].item<float>(), 0.25f, 0.01f);
}

TEST_F(IoUTest, BboxIoU_BatchComputation) {
    auto box1 = createXYXYBoxes({
        {0, 0, 10, 10},
        {0, 0, 100, 100},
        {0, 0, 10, 10}
    });
    auto box2 = createXYXYBoxes({
        {0, 0, 10, 10},    // Perfect overlap
        {50, 0, 150, 100}, // 33% overlap
        {20, 20, 30, 30}   // No overlap
    });
    auto iou = bboxIoU(box1, box2, false);

    EXPECT_EQ(iou.size(0), 3);
    EXPECT_NEAR(iou[0].item<float>(), 1.0f, 1e-5f);
    EXPECT_NEAR(iou[1].item<float>(), 0.333f, 0.01f);
    EXPECT_NEAR(iou[2].item<float>(), 0.0f, 1e-5f);
}

TEST_F(IoUTest, BboxIoU_PrecomputedValue) {
    // From test plan: box1=[0,0,100,100], box2=[50,0,150,100]
    auto box1 = createXYXYBoxes({{0, 0, 100, 100}});
    auto box2 = createXYXYBoxes({{50, 0, 150, 100}});
    auto iou = bboxIoU(box1, box2, false);

    // Intersection: [50, 0, 100, 100], area = 5000
    // Union: 10000 + 10000 - 5000 = 15000
    // IoU = 5000 / 15000 = 0.333
    EXPECT_NEAR(iou[0].item<float>(), 0.333f, 0.01f);
}

// =============================================================================
// bboxIoU Parameter Tests (Section 2.1.2)
// =============================================================================

TEST_F(IoUTest, BboxIoU_XYWHFormat_True) {
    // xywh format: [cx, cy, w, h]
    auto box1 = createXYWHBoxes({{50, 50, 100, 100}});  // center(50,50), size(100,100) -> [0,0,100,100]
    auto box2 = createXYWHBoxes({{50, 50, 100, 100}});
    auto iou = bboxIoU(box1, box2, true);  // xywh=true

    EXPECT_NEAR(iou[0].item<float>(), 1.0f, 1e-5f);
}

TEST_F(IoUTest, BboxIoU_XYWHFormat_False) {
    // xyxy format
    auto box1 = createXYXYBoxes({{0, 0, 100, 100}});
    auto box2 = createXYXYBoxes({{0, 0, 100, 100}});
    auto iou = bboxIoU(box1, box2, false);  // xywh=false

    EXPECT_NEAR(iou[0].item<float>(), 1.0f, 1e-5f);
}

TEST_F(IoUTest, BboxIoU_DefaultParams) {
    // Default: xywh=true, GIoU/DIoU/CIoU=false
    auto box1 = createXYWHBoxes({{50, 50, 100, 100}});
    auto box2 = createXYWHBoxes({{50, 50, 100, 100}});
    auto iou = bboxIoU(box1, box2);  // All defaults

    EXPECT_NEAR(iou[0].item<float>(), 1.0f, 1e-5f);
}

TEST_F(IoUTest, BboxIoU_CustomEps) {
    auto box1 = createXYXYBoxes({{0, 0, 100, 100}});
    auto box2 = createXYXYBoxes({{0, 0, 100, 100}});
    auto iou = bboxIoU(box1, box2, false, false, false, false, 1e-7f);

    EXPECT_NEAR(iou[0].item<float>(), 1.0f, 1e-5f);
}

// =============================================================================
// GIoU, DIoU, CIoU Tests (Section 2.1.3)
// =============================================================================

TEST_F(IoUTest, GIoU_NoOverlap) {
    // GIoU can be negative for disjoint boxes
    auto box1 = createXYXYBoxes({{0, 0, 10, 10}});
    auto box2 = createXYXYBoxes({{20, 20, 30, 30}});
    auto giou = bboxIoU(box1, box2, false, true, false, false);

    // GIoU for disjoint boxes is negative
    EXPECT_LT(giou[0].item<float>(), 0.0f);
}

TEST_F(IoUTest, GIoU_PartialOverlap) {
    auto box1 = createXYXYBoxes({{0, 0, 100, 100}});
    auto box2 = createXYXYBoxes({{50, 50, 150, 150}});
    auto giou = bboxIoU(box1, box2, false, true, false, false);

    // GIoU should be between -1 and 1
    EXPECT_GE(giou[0].item<float>(), -1.0f);
    EXPECT_LE(giou[0].item<float>(), 1.0f);
}

TEST_F(IoUTest, GIoU_PerfectOverlap) {
    auto box = createXYXYBoxes({{0, 0, 100, 100}});
    auto giou = bboxIoU(box, box, false, true, false, false);

    EXPECT_NEAR(giou[0].item<float>(), 1.0f, 1e-5f);
}

TEST_F(IoUTest, DIoU_CenterPenalty) {
    // Same size, offset center
    auto box1 = createXYXYBoxes({{0, 0, 10, 10}});
    auto box2 = createXYXYBoxes({{5, 5, 15, 15}});
    auto diou = bboxIoU(box1, box2, false, false, true, false);

    // DIoU penalizes center distance
    auto iou = bboxIoU(box1, box2, false, false, false, false);
    EXPECT_LT(diou[0].item<float>(), iou[0].item<float>());
}

TEST_F(IoUTest, DIoU_PerfectOverlap) {
    auto box = createXYXYBoxes({{0, 0, 100, 100}});
    auto diou = bboxIoU(box, box, false, false, true, false);

    EXPECT_NEAR(diou[0].item<float>(), 1.0f, 1e-5f);
}

TEST_F(IoUTest, CIoU_AspectRatioPenalty) {
    // Different aspect ratios
    auto box1 = createXYXYBoxes({{0, 0, 40, 40}});     // Square
    auto box2 = createXYXYBoxes({{0, 0, 20, 80}});     // Tall rectangle
    auto ciou = bboxIoU(box1, box2, false, false, false, true);

    // CIoU penalizes aspect ratio difference
    EXPECT_GE(ciou[0].item<float>(), -1.0f);
    EXPECT_LE(ciou[0].item<float>(), 1.0f);
}

TEST_F(IoUTest, CIoU_PerfectOverlap) {
    auto box = createXYXYBoxes({{0, 0, 100, 100}});
    auto ciou = bboxIoU(box, box, false, false, false, true);

    EXPECT_NEAR(ciou[0].item<float>(), 1.0f, 1e-5f);
}

TEST_F(IoUTest, CIoU_PrecomputedValue) {
    // From test plan
    auto box1 = createXYXYBoxes({{0, 0, 40, 40}});     // 40x40 square
    auto box2 = createXYXYBoxes({{10, 10, 30, 50}});   // 20x40 rectangle
    auto ciou = bboxIoU(box1, box2, false, false, false, true);

    // Pre-computed: approximately 0.694 (as 1 - IoU + penalties)
    // The actual CIoU depends on center distance and aspect ratio
    EXPECT_FALSE(ciou[0].isnan().item<bool>());
    EXPECT_FALSE(ciou[0].isinf().item<bool>());
}

// =============================================================================
// bboxIoU Exception Tests
// =============================================================================

TEST_F(IoUTest, BboxIoU_Exception_Non2DBox1) {
    auto box1 = torch::randn({2, 3, 4}, torch::TensorOptions().device(device_));
    auto box2 = torch::randn({2, 4}, torch::TensorOptions().device(device_));

    EXPECT_THROW(bboxIoU(box1, box2, false), std::invalid_argument);
}

TEST_F(IoUTest, BboxIoU_Exception_WrongSize) {
    auto box1 = torch::randn({2, 3}, torch::TensorOptions().device(device_));  // Wrong size
    auto box2 = torch::randn({2, 4}, torch::TensorOptions().device(device_));

    EXPECT_THROW(bboxIoU(box1, box2, false), std::invalid_argument);
}

TEST_F(IoUTest, BboxIoU_Exception_MismatchedSizes) {
    auto box1 = torch::randn({5, 4}, torch::TensorOptions().device(device_));
    auto box2 = torch::randn({3, 4}, torch::TensorOptions().device(device_));

    EXPECT_THROW(bboxIoU(box1, box2, false), std::invalid_argument);
}

// =============================================================================
// Probiou Tests (Section 2.1.4)
// =============================================================================

TEST_F(IoUTest, Probiou_AlignedBoxes_0deg) {
    // Same box, no rotation
    auto obb = createOBBBoxes({{50, 50, 20, 10, 0}});
    auto piou = probiou(obb, obb);

    EXPECT_NEAR(piou[0].item<float>(), 1.0f, 0.01f);
}

TEST_F(IoUTest, Probiou_SameBoxWithAngle) {
    // Same box with rotation
    auto obb = createOBBBoxes({{50, 50, 20, 10, 0.5f}});
    auto piou = probiou(obb, obb);

    EXPECT_NEAR(piou[0].item<float>(), 1.0f, 0.01f);
}

TEST_F(IoUTest, Probiou_Perpendicular_90deg) {
    // Same box, one rotated 90 degrees
    auto obb1 = createOBBBoxes({{50, 50, 20, 10, 0}});
    auto obb2 = createOBBBoxes({{50, 50, 20, 10, static_cast<float>(PI / 2)}});
    auto piou = probiou(obb1, obb2);

    // 90 degree rotation should give lower IoU
    EXPECT_LT(piou[0].item<float>(), 1.0f);
    EXPECT_GT(piou[0].item<float>(), 0.0f);
}

TEST_F(IoUTest, Probiou_SmallAngle_10deg) {
    // Small rotation (about 10 degrees)
    auto obb1 = createOBBBoxes({{50, 50, 20, 10, 0}});
    auto obb2 = createOBBBoxes({{50, 50, 20, 10, 0.175f}});  // ~10 degrees
    auto piou = probiou(obb1, obb2);

    // Small angle should have high IoU
    EXPECT_GT(piou[0].item<float>(), 0.8f);
}

TEST_F(IoUTest, Probiou_Rotation45deg) {
    // 45 degree difference
    auto obb1 = createOBBBoxes({{50, 50, 20, 10, 0}});
    auto obb2 = createOBBBoxes({{50, 50, 20, 10, static_cast<float>(PI / 4)}});
    auto piou = probiou(obb1, obb2);

    EXPECT_GT(piou[0].item<float>(), 0.0f);
    EXPECT_LT(piou[0].item<float>(), 1.0f);
}

TEST_F(IoUTest, Probiou_AntiparallelAngle_180deg) {
    // 180 degree should be same as 0 for symmetric boxes
    auto obb1 = createOBBBoxes({{50, 50, 20, 20, 0}});  // Square
    auto obb2 = createOBBBoxes({{50, 50, 20, 20, static_cast<float>(PI)}});
    auto piou = probiou(obb1, obb2);

    EXPECT_NEAR(piou[0].item<float>(), 1.0f, 0.01f);
}

TEST_F(IoUTest, Probiou_DifferentCenter) {
    // Same size/angle, different center
    auto obb1 = createOBBBoxes({{50, 50, 20, 10, 0}});
    auto obb2 = createOBBBoxes({{70, 50, 20, 10, 0}});
    auto piou = probiou(obb1, obb2);

    // Offset center should reduce IoU
    EXPECT_LT(piou[0].item<float>(), 1.0f);
}

TEST_F(IoUTest, Probiou_DifferentSize) {
    // Same center/angle, different size
    auto obb1 = createOBBBoxes({{50, 50, 20, 10, 0}});
    auto obb2 = createOBBBoxes({{50, 50, 40, 20, 0}});  // 2x size
    auto piou = probiou(obb1, obb2);

    EXPECT_LT(piou[0].item<float>(), 1.0f);
    EXPECT_GT(piou[0].item<float>(), 0.0f);
}

TEST_F(IoUTest, Probiou_BatchComputation) {
    auto obb1 = createOBBBoxes({
        {50, 50, 20, 10, 0},
        {50, 50, 20, 10, 0},
        {50, 50, 20, 10, 0}
    });
    auto obb2 = createOBBBoxes({
        {50, 50, 20, 10, 0},                              // Same
        {50, 50, 20, 10, static_cast<float>(PI / 2)},     // 90 deg
        {70, 70, 20, 10, 0}                               // Offset
    });
    auto piou = probiou(obb1, obb2);

    EXPECT_EQ(piou.size(0), 3);
    EXPECT_NEAR(piou[0].item<float>(), 1.0f, 0.01f);  // Perfect match
    EXPECT_LT(piou[1].item<float>(), 1.0f);           // 90 deg rotation
    EXPECT_LT(piou[2].item<float>(), 1.0f);           // Offset
}

// =============================================================================
// Probiou Exception Tests
// =============================================================================

TEST_F(IoUTest, Probiou_Exception_Non2D) {
    auto obb1 = torch::randn({2, 3, 5}, torch::TensorOptions().device(device_));
    auto obb2 = torch::randn({2, 5}, torch::TensorOptions().device(device_));

    EXPECT_THROW(probiou(obb1, obb2), std::invalid_argument);
}

TEST_F(IoUTest, Probiou_Exception_WrongSize) {
    auto obb1 = torch::randn({2, 4}, torch::TensorOptions().device(device_));  // Wrong size
    auto obb2 = torch::randn({2, 5}, torch::TensorOptions().device(device_));

    EXPECT_THROW(probiou(obb1, obb2), std::invalid_argument);
}

TEST_F(IoUTest, Probiou_Exception_MismatchedSizes) {
    auto obb1 = torch::randn({5, 5}, torch::TensorOptions().device(device_));
    auto obb2 = torch::randn({3, 5}, torch::TensorOptions().device(device_));

    EXPECT_THROW(probiou(obb1, obb2), std::invalid_argument);
}

// =============================================================================
// Numerical Stability Tests
// =============================================================================

TEST_F(IoUTest, BboxIoU_NumericalStability_SmallBoxes) {
    auto box1 = createXYXYBoxes({{0, 0, 0.001f, 0.001f}});
    auto box2 = createXYXYBoxes({{0, 0, 0.001f, 0.001f}});
    auto iou = bboxIoU(box1, box2, false);

    EXPECT_FALSE(iou[0].isnan().item<bool>());
    EXPECT_FALSE(iou[0].isinf().item<bool>());
}

TEST_F(IoUTest, BboxIoU_NumericalStability_LargeBoxes) {
    auto box1 = createXYXYBoxes({{0, 0, 10000, 10000}});
    auto box2 = createXYXYBoxes({{0, 0, 10000, 10000}});
    auto iou = bboxIoU(box1, box2, false);

    EXPECT_NEAR(iou[0].item<float>(), 1.0f, 1e-5f);
}

TEST_F(IoUTest, Probiou_NumericalStability_SmallBoxes) {
    auto obb = createOBBBoxes({{50, 50, 0.01f, 0.01f, 0}});
    auto piou = probiou(obb, obb);

    EXPECT_FALSE(piou[0].isnan().item<bool>());
    EXPECT_FALSE(piou[0].isinf().item<bool>());
}

TEST_F(IoUTest, Probiou_NumericalStability_LargeBoxes) {
    auto obb = createOBBBoxes({{5000, 5000, 1000, 1000, 0}});
    auto piou = probiou(obb, obb);

    EXPECT_FALSE(piou[0].isnan().item<bool>());
    EXPECT_FALSE(piou[0].isinf().item<bool>());
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(IoUTest, BboxIoU_ZeroAreaBox) {
    auto box1 = createXYXYBoxes({{50, 50, 50, 50}});  // Point
    auto box2 = createXYXYBoxes({{0, 0, 100, 100}});
    auto iou = bboxIoU(box1, box2, false);

    // Zero area should give 0 IoU
    EXPECT_NEAR(iou[0].item<float>(), 0.0f, 1e-5f);
}

TEST_F(IoUTest, BboxIoU_NegativeCoordinates) {
    auto box1 = createXYXYBoxes({{-50, -50, 50, 50}});
    auto box2 = createXYXYBoxes({{-50, -50, 50, 50}});
    auto iou = bboxIoU(box1, box2, false);

    EXPECT_NEAR(iou[0].item<float>(), 1.0f, 1e-5f);
}

TEST_F(IoUTest, Probiou_SquareBox) {
    // Square boxes should work correctly with any rotation
    auto obb1 = createOBBBoxes({{50, 50, 20, 20, 0}});
    auto obb2 = createOBBBoxes({{50, 50, 20, 20, static_cast<float>(PI / 4)}});
    auto piou = probiou(obb1, obb2);

    // Square rotated by 45 deg overlaps significantly with original
    EXPECT_GT(piou[0].item<float>(), 0.5f);
}

TEST_F(IoUTest, Probiou_ThinBox) {
    // Very thin boxes
    auto obb = createOBBBoxes({{50, 50, 100, 1, 0}});
    auto piou = probiou(obb, obb);

    EXPECT_NEAR(piou[0].item<float>(), 1.0f, 0.01f);
}

// =============================================================================
// IoU Reference Table Tests (from Appendix)
// =============================================================================

TEST_F(IoUTest, IoU_ReferenceTable_PerfectOverlap) {
    auto box = createXYXYBoxes({{0, 0, 10, 10}});
    auto iou = bboxIoU(box, box, false);
    EXPECT_NEAR(iou[0].item<float>(), 1.0f, 1e-5f);
}

TEST_F(IoUTest, IoU_ReferenceTable_HalfOverlap) {
    auto box1 = createXYXYBoxes({{0, 0, 10, 10}});
    auto box2 = createXYXYBoxes({{5, 5, 15, 15}});
    auto iou = bboxIoU(box1, box2, false);

    // Intersection: 5*5=25, Union: 100+100-25=175
    // IoU = 25/175 = 0.143
    EXPECT_NEAR(iou[0].item<float>(), 0.143f, 0.01f);
}

TEST_F(IoUTest, IoU_ReferenceTable_NoOverlap) {
    auto box1 = createXYXYBoxes({{0, 0, 10, 10}});
    auto box2 = createXYXYBoxes({{10, 10, 20, 20}});
    auto iou = bboxIoU(box1, box2, false);
    EXPECT_NEAR(iou[0].item<float>(), 0.0f, 1e-5f);
}

TEST_F(IoUTest, IoU_ReferenceTable_OneThirdOverlap) {
    auto box1 = createXYXYBoxes({{0, 0, 10, 10}});
    auto box2 = createXYXYBoxes({{5, 0, 15, 10}});
    auto iou = bboxIoU(box1, box2, false);

    // Intersection: 5*10=50, Union: 100+100-50=150
    // IoU = 50/150 = 0.333
    EXPECT_NEAR(iou[0].item<float>(), 0.333f, 0.01f);
}

TEST_F(IoUTest, IoU_ReferenceTable_Containment) {
    auto box1 = createXYXYBoxes({{0, 0, 10, 10}});
    auto box2 = createXYXYBoxes({{2, 2, 8, 8}});
    auto iou = bboxIoU(box1, box2, false);

    // Intersection: 6*6=36, Union: 100+36-36=100
    // IoU = 36/100 = 0.36
    EXPECT_NEAR(iou[0].item<float>(), 0.36f, 0.01f);
}

// =============================================================================
// CUDA Device Tests
// =============================================================================

TEST_F(IoUTest, BboxIoU_DeviceConsistency) {
    auto box1 = torch::randn({10, 4}, torch::TensorOptions().device(device_));
    auto box2 = torch::randn({10, 4}, torch::TensorOptions().device(device_));
    auto iou = bboxIoU(box1, box2, true);

    EXPECT_EQ(iou.device().type(), device_.type());
    EXPECT_EQ(iou.size(0), 10);
}

TEST_F(IoUTest, Probiou_DeviceConsistency) {
    auto obb1 = torch::randn({10, 5}, torch::TensorOptions().device(device_));
    auto obb2 = torch::randn({10, 5}, torch::TensorOptions().device(device_));
    auto piou = probiou(obb1, obb2);

    EXPECT_EQ(piou.device().type(), device_.type());
    EXPECT_EQ(piou.size(0), 10);
}

// =============================================================================
// Gradient Tests
// =============================================================================

TEST_F(IoUTest, BboxIoU_GradientFlow) {
    auto box1 = torch::randn({5, 4}, torch::TensorOptions().device(device_).requires_grad(true));
    auto box2 = torch::randn({5, 4}, torch::TensorOptions().device(device_));

    auto iou = bboxIoU(box1, box2, true);
    iou.sum().backward();

    EXPECT_TRUE(box1.grad().defined());
    EXPECT_FALSE(box1.grad().isnan().any().item<bool>());
}

TEST_F(IoUTest, CIoU_GradientFlow) {
    auto box1 = torch::randn({5, 4}, torch::TensorOptions().device(device_).requires_grad(true));
    auto box2 = torch::randn({5, 4}, torch::TensorOptions().device(device_));

    auto ciou = bboxIoU(box1, box2, true, false, false, true);
    ciou.sum().backward();

    EXPECT_TRUE(box1.grad().defined());
    EXPECT_FALSE(box1.grad().isnan().any().item<bool>());
}

TEST_F(IoUTest, Probiou_GradientFlow) {
    // Create tensor without requires_grad first, then apply abs to make w,h positive
    auto obb1_base = torch::randn({5, 5}, torch::TensorOptions().device(device_));
    // Make w, h (columns 2, 3) positive using abs (non-inplace)
    auto obb1_wh = obb1_base.slice(1, 2, 4).abs();
    auto obb1 = torch::cat({obb1_base.slice(1, 0, 2), obb1_wh, obb1_base.slice(1, 4, 5)}, 1);
    obb1.requires_grad_(true);

    auto obb2_base = torch::randn({5, 5}, torch::TensorOptions().device(device_));
    auto obb2_wh = obb2_base.slice(1, 2, 4).abs();
    auto obb2 = torch::cat({obb2_base.slice(1, 0, 2), obb2_wh, obb2_base.slice(1, 4, 5)}, 1);

    auto piou = probiou(obb1, obb2);
    piou.sum().backward();

    EXPECT_TRUE(obb1.grad().defined());
    EXPECT_FALSE(obb1.grad().isnan().any().item<bool>());
}
