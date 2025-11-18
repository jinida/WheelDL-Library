#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Utils/IoU.h"
#include <torch/torch.h>

using namespace WheelDL::Model::Utils;

/**
 * @class IoUComputationTest
 * @brief Unit tests for IoU computation functions
 *
 * Tests cover:
 * - Standard IoU computation
 * - GIoU (Generalized IoU)
 * - DIoU (Distance IoU)
 * - CIoU (Complete IoU)
 * - Edge cases (zero area, large coordinates, etc.)
 * - Batch processing
 * - Different formats (xywh vs xyxy)
 */
class IoUComputationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set random seed for reproducibility
        torch::manual_seed(42);

        // Tolerance for floating point comparisons
        tolerance = 1e-5f;
    }

    float tolerance;
};

// ============================================================================
// Standard IoU Tests
// ============================================================================

TEST_F(IoUComputationTest, StandardIoU_NoOverlap_ReturnsZero) {
    // Arrange - Two non-overlapping boxes
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{30.0f, 30.0f, 40.0f, 40.0f}});

    // Act
    auto iou = bboxIoU(box1, box2, /*xywh=*/true);

    // Assert
    EXPECT_NEAR(iou.item<float>(), 0.0f, tolerance);
}

TEST_F(IoUComputationTest, StandardIoU_PerfectOverlap_ReturnsOne) {
    // Arrange - Same box
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});

    // Act
    auto iou = bboxIoU(box1, box2, /*xywh=*/true);

    // Assert
    EXPECT_NEAR(iou.item<float>(), 1.0f, tolerance);
}

TEST_F(IoUComputationTest, StandardIoU_PartialOverlap_ReturnsCorrectValue) {
    // Arrange - Two overlapping boxes
    // Box1: center (10, 10), size (20, 20) -> corners (0, 0) to (20, 20)
    // Box2: center (15, 15), size (20, 20) -> corners (5, 5) to (25, 25)
    // Intersection: (5, 5) to (20, 20) = 15x15 = 225
    // Union: 400 + 400 - 225 = 575
    // IoU: 225/575 ≈ 0.391304
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{15.0f, 15.0f, 20.0f, 20.0f}});

    // Act
    auto iou = bboxIoU(box1, box2, /*xywh=*/true);

    // Assert
    float expected_iou = 225.0f / 575.0f;
    EXPECT_NEAR(iou.item<float>(), expected_iou, tolerance);
}

TEST_F(IoUComputationTest, StandardIoU_ContainedBox_ReturnsCorrectValue) {
    // Arrange - Small box fully contained in large box
    // Box1: center (50, 50), size (100, 100) -> corners (0, 0) to (100, 100)
    // Box2: center (50, 50), size (50, 50) -> corners (25, 25) to (75, 75)
    // Intersection: 2500
    // Union: 10000 + 2500 - 2500 = 10000
    // IoU: 2500/10000 = 0.25
    auto box1 = torch::tensor({{50.0f, 50.0f, 100.0f, 100.0f}});
    auto box2 = torch::tensor({{50.0f, 50.0f, 50.0f, 50.0f}});

    // Act
    auto iou = bboxIoU(box1, box2, /*xywh=*/true);

    // Assert
    EXPECT_NEAR(iou.item<float>(), 0.25f, tolerance);
}

TEST_F(IoUComputationTest, StandardIoU_XyxyFormat_ReturnsCorrectValue) {
    // Arrange - Test with xyxy format
    // Box1: (0, 0) to (20, 20)
    // Box2: (10, 10) to (30, 30)
    // Intersection: (10, 10) to (20, 20) = 10x10 = 100
    // Union: 400 + 400 - 100 = 700
    // IoU: 100/700 ≈ 0.142857
    auto box1 = torch::tensor({{0.0f, 0.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{10.0f, 10.0f, 30.0f, 30.0f}});

    // Act
    auto iou = bboxIoU(box1, box2, /*xywh=*/false);

    // Assert
    float expected_iou = 100.0f / 700.0f;
    EXPECT_NEAR(iou.item<float>(), expected_iou, tolerance);
}

TEST_F(IoUComputationTest, StandardIoU_BatchProcessing_ReturnsCorrectMatrix) {
    // Arrange - Multiple boxes
    auto boxes1 = torch::tensor({
        {10.0f, 10.0f, 20.0f, 20.0f},
        {30.0f, 30.0f, 20.0f, 20.0f}
    });
    auto boxes2 = torch::tensor({
        {10.0f, 10.0f, 20.0f, 20.0f},
        {30.0f, 30.0f, 20.0f, 20.0f},
        {50.0f, 50.0f, 20.0f, 20.0f}
    });

    // Act
    auto iou = bboxIoU(boxes1, boxes2, /*xywh=*/true);

    // Assert - Check shape [2, 3]
    EXPECT_EQ(iou.size(0), 2);
    EXPECT_EQ(iou.size(1), 3);

    // Check diagonal-like elements (perfect overlap)
    EXPECT_NEAR(iou[0][0].item<float>(), 1.0f, tolerance);
    EXPECT_NEAR(iou[1][1].item<float>(), 1.0f, tolerance);

    // Check non-overlapping elements
    EXPECT_NEAR(iou[0][2].item<float>(), 0.0f, tolerance);
}

// ============================================================================
// GIoU Tests
// ============================================================================

TEST_F(IoUComputationTest, GIoU_NoOverlap_ReturnsNegativeValue) {
    // Arrange - Non-overlapping boxes
    // GIoU should be negative when boxes don't overlap
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{40.0f, 40.0f, 20.0f, 20.0f}});

    // Act
    auto giou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/true);

    // Assert - Should be negative
    EXPECT_LT(giou.item<float>(), 0.0f);
}

TEST_F(IoUComputationTest, GIoU_PerfectOverlap_ReturnsOne) {
    // Arrange - Same box
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});

    // Act
    auto giou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/true);

    // Assert
    EXPECT_NEAR(giou.item<float>(), 1.0f, tolerance);
}

TEST_F(IoUComputationTest, GIoU_PartialOverlap_ReturnsCorrectValue) {
    // Arrange - Overlapping boxes
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{15.0f, 15.0f, 20.0f, 20.0f}});

    // Act
    auto giou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/true);

    // Assert - GIoU should be >= standard IoU
    auto standard_iou = bboxIoU(box1, box2, /*xywh=*/true);
    EXPECT_GE(giou.item<float>(), standard_iou.item<float>() - tolerance);
}

TEST_F(IoUComputationTest, GIoU_EnclosingBoxComputation_WorksCorrectly) {
    // Arrange - Test enclosing box calculation
    auto box1 = torch::tensor({{10.0f, 10.0f, 10.0f, 10.0f}});
    auto box2 = torch::tensor({{20.0f, 20.0f, 10.0f, 10.0f}});

    // Act
    auto giou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/true);

    // Assert - GIoU should be in valid range [-1, 1]
    EXPECT_GE(giou.item<float>(), -1.0f - tolerance);
    EXPECT_LE(giou.item<float>(), 1.0f + tolerance);
}

TEST_F(IoUComputationTest, GIoU_BetterThanStandardIoU_ForNonOverlapping) {
    // Arrange - Non-overlapping boxes
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{35.0f, 35.0f, 20.0f, 20.0f}});

    // Act
    auto standard_iou = bboxIoU(box1, box2, /*xywh=*/true);
    auto giou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/true);

    // Assert - Standard IoU is 0, but GIoU provides gradient
    EXPECT_NEAR(standard_iou.item<float>(), 0.0f, tolerance);
    EXPECT_LT(giou.item<float>(), 0.0f);  // GIoU is negative
}

// ============================================================================
// DIoU Tests
// ============================================================================

TEST_F(IoUComputationTest, DIoU_DistancePenalty_IsApplied) {
    // Arrange - Two boxes with same IoU but different center distances
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2_near = torch::tensor({{12.0f, 12.0f, 20.0f, 20.0f}});
    auto box2_far = torch::tensor({{15.0f, 15.0f, 20.0f, 20.0f}});

    // Act
    auto diou_near = bboxIoU(box1, box2_near, /*xywh=*/true, /*GIoU=*/false, /*DIoU=*/true);
    auto diou_far = bboxIoU(box1, box2_far, /*xywh=*/true, /*GIoU=*/false, /*DIoU=*/true);

    // Assert - Closer box should have higher DIoU
    EXPECT_GT(diou_near.item<float>(), diou_far.item<float>());
}

TEST_F(IoUComputationTest, DIoU_PerfectOverlap_ReturnsOne) {
    // Arrange - Same box
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});

    // Act
    auto diou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/false, /*DIoU=*/true);

    // Assert
    EXPECT_NEAR(diou.item<float>(), 1.0f, tolerance);
}

TEST_F(IoUComputationTest, DIoU_CenterDistanceConsidered_Correctly) {
    // Arrange - Boxes with overlap but different centers
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{10.0f, 15.0f, 20.0f, 20.0f}});

    // Act
    auto diou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/false, /*DIoU=*/true);
    auto standard_iou = bboxIoU(box1, box2, /*xywh=*/true);

    // Assert - DIoU should be less than standard IoU due to distance penalty
    EXPECT_LT(diou.item<float>(), standard_iou.item<float>());
}

TEST_F(IoUComputationTest, DIoU_DiagonalDistance_ComputedCorrectly) {
    // Arrange - Test diagonal computation
    auto box1 = torch::tensor({{0.0f, 0.0f, 10.0f, 10.0f}});
    auto box2 = torch::tensor({{5.0f, 5.0f, 10.0f, 10.0f}});

    // Act
    auto diou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/false, /*DIoU=*/true);

    // Assert - DIoU should be in valid range
    EXPECT_GE(diou.item<float>(), -1.0f);
    EXPECT_LE(diou.item<float>(), 1.0f);
}

TEST_F(IoUComputationTest, DIoU_BetterThanGIoU_ConsidersDistance) {
    // Arrange - Boxes where distance matters
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{15.0f, 15.0f, 20.0f, 20.0f}});

    // Act
    auto giou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/true);
    auto diou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/false, /*DIoU=*/true);

    // Assert - Both should be valid
    EXPECT_GT(giou.item<float>(), 0.0f);
    EXPECT_GT(diou.item<float>(), 0.0f);
}

// ============================================================================
// CIoU Tests
// ============================================================================

TEST_F(IoUComputationTest, CIoU_AllPenalties_AreApplied) {
    // Arrange - Boxes with different overlap, distance, and aspect ratios
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{15.0f, 15.0f, 20.0f, 30.0f}});  // Different aspect

    // Act
    auto ciou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/false, /*DIoU=*/false, /*CIoU=*/true);

    // Assert - CIoU should be valid
    EXPECT_GE(ciou.item<float>(), -1.0f);
    EXPECT_LE(ciou.item<float>(), 1.0f);
}

TEST_F(IoUComputationTest, CIoU_PerfectOverlap_ReturnsOne) {
    // Arrange - Same box
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});

    // Act
    auto ciou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/false, /*DIoU=*/false, /*CIoU=*/true);

    // Assert
    EXPECT_NEAR(ciou.item<float>(), 1.0f, tolerance);
}

TEST_F(IoUComputationTest, CIoU_AspectRatioTerm_IsConsidered) {
    // Arrange - Boxes with same center but different aspect ratios
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});  // Square
    auto box2_similar = torch::tensor({{10.0f, 10.0f, 20.0f, 22.0f}});  // Similar aspect
    auto box2_different = torch::tensor({{10.0f, 10.0f, 20.0f, 40.0f}});  // Different aspect

    // Act
    auto ciou_similar = bboxIoU(box1, box2_similar, /*xywh=*/true, /*GIoU=*/false, /*DIoU=*/false, /*CIoU=*/true);
    auto ciou_different = bboxIoU(box1, box2_different, /*xywh=*/true, /*GIoU=*/false, /*DIoU=*/false, /*CIoU=*/true);

    // Assert - Similar aspect ratio should have higher CIoU
    EXPECT_GT(ciou_similar.item<float>(), ciou_different.item<float>());
}

TEST_F(IoUComputationTest, CIoU_AlphaVComputation_IsCorrect) {
    // Arrange - Test alpha-v computation
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{12.0f, 12.0f, 20.0f, 30.0f}});

    // Act
    auto ciou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/false, /*DIoU=*/false, /*CIoU=*/true);

    // Assert - Should be valid value
    EXPECT_GE(ciou.item<float>(), -1.0f);
    EXPECT_LE(ciou.item<float>(), 1.0f);
}

TEST_F(IoUComputationTest, CIoU_BestOfAllVariants_MostComprehensive) {
    // Arrange - Test comprehensive penalty
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{15.0f, 15.0f, 25.0f, 25.0f}});

    // Act - Compare all variants
    auto standard_iou = bboxIoU(box1, box2, /*xywh=*/true);
    auto giou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/true);
    auto diou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/false, /*DIoU=*/true);
    auto ciou = bboxIoU(box1, box2, /*xywh=*/true, /*GIoU=*/false, /*DIoU=*/false, /*CIoU=*/true);

    // Assert - All should be valid
    EXPECT_GT(standard_iou.item<float>(), 0.0f);
    EXPECT_GT(giou.item<float>(), 0.0f);
    EXPECT_GT(diou.item<float>(), 0.0f);
    EXPECT_GT(ciou.item<float>(), 0.0f);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(IoUComputationTest, EdgeCase_ZeroAreaBoxes_HandledCorrectly) {
    // Arrange - Box with zero area
    auto box1 = torch::tensor({{10.0f, 10.0f, 0.0f, 0.0f}});
    auto box2 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});

    // Act
    auto iou = bboxIoU(box1, box2, /*xywh=*/true);

    // Assert - Should be 0 (or close to 0)
    EXPECT_NEAR(iou.item<float>(), 0.0f, tolerance);
}

TEST_F(IoUComputationTest, EdgeCase_NegativeCoordinates_WorkCorrectly) {
    // Arrange - Boxes with negative coordinates
    auto box1 = torch::tensor({{-10.0f, -10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{-5.0f, -5.0f, 20.0f, 20.0f}});

    // Act
    auto iou = bboxIoU(box1, box2, /*xywh=*/true);

    // Assert - Should compute valid IoU
    EXPECT_GE(iou.item<float>(), 0.0f);
    EXPECT_LE(iou.item<float>(), 1.0f);
}

TEST_F(IoUComputationTest, EdgeCase_VeryLargeCoordinates_HandledCorrectly) {
    // Arrange - Boxes with large coordinates
    auto box1 = torch::tensor({{1000.0f, 1000.0f, 200.0f, 200.0f}});
    auto box2 = torch::tensor({{1050.0f, 1050.0f, 200.0f, 200.0f}});

    // Act
    auto iou = bboxIoU(box1, box2, /*xywh=*/true);

    // Assert - Should compute valid IoU
    EXPECT_GE(iou.item<float>(), 0.0f);
    EXPECT_LE(iou.item<float>(), 1.0f);
}

TEST_F(IoUComputationTest, EdgeCase_SingleBox_ReturnsCorrectShape) {
    // Arrange - Single box vs single box
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});

    // Act
    auto iou = bboxIoU(box1, box2, /*xywh=*/true);

    // Assert - Shape should be [1, 1]
    EXPECT_EQ(iou.dim(), 2);
    EXPECT_EQ(iou.size(0), 1);
    EXPECT_EQ(iou.size(1), 1);
}

TEST_F(IoUComputationTest, EdgeCase_EmptyTensors_ThrowsException) {
    // Arrange - Invalid empty tensors
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto box2 = torch::empty({0, 4});

    // Act & Assert
    EXPECT_THROW(
        {
            auto iou = bboxIoU(box1, box2, /*xywh=*/true);
        },
        std::exception
    );
}

// ============================================================================
// Batch & Format Tests
// ============================================================================

TEST_F(IoUComputationTest, Batch_MultipleBoxes_ReturnsCorrectMatrix) {
    // Arrange - Test NxM matrix computation
    auto boxes1 = torch::tensor({
        {10.0f, 10.0f, 20.0f, 20.0f},
        {30.0f, 30.0f, 20.0f, 20.0f},
        {50.0f, 50.0f, 20.0f, 20.0f}
    });
    auto boxes2 = torch::tensor({
        {10.0f, 10.0f, 20.0f, 20.0f},
        {50.0f, 50.0f, 20.0f, 20.0f}
    });

    // Act
    auto iou = bboxIoU(boxes1, boxes2, /*xywh=*/true);

    // Assert - Shape [3, 2]
    EXPECT_EQ(iou.size(0), 3);
    EXPECT_EQ(iou.size(1), 2);

    // Check specific values
    EXPECT_NEAR(iou[0][0].item<float>(), 1.0f, tolerance);  // Perfect match
    EXPECT_NEAR(iou[2][1].item<float>(), 1.0f, tolerance);  // Perfect match
    EXPECT_NEAR(iou[0][1].item<float>(), 0.0f, tolerance);  // No overlap
}

TEST_F(IoUComputationTest, Batch_LargeBatch_ProcessesEfficiently) {
    // Arrange - Large batch of 1000+ boxes
    auto boxes1 = torch::rand({1000, 4}) * 100.0f;
    boxes1.slice(1, 2, 4) = torch::abs(boxes1.slice(1, 2, 4)) + 1.0f;  // Ensure positive sizes

    auto boxes2 = torch::rand({100, 4}) * 100.0f;
    boxes2.slice(1, 2, 4) = torch::abs(boxes2.slice(1, 2, 4)) + 1.0f;

    // Act
    auto iou = bboxIoU(boxes1, boxes2, /*xywh=*/true);

    // Assert - Shape [1000, 100]
    EXPECT_EQ(iou.size(0), 1000);
    EXPECT_EQ(iou.size(1), 100);

    // All IoU values should be in [0, 1]
    auto iou_min = iou.min().item<float>();
    auto iou_max = iou.max().item<float>();
    EXPECT_GE(iou_min, 0.0f);
    EXPECT_LE(iou_max, 1.0f);
}

TEST_F(IoUComputationTest, Format_XywhFormat_ConvertsCorrectly) {
    // Arrange - xywh format
    auto box1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});  // center (10, 10), size (20, 20)
    auto box2 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});

    // Act
    auto iou = bboxIoU(box1, box2, /*xywh=*/true);

    // Assert
    EXPECT_NEAR(iou.item<float>(), 1.0f, tolerance);
}

TEST_F(IoUComputationTest, Format_XyxyFormat_WorksDirectly) {
    // Arrange - xyxy format
    auto box1 = torch::tensor({{0.0f, 0.0f, 20.0f, 20.0f}});  // corners (0, 0) to (20, 20)
    auto box2 = torch::tensor({{0.0f, 0.0f, 20.0f, 20.0f}});

    // Act
    auto iou = bboxIoU(box1, box2, /*xywh=*/false);

    // Assert
    EXPECT_NEAR(iou.item<float>(), 1.0f, tolerance);
}
