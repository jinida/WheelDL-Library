#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Utils/IoU.h"
#include <torch/torch.h>

using namespace WheelDL::Model::Utils;

/**
 * @class ProbIoUTest
 * @brief Unit tests for Probabilistic IoU (Probiou) computation
 *
 * Tests cover:
 * - Basic Probiou computation
 * - Rotation handling
 * - Edge cases (zero area, extreme aspect ratios)
 * - Batch processing
 * - Comparison with standard IoU when angle=0
 */
class ProbIoUTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set random seed for reproducibility
        torch::manual_seed(42);

        // Tolerance for floating point comparisons
        tolerance = 1e-4f;  // Slightly looser tolerance for Probiou
    }

    float tolerance;
};

// ============================================================================
// Basic Probiou Tests
// ============================================================================

TEST_F(ProbIoUTest, BasicProbiou_NoRotation_MatchesStandardIoU) {
    // Arrange - Two rotated boxes with angle=0 (no rotation)
    // Should behave like standard IoU
    auto obb1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f, 0.0f}});
    auto obb2 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f, 0.0f}});

    // Act
    auto probiou_result = probiou(obb1, obb2);

    // Assert - Should be close to 1.0 (perfect overlap)
    EXPECT_NEAR(probiou_result.item<float>(), 1.0f, tolerance);
}

TEST_F(ProbIoUTest, BasicProbiou_90DegreeRotation_WorksCorrectly) {
    // Arrange - Square box rotated 90 degrees (should still overlap perfectly)
    auto obb1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f, 0.0f}});
    auto obb2 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f, static_cast<float>(M_PI / 2.0)}});

    // Act
    auto probiou_result = probiou(obb1, obb2);

    // Assert - Should be close to 1.0 for square (rotation doesn't matter)
    EXPECT_NEAR(probiou_result.item<float>(), 1.0f, tolerance);
}

TEST_F(ProbIoUTest, BasicProbiou_PerfectOverlap_ReturnsOne) {
    // Arrange - Identical rotated boxes
    auto obb1 = torch::tensor({{50.0f, 50.0f, 30.0f, 40.0f, 0.5f}});
    auto obb2 = torch::tensor({{50.0f, 50.0f, 30.0f, 40.0f, 0.5f}});

    // Act
    auto probiou_result = probiou(obb1, obb2);

    // Assert
    EXPECT_NEAR(probiou_result.item<float>(), 1.0f, tolerance);
}

TEST_F(ProbIoUTest, BasicProbiou_NoOverlap_ReturnsZero) {
    // Arrange - Non-overlapping rotated boxes
    auto obb1 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f, 0.0f}});
    auto obb2 = torch::tensor({{100.0f, 100.0f, 20.0f, 20.0f, 0.0f}});

    // Act
    auto probiou_result = probiou(obb1, obb2);

    // Assert - Should be close to 0
    EXPECT_NEAR(probiou_result.item<float>(), 0.0f, tolerance);
}

TEST_F(ProbIoUTest, BasicProbiou_SmallAngle_HandledCorrectly) {
    // Arrange - Small rotation angle
    auto obb1 = torch::tensor({{10.0f, 10.0f, 30.0f, 40.0f, 0.0f}});
    auto obb2 = torch::tensor({{10.0f, 10.0f, 30.0f, 40.0f, 0.1f}});

    // Act
    auto probiou_result = probiou(obb1, obb2);

    // Assert - Should be high but less than 1.0
    EXPECT_GT(probiou_result.item<float>(), 0.8f);
    EXPECT_LT(probiou_result.item<float>(), 1.0f + tolerance);
}

// ============================================================================
// Rotation Handling Tests
// ============================================================================

TEST_F(ProbIoUTest, Rotation_LargeAngleDifference_ReducesIoU) {
    // Arrange - Large rotation difference
    auto obb1 = torch::tensor({{10.0f, 10.0f, 40.0f, 20.0f, 0.0f}});  // Horizontal
    auto obb2 = torch::tensor({{10.0f, 10.0f, 40.0f, 20.0f, static_cast<float>(M_PI / 2.0)}});  // Vertical

    // Act
    auto probiou_result = probiou(obb1, obb2);

    // Assert - Should be less than perfect overlap
    EXPECT_LT(probiou_result.item<float>(), 1.0f);
    EXPECT_GT(probiou_result.item<float>(), 0.0f);
}

TEST_F(ProbIoUTest, Rotation_Periodicity_0Equals2Pi) {
    // Arrange - Test angle periodicity (0 == 2π)
    auto obb1 = torch::tensor({{10.0f, 10.0f, 20.0f, 30.0f, 0.0f}});
    auto obb2 = torch::tensor({{10.0f, 10.0f, 20.0f, 30.0f, static_cast<float>(2.0 * M_PI)}});

    // Act
    auto probiou_result = probiou(obb1, obb2);

    // Assert - Should be approximately 1.0
    EXPECT_NEAR(probiou_result.item<float>(), 1.0f, tolerance);
}

TEST_F(ProbIoUTest, Rotation_NegativeAngles_HandledCorrectly) {
    // Arrange - Negative angle
    auto obb1 = torch::tensor({{10.0f, 10.0f, 20.0f, 30.0f, 0.5f}});
    auto obb2 = torch::tensor({{10.0f, 10.0f, 20.0f, 30.0f, -0.5f}});

    // Act
    auto probiou_result = probiou(obb1, obb2);

    // Assert - Should compute valid IoU
    EXPECT_GE(probiou_result.item<float>(), 0.0f);
    EXPECT_LE(probiou_result.item<float>(), 1.0f + tolerance);
}

TEST_F(ProbIoUTest, Rotation_AngleNormalization_WorksCorrectly) {
    // Arrange - Test angle normalization
    auto obb1 = torch::tensor({{10.0f, 10.0f, 20.0f, 30.0f, 0.0f}});
    auto obb2 = torch::tensor({{10.0f, 10.0f, 20.0f, 30.0f, static_cast<float>(4.0 * M_PI)}});

    // Act
    auto probiou_result = probiou(obb1, obb2);

    // Assert - Should be approximately 1.0 (4π == 0)
    EXPECT_NEAR(probiou_result.item<float>(), 1.0f, tolerance);
}

TEST_F(ProbIoUTest, Rotation_VariousAspectRatios_ComputeCorrectly) {
    // Arrange - Different aspect ratios with rotation
    auto obb1 = torch::tensor({{10.0f, 10.0f, 50.0f, 10.0f, 0.0f}});  // Wide
    auto obb2 = torch::tensor({{12.0f, 12.0f, 50.0f, 10.0f, 0.3f}});

    // Act
    auto probiou_result = probiou(obb1, obb2);

    // Assert - Should compute valid IoU
    EXPECT_GE(probiou_result.item<float>(), 0.0f);
    EXPECT_LE(probiou_result.item<float>(), 1.0f + tolerance);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

TEST_F(ProbIoUTest, EdgeCase_ZeroAreaBoxes_HandledGracefully) {
    // Arrange - Box with zero area
    auto obb1 = torch::tensor({{10.0f, 10.0f, 0.0f, 0.0f, 0.0f}});
    auto obb2 = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f, 0.0f}});

    // Act
    auto probiou_result = probiou(obb1, obb2);

    // Assert - Should be close to 0
    EXPECT_NEAR(probiou_result.item<float>(), 0.0f, tolerance);
}

TEST_F(ProbIoUTest, EdgeCase_ExtremeAspectRatio_HandledCorrectly) {
    // Arrange - Very extreme aspect ratio
    auto obb1 = torch::tensor({{10.0f, 10.0f, 100.0f, 1.0f, 0.0f}});
    auto obb2 = torch::tensor({{10.0f, 10.0f, 1.0f, 100.0f, 0.0f}});

    // Act
    auto probiou_result = probiou(obb1, obb2);

    // Assert - Should compute valid value
    EXPECT_GE(probiou_result.item<float>(), 0.0f);
    EXPECT_LE(probiou_result.item<float>(), 1.0f + tolerance);
}

TEST_F(ProbIoUTest, EdgeCase_VerySmallBoxes_ComputeCorrectly) {
    // Arrange - Very small boxes
    auto obb1 = torch::tensor({{0.0f, 0.0f, 0.1f, 0.1f, 0.0f}});
    auto obb2 = torch::tensor({{0.0f, 0.0f, 0.1f, 0.1f, 0.0f}});

    // Act
    auto probiou_result = probiou(obb1, obb2);

    // Assert - Should be close to 1.0
    EXPECT_NEAR(probiou_result.item<float>(), 1.0f, tolerance);
}

// ============================================================================
// Batch Processing Tests
// ============================================================================

TEST_F(ProbIoUTest, Batch_MultipleRotatedBoxes_ReturnsCorrectMatrix) {
    // Arrange - Multiple rotated boxes
    auto obbs1 = torch::tensor({
        {10.0f, 10.0f, 20.0f, 20.0f, 0.0f},
        {30.0f, 30.0f, 20.0f, 20.0f, 0.5f},
        {50.0f, 50.0f, 20.0f, 20.0f, 1.0f}
    });
    auto obbs2 = torch::tensor({
        {10.0f, 10.0f, 20.0f, 20.0f, 0.0f},
        {50.0f, 50.0f, 20.0f, 20.0f, 1.0f}
    });

    // Act
    auto probiou_result = probiou(obbs1, obbs2);

    // Assert - Shape [3, 2]
    EXPECT_EQ(probiou_result.size(0), 3);
    EXPECT_EQ(probiou_result.size(1), 2);

    // Check diagonal-like elements (perfect match)
    EXPECT_NEAR(probiou_result[0][0].item<float>(), 1.0f, tolerance);
    EXPECT_NEAR(probiou_result[2][1].item<float>(), 1.0f, tolerance);

    // Check non-overlapping elements
    EXPECT_NEAR(probiou_result[0][1].item<float>(), 0.0f, tolerance);
}

TEST_F(ProbIoUTest, Batch_LargeBatch_ProcessesEfficiently) {
    // Arrange - Large batch of rotated boxes
    auto obbs1 = torch::rand({500, 5});
    obbs1.slice(1, 0, 2) = obbs1.slice(1, 0, 2) * 100.0f;  // Centers
    obbs1.slice(1, 2, 4) = torch::abs(obbs1.slice(1, 2, 4)) * 20.0f + 1.0f;  // Sizes
    obbs1.slice(1, 4, 5) = obbs1.slice(1, 4, 5) * 2.0f * M_PI;  // Angles

    auto obbs2 = torch::rand({100, 5});
    obbs2.slice(1, 0, 2) = obbs2.slice(1, 0, 2) * 100.0f;
    obbs2.slice(1, 2, 4) = torch::abs(obbs2.slice(1, 2, 4)) * 20.0f + 1.0f;
    obbs2.slice(1, 4, 5) = obbs2.slice(1, 4, 5) * 2.0f * M_PI;

    // Act
    auto probiou_result = probiou(obbs1, obbs2);

    // Assert - Shape [500, 100]
    EXPECT_EQ(probiou_result.size(0), 500);
    EXPECT_EQ(probiou_result.size(1), 100);

    // All Probiou values should be in [0, 1]
    auto probiou_min = probiou_result.min().item<float>();
    auto probiou_max = probiou_result.max().item<float>();
    EXPECT_GE(probiou_min, 0.0f);
    EXPECT_LE(probiou_max, 1.0f + tolerance);
}
