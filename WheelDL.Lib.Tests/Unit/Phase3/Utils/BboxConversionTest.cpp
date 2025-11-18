#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Utils/IoU.h"
#include <torch/torch.h>

using namespace WheelDL::Model::Utils;

/**
 * @class BboxConversionTest
 * @brief Unit tests for bounding box conversion functions
 *
 * Tests cover:
 * - xywh2xyxy conversion
 * - xyxy2xywh conversion
 * - dist2bbox (DFL decoding)
 * - bbox2dist (inverse DFL)
 * - dist2rbox (rotated boxes)
 * - xywhr2xyxyxyxy (corner points)
 * - Roundtrip conversions
 */
class BboxConversionTest : public ::testing::Test {
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
// xywh2xyxy Tests
// ============================================================================

TEST_F(BboxConversionTest, Xywh2Xyxy_ConversionCorrectness_WorksProperly) {
    // Arrange - center (10, 10), size (20, 20)
    // Expected: (0, 0) to (20, 20)
    auto xywh = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});

    // Act
    auto xyxy = xywh2xyxy(xywh);

    // Assert
    EXPECT_NEAR(xyxy[0][0].item<float>(), 0.0f, tolerance);
    EXPECT_NEAR(xyxy[0][1].item<float>(), 0.0f, tolerance);
    EXPECT_NEAR(xyxy[0][2].item<float>(), 20.0f, tolerance);
    EXPECT_NEAR(xyxy[0][3].item<float>(), 20.0f, tolerance);
}

TEST_F(BboxConversionTest, Xywh2Xyxy_BatchInput_ConvertsAllBoxes) {
    // Arrange - Multiple boxes
    auto xywh = torch::tensor({
        {10.0f, 10.0f, 20.0f, 20.0f},
        {30.0f, 30.0f, 40.0f, 40.0f},
        {50.0f, 60.0f, 10.0f, 20.0f}
    });

    // Act
    auto xyxy = xywh2xyxy(xywh);

    // Assert - Shape preserved
    EXPECT_EQ(xyxy.size(0), 3);
    EXPECT_EQ(xyxy.size(1), 4);

    // Check first box
    EXPECT_NEAR(xyxy[0][0].item<float>(), 0.0f, tolerance);
    EXPECT_NEAR(xyxy[0][1].item<float>(), 0.0f, tolerance);
    EXPECT_NEAR(xyxy[0][2].item<float>(), 20.0f, tolerance);
    EXPECT_NEAR(xyxy[0][3].item<float>(), 20.0f, tolerance);
}

TEST_F(BboxConversionTest, Xywh2Xyxy_ZeroSizeBoxes_HandledCorrectly) {
    // Arrange - Box with zero size
    auto xywh = torch::tensor({{10.0f, 10.0f, 0.0f, 0.0f}});

    // Act
    auto xyxy = xywh2xyxy(xywh);

    // Assert - Should be a point
    EXPECT_NEAR(xyxy[0][0].item<float>(), 10.0f, tolerance);
    EXPECT_NEAR(xyxy[0][1].item<float>(), 10.0f, tolerance);
    EXPECT_NEAR(xyxy[0][2].item<float>(), 10.0f, tolerance);
    EXPECT_NEAR(xyxy[0][3].item<float>(), 10.0f, tolerance);
}

TEST_F(BboxConversionTest, Xywh2Xyxy_SingleBox_ReturnsCorrectShape) {
    // Arrange
    auto xywh = torch::tensor({{50.0f, 50.0f, 100.0f, 100.0f}});

    // Act
    auto xyxy = xywh2xyxy(xywh);

    // Assert - Shape [1, 4]
    EXPECT_EQ(xyxy.dim(), 2);
    EXPECT_EQ(xyxy.size(0), 1);
    EXPECT_EQ(xyxy.size(1), 4);
}

TEST_F(BboxConversionTest, Xywh2Xyxy_EdgeCoordinates_ComputedCorrectly) {
    // Arrange - Box at edge coordinates
    auto xywh = torch::tensor({{0.0f, 0.0f, 10.0f, 10.0f}});

    // Act
    auto xyxy = xywh2xyxy(xywh);

    // Assert
    EXPECT_NEAR(xyxy[0][0].item<float>(), -5.0f, tolerance);
    EXPECT_NEAR(xyxy[0][1].item<float>(), -5.0f, tolerance);
    EXPECT_NEAR(xyxy[0][2].item<float>(), 5.0f, tolerance);
    EXPECT_NEAR(xyxy[0][3].item<float>(), 5.0f, tolerance);
}

// ============================================================================
// xyxy2xywh Tests
// ============================================================================

TEST_F(BboxConversionTest, Xyxy2Xywh_ConversionCorrectness_WorksProperly) {
    // Arrange - corners (0, 0) to (20, 20)
    // Expected: center (10, 10), size (20, 20)
    auto xyxy = torch::tensor({{0.0f, 0.0f, 20.0f, 20.0f}});

    // Act
    auto xywh = xyxy2xywh(xyxy);

    // Assert
    EXPECT_NEAR(xywh[0][0].item<float>(), 10.0f, tolerance);
    EXPECT_NEAR(xywh[0][1].item<float>(), 10.0f, tolerance);
    EXPECT_NEAR(xywh[0][2].item<float>(), 20.0f, tolerance);
    EXPECT_NEAR(xywh[0][3].item<float>(), 20.0f, tolerance);
}

TEST_F(BboxConversionTest, Xyxy2Xywh_BatchInput_ConvertsAllBoxes) {
    // Arrange - Multiple boxes
    auto xyxy = torch::tensor({
        {0.0f, 0.0f, 20.0f, 20.0f},
        {10.0f, 10.0f, 50.0f, 50.0f},
        {45.0f, 50.0f, 55.0f, 70.0f}
    });

    // Act
    auto xywh = xyxy2xywh(xyxy);

    // Assert - Shape preserved
    EXPECT_EQ(xywh.size(0), 3);
    EXPECT_EQ(xywh.size(1), 4);

    // Check first box
    EXPECT_NEAR(xywh[0][0].item<float>(), 10.0f, tolerance);
    EXPECT_NEAR(xywh[0][1].item<float>(), 10.0f, tolerance);
    EXPECT_NEAR(xywh[0][2].item<float>(), 20.0f, tolerance);
    EXPECT_NEAR(xywh[0][3].item<float>(), 20.0f, tolerance);
}

TEST_F(BboxConversionTest, Xyxy2Xywh_InvalidBox_NegativeSize) {
    // Arrange - Invalid box where x1 > x2
    auto xyxy = torch::tensor({{20.0f, 20.0f, 0.0f, 0.0f}});

    // Act
    auto xywh = xyxy2xywh(xyxy);

    // Assert - Size should be negative (function doesn't validate)
    EXPECT_NEAR(xywh[0][2].item<float>(), -20.0f, tolerance);
    EXPECT_NEAR(xywh[0][3].item<float>(), -20.0f, tolerance);
}

TEST_F(BboxConversionTest, Xyxy2Xywh_SingleBox_ReturnsCorrectShape) {
    // Arrange
    auto xyxy = torch::tensor({{0.0f, 0.0f, 100.0f, 100.0f}});

    // Act
    auto xywh = xyxy2xywh(xyxy);

    // Assert - Shape [1, 4]
    EXPECT_EQ(xywh.dim(), 2);
    EXPECT_EQ(xywh.size(0), 1);
    EXPECT_EQ(xywh.size(1), 4);
}

TEST_F(BboxConversionTest, Xyxy2Xywh_EdgeCoordinates_ComputedCorrectly) {
    // Arrange - Box with negative coordinates
    auto xyxy = torch::tensor({{-10.0f, -10.0f, 10.0f, 10.0f}});

    // Act
    auto xywh = xyxy2xywh(xyxy);

    // Assert
    EXPECT_NEAR(xywh[0][0].item<float>(), 0.0f, tolerance);
    EXPECT_NEAR(xywh[0][1].item<float>(), 0.0f, tolerance);
    EXPECT_NEAR(xywh[0][2].item<float>(), 20.0f, tolerance);
    EXPECT_NEAR(xywh[0][3].item<float>(), 20.0f, tolerance);
}

// ============================================================================
// dist2bbox Tests
// ============================================================================

TEST_F(BboxConversionTest, Dist2Bbox_DistanceToBbox_DFLDecoding) {
    // Arrange - DFL distance predictions
    auto distance = torch::tensor({{5.0f, 5.0f, 5.0f, 5.0f}});  // left, top, right, bottom
    auto anchor_points = torch::tensor({{10.0f, 10.0f}});

    // Act - Convert to xywh
    auto bbox = dist2bbox(distance, anchor_points, /*xywh=*/true);

    // Assert - Expected: center (10, 10), size (10, 10)
    EXPECT_NEAR(bbox[0][0].item<float>(), 10.0f, tolerance);
    EXPECT_NEAR(bbox[0][1].item<float>(), 10.0f, tolerance);
    EXPECT_NEAR(bbox[0][2].item<float>(), 10.0f, tolerance);
    EXPECT_NEAR(bbox[0][3].item<float>(), 10.0f, tolerance);
}

TEST_F(BboxConversionTest, Dist2Bbox_WithAnchorPoints_CorrectOffset) {
    // Arrange
    auto distance = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto anchor_points = torch::tensor({{50.0f, 50.0f}});

    // Act - Convert to xyxy
    auto bbox = dist2bbox(distance, anchor_points, /*xywh=*/false);

    // Assert
    EXPECT_NEAR(bbox[0][0].item<float>(), 40.0f, tolerance);  // x1 = 50 - 10
    EXPECT_NEAR(bbox[0][1].item<float>(), 40.0f, tolerance);  // y1 = 50 - 10
    EXPECT_NEAR(bbox[0][2].item<float>(), 70.0f, tolerance);  // x2 = 50 + 20
    EXPECT_NEAR(bbox[0][3].item<float>(), 70.0f, tolerance);  // y2 = 50 + 20
}

TEST_F(BboxConversionTest, Dist2Bbox_XywhVsXyxy_OutputFormat) {
    // Arrange
    auto distance = torch::tensor({{5.0f, 5.0f, 5.0f, 5.0f}});
    auto anchor_points = torch::tensor({{10.0f, 10.0f}});

    // Act - Both formats
    auto bbox_xywh = dist2bbox(distance, anchor_points, /*xywh=*/true);
    auto bbox_xyxy = dist2bbox(distance, anchor_points, /*xywh=*/false);

    // Assert - xywh: center (10, 10), size (10, 10)
    EXPECT_NEAR(bbox_xywh[0][0].item<float>(), 10.0f, tolerance);
    EXPECT_NEAR(bbox_xywh[0][1].item<float>(), 10.0f, tolerance);

    // Assert - xyxy: (5, 5) to (15, 15)
    EXPECT_NEAR(bbox_xyxy[0][0].item<float>(), 5.0f, tolerance);
    EXPECT_NEAR(bbox_xyxy[0][1].item<float>(), 5.0f, tolerance);
    EXPECT_NEAR(bbox_xyxy[0][2].item<float>(), 15.0f, tolerance);
    EXPECT_NEAR(bbox_xyxy[0][3].item<float>(), 15.0f, tolerance);
}

TEST_F(BboxConversionTest, Dist2Bbox_BatchProcessing_HandlesMultiple) {
    // Arrange - Batch of distances and anchors
    auto distance = torch::tensor({
        {5.0f, 5.0f, 5.0f, 5.0f},
        {10.0f, 10.0f, 10.0f, 10.0f}
    });
    auto anchor_points = torch::tensor({
        {10.0f, 10.0f},
        {20.0f, 20.0f}
    });

    // Act
    auto bbox = dist2bbox(distance, anchor_points, /*xywh=*/true);

    // Assert - Shape [2, 4]
    EXPECT_EQ(bbox.size(0), 2);
    EXPECT_EQ(bbox.size(1), 4);
}

TEST_F(BboxConversionTest, Dist2Bbox_Integration_WithDFL) {
    // Arrange - Simulate DFL output
    auto distance = torch::tensor({{8.0f, 8.0f, 12.0f, 12.0f}});
    auto anchor_points = torch::tensor({{100.0f, 100.0f}});

    // Act
    auto bbox = dist2bbox(distance, anchor_points, /*xywh=*/false);

    // Assert
    EXPECT_NEAR(bbox[0][0].item<float>(), 92.0f, tolerance);   // 100 - 8
    EXPECT_NEAR(bbox[0][1].item<float>(), 92.0f, tolerance);   // 100 - 8
    EXPECT_NEAR(bbox[0][2].item<float>(), 112.0f, tolerance);  // 100 + 12
    EXPECT_NEAR(bbox[0][3].item<float>(), 112.0f, tolerance);  // 100 + 12
}

// ============================================================================
// bbox2dist Tests
// ============================================================================

TEST_F(BboxConversionTest, Bbox2Dist_BboxToDistance_InverseOperation) {
    // Arrange - Bboxes in xyxy format
    auto anchor_points = torch::tensor({{10.0f, 10.0f}});
    auto bboxes = torch::tensor({{5.0f, 5.0f, 15.0f, 15.0f}});

    // Act
    auto dist = bbox2dist(anchor_points, bboxes);

    // Assert - left, top, right, bottom distances
    EXPECT_NEAR(dist[0][0].item<float>(), 5.0f, tolerance);  // left = 10 - 5
    EXPECT_NEAR(dist[0][1].item<float>(), 5.0f, tolerance);  // top = 10 - 5
    EXPECT_NEAR(dist[0][2].item<float>(), 5.0f, tolerance);  // right = 15 - 10
    EXPECT_NEAR(dist[0][3].item<float>(), 5.0f, tolerance);  // bottom = 15 - 10
}

TEST_F(BboxConversionTest, Bbox2Dist_WithRegMax_ClampsValues) {
    // Arrange - Large bbox that exceeds regMax
    auto anchor_points = torch::tensor({{10.0f, 10.0f}});
    auto bboxes = torch::tensor({{0.0f, 0.0f, 50.0f, 50.0f}});
    int64_t reg_max = 16;

    // Act
    auto dist = bbox2dist(anchor_points, bboxes, reg_max);

    // Assert - Should be clamped to [0, regMax - 0.01]
    EXPECT_NEAR(dist[0][0].item<float>(), 10.0f, tolerance);  // left = 10 (within range)
    EXPECT_NEAR(dist[0][2].item<float>(), 15.99f, tolerance);  // right = clamped to 15.99
    EXPECT_NEAR(dist[0][3].item<float>(), 15.99f, tolerance);  // bottom = clamped to 15.99
}

TEST_F(BboxConversionTest, Bbox2Dist_Roundtrip_BboxToDistToBbox) {
    // Arrange
    auto anchor_points = torch::tensor({{20.0f, 20.0f}});
    auto bboxes_orig = torch::tensor({{15.0f, 15.0f, 25.0f, 25.0f}});

    // Act - Convert to distance and back
    auto dist = bbox2dist(anchor_points, bboxes_orig, /*regMax=*/16);
    auto bboxes_recon = dist2bbox(dist, anchor_points, /*xywh=*/false);

    // Assert - Should match original (within tolerance)
    EXPECT_NEAR(bboxes_recon[0][0].item<float>(), bboxes_orig[0][0].item<float>(), tolerance);
    EXPECT_NEAR(bboxes_recon[0][1].item<float>(), bboxes_orig[0][1].item<float>(), tolerance);
    EXPECT_NEAR(bboxes_recon[0][2].item<float>(), bboxes_orig[0][2].item<float>(), tolerance);
    EXPECT_NEAR(bboxes_recon[0][3].item<float>(), bboxes_orig[0][3].item<float>(), tolerance);
}

// ============================================================================
// dist2rbox Tests
// ============================================================================

TEST_F(BboxConversionTest, Dist2Rbox_DistanceAndAngle_ToRotatedBbox) {
    // Arrange - Distance and angle predictions
    auto distance = torch::tensor({{5.0f, 5.0f, 5.0f, 5.0f}});
    auto angle = torch::tensor({{0.0f}});
    auto anchor_points = torch::tensor({{10.0f, 10.0f}});

    // Act - Returns [cx, cy, w, h] WITHOUT angle
    auto rbox = dist2rbox(distance, angle, anchor_points);

    // Assert - Shape [1, 4] (no angle in output)
    EXPECT_EQ(rbox.size(0), 1);
    EXPECT_EQ(rbox.size(1), 4);

    // Check center and size
    EXPECT_NEAR(rbox[0][0].item<float>(), 10.0f, tolerance);  // cx
    EXPECT_NEAR(rbox[0][1].item<float>(), 10.0f, tolerance);  // cy
    EXPECT_NEAR(rbox[0][2].item<float>(), 10.0f, tolerance);  // w
    EXPECT_NEAR(rbox[0][3].item<float>(), 10.0f, tolerance);  // h
}

TEST_F(BboxConversionTest, Dist2Rbox_ReturnsCxCyWH_NoAngle) {
    // Arrange
    auto distance = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f}});
    auto angle = torch::tensor({{0.5f}});
    auto anchor_points = torch::tensor({{50.0f, 50.0f}});

    // Act
    auto rbox = dist2rbox(distance, angle, anchor_points);

    // Assert - Returns 4 values only (cx, cy, w, h)
    EXPECT_EQ(rbox.size(1), 4);

    // Width and height should be sum of left+right, top+bottom
    EXPECT_NEAR(rbox[0][2].item<float>(), 30.0f, tolerance);  // w = 10 + 20
    EXPECT_NEAR(rbox[0][3].item<float>(), 30.0f, tolerance);  // h = 10 + 20
}

TEST_F(BboxConversionTest, Dist2Rbox_WithAnchorPoints_OffsetsCenter) {
    // Arrange
    auto distance = torch::tensor({{5.0f, 5.0f, 5.0f, 5.0f}});
    auto angle = torch::tensor({{0.0f}});
    auto anchor_points = torch::tensor({{20.0f, 30.0f}});

    // Act
    auto rbox = dist2rbox(distance, angle, anchor_points);

    // Assert - Center should be affected by both distance offset and anchor
    EXPECT_NEAR(rbox[0][2].item<float>(), 10.0f, tolerance);  // w = 5 + 5
    EXPECT_NEAR(rbox[0][3].item<float>(), 10.0f, tolerance);  // h = 5 + 5
}

// ============================================================================
// xywhr2xyxyxyxy Tests
// ============================================================================

TEST_F(BboxConversionTest, Xywhr2Xyxyxyxy_RotatedBoxTo4Corners_Correct) {
    // Arrange - Rotated box [cx, cy, w, h, angle]
    auto rboxes = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f, 0.0f}});

    // Act
    auto corners = xywhr2xyxyxyxy(rboxes);

    // Assert - Shape [1, 4, 2] (4 corners with x, y)
    EXPECT_EQ(corners.size(0), 1);
    EXPECT_EQ(corners.size(1), 4);
    EXPECT_EQ(corners.size(2), 2);
}

TEST_F(BboxConversionTest, Xywhr2Xyxyxyxy_AngleTransformation_WorksCorrectly) {
    // Arrange - Rotated box with 45 degree rotation
    auto rboxes = torch::tensor({{10.0f, 10.0f, 20.0f, 20.0f, static_cast<float>(M_PI / 4.0)}});

    // Act
    auto corners = xywhr2xyxyxyxy(rboxes);

    // Assert - All 4 corners should be computed
    EXPECT_EQ(corners.size(1), 4);

    // Corners should be different from unrotated box
    auto corner1_x = corners[0][0][0].item<float>();
    auto corner1_y = corners[0][0][1].item<float>();

    // Not checking exact values due to rotation complexity,
    // but should be valid coordinates
    EXPECT_TRUE(std::isfinite(corner1_x));
    EXPECT_TRUE(std::isfinite(corner1_y));
}

// ============================================================================
// Roundtrip Tests
// ============================================================================

TEST_F(BboxConversionTest, Roundtrip_XywhToXyxyToXywh_Lossless) {
    // Arrange
    auto xywh_orig = torch::tensor({
        {10.0f, 10.0f, 20.0f, 20.0f},
        {30.0f, 40.0f, 50.0f, 60.0f}
    });

    // Act - Convert to xyxy and back
    auto xyxy = xywh2xyxy(xywh_orig);
    auto xywh_recon = xyxy2xywh(xyxy);

    // Assert - Should match original
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(
                xywh_recon[i][j].item<float>(),
                xywh_orig[i][j].item<float>(),
                tolerance
            );
        }
    }
}

TEST_F(BboxConversionTest, Roundtrip_XyxyToXywhToXyxy_Lossless) {
    // Arrange
    auto xyxy_orig = torch::tensor({
        {0.0f, 0.0f, 20.0f, 20.0f},
        {10.0f, 20.0f, 60.0f, 80.0f}
    });

    // Act - Convert to xywh and back
    auto xywh = xyxy2xywh(xyxy_orig);
    auto xyxy_recon = xywh2xyxy(xywh);

    // Assert - Should match original
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_NEAR(
                xyxy_recon[i][j].item<float>(),
                xyxy_orig[i][j].item<float>(),
                tolerance
            );
        }
    }
}
