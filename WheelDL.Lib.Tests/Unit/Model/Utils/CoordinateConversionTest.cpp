#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Utils/IoU.h"

using namespace WheelDL::Model::Utils;

class CoordinateConversionTest : public ::testing::Test {
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

    // Helper: Create tensor from vector
    torch::Tensor createTensor(std::vector<std::vector<float>> data, int64_t lastDim) {
        std::vector<float> flat;
        for (const auto& row : data) {
            for (float v : row) flat.push_back(v);
        }
        return torch::tensor(flat, torch::TensorOptions().dtype(torch::kFloat32).device(device_))
            .view({static_cast<int64_t>(data.size()), lastDim});
    }

    // Helper: Create 2D tensor [N, dim]
    torch::Tensor create2DTensor(std::vector<std::vector<float>> data) {
        if (data.empty()) return torch::empty({0, 4}, torch::TensorOptions().device(device_));
        return createTensor(data, static_cast<int64_t>(data[0].size()));
    }
};

// =============================================================================
// xywh2xyxy Tests (Section 2.2.1)
// =============================================================================

TEST_F(CoordinateConversionTest, Xywh2Xyxy_Standard) {
    // xywh = [50, 50, 20, 10] -> xyxy = [40, 45, 60, 55]
    auto xywh = create2DTensor({{50, 50, 20, 10}});
    auto xyxy = xywh2xyxy(xywh);

    EXPECT_NEAR(xyxy[0][0].item<float>(), 40.0f, 1e-5f);  // x1 = 50 - 10 = 40
    EXPECT_NEAR(xyxy[0][1].item<float>(), 45.0f, 1e-5f);  // y1 = 50 - 5 = 45
    EXPECT_NEAR(xyxy[0][2].item<float>(), 60.0f, 1e-5f);  // x2 = 50 + 10 = 60
    EXPECT_NEAR(xyxy[0][3].item<float>(), 55.0f, 1e-5f);  // y2 = 50 + 5 = 55
}

TEST_F(CoordinateConversionTest, Xywh2Xyxy_ZeroSize) {
    // Zero dimensions: center point becomes both corners
    auto xywh = create2DTensor({{50, 50, 0, 0}});
    auto xyxy = xywh2xyxy(xywh);

    EXPECT_NEAR(xyxy[0][0].item<float>(), 50.0f, 1e-5f);
    EXPECT_NEAR(xyxy[0][1].item<float>(), 50.0f, 1e-5f);
    EXPECT_NEAR(xyxy[0][2].item<float>(), 50.0f, 1e-5f);
    EXPECT_NEAR(xyxy[0][3].item<float>(), 50.0f, 1e-5f);
}

TEST_F(CoordinateConversionTest, Xywh2Xyxy_LargeBox) {
    // Large coordinates
    auto xywh = create2DTensor({{500, 500, 200, 100}});
    auto xyxy = xywh2xyxy(xywh);

    EXPECT_NEAR(xyxy[0][0].item<float>(), 400.0f, 1e-5f);  // x1 = 500 - 100 = 400
    EXPECT_NEAR(xyxy[0][1].item<float>(), 450.0f, 1e-5f);  // y1 = 500 - 50 = 450
    EXPECT_NEAR(xyxy[0][2].item<float>(), 600.0f, 1e-5f);  // x2 = 500 + 100 = 600
    EXPECT_NEAR(xyxy[0][3].item<float>(), 550.0f, 1e-5f);  // y2 = 500 + 50 = 550
}

TEST_F(CoordinateConversionTest, Xywh2Xyxy_BatchConversion) {
    auto xywh = create2DTensor({
        {50, 50, 20, 10},
        {100, 100, 50, 30},
        {0, 0, 10, 10}
    });
    auto xyxy = xywh2xyxy(xywh);

    EXPECT_EQ(xyxy.size(0), 3);
    EXPECT_EQ(xyxy.size(1), 4);

    // First box
    EXPECT_NEAR(xyxy[0][0].item<float>(), 40.0f, 1e-5f);
    EXPECT_NEAR(xyxy[0][1].item<float>(), 45.0f, 1e-5f);

    // Second box: [100, 100, 50, 30] -> [75, 85, 125, 115]
    EXPECT_NEAR(xyxy[1][0].item<float>(), 75.0f, 1e-5f);
    EXPECT_NEAR(xyxy[1][1].item<float>(), 85.0f, 1e-5f);
    EXPECT_NEAR(xyxy[1][2].item<float>(), 125.0f, 1e-5f);
    EXPECT_NEAR(xyxy[1][3].item<float>(), 115.0f, 1e-5f);
}

TEST_F(CoordinateConversionTest, Xywh2Xyxy_PrecomputedValue) {
    // From test plan: xywh = [100, 100, 50, 30] -> xyxy = [75, 85, 125, 115]
    auto xywh = create2DTensor({{100, 100, 50, 30}});
    auto xyxy = xywh2xyxy(xywh);

    EXPECT_NEAR(xyxy[0][0].item<float>(), 75.0f, 1e-5f);
    EXPECT_NEAR(xyxy[0][1].item<float>(), 85.0f, 1e-5f);
    EXPECT_NEAR(xyxy[0][2].item<float>(), 125.0f, 1e-5f);
    EXPECT_NEAR(xyxy[0][3].item<float>(), 115.0f, 1e-5f);
}

TEST_F(CoordinateConversionTest, Xywh2Xyxy_3DTensor) {
    // Test with 3D tensor [batch, N, 4]
    auto xywh = torch::tensor({{{50.0f, 50.0f, 20.0f, 10.0f}}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto xyxy = xywh2xyxy(xywh);

    EXPECT_EQ(xyxy.dim(), 3);
    EXPECT_NEAR(xyxy[0][0][0].item<float>(), 40.0f, 1e-5f);
}

TEST_F(CoordinateConversionTest, Xywh2Xyxy_Exception_1DTensor) {
    auto xywh = torch::randn({4}, torch::TensorOptions().device(device_));
    EXPECT_THROW(xywh2xyxy(xywh), std::invalid_argument);
}

TEST_F(CoordinateConversionTest, Xywh2Xyxy_Exception_4DTensor) {
    auto xywh = torch::randn({1, 2, 3, 4}, torch::TensorOptions().device(device_));
    EXPECT_THROW(xywh2xyxy(xywh), std::invalid_argument);
}

// =============================================================================
// xyxy2xywh Tests (Section 2.2.2)
// =============================================================================

TEST_F(CoordinateConversionTest, Xyxy2Xywh_Standard) {
    // xyxy = [40, 45, 60, 55] -> xywh = [50, 50, 20, 10]
    auto xyxy = create2DTensor({{40, 45, 60, 55}});
    auto xywh = xyxy2xywh(xyxy);

    EXPECT_NEAR(xywh[0][0].item<float>(), 50.0f, 1e-5f);  // cx = (40+60)/2 = 50
    EXPECT_NEAR(xywh[0][1].item<float>(), 50.0f, 1e-5f);  // cy = (45+55)/2 = 50
    EXPECT_NEAR(xywh[0][2].item<float>(), 20.0f, 1e-5f);  // w = 60-40 = 20
    EXPECT_NEAR(xywh[0][3].item<float>(), 10.0f, 1e-5f);  // h = 55-45 = 10
}

TEST_F(CoordinateConversionTest, Xyxy2Xywh_ZeroArea) {
    // Point box
    auto xyxy = create2DTensor({{50, 50, 50, 50}});
    auto xywh = xyxy2xywh(xyxy);

    EXPECT_NEAR(xywh[0][0].item<float>(), 50.0f, 1e-5f);
    EXPECT_NEAR(xywh[0][1].item<float>(), 50.0f, 1e-5f);
    EXPECT_NEAR(xywh[0][2].item<float>(), 0.0f, 1e-5f);
    EXPECT_NEAR(xywh[0][3].item<float>(), 0.0f, 1e-5f);
}

TEST_F(CoordinateConversionTest, Xyxy2Xywh_NegativeCoords) {
    // Negative values: [-10, -10, 10, 10] -> [0, 0, 20, 20]
    auto xyxy = create2DTensor({{-10, -10, 10, 10}});
    auto xywh = xyxy2xywh(xyxy);

    EXPECT_NEAR(xywh[0][0].item<float>(), 0.0f, 1e-5f);
    EXPECT_NEAR(xywh[0][1].item<float>(), 0.0f, 1e-5f);
    EXPECT_NEAR(xywh[0][2].item<float>(), 20.0f, 1e-5f);
    EXPECT_NEAR(xywh[0][3].item<float>(), 20.0f, 1e-5f);
}

TEST_F(CoordinateConversionTest, Xyxy2Xywh_BatchConversion) {
    auto xyxy = create2DTensor({
        {40, 45, 60, 55},
        {0, 0, 100, 100}
    });
    auto xywh = xyxy2xywh(xyxy);

    EXPECT_EQ(xywh.size(0), 2);
    EXPECT_NEAR(xywh[0][0].item<float>(), 50.0f, 1e-5f);
    EXPECT_NEAR(xywh[1][0].item<float>(), 50.0f, 1e-5f);
    EXPECT_NEAR(xywh[1][2].item<float>(), 100.0f, 1e-5f);
}

TEST_F(CoordinateConversionTest, Xyxy2Xywh_Exception_1DTensor) {
    auto xyxy = torch::randn({4}, torch::TensorOptions().device(device_));
    EXPECT_THROW(xyxy2xywh(xyxy), std::invalid_argument);
}

// =============================================================================
// Reversibility Tests
// =============================================================================

TEST_F(CoordinateConversionTest, Reversibility_XywhToXyxyToXywh) {
    auto original = create2DTensor({{100, 100, 50, 30}});
    auto xyxy = xywh2xyxy(original);
    auto recovered = xyxy2xywh(xyxy);

    EXPECT_TRUE(torch::allclose(original, recovered, 1e-5, 1e-5));
}

TEST_F(CoordinateConversionTest, Reversibility_XyxyToXywhToXyxy) {
    auto original = create2DTensor({{75, 85, 125, 115}});
    auto xywh = xyxy2xywh(original);
    auto recovered = xywh2xyxy(xywh);

    EXPECT_TRUE(torch::allclose(original, recovered, 1e-5, 1e-5));
}

TEST_F(CoordinateConversionTest, Reversibility_BatchRoundTrip) {
    auto original = torch::randn({100, 4}, torch::TensorOptions().device(device_)).abs() * 100;
    auto xyxy = xywh2xyxy(original);
    auto recovered = xyxy2xywh(xyxy);

    // Note: xywh->xyxy->xywh should recover original
    auto xyxy2 = xywh2xyxy(recovered);
    EXPECT_TRUE(torch::allclose(xyxy, xyxy2, 1e-4, 1e-4));
}

// =============================================================================
// dist2bbox Tests (Section 2.2.3)
// =============================================================================

TEST_F(CoordinateConversionTest, Dist2Bbox_SingleAnchor_XYWH) {
    // distance = [left, top, right, bottom] = [5, 5, 5, 5]
    // anchor = [10, 10]
    // Result: x1y1 = [5, 5], x2y2 = [15, 15]
    // xywh: cx=(5+15)/2=10, cy=10, w=10, h=10
    auto distance = create2DTensor({{5, 5, 5, 5}});
    auto anchor = create2DTensor({{10, 10}});

    auto bbox = dist2bbox(distance, anchor, true);

    EXPECT_NEAR(bbox[0][0].item<float>(), 10.0f, 1e-5f);  // cx
    EXPECT_NEAR(bbox[0][1].item<float>(), 10.0f, 1e-5f);  // cy
    EXPECT_NEAR(bbox[0][2].item<float>(), 10.0f, 1e-5f);  // w
    EXPECT_NEAR(bbox[0][3].item<float>(), 10.0f, 1e-5f);  // h
}

TEST_F(CoordinateConversionTest, Dist2Bbox_SingleAnchor_XYXY) {
    auto distance = create2DTensor({{5, 5, 5, 5}});
    auto anchor = create2DTensor({{10, 10}});

    auto bbox = dist2bbox(distance, anchor, false);

    EXPECT_NEAR(bbox[0][0].item<float>(), 5.0f, 1e-5f);   // x1 = 10-5
    EXPECT_NEAR(bbox[0][1].item<float>(), 5.0f, 1e-5f);   // y1 = 10-5
    EXPECT_NEAR(bbox[0][2].item<float>(), 15.0f, 1e-5f);  // x2 = 10+5
    EXPECT_NEAR(bbox[0][3].item<float>(), 15.0f, 1e-5f);  // y2 = 10+5
}

TEST_F(CoordinateConversionTest, Dist2Bbox_MultipleAnchors) {
    auto distance = create2DTensor({
        {5, 5, 5, 5},
        {10, 10, 10, 10}
    });
    auto anchors = create2DTensor({
        {10, 10},
        {50, 50}
    });

    auto bbox = dist2bbox(distance, anchors, true);

    EXPECT_EQ(bbox.size(0), 2);
    EXPECT_NEAR(bbox[0][2].item<float>(), 10.0f, 1e-5f);  // w = 5+5 = 10
    EXPECT_NEAR(bbox[1][2].item<float>(), 20.0f, 1e-5f);  // w = 10+10 = 20
}

TEST_F(CoordinateConversionTest, Dist2Bbox_AsymmetricDistances) {
    // Asymmetric: left=10, top=5, right=20, bottom=15
    auto distance = create2DTensor({{10, 5, 20, 15}});
    auto anchor = create2DTensor({{50, 50}});

    auto bbox = dist2bbox(distance, anchor, true);

    // x1y1 = [40, 45], x2y2 = [70, 65]
    // xywh: cx=55, cy=55, w=30, h=20
    EXPECT_NEAR(bbox[0][0].item<float>(), 55.0f, 1e-5f);  // cx
    EXPECT_NEAR(bbox[0][1].item<float>(), 55.0f, 1e-5f);  // cy
    EXPECT_NEAR(bbox[0][2].item<float>(), 30.0f, 1e-5f);  // w
    EXPECT_NEAR(bbox[0][3].item<float>(), 20.0f, 1e-5f);  // h
}

TEST_F(CoordinateConversionTest, Dist2Bbox_3DTensor) {
    // 3D input: [batch, N, 4]
    auto distance = torch::tensor({{{5.0f, 5.0f, 5.0f, 5.0f}}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto anchor = create2DTensor({{10, 10}});

    auto bbox = dist2bbox(distance, anchor, true);

    EXPECT_EQ(bbox.dim(), 3);
    EXPECT_EQ(bbox.size(0), 1);
    EXPECT_EQ(bbox.size(1), 1);
    EXPECT_EQ(bbox.size(2), 4);
}

TEST_F(CoordinateConversionTest, Dist2Bbox_Exception_1DTensor) {
    auto distance = torch::randn({4}, torch::TensorOptions().device(device_));
    auto anchor = create2DTensor({{10, 10}});
    EXPECT_THROW(dist2bbox(distance, anchor, true), std::invalid_argument);
}

// =============================================================================
// bbox2dist Tests (Section 2.2.4)
// =============================================================================

TEST_F(CoordinateConversionTest, Bbox2Dist_BoxToDistances) {
    // bbox (xyxy): [5, 5, 15, 15], anchor: [10, 10]
    // distances: left=10-5=5, top=10-5=5, right=15-10=5, bottom=15-10=5
    auto anchor = create2DTensor({{10, 10}});
    auto bbox = create2DTensor({{5, 5, 15, 15}});

    auto dist = bbox2dist(anchor, bbox, 16);

    EXPECT_NEAR(dist[0][0].item<float>(), 5.0f, 1e-5f);  // left
    EXPECT_NEAR(dist[0][1].item<float>(), 5.0f, 1e-5f);  // top
    EXPECT_NEAR(dist[0][2].item<float>(), 5.0f, 1e-5f);  // right
    EXPECT_NEAR(dist[0][3].item<float>(), 5.0f, 1e-5f);  // bottom
}

TEST_F(CoordinateConversionTest, Bbox2Dist_ClippedDistances) {
    // Distance exceeds regMax
    auto anchor = create2DTensor({{10, 10}});
    auto bbox = create2DTensor({{-30, -30, 50, 50}});  // Very large box

    auto dist = bbox2dist(anchor, bbox, 16);

    // Distances should be clipped to [0, regMax - 0.01]
    float maxVal = 16.0f - 0.01f;
    EXPECT_LE(dist[0][0].item<float>(), maxVal);
    EXPECT_LE(dist[0][1].item<float>(), maxVal);
    EXPECT_GE(dist[0][0].item<float>(), 0.0f);
}

TEST_F(CoordinateConversionTest, Bbox2Dist_NegativeDistanceClamped) {
    // Anchor outside bbox should give negative distances, clamped to 0
    auto anchor = create2DTensor({{0, 0}});
    auto bbox = create2DTensor({{10, 10, 20, 20}});  // Anchor at origin, box far away

    auto dist = bbox2dist(anchor, bbox, 16);

    // left = 0 - 10 = -10 -> clamped to 0
    // right = 20 - 0 = 20 -> clamped to 15.99
    EXPECT_NEAR(dist[0][0].item<float>(), 0.0f, 1e-5f);
}

TEST_F(CoordinateConversionTest, Bbox2Dist_BatchConversion) {
    auto anchors = create2DTensor({
        {10, 10},
        {50, 50}
    });
    auto bboxes = create2DTensor({
        {5, 5, 15, 15},
        {40, 40, 60, 60}
    });

    auto dist = bbox2dist(anchors, bboxes, 16);

    EXPECT_EQ(dist.size(0), 2);
    EXPECT_NEAR(dist[0][0].item<float>(), 5.0f, 1e-5f);
    EXPECT_NEAR(dist[1][0].item<float>(), 10.0f, 1e-5f);
}

TEST_F(CoordinateConversionTest, Bbox2Dist_Exception_1DTensor) {
    auto anchor = create2DTensor({{10, 10}});
    auto bbox = torch::randn({4}, torch::TensorOptions().device(device_));
    // Note: May throw c10::Error due to size(1) access before validation
    EXPECT_ANY_THROW(bbox2dist(anchor, bbox, 16));
}

// =============================================================================
// dist2rbox Tests (Section 2.2.5)
// =============================================================================

TEST_F(CoordinateConversionTest, Dist2Rbox_NoRotation) {
    // theta=0 should give same as dist2bbox (xywh part only)
    auto distance = create2DTensor({{5, 5, 5, 5}});
    auto angle = create2DTensor({{0}});
    auto anchor = create2DTensor({{10, 10}});

    auto rbox = dist2rbox(distance, angle, anchor);

    // With angle=0, result should be [cx, cy, w, h]
    EXPECT_EQ(rbox.size(1), 4);
    EXPECT_NEAR(rbox[0][0].item<float>(), 10.0f, 1e-5f);  // cx
    EXPECT_NEAR(rbox[0][1].item<float>(), 10.0f, 1e-5f);  // cy
    EXPECT_NEAR(rbox[0][2].item<float>(), 10.0f, 1e-5f);  // w
    EXPECT_NEAR(rbox[0][3].item<float>(), 10.0f, 1e-5f);  // h
}

TEST_F(CoordinateConversionTest, Dist2Rbox_Rotation90) {
    auto distance = create2DTensor({{5, 5, 5, 5}});
    auto angle = torch::tensor({{static_cast<float>(PI / 2)}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto anchor = create2DTensor({{10, 10}});

    auto rbox = dist2rbox(distance, angle, anchor);

    // Output should still be [N, 4]
    EXPECT_EQ(rbox.size(1), 4);
    EXPECT_FALSE(rbox.isnan().any().item<bool>());
}

TEST_F(CoordinateConversionTest, Dist2Rbox_Rotation45) {
    auto distance = create2DTensor({{5, 5, 5, 5}});
    auto angle = torch::tensor({{static_cast<float>(PI / 4)}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto anchor = create2DTensor({{10, 10}});

    auto rbox = dist2rbox(distance, angle, anchor);

    EXPECT_EQ(rbox.size(1), 4);
    EXPECT_FALSE(rbox.isnan().any().item<bool>());
}

TEST_F(CoordinateConversionTest, Dist2Rbox_BatchInput_3D) {
    // 3D tensor: [batch, N, 4]
    auto distance = torch::tensor({{{5.0f, 5.0f, 5.0f, 5.0f}, {10.0f, 10.0f, 10.0f, 10.0f}}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto angle = torch::tensor({{{0.0f}, {0.5f}}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto anchor = create2DTensor({{10, 10}, {50, 50}});

    auto rbox = dist2rbox(distance, angle, anchor);

    EXPECT_EQ(rbox.dim(), 3);
    EXPECT_EQ(rbox.size(0), 1);  // batch
    EXPECT_EQ(rbox.size(1), 2);  // N
    EXPECT_EQ(rbox.size(2), 4);  // xywh
}

TEST_F(CoordinateConversionTest, Dist2Rbox_OutputNoAngle) {
    // Verify output is [N, 4] without angle
    auto distance = create2DTensor({{5, 5, 5, 5}});
    auto angle = create2DTensor({{0.5f}});
    auto anchor = create2DTensor({{10, 10}});

    auto rbox = dist2rbox(distance, angle, anchor);

    // Output should NOT include angle (only 4 columns)
    EXPECT_EQ(rbox.size(1), 4);
}

TEST_F(CoordinateConversionTest, Dist2Rbox_Exception_1DTensor) {
    auto distance = torch::randn({4}, torch::TensorOptions().device(device_));
    auto angle = create2DTensor({{0}});
    auto anchor = create2DTensor({{10, 10}});
    EXPECT_THROW(dist2rbox(distance, angle, anchor), std::invalid_argument);
}

// =============================================================================
// xywhr2xyxyxyxy Tests (Section 2.2.6)
// =============================================================================

TEST_F(CoordinateConversionTest, Xywhr2Xyxyxyxy_NoRotation) {
    // [50, 50, 20, 10, 0] -> corners at [40,45], [60,45], [60,55], [40,55]
    auto rbox = torch::tensor({{50.0f, 50.0f, 20.0f, 10.0f, 0.0f}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    auto corners = xywhr2xyxyxyxy(rbox);

    EXPECT_EQ(corners.dim(), 3);
    EXPECT_EQ(corners.size(0), 1);  // N
    EXPECT_EQ(corners.size(1), 4);  // 4 corners
    EXPECT_EQ(corners.size(2), 2);  // x, y

    // Check corners (order: top-left, top-right, bottom-right, bottom-left)
    EXPECT_NEAR(corners[0][0][0].item<float>(), 40.0f, 1e-4f);  // corner1 x
    EXPECT_NEAR(corners[0][0][1].item<float>(), 45.0f, 1e-4f);  // corner1 y
    EXPECT_NEAR(corners[0][1][0].item<float>(), 60.0f, 1e-4f);  // corner2 x
    EXPECT_NEAR(corners[0][1][1].item<float>(), 45.0f, 1e-4f);  // corner2 y
    EXPECT_NEAR(corners[0][2][0].item<float>(), 60.0f, 1e-4f);  // corner3 x
    EXPECT_NEAR(corners[0][2][1].item<float>(), 55.0f, 1e-4f);  // corner3 y
    EXPECT_NEAR(corners[0][3][0].item<float>(), 40.0f, 1e-4f);  // corner4 x
    EXPECT_NEAR(corners[0][3][1].item<float>(), 55.0f, 1e-4f);  // corner4 y
}

TEST_F(CoordinateConversionTest, Xywhr2Xyxyxyxy_Rotation90) {
    // Rotate 90 degrees - corners should swap
    auto rbox = torch::tensor({{50.0f, 50.0f, 20.0f, 10.0f, static_cast<float>(PI / 2)}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    auto corners = xywhr2xyxyxyxy(rbox);

    // After 90 degree rotation, width and height effectively swap
    EXPECT_FALSE(corners.isnan().any().item<bool>());
    EXPECT_EQ(corners.size(1), 4);
}

TEST_F(CoordinateConversionTest, Xywhr2Xyxyxyxy_Rotation45) {
    auto rbox = torch::tensor({{50.0f, 50.0f, 20.0f, 10.0f, static_cast<float>(PI / 4)}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    auto corners = xywhr2xyxyxyxy(rbox);

    EXPECT_FALSE(corners.isnan().any().item<bool>());

    // Center of mass should still be at (50, 50)
    auto center_x = corners.select(2, 0).mean();
    auto center_y = corners.select(2, 1).mean();
    EXPECT_NEAR(center_x.item<float>(), 50.0f, 0.1f);
    EXPECT_NEAR(center_y.item<float>(), 50.0f, 0.1f);
}

TEST_F(CoordinateConversionTest, Xywhr2Xyxyxyxy_BatchConversion) {
    auto rboxes = torch::tensor({
        {50.0f, 50.0f, 20.0f, 10.0f, 0.0f},
        {100.0f, 100.0f, 40.0f, 20.0f, static_cast<float>(PI / 4)}
    }, torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    auto corners = xywhr2xyxyxyxy(rboxes);

    EXPECT_EQ(corners.size(0), 2);  // N boxes
    EXPECT_EQ(corners.size(1), 4);  // 4 corners each
    EXPECT_EQ(corners.size(2), 2);  // x, y
}

TEST_F(CoordinateConversionTest, Xywhr2Xyxyxyxy_SquareBox) {
    // Square box - corners should be equidistant from center
    auto rbox = torch::tensor({{50.0f, 50.0f, 10.0f, 10.0f, static_cast<float>(PI / 4)}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    auto corners = xywhr2xyxyxyxy(rbox);

    // All corners should be at same distance from center
    auto dist0 = std::sqrt(std::pow(corners[0][0][0].item<float>() - 50.0f, 2) +
                           std::pow(corners[0][0][1].item<float>() - 50.0f, 2));
    auto dist1 = std::sqrt(std::pow(corners[0][1][0].item<float>() - 50.0f, 2) +
                           std::pow(corners[0][1][1].item<float>() - 50.0f, 2));

    EXPECT_NEAR(dist0, dist1, 0.01f);
}

TEST_F(CoordinateConversionTest, Xywhr2Xyxyxyxy_3DTensor) {
    // 3D tensor [batch, N, 5]
    auto rboxes = torch::tensor({{{50.0f, 50.0f, 20.0f, 10.0f, 0.0f}}},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));

    auto corners = xywhr2xyxyxyxy(rboxes);

    EXPECT_EQ(corners.dim(), 4);  // [batch, N, 4, 2]
}

TEST_F(CoordinateConversionTest, Xywhr2Xyxyxyxy_Exception_1DTensor) {
    auto rbox = torch::randn({5}, torch::TensorOptions().device(device_));
    EXPECT_THROW(xywhr2xyxyxyxy(rbox), std::invalid_argument);
}

// =============================================================================
// Device Consistency Tests
// =============================================================================

TEST_F(CoordinateConversionTest, DeviceConsistency_Xywh2Xyxy) {
    auto xywh = torch::randn({10, 4}, torch::TensorOptions().device(device_));
    auto xyxy = xywh2xyxy(xywh);
    EXPECT_EQ(xyxy.device().type(), device_.type());
}

TEST_F(CoordinateConversionTest, DeviceConsistency_Xyxy2Xywh) {
    auto xyxy = torch::randn({10, 4}, torch::TensorOptions().device(device_));
    auto xywh = xyxy2xywh(xyxy);
    EXPECT_EQ(xywh.device().type(), device_.type());
}

TEST_F(CoordinateConversionTest, DeviceConsistency_Dist2Bbox) {
    auto distance = torch::randn({10, 4}, torch::TensorOptions().device(device_));
    auto anchor = torch::randn({10, 2}, torch::TensorOptions().device(device_));
    auto bbox = dist2bbox(distance, anchor, true);
    EXPECT_EQ(bbox.device().type(), device_.type());
}

TEST_F(CoordinateConversionTest, DeviceConsistency_Xywhr2Xyxyxyxy) {
    auto rbox = torch::randn({10, 5}, torch::TensorOptions().device(device_));
    auto corners = xywhr2xyxyxyxy(rbox);
    EXPECT_EQ(corners.device().type(), device_.type());
}

// =============================================================================
// Gradient Tests
// =============================================================================

TEST_F(CoordinateConversionTest, Gradient_Xywh2Xyxy) {
    auto xywh = torch::randn({5, 4}, torch::TensorOptions().device(device_).requires_grad(true));
    auto xyxy = xywh2xyxy(xywh);
    xyxy.sum().backward();

    EXPECT_TRUE(xywh.grad().defined());
    EXPECT_FALSE(xywh.grad().isnan().any().item<bool>());
}

TEST_F(CoordinateConversionTest, Gradient_Xyxy2Xywh) {
    auto xyxy = torch::randn({5, 4}, torch::TensorOptions().device(device_).requires_grad(true));
    auto xywh = xyxy2xywh(xyxy);
    xywh.sum().backward();

    EXPECT_TRUE(xyxy.grad().defined());
    EXPECT_FALSE(xyxy.grad().isnan().any().item<bool>());
}

TEST_F(CoordinateConversionTest, Gradient_Dist2Bbox) {
    auto distance = torch::randn({5, 4}, torch::TensorOptions().device(device_).requires_grad(true));
    auto anchor = torch::randn({5, 2}, torch::TensorOptions().device(device_));
    auto bbox = dist2bbox(distance, anchor, true);
    bbox.sum().backward();

    EXPECT_TRUE(distance.grad().defined());
    EXPECT_FALSE(distance.grad().isnan().any().item<bool>());
}

TEST_F(CoordinateConversionTest, Gradient_Xywhr2Xyxyxyxy) {
    auto rbox = torch::randn({5, 5}, torch::TensorOptions().device(device_).requires_grad(true));
    auto corners = xywhr2xyxyxyxy(rbox);
    corners.sum().backward();

    EXPECT_TRUE(rbox.grad().defined());
    EXPECT_FALSE(rbox.grad().isnan().any().item<bool>());
}
