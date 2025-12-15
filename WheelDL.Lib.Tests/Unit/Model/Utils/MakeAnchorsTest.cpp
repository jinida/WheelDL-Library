#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Utils/IoU.h"

using namespace WheelDL::Model::Utils;

class MakeAnchorsTest : public ::testing::Test {
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

    // Helper: Create feature tensor [batch, channels, height, width]
    torch::Tensor createFeatureTensor(int64_t batch, int64_t channels, int64_t height, int64_t width) {
        return torch::randn({batch, channels, height, width},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    }

    // Helper: Create stride tensor
    torch::Tensor createStrides(std::vector<float> values) {
        return torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    }
};

// =============================================================================
// makeAnchors Feature Tensor Overload Tests (Section 2.3.1)
// =============================================================================

TEST_F(MakeAnchorsTest, FeatureTensor_SingleFeatureMap) {
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 80, 80));

    auto strides = createStrides({8.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    // Should have 80*80 = 6400 anchor points
    EXPECT_EQ(anchors.size(0), 6400);
    EXPECT_EQ(anchors.size(1), 2);  // x, y coordinates
    EXPECT_EQ(strideTensor.size(0), 6400);
    EXPECT_EQ(strideTensor.size(1), 1);
}

TEST_F(MakeAnchorsTest, FeatureTensor_MultiScale) {
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 80, 80));  // P3
    feats.push_back(createFeatureTensor(1, 256, 40, 40));  // P4
    feats.push_back(createFeatureTensor(1, 256, 20, 20));  // P5

    auto strides = createStrides({8.0f, 16.0f, 32.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    // Total: 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400
    EXPECT_EQ(anchors.size(0), 8400);
    EXPECT_EQ(strideTensor.size(0), 8400);
}

TEST_F(MakeAnchorsTest, FeatureTensor_StrideValues) {
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 4, 4));

    auto strides = createStrides({8.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    // All stride values should be 8.0
    EXPECT_TRUE(torch::allclose(strideTensor, torch::full({16, 1}, 8.0f, strideTensor.options())));
}

TEST_F(MakeAnchorsTest, FeatureTensor_FeatureSizes) {
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 10, 10));
    feats.push_back(createFeatureTensor(1, 256, 5, 5));

    auto strides = createStrides({8.0f, 16.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    // Total: 10*10 + 5*5 = 100 + 25 = 125
    EXPECT_EQ(anchors.size(0), 125);

    // Check stride tensor has correct values
    auto strideData = strideTensor.cpu();
    // First 100 should be 8.0, next 25 should be 16.0
    EXPECT_NEAR(strideData[0][0].item<float>(), 8.0f, 1e-5f);
    EXPECT_NEAR(strideData[99][0].item<float>(), 8.0f, 1e-5f);
    EXPECT_NEAR(strideData[100][0].item<float>(), 16.0f, 1e-5f);
    EXPECT_NEAR(strideData[124][0].item<float>(), 16.0f, 1e-5f);
}

TEST_F(MakeAnchorsTest, FeatureTensor_CenterAlignment) {
    // With offset=0.5, anchors should be at cell centers
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 4, 4));

    auto strides = createStrides({8.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    // First anchor should be at (0.5, 0.5) in grid coords
    // In image coords: (0.5 * 8, 0.5 * 8) = (4, 4)
    // Note: makeAnchors returns grid coords, not image coords
    auto anchorsCpu = anchors.cpu();
    EXPECT_NEAR(anchorsCpu[0][0].item<float>(), 0.5f, 1e-5f);  // x
    EXPECT_NEAR(anchorsCpu[0][1].item<float>(), 0.5f, 1e-5f);  // y
}

TEST_F(MakeAnchorsTest, FeatureTensor_CustomOffset_Zero) {
    // With offset=0.0, anchors should be at cell corners
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 4, 4));

    auto strides = createStrides({8.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.0f);

    auto anchorsCpu = anchors.cpu();
    EXPECT_NEAR(anchorsCpu[0][0].item<float>(), 0.0f, 1e-5f);
    EXPECT_NEAR(anchorsCpu[0][1].item<float>(), 0.0f, 1e-5f);
}

TEST_F(MakeAnchorsTest, FeatureTensor_PrecomputedValues) {
    // From test plan: 4x4 feature map, stride=8
    // Anchor centers at (0.5+i)*stride for offset=0.5
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 4, 4));

    auto strides = createStrides({8.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    auto anchorsCpu = anchors.cpu();

    // First row: y=0.5, x=0.5,1.5,2.5,3.5
    EXPECT_NEAR(anchorsCpu[0][0].item<float>(), 0.5f, 1e-5f);  // (0,0) -> x=0.5
    EXPECT_NEAR(anchorsCpu[0][1].item<float>(), 0.5f, 1e-5f);  // (0,0) -> y=0.5
    EXPECT_NEAR(anchorsCpu[1][0].item<float>(), 1.5f, 1e-5f);  // (0,1) -> x=1.5
    EXPECT_NEAR(anchorsCpu[1][1].item<float>(), 0.5f, 1e-5f);  // (0,1) -> y=0.5

    // Second row starts at index 4
    EXPECT_NEAR(anchorsCpu[4][0].item<float>(), 0.5f, 1e-5f);  // (1,0) -> x=0.5
    EXPECT_NEAR(anchorsCpu[4][1].item<float>(), 1.5f, 1e-5f);  // (1,0) -> y=1.5
}

TEST_F(MakeAnchorsTest, FeatureTensor_Exception_EmptyFeats) {
    std::vector<torch::Tensor> feats;  // Empty
    auto strides = createStrides({8.0f});

    EXPECT_THROW(makeAnchors(feats, strides, 0.5f), std::invalid_argument);
}

// =============================================================================
// makeAnchors Shape-based Overload Tests (Section 2.3.2)
// =============================================================================

TEST_F(MakeAnchorsTest, ShapeBased_SingleScale) {
    std::vector<std::pair<int64_t, int64_t>> featShapes = {{80, 80}};
    auto strides = createStrides({8.0f});

    auto [anchors, strideTensor] = makeAnchors(featShapes, strides, torch::kFloat32, device_, 0.5f);

    EXPECT_EQ(anchors.size(0), 6400);  // 80*80
    EXPECT_EQ(anchors.size(1), 2);
}

TEST_F(MakeAnchorsTest, ShapeBased_MultiScale) {
    std::vector<std::pair<int64_t, int64_t>> featShapes = {{80, 80}, {40, 40}, {20, 20}};
    auto strides = createStrides({8.0f, 16.0f, 32.0f});

    auto [anchors, strideTensor] = makeAnchors(featShapes, strides, torch::kFloat32, device_, 0.5f);

    // Total: 80*80 + 40*40 + 20*20 = 8400
    EXPECT_EQ(anchors.size(0), 8400);
}

TEST_F(MakeAnchorsTest, ShapeBased_DtypeFloat32) {
    std::vector<std::pair<int64_t, int64_t>> featShapes = {{10, 10}};
    auto strides = createStrides({8.0f});

    auto [anchors, strideTensor] = makeAnchors(featShapes, strides, torch::kFloat32, device_, 0.5f);

    EXPECT_EQ(anchors.dtype(), torch::kFloat32);
    EXPECT_EQ(strideTensor.dtype(), torch::kFloat32);
}

TEST_F(MakeAnchorsTest, ShapeBased_DtypeFloat16) {
    std::vector<std::pair<int64_t, int64_t>> featShapes = {{10, 10}};
    auto strides = torch::tensor({8.0f}, torch::TensorOptions().dtype(torch::kFloat16).device(device_));

    auto [anchors, strideTensor] = makeAnchors(featShapes, strides, torch::kFloat16, device_, 0.5f);

    EXPECT_EQ(anchors.dtype(), torch::kFloat16);
    EXPECT_EQ(strideTensor.dtype(), torch::kFloat16);
}

TEST_F(MakeAnchorsTest, ShapeBased_DeviceCPU) {
    std::vector<std::pair<int64_t, int64_t>> featShapes = {{10, 10}};
    auto strides = torch::tensor({8.0f}, torch::TensorOptions().device(torch::kCPU));

    auto [anchors, strideTensor] = makeAnchors(featShapes, strides, torch::kFloat32, torch::kCPU, 0.5f);

    EXPECT_EQ(anchors.device().type(), torch::kCPU);
    EXPECT_EQ(strideTensor.device().type(), torch::kCPU);
}

TEST_F(MakeAnchorsTest, ShapeBased_DeviceCUDA) {
#ifdef USE_CUDA
    if (!torch::cuda::is_available()) {
        return;  // CUDA not available
    }

    std::vector<std::pair<int64_t, int64_t>> featShapes = {{10, 10}};
    auto strides = torch::tensor({8.0f}, torch::TensorOptions().device(torch::kCUDA));

    auto [anchors, strideTensor] = makeAnchors(featShapes, strides, torch::kFloat32, torch::kCUDA, 0.5f);

    EXPECT_EQ(anchors.device().type(), torch::kCUDA);
    EXPECT_EQ(strideTensor.device().type(), torch::kCUDA);
#else
    return;  // CUDA not enabled
#endif
}

TEST_F(MakeAnchorsTest, ShapeBased_CustomOffset) {
    std::vector<std::pair<int64_t, int64_t>> featShapes = {{4, 4}};
    auto strides = createStrides({8.0f});

    auto [anchors, strideTensor] = makeAnchors(featShapes, strides, torch::kFloat32, device_, 0.25f);

    auto anchorsCpu = anchors.cpu();
    // With offset=0.25, first anchor at (0.25, 0.25)
    EXPECT_NEAR(anchorsCpu[0][0].item<float>(), 0.25f, 1e-5f);
    EXPECT_NEAR(anchorsCpu[0][1].item<float>(), 0.25f, 1e-5f);
}

TEST_F(MakeAnchorsTest, ShapeBased_Exception_EmptyShapes) {
    std::vector<std::pair<int64_t, int64_t>> featShapes;  // Empty
    auto strides = createStrides({8.0f});

    EXPECT_THROW(makeAnchors(featShapes, strides, torch::kFloat32, device_, 0.5f), std::invalid_argument);
}

// =============================================================================
// Consistency Tests between Overloads
// =============================================================================

TEST_F(MakeAnchorsTest, Consistency_BothOverloads) {
    // Both overloads should produce the same results for equivalent inputs
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 10, 10));

    std::vector<std::pair<int64_t, int64_t>> featShapes = {{10, 10}};

    auto strides = createStrides({8.0f});

    auto [anchors1, strides1] = makeAnchors(feats, strides, 0.5f);
    auto [anchors2, strides2] = makeAnchors(featShapes, strides, torch::kFloat32, device_, 0.5f);

    EXPECT_EQ(anchors1.size(0), anchors2.size(0));
    EXPECT_TRUE(torch::allclose(anchors1, anchors2, 1e-5, 1e-5));
    EXPECT_TRUE(torch::allclose(strides1, strides2, 1e-5, 1e-5));
}

TEST_F(MakeAnchorsTest, Consistency_MultiScale) {
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 20, 20));
    feats.push_back(createFeatureTensor(1, 256, 10, 10));

    std::vector<std::pair<int64_t, int64_t>> featShapes = {{20, 20}, {10, 10}};

    auto strides = createStrides({8.0f, 16.0f});

    auto [anchors1, strides1] = makeAnchors(feats, strides, 0.5f);
    auto [anchors2, strides2] = makeAnchors(featShapes, strides, torch::kFloat32, device_, 0.5f);

    EXPECT_EQ(anchors1.size(0), anchors2.size(0));
    EXPECT_TRUE(torch::allclose(anchors1, anchors2, 1e-5, 1e-5));
}

// =============================================================================
// Grid Pattern Tests
// =============================================================================

TEST_F(MakeAnchorsTest, GridPattern_RowMajorOrder) {
    // Verify anchors are in row-major order (y then x)
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 3, 3));

    auto strides = createStrides({1.0f});  // stride=1 for easy verification

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    auto anchorsCpu = anchors.cpu();

    // Row 0: (0.5,0.5), (1.5,0.5), (2.5,0.5)
    // Row 1: (0.5,1.5), (1.5,1.5), (2.5,1.5)
    // Row 2: (0.5,2.5), (1.5,2.5), (2.5,2.5)

    // Check first row
    EXPECT_NEAR(anchorsCpu[0][0].item<float>(), 0.5f, 1e-5f);  // x
    EXPECT_NEAR(anchorsCpu[0][1].item<float>(), 0.5f, 1e-5f);  // y
    EXPECT_NEAR(anchorsCpu[2][0].item<float>(), 2.5f, 1e-5f);  // x
    EXPECT_NEAR(anchorsCpu[2][1].item<float>(), 0.5f, 1e-5f);  // y

    // Check second row starts at index 3
    EXPECT_NEAR(anchorsCpu[3][0].item<float>(), 0.5f, 1e-5f);  // x
    EXPECT_NEAR(anchorsCpu[3][1].item<float>(), 1.5f, 1e-5f);  // y
}

TEST_F(MakeAnchorsTest, GridPattern_NonSquareFeatureMap) {
    // Non-square feature map: 3x5
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 3, 5));

    auto strides = createStrides({8.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    // Should have 3*5 = 15 anchor points
    EXPECT_EQ(anchors.size(0), 15);

    auto anchorsCpu = anchors.cpu();
    // Last anchor should be at (4.5, 2.5)
    EXPECT_NEAR(anchorsCpu[14][0].item<float>(), 4.5f, 1e-5f);  // x
    EXPECT_NEAR(anchorsCpu[14][1].item<float>(), 2.5f, 1e-5f);  // y
}

// =============================================================================
// Stride Tensor Tests
// =============================================================================

TEST_F(MakeAnchorsTest, StrideTensor_SingleScale) {
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 5, 5));

    auto strides = createStrides({16.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    // All 25 stride values should be 16.0
    auto allEqual = (strideTensor == 16.0f).all().item<bool>();
    EXPECT_TRUE(allEqual);
}

TEST_F(MakeAnchorsTest, StrideTensor_MultiScale) {
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 4, 4));   // 16 anchors
    feats.push_back(createFeatureTensor(1, 256, 2, 2));   // 4 anchors

    auto strides = createStrides({8.0f, 16.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    auto strideCpu = strideTensor.cpu();

    // First 16 should be 8.0
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(strideCpu[i][0].item<float>(), 8.0f, 1e-5f);
    }
    // Last 4 should be 16.0
    for (int i = 16; i < 20; ++i) {
        EXPECT_NEAR(strideCpu[i][0].item<float>(), 16.0f, 1e-5f);
    }
}

// =============================================================================
// Device Consistency Tests
// =============================================================================

TEST_F(MakeAnchorsTest, DeviceConsistency_FeatureTensor) {
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 10, 10));

    auto strides = createStrides({8.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    EXPECT_EQ(anchors.device().type(), device_.type());
    EXPECT_EQ(strideTensor.device().type(), device_.type());
}

TEST_F(MakeAnchorsTest, DeviceConsistency_ShapeBased) {
    std::vector<std::pair<int64_t, int64_t>> featShapes = {{10, 10}};
    auto strides = createStrides({8.0f});

    auto [anchors, strideTensor] = makeAnchors(featShapes, strides, torch::kFloat32, device_, 0.5f);

    EXPECT_EQ(anchors.device().type(), device_.type());
    EXPECT_EQ(strideTensor.device().type(), device_.type());
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(MakeAnchorsTest, EdgeCase_SingleCell) {
    // 1x1 feature map
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 1, 1));

    auto strides = createStrides({32.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    EXPECT_EQ(anchors.size(0), 1);

    auto anchorsCpu = anchors.cpu();
    EXPECT_NEAR(anchorsCpu[0][0].item<float>(), 0.5f, 1e-5f);
    EXPECT_NEAR(anchorsCpu[0][1].item<float>(), 0.5f, 1e-5f);
}

TEST_F(MakeAnchorsTest, EdgeCase_LargeFeatureMap) {
    // Large feature map: 160x160
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 64, 160, 160));

    auto strides = createStrides({4.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    EXPECT_EQ(anchors.size(0), 25600);  // 160*160
    EXPECT_FALSE(anchors.isnan().any().item<bool>());
}

TEST_F(MakeAnchorsTest, EdgeCase_ManyScales) {
    // 5 scale levels
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(1, 256, 80, 80));
    feats.push_back(createFeatureTensor(1, 256, 40, 40));
    feats.push_back(createFeatureTensor(1, 256, 20, 20));
    feats.push_back(createFeatureTensor(1, 256, 10, 10));
    feats.push_back(createFeatureTensor(1, 256, 5, 5));

    auto strides = createStrides({8.0f, 16.0f, 32.0f, 64.0f, 128.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    // Total: 6400 + 1600 + 400 + 100 + 25 = 8525
    EXPECT_EQ(anchors.size(0), 8525);
}

TEST_F(MakeAnchorsTest, EdgeCase_DifferentBatchSizes) {
    // Feature tensors with different batch sizes (should work, only H,W matter)
    std::vector<torch::Tensor> feats;
    feats.push_back(createFeatureTensor(4, 256, 10, 10));  // batch=4

    auto strides = createStrides({8.0f});

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    // Should still be 100 anchors (batch size doesn't affect anchor count)
    EXPECT_EQ(anchors.size(0), 100);
}
