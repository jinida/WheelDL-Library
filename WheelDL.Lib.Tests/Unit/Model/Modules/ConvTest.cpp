/**
 * @file ConvTest.cpp
 * @brief Unit tests for Conv modules
 *
 * Tests cover:
 * - Conv, DWConv, Conv2, LightConv, GhostConv, RepConv
 * - ConvTranspose, Focus, Concat, Index
 * - Upsample, MaxPool2d, AvgPool2d, GlobalAvgPool, NaiveConv
 * - autoPad utility function
 *
 * @author WheelDL Team
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "Model/Modules/Conv.h"

using namespace WheelDL::Model::Modules;

// ============================================================================
// Test Fixture
// ============================================================================

class ConvTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
    }

    torch::Tensor createInput(int64_t n = 1, int64_t c = 64, int64_t h = 32, int64_t w = 32) {
        return torch::randn({n, c, h, w});
    }
};

// ============================================================================
// autoPad Tests
// ============================================================================

TEST_F(ConvTest, AutoPad_Kernel1) {
    auto pad = autoPad(1);
    EXPECT_EQ(pad, 0);
}

TEST_F(ConvTest, AutoPad_Kernel3) {
    auto pad = autoPad(3);
    EXPECT_EQ(pad, 1);
}

TEST_F(ConvTest, AutoPad_Kernel5) {
    auto pad = autoPad(5);
    EXPECT_EQ(pad, 2);
}

TEST_F(ConvTest, AutoPad_Kernel7) {
    auto pad = autoPad(7);
    EXPECT_EQ(pad, 3);
}

TEST_F(ConvTest, AutoPad_WithDilation) {
    // With dilation=2, effective kernel size increases
    auto pad = autoPad(3, std::nullopt, 2);
    EXPECT_EQ(pad, 2);  // (k * d - 1) / 2 = (3 * 2 - 1) / 2 = 2
}

TEST_F(ConvTest, AutoPad_WithExplicitPadding) {
    // When explicit padding is provided, it should be used
    auto pad = autoPad(3, 5);
    EXPECT_EQ(pad, 5);
}

// ============================================================================
// Conv Tests
// ============================================================================

TEST_F(ConvTest, Conv_Constructor_Default) {
    Conv conv(64, 128);

    EXPECT_TRUE(conv.ptr());
}

TEST_F(ConvTest, Conv_Constructor_CustomKernel) {
    Conv conv(64, 128, 3, 2);  // k=3, s=2

    auto input = createInput(1, 64, 32, 32);
    auto output = conv->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 16);  // 32 / 2 = 16
    EXPECT_EQ(output.size(3), 16);
}

TEST_F(ConvTest, Conv_Constructor_Groups) {
    // Groups > 1
    Conv conv(64, 64, 3, 1, std::nullopt, 4);  // groups=4

    auto input = createInput(1, 64, 16, 16);
    auto output = conv->forward(input);

    EXPECT_EQ(output.size(1), 64);
}

TEST_F(ConvTest, Conv_Constructor_Dilation) {
    Conv conv(64, 64, 3, 1, std::nullopt, 1, 2);  // dilation=2

    auto input = createInput(1, 64, 32, 32);
    auto output = conv->forward(input);

    EXPECT_EQ(output.size(1), 64);
}

TEST_F(ConvTest, Conv_Constructor_NoActivation) {
    Conv conv(64, 128, 1, 1, std::nullopt, 1, 1, "");  // act=""

    auto input = createInput(1, 64, 16, 16);
    auto output = conv->forward(input);

    EXPECT_EQ(output.size(1), 128);
}

TEST_F(ConvTest, Conv_Forward_BasicShape) {
    Conv conv(3, 64, 3, 1);

    auto input = torch::randn({1, 3, 64, 64});
    auto output = conv->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 64);
    EXPECT_EQ(output.size(3), 64);
}

TEST_F(ConvTest, Conv_Forward_BatchSize) {
    Conv conv(64, 128);

    auto input = createInput(4, 64, 16, 16);
    auto output = conv->forward(input);

    EXPECT_EQ(output.size(0), 4);  // Batch preserved
    EXPECT_EQ(output.size(1), 128);
}

TEST_F(ConvTest, Conv_Forward_Stride) {
    Conv conv(64, 128, 3, 2);  // s=2

    auto input = createInput(1, 64, 32, 32);
    auto output = conv->forward(input);

    EXPECT_EQ(output.size(2), 16);  // H/2
    EXPECT_EQ(output.size(3), 16);  // W/2
}

// ============================================================================
// DWConv Tests
// ============================================================================

TEST_F(ConvTest, DWConv_Constructor) {
    DWConv dwconv(64, 64, 3);

    EXPECT_TRUE(dwconv.ptr());
}

TEST_F(ConvTest, DWConv_Forward) {
    DWConv dwconv(64, 64, 3);

    auto input = createInput(1, 64, 32, 32);
    auto output = dwconv->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(ConvTest, DWConv_WithStride) {
    DWConv dwconv(64, 64, 3, 2);  // s=2

    auto input = createInput(1, 64, 32, 32);
    auto output = dwconv->forward(input);

    EXPECT_EQ(output.size(2), 16);
    EXPECT_EQ(output.size(3), 16);
}

// ============================================================================
// Conv2 Tests
// ============================================================================

TEST_F(ConvTest, Conv2_Constructor) {
    Conv2 conv2(64, 128);

    EXPECT_TRUE(conv2.ptr());
}

TEST_F(ConvTest, Conv2_Forward) {
    Conv2 conv2(64, 128);

    auto input = createInput(1, 64, 32, 32);
    auto output = conv2->forward(input);

    EXPECT_EQ(output.size(1), 128);
}

TEST_F(ConvTest, Conv2_CustomParams) {
    Conv2 conv2(64, 128, 3, 2);  // k=3, s=2

    auto input = createInput(1, 64, 32, 32);
    auto output = conv2->forward(input);

    EXPECT_EQ(output.size(2), 16);  // H/2
}

// ============================================================================
// LightConv Tests
// ============================================================================

TEST_F(ConvTest, LightConv_Constructor) {
    LightConv lightconv(64, 128);

    EXPECT_TRUE(lightconv.ptr());
}

TEST_F(ConvTest, LightConv_Forward) {
    LightConv lightconv(64, 128, 3);

    auto input = createInput(1, 64, 32, 32);
    auto output = lightconv->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 128);
}

TEST_F(ConvTest, LightConv_CustomKernel) {
    LightConv lightconv(64, 128, 5);  // k=5 for DWConv

    auto input = createInput(1, 64, 32, 32);
    auto output = lightconv->forward(input);

    EXPECT_EQ(output.size(1), 128);
}

// ============================================================================
// GhostConv Tests
// ============================================================================

TEST_F(ConvTest, GhostConv_Constructor) {
    GhostConv ghostconv(64, 128);

    EXPECT_TRUE(ghostconv.ptr());
}

TEST_F(ConvTest, GhostConv_Forward) {
    GhostConv ghostconv(64, 128);

    auto input = createInput(1, 64, 32, 32);
    auto output = ghostconv->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 128);
}

TEST_F(ConvTest, GhostConv_WithStride) {
    GhostConv ghostconv(64, 128, 3, 2);

    auto input = createInput(1, 64, 32, 32);
    auto output = ghostconv->forward(input);

    EXPECT_EQ(output.size(2), 16);
}

// ============================================================================
// RepConv Tests
// ============================================================================

TEST_F(ConvTest, RepConv_Constructor) {
    RepConv repconv(64, 64, 3, 1, 1);

    EXPECT_TRUE(repconv.ptr());
}

TEST_F(ConvTest, RepConv_Forward_Training) {
    RepConv repconv(64, 64, 3, 1, 1);
    repconv->train();

    auto input = createInput(1, 64, 32, 32);
    auto output = repconv->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
}

TEST_F(ConvTest, RepConv_BnIdentity) {
    // With bn=true, includes batch norm identity branch
    RepConv repconv(64, 64, 3, 1, 1, 1, 1, "SiLU", true);

    auto input = createInput(1, 64, 32, 32);
    auto output = repconv->forward(input);

    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// ConvTranspose Tests
// ============================================================================

TEST_F(ConvTest, ConvTranspose_Constructor) {
    ConvTranspose convt(64, 32);

    EXPECT_TRUE(convt.ptr());
}

TEST_F(ConvTest, ConvTranspose_Forward) {
    ConvTranspose convt(64, 32, 2, 2);  // k=2, s=2

    auto input = createInput(1, 64, 16, 16);
    auto output = convt->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 32);
    EXPECT_EQ(output.size(2), 32);  // 16 * 2 = 32
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(ConvTest, ConvTranspose_NoBn) {
    ConvTranspose convt(64, 32, 2, 2, 0, false);  // bn=false

    auto input = createInput(1, 64, 16, 16);
    auto output = convt->forward(input);

    EXPECT_EQ(output.size(1), 32);
}

// ============================================================================
// Focus Tests
// ============================================================================

TEST_F(ConvTest, Focus_Constructor) {
    Focus focus(3, 64);

    EXPECT_TRUE(focus.ptr());
}

TEST_F(ConvTest, Focus_Forward) {
    Focus focus(3, 64);

    auto input = torch::randn({1, 3, 64, 64});
    auto output = focus->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);  // H/2
    EXPECT_EQ(output.size(3), 32);  // W/2
}

// ============================================================================
// Concat Tests
// ============================================================================

TEST_F(ConvTest, Concat_Constructor) {
    Concat concat(1);

    EXPECT_TRUE(concat.ptr());
}

TEST_F(ConvTest, Concat_Forward_TwoTensors) {
    Concat concat(1);  // dim=1 (channels)

    auto t1 = torch::randn({1, 32, 16, 16});
    auto t2 = torch::randn({1, 64, 16, 16});

    auto output = concat->forward({t1, t2});

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 96);  // 32 + 64
    EXPECT_EQ(output.size(2), 16);
    EXPECT_EQ(output.size(3), 16);
}

TEST_F(ConvTest, Concat_Forward_MultipleTensors) {
    Concat concat(1);

    auto t1 = torch::randn({1, 32, 16, 16});
    auto t2 = torch::randn({1, 32, 16, 16});
    auto t3 = torch::randn({1, 32, 16, 16});
    auto t4 = torch::randn({1, 32, 16, 16});

    auto output = concat->forward({t1, t2, t3, t4});

    EXPECT_EQ(output.size(1), 128);  // 32 * 4
}

TEST_F(ConvTest, Concat_DifferentDimension) {
    Concat concat(2);  // dim=2 (height)

    auto t1 = torch::randn({1, 32, 8, 16});
    auto t2 = torch::randn({1, 32, 8, 16});

    auto output = concat->forward({t1, t2});

    EXPECT_EQ(output.size(2), 16);  // 8 + 8
}

// ============================================================================
// Index Tests
// ============================================================================

TEST_F(ConvTest, Index_Constructor) {
    Index index(0);

    EXPECT_TRUE(index.ptr());
}

TEST_F(ConvTest, Index_Forward) {
    Index index(1);  // Select index 1

    auto t0 = torch::randn({1, 32, 16, 16});
    auto t1 = torch::randn({1, 64, 16, 16});
    auto t2 = torch::randn({1, 128, 16, 16});

    auto output = index->forward({t0, t1, t2});

    EXPECT_EQ(output.size(1), 64);  // Selected t1
}

TEST_F(ConvTest, Index_SelectFirst) {
    Index index(0);

    auto t0 = torch::randn({1, 32, 16, 16});
    auto t1 = torch::randn({1, 64, 16, 16});

    auto output = index->forward({t0, t1});

    EXPECT_EQ(output.size(1), 32);
}

// ============================================================================
// Upsample Tests
// ============================================================================

TEST_F(ConvTest, Upsample_Constructor) {
    Upsample upsample(2.0);

    EXPECT_TRUE(upsample.ptr());
}

TEST_F(ConvTest, Upsample_Forward) {
    Upsample upsample(2.0);

    auto input = createInput(1, 64, 16, 16);
    auto output = upsample->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);  // 16 * 2
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(ConvTest, Upsample_Scale4) {
    Upsample upsample(4.0);

    auto input = createInput(1, 64, 8, 8);
    auto output = upsample->forward(input);

    EXPECT_EQ(output.size(2), 32);  // 8 * 4
    EXPECT_EQ(output.size(3), 32);
}

// ============================================================================
// MaxPool2d Tests
// ============================================================================

TEST_F(ConvTest, MaxPool2d_Constructor) {
    MaxPool2d pool(2);

    EXPECT_TRUE(pool.ptr());
}

TEST_F(ConvTest, MaxPool2d_Forward) {
    MaxPool2d pool(2, 2);  // k=2, s=2

    auto input = createInput(1, 64, 32, 32);
    auto output = pool->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 16);  // H/2
    EXPECT_EQ(output.size(3), 16);
}

TEST_F(ConvTest, MaxPool2d_CustomParams) {
    MaxPool2d pool(3, 2, 1);  // k=3, s=2, p=1

    auto input = createInput(1, 64, 32, 32);
    auto output = pool->forward(input);

    EXPECT_EQ(output.size(2), 16);
}

// ============================================================================
// AvgPool2d Tests
// ============================================================================

TEST_F(ConvTest, AvgPool2d_Constructor) {
    AvgPool2d pool(2);

    EXPECT_TRUE(pool.ptr());
}

TEST_F(ConvTest, AvgPool2d_Forward) {
    AvgPool2d pool(2, 2);

    auto input = createInput(1, 64, 32, 32);
    auto output = pool->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 16);
    EXPECT_EQ(output.size(3), 16);
}

// ============================================================================
// GlobalAvgPool Tests
// ============================================================================

TEST_F(ConvTest, GlobalAvgPool_Constructor) {
    GlobalAvgPool gap;

    EXPECT_TRUE(gap.ptr());
}

TEST_F(ConvTest, GlobalAvgPool_Forward) {
    GlobalAvgPool gap;

    auto input = createInput(1, 64, 32, 32);
    auto output = gap->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 1);  // Global pool -> 1x1
    EXPECT_EQ(output.size(3), 1);
}

TEST_F(ConvTest, GlobalAvgPool_DifferentSizes) {
    GlobalAvgPool gap;

    auto input1 = createInput(2, 128, 7, 7);
    auto output1 = gap->forward(input1);
    EXPECT_EQ(output1.size(2), 1);

    auto input2 = createInput(4, 256, 14, 14);
    auto output2 = gap->forward(input2);
    EXPECT_EQ(output2.size(2), 1);
}

// ============================================================================
// NaiveConv Tests
// ============================================================================

TEST_F(ConvTest, NaiveConv_Constructor) {
    NaiveConv conv(64, 128);

    EXPECT_TRUE(conv.ptr());
}

TEST_F(ConvTest, NaiveConv_Forward) {
    NaiveConv conv(64, 128, 3);

    auto input = createInput(1, 64, 32, 32);
    auto output = conv->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 128);
}

TEST_F(ConvTest, NaiveConv_NoBNorAct) {
    // NaiveConv should not have BatchNorm or Activation
    NaiveConv conv(64, 64, 1);

    auto input = createInput(1, 64, 16, 16);
    auto output = conv->forward(input);

    // Output can be negative (no ReLU)
    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// DWConvTranspose2d Tests
// ============================================================================

TEST_F(ConvTest, DWConvTranspose2d_Constructor) {
    DWConvTranspose2d dwt(64, 64, 2, 2);

    EXPECT_TRUE(dwt.ptr());
}

TEST_F(ConvTest, DWConvTranspose2d_Forward) {
    DWConvTranspose2d dwt(64, 64, 2, 2);

    auto input = createInput(1, 64, 16, 16);
    auto output = dwt->forward(input);

    EXPECT_EQ(output.size(2), 32);  // Upsampled
}

// ============================================================================
// Batch Size Tests
// ============================================================================

TEST_F(ConvTest, AllModules_BatchSize) {
    auto input = createInput(4, 64, 16, 16);

    Conv conv(64, 64);
    EXPECT_EQ(conv->forward(input).size(0), 4);

    DWConv dwconv(64, 64, 3);
    EXPECT_EQ(dwconv->forward(input).size(0), 4);

    Conv2 conv2(64, 64);
    EXPECT_EQ(conv2->forward(input).size(0), 4);

    LightConv lightconv(64, 64);
    EXPECT_EQ(lightconv->forward(input).size(0), 4);

    GhostConv ghostconv(64, 64);
    EXPECT_EQ(ghostconv->forward(input).size(0), 4);
}

// ============================================================================
// Gradient Tests
// ============================================================================

TEST_F(ConvTest, Conv_Gradient) {
    Conv conv(64, 128);

    auto input = torch::randn({1, 64, 16, 16}, torch::requires_grad(true));
    auto output = conv->forward(input);
    auto loss = output.sum();
    loss.backward();

    EXPECT_TRUE(input.grad().defined());
    EXPECT_FALSE(input.grad().isnan().any().item<bool>());
}

