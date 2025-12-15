/**
 * @file AttentionTest.cpp
 * @brief Unit tests for Attention modules
 *
 * Tests cover:
 * - ChannelAttention, SpatialAttention, CBAM
 * - Attention, PSABlock, PSA
 *
 * @author WheelDL Team
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "Model/Modules/Attention.h"

using namespace WheelDL::Model::Modules;

// ============================================================================
// Test Fixture
// ============================================================================

class AttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
    }

    torch::Tensor createInput(int64_t n = 1, int64_t c = 64, int64_t h = 32, int64_t w = 32) {
        return torch::randn({n, c, h, w});
    }
};

// ============================================================================
// ChannelAttention Tests
// ============================================================================

TEST_F(AttentionTest, ChannelAttention_Constructor) {
    ChannelAttention ca(64);

    EXPECT_TRUE(ca.ptr());
}

TEST_F(AttentionTest, ChannelAttention_Forward) {
    ChannelAttention ca(64);

    auto input = createInput(1, 64, 32, 32);
    auto output = ca->forward(input);

    // ChannelAttention preserves shape
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(AttentionTest, ChannelAttention_DifferentChannels) {
    ChannelAttention ca(128);

    auto input = createInput(1, 128, 16, 16);
    auto output = ca->forward(input);

    EXPECT_EQ(output.size(1), 128);
}

TEST_F(AttentionTest, ChannelAttention_BatchSize) {
    ChannelAttention ca(64);

    auto input = createInput(4, 64, 16, 16);
    auto output = ca->forward(input);

    EXPECT_EQ(output.size(0), 4);  // Batch preserved
}

TEST_F(AttentionTest, ChannelAttention_NoNaN) {
    ChannelAttention ca(64);

    auto input = createInput(1, 64, 32, 32);
    auto output = ca->forward(input);

    EXPECT_FALSE(output.isnan().any().item<bool>());
    EXPECT_FALSE(output.isinf().any().item<bool>());
}

// ============================================================================
// SpatialAttention Tests
// ============================================================================

TEST_F(AttentionTest, SpatialAttention_Constructor_k7) {
    SpatialAttention sa(7);

    EXPECT_TRUE(sa.ptr());
}

TEST_F(AttentionTest, SpatialAttention_Constructor_k3) {
    SpatialAttention sa(3);

    EXPECT_TRUE(sa.ptr());
}

TEST_F(AttentionTest, SpatialAttention_Constructor_Default) {
    SpatialAttention sa;  // Default kernel size = 7

    EXPECT_TRUE(sa.ptr());
}

TEST_F(AttentionTest, SpatialAttention_Forward) {
    SpatialAttention sa(7);

    auto input = createInput(1, 64, 32, 32);
    auto output = sa->forward(input);

    // SpatialAttention preserves shape
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(AttentionTest, SpatialAttention_Forward_k3) {
    SpatialAttention sa(3);

    auto input = createInput(1, 64, 32, 32);
    auto output = sa->forward(input);

    EXPECT_EQ(output.size(1), 64);
}

TEST_F(AttentionTest, SpatialAttention_BatchSize) {
    SpatialAttention sa(7);

    auto input = createInput(4, 64, 16, 16);
    auto output = sa->forward(input);

    EXPECT_EQ(output.size(0), 4);
}

TEST_F(AttentionTest, SpatialAttention_NoNaN) {
    SpatialAttention sa(7);

    auto input = createInput(1, 64, 32, 32);
    auto output = sa->forward(input);

    EXPECT_FALSE(output.isnan().any().item<bool>());
    EXPECT_FALSE(output.isinf().any().item<bool>());
}

// ============================================================================
// CBAM Tests
// ============================================================================

TEST_F(AttentionTest, CBAM_Constructor) {
    CBAM cbam(64);

    EXPECT_TRUE(cbam.ptr());
}

TEST_F(AttentionTest, CBAM_Constructor_CustomKernel) {
    CBAM cbam(64, 3);

    EXPECT_TRUE(cbam.ptr());
}

TEST_F(AttentionTest, CBAM_Forward) {
    CBAM cbam(64);

    auto input = createInput(1, 64, 32, 32);
    auto output = cbam->forward(input);

    // CBAM preserves shape
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(AttentionTest, CBAM_DifferentChannels) {
    CBAM cbam(128);

    auto input = createInput(1, 128, 16, 16);
    auto output = cbam->forward(input);

    EXPECT_EQ(output.size(1), 128);
}

TEST_F(AttentionTest, CBAM_BatchSize) {
    CBAM cbam(64);

    auto input = createInput(4, 64, 16, 16);
    auto output = cbam->forward(input);

    EXPECT_EQ(output.size(0), 4);
}

TEST_F(AttentionTest, CBAM_NoNaN) {
    CBAM cbam(64);

    auto input = createInput(1, 64, 32, 32);
    auto output = cbam->forward(input);

    EXPECT_FALSE(output.isnan().any().item<bool>());
    EXPECT_FALSE(output.isinf().any().item<bool>());
}

TEST_F(AttentionTest, CBAM_Gradient) {
    CBAM cbam(64);

    auto input = torch::randn({1, 64, 16, 16}, torch::requires_grad(true));
    auto output = cbam->forward(input);
    auto loss = output.sum();
    loss.backward();

    EXPECT_TRUE(input.grad().defined());
    EXPECT_FALSE(input.grad().isnan().any().item<bool>());
}

// ============================================================================
// Attention Tests
// ============================================================================

TEST_F(AttentionTest, Attention_Constructor) {
    Attention attn(64, 8);

    EXPECT_TRUE(attn.ptr());
}

TEST_F(AttentionTest, Attention_Constructor_DefaultHeads) {
    Attention attn(64);  // Default numHeads = 8

    EXPECT_TRUE(attn.ptr());
}

TEST_F(AttentionTest, Attention_Constructor_CustomAttnRatio) {
    Attention attn(64, 8, 0.25);

    EXPECT_TRUE(attn.ptr());
}

TEST_F(AttentionTest, Attention_Forward) {
    Attention attn(64, 8);

    auto input = createInput(1, 64, 16, 16);
    auto output = attn->forward(input);

    // Attention preserves shape
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 16);
    EXPECT_EQ(output.size(3), 16);
}

TEST_F(AttentionTest, Attention_DivisibleHeads) {
    // dim=64 is divisible by numHeads=8
    Attention attn(64, 8);

    auto input = createInput(1, 64, 16, 16);
    auto output = attn->forward(input);

    EXPECT_EQ(output.size(1), 64);
}

TEST_F(AttentionTest, Attention_DifferentHeads) {
    // dim=64 is divisible by numHeads=4
    Attention attn(64, 4);

    auto input = createInput(1, 64, 16, 16);
    auto output = attn->forward(input);

    EXPECT_EQ(output.size(1), 64);
}

TEST_F(AttentionTest, Attention_BatchSize) {
    Attention attn(64, 8);

    auto input = createInput(4, 64, 16, 16);
    auto output = attn->forward(input);

    EXPECT_EQ(output.size(0), 4);
}

TEST_F(AttentionTest, Attention_NoNaN) {
    Attention attn(64, 8);

    auto input = createInput(1, 64, 16, 16);
    auto output = attn->forward(input);

    EXPECT_FALSE(output.isnan().any().item<bool>());
    EXPECT_FALSE(output.isinf().any().item<bool>());
}

TEST_F(AttentionTest, Attention_Gradient) {
    Attention attn(64, 8);

    auto input = torch::randn({1, 64, 16, 16}, torch::requires_grad(true));
    auto output = attn->forward(input);
    auto loss = output.sum();
    loss.backward();

    EXPECT_TRUE(input.grad().defined());
    EXPECT_FALSE(input.grad().isnan().any().item<bool>());
}

// ============================================================================
// PSABlock Tests
// ============================================================================

TEST_F(AttentionTest, PSABlock_Constructor) {
    PSABlock psablock(64);

    EXPECT_TRUE(psablock.ptr());
}

TEST_F(AttentionTest, PSABlock_Constructor_CustomParams) {
    PSABlock psablock(64, 0.25, 8, false);

    EXPECT_TRUE(psablock.ptr());
}

TEST_F(AttentionTest, PSABlock_Forward) {
    PSABlock psablock(64);

    auto input = createInput(1, 64, 16, 16);
    auto output = psablock->forward(input);

    // PSABlock preserves shape
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 16);
    EXPECT_EQ(output.size(3), 16);
}

TEST_F(AttentionTest, PSABlock_WithShortcut) {
    PSABlock psablock(64, 0.5, 4, true);

    auto input = createInput(1, 64, 16, 16);
    auto output = psablock->forward(input);

    EXPECT_EQ(output.size(1), 64);
}

TEST_F(AttentionTest, PSABlock_NoShortcut) {
    PSABlock psablock(64, 0.5, 4, false);

    auto input = createInput(1, 64, 16, 16);
    auto output = psablock->forward(input);

    EXPECT_EQ(output.size(1), 64);
}

TEST_F(AttentionTest, PSABlock_DifferentChannels) {
    PSABlock psablock(128);

    auto input = createInput(1, 128, 16, 16);
    auto output = psablock->forward(input);

    EXPECT_EQ(output.size(1), 128);
}

TEST_F(AttentionTest, PSABlock_BatchSize) {
    PSABlock psablock(64);

    auto input = createInput(4, 64, 16, 16);
    auto output = psablock->forward(input);

    EXPECT_EQ(output.size(0), 4);
}

TEST_F(AttentionTest, PSABlock_NoNaN) {
    PSABlock psablock(64);

    auto input = createInput(1, 64, 16, 16);
    auto output = psablock->forward(input);

    EXPECT_FALSE(output.isnan().any().item<bool>());
    EXPECT_FALSE(output.isinf().any().item<bool>());
}

TEST_F(AttentionTest, PSABlock_Gradient) {
    PSABlock psablock(64);

    auto input = torch::randn({1, 64, 16, 16}, torch::requires_grad(true));
    auto output = psablock->forward(input);
    auto loss = output.sum();
    loss.backward();

    EXPECT_TRUE(input.grad().defined());
    EXPECT_FALSE(input.grad().isnan().any().item<bool>());
}

// ============================================================================
// PSA Tests
// ============================================================================

TEST_F(AttentionTest, PSA_Constructor) {
    PSA psa(64, 64);

    EXPECT_TRUE(psa.ptr());
}

TEST_F(AttentionTest, PSA_Constructor_CustomExpansion) {
    PSA psa(64, 64, 0.25);

    EXPECT_TRUE(psa.ptr());
}

TEST_F(AttentionTest, PSA_Forward) {
    PSA psa(64, 64);

    auto input = createInput(1, 64, 16, 16);
    auto output = psa->forward(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 16);
    EXPECT_EQ(output.size(3), 16);
}

// NOTE: PSA requires c1 == c2, so this test is invalid
// TEST_F(AttentionTest, PSA_DifferentChannels) {
//     PSA psa(64, 128);
//
//     auto input = createInput(1, 64, 16, 16);
//     auto output = psa->forward(input);
//
//     EXPECT_EQ(output.size(1), 128);
// }

TEST_F(AttentionTest, PSA_SameChannels) {
    PSA psa(128, 128);

    auto input = createInput(1, 128, 16, 16);
    auto output = psa->forward(input);

    EXPECT_EQ(output.size(1), 128);
}

TEST_F(AttentionTest, PSA_BatchSize) {
    PSA psa(64, 64);

    auto input = createInput(4, 64, 16, 16);
    auto output = psa->forward(input);

    EXPECT_EQ(output.size(0), 4);
}

TEST_F(AttentionTest, PSA_NoNaN) {
    PSA psa(64, 64);

    auto input = createInput(1, 64, 16, 16);
    auto output = psa->forward(input);

    EXPECT_FALSE(output.isnan().any().item<bool>());
    EXPECT_FALSE(output.isinf().any().item<bool>());
}

TEST_F(AttentionTest, PSA_Gradient) {
    PSA psa(64, 64);

    auto input = torch::randn({1, 64, 16, 16}, torch::requires_grad(true));
    auto output = psa->forward(input);
    auto loss = output.sum();
    loss.backward();

    EXPECT_TRUE(input.grad().defined());
    EXPECT_FALSE(input.grad().isnan().any().item<bool>());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(AttentionTest, AllModules_SmallInput) {
    auto input = createInput(1, 64, 8, 8);

    ChannelAttention ca(64);
    EXPECT_NO_THROW(ca->forward(input));

    SpatialAttention sa(7);
    EXPECT_NO_THROW(sa->forward(input));

    CBAM cbam(64);
    EXPECT_NO_THROW(cbam->forward(input));

    Attention attn(64, 8);
    EXPECT_NO_THROW(attn->forward(input));

    PSABlock psablock(64);
    EXPECT_NO_THROW(psablock->forward(input));

    PSA psa(64, 64);
    EXPECT_NO_THROW(psa->forward(input));
}

TEST_F(AttentionTest, AllModules_LargeInput) {
    auto input = createInput(2, 256, 32, 32);

    ChannelAttention ca(256);
    auto out1 = ca->forward(input);
    EXPECT_EQ(out1.size(1), 256);

    SpatialAttention sa(7);
    auto out2 = sa->forward(input);
    EXPECT_EQ(out2.size(1), 256);

    CBAM cbam(256);
    auto out3 = cbam->forward(input);
    EXPECT_EQ(out3.size(1), 256);

    Attention attn(256, 8);
    auto out4 = attn->forward(input);
    EXPECT_EQ(out4.size(1), 256);
}

TEST_F(AttentionTest, Sequential_ChannelAndSpatial) {
    // Test channel attention followed by spatial attention
    ChannelAttention ca(64);
    SpatialAttention sa(7);

    auto input = createInput(1, 64, 32, 32);
    auto after_channel = ca->forward(input);
    auto output = sa->forward(after_channel);

    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(AttentionTest, ChannelAttention_SingleChannel) {
    ChannelAttention ca(1);

    auto input = torch::randn({1, 1, 16, 16});
    auto output = ca->forward(input);

    EXPECT_EQ(output.size(1), 1);
}

TEST_F(AttentionTest, Attention_LargeHeads) {
    // Test with larger number of heads
    Attention attn(128, 16);

    auto input = createInput(1, 128, 8, 8);
    auto output = attn->forward(input);

    EXPECT_EQ(output.size(1), 128);
}

TEST_F(AttentionTest, PSA_SmallExpansion) {
    PSA psa(64, 64, 0.125);

    auto input = createInput(1, 64, 16, 16);
    auto output = psa->forward(input);

    EXPECT_EQ(output.size(1), 64);
}

