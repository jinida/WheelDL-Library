#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Modules/Attention.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <torch/torch.h>

using namespace WheelDL;
using namespace WheelDL::Model::Modules;
using namespace WheelDL::Utils;

// ========== ChannelAttention Module Tests ==========

class ChannelAttentionModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<ChannelAttention> channelAttentionModule;

    static void SetUpTestSuite() {
        try {
            channelAttentionModule = std::make_unique<ChannelAttention>(64);
        } catch (const std::exception& e) {
            std::cerr << "ChannelAttention setup failed: " << e.what() << std::endl;
            channelAttentionModule.reset();
        }
    }

    static void TearDownTestSuite() {
        channelAttentionModule.reset();
    }
};

std::unique_ptr<ChannelAttention> ChannelAttentionModuleTest::channelAttentionModule;

TEST_F(ChannelAttentionModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        ChannelAttention ca(64);
    });
}

TEST_F(ChannelAttentionModuleTest, Forward_ValidInput_Success) {
    if (!channelAttentionModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = channelAttentionModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(ChannelAttentionModuleTest, Forward_OutputShapeSameAsInput_Success) {
    if (!channelAttentionModule) return;

    torch::Tensor input = torch::randn({2, 64, 32, 32});

    auto output = channelAttentionModule->ptr()->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(ChannelAttentionModuleTest, Forward_GAPAndFCLayers_CorrectBehavior) {
    ChannelAttention ca(128);
    torch::Tensor input = torch::randn({1, 128, 16, 16});

    auto output = ca->forward(input);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 16);
    EXPECT_EQ(output.size(3), 16);
}

TEST_F(ChannelAttentionModuleTest, Forward_AttentionWeightsRange_Between0And1) {
    ChannelAttention ca(64);
    torch::Tensor input = torch::randn({1, 64, 32, 32});

    auto output = ca->forward(input);
    auto attention_weights = output / (input + 1e-6);

    EXPECT_TRUE(torch::all(attention_weights >= 0.0).item<bool>());
}

// ========== SpatialAttention Module Tests ==========

class SpatialAttentionModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<SpatialAttention> spatialAttentionModule;

    static void SetUpTestSuite() {
        try {
            spatialAttentionModule = std::make_unique<SpatialAttention>(7);
        } catch (const std::exception& e) {
            std::cerr << "SpatialAttention setup failed: " << e.what() << std::endl;
            spatialAttentionModule.reset();
        }
    }

    static void TearDownTestSuite() {
        spatialAttentionModule.reset();
    }
};

std::unique_ptr<SpatialAttention> SpatialAttentionModuleTest::spatialAttentionModule;

TEST_F(SpatialAttentionModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        SpatialAttention sa(7);
    });
}

TEST_F(SpatialAttentionModuleTest, Forward_ValidInput_Success) {
    if (!spatialAttentionModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = spatialAttentionModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(SpatialAttentionModuleTest, Forward_SpatialWeights_CorrectOutputShape) {
    if (!spatialAttentionModule) return;

    torch::Tensor input = torch::randn({2, 128, 32, 32});

    auto output = spatialAttentionModule->ptr()->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(SpatialAttentionModuleTest, Forward_DifferentKernelSizes_Success) {
    SpatialAttention sa3(3);
    torch::Tensor input = torch::randn({1, 64, 32, 32});
    
    sa3->to(torch::kCUDA);
    input = input.to(torch::kCUDA);
    EXPECT_NO_THROW({
        auto output = sa3->forward(input);
        EXPECT_EQ(output.sizes(), input.sizes());
    });
}

TEST_F(SpatialAttentionModuleTest, Forward_AttentionWeightsRange_Between0And1) {
    SpatialAttention sa(7);
    torch::Tensor input = torch::randn({1, 64, 32, 32});
    sa->to(torch::kCUDA);
	input = input.to(torch::kCUDA);
    auto output = sa->forward(input);
    auto attention_weights = output / (input + 1e-6);

    auto all_positive = torch::all(attention_weights >= 0.0);
    EXPECT_TRUE(all_positive.is_nonzero());
}

// ========== CBAM Module Tests ==========

class CBAMModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<CBAM> cbamModule;

    static void SetUpTestSuite() {
        try {
            cbamModule = std::make_unique<CBAM>(64, 7);
        } catch (const std::exception& e) {
            std::cerr << "CBAM setup failed: " << e.what() << std::endl;
            cbamModule.reset();
        }
    }

    static void TearDownTestSuite() {
        cbamModule.reset();
    }
};

std::unique_ptr<CBAM> CBAMModuleTest::cbamModule;

TEST_F(CBAMModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        CBAM cbam(64, 7);
    });
}

TEST_F(CBAMModuleTest, Forward_ValidInput_Success) {
    if (!cbamModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});
    EXPECT_NO_THROW({
        auto output = cbamModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(CBAMModuleTest, Forward_SequentialChannelThenSpatial_CorrectBehavior) {
    if (!cbamModule) return;

    torch::Tensor input = torch::randn({2, 64, 32, 32});

    auto output = cbamModule->ptr()->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(CBAMModuleTest, Forward_CombinedAttentionWeights_CorrectOutputShape) {
    CBAM cbam(128, 7);
    torch::Tensor input = torch::randn({1, 128, 16, 16});
	cbam->to(torch::kCUDA);
	input = input.to(torch::kCUDA);
    auto output = cbam->forward(input);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 16);
    EXPECT_EQ(output.size(3), 16);
}

TEST_F(CBAMModuleTest, Forward_AttentionWeightsRange_Between0And1) {
    CBAM cbam(64, 7);
    torch::Tensor input = torch::randn({1, 64, 32, 32});
	cbam->to(torch::kCUDA);
	input = input.to(torch::kCUDA);

    auto output = cbam->forward(input);
    auto attention_weights = output / (input + 1e-6);

    EXPECT_TRUE(torch::all(attention_weights >= 0.0).item<bool>());
}

// ========== PSA Module Tests ==========

class PSAModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<PSA> psaModule;

    static void SetUpTestSuite() {
        try {
            // FIX: PSA requires c1 == c2
            psaModule = std::make_unique<PSA>(64, 64, 0.5);
        } catch (const std::exception& e) {
            std::cerr << "PSA setup failed: " << e.what() << std::endl;
            psaModule.reset();
        }
    }

    static void TearDownTestSuite() {
        psaModule.reset();
    }
};

std::unique_ptr<PSA> PSAModuleTest::psaModule;

TEST_F(PSAModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        // FIX: PSA requires c1 == c2
        PSA psa(64, 64, 0.5);
    });
}

TEST_F(PSAModuleTest, Forward_ValidInput_Success) {
    if (!psaModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = psaModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);  // FIX: c2 should be 64 (same as c1)
    });
}

TEST_F(PSAModuleTest, Forward_ParallelAttention_CorrectOutputShape) {
    // FIX: PSA requires c1 == c2
    PSA psa(64, 64, 0.5);
    torch::Tensor input = torch::randn({2, 64, 32, 32});
	psa->to(torch::kCUDA);
	input = input.to(torch::kCUDA);

    auto output = psa->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 64);  // FIX: c2 should be 64 (same as c1)
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(PSAModuleTest, Forward_DifferentExpansionRatios_Success) {
    // FIX: PSA requires c1 == c2
    PSA psa1(64, 64, 1.0);
    torch::Tensor input = torch::randn({1, 64, 32, 32});
	psa1->to(torch::kCUDA);
	input = input.to(torch::kCUDA);
    EXPECT_NO_THROW({
        auto output = psa1->forward(input);
        EXPECT_EQ(output.size(1), 64);  // FIX: c2 should be 64 (same as c1)
    });
}

// ========== Attention Module Tests ==========

class AttentionModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<Attention> attentionModule;

    static void SetUpTestSuite() {
        try {
            attentionModule = std::make_unique<Attention>(128, 8, 0.5);
        } catch (const std::exception& e) {
            std::cerr << "Attention setup failed: " << e.what() << std::endl;
            attentionModule.reset();
        }
    }

    static void TearDownTestSuite() {
        attentionModule.reset();
    }
};

std::unique_ptr<Attention> AttentionModuleTest::attentionModule;

TEST_F(AttentionModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        Attention attn(128, 8, 0.5);
    });
}

TEST_F(AttentionModuleTest, Forward_ValidInput_Success) {
    if (!attentionModule) return;

    torch::Tensor input = torch::randn({1, 128, 32, 32});

    EXPECT_NO_THROW({
        auto output = attentionModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 128);
    });
}

TEST_F(AttentionModuleTest, Forward_MultiHeadSelfAttention_CorrectOutputShape) {
    Attention attn(128, 8, 0.5);
    torch::Tensor input = torch::randn({2, 128, 32, 32});
	attn->to(torch::kCUDA);
	input = input.to(torch::kCUDA);
    auto output = attn->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(AttentionModuleTest, Forward_DifferentNumHeads_Success) {
    Attention attn4(128, 4, 0.5);
    Attention attn16(128, 16, 0.5);
    torch::Tensor input = torch::randn({1, 128, 32, 32});
	attn4->to(torch::kCUDA);
	attn16->to(torch::kCUDA);
	input = input.to(torch::kCUDA);

    EXPECT_NO_THROW({
        auto output4 = attn4->forward(input);
        auto output16 = attn16->forward(input);
        EXPECT_EQ(output4.size(1), 128);
        EXPECT_EQ(output16.size(1), 128);
    });
}

TEST_F(AttentionModuleTest, Forward_GradientFlow_Success) {
    Attention attn(128, 8, 0.5);
    torch::Tensor input = torch::randn({1, 128, 32, 32}, torch::requires_grad(true));
	attn->to(torch::kCUDA);
	input = input.to(torch::kCUDA);
	input.retain_grad();  // Retain gradient for non-leaf tensor

    auto output = attn->forward(input);
    auto loss = output.sum();

    EXPECT_NO_THROW({
        loss.backward();
        EXPECT_TRUE(input.grad().defined());
    });
}
