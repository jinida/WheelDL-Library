/**
 * @file AttentionFactoryTest.cpp
 * @brief Unit tests for Attention Factory classes
 *
 * Tests cover:
 * - PSAFactory, CBAMFactory, AttentionFactory
 * - Default/Custom arguments, scaling, forward pass
 *
 * @author WheelDL Team
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include "Model/Builder/Factory/AttentionFactory.h"
#include "Model/Builder/Factory/ModuleFactoryRegistry.h"

using namespace WheelDL::Model::Builder;

// ============================================================================
// Test Fixture
// ============================================================================

class AttentionFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = &ModuleFactoryRegistry::instance();
    }

    ModuleBuildContext createContext(int64_t inputCh = 64, double widthMult = 1.0,
        double depthMult = 1.0, int64_t repeats = 1) {
        ModuleBuildContext ctx;
        ctx.inputChannels = inputCh;
        ctx.repeats = repeats;
        ctx.defaultAct = "SiLU";
        ctx.numClasses = 80;
        ctx.headChannels = {};
        // applyScale lambda to simulate width/depth multiplier
        ctx.applyScale = [widthMult, depthMult](int64_t channels, int64_t reps) {
            int64_t scaledCh = static_cast<int64_t>(std::round(channels * widthMult));
            int64_t scaledReps = static_cast<int64_t>(std::ceil(reps * depthMult));
            return std::make_pair(scaledCh > 0 ? scaledCh : 1, scaledReps > 0 ? scaledReps : 1);
        };
        return ctx;
    }

    ModuleFactoryRegistry* registry_;
};

// ============================================================================
// PSAFactory Tests
// ============================================================================

TEST_F(AttentionFactoryTest, PSA_SupportedTypes) {
    auto* factory = registry_->getFactory("PSA");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "PSA");
}

TEST_F(AttentionFactoryTest, PSA_DefaultArgs) {
    auto* factory = registry_->getFactory("PSA");
    ASSERT_NE(factory, nullptr);

    // PSA requires c1 == c2, default outCh is 256
    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(256);  // inputCh must match default outCh

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 256);  // Default outCh
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(AttentionFactoryTest, PSA_CustomChannels) {
    auto* factory = registry_->getFactory("PSA");
    ASSERT_NE(factory, nullptr);

    // PSA requires c1 == c2
    YAML::Node args = YAML::Load("[128]");
    auto ctx = createContext(128);  // inputCh must match outCh

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(AttentionFactoryTest, PSA_WidthScaling) {
    auto* factory = registry_->getFactory("PSA");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128]");
    auto ctx = createContext(64, 0.5);  // widthMult=0.5

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);  // 128 * 0.5 = 64
}

TEST_F(AttentionFactoryTest, PSA_ForwardPass) {
    auto* factory = registry_->getFactory("PSA");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 16, 16 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// CBAMFactory Tests
// ============================================================================

TEST_F(AttentionFactoryTest, CBAM_SupportedTypes) {
    auto* factory = registry_->getFactory("CBAM");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "CBAM");
}

TEST_F(AttentionFactoryTest, CBAM_NoArgs) {
    auto* factory = registry_->getFactory("CBAM");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    // CBAM preserves input channels
    EXPECT_EQ(result.outputChannels, 64);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(AttentionFactoryTest, CBAM_DifferentChannels) {
    auto* factory = registry_->getFactory("CBAM");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(128);

    auto result = factory->create(args, ctx);

    // CBAM preserves input channels
    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(AttentionFactoryTest, CBAM_ForwardPass) {
    auto* factory = registry_->getFactory("CBAM");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    // CBAM preserves spatial dimensions
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

// ============================================================================
// AttentionFactory Tests
// ============================================================================

TEST_F(AttentionFactoryTest, Attention_SupportedTypes) {
    auto* factory = registry_->getFactory("Attention");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "Attention");
}

TEST_F(AttentionFactoryTest, Attention_DefaultArgs) {
    auto* factory = registry_->getFactory("Attention");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    // Attention preserves input channels, default numHeads=8
    EXPECT_EQ(result.outputChannels, 64);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(AttentionFactoryTest, Attention_CustomHeads) {
    auto* factory = registry_->getFactory("Attention");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[4]");  // numHeads=4
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(AttentionFactoryTest, Attention_ForwardPass) {
    auto* factory = registry_->getFactory("Attention");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[8]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 16, 16 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// Width Scaling Tests
// ============================================================================

TEST_F(AttentionFactoryTest, PSA_WidthScaling_2x) {
    auto* factory = registry_->getFactory("PSA");
    ASSERT_NE(factory, nullptr);

    // PSA requires c1 == c2, so inputChannels must match scaled output
    // args = [64], widthMult = 2.0 -> output = 128
    // So inputChannels must also be 128
    YAML::Node args = YAML::Load("[64]");
    auto ctx = createContext(128, 2.0);  // inputChannels=128 to match scaled output

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);  // 64 * 2.0 = 128
}

// ============================================================================
// Batch Size Tests
// ============================================================================

TEST_F(AttentionFactoryTest, CBAM_BatchSize) {
    auto* factory = registry_->getFactory("CBAM");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 4, 64, 16, 16 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 4);  // Batch preserved
    EXPECT_EQ(output.size(1), 64);
}

TEST_F(AttentionFactoryTest, Attention_BatchSize) {
    auto* factory = registry_->getFactory("Attention");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 4, 64, 16, 16 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 4);  // Batch preserved
}
