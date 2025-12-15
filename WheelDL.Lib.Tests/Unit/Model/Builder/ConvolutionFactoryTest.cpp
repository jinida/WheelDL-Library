/**
 * @file ConvolutionFactoryTest.cpp
 * @brief Unit tests for Convolution Factory classes
 *
 * Tests cover:
 * - ConvFactory, DWConvFactory, RepConvFactory, GhostConvFactory
 * - FocusFactory, ConvTransposeFactory, NaiveConvFactory
 * - Default/Custom arguments, scaling, forward pass
 *
 * @author WheelDL Team
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include "Model/Builder/Factory/ConvolutionFactory.h"
#include "Model/Builder/Factory/ModuleFactoryRegistry.h"

using namespace WheelDL::Model::Builder;

// ============================================================================
// Test Fixture
// ============================================================================

class ConvolutionFactoryTest : public ::testing::Test {
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
// ConvFactory Tests
// ============================================================================

TEST_F(ConvolutionFactoryTest, Conv_SupportedTypes) {
    auto* factory = registry_->getFactory("Conv");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "Conv");
}

TEST_F(ConvolutionFactoryTest, Conv_DefaultArgs) {
    auto* factory = registry_->getFactory("Conv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(32);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);  // Default outCh
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(ConvolutionFactoryTest, Conv_CustomArgs) {
    auto* factory = registry_->getFactory("Conv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 3, 2]");  // outCh=128, k=3, s=2
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(ConvolutionFactoryTest, Conv_WidthScaling) {
    auto* factory = registry_->getFactory("Conv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64]");
    auto ctx = createContext(32, 0.5);  // widthMult=0.5

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 32);  // 64 * 0.5 = 32
}

TEST_F(ConvolutionFactoryTest, Conv_ForwardPass) {
    auto* factory = registry_->getFactory("Conv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64, 3, 1]");
    auto ctx = createContext(32);

    auto result = factory->create(args, ctx);

    // Create input tensor and run forward pass
    auto input = torch::randn({ 1, 32, 64, 64 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);  // outCh
    EXPECT_EQ(output.size(2), 64);  // same H (padding auto)
    EXPECT_EQ(output.size(3), 64);  // same W (padding auto)
}

TEST_F(ConvolutionFactoryTest, Conv_ForwardPass_Stride2) {
    auto* factory = registry_->getFactory("Conv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64, 3, 2]");  // stride=2
    auto ctx = createContext(32);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 32, 64, 64 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);  // H/2
    EXPECT_EQ(output.size(3), 32);  // W/2
}

// ============================================================================
// DWConvFactory Tests
// ============================================================================

TEST_F(ConvolutionFactoryTest, DWConv_SupportedTypes) {
    auto* factory = registry_->getFactory("DWConv");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "DWConv");
}

TEST_F(ConvolutionFactoryTest, DWConv_DefaultArgs) {
    auto* factory = registry_->getFactory("DWConv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(ConvolutionFactoryTest, DWConv_CustomArgs) {
    auto* factory = registry_->getFactory("DWConv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 3, 2]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(ConvolutionFactoryTest, DWConv_ForwardPass) {
    auto* factory = registry_->getFactory("DWConv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64, 3, 1]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// RepConvFactory Tests
// ============================================================================

TEST_F(ConvolutionFactoryTest, RepConv_SupportedTypes) {
    auto* factory = registry_->getFactory("RepConv");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "RepConv");
}

TEST_F(ConvolutionFactoryTest, RepConv_DefaultArgs) {
    auto* factory = registry_->getFactory("RepConv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(32);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);  // Default
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(ConvolutionFactoryTest, RepConv_CustomArgs) {
    auto* factory = registry_->getFactory("RepConv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 3, 1]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(ConvolutionFactoryTest, RepConv_ForwardPass) {
    auto* factory = registry_->getFactory("RepConv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// GhostConvFactory Tests
// ============================================================================

TEST_F(ConvolutionFactoryTest, GhostConv_SupportedTypes) {
    auto* factory = registry_->getFactory("GhostConv");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "GhostConv");
}

TEST_F(ConvolutionFactoryTest, GhostConv_DefaultArgs) {
    auto* factory = registry_->getFactory("GhostConv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(32);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(ConvolutionFactoryTest, GhostConv_CustomArgs) {
    auto* factory = registry_->getFactory("GhostConv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 3, 2]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(ConvolutionFactoryTest, GhostConv_ForwardPass) {
    auto* factory = registry_->getFactory("GhostConv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64, 1, 1]");
    auto ctx = createContext(32);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 32, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// FocusFactory Tests
// ============================================================================

TEST_F(ConvolutionFactoryTest, Focus_SupportedTypes) {
    auto* factory = registry_->getFactory("Focus");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "Focus");
}

TEST_F(ConvolutionFactoryTest, Focus_DefaultArgs) {
    auto* factory = registry_->getFactory("Focus");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(3);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(ConvolutionFactoryTest, Focus_CustomArgs) {
    auto* factory = registry_->getFactory("Focus");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 3]");
    auto ctx = createContext(3);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(ConvolutionFactoryTest, Focus_ForwardPass) {
    auto* factory = registry_->getFactory("Focus");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64]");
    auto ctx = createContext(3);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 3, 64, 64 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    // Focus reduces spatial dimensions by 2
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

// ============================================================================
// ConvTransposeFactory Tests
// ============================================================================

TEST_F(ConvolutionFactoryTest, ConvTranspose_SupportedTypes) {
    auto* factory = registry_->getFactory("ConvTranspose");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 2);
    EXPECT_TRUE(std::find(types.begin(), types.end(), "ConvTranspose") != types.end());
    EXPECT_TRUE(std::find(types.begin(), types.end(), "nn.ConvTranspose2d") != types.end());
}

TEST_F(ConvolutionFactoryTest, ConvTranspose_DefaultArgs) {
    auto* factory = registry_->getFactory("ConvTranspose");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);  // Default outCh
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(ConvolutionFactoryTest, ConvTranspose_CustomArgs) {
    auto* factory = registry_->getFactory("ConvTranspose");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 4, 2]");  // outCh=128, k=4, s=2
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(ConvolutionFactoryTest, ConvTranspose_Alias) {
    auto* factory1 = registry_->getFactory("ConvTranspose");
    auto* factory2 = registry_->getFactory("nn.ConvTranspose2d");

    EXPECT_NE(factory1, nullptr);
    EXPECT_NE(factory2, nullptr);
    EXPECT_EQ(factory1, factory2);
}

TEST_F(ConvolutionFactoryTest, ConvTranspose_ForwardPass) {
    auto* factory = registry_->getFactory("ConvTranspose");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64, 2, 2]");  // k=2, s=2 for 2x upsampling
    auto ctx = createContext(32);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 32, 16, 16 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);  // H * 2
    EXPECT_EQ(output.size(3), 32);  // W * 2
}

// ============================================================================
// NaiveConvFactory Tests
// ============================================================================

TEST_F(ConvolutionFactoryTest, NaiveConv_SupportedTypes) {
    auto* factory = registry_->getFactory("NaiveConv");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "NaiveConv");
}

TEST_F(ConvolutionFactoryTest, NaiveConv_DefaultArgs) {
    auto* factory = registry_->getFactory("NaiveConv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(32);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(ConvolutionFactoryTest, NaiveConv_CustomArgs) {
    auto* factory = registry_->getFactory("NaiveConv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 3, 1]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(ConvolutionFactoryTest, NaiveConv_WithPadding) {
    auto* factory = registry_->getFactory("NaiveConv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64, 3, 1, 1]");  // outCh, k, s, padding
    auto ctx = createContext(32);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);
}

TEST_F(ConvolutionFactoryTest, NaiveConv_ForwardPass) {
    auto* factory = registry_->getFactory("NaiveConv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64, 3, 1, 1]");  // k=3, s=1, p=1 for same size
    auto ctx = createContext(32);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 32, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);  // Same H with padding
    EXPECT_EQ(output.size(3), 32);  // Same W with padding
}

TEST_F(ConvolutionFactoryTest, NaiveConv_WithGroups) {
    auto* factory = registry_->getFactory("NaiveConv");
    ASSERT_NE(factory, nullptr);

    // [outCh, k, s, p, groups, dilation]
    YAML::Node args = YAML::Load("[64, 3, 1, 1, 1, 1]");
    auto ctx = createContext(32);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);
}

// ============================================================================
// Width Scaling Tests
// ============================================================================

TEST_F(ConvolutionFactoryTest, AllFactories_WidthScaling) {
    std::vector<std::string> factoryNames = {
        "Conv", "DWConv", "RepConv", "GhostConv", "Focus", "ConvTranspose", "NaiveConv"
    };

    YAML::Node args = YAML::Load("[64]");
    auto ctx = createContext(32, 0.5);  // widthMult=0.5

    for (const auto& name : factoryNames) {
        auto* factory = registry_->getFactory(name);
        ASSERT_NE(factory, nullptr) << "Factory not found: " << name;

        auto result = factory->create(args, ctx);
        EXPECT_EQ(result.outputChannels, 32)
            << "Width scaling failed for " << name << ": expected 32, got " << result.outputChannels;
    }
}

TEST_F(ConvolutionFactoryTest, AllFactories_WidthScaling_2x) {
    std::vector<std::string> factoryNames = {
        "Conv", "DWConv", "RepConv", "GhostConv", "Focus", "ConvTranspose", "NaiveConv"
    };

    YAML::Node args = YAML::Load("[64]");
    auto ctx = createContext(32, 2.0);  // widthMult=2.0

    for (const auto& name : factoryNames) {
        auto* factory = registry_->getFactory(name);
        ASSERT_NE(factory, nullptr) << "Factory not found: " << name;

        auto result = factory->create(args, ctx);
        EXPECT_EQ(result.outputChannels, 128)
            << "Width scaling failed for " << name << ": expected 128, got " << result.outputChannels;
    }
}

// ============================================================================
// Batch Size Tests
// ============================================================================

TEST_F(ConvolutionFactoryTest, Conv_BatchSize) {
    auto* factory = registry_->getFactory("Conv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64, 3, 1]");
    auto ctx = createContext(32);

    auto result = factory->create(args, ctx);

    // Test with batch size > 1
    auto input = torch::randn({ 4, 32, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 4);  // Batch preserved
    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ConvolutionFactoryTest, Conv_MinimalInput) {
    auto* factory = registry_->getFactory("Conv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[16, 1, 1]");
    auto ctx = createContext(8);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 8, 4, 4 });  // Small spatial size
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(2), 4);
    EXPECT_EQ(output.size(3), 4);
}

TEST_F(ConvolutionFactoryTest, Conv_LargeChannels) {
    auto* factory = registry_->getFactory("Conv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[512]");
    auto ctx = createContext(256);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 512);
}
