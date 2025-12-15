/**
 * @file BlockFactoryTest.cpp
 * @brief Unit tests for Block Factory classes
 *
 * Tests cover:
 * - Bottleneck, C2f, C2, C3, C3x, C3k2, C2fPSA, C2PSA, RepC3
 * - C3Ghost, BottleneckCSP, SPP, SPPF
 * - ResNetBlock, ResNetLayer, DBlock, CNXBlock, TransformerBlock
 *
 * @author WheelDL Team
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include "Model/Builder/Factory/BlockFactory.h"
#include "Model/Builder/Factory/ModuleFactoryRegistry.h"

using namespace WheelDL::Model::Builder;

// ============================================================================
// Test Fixture
// ============================================================================

class BlockFactoryTest : public ::testing::Test {
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
// Bottleneck Factory Tests
// ============================================================================

TEST_F(BlockFactoryTest, Bottleneck_SupportedTypes) {
    auto* factory = registry_->getFactory("Bottleneck");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "Bottleneck");
}

TEST_F(BlockFactoryTest, Bottleneck_DefaultArgs) {
    auto* factory = registry_->getFactory("Bottleneck");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 256);  // Default
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(BlockFactoryTest, Bottleneck_CustomArgs) {
    auto* factory = registry_->getFactory("Bottleneck");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, false, 1, 0.5]");  // outCh, shortcut, g, e
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, Bottleneck_ForwardPass) {
    auto* factory = registry_->getFactory("Bottleneck");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// C2f Factory Tests
// ============================================================================

TEST_F(BlockFactoryTest, C2f_SupportedTypes) {
    auto* factory = registry_->getFactory("C2f");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types[0], "C2f");
}

TEST_F(BlockFactoryTest, C2f_DefaultArgs) {
    auto* factory = registry_->getFactory("C2f");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 256);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(BlockFactoryTest, C2f_WithRepeats) {
    auto* factory = registry_->getFactory("C2f");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128]");
    auto ctx = createContext(64, 1.0, 1.0, 3);  // repeats=3

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, C2f_ForwardPass) {
    auto* factory = registry_->getFactory("C2f");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64]");
    auto ctx = createContext(64, 1.0, 1.0, 1);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// C2 Factory Tests
// ============================================================================

TEST_F(BlockFactoryTest, C2_DefaultArgs) {
    auto* factory = registry_->getFactory("C2");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 256);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(BlockFactoryTest, C2_ForwardPass) {
    auto* factory = registry_->getFactory("C2");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64]");
    auto ctx = createContext(64, 1.0, 1.0, 1);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// C3 Factory Tests
// ============================================================================

TEST_F(BlockFactoryTest, C3_DefaultArgs) {
    auto* factory = registry_->getFactory("C3");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 256);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(BlockFactoryTest, C3_ForwardPass) {
    auto* factory = registry_->getFactory("C3");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64]");
    auto ctx = createContext(64, 1.0, 1.0, 1);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// C3x Factory Tests
// ============================================================================

TEST_F(BlockFactoryTest, C3x_DefaultArgs) {
    auto* factory = registry_->getFactory("C3x");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 256);
    EXPECT_TRUE(result.module.ptr());
}

// ============================================================================
// C3k2 Factory Tests (Exception Tests)
// ============================================================================

TEST_F(BlockFactoryTest, C3k2_ValidArgs) {
    auto* factory = registry_->getFactory("C3k2");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, C3k2_EmptyArgs_ThrowsException) {
    auto* factory = registry_->getFactory("C3k2");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    EXPECT_THROW(factory->create(args, ctx), std::runtime_error);
}

TEST_F(BlockFactoryTest, C3k2_WithC3kFlag) {
    auto* factory = registry_->getFactory("C3k2");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 0]");  // c3k=false
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

// ============================================================================
// C2fPSA Factory Tests (Exception Tests)
// ============================================================================

TEST_F(BlockFactoryTest, C2fPSA_ValidArgs) {
    auto* factory = registry_->getFactory("C2fPSA");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, C2fPSA_EmptyArgs_ThrowsException) {
    auto* factory = registry_->getFactory("C2fPSA");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    EXPECT_THROW(factory->create(args, ctx), std::runtime_error);
}

// ============================================================================
// C2PSA Factory Tests (Exception Tests)
// ============================================================================

TEST_F(BlockFactoryTest, C2PSA_ValidArgs) {
    auto* factory = registry_->getFactory("C2PSA");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, C2PSA_EmptyArgs_ThrowsException) {
    auto* factory = registry_->getFactory("C2PSA");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    EXPECT_THROW(factory->create(args, ctx), std::runtime_error);
}

// ============================================================================
// RepC3 Factory Tests (Exception Tests)
// ============================================================================

TEST_F(BlockFactoryTest, RepC3_ValidArgs) {
    auto* factory = registry_->getFactory("RepC3");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, RepC3_EmptyArgs_ThrowsException) {
    auto* factory = registry_->getFactory("RepC3");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    EXPECT_THROW(factory->create(args, ctx), std::runtime_error);
}

// ============================================================================
// C3Ghost Factory Tests
// ============================================================================

TEST_F(BlockFactoryTest, C3Ghost_DefaultArgs) {
    auto* factory = registry_->getFactory("C3Ghost");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 256);
    EXPECT_TRUE(result.module.ptr());
}

// ============================================================================
// BottleneckCSP Factory Tests (Exception Tests)
// ============================================================================

TEST_F(BlockFactoryTest, BottleneckCSP_ValidArgs) {
    auto* factory = registry_->getFactory("BottleneckCSP");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, BottleneckCSP_EmptyArgs_ThrowsException) {
    auto* factory = registry_->getFactory("BottleneckCSP");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    EXPECT_THROW(factory->create(args, ctx), std::runtime_error);
}

// ============================================================================
// SPP Factory Tests (Exception Tests)
// ============================================================================

TEST_F(BlockFactoryTest, SPP_ValidArgs) {
    auto* factory = registry_->getFactory("SPP");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 5]");  // outCh, kernel_size
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, SPP_InsufficientArgs_ThrowsException) {
    auto* factory = registry_->getFactory("SPP");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128]");  // Missing kernel_size
    auto ctx = createContext(64);

    EXPECT_THROW(factory->create(args, ctx), std::runtime_error);
}

TEST_F(BlockFactoryTest, SPP_EmptyArgs_ThrowsException) {
    auto* factory = registry_->getFactory("SPP");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    EXPECT_THROW(factory->create(args, ctx), std::runtime_error);
}

// ============================================================================
// SPPF Factory Tests (Exception Tests)
// ============================================================================

TEST_F(BlockFactoryTest, SPPF_ValidArgs) {
    auto* factory = registry_->getFactory("SPPF");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 5]");  // outCh, kernel_size
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, SPPF_InsufficientArgs_ThrowsException) {
    auto* factory = registry_->getFactory("SPPF");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128]");  // Missing kernel_size
    auto ctx = createContext(64);

    EXPECT_THROW(factory->create(args, ctx), std::runtime_error);
}

TEST_F(BlockFactoryTest, SPPF_ForwardPass) {
    auto* factory = registry_->getFactory("SPPF");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64, 5]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// ResNetBlock Factory Tests
// ============================================================================

TEST_F(BlockFactoryTest, ResNetBlock_DefaultArgs) {
    auto* factory = registry_->getFactory("ResNetBlock");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 256);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(BlockFactoryTest, ResNetBlock_CustomArgs) {
    auto* factory = registry_->getFactory("ResNetBlock");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 2, 4.0]");  // outCh, stride, expansion
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

// ============================================================================
// ResNetLayer Factory Tests
// ============================================================================

TEST_F(BlockFactoryTest, ResNetLayer_DefaultArgs) {
    auto* factory = registry_->getFactory("ResNetLayer");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_TRUE(result.module.ptr());
}

TEST_F(BlockFactoryTest, ResNetLayer_IsFirst) {
    auto* factory = registry_->getFactory("ResNetLayer");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64, 1, true, 4.0]");  // isFirst=true
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    // isFirst=true: outputChannels = scaled.first
    EXPECT_EQ(result.outputChannels, 64);
}

TEST_F(BlockFactoryTest, ResNetLayer_NotFirst) {
    auto* factory = registry_->getFactory("ResNetLayer");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64, 1, false, 4.0]");  // isFirst=false
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    // isFirst=false: outputChannels = scaled.first * e = 64 * 4 = 256
    EXPECT_EQ(result.outputChannels, 256);
}

// ============================================================================
// DBlock Factory Tests
// ============================================================================

TEST_F(BlockFactoryTest, DBlock_SupportedTypes) {
    auto* factory = registry_->getFactory("DBlock");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 2);
    EXPECT_TRUE(std::find(types.begin(), types.end(), "DBlock") != types.end());
    EXPECT_TRUE(std::find(types.begin(), types.end(), "DenseBlock") != types.end());
}

TEST_F(BlockFactoryTest, DBlock_DefaultArgs) {
    auto* factory = registry_->getFactory("DBlock");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 256);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(BlockFactoryTest, DBlock_Alias) {
    auto* factory1 = registry_->getFactory("DBlock");
    auto* factory2 = registry_->getFactory("DenseBlock");

    EXPECT_NE(factory1, nullptr);
    EXPECT_NE(factory2, nullptr);
    EXPECT_EQ(factory1, factory2);
}

// ============================================================================
// CNXBlock Factory Tests
// ============================================================================

TEST_F(BlockFactoryTest, CNXBlock_SupportedTypes) {
    auto* factory = registry_->getFactory("CNXBlock");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 2);
    EXPECT_TRUE(std::find(types.begin(), types.end(), "CNXBlock") != types.end());
    EXPECT_TRUE(std::find(types.begin(), types.end(), "ConvNeXtBlock") != types.end());
}

TEST_F(BlockFactoryTest, CNXBlock_DefaultArgs) {
    auto* factory = registry_->getFactory("CNXBlock");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 256);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(BlockFactoryTest, CNXBlock_Alias) {
    auto* factory1 = registry_->getFactory("CNXBlock");
    auto* factory2 = registry_->getFactory("ConvNeXtBlock");

    EXPECT_NE(factory1, nullptr);
    EXPECT_NE(factory2, nullptr);
    EXPECT_EQ(factory1, factory2);
}

TEST_F(BlockFactoryTest, CNXBlock_CustomLayerScale) {
    auto* factory = registry_->getFactory("CNXBlock");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 0.001]");  // channels, layerScaleInit
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

// ============================================================================
// TransformerBlock Factory Tests
// ============================================================================

TEST_F(BlockFactoryTest, TransformerBlock_DefaultArgs) {
    auto* factory = registry_->getFactory("TransformerBlock");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 256);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(BlockFactoryTest, TransformerBlock_CustomArgs) {
    auto* factory = registry_->getFactory("TransformerBlock");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 4, 2]");  // outCh, numHeads, numLayers
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 128);
}

// ============================================================================
// Width Scaling Tests
// ============================================================================

TEST_F(BlockFactoryTest, AllFactories_WidthScaling) {
    std::vector<std::string> factoryNames = {
        "Bottleneck", "C2f", "C2", "C3", "C3x", "C3Ghost",
        "ResNetBlock", "TransformerBlock"
    };

    YAML::Node args = YAML::Load("[64]");
    auto ctx = createContext(32, 0.5);  // widthMult=0.5

    for (const auto& name : factoryNames) {
        auto* factory = registry_->getFactory(name);
        ASSERT_NE(factory, nullptr) << "Factory not found: " << name;

        auto result = factory->create(args, ctx);
        EXPECT_EQ(result.outputChannels, 32)
            << "Width scaling failed for " << name;
    }
}

// ============================================================================
// Depth Scaling Tests
// ============================================================================

TEST_F(BlockFactoryTest, C2f_DepthScaling) {
    auto* factory = registry_->getFactory("C2f");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128]");
    auto ctx = createContext(64, 1.0, 0.5, 4);  // depthMult=0.5, repeats=4

    auto result = factory->create(args, ctx);

    // With depth scaling, repeats should be scaled
    EXPECT_EQ(result.outputChannels, 128);
}

// ============================================================================
// Exception Message Tests
// ============================================================================

TEST_F(BlockFactoryTest, C3k2_ExceptionMessage) {
    auto* factory = registry_->getFactory("C3k2");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    try {
        factory->create(args, ctx);
        FAIL() << "Expected std::runtime_error";
    }
    catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("C3k2") != std::string::npos)
            << "Error message should mention C3k2: " << msg;
    }
}

TEST_F(BlockFactoryTest, SPP_ExceptionMessage) {
    auto* factory = registry_->getFactory("SPP");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[64]");
    auto ctx = createContext(64);

    try {
        factory->create(args, ctx);
        FAIL() << "Expected std::runtime_error";
    }
    catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("SPP") != std::string::npos)
            << "Error message should mention SPP: " << msg;
    }
}
