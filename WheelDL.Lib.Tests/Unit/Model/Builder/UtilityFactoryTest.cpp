/**
 * @file UtilityFactoryTest.cpp
 * @brief Unit tests for Utility Factory classes
 *
 * Tests cover:
 * - UpsampleFactory, ConcatFactory, MaxPool2dFactory
 * - AvgPool2dFactory, GlobalAvgPoolFactory
 * - Default/Custom arguments, forward pass, exception handling
 *
 * @author WheelDL Team
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include "Model/Builder/Factory/UtilityFactory.h"
#include "Model/Builder/Factory/ModuleFactoryRegistry.h"

using namespace WheelDL::Model::Builder;

// ============================================================================
// Test Fixture
// ============================================================================

class UtilityFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = &ModuleFactoryRegistry::instance();
    }

    ModuleBuildContext createContext(int64_t inputCh = 64) {
        ModuleBuildContext ctx;
        ctx.inputChannels = inputCh;
        ctx.repeats = 1;
        ctx.defaultAct = "SiLU";
        ctx.numClasses = 80;
        ctx.headChannels = {};
        ctx.applyScale = [](int64_t channels, int64_t reps) {
            return std::make_pair(channels, reps);
        };
        return ctx;
    }

    ModuleFactoryRegistry* registry_;
};

// ============================================================================
// UpsampleFactory Tests
// ============================================================================

TEST_F(UtilityFactoryTest, Upsample_SupportedTypes) {
    auto* factory = registry_->getFactory("Upsample");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 2);
    EXPECT_TRUE(std::find(types.begin(), types.end(), "Upsample") != types.end());
    EXPECT_TRUE(std::find(types.begin(), types.end(), "nn.Upsample") != types.end());
}

TEST_F(UtilityFactoryTest, Upsample_DefaultArgs) {
    auto* factory = registry_->getFactory("Upsample");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);  // Preserves channels
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(UtilityFactoryTest, Upsample_CustomScale) {
    auto* factory = registry_->getFactory("Upsample");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[4.0]");  // scale=4.0
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(UtilityFactoryTest, Upsample_BilinearMode) {
    auto* factory = registry_->getFactory("Upsample");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[2.0, bilinear]");  // scale=2.0, mode=bilinear
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_TRUE(result.module.ptr());
}

TEST_F(UtilityFactoryTest, Upsample_ForwardPass) {
    auto* factory = registry_->getFactory("Upsample");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[2.0]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 16, 16 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);  // 16 * 2
    EXPECT_EQ(output.size(3), 32);  // 16 * 2
}

TEST_F(UtilityFactoryTest, Upsample_Alias) {
    auto* factory1 = registry_->getFactory("Upsample");
    auto* factory2 = registry_->getFactory("nn.Upsample");

    EXPECT_NE(factory1, nullptr);
    EXPECT_NE(factory2, nullptr);
    EXPECT_EQ(factory1, factory2);
}

// ============================================================================
// ConcatFactory Tests
// ============================================================================

TEST_F(UtilityFactoryTest, Concat_SupportedTypes) {
    auto* factory = registry_->getFactory("Concat");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "Concat");
}

TEST_F(UtilityFactoryTest, Concat_DefaultDim) {
    auto* factory = registry_->getFactory("Concat");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(UtilityFactoryTest, Concat_CustomDim) {
    auto* factory = registry_->getFactory("Concat");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[2]");  // dim=2 (height)
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_TRUE(result.module.ptr());
}

TEST_F(UtilityFactoryTest, Concat_NegativeDim) {
    auto* factory = registry_->getFactory("Concat");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[-1]");  // Last dimension
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_TRUE(result.module.ptr());
}

TEST_F(UtilityFactoryTest, Concat_InvalidDim_TooLarge) {
    auto* factory = registry_->getFactory("Concat");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[11]");  // > 10
    auto ctx = createContext(64);

    EXPECT_THROW(factory->create(args, ctx), std::runtime_error);
}

TEST_F(UtilityFactoryTest, Concat_InvalidDim_TooSmall) {
    auto* factory = registry_->getFactory("Concat");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[-5]");  // < -4
    auto ctx = createContext(64);

    EXPECT_THROW(factory->create(args, ctx), std::runtime_error);
}

TEST_F(UtilityFactoryTest, Concat_ExceptionMessage) {
    auto* factory = registry_->getFactory("Concat");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[100]");
    auto ctx = createContext(64);

    try {
        factory->create(args, ctx);
        FAIL() << "Expected std::runtime_error";
    }
    catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("dimension") != std::string::npos ||
                    msg.find("Concat") != std::string::npos)
            << "Error message should mention dimension: " << msg;
    }
}

// ============================================================================
// MaxPool2dFactory Tests
// ============================================================================

TEST_F(UtilityFactoryTest, MaxPool2d_SupportedTypes) {
    auto* factory = registry_->getFactory("MaxPool2d");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 2);
    EXPECT_TRUE(std::find(types.begin(), types.end(), "MaxPool2d") != types.end());
    EXPECT_TRUE(std::find(types.begin(), types.end(), "nn.MaxPool2d") != types.end());
}

TEST_F(UtilityFactoryTest, MaxPool2d_DefaultArgs) {
    auto* factory = registry_->getFactory("MaxPool2d");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);  // Preserves channels
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(UtilityFactoryTest, MaxPool2d_CustomArgs) {
    auto* factory = registry_->getFactory("MaxPool2d");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[3, 2, 1]");  // k=3, s=2, p=1
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_TRUE(result.module.ptr());
}

TEST_F(UtilityFactoryTest, MaxPool2d_ForwardPass) {
    auto* factory = registry_->getFactory("MaxPool2d");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[2, 2]");  // k=2, s=2
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 16);  // 32 / 2
    EXPECT_EQ(output.size(3), 16);
}

TEST_F(UtilityFactoryTest, MaxPool2d_Alias) {
    auto* factory1 = registry_->getFactory("MaxPool2d");
    auto* factory2 = registry_->getFactory("nn.MaxPool2d");

    EXPECT_NE(factory1, nullptr);
    EXPECT_NE(factory2, nullptr);
    EXPECT_EQ(factory1, factory2);
}

// ============================================================================
// AvgPool2dFactory Tests
// ============================================================================

TEST_F(UtilityFactoryTest, AvgPool2d_SupportedTypes) {
    auto* factory = registry_->getFactory("AvgPool2d");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 2);
    EXPECT_TRUE(std::find(types.begin(), types.end(), "AvgPool2d") != types.end());
    EXPECT_TRUE(std::find(types.begin(), types.end(), "nn.AvgPool2d") != types.end());
}

TEST_F(UtilityFactoryTest, AvgPool2d_DefaultArgs) {
    auto* factory = registry_->getFactory("AvgPool2d");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(UtilityFactoryTest, AvgPool2d_CustomArgs) {
    auto* factory = registry_->getFactory("AvgPool2d");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[3, 2, 1]");  // k=3, s=2, p=1
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_TRUE(result.module.ptr());
}

TEST_F(UtilityFactoryTest, AvgPool2d_ForwardPass) {
    auto* factory = registry_->getFactory("AvgPool2d");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[2, 2]");  // k=2, s=2
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 16);
    EXPECT_EQ(output.size(3), 16);
}

TEST_F(UtilityFactoryTest, AvgPool2d_Alias) {
    auto* factory1 = registry_->getFactory("AvgPool2d");
    auto* factory2 = registry_->getFactory("nn.AvgPool2d");

    EXPECT_NE(factory1, nullptr);
    EXPECT_NE(factory2, nullptr);
    EXPECT_EQ(factory1, factory2);
}

// ============================================================================
// GlobalAvgPoolFactory Tests
// ============================================================================

TEST_F(UtilityFactoryTest, GlobalAvgPool_SupportedTypes) {
    auto* factory = registry_->getFactory("GlobalAvgPool");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 2);
    EXPECT_TRUE(std::find(types.begin(), types.end(), "GlobalAvgPool") != types.end());
    EXPECT_TRUE(std::find(types.begin(), types.end(), "nn.AdaptiveAvgPool2d") != types.end());
}

TEST_F(UtilityFactoryTest, GlobalAvgPool_NoArgs) {
    auto* factory = registry_->getFactory("GlobalAvgPool");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 64);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(UtilityFactoryTest, GlobalAvgPool_ForwardPass) {
    auto* factory = registry_->getFactory("GlobalAvgPool");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 1, 64, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 1);  // Global pool -> 1x1
    EXPECT_EQ(output.size(3), 1);
}

TEST_F(UtilityFactoryTest, GlobalAvgPool_Alias) {
    auto* factory1 = registry_->getFactory("GlobalAvgPool");
    auto* factory2 = registry_->getFactory("nn.AdaptiveAvgPool2d");

    EXPECT_NE(factory1, nullptr);
    EXPECT_NE(factory2, nullptr);
    EXPECT_EQ(factory1, factory2);
}

// ============================================================================
// Batch Size Tests
// ============================================================================

TEST_F(UtilityFactoryTest, Upsample_BatchSize) {
    auto* factory = registry_->getFactory("Upsample");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[2.0]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 4, 64, 16, 16 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 4);  // Batch preserved
}

TEST_F(UtilityFactoryTest, MaxPool2d_BatchSize) {
    auto* factory = registry_->getFactory("MaxPool2d");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[2, 2]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    auto input = torch::randn({ 4, 64, 32, 32 });
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 4);  // Batch preserved
}
