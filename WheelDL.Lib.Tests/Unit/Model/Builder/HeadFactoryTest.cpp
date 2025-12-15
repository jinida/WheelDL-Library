/**
 * @file HeadFactoryTest.cpp
 * @brief Unit tests for Head Factory classes
 *
 * Tests cover:
 * - DetectFactory, OBBFactory, ClassifyFactory, SegmentFactory, AnomalyFactory
 * - Default arguments, creation, exception handling
 *
 * @author WheelDL Team
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include "Model/Builder/Factory/HeadFactory.h"
#include "Model/Builder/Factory/ModuleFactoryRegistry.h"

using namespace WheelDL::Model::Builder;

// ============================================================================
// Test Fixture
// ============================================================================

class HeadFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = &ModuleFactoryRegistry::instance();
    }

    ModuleBuildContext createContext(int64_t inputCh = 64, int64_t numClasses = 80,
        std::vector<int64_t> headChannels = { 64, 128, 256 }) {
        ModuleBuildContext ctx;
        ctx.inputChannels = inputCh;
        ctx.repeats = 1;
        ctx.defaultAct = "SiLU";
        ctx.numClasses = numClasses;
        ctx.headChannels = headChannels;
        // applyScale lambda (identity for head tests)
        ctx.applyScale = [](int64_t channels, int64_t reps) {
            return std::make_pair(channels, reps);
        };
        return ctx;
    }

    ModuleFactoryRegistry* registry_;
};

// ============================================================================
// DetectFactory Tests
// ============================================================================

TEST_F(HeadFactoryTest, Detect_SupportedTypes) {
    auto* factory = registry_->getFactory("Detect");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "Detect");
}

TEST_F(HeadFactoryTest, Detect_Creation) {
    auto* factory = registry_->getFactory("Detect");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64, 80);

    auto result = factory->create(args, ctx);

    // outputChannels = numClasses + 64
    EXPECT_EQ(result.outputChannels, 80 + 64);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(HeadFactoryTest, Detect_DifferentNumClasses) {
    auto* factory = registry_->getFactory("Detect");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64, 20);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 20 + 64);
}

// ============================================================================
// OBBFactory Tests
// ============================================================================

TEST_F(HeadFactoryTest, OBB_SupportedTypes) {
    auto* factory = registry_->getFactory("OBB");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "OBB");
}

TEST_F(HeadFactoryTest, OBB_Creation) {
    auto* factory = registry_->getFactory("OBB");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64, 80);

    auto result = factory->create(args, ctx);

    // outputChannels = numClasses + 64 + 1 (angle)
    EXPECT_EQ(result.outputChannels, 80 + 64 + 1);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(HeadFactoryTest, OBB_DifferentNumClasses) {
    auto* factory = registry_->getFactory("OBB");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64, 15);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 15 + 64 + 1);
}

// ============================================================================
// ClassifyFactory Tests
// ============================================================================

TEST_F(HeadFactoryTest, Classify_SupportedTypes) {
    auto* factory = registry_->getFactory("Classify");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "Classify");
}

TEST_F(HeadFactoryTest, Classify_Creation) {
    auto* factory = registry_->getFactory("Classify");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(512, 1000);

    auto result = factory->create(args, ctx);

    // outputChannels = numClasses
    EXPECT_EQ(result.outputChannels, 1000);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(HeadFactoryTest, Classify_DifferentClasses) {
    auto* factory = registry_->getFactory("Classify");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(256, 10);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 10);
}

// ============================================================================
// SegmentFactory Tests
// ============================================================================

TEST_F(HeadFactoryTest, Segment_SupportedTypes) {
    auto* factory = registry_->getFactory("Segment");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "Segment");
}

TEST_F(HeadFactoryTest, Segment_Creation) {
    auto* factory = registry_->getFactory("Segment");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64, 80);

    auto result = factory->create(args, ctx);

    // outputChannels = numClasses
    EXPECT_EQ(result.outputChannels, 80);
    EXPECT_TRUE(result.module.ptr());
}

TEST_F(HeadFactoryTest, Segment_DifferentClasses) {
    auto* factory = registry_->getFactory("Segment");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(128, 21);

    auto result = factory->create(args, ctx);

    EXPECT_EQ(result.outputChannels, 21);
}

// ============================================================================
// AnomalyFactory Tests
// ============================================================================

TEST_F(HeadFactoryTest, Anomaly_SupportedTypes) {
    auto* factory = registry_->getFactory("Anomaly");
    ASSERT_NE(factory, nullptr);

    auto types = factory->supportedTypes();
    EXPECT_EQ(types.size(), 1);
    EXPECT_EQ(types[0], "Anomaly");
}

TEST_F(HeadFactoryTest, Anomaly_EfficientAD_Default) {
    auto* factory = registry_->getFactory("Anomaly");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");  // Default: EfficientAD
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_TRUE(result.module.ptr());
}

TEST_F(HeadFactoryTest, Anomaly_EfficientAD_Explicit) {
    auto* factory = registry_->getFactory("Anomaly");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[EfficientAD, 384, true]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_TRUE(result.module.ptr());
}

TEST_F(HeadFactoryTest, Anomaly_EfficientAD_CustomChannels) {
    auto* factory = registry_->getFactory("Anomaly");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[EfficientAD, 512, false]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);

    EXPECT_TRUE(result.module.ptr());
}

TEST_F(HeadFactoryTest, Anomaly_PatchCore) {
    auto* factory = registry_->getFactory("Anomaly");
    ASSERT_NE(factory, nullptr);

    // PatchCore requires pretrained model file (wide_resnet50_2.pt) which may not exist
    // Test that the factory recognizes PatchCore as valid type
    YAML::Node args = YAML::Load("[PatchCore, 9, 25600]");
    auto ctx = createContext(64);

    // If file doesn't exist, construction will throw
    try {
        auto result = factory->create(args, ctx);
        EXPECT_TRUE(result.module.ptr());
    }
    catch (const c10::Error&) {
        // Expected if pretrained model file doesn't exist
        SUCCEED();
    }
}

TEST_F(HeadFactoryTest, Anomaly_PatchCore_CustomParams) {
    auto* factory = registry_->getFactory("Anomaly");
    ASSERT_NE(factory, nullptr);

    // PatchCore requires pretrained model file (wide_resnet50_2.pt) which may not exist
    YAML::Node args = YAML::Load("[PatchCore, 5, 10000]");
    auto ctx = createContext(64);

    try {
        auto result = factory->create(args, ctx);
        EXPECT_TRUE(result.module.ptr());
    }
    catch (const c10::Error&) {
        // Expected if pretrained model file doesn't exist
        SUCCEED();
    }
}

TEST_F(HeadFactoryTest, Anomaly_SimpleNet_ThrowsException) {
    auto* factory = registry_->getFactory("Anomaly");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[SimpleNet]");
    auto ctx = createContext(64);

    EXPECT_THROW(factory->create(args, ctx), std::runtime_error);
}

TEST_F(HeadFactoryTest, Anomaly_UnknownType_ThrowsException) {
    auto* factory = registry_->getFactory("Anomaly");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[UnknownModel]");
    auto ctx = createContext(64);

    EXPECT_THROW(factory->create(args, ctx), std::runtime_error);
}

// ============================================================================
// Exception Message Tests
// ============================================================================

TEST_F(HeadFactoryTest, Anomaly_SimpleNet_ExceptionMessage) {
    auto* factory = registry_->getFactory("Anomaly");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[SimpleNet]");
    auto ctx = createContext(64);

    try {
        factory->create(args, ctx);
        FAIL() << "Expected std::runtime_error";
    }
    catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("SimpleNet") != std::string::npos)
            << "Error message should mention SimpleNet: " << msg;
    }
}

TEST_F(HeadFactoryTest, Anomaly_UnknownType_ExceptionMessage) {
    auto* factory = registry_->getFactory("Anomaly");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[CustomAnomaly]");
    auto ctx = createContext(64);

    try {
        factory->create(args, ctx);
        FAIL() << "Expected std::runtime_error";
    }
    catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("Unknown") != std::string::npos ||
                    msg.find("CustomAnomaly") != std::string::npos)
            << "Error message should mention unknown type: " << msg;
    }
}

// ============================================================================
// Head Channels Tests
// ============================================================================

TEST_F(HeadFactoryTest, Detect_DifferentHeadChannels) {
    auto* factory = registry_->getFactory("Detect");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64, 80, { 128, 256, 512 });

    auto result = factory->create(args, ctx);

    EXPECT_TRUE(result.module.ptr());
}

TEST_F(HeadFactoryTest, OBB_DifferentHeadChannels) {
    auto* factory = registry_->getFactory("OBB");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64, 80, { 128, 256, 512 });

    auto result = factory->create(args, ctx);

    EXPECT_TRUE(result.module.ptr());
}
