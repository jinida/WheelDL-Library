/**
 * @file ModuleFactoryRegistryTest.cpp
 * @brief Unit tests for ModuleFactoryRegistry singleton class
 *
 * Tests cover:
 * - Singleton pattern verification
 * - registerFactory() with valid/invalid inputs
 * - registerAlias() with valid/invalid inputs
 * - getFactory() for existing/non-existing types
 * - registerDefaultFactories() validation
 *
 * @author WheelDL Team
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "Model/Builder/Factory/ModuleFactoryRegistry.h"
#include "Model/Builder/Factory/ModuleFactory.h"

using namespace WheelDL::Model::Builder;

// ============================================================================
// Mock Factory for Testing
// ============================================================================

class MockModuleFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return { "MockType" };
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
        const ModuleBuildContext& ctx) override {
        ModuleBuildResult result;
        result.outputChannels = ctx.inputChannels;
        return result;
    }
};

// ============================================================================
// ModuleFactoryRegistry Test Fixture
// ============================================================================

class ModuleFactoryRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Get the singleton instance
        registry_ = &ModuleFactoryRegistry::instance();
    }

    void TearDown() override {}

    ModuleFactoryRegistry* registry_;
};

// ============================================================================
// Singleton Tests
// ============================================================================

TEST_F(ModuleFactoryRegistryTest, Instance_ReturnsSameInstance) {
    auto& instance1 = ModuleFactoryRegistry::instance();
    auto& instance2 = ModuleFactoryRegistry::instance();

    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(ModuleFactoryRegistryTest, Instance_NotNull) {
    EXPECT_NE(registry_, nullptr);
}

// ============================================================================
// registerFactory Tests
// ============================================================================

TEST_F(ModuleFactoryRegistryTest, RegisterFactory_NullFactory_ThrowsInvalidArgument) {
    EXPECT_THROW(
        registry_->registerFactory("NullTest", nullptr),
        std::invalid_argument
    );
}

TEST_F(ModuleFactoryRegistryTest, RegisterFactory_EmptyType_ThrowsInvalidArgument) {
    auto factory = std::make_shared<MockModuleFactory>();

    EXPECT_THROW(
        registry_->registerFactory("", factory),
        std::invalid_argument
    );
}

TEST_F(ModuleFactoryRegistryTest, RegisterFactory_DuplicateType_ThrowsRuntimeError) {
    // "Conv" is already registered by registerDefaultFactories()
    auto factory = std::make_shared<MockModuleFactory>();

    EXPECT_THROW(
        registry_->registerFactory("Conv", factory),
        std::runtime_error
    );
}

TEST_F(ModuleFactoryRegistryTest, RegisterFactory_ValidFactory_Succeeds) {
    auto factory = std::make_shared<MockModuleFactory>();
    std::string uniqueType = "UniqueTestType_" + std::to_string(reinterpret_cast<uintptr_t>(this));

    // Should not throw
    EXPECT_NO_THROW(registry_->registerFactory(uniqueType, factory));

    // Should be retrievable
    auto* retrieved = registry_->getFactory(uniqueType);
    EXPECT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved, factory.get());
}

TEST_F(ModuleFactoryRegistryTest, RegisterFactory_NullFactory_ExceptionMessage) {
    try {
        registry_->registerFactory("TestNull", nullptr);
        FAIL() << "Expected std::invalid_argument";
    }
    catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("null") != std::string::npos ||
            msg.find("Null") != std::string::npos)
            << "Error message should mention null: " << msg;
    }
}

TEST_F(ModuleFactoryRegistryTest, RegisterFactory_EmptyType_ExceptionMessage) {
    auto factory = std::make_shared<MockModuleFactory>();

    try {
        registry_->registerFactory("", factory);
        FAIL() << "Expected std::invalid_argument";
    }
    catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("empty") != std::string::npos)
            << "Error message should mention empty: " << msg;
    }
}

// ============================================================================
// registerAlias Tests
// ============================================================================

TEST_F(ModuleFactoryRegistryTest, RegisterAlias_NonExistentType_ThrowsRuntimeError) {
    EXPECT_THROW(
        registry_->registerAlias("AliasForNonExistent", "NonExistentType"),
        std::runtime_error
    );
}

TEST_F(ModuleFactoryRegistryTest, RegisterAlias_DuplicateAlias_ThrowsRuntimeError) {
    // "nn.Upsample" is already registered as an alias for "Upsample"
    EXPECT_THROW(
        registry_->registerAlias("nn.Upsample", "Concat"),
        std::runtime_error
    );
}

TEST_F(ModuleFactoryRegistryTest, RegisterAlias_ValidAlias_Succeeds) {
    // First register a unique factory
    auto factory = std::make_shared<MockModuleFactory>();
    std::string uniqueType = "AliasTestType_" + std::to_string(reinterpret_cast<uintptr_t>(this));
    std::string uniqueAlias = "AliasTestAlias_" + std::to_string(reinterpret_cast<uintptr_t>(this));

    registry_->registerFactory(uniqueType, factory);

    // Register alias
    EXPECT_NO_THROW(registry_->registerAlias(uniqueAlias, uniqueType));

    // Both should return the same factory
    auto* original = registry_->getFactory(uniqueType);
    auto* aliased = registry_->getFactory(uniqueAlias);
    EXPECT_EQ(original, aliased);
}

TEST_F(ModuleFactoryRegistryTest, RegisterAlias_NonExistent_ExceptionMessage) {
    try {
        registry_->registerAlias("TestAlias", "NonExistentFactory");
        FAIL() << "Expected std::runtime_error";
    }
    catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_TRUE(msg.find("not found") != std::string::npos)
            << "Error message should mention 'not found': " << msg;
    }
}

// ============================================================================
// getFactory Tests
// ============================================================================

TEST_F(ModuleFactoryRegistryTest, GetFactory_ExistingType_ReturnsFactory) {
    auto* factory = registry_->getFactory("Conv");

    EXPECT_NE(factory, nullptr);
}

TEST_F(ModuleFactoryRegistryTest, GetFactory_NonExistentType_ReturnsNullptr) {
    auto* factory = registry_->getFactory("NonExistentModuleType");

    EXPECT_EQ(factory, nullptr);
}

TEST_F(ModuleFactoryRegistryTest, GetFactory_EmptyString_ReturnsNullptr) {
    auto* factory = registry_->getFactory("");

    EXPECT_EQ(factory, nullptr);
}

TEST_F(ModuleFactoryRegistryTest, GetFactory_AliasType_ReturnsSameFactory) {
    auto* upsample = registry_->getFactory("Upsample");
    auto* nnUpsample = registry_->getFactory("nn.Upsample");

    EXPECT_NE(upsample, nullptr);
    EXPECT_NE(nnUpsample, nullptr);
    EXPECT_EQ(upsample, nnUpsample);
}

// ============================================================================
// registerDefaultFactories Tests
// ============================================================================

TEST_F(ModuleFactoryRegistryTest, DefaultFactories_ConvolutionFactoriesRegistered) {
    EXPECT_NE(registry_->getFactory("Conv"), nullptr);
    EXPECT_NE(registry_->getFactory("NaiveConv"), nullptr);
    EXPECT_NE(registry_->getFactory("DWConv"), nullptr);
    EXPECT_NE(registry_->getFactory("RepConv"), nullptr);
    EXPECT_NE(registry_->getFactory("GhostConv"), nullptr);
    EXPECT_NE(registry_->getFactory("Focus"), nullptr);
    EXPECT_NE(registry_->getFactory("ConvTranspose"), nullptr);
}

TEST_F(ModuleFactoryRegistryTest, DefaultFactories_BlockFactoriesRegistered) {
    EXPECT_NE(registry_->getFactory("Bottleneck"), nullptr);
    EXPECT_NE(registry_->getFactory("C2f"), nullptr);
    EXPECT_NE(registry_->getFactory("C2"), nullptr);
    EXPECT_NE(registry_->getFactory("C3"), nullptr);
    EXPECT_NE(registry_->getFactory("C3x"), nullptr);
    EXPECT_NE(registry_->getFactory("C3k2"), nullptr);
    EXPECT_NE(registry_->getFactory("C2fPSA"), nullptr);
    EXPECT_NE(registry_->getFactory("C2PSA"), nullptr);
    EXPECT_NE(registry_->getFactory("RepC3"), nullptr);
    EXPECT_NE(registry_->getFactory("C3Ghost"), nullptr);
    EXPECT_NE(registry_->getFactory("BottleneckCSP"), nullptr);
    EXPECT_NE(registry_->getFactory("SPP"), nullptr);
    EXPECT_NE(registry_->getFactory("SPPF"), nullptr);
    EXPECT_NE(registry_->getFactory("ResNetLayer"), nullptr);
    EXPECT_NE(registry_->getFactory("ResNetBlock"), nullptr);
    EXPECT_NE(registry_->getFactory("DBlock"), nullptr);
    EXPECT_NE(registry_->getFactory("CNXBlock"), nullptr);
    EXPECT_NE(registry_->getFactory("TransformerBlock"), nullptr);
}

TEST_F(ModuleFactoryRegistryTest, DefaultFactories_UtilityFactoriesRegistered) {
    EXPECT_NE(registry_->getFactory("Upsample"), nullptr);
    EXPECT_NE(registry_->getFactory("Concat"), nullptr);
    EXPECT_NE(registry_->getFactory("MaxPool2d"), nullptr);
    EXPECT_NE(registry_->getFactory("AvgPool2d"), nullptr);
    EXPECT_NE(registry_->getFactory("GlobalAvgPool"), nullptr);
}

TEST_F(ModuleFactoryRegistryTest, DefaultFactories_AttentionFactoriesRegistered) {
    EXPECT_NE(registry_->getFactory("PSA"), nullptr);
    EXPECT_NE(registry_->getFactory("CBAM"), nullptr);
    EXPECT_NE(registry_->getFactory("Attention"), nullptr);
}

TEST_F(ModuleFactoryRegistryTest, DefaultFactories_TransformerFactoriesRegistered) {
    EXPECT_NE(registry_->getFactory("LayerNorm2d"), nullptr);
    EXPECT_NE(registry_->getFactory("MLPBlock"), nullptr);
    EXPECT_NE(registry_->getFactory("MLP"), nullptr);
    EXPECT_NE(registry_->getFactory("TransformerLayer"), nullptr);
    EXPECT_NE(registry_->getFactory("TransformerEncoderLayer"), nullptr);
    EXPECT_NE(registry_->getFactory("AIFI"), nullptr);
    EXPECT_NE(registry_->getFactory("MSDeformAttn"), nullptr);
    EXPECT_NE(registry_->getFactory("DeformableTransformerDecoderLayer"), nullptr);
    EXPECT_NE(registry_->getFactory("DeformableTransformerDecoder"), nullptr);
}

TEST_F(ModuleFactoryRegistryTest, DefaultFactories_HeadFactoriesRegistered) {
    EXPECT_NE(registry_->getFactory("Detect"), nullptr);
    EXPECT_NE(registry_->getFactory("OBB"), nullptr);
    EXPECT_NE(registry_->getFactory("Classify"), nullptr);
    EXPECT_NE(registry_->getFactory("Segment"), nullptr);
    EXPECT_NE(registry_->getFactory("Anomaly"), nullptr);
}

TEST_F(ModuleFactoryRegistryTest, DefaultFactories_AliasesRegistered) {
    // Convolution aliases
    EXPECT_NE(registry_->getFactory("nn.ConvTranspose2d"), nullptr);
    EXPECT_EQ(registry_->getFactory("nn.ConvTranspose2d"), registry_->getFactory("ConvTranspose"));

    // Block aliases
    EXPECT_NE(registry_->getFactory("DenseBlock"), nullptr);
    EXPECT_EQ(registry_->getFactory("DenseBlock"), registry_->getFactory("DBlock"));
    EXPECT_NE(registry_->getFactory("ConvNeXtBlock"), nullptr);
    EXPECT_EQ(registry_->getFactory("ConvNeXtBlock"), registry_->getFactory("CNXBlock"));

    // Utility aliases
    EXPECT_NE(registry_->getFactory("nn.Upsample"), nullptr);
    EXPECT_EQ(registry_->getFactory("nn.Upsample"), registry_->getFactory("Upsample"));
    EXPECT_NE(registry_->getFactory("nn.MaxPool2d"), nullptr);
    EXPECT_EQ(registry_->getFactory("nn.MaxPool2d"), registry_->getFactory("MaxPool2d"));
    EXPECT_NE(registry_->getFactory("nn.AvgPool2d"), nullptr);
    EXPECT_EQ(registry_->getFactory("nn.AvgPool2d"), registry_->getFactory("AvgPool2d"));
    EXPECT_NE(registry_->getFactory("nn.AdaptiveAvgPool2d"), nullptr);
    EXPECT_EQ(registry_->getFactory("nn.AdaptiveAvgPool2d"), registry_->getFactory("GlobalAvgPool"));

    // Transformer aliases
    EXPECT_NE(registry_->getFactory("nn.LayerNorm"), nullptr);
    EXPECT_EQ(registry_->getFactory("nn.LayerNorm"), registry_->getFactory("LayerNorm2d"));
}

// ============================================================================
// Factory supportedTypes() Tests
// ============================================================================

TEST_F(ModuleFactoryRegistryTest, Factory_SupportedTypes_NotEmpty) {
    auto* convFactory = registry_->getFactory("Conv");
    ASSERT_NE(convFactory, nullptr);

    auto types = convFactory->supportedTypes();
    EXPECT_FALSE(types.empty());
    EXPECT_TRUE(std::find(types.begin(), types.end(), "Conv") != types.end());
}

TEST_F(ModuleFactoryRegistryTest, Factory_SupportedTypes_ContainsCorrectType) {
    auto* c2fFactory = registry_->getFactory("C2f");
    ASSERT_NE(c2fFactory, nullptr);

    auto types = c2fFactory->supportedTypes();
    EXPECT_TRUE(std::find(types.begin(), types.end(), "C2f") != types.end());
}
