/**
 * @file ModelBuilderTest.cpp
 * @brief Unit tests for ModelBuilder class
 *
 * Tests cover:
 * - Constructor, setters (method chaining)
 * - Build with valid/invalid YAML
 * - Getters: getHeadChannels, getSaveIndices, getFromIndices
 *
 * @author WheelDL Team
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include "Model/Builder/ModelBuilder.h"
#include "Utils/Error/WheelLibException.h"
#include <filesystem>
#include <iostream>

using namespace WheelDL::Model::Builder;

// ============================================================================
// Skip Test Macro for GTest 1.8.x compatibility
// ============================================================================

#define SKIP_IF_FILE_NOT_EXISTS(path) \
    if (!std::filesystem::exists(path)) { \
        std::cout << "[  SKIPPED ] Test YAML file not found: " << path << std::endl; \
        SUCCEED(); \
        return; \
    }

// ============================================================================
// Test Fixture
// ============================================================================

class ModelBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);

        // Construct paths to test YAML files
        std::filesystem::path testDir = std::filesystem::current_path();

        // Navigate to find the test data directory
        while (!std::filesystem::exists(testDir / "WheelDL.Lib.Tests") && testDir.has_parent_path()) {
            testDir = testDir.parent_path();
        }

        testDataPath_ = testDir / "WheelDL.Lib.Tests" / "Data" / "ModelBuilderTest";
        simpleModelPath_ = (testDataPath_ / "simple_model.yaml").string();
        classificationModelPath_ = (testDataPath_ / "classification_model.yaml").string();
        scaledModelPath_ = (testDataPath_ / "scaled_model.yaml").string();
    }

    std::filesystem::path testDataPath_;
    std::string simpleModelPath_;
    std::string classificationModelPath_;
    std::string scaledModelPath_;
};

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_F(ModelBuilderTest, Constructor_Default) {
    ModelBuilder builder;

    // Default construction should not throw
    SUCCEED();
}

TEST_F(ModelBuilderTest, Constructor_DefaultValues) {
    ModelBuilder builder;

    // Before build, getters should return empty vectors
    EXPECT_TRUE(builder.getHeadChannels().empty());
    EXPECT_TRUE(builder.getSaveIndices().empty());
    EXPECT_TRUE(builder.getFromIndices().empty());
}

// ============================================================================
// Setter Method Chaining Tests
// ============================================================================

TEST_F(ModelBuilderTest, SetYamlPath_Valid) {
    SKIP_IF_FILE_NOT_EXISTS(simpleModelPath_);

    ModelBuilder builder;

    // setYamlPath should return reference for chaining
    auto& result = builder.setYamlPath(simpleModelPath_);

    EXPECT_EQ(&result, &builder);
}

TEST_F(ModelBuilderTest, SetInputChannels_Valid) {
    ModelBuilder builder;

    // setInputChannels should return reference for chaining
    auto& result = builder.setInputChannels(3);

    EXPECT_EQ(&result, &builder);
}

TEST_F(ModelBuilderTest, SetNumClasses_Valid) {
    ModelBuilder builder;

    // setNumClasses should return reference for chaining
    auto& result = builder.setNumClasses(80);

    EXPECT_EQ(&result, &builder);
}

TEST_F(ModelBuilderTest, SetImageSize_Valid) {
    ModelBuilder builder;

    // setImageSize should return reference for chaining
    auto& result = builder.setImageSize(640);

    EXPECT_EQ(&result, &builder);
}

TEST_F(ModelBuilderTest, MethodChaining_AllSetters) {
    SKIP_IF_FILE_NOT_EXISTS(simpleModelPath_);

    ModelBuilder builder;

    // Chain all setters together
    auto& result = builder
        .setYamlPath(simpleModelPath_)
        .setInputChannels(3)
        .setNumClasses(10)
        .setImageSize(640);

    // Final result should be reference to builder
    EXPECT_EQ(&result, &builder);
}

// ============================================================================
// Build Tests - Valid Models
// ============================================================================

TEST_F(ModelBuilderTest, Build_SimpleModel) {
    SKIP_IF_FILE_NOT_EXISTS(simpleModelPath_);

    ModelBuilder builder;
    builder
        .setYamlPath(simpleModelPath_)
        .setInputChannels(3)
        .setNumClasses(10)
        .setImageSize(640);

    auto model = builder.build();

    // Model should have layers
    EXPECT_GT(model->size(), 0);
}

TEST_F(ModelBuilderTest, Build_ClassificationModel) {
    SKIP_IF_FILE_NOT_EXISTS(classificationModelPath_);

    ModelBuilder builder;
    builder
        .setYamlPath(classificationModelPath_)
        .setInputChannels(3)
        .setNumClasses(100)
        .setImageSize(224);

    auto model = builder.build();

    EXPECT_GT(model->size(), 0);
}

TEST_F(ModelBuilderTest, Build_ScaledModel) {
    SKIP_IF_FILE_NOT_EXISTS(scaledModelPath_);

    ModelBuilder builder;
    builder
        .setYamlPath(scaledModelPath_)
        .setInputChannels(3)
        .setNumClasses(20)
        .setImageSize(320);

    auto model = builder.build();

    EXPECT_GT(model->size(), 0);
}

// ============================================================================
// Build Tests - Invalid YAML
// ============================================================================

TEST_F(ModelBuilderTest, Build_InvalidYaml_NonExistentFile) {
    ModelBuilder builder;

    // Non-existent file should throw
    EXPECT_THROW(
        builder.setYamlPath("non_existent_file.yaml"),
        std::exception
    );
}

TEST_F(ModelBuilderTest, Build_InvalidYaml_EmptyPath) {
    ModelBuilder builder;

    // Empty path should throw
    EXPECT_THROW(
        builder.setYamlPath(""),
        std::exception
    );
}

TEST_F(ModelBuilderTest, Build_WithoutYamlPath) {
    ModelBuilder builder;
    builder.setInputChannels(3).setNumClasses(10);

    // Building without YAML path should throw
    EXPECT_THROW(builder.build(), std::exception);
}

// ============================================================================
// Getter Tests - After Build
// ============================================================================

TEST_F(ModelBuilderTest, GetHeadChannels_AfterBuild) {
    SKIP_IF_FILE_NOT_EXISTS(simpleModelPath_);

    ModelBuilder builder;
    builder.setYamlPath(simpleModelPath_)
           .setInputChannels(3)
           .setNumClasses(10);

    auto model = builder.build();
    auto headChannels = builder.getHeadChannels();

    // Detection head should have multiple channel entries (for multi-scale)
    EXPECT_FALSE(headChannels.empty());
}

TEST_F(ModelBuilderTest, GetSaveIndices_AfterBuild) {
    SKIP_IF_FILE_NOT_EXISTS(simpleModelPath_);

    ModelBuilder builder;
    builder.setYamlPath(simpleModelPath_)
           .setInputChannels(3)
           .setNumClasses(10);

    auto model = builder.build();
    auto saveIndices = builder.getSaveIndices();

    // Save indices should be populated for skip connections
    EXPECT_FALSE(saveIndices.empty());

    // All indices should be non-negative
    for (int64_t idx : saveIndices) {
        EXPECT_GE(idx, 0);
    }

    // Indices should be sorted and unique
    for (size_t i = 1; i < saveIndices.size(); ++i) {
        EXPECT_LT(saveIndices[i - 1], saveIndices[i]);
    }
}

TEST_F(ModelBuilderTest, GetFromIndices_AfterBuild) {
    SKIP_IF_FILE_NOT_EXISTS(simpleModelPath_);

    ModelBuilder builder;
    builder.setYamlPath(simpleModelPath_)
           .setInputChannels(3)
           .setNumClasses(10);

    auto model = builder.build();
    auto fromIndices = builder.getFromIndices();

    // FromIndices should have entries for each layer
    EXPECT_FALSE(fromIndices.empty());

    // Each layer should have at least one "from" index
    for (const auto& indices : fromIndices) {
        EXPECT_FALSE(indices.empty());
    }
}

// ============================================================================
// Multiple Build Tests
// ============================================================================

TEST_F(ModelBuilderTest, Build_MultipleTimes) {
    SKIP_IF_FILE_NOT_EXISTS(simpleModelPath_);

    ModelBuilder builder;
    builder.setYamlPath(simpleModelPath_)
           .setInputChannels(3)
           .setNumClasses(10);

    // First build
    auto model1 = builder.build();
    auto headChannels1 = builder.getHeadChannels();

    // Second build should produce same result
    auto model2 = builder.build();
    auto headChannels2 = builder.getHeadChannels();

    EXPECT_EQ(model1->size(), model2->size());
    EXPECT_EQ(headChannels1.size(), headChannels2.size());
}

// ============================================================================
// Different Configuration Tests
// ============================================================================

TEST_F(ModelBuilderTest, Build_DifferentInputChannels) {
    SKIP_IF_FILE_NOT_EXISTS(simpleModelPath_);

    // Build with 3 input channels (RGB)
    ModelBuilder builder1;
    builder1.setYamlPath(simpleModelPath_)
            .setInputChannels(3)
            .setNumClasses(10);
    auto model3ch = builder1.build();

    // Build with 1 input channel (grayscale)
    ModelBuilder builder2;
    builder2.setYamlPath(simpleModelPath_)
            .setInputChannels(1)
            .setNumClasses(10);
    auto model1ch = builder2.build();

    // Both should build successfully
    EXPECT_GT(model3ch->size(), 0);
    EXPECT_GT(model1ch->size(), 0);
}

TEST_F(ModelBuilderTest, Build_DifferentNumClasses) {
    SKIP_IF_FILE_NOT_EXISTS(simpleModelPath_);

    // Build with 10 classes
    ModelBuilder builder1;
    builder1.setYamlPath(simpleModelPath_)
            .setInputChannels(3)
            .setNumClasses(10);
    auto model10 = builder1.build();

    // Build with 80 classes
    ModelBuilder builder2;
    builder2.setYamlPath(simpleModelPath_)
            .setInputChannels(3)
            .setNumClasses(80);
    auto model80 = builder2.build();

    // Both should build successfully
    EXPECT_GT(model10->size(), 0);
    EXPECT_GT(model80->size(), 0);
}

TEST_F(ModelBuilderTest, Build_DifferentImageSizes) {
    SKIP_IF_FILE_NOT_EXISTS(simpleModelPath_);

    // Build with 640 image size
    ModelBuilder builder1;
    builder1.setYamlPath(simpleModelPath_)
            .setInputChannels(3)
            .setNumClasses(10)
            .setImageSize(640);
    auto model640 = builder1.build();

    // Build with 320 image size
    ModelBuilder builder2;
    builder2.setYamlPath(simpleModelPath_)
            .setInputChannels(3)
            .setNumClasses(10)
            .setImageSize(320);
    auto model320 = builder2.build();

    // Both should build successfully
    EXPECT_GT(model640->size(), 0);
    EXPECT_GT(model320->size(), 0);
}

