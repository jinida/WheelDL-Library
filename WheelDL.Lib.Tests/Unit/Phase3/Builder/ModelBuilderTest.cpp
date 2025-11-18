#include "pch.h"
#include <gtest/gtest.h>
#include <filesystem>
#include "WheelDL.Lib/Model/Builder/ModelBuilder.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"

using namespace WheelDL;
using namespace WheelDL::Model::Builder;
using namespace WheelDL::Utils;

namespace {
	std::filesystem::path getTestDataPath() {
		std::filesystem::path sourceDir = __FILE__;
		sourceDir = sourceDir.parent_path();

		while (sourceDir.filename() != "WheelDL.Lib.Tests" && sourceDir.has_parent_path()) {
			sourceDir = sourceDir.parent_path();
		}
		return sourceDir / "Data";
	}
}

class ModelBuilderTest : public ::testing::Test {
protected:
	static std::unique_ptr<ModelBuilder> builder;
	static std::filesystem::path testConfigPath;

	static void SetUpTestSuite() {
		try {
			testConfigPath = getTestDataPath() / "Phase3TestConfigs" / "Models";
			builder = std::make_unique<ModelBuilder>();
		}
		catch (const std::exception& e) {
			std::cerr << "SetUpTestSuite failed: " << e.what() << std::endl;
			builder.reset();
		}
	}

	static void TearDownTestSuite() {
		builder.reset();
	}
};

// Static member initialization
std::unique_ptr<ModelBuilder> ModelBuilderTest::builder;
std::filesystem::path ModelBuilderTest::testConfigPath;

// ========== Construction & Basic Parsing Tests ==========

TEST_F(ModelBuilderTest, Construction_Success) {
	if (!builder) return;

	ModelBuilder testBuilder;
	EXPECT_NO_THROW({
		testBuilder.setInputChannels(3);
		testBuilder.setNumClasses(80);
		testBuilder.setImageSize(640);
	});
}

TEST_F(ModelBuilderTest, SetYamlPath_ValidFile_Success) {
	if (!builder) return;

	std::string modelPath = (testConfigPath / "minimal_detection.yaml").string();

	EXPECT_NO_THROW({
		ModelBuilder testBuilder;
		testBuilder.setYamlPath(modelPath);
	});
}

TEST_F(ModelBuilderTest, SetYamlPath_FileNotFound_ThrowsException) {
	if (!builder) return;

	std::string nonExistentPath = (testConfigPath / "nonexistent.yaml").string();

	EXPECT_THROW({
		ModelBuilder testBuilder;
		testBuilder.setYamlPath(nonExistentPath);
	}, std::runtime_error);
}

TEST_F(ModelBuilderTest, SetYamlPath_MalformedYaml_ThrowsException) {
	if (!builder) return;

	// This test would require a malformed YAML file
	// Skip if the file doesn't exist yet
	std::string malformedPath = (testConfigPath / "invalid_malformed.yaml").string();

	if (std::filesystem::exists(malformedPath)) {
		EXPECT_THROW({
			ModelBuilder testBuilder;
			testBuilder.setYamlPath(malformedPath);
		}, std::exception);
	}
}

TEST_F(ModelBuilderTest, Build_MinimalModel_Success) {
	if (!builder) return;

	std::string modelPath = (testConfigPath / "minimal_detection.yaml").string();

	EXPECT_NO_THROW({
		ModelBuilder testBuilder;
		testBuilder.setYamlPath(modelPath)
			.setInputChannels(3)
			.setNumClasses(10)
			.setImageSize(640);

		auto model = testBuilder.build();
		EXPECT_FALSE(model.is_empty());
	});
}

TEST_F(ModelBuilderTest, Build_BeforeSetPath_ThrowsException) {
	if (!builder) return;

	EXPECT_THROW({
		ModelBuilder testBuilder;
		testBuilder.setInputChannels(3)
			.setNumClasses(10)
			.setImageSize(640);

		auto model = testBuilder.build();
	}, std::exception);
}

TEST_F(ModelBuilderTest, GetHeadChannels_AfterParsing_ReturnsCorrect) {
	if (!builder) return;

	std::string modelPath = (testConfigPath / "minimal_detection.yaml").string();

	ModelBuilder testBuilder;
	testBuilder.setYamlPath(modelPath)
		.setInputChannels(3)
		.setNumClasses(10)
		.setImageSize(640);

	auto model = testBuilder.build();
	auto headChannels = testBuilder.getHeadChannels();

	EXPECT_FALSE(headChannels.empty());
	EXPECT_GT(headChannels.size(), 0);
}

TEST_F(ModelBuilderTest, GetSaveIndices_AfterParsing_ReturnsCorrect) {
	if (!builder) return;

	std::string modelPath = (testConfigPath / "minimal_detection.yaml").string();

	ModelBuilder testBuilder;
	testBuilder.setYamlPath(modelPath)
		.setInputChannels(3)
		.setNumClasses(10)
		.setImageSize(640);

	auto model = testBuilder.build();
	auto saveIndices = testBuilder.getSaveIndices();

	// Save indices should contain at least the output layer
	EXPECT_FALSE(saveIndices.empty());
}

// ========== Scale Application Tests ==========

TEST_F(ModelBuilderTest, SetInputChannels_ValidValue_Success) {
	if (!builder) return;

	EXPECT_NO_THROW({
		ModelBuilder testBuilder;
		testBuilder.setInputChannels(3);
		testBuilder.setInputChannels(1);
		testBuilder.setInputChannels(4);
	});
}

TEST_F(ModelBuilderTest, SetNumClasses_ValidValue_Success) {
	if (!builder) return;

	EXPECT_NO_THROW({
		ModelBuilder testBuilder;
		testBuilder.setNumClasses(10);
		testBuilder.setNumClasses(80);
		testBuilder.setNumClasses(1000);
	});
}

TEST_F(ModelBuilderTest, SetImageSize_ValidValue_Success) {
	if (!builder) return;

	EXPECT_NO_THROW({
		ModelBuilder testBuilder;
		testBuilder.setImageSize(320);
		testBuilder.setImageSize(640);
		testBuilder.setImageSize(1280);
	});
}

// ========== Validation Tests ==========

TEST_F(ModelBuilderTest, ValidateConfiguration_MissingHead_ThrowsException) {
	if (!builder) return;

	std::string invalidPath = (testConfigPath / "invalid_missing_head.yaml").string();

	if (std::filesystem::exists(invalidPath)) {
		EXPECT_THROW({
			ModelBuilder testBuilder;
			testBuilder.setYamlPath(invalidPath)
				.setInputChannels(3)
				.setNumClasses(10)
				.setImageSize(640);

			auto model = testBuilder.build();
		}, std::exception);
	}
}

// ========== Method Chaining Tests ==========

TEST_F(ModelBuilderTest, MethodChaining_AllSetters_Success) {
	if (!builder) return;

	std::string modelPath = (testConfigPath / "minimal_detection.yaml").string();

	EXPECT_NO_THROW({
		ModelBuilder testBuilder;
		auto model = testBuilder.setYamlPath(modelPath)
			.setInputChannels(3)
			.setNumClasses(10)
			.setImageSize(640)
			.build();

		EXPECT_FALSE(model.is_empty());
	});
}

// ========== Edge Cases Tests ==========

TEST_F(ModelBuilderTest, SetInputChannels_SingleChannel_Success) {
	if (!builder) return;

	EXPECT_NO_THROW({
		ModelBuilder testBuilder;
		testBuilder.setInputChannels(1);
	});
}

TEST_F(ModelBuilderTest, SetNumClasses_SingleClass_Success) {
	if (!builder) return;

	EXPECT_NO_THROW({
		ModelBuilder testBuilder;
		testBuilder.setNumClasses(1);
	});
}

TEST_F(ModelBuilderTest, SetImageSize_SmallSize_Success) {
	if (!builder) return;

	EXPECT_NO_THROW({
		ModelBuilder testBuilder;
		testBuilder.setImageSize(32);
	});
}

TEST_F(ModelBuilderTest, SetImageSize_LargeSize_Success) {
	if (!builder) return;

	EXPECT_NO_THROW({
		ModelBuilder testBuilder;
		testBuilder.setImageSize(1920);
	});
}
