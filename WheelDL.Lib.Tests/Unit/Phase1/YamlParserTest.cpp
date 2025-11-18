#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Config/YamlParser.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <yaml-cpp/yaml.h>
#include <filesystem>

using namespace WheelDL;
using namespace WheelDL::Config;
using namespace WheelDL::Utils;

namespace {
	std::filesystem::path getTestDataPath() {
		// Get the directory containing this source file
		std::filesystem::path sourceDir = __FILE__;
		sourceDir = sourceDir.parent_path(); // Remove filename

		// Navigate to WheelDL.Lib.Tests/Data
		while (sourceDir.filename() != "WheelDL.Lib.Tests" && sourceDir.has_parent_path()) {
			sourceDir = sourceDir.parent_path();
		}
		return sourceDir / "Data" / "Phase1TestConfigs";
	}
}

class YamlParserTest : public ::testing::Test {
protected:
	std::filesystem::path testDataPath;

	void SetUp() override {
		testDataPath = getTestDataPath();
	}
};

// ========== parseFrom Tests ==========

TEST_F(YamlParserTest, ParseFrom_ValidFile_Success) {
	std::string testFile = (testDataPath / "test_model.yaml").string();

	YAML::Node node = YamlParser::parseFrom(testFile);

	EXPECT_TRUE(node.IsDefined());
	EXPECT_TRUE(node["num_classes"]);
	EXPECT_EQ(node["num_classes"].as<int>(), 10);
}

TEST_F(YamlParserTest, ParseFrom_FileNotFound_ThrowsException) {
	std::string nonExistentFile = (testDataPath / "nonexistent.yaml").string();

	EXPECT_THROW({
		YamlParser::parseFrom(nonExistentFile);
	}, ConfigurationException);
}

TEST_F(YamlParserTest, ParseFrom_InvalidYaml_ThrowsException) {
	std::string invalidFile = (testDataPath / "test_invalid.yaml").string();

	EXPECT_THROW({
		YamlParser::parseFrom(invalidFile);
	}, ConfigurationException);
}

// ========== validate Tests ==========

TEST_F(YamlParserTest, Validate_DefinedNode_ReturnsTrue) {
	YAML::Node node;
	node["key"] = "value";

	EXPECT_TRUE(YamlParser::validate(node));
}

TEST_F(YamlParserTest, Validate_NullNode_ReturnsFalse) {
	YAML::Node node;

	EXPECT_FALSE(YamlParser::validate(node));
}

// ========== inferTaskTypeFrom Tests ==========

TEST_F(YamlParserTest, InferTaskType_Detection_ReturnsDetection) {
	std::string testFile = (testDataPath / "test_model.yaml").string();
	YAML::Node node = YamlParser::parseFrom(testFile);

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::DETECTION);
}

TEST_F(YamlParserTest, InferTaskType_Segmentation_ReturnsSegmentation) {
	std::string testFile = (testDataPath / "test_model_segment.yaml").string();
	YAML::Node node = YamlParser::parseFrom(testFile);

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::SEGMENTATION);
}

TEST_F(YamlParserTest, InferTaskType_Classification_ReturnsClassification) {
	std::string testFile = (testDataPath / "test_model_classify.yaml").string();
	YAML::Node node = YamlParser::parseFrom(testFile);

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::CLASSIFICATION);
}

TEST_F(YamlParserTest, InferTaskType_NoHead_ReturnsUnknown) {
	YAML::Node node;
	node["num_classes"] = 10;

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::UNKNOWN);
}

TEST_F(YamlParserTest, InferTaskType_InvalidNode_ReturnsUnknown) {
	YAML::Node node;

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::UNKNOWN);
}

// ========== getString Tests ==========

TEST_F(YamlParserTest, GetString_ExistingKey_ReturnsValue) {
	YAML::Node node;
	node["name"] = "TestName";

	std::string result = YamlParser::getString(node, "name", "default");

	EXPECT_EQ(result, "TestName");
}

TEST_F(YamlParserTest, GetString_MissingKey_ReturnsDefault) {
	YAML::Node node;

	std::string result = YamlParser::getString(node, "missing", "default");

	EXPECT_EQ(result, "default");
}

TEST_F(YamlParserTest, GetString_InvalidConversion_ReturnsDefault) {
	YAML::Node node;
	node["number"] = 123;

	std::string result = YamlParser::getString(node, "number", "default");

	// yaml-cpp should convert number to string
	EXPECT_EQ(result, "123");
}

// ========== getInt Tests ==========

TEST_F(YamlParserTest, GetInt_ExistingKey_ReturnsValue) {
	YAML::Node node;
	node["count"] = 42;

	int result = YamlParser::getInt(node, "count", 0);

	EXPECT_EQ(result, 42);
}

TEST_F(YamlParserTest, GetInt_MissingKey_ReturnsDefault) {
	YAML::Node node;

	int result = YamlParser::getInt(node, "missing", 99);

	EXPECT_EQ(result, 99);
}

TEST_F(YamlParserTest, GetInt_InvalidConversion_ReturnsDefault) {
	YAML::Node node;
	node["text"] = "not_a_number";

	int result = YamlParser::getInt(node, "text", 0);

	EXPECT_EQ(result, 0);
}

// ========== getFloat Tests ==========

TEST_F(YamlParserTest, GetFloat_ExistingKey_ReturnsValue) {
	YAML::Node node;
	node["value"] = 3.14f;

	float result = YamlParser::getFloat(node, "value", 0.0f);

	EXPECT_FLOAT_EQ(result, 3.14f);
}

TEST_F(YamlParserTest, GetFloat_MissingKey_ReturnsDefault) {
	YAML::Node node;

	float result = YamlParser::getFloat(node, "missing", 1.5f);

	EXPECT_FLOAT_EQ(result, 1.5f);
}

TEST_F(YamlParserTest, GetFloat_InvalidConversion_ReturnsDefault) {
	YAML::Node node;
	node["text"] = "not_a_float";

	float result = YamlParser::getFloat(node, "text", 0.0f);

	EXPECT_FLOAT_EQ(result, 0.0f);
}

// ========== getBool Tests ==========

TEST_F(YamlParserTest, GetBool_ExistingKey_ReturnsValue) {
	YAML::Node node;
	node["flag"] = true;

	bool result = YamlParser::getBool(node, "flag", false);

	EXPECT_TRUE(result);
}

TEST_F(YamlParserTest, GetBool_MissingKey_ReturnsDefault) {
	YAML::Node node;

	bool result = YamlParser::getBool(node, "missing", true);

	EXPECT_TRUE(result);
}

TEST_F(YamlParserTest, GetBool_InvalidConversion_ReturnsDefault) {
	YAML::Node node;
	node["text"] = "not_a_bool";

	bool result = YamlParser::getBool(node, "text", false);

	EXPECT_FALSE(result);
}

// ========== getClassNames Tests ==========

TEST_F(YamlParserTest, GetClassNames_MapFormat_ReturnsCorrectMap) {
	std::string testFile = (testDataPath / "test_dataset.yaml").string();
	YAML::Node node = YamlParser::parseFrom(testFile);

	std::map<int, std::string> classNames = YamlParser::getClassNames(node);

	EXPECT_EQ(classNames.size(), 4);
	EXPECT_EQ(classNames[0], "cat");
	EXPECT_EQ(classNames[1], "dog");
	EXPECT_EQ(classNames[2], "bird");
	EXPECT_EQ(classNames[3], "fish");
}

TEST_F(YamlParserTest, GetClassNames_ListFormat_ReturnsCorrectMap) {
	std::string testFile = (testDataPath / "test_dataset_list.yaml").string();
	YAML::Node node = YamlParser::parseFrom(testFile);

	std::map<int, std::string> classNames = YamlParser::getClassNames(node);

	EXPECT_EQ(classNames.size(), 4);
	EXPECT_EQ(classNames[0], "person");
	EXPECT_EQ(classNames[1], "car");
	EXPECT_EQ(classNames[2], "truck");
	EXPECT_EQ(classNames[3], "bicycle");
}

TEST_F(YamlParserTest, GetClassNames_NoNames_ReturnsEmptyMap) {
	YAML::Node node;
	node["key"] = "value";

	std::map<int, std::string> classNames = YamlParser::getClassNames(node);

	EXPECT_TRUE(classNames.empty());
}

// ========== checkFileSize Tests ==========

TEST_F(YamlParserTest, CheckFileSize_NormalFile_ReturnsTrue) {
	std::string testFile = (testDataPath / "test_model.yaml").string();

	bool result = YamlParser::checkFileSize(testFile);

	EXPECT_TRUE(result);
}

TEST_F(YamlParserTest, CheckFileSize_NonExistentFile_ReturnsFalse) {
	std::string nonExistentFile = (testDataPath / "nonexistent.yaml").string();

	bool result = YamlParser::checkFileSize(nonExistentFile);

	EXPECT_FALSE(result);
}

TEST_F(YamlParserTest, CheckFileSize_CustomLimit_WorksCorrectly) {
	std::string testFile = (testDataPath / "test_model.yaml").string();

	// Set very small limit (1 byte) - should fail
	bool result = YamlParser::checkFileSize(testFile, 1);

	EXPECT_FALSE(result);
}

// ========== checkDepth Tests ==========

TEST_F(YamlParserTest, CheckDepth_ShallowNode_ReturnsTrue) {
	YAML::Node node;
	node["key"] = "value";

	bool result = YamlParser::checkDepth(node, 10);

	EXPECT_TRUE(result);
}

TEST_F(YamlParserTest, CheckDepth_DeepNode_WorksCorrectly) {
	// Create a nested structure
	YAML::Node node;
	YAML::Node level1;
	YAML::Node level2;
	level2["deep"] = "value";
	level1["mid"] = level2;
	node["root"] = level1;

	// Should pass with limit 10
	EXPECT_TRUE(YamlParser::checkDepth(node, 10));

	// Should fail with limit 1
	EXPECT_FALSE(YamlParser::checkDepth(node, 1));
}
