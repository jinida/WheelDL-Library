#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Config/YamlParser.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>

using namespace WheelDL;
using namespace WheelDL::Config;
using namespace WheelDL::Utils;

namespace {
	std::filesystem::path getTestDataPath()
	{
		// Get the directory containing this source file
		std::filesystem::path sourceDir = __FILE__;
		sourceDir = sourceDir.parent_path(); // Remove filename

		// Navigate to WheelDL.Lib.Tests/Data
		while (sourceDir.filename() != "WheelDL.Lib.Tests" && sourceDir.has_parent_path()) {
			sourceDir = sourceDir.parent_path();
		}
		return sourceDir / "Data" / "EdgeCaseTests";
	}
}

/**
 * @class YamlParserEdgeCaseTest
 * @brief Comprehensive edge case tests for YamlParser module
 *
 * Tests:
 * - File size checking (under limit, at limit, over limit, empty file)
 * - Depth checking (at limit, exceeding limit, deeply nested structures)
 * - Task type inference (all task types, edge cases, missing heads)
 * - Safe getters (missing keys, type conversion failures, valid values)
 * - Class names parsing (map format, sequence format, empty/missing)
 */
class YamlParserEdgeCaseTest : public ::testing::Test
{
protected:
	std::filesystem::path testDataPath;
	std::vector<std::filesystem::path> tempFiles;

	void SetUp() override
	{
		testDataPath = getTestDataPath();

		// Create test directory if it doesn't exist
		if (!std::filesystem::exists(testDataPath)) {
			std::filesystem::create_directories(testDataPath);
		}

		// Create temporary test files
		createTestFiles();
	}

	void TearDown() override
	{
		// Clean up temporary files
		for (const auto& file : tempFiles) {
			if (std::filesystem::exists(file)) {
				try {
					std::filesystem::remove(file);
				}
				catch (...) {
					// Ignore cleanup errors
				}
			}
		}
	}

	void createTestFiles()
	{
		// Create various test YAML files for edge case testing
		createEmptyFile();
		createSmallFile();
		createFileAtLimit();
		createFileBeyondLimit();
		createShallowDepthFile();
		createDepthAtLimitFile();
		createDepthExceedingLimitFile();
		createAllTaskTypeFiles();
		createClassNamesFiles();
	}

	void createEmptyFile()
	{
		auto path = testDataPath / "empty.yaml";
		std::ofstream file(path);
		file.close();
		tempFiles.push_back(path);
	}

	void createSmallFile()
	{
		auto path = testDataPath / "small.yaml";
		std::ofstream file(path);
		file << "key: value\n";
		file << "num: 42\n";
		file.close();
		tempFiles.push_back(path);
	}

	void createFileAtLimit()
	{
		auto path = testDataPath / "at_limit.yaml";
		std::ofstream file(path, std::ios::binary);

		// Create file exactly at 10MB limit
		const size_t targetSize = 10 * 1024 * 1024;
		const std::string line = "key: value\n";
		size_t written = 0;

		while (written + line.size() <= targetSize) {
			file << line;
			written += line.size();
		}

		// Fill remaining bytes to reach exactly targetSize
		while (written < targetSize) {
			file << ' ';
			written++;
		}

		file.close();
		tempFiles.push_back(path);
	}

	void createFileBeyondLimit()
	{
		auto path = testDataPath / "beyond_limit.yaml";
		std::ofstream file(path);

		// Create file exceeding 10MB limit (10MB + 1KB)
		const size_t targetSize = (10 * 1024 * 1024) + 1024;
		const std::string line = "key: value\n";
		size_t written = 0;

		while (written < targetSize) {
			file << line;
			written += line.size();
		}

		file.close();
		tempFiles.push_back(path);
	}

	void createShallowDepthFile()
	{
		auto path = testDataPath / "shallow_depth.yaml";
		std::ofstream file(path);
		file << "level1:\n";
		file << "  level2:\n";
		file << "    level3: value\n";
		file.close();
		tempFiles.push_back(path);
	}

	void createDepthAtLimitFile()
	{
		auto path = testDataPath / "depth_at_limit.yaml";
		std::ofstream file(path);

		// Create nested structure at depth limit (100)
		// The root node is at depth 0, so we need 99 nested levels + 1 final scalar = 100 total depth
		for (int i = 0; i < 99; ++i) {
			for (int j = 0; j < i; ++j) {
				file << "  ";
			}
			file << "level" << i << ":\n";
		}
		// Add final value at depth 99
		for (int j = 0; j < 99; ++j) {
			file << "  ";
		}
		file << "value: final\n";

		file.close();
		tempFiles.push_back(path);
	}

	void createDepthExceedingLimitFile()
	{
		auto path = testDataPath / "depth_exceeding_limit.yaml";
		std::ofstream file(path);

		// Create nested structure exceeding depth limit (101)
		for (int i = 0; i < 101; ++i) {
			for (int j = 0; j < i; ++j) {
				file << "  ";
			}
			file << "level" << i << ":\n";
		}
		// Add final value
		for (int j = 0; j < 101; ++j) {
			file << "  ";
		}
		file << "value: final\n";

		file.close();
		tempFiles.push_back(path);
	}

	void createAllTaskTypeFiles()
	{
		// Detection task (head as sequence)
		{
			auto path = testDataPath / "task_detection_seq.yaml";
			std::ofstream file(path);
			file << "num_classes: 80\n";
			file << "head:\n";
			file << "  - [-1, 1, DetectModule, [80]]\n";
			file.close();
			tempFiles.push_back(path);
		}

		// Detection task (head as map)
		{
			auto path = testDataPath / "task_detection_map.yaml";
			std::ofstream file(path);
			file << "num_classes: 80\n";
			file << "head:\n";
			file << "  type: detection\n";
			file.close();
			tempFiles.push_back(path);
		}

		// Segmentation task (head as sequence)
		{
			auto path = testDataPath / "task_segmentation_seq.yaml";
			std::ofstream file(path);
			file << "num_classes: 80\n";
			file << "head:\n";
			file << "  - [-1, 1, SegmentModule, [80]]\n";
			file.close();
			tempFiles.push_back(path);
		}

		// Segmentation task (head as map)
		{
			auto path = testDataPath / "task_segmentation_map.yaml";
			std::ofstream file(path);
			file << "num_classes: 80\n";
			file << "head:\n";
			file << "  type: segment\n";
			file.close();
			tempFiles.push_back(path);
		}

		// OBB task (head as sequence)
		{
			auto path = testDataPath / "task_obb_seq.yaml";
			std::ofstream file(path);
			file << "num_classes: 80\n";
			file << "head:\n";
			file << "  - [-1, 1, OBBModule, [80]]\n";
			file.close();
			tempFiles.push_back(path);
		}

		// OBB task (head as map)
		{
			auto path = testDataPath / "task_obb_map.yaml";
			std::ofstream file(path);
			file << "num_classes: 80\n";
			file << "head:\n";
			file << "  type: obb\n";
			file.close();
			tempFiles.push_back(path);
		}

		// Classification task with "Classify" (head as sequence)
		{
			auto path = testDataPath / "task_classification_classify.yaml";
			std::ofstream file(path);
			file << "num_classes: 1000\n";
			file << "head:\n";
			file << "  - [-1, 1, ClassifyModule, [1000]]\n";
			file.close();
			tempFiles.push_back(path);
		}

		// Classification task with "ClassificationHead" (head as sequence)
		{
			auto path = testDataPath / "task_classification_head.yaml";
			std::ofstream file(path);
			file << "num_classes: 1000\n";
			file << "head:\n";
			file << "  - [-1, 1, ClassificationHead, [1000]]\n";
			file.close();
			tempFiles.push_back(path);
		}

		// Classification task (head as map)
		{
			auto path = testDataPath / "task_classification_map.yaml";
			std::ofstream file(path);
			file << "num_classes: 1000\n";
			file << "head:\n";
			file << "  type: classification\n";
			file.close();
			tempFiles.push_back(path);
		}

		// Unknown task (no recognizable head)
		{
			auto path = testDataPath / "task_unknown.yaml";
			std::ofstream file(path);
			file << "num_classes: 10\n";
			file << "head:\n";
			file << "  - [-1, 1, CustomModule, [10]]\n";
			file.close();
			tempFiles.push_back(path);
		}

		// Missing head section
		{
			auto path = testDataPath / "task_no_head.yaml";
			std::ofstream file(path);
			file << "num_classes: 10\n";
			file << "backbone:\n";
			file << "  type: resnet\n";
			file.close();
			tempFiles.push_back(path);
		}

		// Empty head section
		{
			auto path = testDataPath / "task_empty_head.yaml";
			std::ofstream file(path);
			file << "num_classes: 10\n";
			file << "head: []\n";
			file.close();
			tempFiles.push_back(path);
		}
	}

	void createClassNamesFiles()
	{
		// Map format
		{
			auto path = testDataPath / "class_names_map.yaml";
			std::ofstream file(path);
			file << "names:\n";
			file << "  0: person\n";
			file << "  1: bicycle\n";
			file << "  2: car\n";
			file.close();
			tempFiles.push_back(path);
		}

		// Sequence format
		{
			auto path = testDataPath / "class_names_seq.yaml";
			std::ofstream file(path);
			file << "names:\n";
			file << "  - person\n";
			file << "  - bicycle\n";
			file << "  - car\n";
			file.close();
			tempFiles.push_back(path);
		}

		// Empty names section
		{
			auto path = testDataPath / "class_names_empty.yaml";
			std::ofstream file(path);
			file << "names: []\n";
			file.close();
			tempFiles.push_back(path);
		}

		// Missing names section
		{
			auto path = testDataPath / "class_names_missing.yaml";
			std::ofstream file(path);
			file << "num_classes: 10\n";
			file.close();
			tempFiles.push_back(path);
		}

		// Large number of classes (sequence)
		{
			auto path = testDataPath / "class_names_large.yaml";
			std::ofstream file(path);
			file << "names:\n";
			for (int i = 0; i < 1000; ++i) {
				file << "  - class_" << i << "\n";
			}
			file.close();
			tempFiles.push_back(path);
		}

		// Non-contiguous map keys
		{
			auto path = testDataPath / "class_names_sparse.yaml";
			std::ofstream file(path);
			file << "names:\n";
			file << "  0: class_0\n";
			file << "  5: class_5\n";
			file << "  10: class_10\n";
			file.close();
			tempFiles.push_back(path);
		}
	}
};

// ========== File Size Checking Tests ==========

TEST_F(YamlParserEdgeCaseTest, FileSize_ValidFileUnder10MB_Success)
{
	auto path = testDataPath / "small.yaml";

	bool result = YamlParser::checkFileSize(path.string());

	EXPECT_TRUE(result);
}

TEST_F(YamlParserEdgeCaseTest, FileSize_ExactlyAt10MB_Success)
{
	auto path = testDataPath / "at_limit.yaml";

	bool result = YamlParser::checkFileSize(path.string());

	EXPECT_TRUE(result);
}

TEST_F(YamlParserEdgeCaseTest, FileSize_Exceeding10MB_ReturnsFalse)
{
	auto path = testDataPath / "beyond_limit.yaml";

	bool result = YamlParser::checkFileSize(path.string());

	EXPECT_FALSE(result);
}

TEST_F(YamlParserEdgeCaseTest, FileSize_EmptyFile_Success)
{
	auto path = testDataPath / "empty.yaml";

	bool result = YamlParser::checkFileSize(path.string());

	EXPECT_TRUE(result);
}

TEST_F(YamlParserEdgeCaseTest, FileSize_NonExistentFile_ReturnsFalse)
{
	auto path = testDataPath / "nonexistent_file_12345.yaml";

	bool result = YamlParser::checkFileSize(path.string());

	EXPECT_FALSE(result);
}

TEST_F(YamlParserEdgeCaseTest, FileSize_CustomLimit_WorksCorrectly)
{
	auto path = testDataPath / "small.yaml";

	// Very small limit (10 bytes)
	bool result = YamlParser::checkFileSize(path.string(), 10);

	EXPECT_FALSE(result);
}

TEST_F(YamlParserEdgeCaseTest, FileSize_CustomLimitLarge_WorksCorrectly)
{
	auto path = testDataPath / "small.yaml";

	// Very large limit (100MB)
	bool result = YamlParser::checkFileSize(path.string(), 100 * 1024 * 1024);

	EXPECT_TRUE(result);
}

TEST_F(YamlParserEdgeCaseTest, ParseFrom_FileTooLarge_ThrowsException)
{
	auto path = testDataPath / "beyond_limit.yaml";

	EXPECT_THROW({
		YamlParser::parseFrom(path.string());
		}, ConfigurationException);
}

// ========== Depth Checking Tests ==========

TEST_F(YamlParserEdgeCaseTest, Depth_ShallowNode_ReturnsTrue)
{
	auto path = testDataPath / "shallow_depth.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	bool result = YamlParser::checkDepth(node, 100);

	EXPECT_TRUE(result);
}

TEST_F(YamlParserEdgeCaseTest, Depth_AtLimit_ReturnsTrue)
{
	auto path = testDataPath / "depth_at_limit.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	bool result = YamlParser::checkDepth(node, 100);

	EXPECT_TRUE(result);
}

TEST_F(YamlParserEdgeCaseTest, Depth_ExceedingLimit_ReturnsFalse)
{
	auto path = testDataPath / "depth_exceeding_limit.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	bool result = YamlParser::checkDepth(node, 100);

	EXPECT_FALSE(result);
}

TEST_F(YamlParserEdgeCaseTest, Depth_CustomLimitLow_WorksCorrectly)
{
	auto path = testDataPath / "shallow_depth.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	// Shallow file has depth of 3, so limit of 2 should fail
	bool result = YamlParser::checkDepth(node, 2);

	EXPECT_FALSE(result);
}

TEST_F(YamlParserEdgeCaseTest, Depth_DeeplyNestedSequences_CheckedCorrectly)
{
	YAML::Node node;
	YAML::Node* current = &node;

	// Create deeply nested sequences - each level contains one sequence
	for (int i = 0; i < 50; ++i) {
		YAML::Node child = YAML::Node(YAML::NodeType::Sequence);
		current->push_back(child);
		current = &((*current)[0]);
	}

	EXPECT_TRUE(YamlParser::checkDepth(node, 100));
	EXPECT_FALSE(YamlParser::checkDepth(node, 30));
}

TEST_F(YamlParserEdgeCaseTest, Depth_DeeplyNestedMaps_CheckedCorrectly)
{
	YAML::Node node;
	YAML::Node* current = &node;

	// Create deeply nested maps
	for (int i = 0; i < 50; ++i) {
		(*current)["level"] = YAML::Node();
		current = &((*current)["level"]);
	}

	EXPECT_TRUE(YamlParser::checkDepth(node, 100));
	EXPECT_FALSE(YamlParser::checkDepth(node, 30));
}

TEST_F(YamlParserEdgeCaseTest, Depth_MixedNestedStructures_CheckedCorrectly)
{
	YAML::Node node;
	node["root"]["seq"].push_back("item1");
	node["root"]["seq"].push_back("item2");
	node["root"]["map"]["nested"]["deep"]["value"] = "test";

	EXPECT_TRUE(YamlParser::checkDepth(node, 10));
	EXPECT_TRUE(YamlParser::checkDepth(node, 100));
}

TEST_F(YamlParserEdgeCaseTest, ParseFrom_DepthExceeded_ThrowsException)
{
	auto path = testDataPath / "depth_exceeding_limit.yaml";

	EXPECT_THROW({
		YamlParser::parseFrom(path.string());
		}, ConfigurationException);
}

// ========== Task Type Inference Tests ==========

TEST_F(YamlParserEdgeCaseTest, InferTaskType_DetectionSequence_ReturnsDetection)
{
	auto path = testDataPath / "task_detection_seq.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::DETECTION);
}

TEST_F(YamlParserEdgeCaseTest, InferTaskType_DetectionMap_ReturnsDetection)
{
	auto path = testDataPath / "task_detection_map.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::DETECTION);
}

TEST_F(YamlParserEdgeCaseTest, InferTaskType_SegmentationSequence_ReturnsSegmentation)
{
	auto path = testDataPath / "task_segmentation_seq.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::SEGMENTATION);
}

TEST_F(YamlParserEdgeCaseTest, InferTaskType_SegmentationMap_ReturnsSegmentation)
{
	auto path = testDataPath / "task_segmentation_map.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::SEGMENTATION);
}

TEST_F(YamlParserEdgeCaseTest, InferTaskType_OBBSequence_ReturnsOBB)
{
	auto path = testDataPath / "task_obb_seq.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::OBB);
}

TEST_F(YamlParserEdgeCaseTest, InferTaskType_OBBMap_ReturnsOBB)
{
	auto path = testDataPath / "task_obb_map.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::OBB);
}

TEST_F(YamlParserEdgeCaseTest, InferTaskType_ClassificationWithClassify_ReturnsClassification)
{
	auto path = testDataPath / "task_classification_classify.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::CLASSIFICATION);
}

TEST_F(YamlParserEdgeCaseTest, InferTaskType_ClassificationWithClassificationHead_ReturnsClassification)
{
	auto path = testDataPath / "task_classification_head.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::CLASSIFICATION);
}

TEST_F(YamlParserEdgeCaseTest, InferTaskType_ClassificationMap_ReturnsClassification)
{
	auto path = testDataPath / "task_classification_map.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::CLASSIFICATION);
}

TEST_F(YamlParserEdgeCaseTest, InferTaskType_UnknownHead_ReturnsUnknown)
{
	auto path = testDataPath / "task_unknown.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::UNKNOWN);
}

TEST_F(YamlParserEdgeCaseTest, InferTaskType_MissingHead_ReturnsUnknown)
{
	auto path = testDataPath / "task_no_head.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::UNKNOWN);
}

TEST_F(YamlParserEdgeCaseTest, InferTaskType_EmptyHead_ReturnsUnknown)
{
	auto path = testDataPath / "task_empty_head.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::UNKNOWN);
}

TEST_F(YamlParserEdgeCaseTest, InferTaskType_InvalidNode_ReturnsUnknown)
{
	YAML::Node node;

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::UNKNOWN);
}

TEST_F(YamlParserEdgeCaseTest, InferTaskType_NullNode_ReturnsUnknown)
{
	YAML::Node node = YAML::Node(YAML::NodeType::Null);

	TaskType taskType = YamlParser::inferTaskTypeFrom(node);

	EXPECT_EQ(taskType, TaskType::UNKNOWN);
}

// ========== Safe Getters Tests ==========

TEST_F(YamlParserEdgeCaseTest, GetString_ExistingKey_ReturnsValue)
{
	YAML::Node node;
	node["name"] = "TestValue";

	std::string result = YamlParser::getString(node, "name", "default");

	EXPECT_EQ(result, "TestValue");
}

TEST_F(YamlParserEdgeCaseTest, GetString_MissingKey_ReturnsDefault)
{
	YAML::Node node;
	node["other"] = "value";

	std::string result = YamlParser::getString(node, "name", "default");

	EXPECT_EQ(result, "default");
}

TEST_F(YamlParserEdgeCaseTest, GetString_InvalidNode_ReturnsDefault)
{
	YAML::Node node;

	std::string result = YamlParser::getString(node, "name", "default");

	EXPECT_EQ(result, "default");
}

TEST_F(YamlParserEdgeCaseTest, GetString_EmptyString_ReturnsEmptyString)
{
	YAML::Node node;
	node["name"] = "";

	std::string result = YamlParser::getString(node, "name", "default");

	EXPECT_EQ(result, "");
}

TEST_F(YamlParserEdgeCaseTest, GetString_NumberAutoConverts_ReturnsString)
{
	YAML::Node node;
	node["value"] = 123;

	std::string result = YamlParser::getString(node, "value", "default");

	EXPECT_EQ(result, "123");
}

TEST_F(YamlParserEdgeCaseTest, GetInt_ExistingKey_ReturnsValue)
{
	YAML::Node node;
	node["count"] = 42;

	int result = YamlParser::getInt(node, "count", 0);

	EXPECT_EQ(result, 42);
}

TEST_F(YamlParserEdgeCaseTest, GetInt_MissingKey_ReturnsDefault)
{
	YAML::Node node;
	node["other"] = 10;

	int result = YamlParser::getInt(node, "count", 99);

	EXPECT_EQ(result, 99);
}

TEST_F(YamlParserEdgeCaseTest, GetInt_InvalidConversion_ReturnsDefault)
{
	YAML::Node node;
	node["text"] = "not_a_number";

	int result = YamlParser::getInt(node, "text", 0);

	EXPECT_EQ(result, 0);
}

TEST_F(YamlParserEdgeCaseTest, GetInt_NegativeValue_ReturnsNegative)
{
	YAML::Node node;
	node["value"] = -42;

	int result = YamlParser::getInt(node, "value", 0);

	EXPECT_EQ(result, -42);
}

TEST_F(YamlParserEdgeCaseTest, GetInt_ZeroValue_ReturnsZero)
{
	YAML::Node node;
	node["value"] = 0;

	int result = YamlParser::getInt(node, "value", 99);

	EXPECT_EQ(result, 0);
}

TEST_F(YamlParserEdgeCaseTest, GetFloat_ExistingKey_ReturnsValue)
{
	YAML::Node node;
	node["value"] = 3.14f;

	float result = YamlParser::getFloat(node, "value", 0.0f);

	EXPECT_FLOAT_EQ(result, 3.14f);
}

TEST_F(YamlParserEdgeCaseTest, GetFloat_MissingKey_ReturnsDefault)
{
	YAML::Node node;
	node["other"] = 1.0f;

	float result = YamlParser::getFloat(node, "value", 2.5f);

	EXPECT_FLOAT_EQ(result, 2.5f);
}

TEST_F(YamlParserEdgeCaseTest, GetFloat_InvalidConversion_ReturnsDefault)
{
	YAML::Node node;
	node["text"] = "not_a_float";

	float result = YamlParser::getFloat(node, "text", 1.0f);

	EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST_F(YamlParserEdgeCaseTest, GetFloat_NegativeValue_ReturnsNegative)
{
	YAML::Node node;
	node["value"] = -3.14f;

	float result = YamlParser::getFloat(node, "value", 0.0f);

	EXPECT_FLOAT_EQ(result, -3.14f);
}

TEST_F(YamlParserEdgeCaseTest, GetFloat_ZeroValue_ReturnsZero)
{
	YAML::Node node;
	node["value"] = 0.0f;

	float result = YamlParser::getFloat(node, "value", 1.0f);

	EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST_F(YamlParserEdgeCaseTest, GetBool_ExistingKey_ReturnsValue)
{
	YAML::Node node;
	node["flag"] = true;

	bool result = YamlParser::getBool(node, "flag", false);

	EXPECT_TRUE(result);
}

TEST_F(YamlParserEdgeCaseTest, GetBool_MissingKey_ReturnsDefault)
{
	YAML::Node node;
	node["other"] = true;

	bool result = YamlParser::getBool(node, "flag", false);

	EXPECT_FALSE(result);
}

TEST_F(YamlParserEdgeCaseTest, GetBool_InvalidConversion_ReturnsDefault)
{
	YAML::Node node;
	node["text"] = "not_a_bool";

	bool result = YamlParser::getBool(node, "text", false);

	EXPECT_FALSE(result);
}

TEST_F(YamlParserEdgeCaseTest, GetBool_FalseValue_ReturnsFalse)
{
	YAML::Node node;
	node["flag"] = false;

	bool result = YamlParser::getBool(node, "flag", true);

	EXPECT_FALSE(result);
}

// ========== Class Names Parsing Tests ==========

TEST_F(YamlParserEdgeCaseTest, GetClassNames_MapFormat_ReturnsCorrectMap)
{
	auto path = testDataPath / "class_names_map.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	std::map<int, std::string> classNames = YamlParser::getClassNames(node);

	EXPECT_EQ(classNames.size(), 3);
	EXPECT_EQ(classNames[0], "person");
	EXPECT_EQ(classNames[1], "bicycle");
	EXPECT_EQ(classNames[2], "car");
}

TEST_F(YamlParserEdgeCaseTest, GetClassNames_SequenceFormat_ReturnsCorrectMap)
{
	auto path = testDataPath / "class_names_seq.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	std::map<int, std::string> classNames = YamlParser::getClassNames(node);

	EXPECT_EQ(classNames.size(), 3);
	EXPECT_EQ(classNames[0], "person");
	EXPECT_EQ(classNames[1], "bicycle");
	EXPECT_EQ(classNames[2], "car");
}

TEST_F(YamlParserEdgeCaseTest, GetClassNames_EmptyNames_ReturnsEmptyMap)
{
	auto path = testDataPath / "class_names_empty.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	std::map<int, std::string> classNames = YamlParser::getClassNames(node);

	EXPECT_TRUE(classNames.empty());
}

TEST_F(YamlParserEdgeCaseTest, GetClassNames_MissingNames_ReturnsEmptyMap)
{
	auto path = testDataPath / "class_names_missing.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	std::map<int, std::string> classNames = YamlParser::getClassNames(node);

	EXPECT_TRUE(classNames.empty());
}

TEST_F(YamlParserEdgeCaseTest, GetClassNames_InvalidNode_ReturnsEmptyMap)
{
	YAML::Node node;

	std::map<int, std::string> classNames = YamlParser::getClassNames(node);

	EXPECT_TRUE(classNames.empty());
}

TEST_F(YamlParserEdgeCaseTest, GetClassNames_LargeNumberOfClasses_WorksCorrectly)
{
	auto path = testDataPath / "class_names_large.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	std::map<int, std::string> classNames = YamlParser::getClassNames(node);

	EXPECT_EQ(classNames.size(), 1000);
	EXPECT_EQ(classNames[0], "class_0");
	EXPECT_EQ(classNames[500], "class_500");
	EXPECT_EQ(classNames[999], "class_999");
}

TEST_F(YamlParserEdgeCaseTest, GetClassNames_SparseMap_WorksCorrectly)
{
	auto path = testDataPath / "class_names_sparse.yaml";
	YAML::Node node = YAML::LoadFile(path.string());

	std::map<int, std::string> classNames = YamlParser::getClassNames(node);

	EXPECT_EQ(classNames.size(), 3);
	EXPECT_EQ(classNames[0], "class_0");
	EXPECT_EQ(classNames[5], "class_5");
	EXPECT_EQ(classNames[10], "class_10");
	EXPECT_EQ(classNames.count(1), 0); // Should not have key 1
}

TEST_F(YamlParserEdgeCaseTest, GetClassNames_SingleClass_WorksCorrectly)
{
	YAML::Node node;
	node["names"].push_back("single_class");

	std::map<int, std::string> classNames = YamlParser::getClassNames(node);

	EXPECT_EQ(classNames.size(), 1);
	EXPECT_EQ(classNames[0], "single_class");
}

// ========== Integration Tests ==========

TEST_F(YamlParserEdgeCaseTest, Integration_ParseValidFileWithAllFeatures_Success)
{
	// Create a comprehensive test file
	auto path = testDataPath / "integration_full.yaml";
	std::ofstream file(path);
	file << "num_classes: 80\n";
	file << "task: detection\n";
	file << "head:\n";
	file << "  - [-1, 1, DetectModule, [80]]\n";
	file << "names:\n";
	file << "  0: person\n";
	file << "  1: bicycle\n";
	file << "hyperparams:\n";
	file << "  epochs: 100\n";
	file << "  batch_size: 16\n";
	file.close();
	tempFiles.push_back(path);

	YAML::Node node;
	EXPECT_NO_THROW({
		node = YamlParser::parseFrom(path.string());
		});

	EXPECT_TRUE(YamlParser::validate(node));
	EXPECT_EQ(YamlParser::inferTaskTypeFrom(node), TaskType::DETECTION);
	EXPECT_EQ(YamlParser::getInt(node, "num_classes", 0), 80);

	auto classNames = YamlParser::getClassNames(node);
	EXPECT_EQ(classNames.size(), 2);
}

TEST_F(YamlParserEdgeCaseTest, Integration_ParseEmptyFile_ThrowsException)
{
	auto path = testDataPath / "empty.yaml";

	EXPECT_THROW({
		YamlParser::parseFrom(path.string());
		}, ConfigurationException);
}

TEST_F(YamlParserEdgeCaseTest, Validate_MultipleNodeTypes_WorksCorrectly)
{
	// Defined map node
	YAML::Node mapNode;
	mapNode["key"] = "value";
	EXPECT_TRUE(YamlParser::validate(mapNode));

	// Defined sequence node
	YAML::Node seqNode;
	seqNode.push_back("item");
	EXPECT_TRUE(YamlParser::validate(seqNode));

	// Undefined node
	YAML::Node undefinedNode;
	EXPECT_FALSE(YamlParser::validate(undefinedNode));

	// Null node
	YAML::Node nullNode = YAML::Node(YAML::NodeType::Null);
	EXPECT_FALSE(YamlParser::validate(nullNode));
}

TEST_F(YamlParserEdgeCaseTest, Integration_AllGettersFallbackToDefaults_Success)
{
	YAML::Node node;
	node["defined_key"] = "value";

	// All missing keys should return defaults
	EXPECT_EQ(YamlParser::getString(node, "missing", "default_str"), "default_str");
	EXPECT_EQ(YamlParser::getInt(node, "missing", 42), 42);
	EXPECT_FLOAT_EQ(YamlParser::getFloat(node, "missing", 3.14f), 3.14f);
	EXPECT_TRUE(YamlParser::getBool(node, "missing", true));
}

// ========== Error Handling Tests ==========

TEST_F(YamlParserEdgeCaseTest, ErrorHandling_ParseInvalidYaml_ThrowsException)
{
	// Create malformed YAML
	auto path = testDataPath / "invalid.yaml";
	std::ofstream file(path);
	file << "key: value\n";
	file << "  invalid indentation\n";
	file << "another: [unclosed bracket\n";
	file.close();
	tempFiles.push_back(path);

	EXPECT_THROW({
		YamlParser::parseFrom(path.string());
		}, ConfigurationException);
}

TEST_F(YamlParserEdgeCaseTest, ErrorHandling_ParseNonExistentFile_ThrowsException)
{
	auto path = testDataPath / "definitely_does_not_exist_12345.yaml";

	EXPECT_THROW({
		YamlParser::parseFrom(path.string());
		}, ConfigurationException);
}

TEST_F(YamlParserEdgeCaseTest, ErrorHandling_GettersWithCorruptedData_ReturnDefaults)
{
	YAML::Node node;

	// Create node with sequence where scalar expected
	node["invalid_int"].push_back("item1");
	node["invalid_int"].push_back("item2");

	// Should return default, not crash
	EXPECT_EQ(YamlParser::getInt(node, "invalid_int", 0), 0);
	EXPECT_FLOAT_EQ(YamlParser::getFloat(node, "invalid_int", 0.0f), 0.0f);
	EXPECT_FALSE(YamlParser::getBool(node, "invalid_int", false));
}