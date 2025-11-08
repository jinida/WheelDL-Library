#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include "WheelDL.Lib/Config/YamlParser.h"
#include "WheelDL.Lib/Utils/Common/Types.h"
#include <filesystem>
#include <fstream>

using namespace WheelDL::Config;
using WheelDL::TaskType;
namespace fs = std::filesystem;

namespace {
	std::filesystem::path getTestDataPath()
	{
		std::filesystem::path sourceDir = __FILE__;
		sourceDir = sourceDir.parent_path();
		while (sourceDir.filename() != "WheelDL.Lib.Tests" && sourceDir.has_parent_path()) {
			sourceDir = sourceDir.parent_path();
		}
		return sourceDir / "Data";
	}
}

/**
 * @class ConfigTaskSpecificTest
 * @brief Tests for task-specific configuration logic
 *
 * Tests:
 * - Task-specific augmentation logic (ANOMALY forces degrees=0, perspective=0, blur=0)
 * - DDP detection with various device string formats
 * - Class names updating num_classes
 * - Cache setting parsing (bool vs string)
 */
class ConfigTaskSpecificTest : public ::testing::Test

{
protected:
	fs::path testConfigDir_;
	fs::path modelConfigPath_;
	fs::path hyperParamPath_;

	void SetUp() override
	{
		testConfigDir_ = getTestDataPath() / "Phase2TestConfigs";

		// Create temporary test config files in the test directory if they don't exist
		createTestConfigs();
	}

	void TearDown() override
	{
		// Optionally clean up temporary configs
	}

	void createTestConfigs()
	{
		// Create model configs for different tasks
		createModelConfig("test_model_detection.yaml", TaskType::DETECTION, 80);
		createModelConfig("test_model_obb.yaml", TaskType::OBB, 80);
		createModelConfig("test_model_anomaly.yaml", TaskType::ANOMALY, 2);
		createModelConfig("test_model_segmentation.yaml", TaskType::SEGMENTATION, 80);
		createModelConfig("test_model_classification.yaml", TaskType::CLASSIFICATION, 1000);

		// Create hyperparameter configs
		createHyperParamConfig("test_hyperparam_default.yaml", false);
		createHyperParamConfig("test_hyperparam_cache_bool.yaml", true);
		createHyperParamConfig("test_hyperparam_with_augments.yaml", false, true);
	}

	void createModelConfig(const std::string& filename, TaskType taskType, int numClasses)
	{
		fs::path path = testConfigDir_ / filename;
		if (fs::exists(path)) return; // Don't overwrite

		std::ofstream file(path);
		file << "# Test Model Configuration\n";
		file << "task: " << taskTypeToString(taskType) << "\n";
		file << "num_classes: " << numClasses << "\n";
		file << "default_act: ReLU\n";
		file << "scale:\n";
		file << "  width_multiple: 1.0\n";
		file << "  depth_multiple: 1.0\n";
		file << "  max_channels: 1024\n";
		file.close();
	}

	void createHyperParamConfig(const std::string& filename, bool useCacheBool, bool withAugments = false)
	{
		fs::path path = testConfigDir_ / filename;
		if (fs::exists(path)) return; // Don't overwrite

		std::ofstream file(path);
		file << "# Test Hyperparameter Configuration\n";
		file << "epochs: 100\n";
		file << "patience: 50\n";
		file << "batch_size: 16\n";
		file << "image_size: 640\n";
		file << "device: cpu\n";
		file << "workers: 4\n";
		file << "optimizer: auto\n";
		file << "seed: 42\n";

		// Cache configuration
		if (useCacheBool) {
			file << "cache: true\n";
		}
		else {
			file << "cache: ram\n";
		}
		file << "cache_size: 256\n";

		// Augmentation parameters
		if (withAugments) {
			file << "degrees: 10.0\n";
			file << "perspective: 0.001\n";
			file << "blur_probability: 0.01\n";
			file << "blur_kernel_size: 3\n";
		}
		else {
			file << "degrees: 0.0\n";
			file << "perspective: 0.0\n";
			file << "blur_probability: 0.0\n";
		}

		file << "hsv_h: 0.015\n";
		file << "hsv_s: 0.7\n";
		file << "hsv_v: 0.4\n";
		file << "translate: 0.1\n";
		file << "scale: 0.5\n";
		file << "shear: 0.0\n";
		file << "flipud: 0.0\n";
		file << "fliplr: 0.5\n";
		file << "mosaic: 1.0\n";
		file.close();
	}

	void createDatasetConfig(const std::string& filename, int numClasses)
	{
		fs::path path = testConfigDir_ / filename;
		if (fs::exists(path)) return;

		std::ofstream file(path);
		file << "# Test Dataset Configuration\n";
		file << "name: test_dataset\n";
		file << "image_path: images/\n";
		file << "label_path: labels/\n";
		file << "class_names:\n";
		for (int i = 0; i < numClasses; ++i) {
			file << "  - class_" << i << "\n";
		}
		file.close();
	}

	std::string taskTypeToString(TaskType type)
	{
		switch (type) {
		case TaskType::DETECTION: return "detect";
		case TaskType::CLASSIFICATION: return "classify";
		case TaskType::SEGMENTATION: return "segment";
		case TaskType::OBB: return "obb";
		case TaskType::ANOMALY: return "anomaly";
		default: return "unknown";
		}
	}
};

// ========== Task-Specific Augmentation Logic Tests ==========

TEST_F(ConfigTaskSpecificTest, AnomalyTask_ForcesDegreesZero)
{
	modelConfigPath_ = testConfigDir_ / "test_model_anomaly.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_with_augments.yaml";

	Configuration config;
	ASSERT_NO_THROW(config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string()));

	EXPECT_EQ(config.getTaskType(), TaskType::ANOMALY);
	// Degrees should be forced to 0 for ANOMALY task
	EXPECT_FLOAT_EQ(config.getDegrees(), 0.0f);
}

TEST_F(ConfigTaskSpecificTest, AnomalyTask_ForcesPerspectiveZero)
{
	modelConfigPath_ = testConfigDir_ / "test_model_anomaly.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_with_augments.yaml";

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string());

	// Perspective should be forced to 0 for ANOMALY task
	EXPECT_FLOAT_EQ(config.getPerspective(), 0.0f);
}

TEST_F(ConfigTaskSpecificTest, AnomalyTask_ForcesBlurZero)
{
	modelConfigPath_ = testConfigDir_ / "test_model_anomaly.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_with_augments.yaml";

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string());

	// Blur probability should be forced to 0 for ANOMALY task
	EXPECT_FLOAT_EQ(config.getBlurProbability(), 0.0f);
}

TEST_F(ConfigTaskSpecificTest, DetectionTask_ForcesDegreesZero)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_with_augments.yaml";

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string());

	EXPECT_EQ(config.getTaskType(), TaskType::DETECTION);
	// Degrees should be forced to 0 for DETECTION task
	EXPECT_FLOAT_EQ(config.getDegrees(), 0.0f);
}

TEST_F(ConfigTaskSpecificTest, OBBTask_ForcesPerspectiveZero)
{
	modelConfigPath_ = testConfigDir_ / "test_model_obb.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_with_augments.yaml";

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string());

	EXPECT_EQ(config.getTaskType(), TaskType::OBB);
	// Perspective should be forced to 0 for OBB task
	EXPECT_FLOAT_EQ(config.getPerspective(), 0.0f);
}

TEST_F(ConfigTaskSpecificTest, SegmentationTask_AllowsAllAugmentations)
{
	modelConfigPath_ = testConfigDir_ / "test_model_segmentation.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_with_augments.yaml";

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string());

	EXPECT_EQ(config.getTaskType(), TaskType::SEGMENTATION);
	// Segmentation should allow all augmentations
	EXPECT_GT(config.getDegrees(), 0.0f);
	EXPECT_GT(config.getPerspective(), 0.0f);
	EXPECT_GT(config.getBlurProbability(), 0.0f);
}

TEST_F(ConfigTaskSpecificTest, ClassificationTask_AllowsAllAugmentations)
{
	modelConfigPath_ = testConfigDir_ / "test_model_classification.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_with_augments.yaml";

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string());

	EXPECT_EQ(config.getTaskType(), TaskType::CLASSIFICATION);
	// Classification should allow all augmentations
	EXPECT_GT(config.getDegrees(), 0.0f);
	EXPECT_GT(config.getPerspective(), 0.0f);
	EXPECT_GT(config.getBlurProbability(), 0.0f);
}

// ========== DDP Detection Tests ==========

TEST_F(ConfigTaskSpecificTest, DDP_SingleDevice_String)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create hyperparam config with single device
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_single_device.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file << "device: \"0\"\n";
	file.close();

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperPath.string());

	EXPECT_FALSE(config.useDDP());
	fs::remove(hyperPath);
}

TEST_F(ConfigTaskSpecificTest, DDP_SingleDevice_Int)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create hyperparam config with single device (int)
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_single_device_int.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file << "device: 0\n";
	file.close();

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperPath.string());

	EXPECT_FALSE(config.useDDP());
	fs::remove(hyperPath);
}

TEST_F(ConfigTaskSpecificTest, DDP_MultipleDevices_CommaString)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create hyperparam config with multiple devices
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_multi_device_comma.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file << "device: \"0,1,2,3\"\n";
	file.close();

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperPath.string());

	EXPECT_TRUE(config.useDDP());
	fs::remove(hyperPath);
}

TEST_F(ConfigTaskSpecificTest, DDP_MultipleDevices_List)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create hyperparam config with device list
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_multi_device_list.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file << "device: [0, 1, 2, 3]\n";
	file.close();

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperPath.string());

	EXPECT_TRUE(config.useDDP());
	fs::remove(hyperPath);
}

TEST_F(ConfigTaskSpecificTest, DDP_TwoDevices)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create hyperparam config with 2 devices
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_two_devices.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file << "device: \"0,1\"\n";
	file.close();

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperPath.string());

	EXPECT_TRUE(config.useDDP());
	fs::remove(hyperPath);
}

TEST_F(ConfigTaskSpecificTest, DDP_EmptyDevice)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create hyperparam config with empty device
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_empty_device.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file << "device: \"\"\n";
	file.close();

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperPath.string());

	EXPECT_FALSE(config.useDDP());
	fs::remove(hyperPath);
}

TEST_F(ConfigTaskSpecificTest, DDP_CUDAString)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create hyperparam config with cuda string
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_cuda_string.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file << "device: \"cuda\"\n";
	file.close();

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperPath.string());

	EXPECT_FALSE(config.useDDP()); // "cuda" has no comma
	fs::remove(hyperPath);
}

// ========== Class Names and num_classes Tests ==========

TEST_F(ConfigTaskSpecificTest, ClassNames_UpdatesNumClasses)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_default.yaml";

	// Create dataset config with class names
	std::string datasetFile = "test_dataset_5classes.yaml";
	createDatasetConfig(datasetFile, 5);
	fs::path datasetPath = testConfigDir_ / datasetFile;

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string(), datasetPath.string());

	// num_classes should remain 80 (from model config), class_names loaded from dataset
	EXPECT_EQ(config.getNumClasses(), 80);
	EXPECT_EQ(config.getClassNames().size(), 5);

	fs::remove(datasetPath);
}

TEST_F(ConfigTaskSpecificTest, ClassNames_PreservesModelNumClasses_IfSet)
{
	// Model config has num_classes = 80
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_default.yaml";

	// Dataset has 5 classes
	std::string datasetFile = "test_dataset_5classes_preserve.yaml";
	createDatasetConfig(datasetFile, 5);
	fs::path datasetPath = testConfigDir_ / datasetFile;

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string(), datasetPath.string());

	// num_classes should remain 80 from model config (preserved)
	EXPECT_EQ(config.getNumClasses(), 80);
	EXPECT_EQ(config.getClassNames().size(), 5);

	fs::remove(datasetPath);
}

TEST_F(ConfigTaskSpecificTest, ClassNames_EmptyDataset)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_default.yaml";

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string());

	// num_classes should be from model config
	EXPECT_EQ(config.getNumClasses(), 80);
	EXPECT_TRUE(config.getClassNames().empty());
}

TEST_F(ConfigTaskSpecificTest, ClassNames_LargeNumberOfClasses)
{
	modelConfigPath_ = testConfigDir_ / "test_model_classification.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_default.yaml";

	// Create dataset with 100 classes
	std::string datasetFile = "test_dataset_100classes.yaml";
	createDatasetConfig(datasetFile, 100);
	fs::path datasetPath = testConfigDir_ / datasetFile;

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string(), datasetPath.string());

	EXPECT_EQ(config.getNumClasses(), 1000); // From model config (preserved)
	EXPECT_EQ(config.getClassNames().size(), 100);

	fs::remove(datasetPath);
}

// ========== Cache Setting Parsing Tests ==========

TEST_F(ConfigTaskSpecificTest, Cache_BoolTrue)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_cache_bool.yaml";

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string());

	// cache: true should be parsed as "ram"
	EXPECT_EQ(config.getCacheType(), "ram");
}

TEST_F(ConfigTaskSpecificTest, Cache_BoolFalse)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create config with cache: false
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_cache_false.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file << "cache: false\n";
	file.close();

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperPath.string());

	// cache: false should be parsed as empty string
	EXPECT_EQ(config.getCacheType(), "");

	fs::remove(hyperPath);
}

TEST_F(ConfigTaskSpecificTest, Cache_StringRam)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_default.yaml";

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string());

	// cache: ram should be parsed as "ram"
	EXPECT_EQ(config.getCacheType(), "ram");
}

TEST_F(ConfigTaskSpecificTest, Cache_StringDisk)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create config with cache: disk
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_cache_disk.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file << "cache: disk\n";
	file.close();

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperPath.string());

	EXPECT_EQ(config.getCacheType(), "disk");

	fs::remove(hyperPath);
}

TEST_F(ConfigTaskSpecificTest, Cache_NotSpecified)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create config without cache setting
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_no_cache.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file.close();

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperPath.string());

	// Default should be empty string
	EXPECT_EQ(config.getCacheType(), "");

	fs::remove(hyperPath);
}

TEST_F(ConfigTaskSpecificTest, CacheSize_Default)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_default.yaml";

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperParamPath_.string());

	EXPECT_EQ(config.getCacheSize(), 256);
}

TEST_F(ConfigTaskSpecificTest, CacheSize_Custom)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create config with custom cache size
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_cache_custom_size.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file << "cache: ram\n";
	file << "cache_size: 1000\n";
	file.close();

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperPath.string());

	EXPECT_EQ(config.getCacheSize(), 1000);

	fs::remove(hyperPath);
}

TEST_F(ConfigTaskSpecificTest, CacheSize_Unlimited)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create config with cache_size: 0 (unlimited)
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_cache_unlimited.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file << "cache: ram\n";
	file << "cache_size: 0\n";
	file.close();

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperPath.string());

	EXPECT_EQ(config.getCacheSize(), 0);

	fs::remove(hyperPath);
}

// ========== Integration Tests ==========

TEST_F(ConfigTaskSpecificTest, Integration_AnomalyWithFullConfig)
{
	modelConfigPath_ = testConfigDir_ / "test_model_anomaly.yaml";

	// Create full config
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_anomaly_full.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 64\n";
	file << "image_size: 256\n";
	file << "device: \"0,1\"\n";
	file << "cache: true\n";
	file << "cache_size: 2000\n";
	file << "degrees: 5.0\n";
	file << "perspective: 0.001\n";
	file << "blur_probability: 0.3\n";
	file << "translate: 0.05\n";
	file << "scale: 0.3\n";
	file.close();

	Configuration config;
	config.loadFromYaml(modelConfigPath_.string(), hyperPath.string());

	// Verify task-specific overrides
	EXPECT_EQ(config.getTaskType(), TaskType::ANOMALY);
	EXPECT_FLOAT_EQ(config.getDegrees(), 0.0f);
	EXPECT_FLOAT_EQ(config.getPerspective(), 0.0f);
	EXPECT_FLOAT_EQ(config.getBlurProbability(), 0.0f);

	// Verify other settings preserved
	EXPECT_FLOAT_EQ(config.getTranslate(), 0.05f);
	EXPECT_FLOAT_EQ(config.getScale(), 0.3f);
	EXPECT_EQ(config.getBatchSize(), 64);
	EXPECT_EQ(config.getCacheType(), "ram");
	EXPECT_TRUE(config.useDDP());

	fs::remove(hyperPath);
}

TEST_F(ConfigTaskSpecificTest, Integration_MultipleTasksSequential)
{
	hyperParamPath_ = testConfigDir_ / "test_hyperparam_with_augments.yaml";

	// Test DETECTION
	Configuration config1;
	config1.loadFromYaml((testConfigDir_ / "test_model_detection.yaml").string(),
		hyperParamPath_.string());
	EXPECT_EQ(config1.getTaskType(), TaskType::DETECTION);
	EXPECT_FLOAT_EQ(config1.getDegrees(), 0.0f);

	// Test OBB
	Configuration config2;
	config2.loadFromYaml((testConfigDir_ / "test_model_obb.yaml").string(),
		hyperParamPath_.string());
	EXPECT_EQ(config2.getTaskType(), TaskType::OBB);
	EXPECT_FLOAT_EQ(config2.getPerspective(), 0.0f);

	// Test SEGMENTATION
	Configuration config3;
	config3.loadFromYaml((testConfigDir_ / "test_model_segmentation.yaml").string(),
		hyperParamPath_.string());
	EXPECT_EQ(config3.getTaskType(), TaskType::SEGMENTATION);
	EXPECT_GT(config3.getDegrees(), 0.0f);
	EXPECT_GT(config3.getPerspective(), 0.0f);
}

// ========== Edge Cases ==========

TEST_F(ConfigTaskSpecificTest, EdgeCase_InvalidDeviceFormat)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create config with unusual device format
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_device_unusual.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file << "device: \"cuda:0\"\n";
	file.close();

	Configuration config;
	EXPECT_NO_THROW(config.loadFromYaml(modelConfigPath_.string(), hyperPath.string()));

	// "cuda:0" has a colon but no comma, so should not be DDP
	EXPECT_FALSE(config.useDDP());

	fs::remove(hyperPath);
}

TEST_F(ConfigTaskSpecificTest, EdgeCase_DeviceWithSpaces)
{
	modelConfigPath_ = testConfigDir_ / "test_model_detection.yaml";

	// Create config with spaces in device string
	fs::path hyperPath = testConfigDir_ / "test_hyperparam_device_spaces.yaml";
	std::ofstream file(hyperPath);
	file << "epochs: 100\n";
	file << "batch_size: 16\n";
	file << "image_size: 640\n";
	file << "device: \"0, 1, 2\"\n";
	file.close();

	Configuration config;
	EXPECT_NO_THROW(config.loadFromYaml(modelConfigPath_.string(), hyperPath.string()));

	// Should still detect DDP (has commas)
	EXPECT_TRUE(config.useDDP());

	fs::remove(hyperPath);
}

TEST_F(ConfigTaskSpecificTest, EdgeCase_ZeroNumClasses)
{
	// Create model with 0 classes
	fs::path modelPath = testConfigDir_ / "test_model_zero_classes.yaml";
	std::ofstream file(modelPath);
	file << "task: detect\n";
	file << "num_classes: 0\n";
	file.close();

	hyperParamPath_ = testConfigDir_ / "test_hyperparam_default.yaml";

	// Create dataset with classes
	std::string datasetFile = "test_dataset_for_zero_model.yaml";
	createDatasetConfig(datasetFile, 10);
	fs::path datasetPath = testConfigDir_ / datasetFile;

	Configuration config;
	config.loadFromYaml(modelPath.string(), hyperParamPath_.string(), datasetPath.string());

	// Should update from dataset
	EXPECT_EQ(config.getNumClasses(), 10);

	fs::remove(modelPath);
	fs::remove(datasetPath);
}
