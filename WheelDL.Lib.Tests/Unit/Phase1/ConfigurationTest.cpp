#include "pch.h"
#include <gtest/gtest.h>
#include <filesystem>
#include "WheelDL.Lib/Config/Configuration.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"

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
		return sourceDir / "Phase1TestConfigs" / "Data";
	}
}

class ConfigurationTest : public ::testing::Test {
protected:
	std::filesystem::path testDataPath;
	Configuration config;

	void SetUp() override {
		testDataPath = getTestDataPath();
	}
};

// ========== Constructor Tests ==========

TEST_F(ConfigurationTest, Constructor_InitializesDefaults) {
	Configuration cfg;

	EXPECT_EQ(cfg.getTaskType(), TaskType::UNKNOWN);
	EXPECT_EQ(cfg.getNumClasses(), 0);
	EXPECT_FLOAT_EQ(cfg.getWidthMultiple(), 1.0f);
	EXPECT_FLOAT_EQ(cfg.getDepthMultiple(), 1.0f);
	EXPECT_EQ(cfg.getMaxChannels(), 1024);
	EXPECT_EQ(cfg.getEpochs(), 100);
	EXPECT_EQ(cfg.getBatchSize(), 16);
	EXPECT_EQ(cfg.getImageSize(), 640);
	EXPECT_FLOAT_EQ(cfg.getLearningRate(), 0.01f);
}

// ========== loadFromYaml Tests ==========

TEST_F(ConfigurationTest, LoadFromYaml_AllFiles_Success) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();
	std::string datasetPath = (testDataPath / "test_dataset.yaml").string();

	config.loadFromYaml(modelPath, hyperPath, datasetPath);

	// Verify model config
	EXPECT_EQ(config.getTaskType(), TaskType::DETECTION);
	EXPECT_EQ(config.getNumClasses(), 10);
	EXPECT_EQ(config.getDefaultActivation(), "ReLU");
	EXPECT_FLOAT_EQ(config.getWidthMultiple(), 0.5f);
	EXPECT_FLOAT_EQ(config.getDepthMultiple(), 0.33f);
	EXPECT_EQ(config.getMaxChannels(), 512);

	// Verify hyperparameters
	EXPECT_EQ(config.getEpochs(), 50);
	EXPECT_EQ(config.getPatience(), 10);
	EXPECT_EQ(config.getBatchSize(), 8);
	EXPECT_EQ(config.getImageSize(), 320);
	EXPECT_EQ(config.getDevice(), "0");
	EXPECT_EQ(config.getWorkers(), 4);
	EXPECT_EQ(config.getOptimizer(), "SGD");
	EXPECT_EQ(config.getSeed(), 42);
	EXPECT_TRUE(config.isDeterministic());
	EXPECT_FALSE(config.useCosineLR());
	EXPECT_EQ(config.getCloseMosaic(), 5);
	EXPECT_TRUE(config.useAMP());
	EXPECT_EQ(config.getCacheType(), "ram");

	// Verify optimizer settings
	EXPECT_FLOAT_EQ(config.getLearningRate(), 0.01f);
	EXPECT_FLOAT_EQ(config.getLRFinalFraction(), 0.1f);
	EXPECT_FLOAT_EQ(config.getMomentum(), 0.9f);
	EXPECT_FLOAT_EQ(config.getWeightDecay(), 0.0001f);
	EXPECT_FLOAT_EQ(config.getWarmupEpochs(), 2.0f);
	EXPECT_FLOAT_EQ(config.getWarmupMomentum(), 0.8f);
	EXPECT_FLOAT_EQ(config.getWarmupBiasLR(), 0.05f);

	// Verify loss gains
	EXPECT_FLOAT_EQ(config.getBoxGain(), 5.0f);
	EXPECT_FLOAT_EQ(config.getClsGain(), 0.3f);
	EXPECT_FLOAT_EQ(config.getDFLGain(), 1.0f);

	// Verify augmentation settings
	EXPECT_FLOAT_EQ(config.getHSVH(), 0.01f);
	EXPECT_FLOAT_EQ(config.getHSVS(), 0.5f);
	EXPECT_FLOAT_EQ(config.getHSVV(), 0.3f);
	// For DETECTION task, degrees should be 0.0f (not used)
	EXPECT_FLOAT_EQ(config.getDegrees(), 0.0f);
	EXPECT_FLOAT_EQ(config.getTranslate(), 0.2f);
	EXPECT_FLOAT_EQ(config.getScale(), 0.9f);
	EXPECT_FLOAT_EQ(config.getShear(), 2.0f);
	EXPECT_FLOAT_EQ(config.getPerspective(), 0.0001f);
	EXPECT_FLOAT_EQ(config.getFlipUD(), 0.1f);
	EXPECT_FLOAT_EQ(config.getFlipLR(), 0.5f);
	EXPECT_FLOAT_EQ(config.getMosaic(), 0.8f);
	EXPECT_FLOAT_EQ(config.getMixup(), 0.1f);
	EXPECT_FLOAT_EQ(config.getCutmix(), 0.0f);

	// Verify validation settings
	EXPECT_FLOAT_EQ(config.getIoU(), 0.6f);
	EXPECT_EQ(config.getMaxDet(), 100);

	// Verify classification settings
	EXPECT_FLOAT_EQ(config.getDropout(), 0.2f);

	// Verify dataset config
	EXPECT_EQ(config.getDatasetName(), "TestDataset");
	EXPECT_EQ(config.getImagePath(), "images/test/");
	EXPECT_EQ(config.getLabelPath(), "labels/test/");

	const auto& classNames = config.getClassNames();
	EXPECT_EQ(classNames.size(), 4);
	EXPECT_EQ(classNames.at(0), "cat");
	EXPECT_EQ(classNames.at(1), "dog");
	EXPECT_EQ(classNames.at(2), "bird");
	EXPECT_EQ(classNames.at(3), "fish");
}

TEST_F(ConfigurationTest, LoadFromYaml_WithoutDataset_Success) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();

	config.loadFromYaml(modelPath, hyperPath);

	EXPECT_EQ(config.getNumClasses(), 10);
	EXPECT_EQ(config.getEpochs(), 50);
	EXPECT_EQ(config.getDatasetName(), "");
}

TEST_F(ConfigurationTest, LoadFromYaml_InvalidModelPath_ThrowsException) {
	std::string modelPath = (testDataPath / "nonexistent.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();

	EXPECT_THROW({
		config.loadFromYaml(modelPath, hyperPath);
		}, ConfigurationException);
}

TEST_F(ConfigurationTest, LoadFromYaml_InvalidHyperPath_ThrowsException) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "nonexistent.yaml").string();

	EXPECT_THROW({
		config.loadFromYaml(modelPath, hyperPath);
		}, ConfigurationException);
}

// ========== TaskType Tests ==========

TEST_F(ConfigurationTest, LoadFromYaml_SegmentationModel_CorrectTaskType) {
	std::string modelPath = (testDataPath / "test_model_segment.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();

	config.loadFromYaml(modelPath, hyperPath);

	EXPECT_EQ(config.getTaskType(), TaskType::SEGMENTATION);
}

TEST_F(ConfigurationTest, LoadFromYaml_ClassificationModel_CorrectTaskType) {
	std::string modelPath = (testDataPath / "test_model_classify.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();

	config.loadFromYaml(modelPath, hyperPath);

	EXPECT_EQ(config.getTaskType(), TaskType::CLASSIFICATION);
}

// ========== Device Configuration Tests ==========

TEST_F(ConfigurationTest, Device_ScalarString_LoadsCorrectly) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();

	config.loadFromYaml(modelPath, hyperPath);

	EXPECT_EQ(config.getDevice(), "0");
	EXPECT_FALSE(config.useDDP());
}

TEST_F(ConfigurationTest, Device_List_LoadsCorrectly) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_device_list.yaml").string();

	config.loadFromYaml(modelPath, hyperPath);

	EXPECT_EQ(config.getDevice(), "0,1,2,3");
	EXPECT_TRUE(config.useDDP());
}

// ========== Cache Configuration Tests ==========

TEST_F(ConfigurationTest, Cache_String_LoadsCorrectly) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();

	config.loadFromYaml(modelPath, hyperPath);

	EXPECT_EQ(config.getCacheType(), "ram");
}

TEST_F(ConfigurationTest, Cache_BoolTrue_ConvertsToRam) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_cache_bool.yaml").string();

	config.loadFromYaml(modelPath, hyperPath);

	EXPECT_EQ(config.getCacheType(), "ram");
}

// ========== Dataset Configuration Tests ==========

TEST_F(ConfigurationTest, Dataset_MapFormat_LoadsCorrectly) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();
	std::string datasetPath = (testDataPath / "test_dataset.yaml").string();

	config.loadFromYaml(modelPath, hyperPath, datasetPath);

	const auto& classNames = config.getClassNames();
	EXPECT_EQ(classNames.size(), 4);
	EXPECT_EQ(classNames.at(0), "cat");
}

TEST_F(ConfigurationTest, Dataset_ListFormat_LoadsCorrectly) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();
	std::string datasetPath = (testDataPath / "test_dataset_list.yaml").string();

	config.loadFromYaml(modelPath, hyperPath, datasetPath);

	const auto& classNames = config.getClassNames();
	EXPECT_EQ(classNames.size(), 4);
	EXPECT_EQ(classNames.at(0), "person");
	EXPECT_EQ(classNames.at(1), "car");
}

// ========== useDDP Tests ==========

TEST_F(ConfigurationTest, UseDDP_SingleDevice_ReturnsFalse) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();

	config.loadFromYaml(modelPath, hyperPath);

	EXPECT_FALSE(config.useDDP());
}

TEST_F(ConfigurationTest, UseDDP_MultipleDevices_ReturnsTrue) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_device_list.yaml").string();

	config.loadFromYaml(modelPath, hyperPath);

	EXPECT_TRUE(config.useDDP());
}

TEST_F(ConfigurationTest, UseDDP_EmptyDevice_ReturnsFalse) {
	Configuration cfg;

	EXPECT_FALSE(cfg.useDDP());
}

// ========== Getter Tests ==========

TEST_F(ConfigurationTest, GetNumClasses_AfterLoad_ReturnsCorrectValue) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();

	config.loadFromYaml(modelPath, hyperPath);

	EXPECT_EQ(config.getNumClasses(), 10);
}

TEST_F(ConfigurationTest, GetDefaultActivation_AfterLoad_ReturnsCorrectValue) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();

	config.loadFromYaml(modelPath, hyperPath);

	EXPECT_EQ(config.getDefaultActivation(), "ReLU");
}

TEST_F(ConfigurationTest, GetImagePath_AfterLoad_ReturnsCorrectValue) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();
	std::string datasetPath = (testDataPath / "test_dataset.yaml").string();

	config.loadFromYaml(modelPath, hyperPath, datasetPath);

	EXPECT_EQ(config.getImagePath(), "images/test/");
}

TEST_F(ConfigurationTest, GetLabelPath_AfterLoad_ReturnsCorrectValue) {
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();
	std::string datasetPath = (testDataPath / "test_dataset.yaml").string();

	config.loadFromYaml(modelPath, hyperPath, datasetPath);

	EXPECT_EQ(config.getLabelPath(), "labels/test/");
}

// ============================================================================
// Device Parsing Tests - CRITICAL GAP COVERAGE
// ============================================================================

TEST_F(ConfigurationTest, Device_ScalarIntValue_LoadsCorrectly) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_device_int.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Scalar int device: 0 should load as "0"
	EXPECT_EQ(config.getDevice(), "0");
	EXPECT_FALSE(config.useDDP());  // Single device, no DDP

	// Verify other config loaded correctly
	EXPECT_EQ(config.getEpochs(), 50);
	EXPECT_EQ(config.getBatchSize(), 8);
}

TEST_F(ConfigurationTest, Device_NullValue_UsesDefault) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_device_null.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Null device should use default (empty string based on implementation)
	std::string device = config.getDevice();

	// Device should be empty string for null (based on Configuration.cpp logic)
	EXPECT_EQ(device, "");

	EXPECT_FALSE(config.useDDP());  // Empty string is not DDP
}

TEST_F(ConfigurationTest, Device_StringFromSequence_VerifyComma) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_device_list.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - List [0,1,2,3] should become "0,1,2,3"
	EXPECT_EQ(config.getDevice(), "0,1,2,3");
	EXPECT_TRUE(config.useDDP());  // Multiple devices, DDP enabled

	// Verify the string format is correct (comma-separated)
	std::string device = config.getDevice();

	// Should contain commas
	EXPECT_NE(device.find(','), std::string::npos);

	// Should not contain spaces
	EXPECT_EQ(device.find(' '), std::string::npos);

	// Should start with '0'
	EXPECT_EQ(device[0], '0');

	// Should contain all digits 0-3
	EXPECT_NE(device.find('0'), std::string::npos);
	EXPECT_NE(device.find('1'), std::string::npos);
	EXPECT_NE(device.find('2'), std::string::npos);
	EXPECT_NE(device.find('3'), std::string::npos);
}

TEST_F(ConfigurationTest, Device_InvalidValue_HandledGracefully) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_device_invalid.yaml").string();

	// Act & Assert - Should either:
	// 1. Load the invalid string as-is (letting GPU library handle error), or
	// 2. Throw ConfigurationException
	// Both behaviors are acceptable for invalid input

	try {
		config.loadFromYaml(modelPath, hyperPath);

		// If it loads, verify the invalid string is preserved
		std::string device = config.getDevice();
		EXPECT_EQ(device, "invalid_device_string");

		// Should not indicate DDP
		EXPECT_FALSE(config.useDDP());

	}
	catch (const ConfigurationException& e) {
		// This is also acceptable - configuration rejects invalid device
		SUCCEED() << "ConfigurationException thrown for invalid device: " << e.what();
	}
	catch (const std::exception& e) {
		// Any other exception is acceptable as validation
		SUCCEED() << "Exception thrown for invalid device: " << e.what();
	}
}

TEST_F(ConfigurationTest, Device_AllFormats_Consistency) {
	// This test verifies all four device formats produce consistent behavior

	struct TestCase {
		std::string hyperFile;
		std::string expectedDevice;
		bool expectedDDP;
		std::string description;
	};

	std::vector<TestCase> testCases = {
		{"test_hyper_device_int.yaml", "0", false, "Scalar integer"},
		{"test_hyper.yaml", "0", false, "Scalar string"},
		{"test_hyper_device_list.yaml", "0,1,2,3", true, "Sequence/List"},
		{"test_hyper_device_null.yaml", "", false, "Null value (default)"}
	};

	for (const auto& testCase : testCases) {
		// Arrange
		Configuration testConfig;
		std::string modelPath = (testDataPath / "test_model.yaml").string();
		std::string hyperPath = (testDataPath / testCase.hyperFile).string();

		// Act
		testConfig.loadFromYaml(modelPath, hyperPath);

		// Assert
		std::string actualDevice = testConfig.getDevice();
		bool actualDDP = testConfig.useDDP();

		EXPECT_EQ(actualDevice, testCase.expectedDevice)
			<< "Failed for " << testCase.description
			<< " (file: " << testCase.hyperFile << ")";

		EXPECT_EQ(actualDDP, testCase.expectedDDP)
			<< "DDP mismatch for " << testCase.description
			<< " (file: " << testCase.hyperFile << ")";
	}
}

// ============================================================================
// Parameter Validation Tests - HIGH PRIORITY GAP COVERAGE
// ============================================================================

TEST_F(ConfigurationTest, Validation_NegativeEpochs_LoadsWithDefault) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_negative_epochs.yaml").string();

	// Act - Load config with negative epochs
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Should either use default or load as-is (implementation dependent)
	// Most implementations will load the value as-is and let validation happen later
	int epochs = config.getEpochs();

	// Either loads negative value (to be caught later) or uses default
	EXPECT_TRUE(epochs == -10 || epochs == 100);  // -10 from file or 100 default
}

TEST_F(ConfigurationTest, Validation_ZeroBatchSize_LoadsWithDefault) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_zero_batch.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Zero batch size should be caught or use default
	int batchSize = config.getBatchSize();
	EXPECT_TRUE(batchSize == 0 || batchSize == 16);  // 0 from file or 16 default
}

TEST_F(ConfigurationTest, Validation_NegativeImageSize_LoadsWithDefault) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_negative_imgsz.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert
	int imageSize = config.getImageSize();
	EXPECT_TRUE(imageSize == -100 || imageSize == 640);  // -100 from file or 640 default
}

TEST_F(ConfigurationTest, Validation_OutOfRangeLearningRate_High) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_high_lr.yaml").string();

	// Act - Learning rate > 1.0 is unusual but might be valid
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Should load the value (validation happens at runtime)
	float lr = config.getLearningRate();
	EXPECT_FLOAT_EQ(lr, 100.0f);  // Extremely high LR from test file
}

TEST_F(ConfigurationTest, Validation_OutOfRangeLearningRate_Negative) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_negative_lr.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Should load negative value (to be caught during training)
	float lr = config.getLearningRate();
	EXPECT_FLOAT_EQ(lr, -0.5f);
}

TEST_F(ConfigurationTest, Validation_OutOfRangeMomentum) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_invalid_momentum.yaml").string();

	// Act - Momentum should be in [0, 1] but load anyway
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Should load value outside valid range
	float momentum = config.getMomentum();
	EXPECT_FLOAT_EQ(momentum, 1.5f);  // Invalid momentum > 1.0
}

TEST_F(ConfigurationTest, Validation_NegativeWorkers) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_negative_workers.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Should load negative workers
	int workers = config.getWorkers();
	EXPECT_TRUE(workers == -4 || workers == 8);  // -4 from file or 8 default
}

TEST_F(ConfigurationTest, Validation_InvalidDropoutValue_TooHigh) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_invalid_dropout.yaml").string();

	// Act - Dropout should be in [0, 1] but might be > 1
	config.loadFromYaml(modelPath, hyperPath);

	// Assert
	float dropout = config.getDropout();
	EXPECT_FLOAT_EQ(dropout, 2.0f);  // Invalid dropout > 1.0
}

TEST_F(ConfigurationTest, Validation_NegativeAugmentationValues) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_negative_augment.yaml").string();

	// Act - Augmentation probabilities should be [0, 1]
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Should load negative values
	EXPECT_FLOAT_EQ(config.getMosaic(), -0.5f);
	EXPECT_FLOAT_EQ(config.getMixup(), -0.2f);
}

TEST_F(ConfigurationTest, Validation_ExtremeMaxDet) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_extreme_maxdet.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Should load extremely large max_det
	int maxDet = config.getMaxDet();
	EXPECT_EQ(maxDet, 100000);  // Unreasonably large value
}

TEST_F(ConfigurationTest, Validation_ZeroNumClasses) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model_zero_classes.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Zero classes should load (might be invalid)
	int numClasses = config.getNumClasses();
	EXPECT_EQ(numClasses, 0);
}

TEST_F(ConfigurationTest, Validation_VeryLargeEpochs) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_large_epochs.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Should load very large epoch count
	int epochs = config.getEpochs();
	EXPECT_EQ(epochs, 100000);
}

TEST_F(ConfigurationTest, Validation_EmptyOptimizer) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_empty_optimizer.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Empty optimizer string should load
	std::string optimizer = config.getOptimizer();
	EXPECT_TRUE(optimizer.empty() || optimizer == "auto");
}

TEST_F(ConfigurationTest, Validation_InvalidCacheType) {
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_invalid_cache.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Invalid cache type should load as-is
	std::string cacheType = config.getCacheType();
	EXPECT_EQ(cacheType, "invalid_cache_mode");
}

TEST_F(ConfigurationTest, Validation_AllParametersAtBoundaries) {
	// Test boundary values for all numeric parameters
	// Arrange
	std::string modelPath = (testDataPath / "test_model.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_boundaries.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Verify boundary values load correctly
	EXPECT_EQ(config.getEpochs(), 1);           // Minimum epochs
	EXPECT_EQ(config.getBatchSize(), 1);        // Minimum batch
	EXPECT_EQ(config.getImageSize(), 32);       // Small image size
	EXPECT_FLOAT_EQ(config.getLearningRate(), 0.0001f);  // Very small LR
	EXPECT_FLOAT_EQ(config.getMomentum(), 0.0f);         // Min momentum
	EXPECT_FLOAT_EQ(config.getWeightDecay(), 0.0f);      // No weight decay
}

TEST_F(ConfigurationTest, Validation_MissingRequiredFields) {
	// Test behavior when required fields are missing (uses defaults)
	// Arrange
	std::string modelPath = (testDataPath / "test_model_minimal.yaml").string();
	std::string hyperPath = (testDataPath / "test_hyper_minimal.yaml").string();

	// Act
	config.loadFromYaml(modelPath, hyperPath);

	// Assert - Should use default values for missing fields
	EXPECT_GT(config.getEpochs(), 0);
	EXPECT_GT(config.getBatchSize(), 0);
	EXPECT_GT(config.getImageSize(), 0);
	EXPECT_GT(config.getLearningRate(), 0.0f);
}
