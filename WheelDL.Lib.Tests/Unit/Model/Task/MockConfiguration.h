#pragma once

/**
 * @file MockConfiguration.h
 * @brief Mock configuration helpers for Model/Task tests
 *
 * Provides helper functions to create mock Configuration objects
 * for testing Task model classes.
 *
 * Phase 1 of Model_Task_Test_Plan.md
 */

#include "Config/Configuration.h"
#include "Utils/Common/Types.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <map>

namespace fs = std::filesystem;

namespace WheelDL {
namespace Test {

/**
 * @class MockConfigurationHelper
 * @brief Helper class for creating mock Configuration objects
 *
 * Creates temporary files and loads Configuration objects for testing.
 * All temporary files are cleaned up when the helper is destroyed.
 */
class MockConfigurationHelper {
public:
    /**
     * @brief Constructor - creates temp directory
     */
    MockConfigurationHelper() {
        _tempDir = (fs::temp_directory_path() / "WheelDL_TaskModel_Test").string();
        fs::remove_all(_tempDir);
        fs::create_directories(_tempDir);
    }

    /**
     * @brief Destructor - cleans up temp directory
     */
    ~MockConfigurationHelper() {
        try {
            fs::remove_all(_tempDir);
        }
        catch (...) {}
    }

    /**
     * @brief Create a mock Configuration for testing
     *
     * @param taskType Task type (CLASSIFICATION, DETECTION, etc.)
     * @param modelYamlPath Path to model YAML file (relative to Config/Valid/model/)
     * @param numClasses Number of classes (default 80)
     * @param imageSize Image size (default 640)
     * @return Shared pointer to Configuration
     */
    std::shared_ptr<Config::Configuration> createMockConfig(
        TaskType taskType,
        const std::string& modelYamlPath,
        int numClasses = 80,
        int imageSize = 640)
    {
        // Create dataset JSON with correct task type
        std::string datasetPath = createDatasetJson(taskType, numClasses);

        // Create hyperparameter YAML
        std::string hypPath = createHypYaml(imageSize);

        // Create configuration and load
        auto config = std::make_shared<Config::Configuration>();
        config->load(modelYamlPath, hypPath, datasetPath);

        return config;
    }

    /**
     * @brief Create a mock Configuration with custom hyperparameters
     *
     * @param taskType Task type
     * @param modelYamlPath Path to model YAML file
     * @param hypParams Map of hyperparameter names to values
     * @param numClasses Number of classes
     * @return Shared pointer to Configuration
     */
    std::shared_ptr<Config::Configuration> createMockConfigWithParams(
        TaskType taskType,
        const std::string& modelYamlPath,
        const std::map<std::string, std::string>& hypParams,
        int numClasses = 80)
    {
        // Create dataset JSON
        std::string datasetPath = createDatasetJson(taskType, numClasses);

        // Create hyperparameter YAML with custom params
        std::string hypPath = createHypYamlWithParams(hypParams);

        // Create and load configuration
        auto config = std::make_shared<Config::Configuration>();
        config->load(modelYamlPath, hypPath, datasetPath);

        return config;
    }

    /**
     * @brief Get the test config base path
     * @return Path to Config/Valid/ directory
     */
    std::string getTestConfigPath() const {
        return "Config/Valid/";
    }

    /**
     * @brief Get full path to a model YAML file
     * @param filename YAML filename (e.g., "cls_ResNet.yaml")
     * @return Full path to model YAML
     */
    std::string getModelYamlPath(const std::string& filename) const {
        return getTestConfigPath() + "model/" + filename;
    }

    /**
     * @brief Get full path to a hyperparameter YAML file
     * @param filename YAML filename (e.g., "cls_hyp_ResNet_256.yaml")
     * @return Full path to hyp YAML
     */
    std::string getHypYamlPath(const std::string& filename) const {
        return getTestConfigPath() + "hyp/" + filename;
    }

private:
    std::string _tempDir;
    int _fileCounter = 0;

    std::string getUniquePath(const std::string& extension) {
        return (fs::path(_tempDir) / ("temp_" + std::to_string(++_fileCounter) + extension)).string();
    }

    std::string taskTypeToJsonString(TaskType type) {
        switch (type) {
            case TaskType::CLASSIFICATION: return "classification";
            case TaskType::DETECTION: return "object_detection";
            case TaskType::OBB: return "obb";
            case TaskType::SEGMENTATION: return "segmentation";
            case TaskType::ANOMALY: return "anomaly_detection";
            default: return "unknown";
        }
    }

    std::string createDatasetJson(TaskType taskType, int numClasses) {
        std::string path = getUniquePath(".json");

        // Build categories array
        std::string categories = "\"class0\"";
        for (int i = 1; i < numClasses; ++i) {
            categories += ", \"class" + std::to_string(i) + "\"";
        }

        std::string content = R"({
    "header": {
        "type": ")" + taskTypeToJsonString(taskType) + R"(",
        "categories": [)" + categories + R"(]
    },
    "annotations": []
})";

        std::ofstream file(path);
        file << content;
        file.close();

        return path;
    }

    std::string createHypYaml(int imageSize) {
        std::string path = getUniquePath(".yaml");

        std::string content = R"(
# Test hyperparameters
epochs: 100
patience: 50
batch_size: 16
image_size: )" + std::to_string(imageSize) + R"(
device: "cpu"
workers: 0
optimizer: auto
seed: 42
deterministic: true

# Optimizer settings
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Loss gains
box: 7.5
cls: 0.5
dfl: 1.5

# Augmentation (disabled for testing)
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0
degrees: 0.0
translate: 0.0
scale: 0.0
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.0
mosaic: 0.0
)";

        std::ofstream file(path);
        file << content;
        file.close();

        return path;
    }

    std::string createHypYamlWithParams(const std::map<std::string, std::string>& params) {
        std::string path = getUniquePath(".yaml");

        std::string content = "# Test hyperparameters\n";
        for (const auto& [key, value] : params) {
            content += key + ": " + value + "\n";
        }

        std::ofstream file(path);
        file << content;
        file.close();

        return path;
    }
};

// ========== Convenience Functions ==========

/**
 * @brief Create a mock Configuration for classification testing
 * @param numClasses Number of classes (default 10)
 * @param imageSize Image size (default 224)
 * @return Shared pointer to Configuration
 */
inline std::shared_ptr<Config::Configuration> createClassificationConfig(
    int numClasses = 10,
    int imageSize = 224)
{
    MockConfigurationHelper helper;
    return helper.createMockConfig(
        TaskType::CLASSIFICATION,
        helper.getModelYamlPath("cls_ConvNext.yaml"),
        numClasses,
        imageSize
    );
}

/**
 * @brief Create a mock Configuration for detection testing
 * @param numClasses Number of classes (default 80)
 * @param imageSize Image size (default 640)
 * @return Shared pointer to Configuration
 */
inline std::shared_ptr<Config::Configuration> createDetectionConfig(
    int numClasses = 80,
    int imageSize = 640)
{
    MockConfigurationHelper helper;
    return helper.createMockConfig(
        TaskType::DETECTION,
        helper.getModelYamlPath("det_yoloxs.yaml"),
        numClasses,
        imageSize
    );
}

/**
 * @brief Create a mock Configuration for OBB testing
 * @param numClasses Number of classes (default 15)
 * @param imageSize Image size (default 640)
 * @return Shared pointer to Configuration
 */
inline std::shared_ptr<Config::Configuration> createOBBConfig(
    int numClasses = 15,
    int imageSize = 640)
{
    MockConfigurationHelper helper;
    return helper.createMockConfig(
        TaskType::OBB,
        helper.getModelYamlPath("obb_yoloxs.yaml"),
        numClasses,
        imageSize
    );
}

/**
 * @brief Create a mock Configuration for segmentation testing
 * @param numClasses Number of classes (default 80)
 * @param imageSize Image size (default 640)
 * @return Shared pointer to Configuration
 */
inline std::shared_ptr<Config::Configuration> createSegmentationConfig(
    int numClasses = 80,
    int imageSize = 640)
{
    MockConfigurationHelper helper;
    return helper.createMockConfig(
        TaskType::SEGMENTATION,
        helper.getModelYamlPath("seg_yjnet.yaml"),
        numClasses,
        imageSize
    );
}

/**
 * @brief Create a mock Configuration for anomaly (EfficientAD) testing
 * @param numClasses Number of classes (default 1)
 * @param imageSize Image size (default 256)
 * @return Shared pointer to Configuration
 */
inline std::shared_ptr<Config::Configuration> createAnomalyEfficientADConfig(
    int numClasses = 1,
    int imageSize = 256)
{
    MockConfigurationHelper helper;
    return helper.createMockConfig(
        TaskType::ANOMALY,
        helper.getModelYamlPath("ano_EfficientAD.yaml"),
        numClasses,
        imageSize
    );
}

/**
 * @brief Create a mock Configuration for anomaly (PatchCore) testing
 * @param numClasses Number of classes (default 1)
 * @param imageSize Image size (default 256)
 * @return Shared pointer to Configuration
 */
inline std::shared_ptr<Config::Configuration> createAnomalyPatchCoreConfig(
    int numClasses = 1,
    int imageSize = 256)
{
    MockConfigurationHelper helper;
    return helper.createMockConfig(
        TaskType::ANOMALY,
        helper.getModelYamlPath("ano_PatchCore.yaml"),
        numClasses,
        imageSize
    );
}

} // namespace Test
} // namespace WheelDL
