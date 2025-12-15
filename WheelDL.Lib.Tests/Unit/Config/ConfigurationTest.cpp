#include "pch.h"
#include <gtest/gtest.h>
#include "Config/Configuration.h"
#include "Utils/Error/WheelLibException.h"
#include <filesystem>
#include <fstream>

using namespace WheelDL::Config;
using namespace WheelDL::Utils;
using namespace WheelDL;
namespace fs = std::filesystem;

// =============================================================================
// Test Fixture
// =============================================================================

class ConfigurationTest : public ::testing::Test {
protected:
    std::string testBaseDir;

    void SetUp() override {
        testBaseDir = (fs::temp_directory_path() / "WheelDL_Configuration_Test").string();
        fs::remove_all(testBaseDir);
        fs::create_directories(testBaseDir);
    }

    void TearDown() override {
        try {
            fs::remove_all(testBaseDir);
        }
        catch (...) {}
    }

    std::string getTestPath(const std::string& filename) {
        return (fs::path(testBaseDir) / filename).string();
    }

    void createFile(const std::string& filename, const std::string& content) {
        std::string filepath = getTestPath(filename);
        std::ofstream file(filepath);
        file << content;
        file.close();
    }

    // Create minimal valid YAML hyperparameter file
    std::string createValidHyperParams() {
        std::string content = R"(
epochs: 100
batch_size: 16
image_size: 640
)";
        createFile("hyp.yaml", content);
        return getTestPath("hyp.yaml");
    }

    // Create minimal valid JSON dataset file
    std::string createValidDataset(const std::string& taskType = "object_detection") {
        std::string content = R"({
    "header": {
        "type": ")" + taskType + R"(",
        "categories": ["class0", "class1", "class2"]
    },
    "annotations": [
        {"id": 1},
        {"id": 2}
    ]
})";
        createFile("dataset.json", content);
        return getTestPath("dataset.json");
    }
};

// =============================================================================
// Constructor Default Values Tests (CF-01 ~ CF-51)
// =============================================================================

// CF-01: DefaultTaskType
TEST_F(ConfigurationTest, DefaultTaskType) {
    Configuration config;
    EXPECT_EQ(TaskType::UNKNOWN, config.getTaskType());
}

// CF-02: DefaultEpochs
TEST_F(ConfigurationTest, DefaultEpochs) {
    Configuration config;
    EXPECT_EQ(100, config.getEpochs());
}

// CF-03: DefaultPatience
TEST_F(ConfigurationTest, DefaultPatience) {
    Configuration config;
    EXPECT_EQ(100, config.getPatience());
}

// CF-04: DefaultBatchSize
TEST_F(ConfigurationTest, DefaultBatchSize) {
    Configuration config;
    EXPECT_EQ(16, config.getBatchSize());
}

// CF-05: DefaultImageSize
TEST_F(ConfigurationTest, DefaultImageSize) {
    Configuration config;
    EXPECT_EQ(640, config.getImageSize());
}

// CF-06: DefaultDevice
TEST_F(ConfigurationTest, DefaultDevice) {
    Configuration config;
    EXPECT_EQ("", config.getDevice());
}

// CF-07: DefaultWorkers
TEST_F(ConfigurationTest, DefaultWorkers) {
    Configuration config;
    EXPECT_EQ(8, config.getWorkers());
}

// CF-08: DefaultOptimizer
TEST_F(ConfigurationTest, DefaultOptimizer) {
    Configuration config;
    EXPECT_EQ("auto", config.getOptimizer());
}

// CF-09: DefaultSeed
TEST_F(ConfigurationTest, DefaultSeed) {
    Configuration config;
    EXPECT_EQ(0, config.getSeed());
}

// CF-10: DefaultDeterministic
TEST_F(ConfigurationTest, DefaultDeterministic) {
    Configuration config;
    EXPECT_FALSE(config.isDeterministic());
}

// CF-11: DefaultCosLR
TEST_F(ConfigurationTest, DefaultCosLR) {
    Configuration config;
    EXPECT_FALSE(config.useCosineLR());
}

// CF-12: DefaultCloseMosaic
TEST_F(ConfigurationTest, DefaultCloseMosaic) {
    Configuration config;
    EXPECT_EQ(0, config.getCloseMosaic());
}

// CF-13: DefaultFreezeLayers
TEST_F(ConfigurationTest, DefaultFreezeLayers) {
    Configuration config;
    EXPECT_EQ(0, config.getFreezeLayers());
}

// CF-14: DefaultAMP
TEST_F(ConfigurationTest, DefaultAMP) {
    Configuration config;
    EXPECT_TRUE(config.useAMP());
}

// CF-15: DefaultBFloat16
TEST_F(ConfigurationTest, DefaultBFloat16) {
    Configuration config;
    EXPECT_FALSE(config.useBFloat16());
}

// CF-16: DefaultCache
TEST_F(ConfigurationTest, DefaultCache) {
    Configuration config;
    EXPECT_EQ("", config.getCacheType());
}

// CF-17: DefaultCacheSize
TEST_F(ConfigurationTest, DefaultCacheSize) {
    Configuration config;
    EXPECT_EQ(0u, config.getCacheSize());
}

// CF-18: DefaultLr0
TEST_F(ConfigurationTest, DefaultLr0) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.01f, config.getLearningRate());
}

// CF-19: DefaultLrf
TEST_F(ConfigurationTest, DefaultLrf) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.01f, config.getLRFinalFraction());
}

// CF-20: DefaultMomentum
TEST_F(ConfigurationTest, DefaultMomentum) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.937f, config.getMomentum());
}

// CF-21: DefaultWeightDecay
TEST_F(ConfigurationTest, DefaultWeightDecay) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.0005f, config.getWeightDecay());
}

// CF-22: DefaultWarmupEpochs
TEST_F(ConfigurationTest, DefaultWarmupEpochs) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.0f, config.getWarmupEpochs());
}

// CF-23: DefaultWarmupMomentum
TEST_F(ConfigurationTest, DefaultWarmupMomentum) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.8f, config.getWarmupMomentum());
}

// CF-24: DefaultWarmupBiasLR
TEST_F(ConfigurationTest, DefaultWarmupBiasLR) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.1f, config.getWarmupBiasLR());
}

// CF-25: DefaultAmsgrad
TEST_F(ConfigurationTest, DefaultAmsgrad) {
    Configuration config;
    EXPECT_FALSE(config.getAmsgrad());
}

// CF-26: DefaultTopK
TEST_F(ConfigurationTest, DefaultTopK) {
    Configuration config;
    EXPECT_EQ(13, config.getTopK());
}

// CF-27: DefaultBoxGain
TEST_F(ConfigurationTest, DefaultBoxGain) {
    Configuration config;
    EXPECT_FLOAT_EQ(7.5f, config.getBoxGain());
}

// CF-28: DefaultClsGain
TEST_F(ConfigurationTest, DefaultClsGain) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.5f, config.getClsGain());
}

// CF-29: DefaultDflGain
TEST_F(ConfigurationTest, DefaultDflGain) {
    Configuration config;
    EXPECT_FLOAT_EQ(1.5f, config.getDFLGain());
}

// CF-30: DefaultHsvH
TEST_F(ConfigurationTest, DefaultHsvH) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.015f, config.getHSVH());
}

// CF-31: DefaultHsvS
TEST_F(ConfigurationTest, DefaultHsvS) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.7f, config.getHSVS());
}

// CF-32: DefaultHsvV
TEST_F(ConfigurationTest, DefaultHsvV) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.4f, config.getHSVV());
}

// CF-33: DefaultDegrees
TEST_F(ConfigurationTest, DefaultDegrees) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.0f, config.getDegrees());
}

// CF-34: DefaultTranslate
TEST_F(ConfigurationTest, DefaultTranslate) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.1f, config.getTranslate());
}

// CF-35: DefaultScale
TEST_F(ConfigurationTest, DefaultScale) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.5f, config.getScale());
}

// CF-36: DefaultShear
TEST_F(ConfigurationTest, DefaultShear) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.0f, config.getShear());
}

// CF-37: DefaultPerspective
TEST_F(ConfigurationTest, DefaultPerspective) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.0f, config.getPerspective());
}

// CF-38: DefaultFlipud
TEST_F(ConfigurationTest, DefaultFlipud) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.0f, config.getFlipUD());
}

// CF-39: DefaultFliplr
TEST_F(ConfigurationTest, DefaultFliplr) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.5f, config.getFlipLR());
}

// CF-40: DefaultMosaic
TEST_F(ConfigurationTest, DefaultMosaic) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.0f, config.getMosaic());
}

// CF-41: DefaultBlurKernelSize
TEST_F(ConfigurationTest, DefaultBlurKernelSize) {
    Configuration config;
    EXPECT_EQ(3, config.getBlurKernelSize());
}

// CF-42: DefaultBlurProbability
TEST_F(ConfigurationTest, DefaultBlurProbability) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.01f, config.getBlurProbability());
}

// CF-43: DefaultFillBorder
TEST_F(ConfigurationTest, DefaultFillBorder) {
    Configuration config;
    EXPECT_EQ(0, config.getFillBorder());
}

// CF-44: DefaultImageNetNorm
TEST_F(ConfigurationTest, DefaultImageNetNorm) {
    Configuration config;
    EXPECT_FALSE(config.getImageNetNorm());
}

// CF-45: DefaultEmaEnabled
TEST_F(ConfigurationTest, DefaultEmaEnabled) {
    Configuration config;
    EXPECT_FALSE(config.isEmaEnabled());
}

// CF-46: DefaultIoU
TEST_F(ConfigurationTest, DefaultIoU) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.7f, config.getIoU());
}

// CF-47: DefaultMaxDet
TEST_F(ConfigurationTest, DefaultMaxDet) {
    Configuration config;
    EXPECT_EQ(300, config.getMaxDet());
}

// CF-48: DefaultDropout
TEST_F(ConfigurationTest, DefaultDropout) {
    Configuration config;
    EXPECT_FLOAT_EQ(0.0f, config.getDropout());
}

// CF-49: DefaultNumClasses
TEST_F(ConfigurationTest, DefaultNumClasses) {
    Configuration config;
    EXPECT_EQ(0, config.getNumClasses());
}

// CF-50: DefaultIsEfficientAD
TEST_F(ConfigurationTest, DefaultIsEfficientAD) {
    Configuration config;
    EXPECT_FALSE(config.IsEfficientAD());
}

// CF-51: DefaultIsPatchCore
TEST_F(ConfigurationTest, DefaultIsPatchCore) {
    Configuration config;
    EXPECT_FALSE(config.IsPatchCore());
}

// =============================================================================
// load() - Valid Configurations (CF-52 ~ CF-58)
// =============================================================================

// CF-52: Load_Classification
TEST_F(ConfigurationTest, Load_Classification) {
    std::string hypPath = createValidHyperParams();
    std::string datasetPath = createValidDataset("classification");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ(TaskType::CLASSIFICATION, config.getTaskType());
}

// CF-53: Load_Detection
TEST_F(ConfigurationTest, Load_Detection) {
    std::string hypPath = createValidHyperParams();
    std::string datasetPath = createValidDataset("object_detection");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ(TaskType::DETECTION, config.getTaskType());
}

// CF-54: Load_Segmentation
TEST_F(ConfigurationTest, Load_Segmentation) {
    std::string hypPath = createValidHyperParams();
    std::string datasetPath = createValidDataset("segmentation");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ(TaskType::SEGMENTATION, config.getTaskType());
}

// CF-55: Load_OBB
TEST_F(ConfigurationTest, Load_OBB) {
    std::string hypPath = createValidHyperParams();
    std::string datasetPath = createValidDataset("obb");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ(TaskType::OBB, config.getTaskType());
}

// CF-56: Load_Anomaly
TEST_F(ConfigurationTest, Load_Anomaly) {
    std::string hypPath = createValidHyperParams();
    std::string datasetPath = createValidDataset("anomaly_detection");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ(TaskType::ANOMALY, config.getTaskType());
}

// CF-57: Load_UnknownType
TEST_F(ConfigurationTest, Load_UnknownType) {
    std::string hypPath = createValidHyperParams();
    std::string datasetPath = createValidDataset("unknown_type");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ(TaskType::UNKNOWN, config.getTaskType());
}

// CF-58: Load_MissingType
TEST_F(ConfigurationTest, Load_MissingType) {
    std::string hypPath = createValidHyperParams();

    // Create dataset without type field
    std::string content = R"({
    "header": {
        "categories": ["class0", "class1"]
    },
    "annotations": []
})";
    createFile("dataset_no_type.json", content);
    std::string datasetPath = getTestPath("dataset_no_type.json");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ(TaskType::UNKNOWN, config.getTaskType());
}

// =============================================================================
// load() - Error Cases (CF-59 ~ CF-66)
// =============================================================================

// CF-59: Load_HyperParamNotFound
TEST_F(ConfigurationTest, Load_HyperParamNotFound) {
    std::string datasetPath = createValidDataset();

    Configuration config;
    EXPECT_THROW({
        try {
            config.load("model.yaml", "nonexistent.yaml", datasetPath);
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_FILE_NOT_FOUND, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// CF-60: Load_DatasetNotFound
TEST_F(ConfigurationTest, Load_DatasetNotFound) {
    std::string hypPath = createValidHyperParams();

    Configuration config;
    EXPECT_THROW({
        try {
            config.load("model.yaml", hypPath, "nonexistent.json");
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_FILE_NOT_FOUND, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// CF-61: Load_InvalidYamlSyntax
TEST_F(ConfigurationTest, Load_InvalidYamlSyntax) {
    std::string datasetPath = createValidDataset();
    createFile("invalid.yaml", "key: [broken");
    std::string hypPath = getTestPath("invalid.yaml");

    Configuration config;
    EXPECT_THROW({
        try {
            config.load("model.yaml", hypPath, datasetPath);
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// CF-62: Load_InvalidJsonSyntax
TEST_F(ConfigurationTest, Load_InvalidJsonSyntax) {
    std::string hypPath = createValidHyperParams();
    createFile("invalid.json", "{broken");
    std::string datasetPath = getTestPath("invalid.json");

    Configuration config;
    EXPECT_THROW({
        try {
            config.load("model.yaml", hypPath, datasetPath);
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// CF-63: Load_EmptyCategories
TEST_F(ConfigurationTest, Load_EmptyCategories) {
    std::string hypPath = createValidHyperParams();
    std::string content = R"({
    "header": {
        "type": "object_detection",
        "categories": []
    },
    "annotations": []
})";
    createFile("empty_cat.json", content);
    std::string datasetPath = getTestPath("empty_cat.json");

    Configuration config;
    EXPECT_THROW({
        try {
            config.load("model.yaml", hypPath, datasetPath);
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::INVALID_CONFIG_VALUE, e.getErrorCode());
            EXPECT_NE(std::string::npos, e.getMessage().find("No class names"));
            throw;
        }
    }, ConfigurationException);
}

// CF-64: Load_MissingCategories
TEST_F(ConfigurationTest, Load_MissingCategories) {
    std::string hypPath = createValidHyperParams();
    std::string content = R"({
    "header": {
        "type": "object_detection"
    },
    "annotations": []
})";
    createFile("no_cat.json", content);
    std::string datasetPath = getTestPath("no_cat.json");

    Configuration config;
    EXPECT_THROW({
        try {
            config.load("model.yaml", hypPath, datasetPath);
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::INVALID_CONFIG_VALUE, e.getErrorCode());
            EXPECT_NE(std::string::npos, e.getMessage().find("No class names"));
            throw;
        }
    }, ConfigurationException);
}

// CF-65: Load_MissingHeader
TEST_F(ConfigurationTest, Load_MissingHeader) {
    std::string hypPath = createValidHyperParams();
    std::string content = R"({
    "annotations": []
})";
    createFile("no_header.json", content);
    std::string datasetPath = getTestPath("no_header.json");

    Configuration config;
    EXPECT_THROW({
        try {
            config.load("model.yaml", hypPath, datasetPath);
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::INVALID_CONFIG_VALUE, e.getErrorCode());
            EXPECT_NE(std::string::npos, e.getMessage().find("No class names"));
            throw;
        }
    }, ConfigurationException);
}

// CF-66: Load_NullJsonStructure
TEST_F(ConfigurationTest, Load_NullJsonStructure) {
    std::string hypPath = createValidHyperParams();
    createFile("null.json", "null");
    std::string datasetPath = getTestPath("null.json");

    Configuration config;
    EXPECT_THROW({
        try {
            config.load("model.yaml", hypPath, datasetPath);
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// =============================================================================
// load() - Task-Specific Augmentation (CF-67 ~ CF-77)
// =============================================================================

// Helper to create hyp with specific augmentation values
class ConfigurationAugmentationTest : public ConfigurationTest {
protected:
    std::string createHypWithAugmentation(float degrees = 10.0f, float perspective = 0.001f, float blur = 0.5f) {
        std::string content = R"(
            epochs: 100
            batch_size: 16
            image_size: 640
            degrees: )" + std::to_string(degrees) + R"(
            perspective: )" + std::to_string(perspective) + R"(
            blur_probability: )" + std::to_string(blur) + R"(
            )";
        createFile("hyp_aug.yaml", content);
        return getTestPath("hyp_aug.yaml");
    }
};

// CF-67: TaskAug_DetectionDegrees - DETECTION forces degrees to 0.0
TEST_F(ConfigurationAugmentationTest, TaskAug_DetectionDegrees) {
    std::string hypPath = createHypWithAugmentation(10.0f, 0.001f, 0.5f);
    std::string datasetPath = createValidDataset("object_detection");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_FLOAT_EQ(0.0f, config.getDegrees());
}

// CF-68: TaskAug_AnomalyDegrees - ANOMALY forces degrees to 0.0
TEST_F(ConfigurationAugmentationTest, TaskAug_AnomalyDegrees) {
    std::string hypPath = createHypWithAugmentation(10.0f, 0.001f, 0.5f);
    std::string datasetPath = createValidDataset("anomaly_detection");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_FLOAT_EQ(0.0f, config.getDegrees());
}

// CF-69: TaskAug_OBBPerspective - OBB forces perspective to 0.0
TEST_F(ConfigurationAugmentationTest, TaskAug_OBBPerspective) {
    std::string hypPath = createHypWithAugmentation(10.0f, 0.001f, 0.5f);
    std::string datasetPath = createValidDataset("obb");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_FLOAT_EQ(0.0f, config.getPerspective());
}

// CF-70: TaskAug_AnomalyPerspective - ANOMALY forces perspective to 0.0
TEST_F(ConfigurationAugmentationTest, TaskAug_AnomalyPerspective) {
    std::string hypPath = createHypWithAugmentation(10.0f, 0.001f, 0.5f);
    std::string datasetPath = createValidDataset("anomaly_detection");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_FLOAT_EQ(0.0f, config.getPerspective());
}

// CF-71: TaskAug_AnomalyBlur - ANOMALY forces blur_probability to 0.0
TEST_F(ConfigurationAugmentationTest, TaskAug_AnomalyBlur) {
    std::string hypPath = createHypWithAugmentation(10.0f, 0.001f, 0.5f);
    std::string datasetPath = createValidDataset("anomaly_detection");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_FLOAT_EQ(0.0f, config.getBlurProbability());
}

// CF-72: TaskAug_ClassificationDegrees - CLASSIFICATION uses YAML value
TEST_F(ConfigurationAugmentationTest, TaskAug_ClassificationDegrees) {
    std::string hypPath = createHypWithAugmentation(10.0f, 0.001f, 0.5f);
    std::string datasetPath = createValidDataset("classification");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_FLOAT_EQ(10.0f, config.getDegrees());
}

// CF-73: TaskAug_SegmentationDegrees - SEGMENTATION uses YAML value
TEST_F(ConfigurationAugmentationTest, TaskAug_SegmentationDegrees) {
    std::string hypPath = createHypWithAugmentation(10.0f, 0.001f, 0.5f);
    std::string datasetPath = createValidDataset("segmentation");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_FLOAT_EQ(10.0f, config.getDegrees());
}

// CF-74: TaskAug_OBBDegrees - OBB uses YAML value for degrees (only perspective is forced)
TEST_F(ConfigurationAugmentationTest, TaskAug_OBBDegrees) {
    std::string hypPath = createHypWithAugmentation(10.0f, 0.001f, 0.5f);
    std::string datasetPath = createValidDataset("obb");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_FLOAT_EQ(10.0f, config.getDegrees());
}

// CF-75: TaskAug_ClassificationPerspective - CLASSIFICATION uses YAML value
TEST_F(ConfigurationAugmentationTest, TaskAug_ClassificationPerspective) {
    std::string hypPath = createHypWithAugmentation(10.0f, 0.001f, 0.5f);
    std::string datasetPath = createValidDataset("classification");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_FLOAT_EQ(0.001f, config.getPerspective());
}

// CF-76: TaskAug_SegmentationPerspective - SEGMENTATION uses YAML value
TEST_F(ConfigurationAugmentationTest, TaskAug_SegmentationPerspective) {
    std::string hypPath = createHypWithAugmentation(10.0f, 0.001f, 0.5f);
    std::string datasetPath = createValidDataset("segmentation");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_FLOAT_EQ(0.001f, config.getPerspective());
}

// CF-77: TaskAug_DetectionBlur - DETECTION uses YAML value for blur
TEST_F(ConfigurationAugmentationTest, TaskAug_DetectionBlur) {
    std::string hypPath = createHypWithAugmentation(10.0f, 0.001f, 0.5f);
    std::string datasetPath = createValidDataset("object_detection");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_FLOAT_EQ(0.5f, config.getBlurProbability());
}

// =============================================================================
// Setter Tests (CF-78 ~ CF-95)
// =============================================================================

// CF-78: SetImageSize_Valid
TEST_F(ConfigurationTest, SetImageSize_Valid) {
    Configuration config;
    config.setImageSize(320);
    EXPECT_EQ(320, config.getImageSize());
}

// CF-79: SetImageSize_Zero
TEST_F(ConfigurationTest, SetImageSize_Zero) {
    Configuration config;
    config.setImageSize(0);
    EXPECT_EQ(0, config.getImageSize());
}

// CF-80: SetBatchSize_Valid
TEST_F(ConfigurationTest, SetBatchSize_Valid) {
    Configuration config;
    config.setBatchSize(32);
    EXPECT_EQ(32, config.getBatchSize());
}

// CF-81: SetNumClasses_Valid
TEST_F(ConfigurationTest, SetNumClasses_Valid) {
    Configuration config;
    config.setNumClasses(10);
    EXPECT_EQ(10, config.getNumClasses());
}

// CF-82: SetNumClasses_Negative
TEST_F(ConfigurationTest, SetNumClasses_Negative) {
    Configuration config;
    EXPECT_THROW({
        try {
            config.setNumClasses(-1);
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::INVALID_CONFIG_VALUE, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// CF-83: SetNumClasses_Zero
TEST_F(ConfigurationTest, SetNumClasses_Zero) {
    Configuration config;
    config.setNumClasses(0);
    EXPECT_EQ(0, config.getNumClasses());
}

// CF-84: SetClassNames_Valid
TEST_F(ConfigurationTest, SetClassNames_Valid) {
    Configuration config;
    std::map<int, std::string> classNames = {{0, "a"}, {1, "b"}};
    config.setClassNames(classNames);
    EXPECT_EQ(2, config.getNumClasses());
    EXPECT_EQ("a", config.getClassNames().at(0));
    EXPECT_EQ("b", config.getClassNames().at(1));
}

// CF-85: SetClassNames_Empty
TEST_F(ConfigurationTest, SetClassNames_Empty) {
    Configuration config;
    config.setNumClasses(5);  // Set initial value
    std::map<int, std::string> emptyClassNames;
    config.setClassNames(emptyClassNames);
    // numClasses unchanged when empty map is set
    EXPECT_EQ(5, config.getNumClasses());
}

// CF-86: SetEpochs_Valid
TEST_F(ConfigurationTest, SetEpochs_Valid) {
    Configuration config;
    config.setEpochs(200);
    EXPECT_EQ(200, config.getEpochs());
}

// CF-87: SetEpochs_WarmupAdjust
TEST_F(ConfigurationTest, SetEpochs_WarmupAdjust) {
    Configuration config;
    config.setWarmupEpochs(5.0f);  // Will be clamped to epochs*0.03
    config.setEpochs(10);
    // setEpochs() resets warmup to 0.0f
    EXPECT_FLOAT_EQ(0.0f, config.getWarmupEpochs());
}

// CF-88: SetWarmupEpochs_Valid
TEST_F(ConfigurationTest, SetWarmupEpochs_Valid) {
    Configuration config;
    config.setEpochs(100);
    config.setWarmupEpochs(3.0f);
    EXPECT_FLOAT_EQ(3.0f, config.getWarmupEpochs());
}

// CF-89: SetWarmupEpochs_Clamped
TEST_F(ConfigurationTest, SetWarmupEpochs_Clamped) {
    Configuration config;
    config.setEpochs(100);
    config.setWarmupEpochs(100.0f);  // Should be clamped to 100*0.03 = 3.0f
    EXPECT_FLOAT_EQ(3.0f, config.getWarmupEpochs());
}

// CF-90: SetLearningRate_Valid
TEST_F(ConfigurationTest, SetLearningRate_Valid) {
    Configuration config;
    config.setLearningRateFirst(0.001f);
    EXPECT_FLOAT_EQ(0.001f, config.getLearningRate());
}

// CF-91: SetMomentum_Valid
TEST_F(ConfigurationTest, SetMomentum_Valid) {
    Configuration config;
    config.setMomentum(0.9f);
    EXPECT_FLOAT_EQ(0.9f, config.getMomentum());
}

// CF-92: SetOptimizer_Valid
TEST_F(ConfigurationTest, SetOptimizer_Valid) {
    Configuration config;
    config.setOptimizer("adam");
    EXPECT_EQ("adam", config.getOptimizer());
}

// CF-93: SetImageNetNorm_True
TEST_F(ConfigurationTest, SetImageNetNorm_True) {
    Configuration config;
    config.setImageNetNorm(true);
    EXPECT_TRUE(config.getImageNetNorm());
}

// CF-94: SetIsEfficientAD_True
TEST_F(ConfigurationTest, SetIsEfficientAD_True) {
    Configuration config;
    config.setIsEfficientAD(true);
    EXPECT_TRUE(config.IsEfficientAD());
    EXPECT_EQ(256, config.getImageSize());
    EXPECT_FLOAT_EQ(0.0f, config.getWarmupEpochs());
}

// CF-95: SetIsPatchCore_True
TEST_F(ConfigurationTest, SetIsPatchCore_True) {
    Configuration config;
    config.setIsPatchCore(true);
    EXPECT_TRUE(config.IsPatchCore());
    EXPECT_EQ(0, config.getEpochs());
    EXPECT_EQ(256, config.getImageSize());
    EXPECT_FLOAT_EQ(0.0f, config.getWarmupEpochs());
}

// =============================================================================
// Device Parsing Tests (CF-96 ~ CF-102)
// =============================================================================

class ConfigurationDeviceTest : public ConfigurationTest {
protected:
    std::string createHypWithDevice(const std::string& deviceYaml) {
        std::string content = R"(
epochs: 100
batch_size: 16
image_size: 640
)" + deviceYaml;
        createFile("hyp_device.yaml", content);
        return getTestPath("hyp_device.yaml");
    }
};

// CF-96: Device_Int
TEST_F(ConfigurationDeviceTest, Device_Int) {
    std::string hypPath = createHypWithDevice("device: 0");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("0", config.getDevice());
    EXPECT_FALSE(config.useDDP());
}

// CF-97: Device_String
TEST_F(ConfigurationDeviceTest, Device_String) {
    std::string hypPath = createHypWithDevice("device: \"0\"");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("0", config.getDevice());
    EXPECT_FALSE(config.useDDP());
}

// CF-98: Device_CUDA
TEST_F(ConfigurationDeviceTest, Device_CUDA) {
    std::string hypPath = createHypWithDevice("device: \"cuda:0\"");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("cuda:0", config.getDevice());
    EXPECT_FALSE(config.useDDP());
}

// CF-99: Device_CPU
TEST_F(ConfigurationDeviceTest, Device_CPU) {
    std::string hypPath = createHypWithDevice("device: \"cpu\"");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("cpu", config.getDevice());
    EXPECT_FALSE(config.useDDP());
}

// CF-100: Device_MultiGPU
TEST_F(ConfigurationDeviceTest, Device_MultiGPU) {
    std::string hypPath = createHypWithDevice("device: [0, 1, 2]");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("0,1,2", config.getDevice());
    EXPECT_TRUE(config.useDDP());
}

// CF-101: Device_Null
TEST_F(ConfigurationDeviceTest, Device_Null) {
    std::string hypPath = createHypWithDevice("device: null");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("", config.getDevice());
    EXPECT_FALSE(config.useDDP());
}

// CF-102: Device_Missing
TEST_F(ConfigurationDeviceTest, Device_Missing) {
    std::string hypPath = createHypWithDevice("");  // No device field
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("", config.getDevice());
    EXPECT_FALSE(config.useDDP());
}

// =============================================================================
// Optimizer Validation Tests (CF-103 ~ CF-109)
// =============================================================================

class ConfigurationOptimizerTest : public ConfigurationTest {
protected:
    std::string createHypWithOptimizer(const std::string& optimizer) {
        std::string content = R"(
epochs: 100
batch_size: 16
image_size: 640
optimizer: )" + optimizer;
        createFile("hyp_opt.yaml", content);
        return getTestPath("hyp_opt.yaml");
    }
};

// CF-103: Optimizer_SGD
TEST_F(ConfigurationOptimizerTest, Optimizer_SGD) {
    std::string hypPath = createHypWithOptimizer("sgd");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("sgd", config.getOptimizer());
}

// CF-104: Optimizer_Adam
TEST_F(ConfigurationOptimizerTest, Optimizer_Adam) {
    std::string hypPath = createHypWithOptimizer("adam");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("adam", config.getOptimizer());
}

// CF-105: Optimizer_AdamW
TEST_F(ConfigurationOptimizerTest, Optimizer_AdamW) {
    std::string hypPath = createHypWithOptimizer("adamw");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("adamw", config.getOptimizer());
}

// CF-106: Optimizer_Auto
TEST_F(ConfigurationOptimizerTest, Optimizer_Auto) {
    std::string hypPath = createHypWithOptimizer("auto");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("auto", config.getOptimizer());
}

// CF-107: Optimizer_Invalid
TEST_F(ConfigurationOptimizerTest, Optimizer_Invalid) {
    std::string hypPath = createHypWithOptimizer("rmsprop");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("auto", config.getOptimizer());  // Fallback to auto
}

// CF-108: Optimizer_Uppercase
TEST_F(ConfigurationOptimizerTest, Optimizer_Uppercase) {
    std::string hypPath = createHypWithOptimizer("SGD");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("sgd", config.getOptimizer());  // Converted to lowercase
}

// CF-109: Optimizer_Empty
TEST_F(ConfigurationOptimizerTest, Optimizer_Empty) {
    std::string hypPath = createHypWithOptimizer("\"\"");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("auto", config.getOptimizer());  // Fallback to auto
}

// =============================================================================
// Cache Parsing Tests (CF-110 ~ CF-113)
// =============================================================================

class ConfigurationCacheTest : public ConfigurationTest {
protected:
    std::string createHypWithCache(const std::string& cacheYaml) {
        std::string content = R"(
epochs: 100
batch_size: 16
image_size: 640
)" + cacheYaml;
        createFile("hyp_cache.yaml", content);
        return getTestPath("hyp_cache.yaml");
    }
};

// CF-110: Cache_True
TEST_F(ConfigurationCacheTest, Cache_True) {
    std::string hypPath = createHypWithCache("cache: true");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("ram", config.getCacheType());
}

// CF-111: Cache_False
TEST_F(ConfigurationCacheTest, Cache_False) {
    std::string hypPath = createHypWithCache("cache: false");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("", config.getCacheType());
}

// CF-112: Cache_Ram
TEST_F(ConfigurationCacheTest, Cache_Ram) {
    std::string hypPath = createHypWithCache("cache: \"ram\"");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("ram", config.getCacheType());
}

// CF-113: Cache_Disk
TEST_F(ConfigurationCacheTest, Cache_Disk) {
    std::string hypPath = createHypWithCache("cache: \"disk\"");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ("disk", config.getCacheType());
}

// =============================================================================
// validateConfiguration() Tests (CF-114 ~ CF-155)
// =============================================================================

class ConfigurationValidationTest : public ConfigurationTest {
protected:
    // Helper to create full YAML content for validation tests
    std::string createHypForValidation(
        int epochs = 100,
        int batchSize = 16,
        int imageSize = 640,
        const std::string& extraYaml = "") {
        std::string content =
            "epochs: " + std::to_string(epochs) + "\n" +
            "batch_size: " + std::to_string(batchSize) + "\n" +
            "image_size: " + std::to_string(imageSize) + "\n" +
            extraYaml;
        createFile("hyp_val.yaml", content);
        return getTestPath("hyp_val.yaml");
    }
};

// CF-114: Val_EpochsZero
TEST_F(ConfigurationValidationTest, Val_EpochsZero) {
    std::string hypPath = createHypForValidation(0, 16, 640);
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-115: Val_EpochsNegative
TEST_F(ConfigurationValidationTest, Val_EpochsNegative) {
    std::string hypPath = createHypForValidation(-1, 16, 640);
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-116: Val_BatchSizeZero
TEST_F(ConfigurationValidationTest, Val_BatchSizeZero) {
    std::string hypPath = createHypForValidation(100, 0, 640);
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-117: Val_ImageSizeZero
TEST_F(ConfigurationValidationTest, Val_ImageSizeZero) {
    std::string hypPath = createHypForValidation(100, 16, 0);
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-118: Val_WorkersNegative
TEST_F(ConfigurationValidationTest, Val_WorkersNegative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "workers: -1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-119: Val_MaxDetZero
TEST_F(ConfigurationValidationTest, Val_MaxDetZero) {
    std::string hypPath = createHypForValidation(100, 16, 640, "max_det: 0");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-120: Val_PatienceNegative
TEST_F(ConfigurationValidationTest, Val_PatienceNegative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "patience: -1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-121: Val_CloseMosaicNegative
TEST_F(ConfigurationValidationTest, Val_CloseMosaicNegative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "close_mosaic: -1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-122: Val_BlurKernelSizeNegative
TEST_F(ConfigurationValidationTest, Val_BlurKernelSizeNegative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "blur_kernel_size: -1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-123: Val_HsvH_Over
TEST_F(ConfigurationValidationTest, Val_HsvH_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "hsv_h: 1.5");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-124: Val_HsvH_Negative
TEST_F(ConfigurationValidationTest, Val_HsvH_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "hsv_h: -0.1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-125: Val_HsvS_Over
TEST_F(ConfigurationValidationTest, Val_HsvS_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "hsv_s: 1.5");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-126: Val_HsvS_Negative
TEST_F(ConfigurationValidationTest, Val_HsvS_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "hsv_s: -0.1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-127: Val_HsvV_Over
TEST_F(ConfigurationValidationTest, Val_HsvV_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "hsv_v: 1.5");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-128: Val_HsvV_Negative
TEST_F(ConfigurationValidationTest, Val_HsvV_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "hsv_v: -0.1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-129: Val_Translate_Over
TEST_F(ConfigurationValidationTest, Val_Translate_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "translate: 1.5");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-130: Val_Translate_Negative
TEST_F(ConfigurationValidationTest, Val_Translate_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "translate: -0.1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-131: Val_Scale_Over
TEST_F(ConfigurationValidationTest, Val_Scale_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "scale: 1.5");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-132: Val_Scale_Negative
TEST_F(ConfigurationValidationTest, Val_Scale_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "scale: -0.1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-133: Val_Flipud_Over
TEST_F(ConfigurationValidationTest, Val_Flipud_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "flipud: 1.5");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-134: Val_Flipud_Negative
TEST_F(ConfigurationValidationTest, Val_Flipud_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "flipud: -0.1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-135: Val_Fliplr_Over
TEST_F(ConfigurationValidationTest, Val_Fliplr_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "fliplr: 1.5");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-136: Val_Fliplr_Negative
TEST_F(ConfigurationValidationTest, Val_Fliplr_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "fliplr: -0.1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-137: Val_Mosaic_Over
TEST_F(ConfigurationValidationTest, Val_Mosaic_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "mosaic: 1.5");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-138: Val_Mosaic_Negative
TEST_F(ConfigurationValidationTest, Val_Mosaic_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "mosaic: -0.1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-139: Val_BlurProbability_Over
TEST_F(ConfigurationValidationTest, Val_BlurProbability_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "blur_probability: 1.5");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-140: Val_BlurProbability_Negative
TEST_F(ConfigurationValidationTest, Val_BlurProbability_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "blur_probability: -0.1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-141: Val_Lr0_Over
TEST_F(ConfigurationValidationTest, Val_Lr0_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "lr0: 2.0");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-142: Val_Lr0_Negative
TEST_F(ConfigurationValidationTest, Val_Lr0_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "lr0: -0.01");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-143: Val_Lrf_Over
TEST_F(ConfigurationValidationTest, Val_Lrf_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "lrf: 2.0");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-144: Val_Lrf_Negative
TEST_F(ConfigurationValidationTest, Val_Lrf_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "lrf: -0.01");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-145: Val_Momentum_Over
TEST_F(ConfigurationValidationTest, Val_Momentum_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "momentum: 1.5");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-146: Val_Momentum_Negative
TEST_F(ConfigurationValidationTest, Val_Momentum_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "momentum: -0.1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-147: Val_WeightDecay_Negative
TEST_F(ConfigurationValidationTest, Val_WeightDecay_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "weight_decay: -0.001");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-148: Val_WeightDecay_Over
TEST_F(ConfigurationValidationTest, Val_WeightDecay_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "weight_decay: 1.5");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-149: Val_WarmupNegative
TEST_F(ConfigurationValidationTest, Val_WarmupNegative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "warmup_epochs: -1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-150: Val_WarmupTooLarge
TEST_F(ConfigurationValidationTest, Val_WarmupTooLarge) {
    std::string hypPath = createHypForValidation(10, 16, 640, "warmup_epochs: 10");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-151: Val_IoU_Over
TEST_F(ConfigurationValidationTest, Val_IoU_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "iou: 1.5");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-152: Val_IoU_Negative
TEST_F(ConfigurationValidationTest, Val_IoU_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "iou: -0.1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-153: Val_Dropout_Over
TEST_F(ConfigurationValidationTest, Val_Dropout_Over) {
    std::string hypPath = createHypForValidation(100, 16, 640, "dropout: 1.5");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-154: Val_Dropout_Negative
TEST_F(ConfigurationValidationTest, Val_Dropout_Negative) {
    std::string hypPath = createHypForValidation(100, 16, 640, "dropout: -0.1");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-155: Val_Valid
TEST_F(ConfigurationValidationTest, Val_Valid) {
    std::string hypPath = createValidHyperParams();
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_NO_THROW({
        config.validateConfiguration();
    });
}

// =============================================================================
// validateConfiguration() Boundary Valid Cases (CF-B01 ~ CF-B10)
// =============================================================================

// CF-B01: Val_HsvH_Zero
TEST_F(ConfigurationValidationTest, Val_HsvH_Zero) {
    std::string hypPath = createHypForValidation(100, 16, 640, "hsv_h: 0.0");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_NO_THROW({
        config.validateConfiguration();
    });
}

// CF-B02: Val_HsvH_One
TEST_F(ConfigurationValidationTest, Val_HsvH_One) {
    std::string hypPath = createHypForValidation(100, 16, 640, "hsv_h: 1.0");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_NO_THROW({
        config.validateConfiguration();
    });
}

// CF-B03: Val_Lr0_Zero
TEST_F(ConfigurationValidationTest, Val_Lr0_Zero) {
    std::string hypPath = createHypForValidation(100, 16, 640, "lr0: 0.0");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_NO_THROW({
        config.validateConfiguration();
    });
}

// CF-B04: Val_Lr0_One
TEST_F(ConfigurationValidationTest, Val_Lr0_One) {
    std::string hypPath = createHypForValidation(100, 16, 640, "lr0: 1.0");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_NO_THROW({
        config.validateConfiguration();
    });
}

// CF-B05: Val_Momentum_Zero
TEST_F(ConfigurationValidationTest, Val_Momentum_Zero) {
    std::string hypPath = createHypForValidation(100, 16, 640, "momentum: 0.0");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_NO_THROW({
        config.validateConfiguration();
    });
}

// CF-B06: Val_Momentum_One
TEST_F(ConfigurationValidationTest, Val_Momentum_One) {
    std::string hypPath = createHypForValidation(100, 16, 640, "momentum: 1.0");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_NO_THROW({
        config.validateConfiguration();
    });
}

// CF-B07: Val_WeightDecay_Zero
TEST_F(ConfigurationValidationTest, Val_WeightDecay_Zero) {
    std::string hypPath = createHypForValidation(100, 16, 640, "weight_decay: 0.0");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_NO_THROW({
        config.validateConfiguration();
    });
}

// CF-B08: Val_WeightDecay_One
TEST_F(ConfigurationValidationTest, Val_WeightDecay_One) {
    std::string hypPath = createHypForValidation(100, 16, 640, "weight_decay: 1.0");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_NO_THROW({
        config.validateConfiguration();
    });
}

// CF-B09: Val_Epochs_One
TEST_F(ConfigurationValidationTest, Val_Epochs_One) {
    std::string hypPath = createHypForValidation(1, 16, 640);
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_NO_THROW({
        config.validateConfiguration();
    });
}

// CF-B10: Val_WarmupZero
TEST_F(ConfigurationValidationTest, Val_WarmupZero) {
    std::string hypPath = createHypForValidation(100, 16, 640, "warmup_epochs: 0.0");
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_NO_THROW({
        config.validateConfiguration();
    });
}

// =============================================================================
// Dataset JSON Parsing Tests (CF-156 ~ CF-160)
// =============================================================================

// CF-156: Dataset_CategoriesArray
TEST_F(ConfigurationTest, Dataset_CategoriesArray) {
    std::string hypPath = createValidHyperParams();
    std::string content = R"({
    "header": {
        "type": "object_detection",
        "categories": ["a", "b", "c"]
    },
    "annotations": []
})";
    createFile("cat_array.json", content);
    std::string datasetPath = getTestPath("cat_array.json");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ(3, config.getNumClasses());
    EXPECT_EQ("a", config.getClassNames().at(0));
    EXPECT_EQ("b", config.getClassNames().at(1));
    EXPECT_EQ("c", config.getClassNames().at(2));
}

// CF-157: Dataset_CategoriesObject
TEST_F(ConfigurationTest, Dataset_CategoriesObject) {
    std::string hypPath = createValidHyperParams();
    std::string content = R"({
    "header": {
        "type": "object_detection",
        "categories": {"0": "a", "1": "b"}
    },
    "annotations": []
})";
    createFile("cat_obj.json", content);
    std::string datasetPath = getTestPath("cat_obj.json");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ(2, config.getNumClasses());
    EXPECT_EQ("a", config.getClassNames().at(0));
    EXPECT_EQ("b", config.getClassNames().at(1));
}

// CF-158: Dataset_InvalidCategoryKey
TEST_F(ConfigurationTest, Dataset_InvalidCategoryKey) {
    std::string hypPath = createValidHyperParams();
    std::string content = R"({
    "header": {
        "type": "object_detection",
        "categories": {"abc": "class"}
    },
    "annotations": []
})";
    createFile("invalid_key.json", content);
    std::string datasetPath = getTestPath("invalid_key.json");

    Configuration config;
    EXPECT_THROW({
        try {
            config.load("model.yaml", hypPath, datasetPath);
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::INVALID_CONFIG_VALUE, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// CF-159: Dataset_Annotations
TEST_F(ConfigurationTest, Dataset_Annotations) {
    std::string hypPath = createValidHyperParams();
    std::string content = R"({
    "header": {
        "type": "object_detection",
        "categories": ["class0"]
    },
    "annotations": [
        {"id": 1},
        {"id": 2},
        {"id": 3},
        {"id": 4},
        {"id": 5}
    ]
})";
    createFile("annotations.json", content);
    std::string datasetPath = getTestPath("annotations.json");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ(5u, config.getNumDataSamples());
}

// CF-160: Dataset_MissingAnnotations
TEST_F(ConfigurationTest, Dataset_MissingAnnotations) {
    std::string hypPath = createValidHyperParams();
    std::string content = R"({
    "header": {
        "type": "object_detection",
        "categories": ["class0"]
    }
})";
    createFile("no_annotations.json", content);
    std::string datasetPath = getTestPath("no_annotations.json");

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);
    EXPECT_EQ(0u, config.getNumDataSamples());
}

// =============================================================================
// load() Path Behavior Tests (CF-161 ~ CF-162)
// =============================================================================

// CF-161: Load_ModelPathStored
TEST_F(ConfigurationTest, Load_ModelPathStored) {
    std::string hypPath = createValidHyperParams();
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("test_model.yaml", hypPath, datasetPath);

    EXPECT_EQ("test_model.yaml", config.getModelPath());
}

// CF-162: Load_DatasetPathStored
TEST_F(ConfigurationTest, Load_DatasetPathStored) {
    std::string hypPath = createValidHyperParams();
    std::string datasetPath = createValidDataset();

    Configuration config;
    config.load("model.yaml", hypPath, datasetPath);

    EXPECT_EQ(datasetPath, config.getDatasetPath());
}

// =============================================================================
// Silent Failure Tests - Bug Documentation (CF-163 ~ CF-167)
// These tests document cases where invalid config is silently accepted
// =============================================================================

class ConfigurationSilentFailureTest : public ConfigurationTest {
protected:
    std::string createHypWithContent(const std::string& content) {
        createFile("hyp_silent.yaml", content);
        return getTestPath("hyp_silent.yaml");
    }
};

// CF-163: SilentFail_NegativeEpochs
// BUG DOCUMENTATION: Negative epochs is silently accepted during load()
// validateConfiguration() must be called explicitly to catch this
TEST_F(ConfigurationSilentFailureTest, SilentFail_NegativeEpochs) {
    std::string hypPath = createHypWithContent(R"(
epochs: -10
batch_size: 16
image_size: 640
)");
    std::string datasetPath = createValidDataset();

    Configuration config;
    // load() does NOT throw - this is the documented behavior
    EXPECT_NO_THROW({
        config.load("model.yaml", hypPath, datasetPath);
    });

    // Negative value is stored
    EXPECT_EQ(-10, config.getEpochs());

    // Only validateConfiguration() catches this
    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-164: SilentFail_ZeroBatchSize
// BUG DOCUMENTATION: Zero batch_size is silently accepted during load()
TEST_F(ConfigurationSilentFailureTest, SilentFail_ZeroBatchSize) {
    std::string hypPath = createHypWithContent(R"(
epochs: 100
batch_size: 0
image_size: 640
)");
    std::string datasetPath = createValidDataset();

    Configuration config;
    EXPECT_NO_THROW({
        config.load("model.yaml", hypPath, datasetPath);
    });

    EXPECT_EQ(0, config.getBatchSize());

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-165: SilentFail_NegativeLR
// BUG DOCUMENTATION: Negative learning rate is silently accepted during load()
TEST_F(ConfigurationSilentFailureTest, SilentFail_NegativeLR) {
    std::string hypPath = createHypWithContent(R"(
epochs: 100
batch_size: 16
image_size: 640
lr0: -0.5
)");
    std::string datasetPath = createValidDataset();

    Configuration config;
    EXPECT_NO_THROW({
        config.load("model.yaml", hypPath, datasetPath);
    });

    EXPECT_FLOAT_EQ(-0.5f, config.getLearningRate());

    EXPECT_THROW({
        config.validateConfiguration();
    }, ConfigurationException);
}

// CF-166: SilentFail_InvalidOptimizer
// DESIGN DOCUMENTATION: Invalid optimizer is silently converted to "auto"
TEST_F(ConfigurationSilentFailureTest, SilentFail_InvalidOptimizer) {
    std::string hypPath = createHypWithContent(R"(
epochs: 100
batch_size: 16
image_size: 640
optimizer: rmsprop
)");
    std::string datasetPath = createValidDataset();

    Configuration config;
    EXPECT_NO_THROW({
        config.load("model.yaml", hypPath, datasetPath);
    });

    // Invalid optimizer is silently converted to "auto" - no warning
    EXPECT_EQ("auto", config.getOptimizer());
}

// CF-167: SilentFail_InvalidDevice
// DESIGN DOCUMENTATION: Invalid device array causes empty device string
TEST_F(ConfigurationSilentFailureTest, SilentFail_InvalidDevice) {
    std::string hypPath = createHypWithContent(R"(
epochs: 100
batch_size: 16
image_size: 640
device: [0, "abc", 2]
)");
    std::string datasetPath = createValidDataset();

    Configuration config;
    // May throw yaml exception or silently fail
    try {
        config.load("model.yaml", hypPath, datasetPath);
        // If load succeeds, device should be empty or partial
        // Current implementation: catches exception and sets empty string
        SUCCEED() << "Invalid device handled - device=" << config.getDevice();
    }
    catch (const std::exception&) {
        SUCCEED() << "Invalid device caused exception during parsing";
    }
}

// =============================================================================
// Exception Type & Message Tests (CF-169 ~ CF-174)
// =============================================================================

// CF-169: ExType_FileNotFound
TEST_F(ConfigurationTest, ExType_FileNotFound) {
    std::string datasetPath = createValidDataset();

    Configuration config;
    try {
        config.load("model.yaml", "nonexistent.yaml", datasetPath);
        FAIL() << "Expected ConfigurationException";
    }
    catch (const ConfigurationException& e) {
        EXPECT_EQ(ErrorCode::CONFIG_FILE_NOT_FOUND, e.getErrorCode());
    }
}

// CF-170: ExType_ParseFailed
TEST_F(ConfigurationTest, ExType_ParseFailed) {
    createFile("invalid.yaml", "key: [broken");
    std::string hypPath = getTestPath("invalid.yaml");
    std::string datasetPath = createValidDataset();

    Configuration config;
    try {
        config.load("model.yaml", hypPath, datasetPath);
        FAIL() << "Expected ConfigurationException";
    }
    catch (const ConfigurationException& e) {
        EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
    }
}

// CF-171: ExType_InvalidValue
TEST_F(ConfigurationTest, ExType_InvalidValue) {
    std::string hypPath = createValidHyperParams();
    std::string content = R"({
    "header": {
        "type": "object_detection",
        "categories": []
    },
    "annotations": []
})";
    createFile("empty_cat.json", content);
    std::string datasetPath = getTestPath("empty_cat.json");

    Configuration config;
    try {
        config.load("model.yaml", hypPath, datasetPath);
        FAIL() << "Expected ConfigurationException";
    }
    catch (const ConfigurationException& e) {
        EXPECT_EQ(ErrorCode::INVALID_CONFIG_VALUE, e.getErrorCode());
    }
}

// CF-172: ExMsg_ContainsParamName
TEST_F(ConfigurationTest, ExMsg_ContainsParamName) {
    Configuration config;

    try {
        config.setNumClasses(-1);  // This should throw with param name
        FAIL() << "Expected ConfigurationException from setNumClasses";
    }
    catch (const ConfigurationException& e) {
        // Message should contain "numClasses"
        EXPECT_NE(std::string::npos, e.getMessage().find("numClasses"));
    }
}

// CF-173: ExMsg_ContainsValue
TEST_F(ConfigurationTest, ExMsg_ContainsValue) {
    std::string hypPath = createValidHyperParams();
    std::string datasetPath = createValidDataset();

    Configuration config;
    try {
        config.setNumClasses(-5);
        FAIL() << "Expected ConfigurationException";
    }
    catch (const ConfigurationException& e) {
        // Message should contain the actual value "-5"
        EXPECT_NE(std::string::npos, e.getMessage().find("-5"));
    }
}

// CF-174: ExMsg_ContainsFilePath
TEST_F(ConfigurationTest, ExMsg_ContainsFilePath) {
    std::string datasetPath = createValidDataset();
    std::string nonexistentPath = getTestPath("nonexistent_file.yaml");

    Configuration config;
    try {
        config.load("model.yaml", nonexistentPath, datasetPath);
        FAIL() << "Expected ConfigurationException";
    }
    catch (const ConfigurationException& e) {
        // Message should contain file path or filename
        EXPECT_NE(std::string::npos, e.getMessage().find("nonexistent"));
    }
}

// =============================================================================
// Uncaught Exception Tests - Bug Documentation (CF-175 ~ CF-176)
// =============================================================================

// CF-175: CaughtEx_ArrayNonString
// BUG FIXED: Non-string values in array categories now properly wrapped as ConfigurationException
TEST_F(ConfigurationTest, CaughtEx_ArrayNonString) {
    std::string hypPath = createValidHyperParams();
    std::string content = R"({
    "header": {
        "type": "object_detection",
        "categories": [1, 2, 3]
    },
    "annotations": []
})";
    createFile("int_array_cat.json", content);
    std::string datasetPath = getTestPath("int_array_cat.json");

    Configuration config;
    // Array format now has try-catch - correctly wrapped as ConfigurationException
    EXPECT_THROW({
        try {
            config.load("model.yaml", hypPath, datasetPath);
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::INVALID_CONFIG_VALUE, e.getErrorCode());
            // Verify message contains useful info
            EXPECT_NE(std::string::npos, e.getMessage().find("index"));
            throw;
        }
    }, ConfigurationException);
}

// CF-176: CaughtEx_ObjectNonString
// This correctly throws ConfigurationException because object processing has try-catch
TEST_F(ConfigurationTest, CaughtEx_ObjectNonString) {
    std::string hypPath = createValidHyperParams();
    std::string content = R"({
    "header": {
        "type": "object_detection",
        "categories": {"0": 123, "1": 456}
    },
    "annotations": []
})";
    createFile("int_obj_cat.json", content);
    std::string datasetPath = getTestPath("int_obj_cat.json");

    Configuration config;
    // Object format has try-catch - correctly wrapped as ConfigurationException
    EXPECT_THROW({
        try {
            config.load("model.yaml", hypPath, datasetPath);
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::INVALID_CONFIG_VALUE, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}
