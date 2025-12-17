#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <regex>
#include <string>

#include "Core/Utils/CheckpointMetadata.h"
#include "Config/Configuration.h"
#include "Utils/Common/Types.h"
#include "Utils/Common/Constants.h"
#include "Unit/Core/CoreTestHelpers.h"

using namespace WheelDL;
using namespace WheelDL::Core::Utils;
using namespace WheelDL::Test::Core;

// ============================================================================
// Test Fixture
// ============================================================================

class CheckpointMetadataTest : public CUDATestFixture {
protected:
    void SetUp() override {
        CUDATestFixture::SetUp();
        _tempManager = std::make_unique<TempFileManager>("CheckpointMetadataTest");
    }

    void TearDown() override {
        _tempManager.reset();
        CUDATestFixture::TearDown();
    }

    std::shared_ptr<Config::Configuration> createConfigWithDevice(const std::string& device) {
        std::string datasetPath = _tempManager->createFile("dataset.json", R"({
    "header": {
        "type": "classification",
        "categories": ["class0", "class1", "class2", "class3", "class4",
                       "class5", "class6", "class7", "class8", "class9"]
    },
    "annotations": []
})");

        std::string deviceStr = device.empty() ? "\"\"" : ("\"" + device + "\"");
        std::string hypContent =
            "epochs: 100\n"
            "patience: 50\n"
            "batch_size: 16\n"
            "image_size: 640\n"
            "device: " + deviceStr + "\n"
            "workers: 0\n"
            "optimizer: SGD\n"
            "seed: 42\n"
            "deterministic: true\n"
            "lr0: 0.01\n"
            "lrf: 0.001\n"
            "momentum: 0.937\n"
            "weight_decay: 0.0005\n"
            "warmup_epochs: 3.0\n"
            "warmup_momentum: 0.8\n"
            "warmup_bias_lr: 0.1\n"
            "box: 7.5\n"
            "cls: 0.5\n"
            "dfl: 1.5\n"
            "hsv_h: 0.0\n"
            "hsv_s: 0.0\n"
            "hsv_v: 0.0\n"
            "degrees: 0.0\n"
            "translate: 0.0\n"
            "scale: 0.0\n"
            "shear: 0.0\n"
            "perspective: 0.0\n"
            "flipud: 0.0\n"
            "fliplr: 0.0\n"
            "mosaic: 0.0\n"
            "cos_lr: false\n"
            "amp: false\n";

        std::string hypPath = _tempManager->createFile("hyp.yaml", hypContent);

        auto config = std::make_shared<Config::Configuration>();
        config->load("Config/Valid/model/cls_ConvNext.yaml", hypPath, datasetPath);
        return config;
    }

    std::shared_ptr<Config::Configuration> createConfigWithParams(
        int epochs, int batchSize, int imageSize,
        const std::string& optimizer, float lr0, float lrf,
        float momentum, float weightDecay, float warmupEpochs,
        bool cosLR, bool amp)
    {
        std::string datasetPath = _tempManager->createFile("dataset.json", R"({
    "header": {
        "type": "classification",
        "categories": ["class0", "class1", "class2", "class3", "class4",
                       "class5", "class6", "class7", "class8", "class9"]
    },
    "annotations": []
})");

        std::string hypContent =
            "epochs: " + std::to_string(epochs) + "\n"
            "patience: 50\n"
            "batch_size: " + std::to_string(batchSize) + "\n"
            "image_size: " + std::to_string(imageSize) + "\n"
            "device: \"0\"\n"
            "workers: 0\n"
            "optimizer: " + optimizer + "\n"
            "seed: 42\n"
            "deterministic: true\n"
            "lr0: " + std::to_string(lr0) + "\n"
            "lrf: " + std::to_string(lrf) + "\n"
            "momentum: " + std::to_string(momentum) + "\n"
            "weight_decay: " + std::to_string(weightDecay) + "\n"
            "warmup_epochs: " + std::to_string(warmupEpochs) + "\n"
            "warmup_momentum: 0.8\n"
            "warmup_bias_lr: 0.1\n"
            "box: 7.5\n"
            "cls: 0.5\n"
            "dfl: 1.5\n"
            "hsv_h: 0.0\n"
            "hsv_s: 0.0\n"
            "hsv_v: 0.0\n"
            "degrees: 0.0\n"
            "translate: 0.0\n"
            "scale: 0.0\n"
            "shear: 0.0\n"
            "perspective: 0.0\n"
            "flipud: 0.0\n"
            "fliplr: 0.0\n"
            "mosaic: 0.0\n"
            "cos_lr: " + std::string(cosLR ? "true" : "false") + "\n"
            "amp: " + std::string(amp ? "true" : "false") + "\n";

        std::string hypPath = _tempManager->createFile("hyp_" + std::to_string(++_fileCounter) + ".yaml", hypContent);

        auto config = std::make_shared<Config::Configuration>();
        config->load("Config/Valid/model/cls_ConvNext.yaml", hypPath, datasetPath);
        return config;
    }

    MetricsData createTestMetrics() {
        MetricsData m;
        m.loss = 0.5f;
        m.accuracy = 0.85f;
        m.precision = 0.82f;
        m.recall = 0.78f;
        m.f1Score = 0.80f;
        m.mAP = 0.75f;
        m.fitness = 0.95f;
        m.threshold = 0.5f;
        m.aucROC = 0.88f;
        return m;
    }

    std::unique_ptr<TempFileManager> _tempManager;
    int _fileCounter = 0;
};

// ============================================================================
// 2.1 FromConfiguration Tests (CM-001 ~ CM-019)
// ============================================================================

// CM-001: FromConfig_VersionInfo
TEST_F(CheckpointMetadataTest, FromConfig_VersionInfo) {
    if (!requireCuda()) return;

    auto config = createConfigWithDevice("0");
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    // Check wheelLibVersion format (e.g., "0.1.0")
    EXPECT_FALSE(metadata.wheelLibVersion.empty());
    std::regex versionRegex(R"(\d+\.\d+\.\d+)");
    EXPECT_TRUE(std::regex_match(metadata.wheelLibVersion, versionRegex))
        << "wheelLibVersion: " << metadata.wheelLibVersion;

    // Check libtorchVersion format
    EXPECT_FALSE(metadata.libtorchVersion.empty());
    EXPECT_TRUE(std::regex_match(metadata.libtorchVersion, versionRegex))
        << "libtorchVersion: " << metadata.libtorchVersion;
}

// CM-002: FromConfig_TaskType
TEST_F(CheckpointMetadataTest, FromConfig_TaskType) {
    if (!requireCuda()) return;

    auto config = createConfigWithDevice("0");
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    EXPECT_EQ(metadata.taskType, config->getTaskType());
}

// CM-003: FromConfig_Epoch
TEST_F(CheckpointMetadataTest, FromConfig_Epoch) {
    if (!requireCuda()) return;

    auto config = createConfigWithDevice("0");
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 50, metrics);

    EXPECT_EQ(metadata.epoch, 50);
}

// CM-004: FromConfig_BestFitness
TEST_F(CheckpointMetadataTest, FromConfig_BestFitness) {
    if (!requireCuda()) return;

    auto config = createConfigWithDevice("0");
    MetricsData metrics;
    metrics.fitness = 0.95f;

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    EXPECT_FLOAT_EQ(metadata.bestFitness, 0.95f);
}

// CM-005: FromConfig_AllMetrics
TEST_F(CheckpointMetadataTest, FromConfig_AllMetrics) {
    if (!requireCuda()) return;

    auto config = createConfigWithDevice("0");
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    EXPECT_FLOAT_EQ(metadata.loss, metrics.loss);
    EXPECT_FLOAT_EQ(metadata.accuracy, metrics.accuracy);
    EXPECT_FLOAT_EQ(metadata.precision, metrics.precision);
    EXPECT_FLOAT_EQ(metadata.recall, metrics.recall);
    EXPECT_FLOAT_EQ(metadata.f1Score, metrics.f1Score);
    EXPECT_FLOAT_EQ(metadata.mAP, metrics.mAP);
    EXPECT_FLOAT_EQ(metadata.threshold, metrics.threshold);
    EXPECT_FLOAT_EQ(metadata.bestFitness, metrics.fitness);
}

// CM-006: FromConfig_HyperParams_Epochs
TEST_F(CheckpointMetadataTest, FromConfig_HyperParams_Epochs) {
    if (!requireCuda()) return;

    auto config = createConfigWithParams(100, 16, 640, "SGD", 0.01f, 0.001f, 0.937f, 0.0005f, 3.0f, false, false);
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    EXPECT_EQ(metadata.hyperParams.at("epochs"), "100");
}

// CM-007: FromConfig_HyperParams_BatchSize
TEST_F(CheckpointMetadataTest, FromConfig_HyperParams_BatchSize) {
    if (!requireCuda()) return;

    auto config = createConfigWithParams(100, 32, 640, "SGD", 0.01f, 0.001f, 0.937f, 0.0005f, 3.0f, false, false);
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    EXPECT_EQ(metadata.hyperParams.at("batch_size"), "32");
}

// CM-008: FromConfig_HyperParams_ImageSize
TEST_F(CheckpointMetadataTest, FromConfig_HyperParams_ImageSize) {
    if (!requireCuda()) return;

    auto config = createConfigWithParams(100, 16, 224, "SGD", 0.01f, 0.001f, 0.937f, 0.0005f, 3.0f, false, false);
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    EXPECT_EQ(metadata.hyperParams.at("image_size"), "224");
}

// CM-009: FromConfig_HyperParams_Optimizer
TEST_F(CheckpointMetadataTest, FromConfig_HyperParams_Optimizer) {
    if (!requireCuda()) return;

    auto config = createConfigWithParams(100, 16, 640, "Adam", 0.01f, 0.001f, 0.937f, 0.0005f, 3.0f, false, false);
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    // Optimizer may be stored as lowercase
    std::string optimizerVal = metadata.hyperParams.at("optimizer");
    std::transform(optimizerVal.begin(), optimizerVal.end(), optimizerVal.begin(), ::tolower);
    EXPECT_EQ(optimizerVal, "adam");
}

// CM-010: FromConfig_HyperParams_LR
TEST_F(CheckpointMetadataTest, FromConfig_HyperParams_LR) {
    if (!requireCuda()) return;

    auto config = createConfigWithParams(100, 16, 640, "SGD", 0.01f, 0.001f, 0.937f, 0.0005f, 3.0f, false, false);
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    // Check lr0 exists and is numeric
    EXPECT_TRUE(metadata.hyperParams.find("lr0") != metadata.hyperParams.end());
    float lr0 = std::stof(metadata.hyperParams.at("lr0"));
    EXPECT_NEAR(lr0, 0.01f, 0.001f);

    // Check lrf exists and is numeric
    EXPECT_TRUE(metadata.hyperParams.find("lrf") != metadata.hyperParams.end());
    float lrf = std::stof(metadata.hyperParams.at("lrf"));
    EXPECT_NEAR(lrf, 0.001f, 0.0001f);
}

// CM-011: FromConfig_HyperParams_Momentum
TEST_F(CheckpointMetadataTest, FromConfig_HyperParams_Momentum) {
    if (!requireCuda()) return;

    auto config = createConfigWithParams(100, 16, 640, "SGD", 0.01f, 0.001f, 0.937f, 0.0005f, 3.0f, false, false);
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    float momentum = std::stof(metadata.hyperParams.at("momentum"));
    EXPECT_NEAR(momentum, 0.937f, 0.001f);
}

// CM-012: FromConfig_HyperParams_WeightDecay
TEST_F(CheckpointMetadataTest, FromConfig_HyperParams_WeightDecay) {
    if (!requireCuda()) return;

    auto config = createConfigWithParams(100, 16, 640, "SGD", 0.01f, 0.001f, 0.937f, 0.0005f, 3.0f, false, false);
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    float weightDecay = std::stof(metadata.hyperParams.at("weight_decay"));
    EXPECT_NEAR(weightDecay, 0.0005f, 0.0001f);
}

// CM-013: FromConfig_HyperParams_WarmupEpochs
TEST_F(CheckpointMetadataTest, FromConfig_HyperParams_WarmupEpochs) {
    if (!requireCuda()) return;

    auto config = createConfigWithParams(100, 16, 640, "SGD", 0.01f, 0.001f, 0.937f, 0.0005f, 5.0f, false, false);
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    float warmupEpochs = std::stof(metadata.hyperParams.at("warmup_epochs"));
    EXPECT_NEAR(warmupEpochs, 5.0f, 0.1f);
}

// CM-014: FromConfig_HyperParams_AMP
TEST_F(CheckpointMetadataTest, FromConfig_HyperParams_AMP) {
    if (!requireCuda()) return;

    auto config = createConfigWithParams(100, 16, 640, "SGD", 0.01f, 0.001f, 0.937f, 0.0005f, 3.0f, false, true);
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    EXPECT_EQ(metadata.hyperParams.at("amp"), "true");
}

// CM-015: FromConfig_HyperParams_CosLR
TEST_F(CheckpointMetadataTest, FromConfig_HyperParams_CosLR) {
    if (!requireCuda()) return;

    auto config = createConfigWithParams(100, 16, 640, "SGD", 0.01f, 0.001f, 0.937f, 0.0005f, 3.0f, true, false);
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    EXPECT_EQ(metadata.hyperParams.at("cos_lr"), "true");
}

// CM-016: FromConfig_Timestamp
TEST_F(CheckpointMetadataTest, FromConfig_Timestamp) {
    if (!requireCuda()) return;

    auto config = createConfigWithDevice("0");
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    // Check ISO 8601 format: YYYY-MM-DDTHH:MM:SS
    EXPECT_FALSE(metadata.timestamp.empty());
    std::regex timestampRegex(R"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})");
    EXPECT_TRUE(std::regex_match(metadata.timestamp, timestampRegex))
        << "Timestamp: " << metadata.timestamp;
}

// CM-017: FromConfig_DeviceType_CPU
TEST_F(CheckpointMetadataTest, FromConfig_DeviceType_CPU) {
    if (!requireCuda()) return;

    auto config = createConfigWithDevice("cpu");
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    EXPECT_EQ(metadata.deviceType, "cpu");
    EXPECT_EQ(metadata.gpuCount, 0);
}

// CM-018: FromConfig_DeviceType_SingleGPU
TEST_F(CheckpointMetadataTest, FromConfig_DeviceType_SingleGPU) {
    if (!requireCuda()) return;

    auto config = createConfigWithDevice("0");
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    EXPECT_EQ(metadata.deviceType, "cuda");
    EXPECT_EQ(metadata.gpuCount, 1);
}

// CM-019: FromConfig_DeviceType_MultiGPU
TEST_F(CheckpointMetadataTest, FromConfig_DeviceType_MultiGPU) {
    if (!requireCuda()) return;

    auto config = createConfigWithDevice("0,1,2");
    auto metrics = createTestMetrics();

    auto metadata = CheckpointMetadata::fromConfiguration(*config, 10, metrics);

    EXPECT_EQ(metadata.deviceType, "cuda");
    EXPECT_EQ(metadata.gpuCount, 3);
}

// ============================================================================
// 2.2 Struct Default Values Tests (CM-020 ~ CM-023)
// ============================================================================

// CM-020: DefaultConstruct_EmptyStrings
TEST_F(CheckpointMetadataTest, DefaultConstruct_EmptyStrings) {
    if (!requireCuda()) return;

    CheckpointMetadata metadata;

    EXPECT_TRUE(metadata.wheelLibVersion.empty());
    EXPECT_TRUE(metadata.libtorchVersion.empty());
    EXPECT_TRUE(metadata.timestamp.empty());
    EXPECT_TRUE(metadata.deviceType.empty());
}

// CM-021: DefaultConstruct_ZeroValues
// Tests value-initialization (brace-initialization) which zero-initializes all members
TEST_F(CheckpointMetadataTest, DefaultConstruct_ZeroValues) {
    if (!requireCuda()) return;

    CheckpointMetadata metadata{};  // Value-initialization zeros all members

    EXPECT_EQ(metadata.epoch, 0);
    EXPECT_FLOAT_EQ(metadata.bestFitness, 0.0f);
    EXPECT_EQ(metadata.gpuCount, 0);
    EXPECT_FLOAT_EQ(metadata.loss, 0.0f);
    EXPECT_FLOAT_EQ(metadata.accuracy, 0.0f);
    EXPECT_FLOAT_EQ(metadata.precision, 0.0f);
    EXPECT_FLOAT_EQ(metadata.recall, 0.0f);
    EXPECT_FLOAT_EQ(metadata.f1Score, 0.0f);
    EXPECT_FLOAT_EQ(metadata.mAP, 0.0f);
    EXPECT_FLOAT_EQ(metadata.threshold, 0.0f);
}

// CM-022: DefaultConstruct_EmptyHyperParams
TEST_F(CheckpointMetadataTest, DefaultConstruct_EmptyHyperParams) {
    if (!requireCuda()) return;

    CheckpointMetadata metadata;

    EXPECT_TRUE(metadata.hyperParams.empty());
}

// CM-023: DefaultConstruct_TaskTypeZeroInitialized
// Value-initialization zero-initializes enum to 0, which is CLASSIFICATION
TEST_F(CheckpointMetadataTest, DefaultConstruct_TaskTypeZeroInitialized) {
    if (!requireCuda()) return;

    CheckpointMetadata metadata{};  // Value-initialization zeros all members

    // Value-initialized taskType is 0 (CLASSIFICATION), not UNKNOWN
    // This is standard C++ behavior for aggregate initialization
    EXPECT_EQ(static_cast<int>(metadata.taskType), static_cast<int>(TaskType::CLASSIFICATION));
}
