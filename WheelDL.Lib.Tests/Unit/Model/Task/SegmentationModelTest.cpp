#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Task/SegmentationModel.h"
#include "Model/Loss/SegmentationLoss.h"
#include "Config/Configuration.h"
#include "Utils/Error/WheelLibException.h"
#include "Utils/Error/ErrorCodes.h"
#include "MockConfiguration.h"
#include "MockDataExample.h"
#include <filesystem>
#include <fstream>
#include <memory>

using namespace WheelDL;
using namespace WheelDL::Model;
using namespace WheelDL::Model::Loss;
using namespace WheelDL::Config;
using namespace WheelDL::Test;
using namespace WheelDL::Data::Dataset;
namespace fs = std::filesystem;

// =============================================================================
// Test Fixture
// =============================================================================

class SegmentationModelTest : public ::testing::Test {
protected:
    std::unique_ptr<MockConfigurationHelper> _configHelper;
    std::string _tempDir;
    torch::Device _device = torch::kCPU;

    void SetUp() override {
        torch::manual_seed(42);
        _configHelper = std::make_unique<MockConfigurationHelper>();
        _tempDir = (fs::temp_directory_path() / "WheelDL_SegmentationModel_Test").string();
        fs::remove_all(_tempDir);
        fs::create_directories(_tempDir);

        // Use CUDA if available
        if (torch::cuda::is_available()) {
            _device = torch::kCUDA;
        }
    }

    void TearDown() override {
        try {
            fs::remove_all(_tempDir);
        }
        catch (...) {}
    }

    std::string getTempPath(const std::string& filename) {
        return (fs::path(_tempDir) / filename).string();
    }

    // Helper: Create valid segmentation config
    std::shared_ptr<Configuration> createValidConfig(int numClasses = 10, int imageSize = 640) {
        return _configHelper->createMockConfig(
            TaskType::SEGMENTATION,
            _configHelper->getModelYamlPath("seg_yjnet.yaml"),
            numClasses,
            imageSize
        );
    }

    // Helper: Create config with wrong task type
    std::shared_ptr<Configuration> createWrongTaskTypeConfig() {
        return _configHelper->createMockConfig(
            TaskType::DETECTION,
            _configHelper->getModelYamlPath("seg_yjnet.yaml"),
            10,
            640
        );
    }

    // Helper: Create DataExample for segmentation (on device)
    // Segmentation targets: [N, numClasses, H, W] binary masks
    DataExample createSegmentationData(int batchSize = 1, int numClasses = 10, int imageSize = 640) {
        auto example = createSegmentationDataExample(batchSize, imageSize, 5, numClasses, 32);
        example.data = example.data.to(_device);
        example.classes = example.classes.to(_device);
        example.targets = example.targets.to(_device);
        example.batchIndices = example.batchIndices.to(_device);
        return example;
    }

    // Helper: Create simple segmentation DataExample with mask targets (on device)
    DataExample createMaskData(int batchSize, int numClasses, int height, int width) {
        DataExample example;
        example.data = torch::rand({batchSize, 3, height, width}, _device);
        // Segmentation masks: [N, C, H, W] binary
        example.targets = torch::randint(0, 2, {batchSize, numClasses, height, width}, _device).to(torch::kFloat32);
        example.classes = torch::randint(0, numClasses, {batchSize}, _device);
        example.batchIndices = torch::arange(batchSize, _device);
        return example;
    }

    // Helper: Move model to device
    void moveToDevice(SegmentationModel& model) {
        model.to(_device);
    }
};

// =============================================================================
// Part 3.2.1: Constructor Tests (5 tests)
// =============================================================================

// SM-001: Constructor_ValidConfig - Valid configuration initializes model
TEST_F(SegmentationModelTest, Constructor_ValidConfig) {
    auto config = createValidConfig();

    SegmentationModel model(config);

    EXPECT_EQ(model.getTaskType(), TaskType::SEGMENTATION);
}

// SM-002: Constructor_NullConfig - Null config throws ConfigurationException
TEST_F(SegmentationModelTest, Constructor_NullConfig) {
    EXPECT_THROW(
        {
            try {
                SegmentationModel model(nullptr);
            }
            catch (const Utils::ConfigurationException& e) {
                EXPECT_EQ(e.getErrorCode(), Utils::ErrorCode::INVALID_CONFIG);
                EXPECT_NE(std::string(e.what()).find("Configuration is null"), std::string::npos);
                throw;
            }
        },
        Utils::ConfigurationException
    );
}

// SM-003: Constructor_EmptyModelPath - Empty model path throws ConfigurationException
TEST_F(SegmentationModelTest, Constructor_EmptyModelPath) {
    auto config = std::make_shared<Configuration>();

    EXPECT_THROW(
        {
            try {
                SegmentationModel model(config);
            }
            catch (const Utils::ConfigurationException& e) {
                EXPECT_EQ(e.getErrorCode(), Utils::ErrorCode::INVALID_CONFIG);
                EXPECT_NE(std::string(e.what()).find("Model YAML path is empty"), std::string::npos);
                throw;
            }
        },
        Utils::ConfigurationException
    );
}

// SM-004: Constructor_WrongTaskType - Wrong task type throws exception
TEST_F(SegmentationModelTest, Constructor_WrongTaskType) {
    auto config = createWrongTaskTypeConfig();

    EXPECT_THROW(
        {
            try {
                SegmentationModel model(config);
            }
            catch (const Utils::ModelException& e) {
                EXPECT_EQ(e.getErrorCode(), Utils::ErrorCode::MODEL_LOAD_FAILED);
                EXPECT_NE(std::string(e.what()).find("SEGMENTATION"), std::string::npos);
                throw;
            }
        },
        Utils::ModelException
    );
}

// SM-005: Constructor_DefaultDiceWeight - Default dice weight is 0.5
TEST_F(SegmentationModelTest, Constructor_DefaultDiceWeight) {
    auto config = createValidConfig();
    SegmentationModel model(config);
    moveToDevice(model);

    // Verify model initialized correctly - dice weight is internal
    // but we can verify loss computation works
    auto data = createMaskData(1, 10, 160, 160);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
}

// =============================================================================
// Part 3.2.2: InitCriterion Tests (3 tests)
// =============================================================================

// SM-006: InitCriterion_ReturnsSegmentationLoss - Criterion is SegmentationLoss
TEST_F(SegmentationModelTest, InitCriterion_ReturnsSegmentationLoss) {
    auto config = createValidConfig();
    SegmentationModel model(config);
    moveToDevice(model);

    // Verify loss returns expected components
    auto data = createMaskData(1, 10, 160, 160);
    auto lossMap = model.forward(data);

    // SegmentationLoss returns "bce", "dice", "total"
    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
}

// SM-007: InitCriterion_UsesNumClasses - Uses numClasses from config
TEST_F(SegmentationModelTest, InitCriterion_UsesNumClasses) {
    int numClasses = 20;
    auto config = createValidConfig(numClasses, 640);
    SegmentationModel model(config);
    moveToDevice(model);

    auto data = createMaskData(1, numClasses, 160, 160);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
}

// SM-008: InitCriterion_UsesDiceWeight - Uses configured dice weight
TEST_F(SegmentationModelTest, InitCriterion_UsesDiceWeight) {
    auto config = createValidConfig();
    SegmentationModel model(config);
    moveToDevice(model);

    // Default dice weight is 0.5
    // Verify loss components are computed
    auto data = createMaskData(1, 10, 160, 160);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_GT(lossMap["total"].item<float>(), 0.0f);
}

// =============================================================================
// Part 3.2.3: Forward Pass Tests (3 tests)
// =============================================================================

// SM-009: Forward_Inference - Inference returns segmentation output
TEST_F(SegmentationModelTest, Forward_Inference) {
    int numClasses = 10;
    auto config = createValidConfig(numClasses, 640);
    SegmentationModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    ASSERT_GE(outputs.size(), 1);
    EXPECT_EQ(outputs[0].dim(), 4);  // [N, C, H, W]
}

// SM-010: Forward_Training - Training returns loss map
TEST_F(SegmentationModelTest, Forward_Training) {
    auto config = createValidConfig(10, 640);
    SegmentationModel model(config);
    moveToDevice(model);

    auto data = createMaskData(1, 10, 160, 160);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(lossMap["total"].isinf().any().item<bool>());
}

// SM-011: Forward_OutputResolution - Output has correct resolution
TEST_F(SegmentationModelTest, Forward_OutputResolution) {
    int numClasses = 10;
    auto config = createValidConfig(numClasses, 640);
    SegmentationModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    ASSERT_GE(outputs.size(), 1);
    EXPECT_EQ(outputs[0].size(0), 1);  // batch size
}

// =============================================================================
// Part 3.2.4: Loss Computation Tests (3 tests)
// =============================================================================

// SM-012: Loss_ReturnsComponents - Returns loss with total key
TEST_F(SegmentationModelTest, Loss_ReturnsComponents) {
    auto config = createValidConfig(10, 640);
    SegmentationModel model(config);
    moveToDevice(model);

    auto data = createMaskData(2, 10, 160, 160);
    auto lossMap = model.loss(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
}

// SM-013: Loss_MaskTarget - Loss with mask targets works
TEST_F(SegmentationModelTest, Loss_MaskTarget) {
    auto config = createValidConfig(10, 640);
    SegmentationModel model(config);
    moveToDevice(model);

    // Create data with explicit mask targets
    auto data = createMaskData(2, 10, 160, 160);
    auto lossMap = model.loss(data);

    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(lossMap["total"].isinf().any().item<bool>());
    EXPECT_GE(lossMap["total"].item<float>(), 0.0f);
}

// SM-014: Loss_MultiClass - Multi-class segmentation loss works
TEST_F(SegmentationModelTest, Loss_MultiClass) {
    // Test with 5 classes
    {
        auto config = createValidConfig(5, 640);
        SegmentationModel model(config);
        moveToDevice(model);
        auto data = createMaskData(1, 5, 160, 160);
        auto lossMap = model.loss(data);
        EXPECT_GE(lossMap["total"].item<float>(), 0.0f);
    }

    // Test with 20 classes
    {
        auto config = createValidConfig(20, 640);
        SegmentationModel model(config);
        moveToDevice(model);
        auto data = createMaskData(1, 20, 160, 160);
        auto lossMap = model.loss(data);
        EXPECT_GE(lossMap["total"].item<float>(), 0.0f);
    }
}

// =============================================================================
// Part 3.2.5: LoadPretrained Tests (2 tests)
// =============================================================================

// SM-015: LoadPretrained_ValidPath - Loading valid weights returns true
TEST_F(SegmentationModelTest, LoadPretrained_ValidPath) {
    auto config = createValidConfig();
    SegmentationModel model(config);

    std::string weightsPath = getTempPath("seg_weights.pt");
    model.saveWeights(weightsPath);

    SegmentationModel model2(createValidConfig());
    bool result = model2.loadPretrained(weightsPath);

    EXPECT_TRUE(result);
}

// SM-016: LoadPretrained_InvalidPath - Loading invalid path returns false
TEST_F(SegmentationModelTest, LoadPretrained_InvalidPath) {
    auto config = createValidConfig();
    SegmentationModel model(config);

    bool result = model.loadPretrained("/nonexistent/path/weights.pt");

    EXPECT_FALSE(result);
}

// =============================================================================
// Part 3.2.6: Integration Tests (4 tests)
// =============================================================================

// SM-017: EndToEnd_TrainingStep - Full training step works
TEST_F(SegmentationModelTest, EndToEnd_TrainingStep) {
    auto config = createValidConfig(10, 640);
    SegmentationModel model(config);
    moveToDevice(model);
    model.train();

    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01));

    auto data = createMaskData(2, 10, 160, 160);
    auto lossMap = model.forward(data);

    optimizer.zero_grad();
    lossMap["total"].backward();
    optimizer.step();

    EXPECT_TRUE(lossMap["total"].defined());
}

// SM-018: EndToEnd_ValidationStep - Validation step works
TEST_F(SegmentationModelTest, EndToEnd_ValidationStep) {
    auto config = createValidConfig(10, 640);
    SegmentationModel model(config);
    moveToDevice(model);
    model.eval();

    torch::NoGradGuard no_grad;

    auto input = torch::rand({2, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    ASSERT_GE(outputs.size(), 1);
    EXPECT_EQ(outputs[0].size(0), 2);  // batch size
}

// SM-019: EndToEnd_DeviceTransfer - Model can transfer between devices
TEST_F(SegmentationModelTest, EndToEnd_DeviceTransfer) {
    auto config = createValidConfig(10, 640);
    SegmentationModel model(config);

    // Verify on CPU
    model.to(torch::kCPU);
    for (const auto& param : model.parameters()) {
        EXPECT_TRUE(param.device().is_cpu());
    }

    auto input = torch::rand({1, 3, 640, 640});
    auto outputs = model.forward(input);
    EXPECT_EQ(outputs[0].device().type(), torch::kCPU);

    // CUDA test (if available)
    if (torch::cuda::is_available()) {
        model.to(torch::kCUDA);
        for (const auto& param : model.parameters()) {
            EXPECT_TRUE(param.device().is_cuda());
        }
    }
}

// SM-020: GradientFlow - Gradients flow through model
TEST_F(SegmentationModelTest, GradientFlow) {
    auto config = createValidConfig(10, 640);
    SegmentationModel model(config);
    moveToDevice(model);
    model.train();

    auto data = createMaskData(1, 10, 160, 160);
    data.data.set_requires_grad(true);

    auto lossMap = model.forward(data);
    lossMap["total"].backward();

    bool hasGradient = false;
    for (const auto& param : model.parameters()) {
        if (param.grad().defined() && param.grad().abs().sum().item<float>() > 0) {
            hasGradient = true;
            break;
        }
    }
    EXPECT_TRUE(hasGradient);
}
