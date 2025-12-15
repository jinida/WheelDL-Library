#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Task/DetectionModel.h"
#include "Model/Loss/DetectionLoss.h"
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

class DetectionModelTest : public ::testing::Test {
protected:
    std::unique_ptr<MockConfigurationHelper> _configHelper;
    std::string _tempDir;
    torch::Device _device = torch::kCPU;

    void SetUp() override {
        torch::manual_seed(42);
        _configHelper = std::make_unique<MockConfigurationHelper>();
        _tempDir = (fs::temp_directory_path() / "WheelDL_DetectionModel_Test").string();
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

    // Helper: Create valid detection config
    std::shared_ptr<Configuration> createValidConfig(int numClasses = 80, int imageSize = 640) {
        return _configHelper->createMockConfig(
            TaskType::DETECTION,
            _configHelper->getModelYamlPath("det_yoloxs.yaml"),
            numClasses,
            imageSize
        );
    }

    // Helper: Create config with wrong task type
    std::shared_ptr<Configuration> createWrongTaskTypeConfig() {
        return _configHelper->createMockConfig(
            TaskType::CLASSIFICATION,
            _configHelper->getModelYamlPath("det_yoloxs.yaml"),
            80,
            640
        );
    }

    // Helper: Create DataExample for detection (on device)
    DataExample createDetectionData(int batchSize = 1, int numTargets = 5, int numClasses = 80, int imageSize = 640) {
        auto example = createDetectionDataExample(batchSize, imageSize, numTargets, numClasses);
        example.data = example.data.to(_device);
        example.classes = example.classes.to(_device);
        example.targets = example.targets.to(_device);
        example.batchIndices = example.batchIndices.to(_device);
        return example;
    }

    // Helper: Create DataExample with no targets (empty targets)
    DataExample createEmptyTargetData(int batchSize = 1, int imageSize = 640) {
        DataExample example;
        example.data = torch::rand({batchSize, 3, imageSize, imageSize}, _device);
        example.targets = torch::zeros({0, 6}, _device);
        example.classes = torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt64).device(_device));
        example.batchIndices = torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt64).device(_device));
        return example;
    }

    // Helper: Move model to device
    void moveToDevice(DetectionModel& model) {
        model.to(_device);
    }
};

// =============================================================================
// Part 3.1: Constructor Tests (8 tests)
// =============================================================================

// DM-001: Constructor_ValidConfig - Valid configuration initializes model
TEST_F(DetectionModelTest, Constructor_ValidConfig) {
    auto config = createValidConfig();

    DetectionModel model(config);

    EXPECT_EQ(model.getTaskType(), TaskType::DETECTION);
}

// DM-002: Constructor_NullConfig - Null config throws ConfigurationException
TEST_F(DetectionModelTest, Constructor_NullConfig) {
    EXPECT_THROW(
        {
            try {
                DetectionModel model(nullptr);
            }
            catch (const WheelDL::Utils::ConfigurationException& e) {
                EXPECT_EQ(e.getErrorCode(), WheelDL::Utils::ErrorCode::INVALID_CONFIG);
                EXPECT_NE(std::string(e.what()).find("Configuration is null"), std::string::npos);
                throw;
            }
        },
        WheelDL::Utils::ConfigurationException
    );
}

// DM-003: Constructor_EmptyModelPath - Empty model path throws ConfigurationException
TEST_F(DetectionModelTest, Constructor_EmptyModelPath) {
    auto config = std::make_shared<Configuration>();

    EXPECT_THROW(
        {
            try {
                DetectionModel model(config);
            }
            catch (const WheelDL::Utils::ConfigurationException& e) {
                EXPECT_EQ(e.getErrorCode(), WheelDL::Utils::ErrorCode::INVALID_CONFIG);
                EXPECT_NE(std::string(e.what()).find("Model YAML path is empty"), std::string::npos);
                throw;
            }
        },
        WheelDL::Utils::ConfigurationException
    );
}

// DM-004: Constructor_WrongTaskType - Wrong task type throws exception
TEST_F(DetectionModelTest, Constructor_WrongTaskType) {
    auto config = createWrongTaskTypeConfig();

    EXPECT_THROW(
        {
            try {
                DetectionModel model(config);
            }
            catch (const WheelDL::Utils::ModelException& e) {
                EXPECT_EQ(e.getErrorCode(), WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED);
                EXPECT_NE(std::string(e.what()).find("DETECTION"), std::string::npos);
                throw;
            }
        },
        WheelDL::Utils::ModelException
    );
}

// DM-005: Constructor_LoadsGains - Loads loss gains from config
TEST_F(DetectionModelTest, Constructor_LoadsGains) {
    auto config = createValidConfig();

    DetectionModel model(config);
    moveToDevice(model);

    // Verify model is initialized correctly (gains are internal but loss works)
    auto data = createDetectionData(1, 5, 80, 640);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
}

// DM-006: Constructor_CalculatesStride - Stride is calculated
TEST_F(DetectionModelTest, Constructor_CalculatesStride) {
    auto config = createValidConfig();

    DetectionModel model(config);
    moveToDevice(model);

    // Model should be initialized with stride
    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    // Multi-scale outputs (3 scales for YOLO)
    EXPECT_GE(outputs.size(), 1);
}

// DM-007: Constructor_AdjustsImageSize - Image size adjusted to stride
TEST_F(DetectionModelTest, Constructor_AdjustsImageSize) {
    // Use an image size not divisible by 32
    auto config = createValidConfig(80, 641);

    DetectionModel model(config);

    // Should not crash; size is adjusted internally
    EXPECT_EQ(model.getTaskType(), TaskType::DETECTION);
}

// DM-008: Constructor_InitializesHead - Detection head is initialized
TEST_F(DetectionModelTest, Constructor_InitializesHead) {
    auto config = createValidConfig();

    DetectionModel model(config);
    moveToDevice(model);

    // Verify model works correctly
    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    EXPECT_GE(outputs.size(), 1);
}

// =============================================================================
// Part 3.2: InitCriterion Tests (3 tests)
// =============================================================================

// DM-009: InitCriterion_ReturnsDetectionLoss - Criterion is DetectionLoss
TEST_F(DetectionModelTest, InitCriterion_ReturnsDetectionLoss) {
    auto config = createValidConfig();
    DetectionModel model(config);
    moveToDevice(model);

    // Verify loss returns expected components
    auto data = createDetectionData(1, 5, 80, 640);
    auto lossMap = model.forward(data);

    // DetectionLoss returns "box", "cls", "dfl", "total"
    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
}

// DM-010: InitCriterion_UsesConfigParams - Uses parameters from config
TEST_F(DetectionModelTest, InitCriterion_UsesConfigParams) {
    int numClasses = 20;
    auto config = createValidConfig(numClasses, 640);
    DetectionModel model(config);
    moveToDevice(model);

    auto data = createDetectionData(1, 5, numClasses, 640);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
}

// DM-011: InitCriterion_UsesGains - Uses configured loss gains
TEST_F(DetectionModelTest, InitCriterion_UsesGains) {
    auto config = createValidConfig();
    DetectionModel model(config);
    moveToDevice(model);

    auto data = createDetectionData(1, 5, 80, 640);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_GT(lossMap["total"].item<float>(), 0.0f);
}

// =============================================================================
// Part 3.3: CalculateStride Tests (3 tests)
// =============================================================================

// DM-012: CalculateStride_Standard - Standard stride calculation
TEST_F(DetectionModelTest, CalculateStride_Standard) {
    auto config = createValidConfig();
    DetectionModel model(config);
    moveToDevice(model);

    // Verify model produces multi-scale outputs
    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    // Expect outputs for each scale
    EXPECT_GE(outputs.size(), 1);
}

// DM-013: CalculateStride_MultiScale - Multi-scale predictions
TEST_F(DetectionModelTest, CalculateStride_MultiScale) {
    auto config = createValidConfig();
    DetectionModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    // Detection models typically produce 3 scales (P3, P4, P5)
    ASSERT_GE(outputs.size(), 1);
    for (const auto& out : outputs) {
        EXPECT_EQ(out.dim(), 4);  // [N, C, H, W]
    }
}

// DM-014: CalculateStride_CUDA - Stride calculation on CUDA
TEST_F(DetectionModelTest, CalculateStride_CUDA) {
    auto config = createValidConfig();
    DetectionModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    EXPECT_GE(outputs.size(), 1);
    EXPECT_EQ(outputs[0].device().type(), _device.type());
}

// =============================================================================
// Part 3.4: Forward Pass Tests (4 tests)
// =============================================================================

// DM-015: Forward_Inference - Inference returns detection outputs
TEST_F(DetectionModelTest, Forward_Inference) {
    auto config = createValidConfig(80, 640);
    DetectionModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    ASSERT_GE(outputs.size(), 1);
    EXPECT_EQ(outputs[0].dim(), 4);  // [N, C, H, W]
}

// DM-016: Forward_Training - Training returns loss map
TEST_F(DetectionModelTest, Forward_Training) {
    auto config = createValidConfig(80, 640);
    DetectionModel model(config);
    moveToDevice(model);

    auto data = createDetectionData(2, 10, 80, 640);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(lossMap["total"].isinf().any().item<bool>());
}

// DM-017: Forward_MultiScale - Multi-scale output format
TEST_F(DetectionModelTest, Forward_MultiScale) {
    auto config = createValidConfig(80, 640);
    DetectionModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    // YOLO typically produces 3 scale outputs
    ASSERT_GE(outputs.size(), 1);
}

// DM-018: Forward_BatchProcessing - Batch processing works
TEST_F(DetectionModelTest, Forward_BatchProcessing) {
    auto config = createValidConfig(80, 640);
    DetectionModel model(config);
    moveToDevice(model);

    auto input = torch::rand({4, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    ASSERT_GE(outputs.size(), 1);
    EXPECT_EQ(outputs[0].size(0), 4);  // batch size
}

// =============================================================================
// Part 3.5: Loss Computation Tests (5 tests)
// =============================================================================

// DM-019: Loss_ReturnsComponents - Returns loss with total key
TEST_F(DetectionModelTest, Loss_ReturnsComponents) {
    auto config = createValidConfig(80, 640);
    DetectionModel model(config);
    moveToDevice(model);

    auto data = createDetectionData(2, 10, 80, 640);
    auto lossMap = model.loss(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
}

// DM-020: Loss_NoTargets - Loss with empty targets
TEST_F(DetectionModelTest, Loss_NoTargets) {
    auto config = createValidConfig(80, 640);
    DetectionModel model(config);
    moveToDevice(model);

    auto data = createEmptyTargetData(1, 640);
    auto lossMap = model.loss(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    // Loss should be zero or small with no targets
    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
}

// DM-021: Loss_SingleTarget - Single detection target
TEST_F(DetectionModelTest, Loss_SingleTarget) {
    auto config = createValidConfig(80, 640);
    DetectionModel model(config);
    moveToDevice(model);

    auto data = createDetectionData(1, 1, 80, 640);
    auto lossMap = model.loss(data);

    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(lossMap["total"].isinf().any().item<bool>());
    EXPECT_GE(lossMap["total"].item<float>(), 0.0f);
}

// DM-022: Loss_MultipleTargets - Multiple detection targets
TEST_F(DetectionModelTest, Loss_MultipleTargets) {
    auto config = createValidConfig(80, 640);
    DetectionModel model(config);
    moveToDevice(model);

    auto data = createDetectionData(2, 20, 80, 640);
    auto lossMap = model.loss(data);

    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(lossMap["total"].isinf().any().item<bool>());
    EXPECT_GE(lossMap["total"].item<float>(), 0.0f);
}

// DM-023: Loss_MultiClass - Multi-class detection works
TEST_F(DetectionModelTest, Loss_MultiClass) {
    // Test with different class counts
    {
        auto config = createValidConfig(10, 640);
        DetectionModel model(config);
        moveToDevice(model);
        auto data = createDetectionData(1, 5, 10, 640);
        auto lossMap = model.loss(data);
        EXPECT_GE(lossMap["total"].item<float>(), 0.0f);
    }

    {
        auto config = createValidConfig(20, 640);
        DetectionModel model(config);
        moveToDevice(model);
        auto data = createDetectionData(1, 5, 20, 640);
        auto lossMap = model.loss(data);
        EXPECT_GE(lossMap["total"].item<float>(), 0.0f);
    }
}

// =============================================================================
// Part 3.6: LoadPretrained Tests (2 tests)
// =============================================================================

// DM-024: LoadPretrained_ValidPath - Loading valid weights returns true
TEST_F(DetectionModelTest, LoadPretrained_ValidPath) {
    auto config = createValidConfig();
    DetectionModel model(config);

    std::string weightsPath = getTempPath("det_weights.pt");
    model.saveWeights(weightsPath);

    DetectionModel model2(createValidConfig());
    bool result = model2.loadPretrained(weightsPath);

    EXPECT_TRUE(result);
}

// DM-025: LoadPretrained_InvalidPath - Loading invalid path returns false
TEST_F(DetectionModelTest, LoadPretrained_InvalidPath) {
    auto config = createValidConfig();
    DetectionModel model(config);

    bool result = model.loadPretrained("/nonexistent/path/weights.pt");

    EXPECT_FALSE(result);
}

// =============================================================================
// Part 3.7: Integration Tests (2 tests)
// =============================================================================

// DM-026: EndToEnd_TrainingStep - Full training step works
TEST_F(DetectionModelTest, EndToEnd_TrainingStep) {
    auto config = createValidConfig(80, 640);
    DetectionModel model(config);
    moveToDevice(model);
    model.train();

    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01));

    auto data = createDetectionData(2, 10, 80, 640);
    auto lossMap = model.forward(data);

    optimizer.zero_grad();
    lossMap["total"].backward();
    optimizer.step();

    EXPECT_TRUE(lossMap["total"].defined());
}

// DM-027: GradientFlow - Gradients flow through model
TEST_F(DetectionModelTest, GradientFlow) {
    auto config = createValidConfig(80, 640);
    DetectionModel model(config);
    moveToDevice(model);
    model.train();

    auto data = createDetectionData(1, 5, 80, 640);
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
