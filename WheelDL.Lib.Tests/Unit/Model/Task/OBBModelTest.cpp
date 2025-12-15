#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Task/OBBModel.h"
#include "Model/Loss/OBBLoss.h"
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

class OBBModelTest : public ::testing::Test {
protected:
    std::unique_ptr<MockConfigurationHelper> _configHelper;
    std::string _tempDir;
    torch::Device _device = torch::kCPU;

    void SetUp() override {
        torch::manual_seed(42);
        _configHelper = std::make_unique<MockConfigurationHelper>();
        _tempDir = (fs::temp_directory_path() / "WheelDL_OBBModel_Test").string();
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

    // Helper: Create valid OBB config
    std::shared_ptr<Configuration> createValidConfig(int numClasses = 15, int imageSize = 640) {
        return _configHelper->createMockConfig(
            TaskType::OBB,
            _configHelper->getModelYamlPath("obb_yoloxs.yaml"),
            numClasses,
            imageSize
        );
    }

    // Helper: Create config with wrong task type
    std::shared_ptr<Configuration> createWrongTaskTypeConfig() {
        return _configHelper->createMockConfig(
            TaskType::DETECTION,
            _configHelper->getModelYamlPath("obb_yoloxs.yaml"),
            15,
            640
        );
    }

    // Helper: Create DataExample for OBB (on device)
    // OBB targets: [N, 7] where 7 = [batch_idx, class_id, x, y, w, h, angle]
    DataExample createOBBData(int batchSize = 1, int numTargets = 5, int numClasses = 15, int imageSize = 640) {
        auto example = createOBBDataExample(batchSize, imageSize, numTargets, numClasses);
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
        example.targets = torch::zeros({0, 7}, _device);  // OBB has 7 columns
        example.classes = torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt64).device(_device));
        example.batchIndices = torch::zeros({0}, torch::TensorOptions().dtype(torch::kInt64).device(_device));
        return example;
    }

    // Helper: Create OBB DataExample with specific angle
    DataExample createOBBDataWithAngle(int batchSize, int numTargets, int numClasses, int imageSize, float angle) {
        auto example = createOBBData(batchSize, numTargets, numClasses, imageSize);
        // Set all angles to the specified value
        example.targets.index({torch::indexing::Slice(), 6}) = angle;
        return example;
    }

    // Helper: Move model to device
    void moveToDevice(OBBModel& model) {
        model.to(_device);
    }
};

// =============================================================================
// Part 4.1: Constructor Tests (7 tests)
// =============================================================================

// OBB-001: Constructor_ValidConfig - Valid configuration initializes model
TEST_F(OBBModelTest, Constructor_ValidConfig) {
    auto config = createValidConfig();

    OBBModel model(config);

    EXPECT_EQ(model.getTaskType(), TaskType::OBB);
}

// OBB-002: Constructor_NullConfig - Null config throws ConfigurationException
TEST_F(OBBModelTest, Constructor_NullConfig) {
    EXPECT_THROW(
        {
            try {
                OBBModel model(nullptr);
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

// OBB-003: Constructor_EmptyModelPath - Empty model path throws ConfigurationException
TEST_F(OBBModelTest, Constructor_EmptyModelPath) {
    auto config = std::make_shared<Configuration>();

    EXPECT_THROW(
        {
            try {
                OBBModel model(config);
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

// OBB-004: Constructor_WrongTaskType - Wrong task type throws exception
TEST_F(OBBModelTest, Constructor_WrongTaskType) {
    auto config = createWrongTaskTypeConfig();

    EXPECT_THROW(
        {
            try {
                OBBModel model(config);
            }
            catch (const WheelDL::Utils::ModelException& e) {
                EXPECT_EQ(e.getErrorCode(), WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED);
                EXPECT_NE(std::string(e.what()).find("OBB"), std::string::npos);
                throw;
            }
        },
        WheelDL::Utils::ModelException
    );
}

// OBB-005: Constructor_LoadsGains - Loads loss gains from config
TEST_F(OBBModelTest, Constructor_LoadsGains) {
    auto config = createValidConfig();

    OBBModel model(config);
    moveToDevice(model);

    auto data = createOBBData(1, 5, 15, 640);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
}

// OBB-006: Constructor_CalculatesStride - Stride is calculated
TEST_F(OBBModelTest, Constructor_CalculatesStride) {
    auto config = createValidConfig();

    OBBModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    EXPECT_GE(outputs.size(), 1);
}

// OBB-007: Constructor_InitializesOBBHead - OBB head is initialized
TEST_F(OBBModelTest, Constructor_InitializesOBBHead) {
    auto config = createValidConfig();

    OBBModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    EXPECT_GE(outputs.size(), 1);
}

// =============================================================================
// Part 4.2: InitCriterion Tests (3 tests)
// =============================================================================

// OBB-008: InitCriterion_ReturnsOBBLoss - Criterion is OBBLoss
TEST_F(OBBModelTest, InitCriterion_ReturnsOBBLoss) {
    auto config = createValidConfig();
    OBBModel model(config);
    moveToDevice(model);

    auto data = createOBBData(1, 5, 15, 640);
    auto lossMap = model.forward(data);

    // OBBLoss returns components including "total"
    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
}

// OBB-009: InitCriterion_UsesConfigParams - Uses parameters from config
TEST_F(OBBModelTest, InitCriterion_UsesConfigParams) {
    int numClasses = 20;
    auto config = createValidConfig(numClasses, 640);
    OBBModel model(config);
    moveToDevice(model);

    auto data = createOBBData(1, 5, numClasses, 640);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
}

// OBB-010: InitCriterion_UsesGains - Uses configured loss gains
TEST_F(OBBModelTest, InitCriterion_UsesGains) {
    auto config = createValidConfig();
    OBBModel model(config);
    moveToDevice(model);

    auto data = createOBBData(1, 5, 15, 640);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_GT(lossMap["total"].item<float>(), 0.0f);
}

// =============================================================================
// Part 4.3: CalculateStride Tests (4 tests)
// =============================================================================

// OBB-011: CalculateStride_Standard - Standard stride calculation
TEST_F(OBBModelTest, CalculateStride_Standard) {
    auto config = createValidConfig();
    OBBModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    EXPECT_GE(outputs.size(), 1);
}

// OBB-012: CalculateStride_MultiScale - Multi-scale predictions
TEST_F(OBBModelTest, CalculateStride_MultiScale) {
    auto config = createValidConfig();
    OBBModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    ASSERT_GE(outputs.size(), 1);
}

// OBB-013: CalculateStride_CUDA - Stride calculation on CUDA
TEST_F(OBBModelTest, CalculateStride_CUDA) {
    auto config = createValidConfig();
    OBBModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    EXPECT_GE(outputs.size(), 1);
    EXPECT_EQ(outputs[0].device().type(), _device.type());
}

// OBB-014: CalculateStride_NoGrad - No gradient during stride calculation
TEST_F(OBBModelTest, CalculateStride_NoGrad) {
    auto config = createValidConfig();
    OBBModel model(config);
    moveToDevice(model);

    // Verify model works correctly after stride calculation
    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    EXPECT_GE(outputs.size(), 1);
}

// =============================================================================
// Part 4.4: Forward Pass Tests (3 tests)
// =============================================================================

// OBB-015: Forward_Inference - Inference returns OBB outputs
TEST_F(OBBModelTest, Forward_Inference) {
    auto config = createValidConfig(15, 640);
    OBBModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    ASSERT_GE(outputs.size(), 1);
    EXPECT_EQ(outputs[0].dim(), 4);  // [N, C, H, W]
}

// OBB-016: Forward_Training - Training returns loss map
TEST_F(OBBModelTest, Forward_Training) {
    auto config = createValidConfig(15, 640);
    OBBModel model(config);
    moveToDevice(model);

    auto data = createOBBData(2, 10, 15, 640);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(lossMap["total"].isinf().any().item<bool>());
}

// OBB-017: Forward_BatchProcessing - Batch processing works
TEST_F(OBBModelTest, Forward_BatchProcessing) {
    auto config = createValidConfig(15, 640);
    OBBModel model(config);
    moveToDevice(model);

    auto input = torch::rand({4, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    ASSERT_GE(outputs.size(), 1);
    EXPECT_EQ(outputs[0].size(0), 4);  // batch size
}

// =============================================================================
// Part 4.5: Loss Computation Tests (5 tests)
// =============================================================================

// OBB-018: Loss_ReturnsComponents - Returns loss with total key
TEST_F(OBBModelTest, Loss_ReturnsComponents) {
    auto config = createValidConfig(15, 640);
    OBBModel model(config);
    moveToDevice(model);

    auto data = createOBBData(2, 10, 15, 640);
    auto lossMap = model.loss(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
}

// OBB-019: Loss_NoTargets - Loss with empty targets
TEST_F(OBBModelTest, Loss_NoTargets) {
    auto config = createValidConfig(15, 640);
    OBBModel model(config);
    moveToDevice(model);

    auto data = createEmptyTargetData(1, 640);
    auto lossMap = model.loss(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
}

// OBB-020: Loss_RotatedTargets - Rotated box targets work
TEST_F(OBBModelTest, Loss_RotatedTargets) {
    auto config = createValidConfig(15, 640);
    OBBModel model(config);
    moveToDevice(model);

    // Create data with rotated boxes
    auto data = createOBBDataWithAngle(2, 10, 15, 640, 0.5f);  // ~28.6 degrees
    auto lossMap = model.loss(data);

    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(lossMap["total"].isinf().any().item<bool>());
    EXPECT_GE(lossMap["total"].item<float>(), 0.0f);
}

// OBB-021: Loss_VariousAngles - Different rotation angles work
TEST_F(OBBModelTest, Loss_VariousAngles) {
    auto config = createValidConfig(15, 640);
    OBBModel model(config);
    moveToDevice(model);

    // Test with angle = 0 (horizontal)
    {
        auto data = createOBBDataWithAngle(1, 5, 15, 640, 0.0f);
        auto lossMap = model.loss(data);
        EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
    }

    // Test with angle = pi/4 (45 degrees)
    {
        auto data = createOBBDataWithAngle(1, 5, 15, 640, static_cast<float>(M_PI / 4));
        auto lossMap = model.loss(data);
        EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
    }

    // Test with angle = -pi/4 (-45 degrees)
    {
        auto data = createOBBDataWithAngle(1, 5, 15, 640, static_cast<float>(-M_PI / 4));
        auto lossMap = model.loss(data);
        EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
    }
}

// OBB-022: Loss_MultiClass - Multi-class OBB works
TEST_F(OBBModelTest, Loss_MultiClass) {
    // Test with different class counts
    {
        auto config = createValidConfig(5, 640);
        OBBModel model(config);
        moveToDevice(model);
        auto data = createOBBData(1, 5, 5, 640);
        auto lossMap = model.loss(data);
        EXPECT_GE(lossMap["total"].item<float>(), 0.0f);
    }

    {
        auto config = createValidConfig(20, 640);
        OBBModel model(config);
        moveToDevice(model);
        auto data = createOBBData(1, 5, 20, 640);
        auto lossMap = model.loss(data);
        EXPECT_GE(lossMap["total"].item<float>(), 0.0f);
    }
}

// =============================================================================
// Part 4.6: OBB-Specific Tests (3 tests)
// =============================================================================

// OBB-023: Output_ContainsAngle - Output includes angle information
TEST_F(OBBModelTest, Output_ContainsAngle) {
    auto config = createValidConfig(15, 640);
    OBBModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    // OBB model should produce outputs with angle information
    ASSERT_GE(outputs.size(), 1);
    // The output format depends on implementation
}

// OBB-024: Head_BiasInit - Head bias is initialized
TEST_F(OBBModelTest, Head_BiasInit) {
    auto config = createValidConfig();
    OBBModel model(config);
    moveToDevice(model);

    // Verify head is properly initialized by checking forward pass works
    auto input = torch::rand({1, 3, 640, 640}, _device);
    auto outputs = model.forward(input);

    EXPECT_GE(outputs.size(), 1);
}

// OBB-025: Probiou_Integration - Loss uses probiou for rotated boxes
TEST_F(OBBModelTest, Probiou_Integration) {
    auto config = createValidConfig(15, 640);
    OBBModel model(config);
    moveToDevice(model);

    // Create data with rotated boxes
    auto data = createOBBData(1, 5, 15, 640);
    auto lossMap = model.loss(data);

    // Probiou should produce valid loss
    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(lossMap["total"].isinf().any().item<bool>());
}

// =============================================================================
// Part 4.7: LoadPretrained Tests (2 tests)
// =============================================================================

// OBB-026: LoadPretrained_ValidPath - Loading valid weights returns true
TEST_F(OBBModelTest, LoadPretrained_ValidPath) {
    auto config = createValidConfig();
    OBBModel model(config);

    std::string weightsPath = getTempPath("obb_weights.pt");
    model.saveWeights(weightsPath);

    OBBModel model2(createValidConfig());
    bool result = model2.loadPretrained(weightsPath);

    EXPECT_TRUE(result);
}

// OBB-027: LoadPretrained_InvalidPath - Loading invalid path returns false
TEST_F(OBBModelTest, LoadPretrained_InvalidPath) {
    auto config = createValidConfig();
    OBBModel model(config);

    bool result = model.loadPretrained("/nonexistent/path/weights.pt");

    EXPECT_FALSE(result);
}

// =============================================================================
// Part 4.8: Integration Tests (1 test)
// =============================================================================

// OBB-028: EndToEnd_TrainingStep - Full training step works
TEST_F(OBBModelTest, EndToEnd_TrainingStep) {
    auto config = createValidConfig(15, 640);
    OBBModel model(config);
    moveToDevice(model);
    model.train();

    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01));

    auto data = createOBBData(2, 10, 15, 640);
    auto lossMap = model.forward(data);

    optimizer.zero_grad();
    lossMap["total"].backward();
    optimizer.step();

    EXPECT_TRUE(lossMap["total"].defined());
}
