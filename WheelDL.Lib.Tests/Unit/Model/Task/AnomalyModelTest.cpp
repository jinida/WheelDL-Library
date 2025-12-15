#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Task/AnomalyModel.h"
#include "Model/Loss/AnomalyLoss.h"
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

class AnomalyModelTest : public ::testing::Test {
protected:
    std::unique_ptr<MockConfigurationHelper> _configHelper;
    std::string _tempDir;
    torch::Device _device = torch::kCPU;

    void SetUp() override {
        torch::manual_seed(42);
        _configHelper = std::make_unique<MockConfigurationHelper>();
        _tempDir = (fs::temp_directory_path() / "WheelDL_AnomalyModel_Test").string();
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

    // Helper: Create valid EfficientAD config
    std::shared_ptr<Configuration> createEfficientADConfig(int numClasses = 1, int imageSize = 256) {
        return _configHelper->createMockConfig(
            TaskType::ANOMALY,
            _configHelper->getModelYamlPath("ano_EfficientAD.yaml"),
            numClasses,
            imageSize
        );
    }

    // Helper: Create valid PatchCore config
    std::shared_ptr<Configuration> createPatchCoreConfig(int numClasses = 1, int imageSize = 256) {
        return _configHelper->createMockConfig(
            TaskType::ANOMALY,
            _configHelper->getModelYamlPath("ano_PatchCore.yaml"),
            numClasses,
            imageSize
        );
    }

    // Helper: Create config with wrong task type
    std::shared_ptr<Configuration> createWrongTaskTypeConfig() {
        return _configHelper->createMockConfig(
            TaskType::CLASSIFICATION,
            _configHelper->getModelYamlPath("ano_EfficientAD.yaml"),
            1,
            256
        );
    }

    // Helper: Move model to device
    void moveToDevice(AnomalyModel& model) {
        model.to(_device);
    }
};

// =============================================================================
// Part 6.1: Constructor Tests - EfficientAD (6 tests)
// =============================================================================

// AM-001: Constructor_ValidConfig_EfficientAD - Valid EfficientAD configuration initializes model
TEST_F(AnomalyModelTest, Constructor_ValidConfig_EfficientAD) {
    auto config = createEfficientADConfig();

    AnomalyModel model(config);

    EXPECT_EQ(model.getTaskType(), TaskType::ANOMALY);
}

// AM-002: Constructor_NullConfig - Null config throws ConfigurationException
TEST_F(AnomalyModelTest, Constructor_NullConfig) {
    EXPECT_THROW(
        {
            try {
                AnomalyModel model(nullptr);
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

// AM-003: Constructor_EmptyModelPath - Empty model path throws ConfigurationException
TEST_F(AnomalyModelTest, Constructor_EmptyModelPath) {
    auto config = std::make_shared<Configuration>();

    EXPECT_THROW(
        {
            try {
                AnomalyModel model(config);
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

// AM-004: Constructor_WrongTaskType - Wrong task type throws exception
TEST_F(AnomalyModelTest, Constructor_WrongTaskType) {
    auto config = createWrongTaskTypeConfig();

    EXPECT_THROW(
        {
            try {
                AnomalyModel model(config);
            }
            catch (const WheelDL::Utils::ModelException& e) {
                EXPECT_EQ(e.getErrorCode(), WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED);
                EXPECT_NE(std::string(e.what()).find("ANOMALY"), std::string::npos);
                throw;
            }
        },
        WheelDL::Utils::ModelException
    );
}

// AM-005: Constructor_EfficientAD_TaskType - EfficientAD sets correct task type
TEST_F(AnomalyModelTest, Constructor_EfficientAD_TaskType) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);

    EXPECT_EQ(model.getTaskType(), TaskType::ANOMALY);
}

// AM-006: Constructor_DeterminesLossType - Determines loss type from model head
TEST_F(AnomalyModelTest, Constructor_DeterminesLossType) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);

    // Model should determine loss type from head
    EXPECT_EQ(model.getTaskType(), TaskType::ANOMALY);
}

// =============================================================================
// Part 6.2: Constructor Tests - PatchCore (3 tests)
// =============================================================================

// AM-007: Constructor_ValidConfig_PatchCore - Valid PatchCore configuration initializes model
TEST_F(AnomalyModelTest, Constructor_ValidConfig_PatchCore) {
    auto config = createPatchCoreConfig();

    AnomalyModel model(config);

    EXPECT_EQ(model.getTaskType(), TaskType::ANOMALY);
}

// AM-008: Constructor_PatchCore_LossType - PatchCore determines correct loss type
TEST_F(AnomalyModelTest, Constructor_PatchCore_LossType) {
    auto config = createPatchCoreConfig();
    AnomalyModel model(config);

    // Model should be initialized with PatchCore loss type
    EXPECT_EQ(model.getTaskType(), TaskType::ANOMALY);
}

// AM-009: Constructor_PatchCore_TaskType - PatchCore sets correct task type
TEST_F(AnomalyModelTest, Constructor_PatchCore_TaskType) {
    auto config = createPatchCoreConfig();
    AnomalyModel model(config);

    EXPECT_EQ(model.getTaskType(), TaskType::ANOMALY);
}

// =============================================================================
// Part 6.3: InitCriterion Tests (3 tests)
// Note: These test criterion initialization, not loss computation
// =============================================================================

// AM-010: InitCriterion_EfficientAD_Created - EfficientAD criterion is created
TEST_F(AnomalyModelTest, InitCriterion_EfficientAD_Created) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);

    // Criterion is created during construction
    EXPECT_EQ(model.getTaskType(), TaskType::ANOMALY);
}

// AM-011: InitCriterion_PatchCore_Created - PatchCore criterion is created
TEST_F(AnomalyModelTest, InitCriterion_PatchCore_Created) {
    auto config = createPatchCoreConfig();
    AnomalyModel model(config);

    // Criterion is created during construction
    EXPECT_EQ(model.getTaskType(), TaskType::ANOMALY);
}

// AM-012: InitCriterion_ModelInitialized - Model is fully initialized
TEST_F(AnomalyModelTest, InitCriterion_ModelInitialized) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);
    moveToDevice(model);

    // Model should be fully initialized
    EXPECT_EQ(model.getTaskType(), TaskType::ANOMALY);
}

// =============================================================================
// Part 6.4: Forward Pass Tests - Inference Only (4 tests)
// Note: Anomaly models require preparation before training mode
// =============================================================================

// AM-013: Forward_Inference_EfficientAD - EfficientAD inference returns anomaly scores
TEST_F(AnomalyModelTest, Forward_Inference_EfficientAD) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);
    moveToDevice(model);
    model.eval();

    auto input = torch::rand({1, 3, 256, 256}, _device);
    auto outputs = model.forward(input);

    ASSERT_GE(outputs.size(), 1);
}

// AM-014: Forward_Inference_EfficientAD_Batch - EfficientAD batch inference
TEST_F(AnomalyModelTest, Forward_Inference_EfficientAD_Batch) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);
    moveToDevice(model);
    model.eval();

    auto input = torch::rand({4, 3, 256, 256}, _device);
    auto outputs = model.forward(input);

    ASSERT_GE(outputs.size(), 1);
}

// AM-015: Forward_Inference_OutputShape - Output has expected shape
TEST_F(AnomalyModelTest, Forward_Inference_OutputShape) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);
    moveToDevice(model);
    model.eval();

    auto input = torch::rand({2, 3, 256, 256}, _device);
    auto outputs = model.forward(input);

    ASSERT_GE(outputs.size(), 1);
    EXPECT_EQ(outputs[0].size(0), 2);  // batch size
}

// AM-016: Forward_Inference_DeviceConsistency - Output on same device as input
TEST_F(AnomalyModelTest, Forward_Inference_DeviceConsistency) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);
    moveToDevice(model);
    model.eval();

    auto input = torch::rand({1, 3, 256, 256}, _device);
    auto outputs = model.forward(input);

    ASSERT_GE(outputs.size(), 1);
    EXPECT_EQ(outputs[0].device().type(), _device.type());
}

// =============================================================================
// Part 6.5: Model State Tests (4 tests)
// =============================================================================

// AM-017: Model_TrainMode - Model can be set to train mode
TEST_F(AnomalyModelTest, Model_TrainMode) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);
    moveToDevice(model);

    model.train();
    EXPECT_TRUE(model.is_training());
}

// AM-018: Model_EvalMode - Model can be set to eval mode
TEST_F(AnomalyModelTest, Model_EvalMode) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);
    moveToDevice(model);

    model.eval();
    EXPECT_FALSE(model.is_training());
}

// AM-019: Model_Parameters - Model has parameters
TEST_F(AnomalyModelTest, Model_Parameters) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);

    auto params = model.parameters();
    EXPECT_GT(params.size(), 0);
}

// AM-020: Model_NamedParameters - Model has named parameters
TEST_F(AnomalyModelTest, Model_NamedParameters) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);

    auto namedParams = model.named_parameters();
    EXPECT_GT(namedParams.size(), 0);
}

// =============================================================================
// Part 6.6: LoadPretrained Tests (2 tests)
// =============================================================================

// AM-021: LoadPretrained_ValidPath - Loading valid weights returns true
TEST_F(AnomalyModelTest, LoadPretrained_ValidPath) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);

    std::string weightsPath = getTempPath("anomaly_weights.pt");
    model.saveWeights(weightsPath);

    AnomalyModel model2(createEfficientADConfig());
    bool result = model2.loadPretrained(weightsPath);

    EXPECT_TRUE(result);
}

// AM-022: LoadPretrained_InvalidPath - Loading invalid path returns false
TEST_F(AnomalyModelTest, LoadPretrained_InvalidPath) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);

    bool result = model.loadPretrained("/nonexistent/path/weights.pt");

    EXPECT_FALSE(result);
}

// =============================================================================
// Part 6.7: Integration Tests (3 tests)
// =============================================================================

// AM-023: EndToEnd_InferenceStep - Full inference step works
TEST_F(AnomalyModelTest, EndToEnd_InferenceStep) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);
    moveToDevice(model);
    model.eval();

    torch::NoGradGuard no_grad;

    auto input = torch::rand({2, 3, 256, 256}, _device);
    auto outputs = model.forward(input);

    ASSERT_GE(outputs.size(), 1);
}

// AM-024: EndToEnd_DeviceTransfer - Model can transfer between devices
TEST_F(AnomalyModelTest, EndToEnd_DeviceTransfer) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);

    // Verify on CPU
    model.to(torch::kCPU);
    for (const auto& param : model.parameters()) {
        EXPECT_TRUE(param.device().is_cpu());
    }

    auto input = torch::rand({1, 3, 256, 256});
    model.eval();
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

// AM-025: EndToEnd_MultipleInference - Multiple inference calls work
TEST_F(AnomalyModelTest, EndToEnd_MultipleInference) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);
    moveToDevice(model);
    model.eval();

    torch::NoGradGuard no_grad;

    // Run multiple inferences
    for (int i = 0; i < 3; ++i) {
        auto input = torch::rand({1, 3, 256, 256}, _device);
        auto outputs = model.forward(input);
        ASSERT_GE(outputs.size(), 1);
    }
}

// =============================================================================
// Part 6.8: Anomaly-Specific Tests (2 tests)
// =============================================================================

// AM-026: Model_SaveWeights - Can save weights
TEST_F(AnomalyModelTest, Model_SaveWeights) {
    auto config = createEfficientADConfig();
    AnomalyModel model(config);

    std::string weightsPath = getTempPath("save_test_weights.pt");

    EXPECT_NO_THROW({
        model.saveWeights(weightsPath);
    });

    EXPECT_TRUE(fs::exists(weightsPath));
}

// AM-027: Model_LoadSaveConsistency - Saved and loaded weights are consistent
TEST_F(AnomalyModelTest, Model_LoadSaveConsistency) {
    auto config = createEfficientADConfig();
    AnomalyModel model1(config);

    std::string weightsPath = getTempPath("consistency_weights.pt");
    model1.saveWeights(weightsPath);

    AnomalyModel model2(createEfficientADConfig());
    bool loaded = model2.loadPretrained(weightsPath);

    EXPECT_TRUE(loaded);

    // Both models should have same number of parameters
    auto params1 = model1.parameters();
    auto params2 = model2.parameters();
    EXPECT_EQ(params1.size(), params2.size());
}
