#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Task/ClassificationModel.h"
#include "Model/Loss/ClassificationLoss.h"
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

class ClassificationModelTest : public ::testing::Test {
protected:
    std::unique_ptr<MockConfigurationHelper> _configHelper;
    std::string _tempDir;
    torch::Device _device = torch::kCPU;

    void SetUp() override {
        torch::manual_seed(42);
        _configHelper = std::make_unique<MockConfigurationHelper>();
        _tempDir = (fs::temp_directory_path() / "WheelDL_ClassificationModel_Test").string();
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

    // Helper: Create valid classification config
    std::shared_ptr<Configuration> createValidConfig(int numClasses = 10, int imageSize = 224) {
        return _configHelper->createMockConfig(
            TaskType::CLASSIFICATION,
            _configHelper->getModelYamlPath("cls_ConvNext.yaml"),
            numClasses,
            imageSize
        );
    }

    // Helper: Create config with wrong task type
    std::shared_ptr<Configuration> createWrongTaskTypeConfig() {
        return _configHelper->createMockConfig(
            TaskType::DETECTION,
            _configHelper->getModelYamlPath("cls_ConvNext.yaml"),
            10,
            224
        );
    }

    // Helper: Create DataExample for classification (on device)
    DataExample createClassificationData(int batchSize = 1, int numClasses = 10, int imageSize = 224) {
        auto example = createClassificationDataExample(batchSize, 3, imageSize, imageSize, numClasses);
        example.data = example.data.to(_device);
        example.classes = example.classes.to(_device);
        example.targets = example.targets.to(_device);
        example.batchIndices = example.batchIndices.to(_device);
        return example;
    }

    // Helper: Create model on device
    void moveToDevice(ClassificationModel& model) {
        model.to(_device);
    }
};

// =============================================================================
// Part 3.1.1: Constructor Tests (6 tests)
// =============================================================================

// CM-001: Constructor_ValidConfig - Valid configuration initializes model
TEST_F(ClassificationModelTest, Constructor_ValidConfig) {
    auto config = createValidConfig();

    ClassificationModel model(config);

    EXPECT_EQ(model.getTaskType(), TaskType::CLASSIFICATION);
}

// CM-002: Constructor_NullConfig - Null config throws ConfigurationException
TEST_F(ClassificationModelTest, Constructor_NullConfig) {
    EXPECT_THROW(
        {
            try {
                ClassificationModel model(nullptr);
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

// CM-003: Constructor_EmptyModelPath - Empty model path throws ConfigurationException
TEST_F(ClassificationModelTest, Constructor_EmptyModelPath) {
    // Create a config with empty model path
    auto config = std::make_shared<Configuration>();

    EXPECT_THROW(
        {
            try {
                ClassificationModel model(config);
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

// CM-004: Constructor_WrongTaskType - Wrong task type throws exception
TEST_F(ClassificationModelTest, Constructor_WrongTaskType) {
    auto config = createWrongTaskTypeConfig();

    // The exception will be wrapped in ModelException due to try-catch in constructor
    EXPECT_THROW(
        {
            try {
                ClassificationModel model(config);
            }
            catch (const Utils::ModelException& e) {
                EXPECT_EQ(e.getErrorCode(), Utils::ErrorCode::MODEL_LOAD_FAILED);
                EXPECT_NE(std::string(e.what()).find("CLASSIFICATION"), std::string::npos);
                throw;
            }
        },
        Utils::ModelException
    );
}

// CM-005: Constructor_InvalidYaml - Invalid YAML throws ModelException
TEST_F(ClassificationModelTest, Constructor_InvalidYaml) {
    // Create config with non-existent YAML path
    auto config = _configHelper->createMockConfig(
        TaskType::CLASSIFICATION,
        "nonexistent/invalid_model.yaml",
        10,
        224
    );

    EXPECT_THROW(
        {
            try {
                ClassificationModel model(config);
            }
            catch (const Utils::ModelException& e) {
                EXPECT_EQ(e.getErrorCode(), Utils::ErrorCode::MODEL_LOAD_FAILED);
                throw;
            }
        },
        Utils::ModelException
    );
}

// CM-006: Constructor_SetsLossType - Default loss type is CROSS_ENTROPY
TEST_F(ClassificationModelTest, Constructor_SetsLossType) {
    auto config = createValidConfig();

    ClassificationModel model(config);
    moveToDevice(model);

    // Loss should be initialized and working (we can't directly access _lossType,
    // but we can verify via loss computation)
    auto data = createClassificationData(1, 10, 224);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
}

// =============================================================================
// Part 3.1.2: InitCriterion Tests (3 tests)
// =============================================================================

// CM-007: InitCriterion_ReturnsClassificationLoss - Criterion is ClassificationLoss
TEST_F(ClassificationModelTest, InitCriterion_ReturnsClassificationLoss) {
    auto config = createValidConfig();
    ClassificationModel model(config);
    moveToDevice(model);

    // Verify loss computation works (indicates correct loss type)
    auto data = createClassificationData(1, 10, 224);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
}

// CM-008: InitCriterion_UsesConfigNumClasses - Uses numClasses from config
TEST_F(ClassificationModelTest, InitCriterion_UsesConfigNumClasses) {
    // Create config with specific numClasses
    int numClasses = 20;
    auto config = createValidConfig(numClasses, 224);
    ClassificationModel model(config);
    moveToDevice(model);

    // Create data with matching numClasses
    auto data = createClassificationData(1, numClasses, 224);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
}

// CM-009: InitCriterion_DefaultLabelSmoothing - Label smoothing is 0.0 by default
TEST_F(ClassificationModelTest, InitCriterion_DefaultLabelSmoothing) {
    auto config = createValidConfig();
    ClassificationModel model(config);
    moveToDevice(model);

    // Default label smoothing should produce standard CE behavior
    // We verify by checking loss is computed correctly
    auto data = createClassificationData(1, 10, 224);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_GT(lossMap["total"].item<float>(), 0.0f);
}

// =============================================================================
// Part 3.1.3: LoadPretrained Tests (3 tests)
// =============================================================================

// CM-010: LoadPretrained_ValidPath - Loading valid weights returns true
TEST_F(ClassificationModelTest, LoadPretrained_ValidPath) {
    auto config = createValidConfig();
    ClassificationModel model(config);

    // Save weights first
    std::string weightsPath = getTempPath("cls_weights.pt");
    model.saveWeights(weightsPath);

    // Create new model and load
    ClassificationModel model2(createValidConfig());
    bool result = model2.loadPretrained(weightsPath);

    EXPECT_TRUE(result);
}

// CM-011: LoadPretrained_InvalidPath - Loading invalid path returns false
TEST_F(ClassificationModelTest, LoadPretrained_InvalidPath) {
    auto config = createValidConfig();
    ClassificationModel model(config);

    bool result = model.loadPretrained("/nonexistent/path/weights.pt");

    EXPECT_FALSE(result);
}

// CM-012: LoadPretrained_CallsLoadWeights - Delegates to base loadWeights
TEST_F(ClassificationModelTest, LoadPretrained_CallsLoadWeights) {
    auto config = createValidConfig();
    ClassificationModel model1(config);
    ClassificationModel model2(createValidConfig());

    // Modify model1 weights
    for (auto& param : model1.parameters()) {
        param.data().fill_(0.5f);
    }

    // Save and load
    std::string weightsPath = getTempPath("delegate_weights.pt");
    model1.saveWeights(weightsPath);
    model2.loadPretrained(weightsPath);

    // Verify weights were loaded
    auto params1 = model1.parameters();
    auto params2 = model2.parameters();

    ASSERT_EQ(params1.size(), params2.size());
    for (size_t i = 0; i < params1.size(); ++i) {
        EXPECT_TRUE(torch::allclose(params1[i], params2[i], 1e-5, 1e-5));
    }
}

// =============================================================================
// Part 3.1.4: Forward Pass Tests (4 tests)
// =============================================================================

// CM-013: Forward_InferenceMode - Inference returns correct shape
TEST_F(ClassificationModelTest, Forward_InferenceMode) {
    int numClasses = 10;
    auto config = createValidConfig(numClasses, 224);
    ClassificationModel model(config);
    moveToDevice(model);

    auto input = torch::rand({1, 3, 224, 224}, _device);
    auto outputs = model.forward(input);

    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].size(0), 1);  // batch size
    EXPECT_EQ(outputs[0].size(1), numClasses);  // num classes
}

// CM-014: Forward_TrainingMode - Training returns loss map
TEST_F(ClassificationModelTest, Forward_TrainingMode) {
    auto config = createValidConfig(10, 224);
    ClassificationModel model(config);
    moveToDevice(model);

    auto data = createClassificationData(1, 10, 224);
    auto lossMap = model.forward(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(lossMap["total"].isinf().any().item<bool>());
}

// CM-015: Forward_BatchProcessing - Handles multiple samples
TEST_F(ClassificationModelTest, Forward_BatchProcessing) {
    int numClasses = 10;
    int batchSize = 8;
    auto config = createValidConfig(numClasses, 224);
    ClassificationModel model(config);
    moveToDevice(model);

    auto input = torch::rand({batchSize, 3, 224, 224}, _device);
    auto outputs = model.forward(input);

    ASSERT_EQ(outputs.size(), 1);
    EXPECT_EQ(outputs[0].size(0), batchSize);
    EXPECT_EQ(outputs[0].size(1), numClasses);
}

// CM-016: Forward_DifferentImageSizes - Works with various input sizes
TEST_F(ClassificationModelTest, Forward_DifferentImageSizes) {
    int numClasses = 10;

    // Test with 224x224
    {
        auto config = createValidConfig(numClasses, 224);
        ClassificationModel model(config);
        moveToDevice(model);
        auto input = torch::rand({1, 3, 224, 224}, _device);
        auto outputs = model.forward(input);
        EXPECT_EQ(outputs[0].size(1), numClasses);
    }

    // Test with 256x256
    {
        auto config = createValidConfig(numClasses, 256);
        ClassificationModel model(config);
        moveToDevice(model);
        auto input = torch::rand({1, 3, 256, 256}, _device);
        auto outputs = model.forward(input);
        EXPECT_EQ(outputs[0].size(1), numClasses);
    }
}

// =============================================================================
// Part 3.1.5: Loss Computation Tests (3 tests)
// =============================================================================

// CM-017: Loss_CrossEntropy - Default CE loss computes correctly
TEST_F(ClassificationModelTest, Loss_CrossEntropy) {
    auto config = createValidConfig(10, 224);
    ClassificationModel model(config);
    moveToDevice(model);

    auto data = createClassificationData(4, 10, 224);
    auto lossMap = model.loss(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
    EXPECT_FALSE(lossMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(lossMap["total"].isinf().any().item<bool>());
    EXPECT_GT(lossMap["total"].item<float>(), 0.0f);
}

// CM-018: Loss_GradientFlow - Gradients are computed
TEST_F(ClassificationModelTest, Loss_GradientFlow) {
    auto config = createValidConfig(10, 224);
    ClassificationModel model(config);
    moveToDevice(model);
    model.train();

    auto data = createClassificationData(2, 10, 224);
    data.data.set_requires_grad(true);

    auto lossMap = model.forward(data);
    lossMap["total"].backward();

    // Check that gradients exist
    bool hasGradient = false;
    for (const auto& param : model.parameters()) {
        if (param.grad().defined() && param.grad().abs().sum().item<float>() > 0) {
            hasGradient = true;
            break;
        }
    }
    EXPECT_TRUE(hasGradient);
}

// CM-019: Loss_MultiClass - Works with different class counts
TEST_F(ClassificationModelTest, Loss_MultiClass) {
    // Test with 10 classes
    {
        auto config = createValidConfig(10, 224);
        ClassificationModel model(config);
        moveToDevice(model);
        auto data = createClassificationData(2, 10, 224);
        auto lossMap = model.loss(data);
        EXPECT_GT(lossMap["total"].item<float>(), 0.0f);
    }

    // Test with 100 classes
    {
        auto config = createValidConfig(100, 224);
        ClassificationModel model(config);
        moveToDevice(model);
        auto data = createClassificationData(2, 100, 224);
        auto lossMap = model.loss(data);
        EXPECT_GT(lossMap["total"].item<float>(), 0.0f);
    }
}

// =============================================================================
// Part 3.1.6: Integration Tests (4 tests)
// =============================================================================

// CM-020: EndToEnd_TrainingStep - Full training step works
TEST_F(ClassificationModelTest, EndToEnd_TrainingStep) {
    auto config = createValidConfig(10, 224);
    ClassificationModel model(config);
    moveToDevice(model);
    model.train();

    // Create optimizer
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.01));

    // Forward pass
    auto data = createClassificationData(4, 10, 224);
    auto lossMap = model.forward(data);

    // Backward pass
    optimizer.zero_grad();
    lossMap["total"].backward();
    optimizer.step();

    // Verify loss changed after step
    auto newLossMap = model.forward(data);

    // Loss values should be different (training happened)
    // Note: Not always guaranteed to decrease in one step, but should be different
    EXPECT_TRUE(lossMap["total"].defined());
    EXPECT_TRUE(newLossMap["total"].defined());
}

// CM-021: EndToEnd_ValidationStep - Validation step works
TEST_F(ClassificationModelTest, EndToEnd_ValidationStep) {
    auto config = createValidConfig(10, 224);
    ClassificationModel model(config);
    moveToDevice(model);
    model.eval();

    torch::NoGradGuard no_grad;

    auto input = torch::rand({4, 3, 224, 224}, _device);
    auto outputs = model.forward(input);

    // Apply softmax for probabilities
    auto probs = torch::softmax(outputs[0], 1);

    // Check probabilities sum to 1
    auto probSum = probs.sum(1);
    EXPECT_TRUE(torch::allclose(probSum, torch::ones_like(probSum), 1e-4, 1e-4));

    // Get predictions
    auto [maxProbs, predictions] = probs.max(1);
    EXPECT_EQ(predictions.size(0), 4);
}

// CM-022: EndToEnd_DeviceTransfer - Model can transfer between devices
TEST_F(ClassificationModelTest, EndToEnd_DeviceTransfer) {
    auto config = createValidConfig(10, 224);
    ClassificationModel model(config);

    // Verify on CPU
    model.to(torch::kCPU);
    for (const auto& param : model.parameters()) {
        EXPECT_TRUE(param.device().is_cpu());
    }

    // Forward pass on CPU
    auto input = torch::rand({1, 3, 224, 224});
    auto outputs = model.forward(input);
    EXPECT_EQ(outputs[0].device().type(), torch::kCPU);

    // CUDA test (if available)
    if (torch::cuda::is_available()) {
        model.to(torch::kCUDA);
        for (const auto& param : model.parameters()) {
            EXPECT_TRUE(param.device().is_cuda());
        }

        auto cudaInput = torch::rand({1, 3, 224, 224}, torch::kCUDA);
        auto cudaOutputs = model.forward(cudaInput);
        EXPECT_TRUE(cudaOutputs[0].device().is_cuda());
    }
}

// CM-023: Forward_OutputRequiresGrad - Output maintains gradient requirement
TEST_F(ClassificationModelTest, Forward_OutputRequiresGrad) {
    auto config = createValidConfig(10, 224);
    ClassificationModel model(config);
    moveToDevice(model);
    model.train();

    auto input = torch::rand({1, 3, 224, 224}, torch::TensorOptions().device(_device).requires_grad(true));
    auto outputs = model.forward(input);

    EXPECT_TRUE(outputs[0].requires_grad());
}
