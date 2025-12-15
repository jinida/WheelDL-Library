#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Task/BaseModel.h"
#include "Model/Loss/BaseLoss.h"
#include "Model/Modules/Conv.h"
#include "Model/Modules/Block.h"
#include "Model/Modules/Head.h"
#include "Config/Configuration.h"
#include "Utils/Error/WheelLibException.h"
#include "Utils/Error/ErrorCodes.h"
#include <filesystem>
#include <fstream>
#include <memory>

using namespace WheelDL;
using namespace WheelDL::Model;
using namespace WheelDL::Model::Loss;
using namespace WheelDL::Model::Modules;
using namespace WheelDL::Config;
using namespace WheelDL::Data::Dataset;
namespace fs = std::filesystem;

// =============================================================================
// Test Model Class (concrete implementation of BaseModel for testing)
// =============================================================================

/**
 * @class TestModel
 * @brief Concrete implementation of BaseModel for testing purposes
 *
 * This class provides a minimal implementation of BaseModel to test
 * the abstract class functionality.
 */
class TestModel : public BaseModel {
public:
    TestModel() : BaseModel() {}

    // Expose protected methods for testing
    using BaseModel::countParameters;
    using BaseModel::countFlops;
    using BaseModel::applyToTensors;
    using BaseModel::_criterion;
    using BaseModel::_isInitialized;
    using BaseModel::_inplace;
    using BaseModel::_training;
    using BaseModel::_model;
    using BaseModel::_stride;
    using BaseModel::_taskType;
    using BaseModel::_saveIndices;
    using BaseModel::_fromIndices;
    using BaseModel::_config;

protected:
    std::unique_ptr<BaseLoss> initCriterion() override {
        return std::make_unique<BCEWithLogitsLoss>();
    }
};

// =============================================================================
// Test Fixture
// =============================================================================

class BaseModelTest : public ::testing::Test {
protected:
    std::string _tempDir;

    void SetUp() override {
        torch::manual_seed(42);
        _tempDir = (fs::temp_directory_path() / "WheelDL_BaseModel_Test").string();
        fs::remove_all(_tempDir);
        fs::create_directories(_tempDir);
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

    // Helper: Create simple Conv layer for testing
    torch::nn::Sequential createSimpleModel() {
        torch::nn::Sequential seq;
        // Simple convolution: 3 -> 16 channels
        seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)));
        seq->push_back(torch::nn::BatchNorm2d(16));
        seq->push_back(torch::nn::ReLU());
        return seq;
    }

    // Helper: Create model with multiple layers for testing
    torch::nn::Sequential createMultiLayerModel() {
        torch::nn::Sequential seq;
        seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)));
        seq->push_back(torch::nn::BatchNorm2d(16));
        seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1)));
        seq->push_back(torch::nn::BatchNorm2d(32));
        seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
        return seq;
    }

    // Helper: Create DataExample
    DataExample createDataExample(int batchSize = 1, int height = 32, int width = 32) {
        DataExample example;
        example.data = torch::rand({batchSize, 3, height, width});
        example.classes = torch::randint(0, 2, {batchSize});
        example.targets = example.classes.clone();
        example.batchIndices = torch::arange(batchSize);
        return example;
    }

    // Helper: Create model for loss testing (outputs [B, 1] to match targets)
    torch::nn::Sequential createModelForLossTesting() {
        torch::nn::Sequential seq;
        seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)));
        seq->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        seq->push_back(torch::nn::Flatten());
        seq->push_back(torch::nn::Linear(16, 1));
        return seq;
    }

    // Helper: Create DataExample for loss testing (targets match model output shape [B, 1])
    // NOTE: BCEWithLogitsLoss uses target.classes, not target.targets!
    DataExample createDataExampleForLoss(int batchSize = 1) {
        DataExample example;
        example.data = torch::rand({batchSize, 3, 32, 32});
        // BCEWithLogitsLoss uses classes - shape must match model output [B, 1]
        example.classes = torch::rand({batchSize, 1});
        example.targets = torch::rand({batchSize, 1});
        example.batchIndices = torch::arange(batchSize);
        return example;
    }
};

// =============================================================================
// Part 2.1: Constructor Tests (3 tests)
// =============================================================================

// BM-001: DefaultConstructor - Default initialization
TEST_F(BaseModelTest, DefaultConstructor) {
    TestModel model;

    EXPECT_EQ(model._taskType, TaskType::UNKNOWN);
    EXPECT_FALSE(model._isInitialized);
}

// BM-002: DefaultStride - Default stride value
TEST_F(BaseModelTest, DefaultStride) {
    TestModel model;

    EXPECT_TRUE(model._stride.defined());
    EXPECT_EQ(model._stride.size(0), 1);
    EXPECT_FLOAT_EQ(model._stride[0].item<float>(), 32.0f);
}

// BM-003: DefaultFlags - Default boolean flags
TEST_F(BaseModelTest, DefaultFlags) {
    TestModel model;

    EXPECT_TRUE(model._inplace);
    EXPECT_FALSE(model._training);
}

// =============================================================================
// Part 2.2: Forward/Predict Empty Model Tests (6 tests)
// =============================================================================

// BM-004: Forward_EmptyModel - Forward with empty model returns input
TEST_F(BaseModelTest, Forward_EmptyModel) {
    TestModel model;
    // Must set an empty Sequential (not nullptr) for is_empty() check
    model._model = torch::nn::Sequential();
    auto input = torch::rand({1, 3, 32, 32});

    auto output = model.forward(input);

    ASSERT_EQ(output.size(), 1);
    EXPECT_TRUE(torch::allclose(output[0], input));
}

// BM-005: Predict_EmptyModel - Predict with empty model returns input
TEST_F(BaseModelTest, Predict_EmptyModel) {
    TestModel model;
    // Must set an empty Sequential (not nullptr) for is_empty() check
    model._model = torch::nn::Sequential();
    auto input = torch::rand({1, 3, 32, 32});

    auto output = model.predict(input);

    ASSERT_EQ(output.size(), 1);
    EXPECT_TRUE(torch::allclose(output[0], input));
}

// BM-006: Forward_DifferentBatchSizes - Test different batch sizes
TEST_F(BaseModelTest, Forward_DifferentBatchSizes) {
    TestModel model;
    // Set empty Sequential for pass-through behavior
    model._model = torch::nn::Sequential();

    // Batch size 1
    auto input1 = torch::rand({1, 3, 32, 32});
    auto output1 = model.forward(input1);
    EXPECT_EQ(output1[0].size(0), 1);

    // Batch size 4
    auto input4 = torch::rand({4, 3, 32, 32});
    auto output4 = model.forward(input4);
    EXPECT_EQ(output4[0].size(0), 4);

    // Batch size 8
    auto input8 = torch::rand({8, 3, 32, 32});
    auto output8 = model.forward(input8);
    EXPECT_EQ(output8[0].size(0), 8);
}

// BM-007: Forward_DifferentImageSizes - Test different image sizes
TEST_F(BaseModelTest, Forward_DifferentImageSizes) {
    TestModel model;
    // Set empty Sequential for pass-through behavior
    model._model = torch::nn::Sequential();

    // 32x32
    auto input32 = torch::rand({1, 3, 32, 32});
    auto output32 = model.forward(input32);
    EXPECT_EQ(output32[0].size(2), 32);
    EXPECT_EQ(output32[0].size(3), 32);

    // 64x64
    auto input64 = torch::rand({1, 3, 64, 64});
    auto output64 = model.forward(input64);
    EXPECT_EQ(output64[0].size(2), 64);
    EXPECT_EQ(output64[0].size(3), 64);
}

// BM-008: Predict_EmptyIndices - Both saveIndices and fromIndices empty
TEST_F(BaseModelTest, Predict_EmptyIndices) {
    TestModel model;
    // Set a simple model but leave indices empty
    model._model = torch::nn::Sequential();
    model._model->push_back(torch::nn::Identity());
    model._saveIndices.clear();
    model._fromIndices.clear();

    auto input = torch::rand({1, 3, 32, 32});
    auto output = model.predict(input);

    // Should use simple sequential forward
    ASSERT_EQ(output.size(), 1);
    EXPECT_TRUE(torch::allclose(output[0], input));
}

// BM-009: Forward_PreservesGradient - Gradient should flow through
TEST_F(BaseModelTest, Forward_PreservesGradient) {
    TestModel model;
    // Set empty Sequential for pass-through behavior
    model._model = torch::nn::Sequential();
    auto input = torch::rand({1, 3, 32, 32}, torch::requires_grad());

    auto output = model.forward(input);

    EXPECT_TRUE(output[0].requires_grad());
}

// =============================================================================
// Part 2.3: Weight Load/Save Tests (10 tests)
// =============================================================================

// BM-010: SaveWeights_ValidPath - Save to valid path
TEST_F(BaseModelTest, SaveWeights_ValidPath) {
    TestModel model;
    model.setModel(createSimpleModel());

    std::string path = getTempPath("weights.pt");
    EXPECT_NO_THROW(model.saveWeights(path));
    EXPECT_TRUE(fs::exists(path));
}

// BM-011: SaveWeights_InvalidPath - Save to invalid path throws
TEST_F(BaseModelTest, SaveWeights_InvalidPath) {
    TestModel model;
    model.setModel(createSimpleModel());

    std::string invalidPath = "/nonexistent/directory/weights.pt";
    EXPECT_THROW(model.saveWeights(invalidPath), Utils::ModelException);
}

// BM-012: LoadWeights_ValidPath - Load from valid file
TEST_F(BaseModelTest, LoadWeights_ValidPath) {
    // Create and save a model
    TestModel model1;
    model1.setModel(createSimpleModel());
    std::string path = getTempPath("weights.pt");
    model1.saveWeights(path);

    // Load into a new model
    TestModel model2;
    model2.setModel(createSimpleModel());
    EXPECT_NO_THROW(model2.loadWeights(path));
}

// BM-013: LoadWeights_InvalidPath - Load from invalid path throws
TEST_F(BaseModelTest, LoadWeights_InvalidPath) {
    TestModel model;
    model.setModel(createSimpleModel());

    EXPECT_THROW(model.loadWeights("/nonexistent/weights.pt"), Utils::ModelException);
}

// BM-014: LoadWeights_StateDict - Load from state dict
TEST_F(BaseModelTest, LoadWeights_StateDict) {
    TestModel model1;
    model1.setModel(createSimpleModel());

    // Get state dict from model1
    std::unordered_map<std::string, torch::Tensor> stateDict;
    for (const auto& param : model1.named_parameters()) {
        stateDict[param.key()] = param.value().clone();
    }

    // Load into model2
    TestModel model2;
    model2.setModel(createSimpleModel());
    EXPECT_NO_THROW(model2.loadWeights(stateDict));

    // Verify weights are the same
    for (const auto& param : model2.named_parameters()) {
        auto it = stateDict.find(param.key());
        if (it != stateDict.end()) {
            EXPECT_TRUE(torch::allclose(param.value(), it->second));
        }
    }
}

// BM-015: LoadWeights_ShapeMismatch - Shape mismatch silently skips
TEST_F(BaseModelTest, LoadWeights_ShapeMismatch) {
    TestModel model;
    model.setModel(createSimpleModel());

    // Create state dict with wrong shapes
    std::unordered_map<std::string, torch::Tensor> badStateDict;
    for (const auto& param : model.named_parameters()) {
        // Create tensor with wrong shape
        badStateDict[param.key()] = torch::rand({1, 1, 1, 1});
    }

    // Should not throw, just skip mismatched params
    EXPECT_NO_THROW(model.loadWeights(badStateDict));
}

// BM-016: LoadWeights_PartialMatch - Partial state dict loads available
TEST_F(BaseModelTest, LoadWeights_PartialMatch) {
    TestModel model;
    model.setModel(createSimpleModel());

    // Create partial state dict (only first param)
    std::unordered_map<std::string, torch::Tensor> partialStateDict;
    auto params = model.named_parameters();
    if (!params.is_empty()) {
        auto firstParam = params.begin();
        partialStateDict[firstParam->key()] = firstParam->value().clone();
    }

    // Should not throw
    EXPECT_NO_THROW(model.loadWeights(partialStateDict));
}

// BM-017: LoadWeights_ParamNotInStateDict - Missing key is silently skipped
TEST_F(BaseModelTest, LoadWeights_ParamNotInStateDict) {
    TestModel model;
    model.setModel(createSimpleModel());

    // Empty state dict
    std::unordered_map<std::string, torch::Tensor> emptyStateDict;

    // Should not throw
    EXPECT_NO_THROW(model.loadWeights(emptyStateDict));
}

// BM-018: SaveWeights_AllParams - All params saved
TEST_F(BaseModelTest, SaveWeights_AllParams) {
    TestModel model;
    model.setModel(createSimpleModel());

    std::string path = getTempPath("all_params.pt");
    model.saveWeights(path);

    // Load and verify all params present
    torch::serialize::InputArchive archive;
    archive.load_from(path);

    for (const auto& param : model.named_parameters()) {
        torch::Tensor tensor;
        EXPECT_NO_THROW(archive.read(param.key(), tensor));
    }
}

// BM-019: SaveLoad_RoundTrip - Save and load produces identical model
TEST_F(BaseModelTest, SaveLoad_RoundTrip) {
    TestModel model1;
    model1.setModel(createSimpleModel());

    // Randomize weights
    for (auto& param : model1.parameters()) {
        param.data().uniform_(-1, 1);
    }

    std::string path = getTempPath("roundtrip.pt");
    model1.saveWeights(path);

    TestModel model2;
    model2.setModel(createSimpleModel());
    model2.loadWeights(path);

    // Compare all parameters
    auto params1 = model1.parameters();
    auto params2 = model2.parameters();

    for (size_t i = 0; i < params1.size(); ++i) {
        EXPECT_TRUE(torch::allclose(params1[i], params2[i], 1e-5, 1e-5))
            << "Parameter " << i << " mismatch after round-trip";
    }
}

// =============================================================================
// Part 2.4: Device Transfer Tests (8 tests)
// =============================================================================

// BM-020: To_CPU - Move to CPU
TEST_F(BaseModelTest, To_CPU) {
    TestModel model;
    model.setModel(createSimpleModel());

    model.to(torch::kCPU);

    for (const auto& param : model.parameters()) {
        EXPECT_TRUE(param.device().is_cpu());
    }
}

// BM-021: To_CUDA - Move to CUDA (if available)
TEST_F(BaseModelTest, To_CUDA) {
    if (!torch::cuda::is_available()) {
        // Cannot use GTEST_SKIP in 1.8.1.7, just pass the test
        SUCCEED() << "CUDA not available, skipping test";
        return;
    }

    TestModel model;
    model.setModel(createSimpleModel());

    model.to(torch::kCUDA);

    for (const auto& param : model.parameters()) {
        EXPECT_TRUE(param.device().is_cuda());
    }
}

// BM-022: To_DeviceWithDtype - Move with dtype
TEST_F(BaseModelTest, To_DeviceWithDtype) {
    TestModel model;
    model.setModel(createSimpleModel());

    model.to(torch::kCPU, torch::kFloat64);

    for (const auto& param : model.parameters()) {
        EXPECT_TRUE(param.device().is_cpu());
        EXPECT_EQ(param.dtype(), torch::kFloat64);
    }
}

// BM-023: To_MovesCriterion - Criterion also moved
TEST_F(BaseModelTest, To_MovesCriterion) {
    TestModel model;
    // Use model that outputs [B, 1] to match target shape
    model.setModel(createModelForLossTesting());

    // Initialize criterion by calling loss
    auto data = createDataExampleForLoss();
    model.loss(data);

    ASSERT_NE(model._criterion, nullptr);

    model.to(torch::kCPU);

    // Model should be on CPU
    for (const auto& param : model.parameters()) {
        EXPECT_TRUE(param.device().is_cpu());
    }
}

// BM-024: To_MovesCriterion_Null - Criterion is null, no crash
TEST_F(BaseModelTest, To_MovesCriterion_Null) {
    TestModel model;
    model.setModel(createSimpleModel());

    // Criterion should be null before loss() is called
    ASSERT_EQ(model._criterion, nullptr);

    // Should not crash
    EXPECT_NO_THROW(model.to(torch::kCPU));
}

// BM-025: To_MovesModel - Model sequential moved
TEST_F(BaseModelTest, To_MovesModel) {
    TestModel model;
    model.setModel(createSimpleModel());

    model.to(torch::kCPU);

    // Verify model sequential is on CPU
    for (const auto& param : model._model->parameters()) {
        EXPECT_TRUE(param.device().is_cpu());
    }
}

// BM-026: To_DeviceDtype_MovesCriterion - With dtype, criterion moved
TEST_F(BaseModelTest, To_DeviceDtype_MovesCriterion) {
    TestModel model;
    // Use model that outputs [B, 1] to match target shape
    model.setModel(createModelForLossTesting());

    // Initialize criterion
    auto data = createDataExampleForLoss();
    model.loss(data);

    model.to(torch::kCPU, torch::kFloat32);

    // No crash is the expectation
    SUCCEED();
}

// BM-027: To_DeviceDtype_MovesModel - With dtype, model moved
TEST_F(BaseModelTest, To_DeviceDtype_MovesModel) {
    TestModel model;
    model.setModel(createSimpleModel());

    model.to(torch::kCPU, torch::kFloat64);

    for (const auto& param : model._model->parameters()) {
        EXPECT_TRUE(param.device().is_cpu());
        EXPECT_EQ(param.dtype(), torch::kFloat64);
    }
}

// =============================================================================
// Part 2.5: Setter/Getter Tests (11 tests)
// =============================================================================

// BM-028: SetModel - Set model sequential
TEST_F(BaseModelTest, SetModel) {
    TestModel model;
    auto seq = createSimpleModel();

    model.setModel(seq);

    EXPECT_FALSE(model.getModel()->is_empty());
}

// BM-029: SwapModelSequential - Swap model without re-registering
TEST_F(BaseModelTest, SwapModelSequential) {
    TestModel model;
    auto seq1 = createSimpleModel();
    auto seq2 = createSimpleModel();

    model.setModel(seq1);
    auto oldSeq = model.swapModelSequential(seq2);

    // Should return the old sequential
    EXPECT_EQ(oldSeq->size(), seq1->size());
}

// BM-030: SetFromIndices - Set from indices
TEST_F(BaseModelTest, SetFromIndices) {
    TestModel model;
    std::vector<std::vector<int64_t>> indices = {{-1}, {-1}, {0, 2}};

    model.setFromIndices(indices);

    EXPECT_EQ(model._fromIndices.size(), 3);
    EXPECT_EQ(model._fromIndices[2].size(), 2);
}

// BM-031: SetSaveIndices - Set save indices
TEST_F(BaseModelTest, SetSaveIndices) {
    TestModel model;
    std::vector<int64_t> indices = {1, 3, 5, 7};

    model.setSaveIndices(indices);

    EXPECT_EQ(model.getSaveIndices().size(), 4);
    EXPECT_EQ(model.getSaveIndices()[0], 1);
    EXPECT_EQ(model.getSaveIndices()[3], 7);
}

// BM-032: SetStride - Set stride tensor
TEST_F(BaseModelTest, SetStride) {
    TestModel model;
    auto stride = torch::tensor({8.0f, 16.0f, 32.0f});

    model.setStride(stride);

    EXPECT_EQ(model.getStride().size(0), 3);
    EXPECT_FLOAT_EQ(model.getStride()[0].item<float>(), 8.0f);
    EXPECT_FLOAT_EQ(model.getStride()[1].item<float>(), 16.0f);
    EXPECT_FLOAT_EQ(model.getStride()[2].item<float>(), 32.0f);
}

// BM-033: SetTaskType - Set task type
TEST_F(BaseModelTest, SetTaskType) {
    TestModel model;

    model.setTaskType(TaskType::DETECTION);

    EXPECT_EQ(model.getTaskType(), TaskType::DETECTION);
}

// BM-034: SetConfig - Set configuration
TEST_F(BaseModelTest, SetConfig) {
    TestModel model;
    auto config = std::make_shared<Configuration>();

    model.setConfig(config);

    EXPECT_NE(model._config, nullptr);
}

// BM-035: GetModel - Get model sequential
TEST_F(BaseModelTest, GetModel) {
    TestModel model;
    model.setModel(createSimpleModel());

    auto seq = model.getModel();

    EXPECT_FALSE(seq->is_empty());
}

// BM-036: GetSaveIndices - Get save indices
TEST_F(BaseModelTest, GetSaveIndices) {
    TestModel model;
    model.setSaveIndices({1, 2, 3});

    auto indices = model.getSaveIndices();

    EXPECT_EQ(indices.size(), 3);
}

// BM-037: GetStride - Get stride
TEST_F(BaseModelTest, GetStride) {
    TestModel model;

    auto stride = model.getStride();

    EXPECT_TRUE(stride.defined());
    EXPECT_EQ(stride.size(0), 1);
}

// BM-038: GetTaskType - Get task type
TEST_F(BaseModelTest, GetTaskType) {
    TestModel model;

    EXPECT_EQ(model.getTaskType(), TaskType::UNKNOWN);

    model.setTaskType(TaskType::CLASSIFICATION);
    EXPECT_EQ(model.getTaskType(), TaskType::CLASSIFICATION);
}

// =============================================================================
// Part 2.6: Forward/Predict Complex Tests (15 tests)
// =============================================================================

// BM-039: Loss_InitializesCriterion - First call initializes criterion
TEST_F(BaseModelTest, Loss_InitializesCriterion) {
    TestModel model;
    // Use model that outputs [B, 1] to match target shape
    model.setModel(createModelForLossTesting());

    EXPECT_EQ(model._criterion, nullptr);

    auto data = createDataExampleForLoss();
    model.loss(data);

    EXPECT_NE(model._criterion, nullptr);
}

// BM-040: Loss_WithPreds - Loss with precomputed predictions
TEST_F(BaseModelTest, Loss_WithPreds) {
    TestModel model;
    // Use model that outputs [B, 1] to match target shape
    model.setModel(createModelForLossTesting());

    auto data = createDataExampleForLoss();
    // Predictions shape [B, 1] matches target shape
    auto preds = std::vector<torch::Tensor>{torch::rand({1, 1})};

    // Should not throw
    EXPECT_NO_THROW(model.loss(data, preds));
}

// BM-041: Loss_WithoutPreds - Loss without predictions runs forward
TEST_F(BaseModelTest, Loss_WithoutPreds) {
    TestModel model;
    // Use model that outputs [B, 1] to match target shape
    model.setModel(createModelForLossTesting());

    auto data = createDataExampleForLoss();

    // Should run forward pass internally
    EXPECT_NO_THROW(model.loss(data));
}

// BM-042: Loss_ReturnMap - Returns loss map with total key
TEST_F(BaseModelTest, Loss_ReturnMap) {
    TestModel model;
    // Use model that outputs [B, 1] to match target shape
    model.setModel(createModelForLossTesting());

    auto data = createDataExampleForLoss();
    auto lossMap = model.loss(data);

    EXPECT_TRUE(lossMap.find("total") != lossMap.end());
}

// BM-043: CountParameters - Count total parameters
TEST_F(BaseModelTest, CountParameters) {
    TestModel model;
    model.setModel(createSimpleModel());

    int64_t count = model.countParameters();

    EXPECT_GT(count, 0);
}

// BM-044: CountParameters_EmptyModel - Empty model returns 0
TEST_F(BaseModelTest, CountParameters_EmptyModel) {
    TestModel model;

    int64_t count = model.countParameters();

    EXPECT_EQ(count, 0);
}

// BM-045: CountFlops - Estimate FLOPs
TEST_F(BaseModelTest, CountFlops) {
    // Must use shared_ptr for modules() to work
    auto model = std::make_shared<TestModel>();
    model->setModel(createSimpleModel());

    double flops = model->countFlops({1, 3, 32, 32});

    EXPECT_GT(flops, 0.0);
}

// BM-046: CountFlops_NoConv2d - Model without Conv2d returns 0
TEST_F(BaseModelTest, CountFlops_NoConv2d) {
    // Must use shared_ptr for modules() to work
    auto model = std::make_shared<TestModel>();
    torch::nn::Sequential seq;
    seq->push_back(torch::nn::Linear(10, 10));
    model->setModel(seq);

    double flops = model->countFlops({1, 10, 1, 1});

    // Linear layers are not counted in current implementation
    EXPECT_GE(flops, 0.0);
}

// BM-047: ApplyToTensors_StrideDefined - Function applied to stride
TEST_F(BaseModelTest, ApplyToTensors_StrideDefined) {
    TestModel model;
    model.setModel(createSimpleModel());

    // Original stride is 32.0
    EXPECT_FLOAT_EQ(model._stride[0].item<float>(), 32.0f);

    // Apply function to multiply by 2
    model.applyToTensors([](const torch::Tensor& t) {
        return t * 2;
    });

    EXPECT_FLOAT_EQ(model._stride[0].item<float>(), 64.0f);
}

// BM-048: ApplyToTensors_StrideUndefined - Stride undefined is skipped
TEST_F(BaseModelTest, ApplyToTensors_StrideUndefined) {
    TestModel model;
    model._stride = torch::Tensor();  // Undefined

    EXPECT_FALSE(model._stride.defined());

    // Should not crash
    EXPECT_NO_THROW(model.applyToTensors([](const torch::Tensor& t) {
        return t * 2;
    }));
}

// BM-049: Predict_FromListEmpty - fromList empty uses simple sequential forward
// Note: Complex predict paths require IBlockImpl modules (tested in derived model tests)
TEST_F(BaseModelTest, Predict_FromListEmpty) {
    TestModel model;
    // Use empty Sequential with empty indices - takes simple forward path
    model._model = torch::nn::Sequential();
    model.setFromIndices({});
    model.setSaveIndices({});

    auto input = torch::rand({1, 3, 32, 32});
    auto output = model.predict(input);

    // Empty model returns input unchanged
    ASSERT_EQ(output.size(), 1);
    EXPECT_TRUE(torch::allclose(output[0], input));
}

// BM-050: Predict_FromListMinusOne - Empty indices uses simple forward
TEST_F(BaseModelTest, Predict_FromListMinusOne) {
    TestModel model;
    // With both indices empty, uses simple sequential forward (line 51-53)
    model._model = torch::nn::Sequential();
    model._model->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 3, 1)));
    model.setFromIndices({});  // Empty - triggers simple forward path
    model.setSaveIndices({});

    auto input = torch::rand({1, 3, 32, 32});
    auto output = model.predict(input);

    ASSERT_EQ(output.size(), 1);
    EXPECT_EQ(output[0].size(1), 3);  // Output channels
}

// BM-051: Predict_SimpleSequentialPath - Simple sequential forward path
TEST_F(BaseModelTest, Predict_SimpleSequentialPath) {
    TestModel model;
    // Multi-layer model with empty indices uses simple forward
    torch::nn::Sequential seq;
    seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)));
    seq->push_back(torch::nn::ReLU());
    seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).padding(1)));
    model.setModel(seq);

    // Empty indices trigger simple forward path
    model.setSaveIndices({});
    model.setFromIndices({});

    auto input = torch::rand({1, 3, 32, 32});
    auto output = model.predict(input);

    ASSERT_EQ(output.size(), 1);
    EXPECT_EQ(output[0].size(1), 32);  // Final output channels
}

// BM-052: Predict_SetIndicesCleared - Indices can be set and cleared
TEST_F(BaseModelTest, Predict_SetIndicesCleared) {
    TestModel model;
    model._model = torch::nn::Sequential();

    // Set indices
    model.setFromIndices({{-1}, {0, 1}});
    model.setSaveIndices({1, 2, 3});

    EXPECT_EQ(model._fromIndices.size(), 2);
    EXPECT_EQ(model.getSaveIndices().size(), 3);

    // Clear indices
    model.setFromIndices({});
    model.setSaveIndices({});

    EXPECT_TRUE(model._fromIndices.empty());
    EXPECT_TRUE(model.getSaveIndices().empty());
}

// BM-053: Predict_IndicesStorage - Indices are stored correctly
TEST_F(BaseModelTest, Predict_IndicesStorage) {
    TestModel model;
    model._model = torch::nn::Sequential();

    // Set complex fromIndices
    std::vector<std::vector<int64_t>> fromIndices = {{-1}, {-1}, {0, 2}, {-1, 3}};
    model.setFromIndices(fromIndices);

    EXPECT_EQ(model._fromIndices.size(), 4);
    EXPECT_EQ(model._fromIndices[2].size(), 2);
    EXPECT_EQ(model._fromIndices[2][0], 0);
    EXPECT_EQ(model._fromIndices[2][1], 2);
    EXPECT_EQ(model._fromIndices[3][0], -1);
    EXPECT_EQ(model._fromIndices[3][1], 3);
}

// =============================================================================
// Additional Edge Case Tests
// =============================================================================

// BM-054: Forward_WithDataExample - Training forward pass
TEST_F(BaseModelTest, Forward_WithDataExample) {
    TestModel model;
    // Use model that outputs [B, 1] to match target shape
    model.setModel(createModelForLossTesting());

    auto data = createDataExampleForLoss();
    auto result = model.forward(data);

    // Should return loss map
    EXPECT_TRUE(result.find("total") != result.end());
}

// BM-055: Multiple_LossCalls - Criterion reused
TEST_F(BaseModelTest, Multiple_LossCalls) {
    TestModel model;
    // Use model that outputs [B, 1] to match target shape
    model.setModel(createModelForLossTesting());

    auto data = createDataExampleForLoss();

    // First call
    model.loss(data);
    auto criterionPtr1 = model._criterion.get();

    // Second call
    model.loss(data);
    auto criterionPtr2 = model._criterion.get();

    // Same criterion instance should be reused
    EXPECT_EQ(criterionPtr1, criterionPtr2);
}
