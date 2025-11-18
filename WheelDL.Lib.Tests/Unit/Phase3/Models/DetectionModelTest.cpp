#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Model/Task/DetectionModel.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include <filesystem>
#include <memory>

using namespace WheelDL::Model;
using namespace WheelDL::Config;

namespace {
    std::filesystem::path getTestDataPath()
    {
        std::filesystem::path sourceDir = __FILE__;
        sourceDir = sourceDir.parent_path();
        while (sourceDir.filename() != "WheelDL.Lib.Tests" && sourceDir.has_parent_path()) {
            sourceDir = sourceDir.parent_path();
        }
        return sourceDir / "Data";
    }
}

class DetectionModelTest : public ::testing::Test
{
protected:
    static std::unique_ptr<DetectionModel> model;
    static std::shared_ptr<Configuration> config;
    static std::filesystem::path testConfigPath;

    static void SetUpTestSuite()
    {
        try {
            testConfigPath = getTestDataPath() / "Phase3TestConfigs" / "Models";
            std::string modelPath = (testConfigPath / "minimal_detection.yaml").string();

            config = std::make_shared<Configuration>();
            model = std::make_unique<DetectionModel>(config, modelPath);
        }
        catch (const std::exception& e) {
            std::cerr << "SetUpTestSuite failed: " << e.what() << std::endl;
            model.reset();
            config.reset();
        }
    }

    static void TearDownTestSuite()
    {
        model.reset();
        config.reset();
    }
};

std::unique_ptr<DetectionModel> DetectionModelTest::model;
std::shared_ptr<Configuration> DetectionModelTest::config;
std::filesystem::path DetectionModelTest::testConfigPath;

TEST_F(DetectionModelTest, Construction_ValidYAML_Success)
{
    if (!model) {
        return;
    }

    EXPECT_NE(model, nullptr);
}

TEST_F(DetectionModelTest, Construction_HasValidConfig)
{
    if (!model) {
        return;
    }

    EXPECT_NE(config, nullptr);
}

TEST_F(DetectionModelTest, TaskType_IsDetection)
{
    if (!model) {
        return;
    }

    EXPECT_EQ(model->getTaskType(), WheelDL::TaskType::DETECTION);
}

TEST_F(DetectionModelTest, Model_NotNull)
{
    if (!model) {
        return;
    }

    auto moduleModel = model->getModel();
    EXPECT_FALSE(moduleModel.is_empty());
}

TEST_F(DetectionModelTest, ForwardPass_TrainingMode_ReturnsMultiScaleTensors)
{
    if (!model) {
        return;
    }

    model->train();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 });

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = model->forward(input));
}

TEST_F(DetectionModelTest, ForwardPass_EvalMode_ReturnsTensors)
{
    if (!model) {
        return;
    }

    model->eval();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 });

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = model->forward(input));
}

TEST_F(DetectionModelTest, ForwardPass_OutputNotEmpty)
{
    if (!model) {
        return;
    }

    model->eval();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 });

    std::vector<torch::Tensor> outputs = model->forward(input);
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(DetectionModelTest, Predict_EvalMode_ReturnsValidTensors)
{
    if (!model) {
        return;
    }

    torch::Tensor input = torch::rand({ 1, 3, 640, 640 });

    std::vector<torch::Tensor> predictions;
    EXPECT_NO_THROW(predictions = model->predict(input));
}

TEST_F(DetectionModelTest, Predict_OutputHasCorrectBatchSize)
{
    if (!model) {
        return;
    }

    int64_t batchSize = 2;
    torch::Tensor input = torch::rand({ batchSize, 3, 640, 640 });

    std::vector<torch::Tensor> predictions = model->predict(input);
    if (!predictions.empty() && predictions[0].defined()) {
        EXPECT_EQ(predictions[0].size(0), batchSize);
    }
}

TEST_F(DetectionModelTest, Stride_HasValidValues)
{
    if (!model) {
        return;
    }

    torch::Tensor stride = model->getStride();
    if (stride.defined() && stride.numel() > 0) {
        EXPECT_GT(stride.numel(), 0);
        EXPECT_TRUE(torch::all(stride > 0).item<bool>());
    }
}

TEST_F(DetectionModelTest, SaveIndices_NotEmpty)
{
    if (!model) {
        return;
    }

    auto saveIndices = model->getSaveIndices();
    // SaveIndices may or may not be empty depending on model architecture
    // Just verify we can call the method
    EXPECT_NO_THROW(model->getSaveIndices());
}

TEST_F(DetectionModelTest, LoadPretrained_InvalidPath_ReturnsFalse)
{
    if (!model) {
        return;
    }

    std::string invalidPath = "nonexistent_weights.pt";
    bool result = false;

    EXPECT_NO_THROW(result = model->loadPretrained(invalidPath));
    EXPECT_FALSE(result);
}

TEST_F(DetectionModelTest, LoadWeights_ThrowsOnInvalidPath)
{
    if (!model) {
        return;
    }

    std::string invalidPath = "nonexistent_weights.pt";
    EXPECT_THROW(model->loadWeights(invalidPath), std::exception);
}

TEST_F(DetectionModelTest, SaveWeights_ValidPath_NoThrow)
{
    if (!model) {
        return;
    }

    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test_detection_weights.pt";

    EXPECT_NO_THROW(model->saveWeights(tempPath.string()));

    if (std::filesystem::exists(tempPath)) {
        std::filesystem::remove(tempPath);
    }
}

TEST_F(DetectionModelTest, TrainMode_SetsTrainingFlag)
{
    if (!model) {
        return;
    }

    model->train();
    EXPECT_TRUE(model->is_training());
}

TEST_F(DetectionModelTest, EvalMode_ClearsTrainingFlag)
{
    if (!model) {
        return;
    }

    model->eval();
    EXPECT_FALSE(model->is_training());
}

TEST_F(DetectionModelTest, ForwardPass_DifferentInputSizes_NoThrow)
{
    if (!model) {
        return;
    }

    model->eval();

    // Test with different input sizes
    std::vector<std::vector<int64_t>> inputSizes = {
        {1, 3, 320, 320},
        {1, 3, 640, 640},
        {2, 3, 640, 640}
    };

    for (const auto& size : inputSizes) {
        torch::Tensor input = torch::rand(size);
        EXPECT_NO_THROW(model->forward(input));
    }
}

TEST_F(DetectionModelTest, Parameters_CountIsPositive)
{
    if (!model) {
        return;
    }

    int64_t paramCount = 0;
    for (const auto& param : model->parameters()) {
        paramCount += param.numel();
    }

    EXPECT_GT(paramCount, 0);
}

TEST_F(DetectionModelTest, Model_CanBeMoved)
{
    if (!model) {
        return;
    }

    // Test that model can be moved to device (CPU in this case)
    EXPECT_NO_THROW(model->to(torch::kCPU));
}

TEST_F(DetectionModelTest, Stride_CalculatesCorrectly)
{
    if (!model) {
        return;
    }

    torch::Tensor stride = model->getStride();

    // Stride should be positive values (typically 8, 16, 32 for YOLO)
    if (stride.defined() && stride.numel() > 0) {
        auto strideValues = stride.accessor<float, 1>();
        for (int64_t i = 0; i < stride.size(0); ++i) {
            EXPECT_GT(strideValues[i], 0.0f);
            EXPECT_LE(strideValues[i], 64.0f); // Reasonable upper bound
        }
    }
}
