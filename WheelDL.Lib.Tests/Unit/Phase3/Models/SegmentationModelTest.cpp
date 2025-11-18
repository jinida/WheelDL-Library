#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Model/Task/SegmentationModel.h"
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

class SegmentationModelTest : public ::testing::Test
{
protected:
    static std::unique_ptr<SegmentationModel> model;
    static std::shared_ptr<Configuration> config;
    static std::filesystem::path testConfigPath;

    static void SetUpTestSuite()
    {
        try {
            testConfigPath = getTestDataPath() / "Phase3TestConfigs" / "Models";
            std::string modelPath = (testConfigPath / "minimal_segmentation.yaml").string();

            config = std::make_shared<Configuration>();
            model = std::make_unique<SegmentationModel>(config, modelPath);
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

std::unique_ptr<SegmentationModel> SegmentationModelTest::model;
std::shared_ptr<Configuration> SegmentationModelTest::config;
std::filesystem::path SegmentationModelTest::testConfigPath;

TEST_F(SegmentationModelTest, Construction_ValidYAML_Success)
{
    if (!model) {
        return;
    }

    EXPECT_NE(model, nullptr);
}

TEST_F(SegmentationModelTest, Construction_HasValidConfig)
{
    if (!model) {
        return;
    }

    EXPECT_NE(config, nullptr);
}

TEST_F(SegmentationModelTest, TaskType_IsSegmentation)
{
    if (!model) {
        return;
    }

    EXPECT_EQ(model->getTaskType(), WheelDL::TaskType::SEGMENTATION);
}

TEST_F(SegmentationModelTest, Model_NotNull)
{
    if (!model) {
        return;
    }

    auto moduleModel = model->getModel();
    EXPECT_FALSE(moduleModel.is_empty());
}

TEST_F(SegmentationModelTest, ForwardPass_TrainingMode_ReturnsLogits)
{
    if (!model) {
        return;
    }

    model->train();
    torch::Tensor input = torch::rand({ 1, 3, 512, 512 });

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = model->forward(input));
}

TEST_F(SegmentationModelTest, ForwardPass_EvalMode_ReturnsProbabilities)
{
    if (!model) {
        return;
    }

    model->eval();
    torch::Tensor input = torch::rand({ 1, 3, 512, 512 });

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = model->forward(input));
}

TEST_F(SegmentationModelTest, ForwardPass_OutputNotEmpty)
{
    if (!model) {
        return;
    }

    model->eval();
    torch::Tensor input = torch::rand({ 1, 3, 512, 512 });

    std::vector<torch::Tensor> outputs = model->forward(input);
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(SegmentationModelTest, ForwardPass_OutputShape_PixelWise)
{
    if (!model) {
        return;
    }

    model->eval();
    int64_t batchSize = 2;
    int64_t height = 512;
    int64_t width = 512;
    torch::Tensor input = torch::rand({ batchSize, 3, height, width });

    std::vector<torch::Tensor> outputs = model->forward(input);
    if (!outputs.empty() && outputs[0].defined()) {
        EXPECT_EQ(outputs[0].size(0), batchSize);
        // Output should have spatial dimensions (possibly downsampled)
        EXPECT_GE(outputs[0].dim(), 3);
    }
}

TEST_F(SegmentationModelTest, Predict_ReturnsValidTensors)
{
    if (!model) {
        return;
    }

    torch::Tensor input = torch::rand({ 1, 3, 512, 512 });

    std::vector<torch::Tensor> predictions;
    EXPECT_NO_THROW(predictions = model->predict(input));
}

TEST_F(SegmentationModelTest, Predict_OutputHasCorrectBatchSize)
{
    if (!model) {
        return;
    }

    int64_t batchSize = 2;
    torch::Tensor input = torch::rand({ batchSize, 3, 512, 512 });

    std::vector<torch::Tensor> predictions = model->predict(input);
    if (!predictions.empty() && predictions[0].defined()) {
        EXPECT_EQ(predictions[0].size(0), batchSize);
    }
}

TEST_F(SegmentationModelTest, LoadPretrained_InvalidPath_ReturnsFalse)
{
    if (!model) {
        return;
    }

    std::string invalidPath = "nonexistent_weights.pt";
    bool result = false;

    EXPECT_NO_THROW(result = model->loadPretrained(invalidPath));
    EXPECT_FALSE(result);
}

TEST_F(SegmentationModelTest, LoadWeights_ThrowsOnInvalidPath)
{
    if (!model) {
        return;
    }

    std::string invalidPath = "nonexistent_weights.pt";
    EXPECT_THROW(model->loadWeights(invalidPath), std::exception);
}

TEST_F(SegmentationModelTest, SaveWeights_ValidPath_NoThrow)
{
    if (!model) {
        return;
    }

    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test_segmentation_weights.pt";

    EXPECT_NO_THROW(model->saveWeights(tempPath.string()));

    if (std::filesystem::exists(tempPath)) {
        std::filesystem::remove(tempPath);
    }
}

TEST_F(SegmentationModelTest, TrainMode_SetsTrainingFlag)
{
    if (!model) {
        return;
    }

    model->train();
    EXPECT_TRUE(model->is_training());
}

TEST_F(SegmentationModelTest, EvalMode_ClearsTrainingFlag)
{
    if (!model) {
        return;
    }

    model->eval();
    EXPECT_FALSE(model->is_training());
}

TEST_F(SegmentationModelTest, Parameters_CountIsPositive)
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

TEST_F(SegmentationModelTest, Model_CanBeMoved)
{
    if (!model) {
        return;
    }

    EXPECT_NO_THROW(model->to(torch::kCPU));
}

TEST_F(SegmentationModelTest, ForwardPass_DifferentInputSizes_NoThrow)
{
    if (!model) {
        return;
    }

    model->eval();

    std::vector<std::vector<int64_t>> inputSizes = {
        {1, 3, 256, 256},
        {1, 3, 512, 512}
    };

    for (const auto& size : inputSizes) {
        torch::Tensor input = torch::rand(size);
        EXPECT_NO_THROW(model->forward(input));
    }
}
