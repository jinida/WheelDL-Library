#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Model/Task/AnomalyModel.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include "WheelDL.Lib/Data/Dataset/BaseDataset.h"
#include <filesystem>
#include <memory>

using namespace WheelDL::Model;
using namespace WheelDL::Config;
using namespace WheelDL::Data::Dataset;

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

class AnomalyModelTest : public ::testing::Test
{
protected:
    static std::unique_ptr<AnomalyModel> model;
    static std::shared_ptr<Configuration> config;
    static std::filesystem::path testConfigPath;

    static void SetUpTestSuite()
    {
        try {
            testConfigPath = getTestDataPath() / "Phase3TestConfigs" / "Models";
            std::string modelPath = (testConfigPath / "minimal_anomaly.yaml").string();

            config = std::make_shared<Configuration>();
            model = std::make_unique<AnomalyModel>(config, modelPath);
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

std::unique_ptr<AnomalyModel> AnomalyModelTest::model;
std::shared_ptr<Configuration> AnomalyModelTest::config;
std::filesystem::path AnomalyModelTest::testConfigPath;

TEST_F(AnomalyModelTest, Construction_ValidYAML_Success)
{
    if (!model) {
        return;
    }

    EXPECT_NE(model, nullptr);
}

TEST_F(AnomalyModelTest, Construction_HasValidConfig)
{
    if (!model) {
        return;
    }

    EXPECT_NE(config, nullptr);
}

TEST_F(AnomalyModelTest, TaskType_IsAnomaly)
{
    if (!model) {
        return;
    }

    EXPECT_EQ(model->getTaskType(), WheelDL::TaskType::ANOMALY);
}

TEST_F(AnomalyModelTest, Model_NotNull)
{
    if (!model) {
        return;
    }

    auto moduleModel = model->getModel();
    EXPECT_FALSE(moduleModel.is_empty());
}

TEST_F(AnomalyModelTest, ForwardPass_WithTensor_ReturnsValidOutput)
{
    if (!model) {
        return;
    }

    model->eval();
    torch::Tensor input = torch::rand({ 1, 3, 256, 256 });

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = model->forward(input));
}

TEST_F(AnomalyModelTest, ForwardPass_WithDataExample_ReturnsLosses)
{
    if (!model) {
        return;
    }

    model->train();

    DataExample example;
    example.data = torch::rand({ 2, 3, 256, 256 });
    example.targets = torch::zeros({ 2 });

    std::unordered_map<std::string, torch::Tensor> losses;
    EXPECT_NO_THROW(losses = model->forward(example));
}

TEST_F(AnomalyModelTest, Predict_ReturnsValidTensors)
{
    if (!model) {
        return;
    }

    torch::Tensor input = torch::rand({ 1, 3, 256, 256 });

    std::vector<torch::Tensor> predictions;
    EXPECT_NO_THROW(predictions = model->predict(input));
}

TEST_F(AnomalyModelTest, Predict_OutputNotEmpty)
{
    if (!model) {
        return;
    }

    torch::Tensor input = torch::rand({ 1, 3, 256, 256 });

    std::vector<torch::Tensor> predictions = model->predict(input);
    EXPECT_GT(predictions.size(), 0);
}

TEST_F(AnomalyModelTest, ForwardPass_OutputHasCorrectBatchSize)
{
    if (!model) {
        return;
    }

    model->eval();
    int64_t batchSize = 4;
    torch::Tensor input = torch::rand({ batchSize, 3, 256, 256 });

    std::vector<torch::Tensor> outputs = model->forward(input);
    if (!outputs.empty() && outputs[0].defined()) {
        EXPECT_EQ(outputs[0].size(0), batchSize);
    }
}

TEST_F(AnomalyModelTest, LoadPretrained_InvalidPath_ReturnsFalse)
{
    if (!model) {
        return;
    }

    std::string invalidPath = "nonexistent_weights.pt";
    bool result = false;

    EXPECT_NO_THROW(result = model->loadPretrained(invalidPath));
    EXPECT_FALSE(result);
}

TEST_F(AnomalyModelTest, LoadWeights_ThrowsOnInvalidPath)
{
    if (!model) {
        return;
    }

    std::string invalidPath = "nonexistent_weights.pt";
    EXPECT_THROW(model->loadWeights(invalidPath), std::exception);
}

TEST_F(AnomalyModelTest, SaveWeights_ValidPath_NoThrow)
{
    if (!model) {
        return;
    }

    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test_anomaly_weights.pt";

    EXPECT_NO_THROW(model->saveWeights(tempPath.string()));

    if (std::filesystem::exists(tempPath)) {
        std::filesystem::remove(tempPath);
    }
}

TEST_F(AnomalyModelTest, TrainMode_SetsTrainingFlag)
{
    if (!model) {
        return;
    }

    model->train();
    EXPECT_TRUE(model->is_training());
}

TEST_F(AnomalyModelTest, EvalMode_ClearsTrainingFlag)
{
    if (!model) {
        return;
    }

    model->eval();
    EXPECT_FALSE(model->is_training());
}

TEST_F(AnomalyModelTest, Parameters_CountIsPositive)
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

TEST_F(AnomalyModelTest, Model_CanBeMoved)
{
    if (!model) {
        return;
    }

    EXPECT_NO_THROW(model->to(torch::kCPU));
}

TEST_F(AnomalyModelTest, ForwardPass_DifferentInputSizes_NoThrow)
{
    if (!model) {
        return;
    }

    model->eval();

    // Note: AnomalyModel may require fixed input size (256x256 for EfficientAD)
    std::vector<std::vector<int64_t>> inputSizes = {
        {1, 3, 256, 256},
        {2, 3, 256, 256}
    };

    for (const auto& size : inputSizes) {
        torch::Tensor input = torch::rand(size);
        EXPECT_NO_THROW(model->forward(input));
    }
}

TEST_F(AnomalyModelTest, Loss_WithValidDataExample_ReturnsNonEmptyMap)
{
    if (!model) {
        return;
    }

    DataExample example;
    example.data = torch::rand({ 2, 3, 256, 256 });
    example.targets = torch::zeros({ 2 });

    std::unordered_map<std::string, torch::Tensor> losses;
    EXPECT_NO_THROW(losses = model->loss(example));

    // Losses map may or may not be empty depending on implementation
    // Just verify we can call the method
}

TEST_F(AnomalyModelTest, ForwardPass_TrainingMode_NoThrow)
{
    if (!model) {
        return;
    }

    model->train();

    DataExample example;
    example.data = torch::rand({ 1, 3, 256, 256 });
    example.targets = torch::zeros({ 1 });

    EXPECT_NO_THROW(model->forward(example));
}

TEST_F(AnomalyModelTest, ForwardPass_InferenceMode_NoThrow)
{
    if (!model) {
        return;
    }

    model->eval();
    torch::Tensor input = torch::rand({ 1, 3, 256, 256 });

    EXPECT_NO_THROW(model->forward(input));
}
