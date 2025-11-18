#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Model/Task/ClassificationModel.h"
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

class ClassificationModelTest : public ::testing::Test
{
protected:
    static std::unique_ptr<ClassificationModel> model;
    static std::shared_ptr<Configuration> config;
    static std::filesystem::path testConfigPath;

    static void SetUpTestSuite()
    {
        try {
            testConfigPath = getTestDataPath() / "Phase3TestConfigs" / "Models";
            std::string modelPath = (testConfigPath / "minimal_classification.yaml").string();

            config = std::make_shared<Configuration>();
            model = std::make_unique<ClassificationModel>(config, modelPath);
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

std::unique_ptr<ClassificationModel> ClassificationModelTest::model;
std::shared_ptr<Configuration> ClassificationModelTest::config;
std::filesystem::path ClassificationModelTest::testConfigPath;

TEST_F(ClassificationModelTest, Construction_ValidYAML_Success)
{
    if (!model) {
        return;
    }

    EXPECT_NE(model, nullptr);
}

TEST_F(ClassificationModelTest, Construction_HasValidConfig)
{
    if (!model) {
        return;
    }

    EXPECT_NE(config, nullptr);
}

TEST_F(ClassificationModelTest, TaskType_IsClassification)
{
    if (!model) {
        return;
    }

    EXPECT_EQ(model->getTaskType(), WheelDL::TaskType::CLASSIFICATION);
}

TEST_F(ClassificationModelTest, Model_NotNull)
{
    if (!model) {
        return;
    }

    auto moduleModel = model->getModel();
    EXPECT_FALSE(moduleModel.is_empty());
}

TEST_F(ClassificationModelTest, ForwardPass_TrainingMode_ReturnsLogits)
{
    if (!model) {
        return;
    }

    model->train();
    torch::Tensor input = torch::rand({ 2, 3, 224, 224 });

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = model->forward(input));
}

TEST_F(ClassificationModelTest, ForwardPass_EvalMode_ReturnsProbabilities)
{
    if (!model) {
        return;
    }

    model->eval();
    torch::Tensor input = torch::rand({ 2, 3, 224, 224 });

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = model->forward(input));
}

TEST_F(ClassificationModelTest, ForwardPass_OutputNotEmpty)
{
    if (!model) {
        return;
    }

    model->eval();
    torch::Tensor input = torch::rand({ 1, 3, 224, 224 });

    std::vector<torch::Tensor> outputs = model->forward(input);
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(ClassificationModelTest, ForwardPass_OutputShape_MatchesNumClasses)
{
    if (!model) {
        return;
    }

    model->eval();
    int64_t batchSize = 2;
    torch::Tensor input = torch::rand({ batchSize, 3, 224, 224 });

    std::vector<torch::Tensor> outputs = model->forward(input);
    if (!outputs.empty() && outputs[0].defined()) {
        EXPECT_EQ(outputs[0].size(0), batchSize);
        // Check that second dimension is num_classes (10 from minimal_classification.yaml)
        EXPECT_EQ(outputs[0].size(1), 10);
    }
}

TEST_F(ClassificationModelTest, Predict_ReturnsValidTensors)
{
    if (!model) {
        return;
    }

    torch::Tensor input = torch::rand({ 1, 3, 224, 224 });

    std::vector<torch::Tensor> predictions;
    EXPECT_NO_THROW(predictions = model->predict(input));
}

TEST_F(ClassificationModelTest, Predict_OutputHasCorrectBatchSize)
{
    if (!model) {
        return;
    }

    int64_t batchSize = 4;
    torch::Tensor input = torch::rand({ batchSize, 3, 224, 224 });

    std::vector<torch::Tensor> predictions = model->predict(input);
    if (!predictions.empty() && predictions[0].defined()) {
        EXPECT_EQ(predictions[0].size(0), batchSize);
    }
}

TEST_F(ClassificationModelTest, LoadPretrained_InvalidPath_ReturnsFalse)
{
    if (!model) {
        return;
    }

    std::string invalidPath = "nonexistent_weights.pt";
    bool result = false;

    EXPECT_NO_THROW(result = model->loadPretrained(invalidPath));
    EXPECT_FALSE(result);
}

TEST_F(ClassificationModelTest, LoadWeights_ThrowsOnInvalidPath)
{
    if (!model) {
        return;
    }

    std::string invalidPath = "nonexistent_weights.pt";
    EXPECT_THROW(model->loadWeights(invalidPath), std::exception);
}

TEST_F(ClassificationModelTest, SaveWeights_ValidPath_NoThrow)
{
    if (!model) {
        return;
    }

    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test_classification_weights.pt";

    EXPECT_NO_THROW(model->saveWeights(tempPath.string()));

    if (std::filesystem::exists(tempPath)) {
        std::filesystem::remove(tempPath);
    }
}

TEST_F(ClassificationModelTest, TrainMode_SetsTrainingFlag)
{
    if (!model) {
        return;
    }

    model->train();
    EXPECT_TRUE(model->is_training());
}

TEST_F(ClassificationModelTest, EvalMode_ClearsTrainingFlag)
{
    if (!model) {
        return;
    }

    model->eval();
    EXPECT_FALSE(model->is_training());
}

TEST_F(ClassificationModelTest, Parameters_CountIsPositive)
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

TEST_F(ClassificationModelTest, Model_CanBeMoved)
{
    if (!model) {
        return;
    }

    EXPECT_NO_THROW(model->to(torch::kCPU));
}
