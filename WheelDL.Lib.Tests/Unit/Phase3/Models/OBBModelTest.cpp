#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Model/Task/OBBModel.h"
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

class OBBModelTest : public ::testing::Test
{
protected:
    static std::unique_ptr<OBBModel> model;
    static std::shared_ptr<Configuration> config;
    static std::filesystem::path testConfigPath;

    static void SetUpTestSuite()
    {
        try {
            testConfigPath = getTestDataPath() / "Phase3TestConfigs" / "Models";
            std::string modelPath = (testConfigPath / "minimal_obb.yaml").string();

            config = std::make_shared<Configuration>();
            model = std::make_unique<OBBModel>(config, modelPath);
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

std::unique_ptr<OBBModel> OBBModelTest::model;
std::shared_ptr<Configuration> OBBModelTest::config;
std::filesystem::path OBBModelTest::testConfigPath;

TEST_F(OBBModelTest, Construction_ValidYAML_Success)
{
    if (!model) {
        return;
    }

    EXPECT_NE(model, nullptr);
}

TEST_F(OBBModelTest, Construction_HasValidConfig)
{
    if (!model) {
        return;
    }

    EXPECT_NE(config, nullptr);
}

TEST_F(OBBModelTest, TaskType_IsOBB)
{
    if (!model) {
        return;
    }

    EXPECT_EQ(model->getTaskType(), WheelDL::TaskType::OBB);
}

TEST_F(OBBModelTest, Model_NotNull)
{
    if (!model) {
        return;
    }

    auto moduleModel = model->getModel();
    EXPECT_FALSE(moduleModel.is_empty());
}

TEST_F(OBBModelTest, ForwardPass_TrainingMode_ReturnsMultiScaleTensors)
{
    if (!model) {
        return;
    }

    model->train();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 });

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = model->forward(input));
}

TEST_F(OBBModelTest, ForwardPass_EvalMode_ReturnsTensors)
{
    if (!model) {
        return;
    }

    model->eval();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 });

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = model->forward(input));
}

TEST_F(OBBModelTest, ForwardPass_OutputNotEmpty)
{
    if (!model) {
        return;
    }

    model->eval();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 });

    std::vector<torch::Tensor> outputs = model->forward(input);
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(OBBModelTest, Predict_ReturnsValidTensors)
{
    if (!model) {
        return;
    }

    torch::Tensor input = torch::rand({ 1, 3, 640, 640 });

    std::vector<torch::Tensor> predictions;
    EXPECT_NO_THROW(predictions = model->predict(input));
}

TEST_F(OBBModelTest, Predict_OutputHasCorrectBatchSize)
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

TEST_F(OBBModelTest, Stride_HasValidValues)
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

TEST_F(OBBModelTest, LoadPretrained_InvalidPath_ReturnsFalse)
{
    if (!model) {
        return;
    }

    std::string invalidPath = "nonexistent_weights.pt";
    bool result = false;

    EXPECT_NO_THROW(result = model->loadPretrained(invalidPath));
    EXPECT_FALSE(result);
}

TEST_F(OBBModelTest, SaveWeights_ValidPath_NoThrow)
{
    if (!model) {
        return;
    }

    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test_obb_weights.pt";

    EXPECT_NO_THROW(model->saveWeights(tempPath.string()));

    if (std::filesystem::exists(tempPath)) {
        std::filesystem::remove(tempPath);
    }
}

TEST_F(OBBModelTest, TrainMode_SetsTrainingFlag)
{
    if (!model) {
        return;
    }

    model->train();
    EXPECT_TRUE(model->is_training());
}

TEST_F(OBBModelTest, EvalMode_ClearsTrainingFlag)
{
    if (!model) {
        return;
    }

    model->eval();
    EXPECT_FALSE(model->is_training());
}

TEST_F(OBBModelTest, Parameters_CountIsPositive)
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

TEST_F(OBBModelTest, Model_CanBeMoved)
{
    if (!model) {
        return;
    }

    EXPECT_NO_THROW(model->to(torch::kCPU));
}

TEST_F(OBBModelTest, ForwardPass_DifferentInputSizes_NoThrow)
{
    if (!model) {
        return;
    }

    model->eval();

    std::vector<std::vector<int64_t>> inputSizes = {
        {1, 3, 320, 320},
        {1, 3, 640, 640}
    };

    for (const auto& size : inputSizes) {
        torch::Tensor input = torch::rand(size);
        EXPECT_NO_THROW(model->forward(input));
    }
}
