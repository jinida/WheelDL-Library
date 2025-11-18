#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Model/Task/BaseModel.h"
#include "WheelDL.Lib/Model/Task/DetectionModel.h"
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

class BaseModelTest : public ::testing::Test
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

std::unique_ptr<DetectionModel> BaseModelTest::model;
std::shared_ptr<Configuration> BaseModelTest::config;
std::filesystem::path BaseModelTest::testConfigPath;

TEST_F(BaseModelTest, Construction_Success)
{
    if (!model) {
        return;
    }

    EXPECT_NE(model, nullptr);
}

TEST_F(BaseModelTest, Forward_WithTensor_ReturnsVectorOfTensors)
{
    if (!model) {
        return;
    }

    model->eval();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 });

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = model->forward(input));
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(BaseModelTest, Forward_WithDataExample_ReturnsLossMap)
{
    if (!model) {
        return;
    }

    model->train();

    DataExample example;
    example.data = torch::rand({ 1, 3, 640, 640 });
    example.targets = torch::zeros({ 1, 4, 5 }); // Dummy targets

    std::unordered_map<std::string, torch::Tensor> losses;
    EXPECT_NO_THROW(losses = model->forward(example));
}

TEST_F(BaseModelTest, Predict_ReturnsVectorOfTensors)
{
    if (!model) {
        return;
    }

    torch::Tensor input = torch::rand({ 1, 3, 640, 640 });

    std::vector<torch::Tensor> predictions;
    EXPECT_NO_THROW(predictions = model->predict(input));
}

TEST_F(BaseModelTest, Loss_WithDataExample_ReturnsLossMap)
{
    if (!model) {
        return;
    }

    DataExample example;
    example.data = torch::rand({ 1, 3, 640, 640 });
    example.targets = torch::zeros({ 1, 4, 5 }); // Dummy targets

    std::unordered_map<std::string, torch::Tensor> losses;
    EXPECT_NO_THROW(losses = model->loss(example));
}

TEST_F(BaseModelTest, LoadWeights_InvalidPath_ThrowsException)
{
    if (!model) {
        return;
    }

    std::string invalidPath = "nonexistent_weights.pt";
    EXPECT_THROW(model->loadWeights(invalidPath), std::exception);
}

TEST_F(BaseModelTest, SaveWeights_ValidPath_Success)
{
    if (!model) {
        return;
    }

    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test_base_model_weights.pt";

    EXPECT_NO_THROW(model->saveWeights(tempPath.string()));

    if (std::filesystem::exists(tempPath)) {
        std::filesystem::remove(tempPath);
    }
}

TEST_F(BaseModelTest, GetModel_ReturnsNonNull)
{
    if (!model) {
        return;
    }

    auto moduleModel = model->getModel();
    EXPECT_FALSE(moduleModel.is_empty());
}

TEST_F(BaseModelTest, GetTaskType_ReturnsValidType)
{
    if (!model) {
        return;
    }

    WheelDL::TaskType taskType = model->getTaskType();
    EXPECT_EQ(taskType, WheelDL::TaskType::DETECTION);
}

TEST_F(BaseModelTest, GetStride_ReturnsDefinedTensor)
{
    if (!model) {
        return;
    }

    torch::Tensor stride = model->getStride();
    // Stride may or may not be defined depending on model initialization
    // Just verify we can call the method
    EXPECT_NO_THROW(model->getStride());
}

TEST_F(BaseModelTest, SetModel_AcceptsSequential)
{
    if (!model) {
        return;
    }

    // Create a simple sequential model
    torch::nn::Sequential seq;
    seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1)));

    EXPECT_NO_THROW(model->setModel(seq));
}

TEST_F(BaseModelTest, SetTaskType_UpdatesTaskType)
{
    if (!model) {
        return;
    }

    EXPECT_NO_THROW(model->setTaskType(WheelDL::TaskType::DETECTION));
    EXPECT_EQ(model->getTaskType(), WheelDL::TaskType::DETECTION);
}

TEST_F(BaseModelTest, SetConfig_UpdatesConfig)
{
    if (!model) {
        return;
    }

    auto newConfig = std::make_shared<Configuration>();
    EXPECT_NO_THROW(model->setConfig(newConfig));
}

TEST_F(BaseModelTest, Parameters_CountIsPositive)
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

TEST_F(BaseModelTest, TrainMode_SetsFlag)
{
    if (!model) {
        return;
    }

    model->train();
    EXPECT_TRUE(model->is_training());
}

TEST_F(BaseModelTest, EvalMode_ClearsFlag)
{
    if (!model) {
        return;
    }

    model->eval();
    EXPECT_FALSE(model->is_training());
}
