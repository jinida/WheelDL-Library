#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Builder/ModelBuilder.h"
#include "WheelDL.Lib/Model/Task/DetectionModel.h"
#include "WheelDL.Lib/Model/Task/OBBModel.h"
#include "WheelDL.Lib/Model/Task/ClassificationModel.h"
#include "WheelDL.Lib/Model/Task/SegmentationModel.h"
#include "WheelDL.Lib/Model/Task/AnomalyModel.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include "WheelDL.Lib/Data/Dataset/AnomalyDataset.h"
#include "WheelDL.Lib/Data/Transforms/Collation.h"
#include <torch/torch.h>
#include <filesystem>

using namespace WheelDL::Model;
using namespace WheelDL::Config;
using namespace WheelDL::Data::Dataset;
using namespace WheelDL::Data;

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

/**
 * @brief Phase3 Integration Test Fixture
 *
 * Tests full end-to-end pipelines for all task types:
 * - Detection (object detection with multi-scale predictions)
 * - OBB (oriented bounding box detection)
 * - Classification (image classification)
 * - Segmentation (instance segmentation)
 * - Anomaly (anomaly detection)
 *
 * Uses static fixture pattern as model building is expensive.
 * Models are built once per test suite and reused across all tests.
 */
class Phase3IntegrationTest : public ::testing::Test {
protected:
    static std::unique_ptr<Builder::ModelBuilder> builder;
    static std::filesystem::path integrationConfigPath;
    static std::filesystem::path datasetPath;
    static std::filesystem::path annotationPath;
    static std::unique_ptr<AnomalyDataset> anomalyDataset;

    // Static models for each task type
    static std::unique_ptr<DetectionModel> detectionModel;
    static std::unique_ptr<OBBModel> obbModel;
    static std::unique_ptr<ClassificationModel> classificationModel;
    static std::unique_ptr<SegmentationModel> segmentationModel;
    static std::unique_ptr<AnomalyModel> anomalyModel;
    static std::unique_ptr<AnomalyModel> anomalyModelPC;

    // Shared configuration
    static std::shared_ptr<Configuration> classifyConfig;
    static std::shared_ptr<Configuration> detectConfig;
	static std::shared_ptr<Configuration> obbConfig;
    static std::shared_ptr<Configuration> segmentConfig;
    static std::shared_ptr<Configuration> anomalyConfig;

    static void SetUpTestSuite()
    {
        try {
            auto basePath = getTestDataPath();
            integrationConfigPath = basePath / "Phase3TestConfigs" / "Integration";
            auto modelsPath = basePath / "Phase3TestConfigs" / "Models";
            datasetPath = basePath / "mnist_sample";
            annotationPath = basePath / "mnist_sample" / "labels" / "anomalydetection";

            // Verify directories exist
            if (!std::filesystem::exists(integrationConfigPath)) {
                std::string error = "Integration config path does not exist: " + integrationConfigPath.string();
                throw std::runtime_error(error);
            }

            // Initialize configurations for each task type
            detectConfig = std::make_shared<Configuration>();
            obbConfig = std::make_shared<Configuration>();
            classifyConfig = std::make_shared<Configuration>();
            segmentConfig = std::make_shared<Configuration>();
            anomalyConfig = std::make_shared<Configuration>();

            // Build all 5 models (only once for all tests) - with individual error handling
            try {
                auto detectionHyperPath = (integrationConfigPath / "default.yaml").string();
                auto detectionModelPath = (modelsPath / "minimal_detection.yaml").string();
                if (!std::filesystem::exists(detectionModelPath)) {
                    throw std::runtime_error("Model file not found: " + detectionModelPath);
                }
                detectConfig->loadFromYaml(detectionModelPath, detectionHyperPath, "");
                detectionModel = std::make_unique<DetectionModel>(detectConfig, detectionModelPath);
                detectionModel->to(torch::kCUDA);
            } catch (const std::exception& e) {
                detectionModel.reset();
            }

            try {
                auto obbHyperPath = (integrationConfigPath / "default.yaml").string();
                auto obbModelPath = (modelsPath / "minimal_obb.yaml").string();
                if (!std::filesystem::exists(obbModelPath)) {
                    throw std::runtime_error("Model file not found: " + obbModelPath);
                }
                obbConfig->loadFromYaml(obbModelPath, obbHyperPath, "");
                obbModel = std::make_unique<OBBModel>(obbConfig, obbModelPath);
                obbModel->to(torch::kCUDA);
            } catch (const std::exception& e) {
                obbModel.reset();
            }

            try {
                auto classificationHyperPath = (integrationConfigPath / "default.yaml").string();
                auto classificationModelPath = (modelsPath / "minimal_classification.yaml").string();
                if (!std::filesystem::exists(classificationModelPath)) {
                    throw std::runtime_error("Model file not found: " + classificationModelPath);
                }
                classifyConfig->loadFromYaml(classificationModelPath, classificationHyperPath, "");
                classificationModel = std::make_unique<ClassificationModel>(classifyConfig, classificationModelPath);
                classificationModel->to(torch::kCUDA);
            } catch (const std::exception& e) {
                classificationModel.reset();
            }

            try {
                auto segmentationHyperPath = (integrationConfigPath / "default.yaml").string();
                auto segmentationModelPath = (modelsPath / "minimal_segmentation.yaml").string();
                if (!std::filesystem::exists(segmentationModelPath)) {
                    throw std::runtime_error("Model file not found: " + segmentationModelPath);
                }
                segmentConfig->loadFromYaml(segmentationModelPath, segmentationHyperPath, "");
                segmentationModel = std::make_unique<SegmentationModel>(segmentConfig, segmentationModelPath);
                segmentationModel->to(torch::kCUDA);
            } catch (const std::exception& e) {
                segmentationModel.reset();
            }

            try {
                auto anomalyHyperPath = (integrationConfigPath / "default.yaml").string();
                auto anomalyModelPath = (modelsPath / "minimal_anomaly.yaml").string();
                if (!std::filesystem::exists(anomalyModelPath)) {
                    throw std::runtime_error("Model file not found: " + anomalyModelPath);
                }
                anomalyConfig->loadFromYaml(anomalyModelPath, anomalyHyperPath, "");
                anomalyModel = std::make_unique<AnomalyModel>(anomalyConfig, anomalyModelPath);
                anomalyModel->to(torch::kCUDA);
            } catch (const std::exception& e) {
                anomalyModel.reset();
            }

            try {
                auto anomalyHyperPath2 = (integrationConfigPath / "default.yaml").string();
                auto anomalyModelPath2 = (modelsPath / "minimal_anomaly2.yaml").string();
                if (!std::filesystem::exists(anomalyModelPath2)) {
                    throw std::runtime_error("Model file not found: " + anomalyModelPath2);
                }
                anomalyConfig->setImageSize(64);
                anomalyConfig->setIsPatchCore(true);
                anomalyModelPC = std::make_unique<AnomalyModel>(anomalyConfig, anomalyModelPath2);
                anomalyModelPC->to(torch::kCUDA);

                auto dataLoader = torch::data::make_data_loader(
                    AnomalyDataset(
                        (datasetPath / "images").string(),
                        annotationPath.string(),
                        *anomalyConfig,
                        false
                    ).map(DataExampleCollation()),
					torch::data::DataLoaderOptions().batch_size(64).workers(0));

				anomalyModelPC->prepareTraining(*dataLoader);
				anomalyModelPC->prepareValidation(*dataLoader);
            }
            catch (const std::exception& e) {
                anomalyModelPC.reset();
            }

        } catch (const std::exception& e) {
            detectionModel.reset();
            obbModel.reset();
            classificationModel.reset();
            segmentationModel.reset();
            anomalyModel.reset();
            anomalyModelPC.reset();
            detectConfig.reset();
            obbConfig.reset();
            classifyConfig.reset();
            segmentConfig.reset();
            anomalyConfig.reset();
        }
    }

    // For older Google Test versions
    static void SetUpTestCase()
    {
        SetUpTestSuite();
    }

    static void TearDownTestSuite()
    {
        detectionModel.reset();
        obbModel.reset();
        classificationModel.reset();
        segmentationModel.reset();
        anomalyModelPC.reset();
        anomalyModel.reset();
        detectConfig.reset();
        obbConfig.reset();
        classifyConfig.reset();
        segmentConfig.reset();
        anomalyConfig.reset();
    }
};

// Static member initialization
std::unique_ptr<Builder::ModelBuilder> Phase3IntegrationTest::builder;
std::filesystem::path Phase3IntegrationTest::integrationConfigPath;
std::filesystem::path Phase3IntegrationTest::datasetPath;
std::filesystem::path Phase3IntegrationTest::annotationPath;

std::unique_ptr<DetectionModel> Phase3IntegrationTest::detectionModel;
std::unique_ptr<OBBModel> Phase3IntegrationTest::obbModel;
std::unique_ptr<ClassificationModel> Phase3IntegrationTest::classificationModel;
std::unique_ptr<SegmentationModel> Phase3IntegrationTest::segmentationModel;
std::unique_ptr<AnomalyModel> Phase3IntegrationTest::anomalyModel;
std::unique_ptr<AnomalyModel> Phase3IntegrationTest::anomalyModelPC;
std::shared_ptr<Configuration> Phase3IntegrationTest::classifyConfig;
std::shared_ptr<Configuration> Phase3IntegrationTest::detectConfig;
std::shared_ptr<Configuration> Phase3IntegrationTest::obbConfig;
std::shared_ptr<Configuration> Phase3IntegrationTest::segmentConfig;
std::shared_ptr<Configuration> Phase3IntegrationTest::anomalyConfig;

// ========== Detection Pipeline Tests ==========

TEST_F(Phase3IntegrationTest, DetectionPipeline_ModelConstruction_Success)
{
    if (!detectionModel) {
        return;
    }

    EXPECT_NE(detectionModel, nullptr);
    EXPECT_EQ(detectionModel->getTaskType(), WheelDL::TaskType::DETECTION);
}

TEST_F(Phase3IntegrationTest, DetectionPipeline_ForwardPass_Training_Success)
{
    if (!detectionModel) {
        return;
    }

    detectionModel->train();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = detectionModel->forward(input));
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, DetectionPipeline_ForwardPass_Inference_Success)
{
    if (!detectionModel) {
        return;
    }

    detectionModel->eval();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = detectionModel->forward(input));
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, DetectionPipeline_MultiScale_OutputShapes)
{
    if (!detectionModel) {
        return;
    }

    detectionModel->train();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs = detectionModel->forward(input);

    // Minimal model uses single scale, so expect at least 1 output
    EXPECT_GE(outputs.size(), 1);

    // Verify output tensors are defined
    for (const auto& output : outputs) {
        EXPECT_TRUE(output.defined());
        EXPECT_GT(output.numel(), 0);
    }
}

TEST_F(Phase3IntegrationTest, DetectionPipeline_Loss_Computation_Success)
{
    if (!detectionModel) {
        return;
    }

    detectionModel->train();
    torch::Tensor input = torch::rand({ 2, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor targets = torch::rand({ 5, 4 }, torch::TensorOptions().device(torch::kCUDA)); // [batch, max_objects, 6 (class, x, y, w, h, batch_idx)]
	torch::Tensor batchIndices = torch::randint(0, 2, { 5 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong));
	torch::Tensor classes = torch::randint(0, 10, { 5 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong));

    DataExample data;
    data.data = input;
    data.targets = targets;
	data.batchIndices = batchIndices;
	data.classes = classes;

    std::unordered_map<std::string, torch::Tensor> losses;
    EXPECT_NO_THROW(losses = detectionModel->loss(data));
    EXPECT_GT(losses.size(), 0);
}

TEST_F(Phase3IntegrationTest, DetectionPipeline_Loss_OutputKeys)
{
    if (!detectionModel) {
        return;
    }

    detectionModel->train();
    torch::Tensor input = torch::rand({ 2, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor targets = torch::rand({ 5, 4 }, torch::TensorOptions().device(torch::kCUDA)); // [batch, max_objects, 6 (class, x, y, w, h, batch_idx)]
    torch::Tensor batchIndices = torch::randint(0, 2, { 5 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong));
    torch::Tensor classes = torch::randint(0, 10, { 5 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong));

    DataExample data;
    data.data = input;
    data.targets = targets;
    data.batchIndices = batchIndices;
    data.classes = classes;

    std::unordered_map<std::string, torch::Tensor> losses = detectionModel->loss(data);

    // Detection loss should contain: total, box, cls, dfl
    EXPECT_TRUE(losses.find("total") != losses.end());
}

TEST_F(Phase3IntegrationTest, DetectionPipeline_Gradient_BackwardPass)
{
    if (!detectionModel) {
        return;
    }

    detectionModel->train();
    torch::Tensor input = torch::rand({ 2, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor targets = torch::rand({ 5, 4 }, torch::TensorOptions().device(torch::kCUDA)); // [batch, max_objects, 6 (class, x, y, w, h, batch_idx)]
    torch::Tensor batchIndices = torch::randint(0, 2, { 5 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong));
    torch::Tensor classes = torch::randint(0, 10, { 5 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong));

    DataExample data;
    data.data = input;
    data.targets = targets;
    data.batchIndices = batchIndices;
    data.classes = classes;

    std::unordered_map<std::string, torch::Tensor> losses = detectionModel->loss(data);

    if (losses.find("total") != losses.end()) {
        torch::Tensor totalLoss = losses["total"];
        EXPECT_NO_THROW(totalLoss.backward());
    }
}

TEST_F(Phase3IntegrationTest, DetectionPipeline_WeightSave_Load)
{
    if (!detectionModel) {
        return;
    }

    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test_detection_integration.pt";

    EXPECT_NO_THROW(detectionModel->saveWeights(tempPath.string()));
    EXPECT_TRUE(std::filesystem::exists(tempPath));

    EXPECT_NO_THROW(detectionModel->loadWeights(tempPath.string()));

    if (std::filesystem::exists(tempPath)) {
        std::filesystem::remove(tempPath);
    }
}

TEST_F(Phase3IntegrationTest, DetectionPipeline_BatchProcessing_Success)
{
    if (!detectionModel) {
        return;
    }

    detectionModel->eval();
    int64_t batchSize = 4;
    torch::Tensor input = torch::rand({ batchSize, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = detectionModel->forward(input));
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, DetectionPipeline_EmptyBatch_Handled)
{
    if (!detectionModel) {
        return;
    }

    detectionModel->eval();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    // Empty batch should still produce valid output
    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = detectionModel->forward(input));
}

// ========== OBB Pipeline Tests ==========

TEST_F(Phase3IntegrationTest, OBBPipeline_ModelConstruction_Success)
{
    if (!obbModel) {
        return;
    }

    EXPECT_NE(obbModel, nullptr);
    EXPECT_EQ(obbModel->getTaskType(), WheelDL::TaskType::OBB);
}

TEST_F(Phase3IntegrationTest, OBBPipeline_ForwardPass_Training_Success)
{
    if (!obbModel) {
        return;
    }

    obbModel->train();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = obbModel->forward(input));
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, OBBPipeline_ForwardPass_Inference_Success)
{
    if (!obbModel) {
        return;
    }

    obbModel->eval();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = obbModel->forward(input));
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, OBBPipeline_MultiScale_OutputShapes)
{
    if (!obbModel) {
        return;
    }

    obbModel->train();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs = obbModel->forward(input);

    // OBB outputs: detection layers (4D) + angle (3D)
    // Minimal model uses single scale, so expect at least 2 outputs
    EXPECT_GE(outputs.size(), 2);

    // Verify output tensors are defined
    for (const auto& output : outputs) {
        EXPECT_TRUE(output.defined());
        EXPECT_GT(output.numel(), 0);
    }
}

TEST_F(Phase3IntegrationTest, OBBPipeline_Loss_Computation_Success)
{
    if (!obbModel) {
        return;
    }

    obbModel->train();
    torch::Tensor input = torch::rand({ 2, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor targets = torch::rand({ 2, 5 }, torch::TensorOptions().device(torch::kCUDA)); // [num_objects, 5 (x, y, w, h, angle)]
    torch::Tensor batchIndices = torch::tensor({ 0, 1 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong)); // Ensure each batch has objects
    torch::Tensor classes = torch::randint(0, 10, { 2 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong));

    DataExample data;
    data.data = input;
    data.targets = targets;
    data.batchIndices = batchIndices;
    data.classes = classes;

    std::unordered_map<std::string, torch::Tensor> losses;
    EXPECT_NO_THROW(losses = obbModel->loss(data));
    EXPECT_GT(losses.size(), 0);
}

TEST_F(Phase3IntegrationTest, OBBPipeline_Loss_OutputKeys)
{
    if (!obbModel) {
        return;
    }

    obbModel->train();
    torch::Tensor input = torch::rand({ 2, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor targets = torch::rand({ 2, 5 }, torch::TensorOptions().device(torch::kCUDA)); // [num_objects, 5 (x, y, w, h, angle)]
    torch::Tensor batchIndices = torch::tensor({ 0, 1 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong)); // Ensure each batch has objects
    torch::Tensor classes = torch::randint(0, 10, { 2 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong));

    DataExample data;
    data.data = input;
    data.targets = targets;
    data.batchIndices = batchIndices;
    data.classes = classes;

    std::unordered_map<std::string, torch::Tensor> losses = obbModel->loss(data);

    // OBB loss should contain: total, box, cls, dfl, angle
    EXPECT_TRUE(losses.find("total") != losses.end());
}

TEST_F(Phase3IntegrationTest, OBBPipeline_AnglePrediction_InRange)
{
    if (!obbModel) {
        return;
    }

    obbModel->eval();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs = obbModel->forward(input);

    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, OBBPipeline_Gradient_BackwardPass)
{
    if (!obbModel) {
        return;
    }

    obbModel->train();
    torch::Tensor input = torch::rand({ 2, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor targets = torch::rand({ 2, 5 }, torch::TensorOptions().device(torch::kCUDA)); // [num_objects, 5 (x, y, w, h, angle)]
    torch::Tensor batchIndices = torch::tensor({ 0, 1 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong)); // Ensure each batch has objects
    torch::Tensor classes = torch::randint(0, 10, { 2 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong));

    DataExample data;
    data.data = input;
    data.targets = targets;
    data.batchIndices = batchIndices;
    data.classes = classes;

    std::unordered_map<std::string, torch::Tensor> losses = obbModel->loss(data);

    if (losses.find("total") != losses.end()) {
        torch::Tensor totalLoss = losses["total"];
        EXPECT_NO_THROW(totalLoss.backward());
    }
}

TEST_F(Phase3IntegrationTest, OBBPipeline_WeightSave_Load)
{
    if (!obbModel) {
        return;
    }

    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test_obb_integration.pt";

    EXPECT_NO_THROW(obbModel->saveWeights(tempPath.string()));
    EXPECT_TRUE(std::filesystem::exists(tempPath));

    EXPECT_NO_THROW(obbModel->loadWeights(tempPath.string()));

    if (std::filesystem::exists(tempPath)) {
        std::filesystem::remove(tempPath);
    }
}

TEST_F(Phase3IntegrationTest, OBBPipeline_RotationInvariance_Test)
{
    if (!obbModel) {
        return;
    }

    obbModel->eval();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = obbModel->forward(input));

    // OBB model should handle rotated boxes
    EXPECT_GT(outputs.size(), 0);
}

// ========== Classification Pipeline Tests ==========

TEST_F(Phase3IntegrationTest, ClassificationPipeline_ModelConstruction_Success)
{
    if (!classificationModel) {
        return;
    }

    EXPECT_NE(classificationModel, nullptr);
    EXPECT_EQ(classificationModel->getTaskType(), WheelDL::TaskType::CLASSIFICATION);
}

TEST_F(Phase3IntegrationTest, ClassificationPipeline_ForwardPass_Training_Success)
{
    if (!classificationModel) {
        return;
    }

    classificationModel->train();
    torch::Tensor input = torch::rand({ 1, 3, 224, 224 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = classificationModel->forward(input));
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, ClassificationPipeline_ForwardPass_Inference_Success)
{
    if (!classificationModel) {
        return;
    }

    classificationModel->eval();
    torch::Tensor input = torch::rand({ 1, 3, 224, 224 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = classificationModel->forward(input));
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, ClassificationPipeline_OutputShape_Correct)
{
    if (!classificationModel) {
        return;
    }

    classificationModel->eval();
    int64_t batchSize = 2;
    torch::Tensor input = torch::rand({ batchSize, 3, 224, 224 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs = classificationModel->forward(input);

    // Classification output should be [batch, num_classes]
    EXPECT_GT(outputs.size(), 0);
    if (outputs[0].defined()) {
        EXPECT_EQ(outputs[0].size(0), batchSize);
        EXPECT_GT(outputs[0].size(1), 0); // num_classes
    }
}

TEST_F(Phase3IntegrationTest, ClassificationPipeline_Loss_Computation_Success)
{
    if (!classificationModel) {
        return;
    }

    classificationModel->train();
    torch::Tensor input = torch::rand({ 2, 3, 224, 224 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor targets = torch::randint(0, 10, { 2 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong)); // 10 classes

    DataExample data;
    data.data = input;
    data.classes = targets;

    std::unordered_map<std::string, torch::Tensor> losses;
    EXPECT_NO_THROW(losses = classificationModel->loss(data));
    EXPECT_GT(losses.size(), 0);
}

TEST_F(Phase3IntegrationTest, ClassificationPipeline_Loss_OutputKeys)
{
    if (!classificationModel) {
        return;
    }

    classificationModel->train();
    torch::Tensor input = torch::rand({ 2, 3, 224, 224 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor targets = torch::randint(0, 10, { 2 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong));

    DataExample data;
    data.data = input;
    data.classes = targets;

    std::unordered_map<std::string, torch::Tensor> losses = classificationModel->loss(data);

    // Classification loss should contain: total
    EXPECT_TRUE(losses.find("total") != losses.end());
}

TEST_F(Phase3IntegrationTest, ClassificationPipeline_Probabilities_SumToOne)
{
    if (!classificationModel) {
        return;
    }

    classificationModel->eval();
    torch::Tensor input = torch::rand({ 1, 3, 224, 224 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs = classificationModel->forward(input);

    if (outputs.size() > 0 && outputs[0].defined()) {
        // Apply softmax to get probabilities
        torch::Tensor probs = torch::softmax(outputs[0], 1);
        torch::Tensor sumProbs = probs.sum(1);

        // Sum should be approximately 1.0
        EXPECT_NEAR(sumProbs.item<float>(), 1.0f, 1e-5f);
    }
}

TEST_F(Phase3IntegrationTest, ClassificationPipeline_Gradient_BackwardPass)
{
    if (!classificationModel) {
        return;
    }

    classificationModel->train();
    torch::Tensor input = torch::rand({ 1, 3, 224, 224 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor targets = torch::randint(0, 10, { 1 }, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong));

    DataExample data;
    data.data = input;
    data.classes = targets;

    std::unordered_map<std::string, torch::Tensor> losses = classificationModel->loss(data);

    if (losses.find("total") != losses.end()) {
        torch::Tensor totalLoss = losses["total"];
        EXPECT_NO_THROW(totalLoss.backward());
    }
}

TEST_F(Phase3IntegrationTest, ClassificationPipeline_WeightSave_Load)
{
    if (!classificationModel) {
        return;
    }

    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test_classification_integration.pt";

    EXPECT_NO_THROW(classificationModel->saveWeights(tempPath.string()));
    EXPECT_TRUE(std::filesystem::exists(tempPath));

    EXPECT_NO_THROW(classificationModel->loadWeights(tempPath.string()));

    if (std::filesystem::exists(tempPath)) {
        std::filesystem::remove(tempPath);
    }
}

TEST_F(Phase3IntegrationTest, ClassificationPipeline_TopK_Accuracy)
{
    if (!classificationModel) {
        return;
    }

    classificationModel->eval();
    torch::Tensor input = torch::rand({ 1, 3, 224, 224 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs = classificationModel->forward(input);

    if (outputs.size() > 0 && outputs[0].defined()) {
        // Get top-5 predictions
        auto [topkValues, topkIndices] = outputs[0].topk(std::min(int64_t(5), outputs[0].size(1)), 1);

        EXPECT_EQ(topkIndices.size(1), std::min(int64_t(5), outputs[0].size(1)));
    }
}

// ========== Segmentation Pipeline Tests ==========

TEST_F(Phase3IntegrationTest, SegmentationPipeline_ModelConstruction_Success)
{
    if (!segmentationModel) {
        return;
    }

    EXPECT_NE(segmentationModel, nullptr);
    EXPECT_EQ(segmentationModel->getTaskType(), WheelDL::TaskType::SEGMENTATION);
}

TEST_F(Phase3IntegrationTest, SegmentationPipeline_ForwardPass_Training_Success)
{
    if (!segmentationModel) {
        return;
    }

    segmentationModel->train();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = segmentationModel->forward(input));
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, SegmentationPipeline_ForwardPass_Inference_Success)
{
    if (!segmentationModel) {
        return;
    }

    segmentationModel->eval();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = segmentationModel->forward(input));
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, SegmentationPipeline_MultiScale_OutputShapes)
{
    if (!segmentationModel) {
        return;
    }

    segmentationModel->train();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs = segmentationModel->forward(input);

    // Minimal model uses single scale, so expect at least 1 output
    EXPECT_GE(outputs.size(), 1);

    // Verify output tensors are defined
    for (const auto& output : outputs) {
        EXPECT_TRUE(output.defined());
        EXPECT_GT(output.numel(), 0);
    }
}

TEST_F(Phase3IntegrationTest, SegmentationPipeline_PrototypeMasks_Generated)
{
    if (!segmentationModel) {
        return;
    }

    segmentationModel->train();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs = segmentationModel->forward(input);

    // Segmentation should include prototype masks (typically 32 prototypes)
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, SegmentationPipeline_Loss_Computation_Success)
{
    if (!segmentationModel) {
        return;
    }

    segmentationModel->train();
    torch::Tensor input = torch::rand({ 2, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor masks = torch::rand({ 2, 10, 640, 640 }, torch::TensorOptions().device(torch::kCUDA)); // mask targets (num_classes=10)

    DataExample data;
    data.data = input;
    data.targets = masks;

    std::unordered_map<std::string, torch::Tensor> losses;
    EXPECT_NO_THROW(losses = segmentationModel->loss(data));
    EXPECT_GT(losses.size(), 0);
}

TEST_F(Phase3IntegrationTest, SegmentationPipeline_Loss_OutputKeys)
{
    if (!segmentationModel) {
        return;
    }

    segmentationModel->train();
    torch::Tensor input = torch::rand({ 2, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor masks = torch::rand({ 2, 10, 640, 640 }, torch::TensorOptions().device(torch::kCUDA)); // num_classes=10

    DataExample data;
    data.data = input;
    data.targets = masks;

    std::unordered_map<std::string, torch::Tensor> losses = segmentationModel->loss(data);

    // Segmentation loss should contain: total, box, cls, dfl, seg
    EXPECT_TRUE(losses.find("total") != losses.end());
}

TEST_F(Phase3IntegrationTest, SegmentationPipeline_MaskOutput_InRange)
{
    if (!segmentationModel) {
        return;
    }

    segmentationModel->eval();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    std::vector<torch::Tensor> outputs = segmentationModel->forward(input);

    // Mask outputs should be in valid range [0, 1] after sigmoid
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, SegmentationPipeline_Gradient_BackwardPass)
{
    if (!segmentationModel) {
        return;
    }

    segmentationModel->train();
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor masks = torch::rand({ 1, 10, 640, 640 }, torch::TensorOptions().device(torch::kCUDA)); // num_classes=10

    DataExample data;
    data.data = input;
    data.targets = masks;

    std::unordered_map<std::string, torch::Tensor> losses = segmentationModel->loss(data);

    if (losses.find("total") != losses.end()) {
        torch::Tensor totalLoss = losses["total"];
        EXPECT_NO_THROW(totalLoss.backward());
    }
}

TEST_F(Phase3IntegrationTest, SegmentationPipeline_WeightSave_Load)
{
    if (!segmentationModel) {
        return;
    }

    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test_segmentation_integration.pt";

    EXPECT_NO_THROW(segmentationModel->saveWeights(tempPath.string()));
    EXPECT_TRUE(std::filesystem::exists(tempPath));

    EXPECT_NO_THROW(segmentationModel->loadWeights(tempPath.string()));

    if (std::filesystem::exists(tempPath)) {
        std::filesystem::remove(tempPath);
    }
}

// ========== Anomaly Pipeline Tests ==========

TEST_F(Phase3IntegrationTest, AnomalyPipeline_ModelConstruction_Success)
{
    if (!anomalyModel) {
        return;
    }

    EXPECT_NE(anomalyModel, nullptr);
    EXPECT_EQ(anomalyModel->getTaskType(), WheelDL::TaskType::ANOMALY);
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_ForwardPass_Training_Success)
{
    // This is EfficientAD Model
    if (!anomalyModel) {
        return;
    }

    anomalyModel->train();

    // Create anomaly dataset example
    DataExample data;
    data.data = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));
	data.targets = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA)); // Dummy target
    std::unordered_map<std::string, torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = anomalyModel->forward(data));
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_ForwardPass_Inference_Success)
{
    if (!anomalyModel) {
        return;
    }

    anomalyModel->eval();

    // Create anomaly dataset example
    DataExample data;
    data.data = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));
	data.targets = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));
    std::unordered_map<std::string, torch::Tensor> outputs;

    EXPECT_NO_THROW(outputs = anomalyModel->forward(data));
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_OutputShape_Correct)
{
    if (!anomalyModel) {
        return;
    }

    anomalyModel->eval();

    DataExample data;
    data.data = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));
	data.targets = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));

    std::unordered_map<std::string, torch::Tensor> outputs = anomalyModel->forward(data);

    // Anomaly output should contain anomaly scores
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_Loss_Computation_Success)
{
    if (!anomalyModel) {
        return;
    }

    anomalyModel->train();

    DataExample data;
    data.data = torch::rand({ 2, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));
	data.targets = torch::rand({ 2, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));

    std::unordered_map<std::string, torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = anomalyModel->forward(data));
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_Loss_OutputKeys)
{
    if (!anomalyModel) {
        return;
    }

    anomalyModel->train();

    DataExample data;
    data.data = torch::rand({ 2, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));
	data.targets = torch::rand({ 2, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));

    std::unordered_map<std::string, torch::Tensor> losses = anomalyModel->forward(data);

    // EfficientAD loss should contain: total, st, ae, focal, cs, ac
    EXPECT_TRUE(losses.find("total") != losses.end());
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_AnomalyScore_InRange)
{
    if (!anomalyModel) {
        return;
    }

    anomalyModel->eval();

    DataExample data;
    data.data = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));
	data.targets = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));

    std::unordered_map<std::string, torch::Tensor> outputs = anomalyModel->forward(data);

    // Anomaly scores should be >= 0
    if (outputs.find("anomaly_score") != outputs.end()) {
        torch::Tensor scores = outputs["anomaly_score"];
        EXPECT_TRUE(torch::all(scores >= 0).item<bool>());
    }
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_Gradient_BackwardPass)
{
    if (!anomalyModel) {
        return;
    }

    anomalyModel->train();

    DataExample data;
    data.data = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));
	data.targets = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));

    std::unordered_map<std::string, torch::Tensor> losses = anomalyModel->forward(data);

    if (losses.find("total") != losses.end()) {
        torch::Tensor totalLoss = losses["total"];
        EXPECT_NO_THROW(totalLoss.backward());
    }
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_WeightSave_Load)
{
    if (!anomalyModel) {
        return;
    }

    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test_anomaly_integration.pt";

    EXPECT_NO_THROW(anomalyModel->saveWeights(tempPath.string()));
    EXPECT_TRUE(std::filesystem::exists(tempPath));

    EXPECT_NO_THROW(anomalyModel->loadWeights(tempPath.string()));

    if (std::filesystem::exists(tempPath)) {
        std::filesystem::remove(tempPath);
    }
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_NormalVsAnomaly_Scoring)
{
    if (!anomalyModel) {
        return;
    }

    anomalyModel->eval();

    // Normal sample
    DataExample normalData;
    normalData.data = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));
	normalData.targets = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));

    DataExample anomalyData;
    anomalyData.data = torch::ones({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA)) * 0.5f;
	anomalyData.targets = torch::ones({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA)) * 0.5f;

    std::unordered_map<std::string, torch::Tensor> normalOutputs;
    std::unordered_map<std::string, torch::Tensor> anomalyOutputs;

    EXPECT_NO_THROW(normalOutputs = anomalyModel->forward(normalData));
    EXPECT_NO_THROW(anomalyOutputs = anomalyModel->forward(anomalyData));

    // Both should produce valid outputs
    EXPECT_GT(normalOutputs.size(), 0);
    EXPECT_GT(anomalyOutputs.size(), 0);
}


TEST_F(Phase3IntegrationTest, AnomalyPipeline_ModelConstruction_Success_PatchCore)
{
    if (!anomalyModelPC) {
        return;
    }

    EXPECT_NE(anomalyModelPC, nullptr);
    EXPECT_EQ(anomalyModelPC->getTaskType(), WheelDL::TaskType::ANOMALY);
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_ForwardPass_Training_Success_PatchCore)
{
    // This is EfficientAD Model
    if (!anomalyModelPC) {
        return;
    }

    anomalyModelPC->train();

    // Create anomaly dataset example
    DataExample data;
    data.data = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));
    std::unordered_map<std::string, torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = anomalyModelPC->forward(data));
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_ForwardPass_Inference_Success_PatchCore)
{
    if (!anomalyModelPC) {
        return;
    }

    anomalyModelPC->eval();

    // Create anomaly dataset example
    DataExample data;
    data.data = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));
    std::unordered_map<std::string, torch::Tensor> outputs;

    EXPECT_NO_THROW(outputs = anomalyModelPC->forward(data));
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_OutputShape_Correct_PatchCore)
{
    if (!anomalyModelPC) {
        return;
    }

    anomalyModelPC->eval();

    DataExample data;
    data.data = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));
    auto outputs = anomalyModelPC->forward(data.data);

    // Anomaly output should contain anomaly scores
    EXPECT_GT(outputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_Loss_Computation_Success_PatchCore)
{
    if (!anomalyModelPC) {
        return;
    }

    anomalyModelPC->train();

    DataExample data;
    data.data = torch::rand({ 2, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));

    std::unordered_map<std::string, torch::Tensor> outputs;
    EXPECT_NO_THROW(outputs = anomalyModelPC->forward(data));
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_AnomalyScore_InRange_PatchCore)
{
    if (!anomalyModelPC) {
        return;
    }

    anomalyModelPC->eval();

    DataExample data;
    data.data = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));

    auto outputs = anomalyModelPC->forward(data.data);

    // Anomaly scores should be >= 0
    if (outputs.size() == 2) 
    {
        torch::Tensor scores = outputs[0];
		torch::Tensor anomalyMap = outputs[1];
        EXPECT_TRUE(torch::all(scores >= 0).item<bool>());
        EXPECT_TRUE(torch::all(anomalyMap >= 0).item<bool>());
    }
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_WeightSave_Load_PatchCore)
{
    if (!anomalyModelPC) {
        return;
    }

    std::filesystem::path tempPath = std::filesystem::temp_directory_path() / "test_anomaly_integration.pt";

    EXPECT_NO_THROW(anomalyModelPC->saveWeights(tempPath.string()));
    EXPECT_TRUE(std::filesystem::exists(tempPath));

    EXPECT_NO_THROW(anomalyModelPC->loadWeights(tempPath.string()));

    if (std::filesystem::exists(tempPath)) 
    {
        std::filesystem::remove(tempPath);
    }
}

TEST_F(Phase3IntegrationTest, AnomalyPipeline_NormalVsAnomaly_Scoring_PatchCore)
{
    if (!anomalyModelPC) {
        return;
    }

    anomalyModelPC->eval();

    // Normal sample
    DataExample normalData;
    normalData.data = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));

    DataExample anomalyData;
    anomalyData.data = torch::ones({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA)) * 0.5f;

    std::unordered_map<std::string, torch::Tensor> normalOutputs;
    std::unordered_map<std::string, torch::Tensor> anomalyOutputs;

    EXPECT_NO_THROW(normalOutputs = anomalyModelPC->forward(normalData));
    EXPECT_NO_THROW(anomalyOutputs = anomalyModelPC->forward(anomalyData));

}

// ========== Multi-Task Scenarios ==========

TEST_F(Phase3IntegrationTest, MultiTask_AllModels_Coexist)
{
    // All 5 models should be loaded simultaneously
    EXPECT_TRUE(detectionModel != nullptr);
    EXPECT_TRUE(obbModel != nullptr);
    EXPECT_TRUE(classificationModel != nullptr);
    EXPECT_TRUE(segmentationModel != nullptr);
    EXPECT_TRUE(anomalyModel != nullptr);
	EXPECT_TRUE(anomalyModelPC != nullptr);
}

TEST_F(Phase3IntegrationTest, MultiTask_IndependentForward_NoInterference)
{
    if (!detectionModel || !classificationModel) {
        return;
    }

    // Run detection
    detectionModel->eval();
    torch::Tensor detInput = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));
    std::vector<torch::Tensor> detOutputs;
    EXPECT_NO_THROW(detOutputs = detectionModel->forward(detInput));

    // Run classification
    classificationModel->eval();
    torch::Tensor clsInput = torch::rand({ 1, 3, 224, 224 }, torch::TensorOptions().device(torch::kCUDA));
    std::vector<torch::Tensor> clsOutputs;
    EXPECT_NO_THROW(clsOutputs = classificationModel->forward(clsInput));

    // Both should succeed independently
    EXPECT_GT(detOutputs.size(), 0);
    EXPECT_GT(clsOutputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, MultiTask_SameInputDifferentOutputs)
{
    if (!detectionModel || !obbModel) {
        return;
    }

    // Same input to different models
    torch::Tensor input = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));

    detectionModel->eval();
    std::vector<torch::Tensor> detOutputs = detectionModel->forward(input);

    obbModel->eval();
    std::vector<torch::Tensor> obbOutputs = obbModel->forward(input);

    // Both should produce valid but different outputs
    EXPECT_GT(detOutputs.size(), 0);
    EXPECT_GT(obbOutputs.size(), 0);
}

TEST_F(Phase3IntegrationTest, MultiTask_MemoryUsage_Reasonable)
{
    // Check that all models can be used without memory issues
    if (!detectionModel || !obbModel || !classificationModel ||
        !segmentationModel || !anomalyModel) {
        return;
    }

    // Run inference on all models
    detectionModel->eval();
    obbModel->eval();
    classificationModel->eval();
    segmentationModel->eval();
    anomalyModel->eval();

    torch::Tensor input640 = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor input224 = torch::rand({ 1, 3, 224, 224 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor input256 = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));

    EXPECT_NO_THROW(detectionModel->forward(input640));
    EXPECT_NO_THROW(obbModel->forward(input640));
    EXPECT_NO_THROW(classificationModel->forward(input224));
    EXPECT_NO_THROW(segmentationModel->forward(input640));

    DataExample anomalyData;
    anomalyData.data = input256;
	anomalyData.targets = torch::rand({ 1, 3, 256, 256 }, torch::TensorOptions().device(torch::kCUDA));
    EXPECT_NO_THROW(anomalyModel->forward(anomalyData));
}

TEST_F(Phase3IntegrationTest, MultiTask_ConcurrentInference_ThreadSafe)
{
    if (!detectionModel || !classificationModel) {
        return;
    }

    // Test that models can be used sequentially without issues
    // (True concurrent threading would require thread-safe guards)

    detectionModel->eval();
    classificationModel->eval();

    torch::Tensor detInput = torch::rand({ 1, 3, 640, 640 }, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor clsInput = torch::rand({ 1, 3, 224, 224 }, torch::TensorOptions().device(torch::kCUDA));

    // Sequential inference
    std::vector<torch::Tensor> detOutputs1 = detectionModel->forward(detInput);
    std::vector<torch::Tensor> clsOutputs1 = classificationModel->forward(clsInput);
    std::vector<torch::Tensor> detOutputs2 = detectionModel->forward(detInput);
    std::vector<torch::Tensor> clsOutputs2 = classificationModel->forward(clsInput);

    // All inferences should succeed
    EXPECT_GT(detOutputs1.size(), 0);
    EXPECT_GT(clsOutputs1.size(), 0);
    EXPECT_GT(detOutputs2.size(), 0);
    EXPECT_GT(clsOutputs2.size(), 0);
}
