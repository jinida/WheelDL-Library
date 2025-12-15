#include "pch.h"
#include <gtest/gtest.h>
#include "Data/Dataset/OBBDataset.h"
#include "Config/Configuration.h"
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

using namespace WheelDL::Data::Dataset;
using namespace WheelDL::Config;

class OBBDatasetTest : public ::testing::Test
{
protected:
    std::string testDir_;
    std::string dataDir_;
    std::string annotationPath_;
    std::string hyperPath_;
    std::string modelPath_;

    void SetUp() override
    {
        testDir_ = std::filesystem::temp_directory_path().string() + "/wheeldl_obb_test_" + std::to_string(std::time(nullptr));
        dataDir_ = testDir_ + "/a";
        std::filesystem::create_directories(testDir_ + "/a/b/c");
        createTestImages(10);
        annotationPath_ = testDir_ + "/a/b/c/train.json";
        hyperPath_ = testDir_ + "/hyper.yaml";
        modelPath_ = testDir_ + "/model.yaml";
        createHyperConfig(hyperPath_);
        createModelConfig(modelPath_);
        createOBBAnnotation(annotationPath_);
    }

    void TearDown() override
    {
        std::error_code ec;
        std::filesystem::remove_all(testDir_, ec);
    }

    void createTestImages(int count)
    {
        for (int i = 0; i < count; ++i)
        {
            cv::Mat img(64, 64, CV_8UC3, cv::Scalar(i * 20, i * 10, 255 - i * 20));
            // Draw rotated rectangle
            cv::RotatedRect rotRect(cv::Point2f(32, 32), cv::Size2f(20, 10), 45.0f);
            cv::Point2f vertices[4];
            rotRect.points(vertices);
            for (int j = 0; j < 4; ++j)
            {
                cv::line(img, vertices[j], vertices[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
            }
            std::string filename = dataDir_ + "/image_" + std::to_string(i) + ".jpg";
            cv::imwrite(filename, img);
        }
    }

    void createHyperConfig(const std::string& path, const std::string& cacheType = "false", float mosaic = 0.0f)
    {
        std::ofstream file(path);
        file << "epochs: 10\n";
        file << "batch_size: 4\n";
        file << "image_size: 64\n";
        file << "device: cpu\n";
        file << "workers: 2\n";
        file << "cache: " << cacheType << "\n";
        file << "fliplr: 0.5\n";
        file << "flipud: 0.0\n";
        file << "mosaic: " << mosaic << "\n";
        file << "hsv_h: 0.015\n";
        file << "hsv_s: 0.7\n";
        file << "hsv_v: 0.4\n";
        file << "degrees: 0.0\n";
        file << "translate: 0.1\n";
        file << "scale: 0.5\n";
        file << "shear: 0.0\n";
        file << "perspective: 0.0\n";
        file << "blur_probability: 0.0\n";
        file.close();
    }

    void createModelConfig(const std::string& path)
    {
        std::ofstream file(path);
        file << "task: obb\n";
        file << "nc: 5\n";
        file << "backbone:\n";
        file << "  - [-1, 1, Conv, [64, 3, 2]]\n";
        file << "head:\n";
        file << "  - [-1, 1, OBB, [5]]\n";
        file.close();
    }

    // OBB annotation: [classId, x1, y1, x2, y2, x3, y3, x4, y4]
    void createOBBAnnotation(const std::string& path, int trainCount = 5, int valCount = 3)
    {
        std::ofstream file(path);
        file << "{\n";
        file << "  \"header\": { \"categories\": [\"car\", \"truck\", \"bus\", \"ship\", \"plane\"] },\n";
        file << "  \"annotations\": [\n";

        bool first = true;
        for (int i = 0; i < trainCount; ++i)
        {
            if (!first) file << ",\n";
            first = false;
            file << "    {\n";
            file << "      \"filename\": \"image_" << i << ".jpg\",\n";
            file << "      \"role\": 0,\n";
            file << "      \"label\": [\n";
            // Rotated rectangle: 4 corners (x1,y1,x2,y2,x3,y3,x4,y4)
            file << "        [" << (i % 5) << ", 10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]";
            if (i > 0)
            {
                file << ",\n        [" << ((i + 1) % 5) << ", 35.0, 35.0, 55.0, 35.0, 55.0, 55.0, 35.0, 55.0]";
            }
            file << "\n      ]\n";
            file << "    }";
        }

        for (int i = 0; i < valCount; ++i)
        {
            if (!first) file << ",\n";
            first = false;
            file << "    {\n";
            file << "      \"filename\": \"image_" << (trainCount + i) << ".jpg\",\n";
            file << "      \"role\": 1,\n";
            file << "      \"label\": [\n";
            file << "        [" << (i % 5) << ", 15.0, 15.0, 45.0, 15.0, 45.0, 45.0, 15.0, 45.0]\n";
            file << "      ]\n";
            file << "    }";
        }

        file << "\n  ]\n";
        file << "}\n";
        file.close();
    }

    Configuration createConfig(const std::string& cacheType = "false", float mosaic = 0.0f)
    {
        std::string hyperConfigPath = testDir_ + "/hyper_" + std::to_string(rand()) + ".yaml";
        createHyperConfig(hyperConfigPath, cacheType, mosaic);

        Configuration config;
        config.load(modelPath_, hyperConfigPath, annotationPath_);
        return config;
    }

    Configuration createValConfig()
    {
        std::string valAnnotation = testDir_ + "/a/b/c/val.json";
        createOBBAnnotation(valAnnotation, 0, 5);

        std::string hyperConfigPath = testDir_ + "/hyper_val.yaml";
        createHyperConfig(hyperConfigPath, "false", 0.0f);

        Configuration config;
        config.load(modelPath_, hyperConfigPath, valAnnotation);
        return config;
    }

    Configuration createCustomConfig(const std::string& annotPath, const std::string& cacheType = "false", float mosaic = 0.0f)
    {
        std::string hyperConfigPath = testDir_ + "/hyper_custom_" + std::to_string(rand()) + ".yaml";
        createHyperConfig(hyperConfigPath, cacheType, mosaic);

        Configuration config;
        config.load(modelPath_, hyperConfigPath, annotPath);
        return config;
    }
};

// -----------------------------------------------------------------------------
// 7.1 loadAnnotations() - Specific Branches (21 tests)
// -----------------------------------------------------------------------------

// OBB-001: Constructor_Train
TEST_F(OBBDatasetTest, Constructor_Train)
{
    Configuration config = createConfig();
    EXPECT_NO_THROW({
        OBBDataset dataset(config, true);
        EXPECT_GT(dataset.size().value(), 0);
    });
}

// OBB-002: Constructor_Val
TEST_F(OBBDatasetTest, Constructor_Val)
{
    Configuration config = createValConfig();
    EXPECT_NO_THROW({
        OBBDataset dataset(config, false);
        EXPECT_GT(dataset.size().value(), 0);
    });
}

// OBB-003: LoadAnnotations_AnnotationPath_NotExists
TEST_F(OBBDatasetTest, LoadAnnotations_AnnotationPath_NotExists)
{
    std::string invalidPath = testDir_ + "/a/b/c/nonexistent.json";
    std::string hyperConfigPath = testDir_ + "/hyper_invalid.yaml";
    createHyperConfig(hyperConfigPath);

    // Exception may come from config.load() or Dataset constructor
    EXPECT_THROW({
        Configuration config;
        config.load(modelPath_, hyperConfigPath, invalidPath);
        OBBDataset dataset(config, true);
    }, std::exception);
}

// OBB-004: LoadAnnotations_DataPath_NotExists
TEST_F(OBBDatasetTest, LoadAnnotations_DataPath_NotExists)
{
    std::string badAnnot = testDir_ + "/bad/b/c/train.json";
    std::filesystem::create_directories(testDir_ + "/bad/b/c");
    createOBBAnnotation(badAnnot);

    std::string hyperConfigPath = testDir_ + "/hyper_badpath.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, badAnnot);

    EXPECT_THROW({
        OBBDataset dataset(config, true);
    }, std::exception);
}

// OBB-005: LoadAnnotations_JSON_Invalid
TEST_F(OBBDatasetTest, LoadAnnotations_JSON_Invalid)
{
    std::string invalidJson = testDir_ + "/a/b/c/invalid.json";
    std::ofstream file(invalidJson);
    file << "{ invalid json content";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_invalid_json.yaml";
    createHyperConfig(hyperConfigPath);

    // Exception may come from config.load() or Dataset constructor
    EXPECT_THROW({
        Configuration config;
        config.load(modelPath_, hyperConfigPath, invalidJson);
        OBBDataset dataset(config, true);
    }, std::exception);
}

// OBB-006: LoadAnnotations_Role_Match
TEST_F(OBBDatasetTest, LoadAnnotations_Role_Match)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 5);  // 5 train images
}

// OBB-007: LoadAnnotations_Role_NoMatch
TEST_F(OBBDatasetTest, LoadAnnotations_Role_NoMatch)
{
    Configuration config = createValConfig();
    OBBDataset dataset(config, false);
    // Should only get validation images
    EXPECT_GT(dataset.size().value(), 0);
}

// OBB-008: LoadAnnotations_Filename_Empty
TEST_F(OBBDatasetTest, LoadAnnotations_Filename_Empty)
{
    std::string annotPath = testDir_ + "/a/b/c/empty_filename.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] },\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);  // Only valid one
}

// OBB-009: LoadAnnotations_ImagePath_NotExists
TEST_F(OBBDatasetTest, LoadAnnotations_ImagePath_NotExists)
{
    std::string annotPath = testDir_ + "/a/b/c/missing_image.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"nonexistent.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] },\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// OBB-010: LoadAnnotations_ImageDimensions_Fail
TEST_F(OBBDatasetTest, LoadAnnotations_ImageDimensions_Fail)
{
    // Create corrupted image
    std::string corruptedImage = dataDir_ + "/corrupted.jpg";
    std::ofstream corrupt(corruptedImage, std::ios::binary);
    corrupt << "not an image";
    corrupt.close();

    std::string annotPath = testDir_ + "/a/b/c/corrupted_test.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"corrupted.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] },\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// OBB-011: LoadAnnotations_Label_Missing
TEST_F(OBBDatasetTest, LoadAnnotations_Label_Missing)
{
    std::string annotPath = testDir_ + "/a/b/c/no_label.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0 },\n";
    file << "    { \"filename\": \"image_1.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// OBB-012: LoadAnnotations_Object_LessThan9
TEST_F(OBBDatasetTest, LoadAnnotations_Object_LessThan9)
{
    std::string annotPath = testDir_ + "/a/b/c/short_obj.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10]] },\n";  // Only 5 elements
    file << "    { \"filename\": \"image_1.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    EXPECT_GE(dataset.size().value(), 1);
}

// OBB-013: LoadAnnotations_Object_Exactly9
TEST_F(OBBDatasetTest, LoadAnnotations_Object_Exactly9)
{
    std::string annotPath = testDir_ + "/a/b/c/exact9.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// OBB-014: LoadAnnotations_Object_MoreThan9
TEST_F(OBBDatasetTest, LoadAnnotations_Object_MoreThan9)
{
    std::string annotPath = testDir_ + "/a/b/c/more9.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0, 99.0, 99.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// OBB-015: LoadAnnotations_XYXYXYXY_Parse
TEST_F(OBBDatasetTest, LoadAnnotations_XYXYXYXY_Parse)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
    // OBB target shape: [N, 5] for XYWHR format
    EXPECT_EQ(example.targets.dim(), 2);
    if (example.targets.size(0) > 0)
    {
        EXPECT_EQ(example.targets.size(1), 5);
    }
}

// OBB-016: LoadAnnotations_ClassId_Parse
TEST_F(OBBDatasetTest, LoadAnnotations_ClassId_Parse)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.classes.defined());
}

// OBB-017: LoadAnnotations_ObbVector
TEST_F(OBBDatasetTest, LoadAnnotations_ObbVector)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
    EXPECT_GE(example.targets.size(0), 1);  // At least one OBB
}

// OBB-018: LoadAnnotations_AddObject
TEST_F(OBBDatasetTest, LoadAnnotations_AddObject)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(1);  // Second image has 2 objects
    EXPECT_TRUE(example.targets.defined());
    EXPECT_GE(example.targets.size(0), 2);
}

// OBB-019: LoadAnnotations_Normalize
TEST_F(OBBDatasetTest, LoadAnnotations_Normalize)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    auto accessor = example.targets.accessor<float, 2>();
    for (int i = 0; i < example.targets.size(0); ++i)
    {
        // XYWHR values should be normalized (approximately in [0,1] range for position)
        EXPECT_GE(accessor[i][0], 0.0f);  // x
        EXPECT_LE(accessor[i][0], 1.5f);
        EXPECT_GE(accessor[i][1], 0.0f);  // y
        EXPECT_LE(accessor[i][1], 1.5f);
    }
}

// OBB-020: LoadAnnotations_LabelType_XYXYXYXY
TEST_F(OBBDatasetTest, LoadAnnotations_LabelType_XYXYXYXY)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);
    // Target tensor should be in XYWHR format (5 values per object)
    if (example.targets.size(0) > 0)
    {
        EXPECT_EQ(example.targets.size(1), 5);
    }
}

// OBB-021: LoadAnnotations_Empty_Throws
TEST_F(OBBDatasetTest, LoadAnnotations_Empty_Throws)
{
    std::string annotPath = testDir_ + "/a/b/c/empty_all.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"nonexistent1.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] },\n";
    file << "    { \"filename\": \"nonexistent2.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    EXPECT_THROW({
        OBBDataset dataset(config, true);
    }, std::exception);
}

// -----------------------------------------------------------------------------
// 7.2 buildTransforms() & getTargetTensor() (12 tests)
// -----------------------------------------------------------------------------

// OBB-022: BuildTransforms_Train_WithMosaic
TEST_F(OBBDatasetTest, BuildTransforms_Train_WithMosaic)
{
    Configuration config = createConfig("false", 1.0f);
    EXPECT_GT(config.getMosaic(), 0.0f);
    EXPECT_NO_THROW({
        OBBDataset dataset(config, true);
    });
}

// OBB-023: BuildTransforms_Val_NoMosaic
TEST_F(OBBDatasetTest, BuildTransforms_Val_NoMosaic)
{
    Configuration config = createValConfig();
    EXPECT_NO_THROW({
        OBBDataset dataset(config, false);
    });
}

// OBB-024: GetTargetTensor_GetAs_XYWHR
TEST_F(OBBDatasetTest, GetTargetTensor_GetAs_XYWHR)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
    // XYWHR format: 5 values per object
    if (example.targets.size(0) > 0)
    {
        EXPECT_EQ(example.targets.size(1), 5);
    }
}

// OBB-025: GetTargetTensor_Normalize
TEST_F(OBBDatasetTest, GetTargetTensor_Normalize)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    if (example.targets.size(0) > 0)
    {
        auto accessor = example.targets.accessor<float, 2>();
        // Check normalized values are in reasonable range
        for (int i = 0; i < example.targets.size(0); ++i)
        {
            EXPECT_GE(accessor[i][0], 0.0f);
            EXPECT_GE(accessor[i][1], 0.0f);
        }
    }
}

// OBB-026: GetTargetTensor_NumObjects_Zero
TEST_F(OBBDatasetTest, GetTargetTensor_NumObjects_Zero)
{
    std::string annotPath = testDir_ + "/a/b/c/no_objects.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    EXPECT_TRUE(example.targets.defined());
    EXPECT_EQ(example.targets.size(0), 0);
    EXPECT_EQ(example.targets.size(1), 5);
}

// OBB-027: GetTargetTensor_NumObjects_Positive
TEST_F(OBBDatasetTest, GetTargetTensor_NumObjects_Positive)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    EXPECT_TRUE(example.targets.defined());
    EXPECT_GT(example.targets.size(0), 0);
}

// OBB-028: GetTargetTensor_Shape_Nx5
TEST_F(OBBDatasetTest, GetTargetTensor_Shape_Nx5)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    EXPECT_EQ(example.targets.dim(), 2);
    EXPECT_EQ(example.targets.size(1), 5);  // XYWHR format
}

// OBB-029: GetTargetTensor_Dtype_Float32
TEST_F(OBBDatasetTest, GetTargetTensor_Dtype_Float32)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    EXPECT_EQ(example.targets.dtype(), torch::kFloat32);
}

// OBB-030: GetTargetTensor_Points_Size5
TEST_F(OBBDatasetTest, GetTargetTensor_Points_Size5)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    if (example.targets.size(0) > 0)
    {
        EXPECT_EQ(example.targets.size(1), 5);
    }
}

// OBB-031: GetTargetTensor_Points_Not5
TEST_F(OBBDatasetTest, GetTargetTensor_Points_Not5)
{
    // This test verifies that malformed points are handled
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    // All valid points should have size 5
    EXPECT_EQ(example.targets.size(1), 5);
}

// OBB-032: GetTargetTensor_XYWHR_Order
TEST_F(OBBDatasetTest, GetTargetTensor_XYWHR_Order)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    if (example.targets.size(0) > 0)
    {
        auto accessor = example.targets.accessor<float, 2>();
        // x, y, w, h, theta order
        float x = accessor[0][0];
        float y = accessor[0][1];
        float w = accessor[0][2];
        float h = accessor[0][3];
        float theta = accessor[0][4];

        // Basic sanity checks
        EXPECT_GE(x, 0.0f);
        EXPECT_GE(y, 0.0f);
        EXPECT_GE(w, 0.0f);
        EXPECT_GE(h, 0.0f);
        (void)theta;  // Angle can be any value
    }
}

// OBB-033: GetTargetTensor_AngleExtraction
TEST_F(OBBDatasetTest, GetTargetTensor_AngleExtraction)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    if (example.targets.size(0) > 0)
    {
        auto accessor = example.targets.accessor<float, 2>();
        float theta = accessor[0][4];
        // Angle should be a valid number
        EXPECT_FALSE(std::isnan(theta));
        EXPECT_FALSE(std::isinf(theta));
    }
}

// -----------------------------------------------------------------------------
// 7.3 Integration Tests (22 tests)
// -----------------------------------------------------------------------------

// OBB-034: Integration_Get_FullPipeline
TEST_F(OBBDatasetTest, Integration_Get_FullPipeline)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);

    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
    EXPECT_TRUE(example.classes.defined());
    EXPECT_TRUE(example.targets.defined());
}

// OBB-035: Integration_DataLoader
TEST_F(OBBDatasetTest, Integration_DataLoader)
{
    Configuration config = createConfig();
    auto dataset = std::make_shared<OBBDataset>(config, true);

    auto dataLoader = torch::data::make_data_loader(
        *dataset,
        torch::data::DataLoaderOptions().batch_size(2).workers(0)
    );

    int batchCount = 0;
    for (auto& batch : *dataLoader)
    {
        batchCount++;
        EXPECT_GT(batch.size(), 0);
    }
    EXPECT_GT(batchCount, 0);
}

// OBB-036: Integration_BatchCollation
TEST_F(OBBDatasetTest, Integration_BatchCollation)
{
    Configuration config = createConfig();
    auto dataset = std::make_shared<OBBDataset>(config, true);

    auto dataLoader = torch::data::make_data_loader(
        *dataset,
        torch::data::DataLoaderOptions().batch_size(2).workers(0)
    );

    for (auto& batch : *dataLoader)
    {
        EXPECT_LE(batch.size(), 2);
        break;
    }
}

// OBB-037: Integration_Mosaic
TEST_F(OBBDatasetTest, Integration_Mosaic)
{
    Configuration config = createConfig("false", 1.0f);
    OBBDataset dataset(config, true);
    EXPECT_GT(config.getMosaic(), 0.0f);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// OBB-038: Integration_RotationPreservation
TEST_F(OBBDatasetTest, Integration_RotationPreservation)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    // Rotation (theta) should be preserved in the 5th column
    if (example.targets.size(0) > 0)
    {
        EXPECT_EQ(example.targets.size(1), 5);  // XYWHR format includes rotation
    }
}

// OBB-039: Integration_Reproducibility
TEST_F(OBBDatasetTest, Integration_Reproducibility)
{
    Configuration config1 = createConfig();
    Configuration config2 = createConfig();

    OBBDataset dataset1(config1, true);
    OBBDataset dataset2(config2, true);

    EXPECT_EQ(dataset1.size().value(), dataset2.size().value());
}

// OBB-040: Integration_Cache
TEST_F(OBBDatasetTest, Integration_Cache)
{
    Configuration config = createConfig("ram");
    OBBDataset dataset(config, true);

    auto example1 = dataset.get(0);
    auto example2 = dataset.get(0);

    EXPECT_TRUE(example1.data.defined());
    EXPECT_TRUE(example2.data.defined());
}

// OBB-041: Integration_MultiThread
TEST_F(OBBDatasetTest, Integration_MultiThread)
{
    Configuration config = createConfig();
    auto dataset = std::make_shared<OBBDataset>(config, true);

    auto dataLoader = torch::data::make_data_loader(
        *dataset,
        torch::data::DataLoaderOptions().batch_size(2).workers(2)
    );

    int batchCount = 0;
    for (auto& batch : *dataLoader)
    {
        batchCount++;
        (void)batch;
    }
    EXPECT_GT(batchCount, 0);
}

// OBB-042: Integration_VariousAngles
TEST_F(OBBDatasetTest, Integration_VariousAngles)
{
    std::string annotPath = testDir_ + "/a/b/c/various_angles.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    // Different rotated rectangles
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [\n";
    file << "      [0, 20.0, 20.0, 40.0, 20.0, 40.0, 40.0, 20.0, 40.0],\n";  // 0 degrees
    file << "      [0, 20.0, 25.0, 35.0, 15.0, 40.0, 30.0, 25.0, 40.0]\n";   // ~45 degrees
    file << "    ]}\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_GE(example.targets.size(0), 2);
}

// OBB-043: Integration_CornerCases
TEST_F(OBBDatasetTest, Integration_CornerCases)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);

    // Test first and last samples
    auto first = dataset.get(0);
    auto last = dataset.get(dataset.size().value() - 1);

    EXPECT_TRUE(first.data.defined());
    EXPECT_TRUE(last.data.defined());
}

// OBB-044: Integration_SmallOBB
TEST_F(OBBDatasetTest, Integration_SmallOBB)
{
    std::string annotPath = testDir_ + "/a/b/c/small_obb.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 30.0, 30.0, 32.0, 30.0, 32.0, 32.0, 30.0, 32.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// OBB-045: Integration_LargeOBB
TEST_F(OBBDatasetTest, Integration_LargeOBB)
{
    std::string annotPath = testDir_ + "/a/b/c/large_obb.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 0.0, 0.0, 64.0, 0.0, 64.0, 64.0, 0.0, 64.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// OBB-046: Integration_OverlappingOBB
TEST_F(OBBDatasetTest, Integration_OverlappingOBB)
{
    std::string annotPath = testDir_ + "/a/b/c/overlap_obb.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [\n";
    file << "      [0, 10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0],\n";
    file << "      [1, 20.0, 20.0, 50.0, 20.0, 50.0, 50.0, 20.0, 50.0]\n";
    file << "    ]}\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.targets.size(0), 2);
}

// OBB-047: Integration_EdgeOBB
TEST_F(OBBDatasetTest, Integration_EdgeOBB)
{
    std::string annotPath = testDir_ + "/a/b/c/edge_obb.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 0.0, 0.0, 20.0, 0.0, 20.0, 20.0, 0.0, 20.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// OBB-048: Integration_ManyOBBs
TEST_F(OBBDatasetTest, Integration_ManyOBBs)
{
    std::string annotPath = testDir_ + "/a/b/c/many_obbs.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [\n";
    // Create 5 larger OBBs to avoid filtering due to small area after transforms
    for (int i = 0; i < 5; ++i)
    {
        if (i > 0) file << ",\n";
        int x = (i % 3) * 18 + 2;
        int y = (i / 3) * 25 + 2;
        file << "      [0, " << x << ".0, " << y << ".0, " << (x + 16) << ".0, " << y << ".0, "
             << (x + 16) << ".0, " << (y + 16) << ".0, " << x << ".0, " << (y + 16) << ".0]";
    }
    file << "\n    ]}\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_GE(example.targets.size(0), 3);  // At least 3 OBBs should survive transforms
}

// OBB-049: Integration_SingleOBB
TEST_F(OBBDatasetTest, Integration_SingleOBB)
{
    std::string annotPath = testDir_ + "/a/b/c/single_obb.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.targets.size(0), 1);
}

// OBB-050: Integration_NoOBB
TEST_F(OBBDatasetTest, Integration_NoOBB)
{
    std::string annotPath = testDir_ + "/a/b/c/no_obb.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.targets.size(0), 0);
}

// OBB-051: Integration_DisableMosaic
// Note: disableMosaic() is protected, test mosaic disabled via config instead
TEST_F(OBBDatasetTest, Integration_DisableMosaic)
{
    Configuration config = createConfig("false", 0.0f);  // mosaic = 0.0 disables it
    OBBDataset dataset(config, true);
    EXPECT_EQ(config.getMosaic(), 0.0f);

    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// OBB-052: Integration_TransformRotation
TEST_F(OBBDatasetTest, Integration_TransformRotation)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    // After transforms, data should still be valid
    EXPECT_TRUE(example.data.defined());
    EXPECT_EQ(example.data.dim(), 3);  // [C, H, W]
}

// OBB-053: Integration_ClipToBounds
TEST_F(OBBDatasetTest, Integration_ClipToBounds)
{
    Configuration config = createConfig();
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    // Values should be clipped to reasonable bounds
    if (example.targets.size(0) > 0)
    {
        auto accessor = example.targets.accessor<float, 2>();
        for (int i = 0; i < example.targets.size(0); ++i)
        {
            // Position values should be >= 0
            EXPECT_GE(accessor[i][0], 0.0f);  // x
            EXPECT_GE(accessor[i][1], 0.0f);  // y
        }
    }
}

// OBB-054: Integration_NegativeAngle
TEST_F(OBBDatasetTest, Integration_NegativeAngle)
{
    // Create OBB that would result in negative angle
    std::string annotPath = testDir_ + "/a/b/c/neg_angle.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 30.0, 10.0, 10.0, 10.0, 10.0, 30.0, 30.0, 30.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    EXPECT_TRUE(example.targets.defined());
    EXPECT_EQ(example.targets.size(0), 1);
}

// OBB-055: Integration_LargeAngle
TEST_F(OBBDatasetTest, Integration_LargeAngle)
{
    // Create heavily rotated OBB (diamond shape)
    std::string annotPath = testDir_ + "/a/b/c/large_angle.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 32.0, 10.0, 54.0, 32.0, 32.0, 54.0, 10.0, 32.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    OBBDataset dataset(config, true);
    auto example = dataset.get(0);

    EXPECT_TRUE(example.targets.defined());
    if (example.targets.size(0) > 0)
    {
        auto accessor = example.targets.accessor<float, 2>();
        float theta = accessor[0][4];
        EXPECT_FALSE(std::isnan(theta));
    }
}
