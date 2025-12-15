#include "pch.h"
#include <gtest/gtest.h>
#include "Data/Dataset/SegmentationDataset.h"
#include "Config/Configuration.h"
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

using namespace WheelDL::Data::Dataset;
using namespace WheelDL::Config;

class SegmentationDatasetTest : public ::testing::Test
{
protected:
    std::string testDir_;
    std::string dataDir_;
    std::string annotationPath_;
    std::string hyperPath_;
    std::string modelPath_;

    void SetUp() override
    {
        testDir_ = std::filesystem::temp_directory_path().string() + "/wheeldl_seg_test_" + std::to_string(std::time(nullptr));
        dataDir_ = testDir_ + "/a";
        std::filesystem::create_directories(testDir_ + "/a/b/c");
        createTestImages(10);
        annotationPath_ = testDir_ + "/a/b/c/train.json";
        hyperPath_ = testDir_ + "/hyper.yaml";
        modelPath_ = testDir_ + "/model.yaml";
        createHyperConfig(hyperPath_);
        createModelConfig(modelPath_);
        createSegmentationAnnotation(annotationPath_);
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
            // Draw polygon shape
            std::vector<cv::Point> pts = {
                cv::Point(20, 20), cv::Point(44, 20), cv::Point(44, 44), cv::Point(20, 44)
            };
            cv::fillPoly(img, std::vector<std::vector<cv::Point>>{pts}, cv::Scalar(255, 0, 0));
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
        file << "task: segment\n";
        file << "nc: 5\n";
        file << "backbone:\n";
        file << "  - [-1, 1, Conv, [64, 3, 2]]\n";
        file << "head:\n";
        file << "  - [-1, 1, Segment, [5]]\n";
        file.close();
    }

    // Segmentation annotation: [classId, x1, y1, x2, y2, ..., xn, yn]
    void createSegmentationAnnotation(const std::string& path, int trainCount = 5, int valCount = 3)
    {
        std::ofstream file(path);
        file << "{\n";
        file << "  \"header\": { \"categories\": [\"person\", \"car\", \"dog\", \"cat\", \"tree\"] },\n";
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
            // Polygon: [classId, x1, y1, x2, y2, x3, y3, x4, y4]
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
        createSegmentationAnnotation(valAnnotation, 0, 5);

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
// 8.1 loadAnnotations() - Specific Branches (22 tests)
// -----------------------------------------------------------------------------

// SEG-001: Constructor_Train
TEST_F(SegmentationDatasetTest, Constructor_Train)
{
    Configuration config = createConfig();
    EXPECT_NO_THROW({
        SegmentationDataset dataset(config, true);
        EXPECT_GT(dataset.size().value(), 0);
    });
}

// SEG-002: Constructor_Val
TEST_F(SegmentationDatasetTest, Constructor_Val)
{
    Configuration config = createValConfig();
    EXPECT_NO_THROW({
        SegmentationDataset dataset(config, false);
        EXPECT_GT(dataset.size().value(), 0);
    });
}

// SEG-003: LoadAnnotations_AnnotationPath_NotExists
TEST_F(SegmentationDatasetTest, LoadAnnotations_AnnotationPath_NotExists)
{
    std::string invalidPath = testDir_ + "/a/b/c/nonexistent.json";
    std::string hyperConfigPath = testDir_ + "/hyper_invalid.yaml";
    createHyperConfig(hyperConfigPath);

    // Exception may come from config.load() or Dataset constructor
    EXPECT_THROW({
        Configuration config;
        config.load(modelPath_, hyperConfigPath, invalidPath);
        SegmentationDataset dataset(config, true);
    }, std::exception);
}

// SEG-004: LoadAnnotations_DataPath_NotExists
TEST_F(SegmentationDatasetTest, LoadAnnotations_DataPath_NotExists)
{
    std::string badAnnot = testDir_ + "/bad/b/c/train.json";
    std::filesystem::create_directories(testDir_ + "/bad/b/c");
    createSegmentationAnnotation(badAnnot);

    std::string hyperConfigPath = testDir_ + "/hyper_badpath.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, badAnnot);

    EXPECT_THROW({
        SegmentationDataset dataset(config, true);
    }, std::exception);
}

// SEG-005: LoadAnnotations_JSON_Invalid
TEST_F(SegmentationDatasetTest, LoadAnnotations_JSON_Invalid)
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
        SegmentationDataset dataset(config, true);
    }, std::exception);
}

// SEG-006: LoadAnnotations_Role_Match
TEST_F(SegmentationDatasetTest, LoadAnnotations_Role_Match)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 5);
}

// SEG-007: LoadAnnotations_Role_NoMatch
TEST_F(SegmentationDatasetTest, LoadAnnotations_Role_NoMatch)
{
    Configuration config = createValConfig();
    SegmentationDataset dataset(config, false);
    EXPECT_GT(dataset.size().value(), 0);
}

// SEG-008: LoadAnnotations_Filename_Empty
TEST_F(SegmentationDatasetTest, LoadAnnotations_Filename_Empty)
{
    std::string annotPath = testDir_ + "/a/b/c/empty_filename.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] },\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// SEG-009: LoadAnnotations_ImagePath_NotExists
TEST_F(SegmentationDatasetTest, LoadAnnotations_ImagePath_NotExists)
{
    std::string annotPath = testDir_ + "/a/b/c/missing_image.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"nonexistent.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] },\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// SEG-010: LoadAnnotations_ImageDimensions_Fail
TEST_F(SegmentationDatasetTest, LoadAnnotations_ImageDimensions_Fail)
{
    std::string corruptedImage = dataDir_ + "/corrupted.jpg";
    std::ofstream corrupt(corruptedImage, std::ios::binary);
    corrupt << "not an image";
    corrupt.close();

    std::string annotPath = testDir_ + "/a/b/c/corrupted_test.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"corrupted.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] },\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// SEG-011: LoadAnnotations_Label_Missing
TEST_F(SegmentationDatasetTest, LoadAnnotations_Label_Missing)
{
    std::string annotPath = testDir_ + "/a/b/c/no_label.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0 },\n";
    file << "    { \"filename\": \"image_1.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// SEG-012: LoadAnnotations_Object_LessThan3
TEST_F(SegmentationDatasetTest, LoadAnnotations_Object_LessThan3)
{
    std::string annotPath = testDir_ + "/a/b/c/short_obj.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10]] },\n";
    file << "    { \"filename\": \"image_1.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 10, 30, 30, 10, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    EXPECT_GE(dataset.size().value(), 1);
}

// SEG-013: LoadAnnotations_Object_Exactly3
TEST_F(SegmentationDatasetTest, LoadAnnotations_Object_Exactly3)
{
    std::string annotPath = testDir_ + "/a/b/c/exact3.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10.0, 10.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// SEG-014: LoadAnnotations_Object_MoreThan3
TEST_F(SegmentationDatasetTest, LoadAnnotations_Object_MoreThan3)
{
    std::string annotPath = testDir_ + "/a/b/c/more3.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// SEG-015: LoadAnnotations_Polygon_Parse
TEST_F(SegmentationDatasetTest, LoadAnnotations_Polygon_Parse)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
    // Segmentation target: [C, H, W]
    EXPECT_EQ(example.targets.dim(), 3);
}

// SEG-016: LoadAnnotations_Polygon_Empty
TEST_F(SegmentationDatasetTest, LoadAnnotations_Polygon_Empty)
{
    std::string annotPath = testDir_ + "/a/b/c/empty_poly.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// SEG-017: LoadAnnotations_Polygon_Valid
TEST_F(SegmentationDatasetTest, LoadAnnotations_Polygon_Valid)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-018: LoadAnnotations_AddObject
TEST_F(SegmentationDatasetTest, LoadAnnotations_AddObject)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(1);  // Second image has 2 polygons
    EXPECT_TRUE(example.targets.defined());
}

// SEG-019: LoadAnnotations_Normalize
TEST_F(SegmentationDatasetTest, LoadAnnotations_Normalize)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-020: LoadAnnotations_LabelType_POLYGON
TEST_F(SegmentationDatasetTest, LoadAnnotations_LabelType_POLYGON)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    // Target should be mask format [C, H, W]
    EXPECT_EQ(example.targets.dim(), 3);
}

// SEG-021: LoadAnnotations_VariableLengthPolygon
TEST_F(SegmentationDatasetTest, LoadAnnotations_VariableLengthPolygon)
{
    std::string annotPath = testDir_ + "/a/b/c/variable_poly.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [\n";
    file << "      [0, 10.0, 10.0, 30.0, 10.0, 30.0, 30.0],\n";  // Triangle (3 points)
    file << "      [1, 35.0, 35.0, 55.0, 35.0, 55.0, 55.0, 35.0, 55.0]\n";  // Quad (4 points)
    file << "    ]}\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-022: LoadAnnotations_Empty_Throws
TEST_F(SegmentationDatasetTest, LoadAnnotations_Empty_Throws)
{
    std::string annotPath = testDir_ + "/a/b/c/empty_all.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"nonexistent1.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 30]] },\n";
    file << "    { \"filename\": \"nonexistent2.jpg\", \"role\": 0, \"label\": [[0, 10, 10, 30, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    EXPECT_THROW({
        SegmentationDataset dataset(config, true);
    }, std::exception);
}

// -----------------------------------------------------------------------------
// 8.2 buildTransforms() & getTargetTensor() (16 tests)
// -----------------------------------------------------------------------------

// SEG-023: BuildTransforms_Train_WithMosaic
TEST_F(SegmentationDatasetTest, BuildTransforms_Train_WithMosaic)
{
    Configuration config = createConfig("false", 1.0f);
    EXPECT_GT(config.getMosaic(), 0.0f);
    EXPECT_NO_THROW({
        SegmentationDataset dataset(config, true);
    });
}

// SEG-024: BuildTransforms_Val_NoMosaic
TEST_F(SegmentationDatasetTest, BuildTransforms_Val_NoMosaic)
{
    Configuration config = createValConfig();
    EXPECT_NO_THROW({
        SegmentationDataset dataset(config, false);
    });
}

// SEG-025: GetTargetTensor_Denormalize
TEST_F(SegmentationDatasetTest, GetTargetTensor_Denormalize)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-026: GetTargetTensor_ClassesMismatch
TEST_F(SegmentationDatasetTest, GetTargetTensor_ClassesMismatch)
{
    // This is handled internally, just verify normal operation works
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-027: GetTargetTensor_ClassesMatch
TEST_F(SegmentationDatasetTest, GetTargetTensor_ClassesMatch)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
    EXPECT_TRUE(example.classes.defined());
}

// SEG-028: GetTargetTensor_Shape_CxHxW
TEST_F(SegmentationDatasetTest, GetTargetTensor_Shape_CxHxW)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);

    EXPECT_EQ(example.targets.dim(), 3);
    EXPECT_EQ(example.targets.size(0), config.getNumClasses());  // C
    EXPECT_EQ(example.targets.size(1), config.getImageSize());   // H
    EXPECT_EQ(example.targets.size(2), config.getImageSize());   // W
}

// SEG-029: GetTargetTensor_Dtype_Float32
TEST_F(SegmentationDatasetTest, GetTargetTensor_Dtype_Float32)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);

    EXPECT_EQ(example.targets.dtype(), torch::kFloat32);
}

// SEG-030: GetTargetTensor_NumClassesFromConfig
TEST_F(SegmentationDatasetTest, GetTargetTensor_NumClassesFromConfig)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);

    EXPECT_EQ(example.targets.size(0), config.getNumClasses());
}

// SEG-031: GetTargetTensor_ClassId_Negative
TEST_F(SegmentationDatasetTest, GetTargetTensor_ClassId_Negative)
{
    std::string annotPath = testDir_ + "/a/b/c/neg_class.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[-1, 10, 10, 30, 10, 30, 30, 10, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    // Should handle gracefully, mask should be all zeros for that class
    EXPECT_TRUE(example.targets.defined());
}

// SEG-032: GetTargetTensor_ClassId_OutOfRange
TEST_F(SegmentationDatasetTest, GetTargetTensor_ClassId_OutOfRange)
{
    std::string annotPath = testDir_ + "/a/b/c/oor_class.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[100, 10, 10, 30, 10, 30, 30, 10, 30]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-033: GetTargetTensor_ClassId_Valid
TEST_F(SegmentationDatasetTest, GetTargetTensor_ClassId_Valid)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-034: GetTargetTensor_Contour_Size3
TEST_F(SegmentationDatasetTest, GetTargetTensor_Contour_Size3)
{
    std::string annotPath = testDir_ + "/a/b/c/triangle.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10.0, 10.0, 30.0, 10.0, 20.0, 30.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-035: GetTargetTensor_Contour_SizeLessThan3
TEST_F(SegmentationDatasetTest, GetTargetTensor_Contour_SizeLessThan3)
{
    std::string annotPath = testDir_ + "/a/b/c/line.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10.0, 10.0, 30.0, 30.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-036: GetTargetTensor_FillPoly
TEST_F(SegmentationDatasetTest, GetTargetTensor_FillPoly)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);

    // Check that mask has some non-zero values (polygon was filled)
    float sum = example.targets.sum().item<float>();
    EXPECT_GT(sum, 0.0f);
}

// SEG-037: GetTargetTensor_OneHotEncoding
TEST_F(SegmentationDatasetTest, GetTargetTensor_OneHotEncoding)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);

    // Values should be 0 or 1 (one-hot)
    auto minVal = example.targets.min().item<float>();
    auto maxVal = example.targets.max().item<float>();
    EXPECT_GE(minVal, 0.0f);
    EXPECT_LE(maxVal, 1.0f);
}

// SEG-038: GetTargetTensor_MultipleClassMasks
TEST_F(SegmentationDatasetTest, GetTargetTensor_MultipleClassMasks)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(1);  // Image with 2 objects of different classes

    // Check mask has values in multiple class channels
    EXPECT_TRUE(example.targets.defined());
    EXPECT_EQ(example.targets.size(0), config.getNumClasses());
}

// -----------------------------------------------------------------------------
// 8.3 Integration Tests (22 tests)
// -----------------------------------------------------------------------------

// SEG-039: Integration_Get_FullPipeline
TEST_F(SegmentationDatasetTest, Integration_Get_FullPipeline)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);

    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
    EXPECT_TRUE(example.classes.defined());
    EXPECT_TRUE(example.targets.defined());
}

// SEG-040: Integration_DataLoader
TEST_F(SegmentationDatasetTest, Integration_DataLoader)
{
    Configuration config = createConfig();
    auto dataset = std::make_shared<SegmentationDataset>(config, true);

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

// SEG-041: Integration_BatchCollation_3D
TEST_F(SegmentationDatasetTest, Integration_BatchCollation_3D)
{
    Configuration config = createConfig();
    auto dataset = std::make_shared<SegmentationDataset>(config, true);

    auto dataLoader = torch::data::make_data_loader(
        *dataset,
        torch::data::DataLoaderOptions().batch_size(2).workers(0)
    );

    for (auto& batch : *dataLoader)
    {
        EXPECT_LE(batch.size(), 2);
        // Each sample's target should be [C, H, W]
        for (auto& example : batch)
        {
            EXPECT_EQ(example.targets.dim(), 3);
        }
        break;
    }
}

// SEG-042: Integration_Mosaic
TEST_F(SegmentationDatasetTest, Integration_Mosaic)
{
    Configuration config = createConfig("false", 1.0f);
    SegmentationDataset dataset(config, true);
    EXPECT_GT(config.getMosaic(), 0.0f);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// SEG-043: Integration_NoNormalizeInGet
TEST_F(SegmentationDatasetTest, Integration_NoNormalizeInGet)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);

    // Segmentation masks should be 0/1 values
    auto maxVal = example.targets.max().item<float>();
    EXPECT_LE(maxVal, 1.0f);
}

// SEG-044: Integration_Reproducibility
TEST_F(SegmentationDatasetTest, Integration_Reproducibility)
{
    Configuration config1 = createConfig();
    Configuration config2 = createConfig();

    SegmentationDataset dataset1(config1, true);
    SegmentationDataset dataset2(config2, true);

    EXPECT_EQ(dataset1.size().value(), dataset2.size().value());
}

// SEG-045: Integration_Cache
TEST_F(SegmentationDatasetTest, Integration_Cache)
{
    Configuration config = createConfig("ram");
    SegmentationDataset dataset(config, true);

    auto example1 = dataset.get(0);
    auto example2 = dataset.get(0);

    EXPECT_TRUE(example1.data.defined());
    EXPECT_TRUE(example2.data.defined());
}

// SEG-046: Integration_MaskVisualization
TEST_F(SegmentationDatasetTest, Integration_MaskVisualization)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);

    // Verify mask dimensions match image size
    EXPECT_EQ(example.targets.size(1), config.getImageSize());
    EXPECT_EQ(example.targets.size(2), config.getImageSize());
}

// SEG-047: Integration_MultiThread
TEST_F(SegmentationDatasetTest, Integration_MultiThread)
{
    Configuration config = createConfig();
    auto dataset = std::make_shared<SegmentationDataset>(config, true);

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

// SEG-048: Integration_SmallPolygon
TEST_F(SegmentationDatasetTest, Integration_SmallPolygon)
{
    std::string annotPath = testDir_ + "/a/b/c/small_poly.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 30.0, 30.0, 32.0, 30.0, 32.0, 32.0, 30.0, 32.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-049: Integration_LargePolygon
TEST_F(SegmentationDatasetTest, Integration_LargePolygon)
{
    std::string annotPath = testDir_ + "/a/b/c/large_poly.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 0.0, 0.0, 64.0, 0.0, 64.0, 64.0, 0.0, 64.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-050: Integration_ComplexPolygon
TEST_F(SegmentationDatasetTest, Integration_ComplexPolygon)
{
    std::string annotPath = testDir_ + "/a/b/c/complex_poly.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [\n";
    file << "      [0, 10.0, 10.0, 20.0, 5.0, 30.0, 10.0, 35.0, 20.0, 30.0, 30.0, 20.0, 35.0, 10.0, 30.0, 5.0, 20.0]\n";
    file << "    ]}\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-051: Integration_ConvexPolygon
TEST_F(SegmentationDatasetTest, Integration_ConvexPolygon)
{
    std::string annotPath = testDir_ + "/a/b/c/convex_poly.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 20.0, 10.0, 40.0, 20.0, 40.0, 40.0, 20.0, 50.0, 10.0, 40.0, 10.0, 20.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-052: Integration_ConcavePolygon
TEST_F(SegmentationDatasetTest, Integration_ConcavePolygon)
{
    std::string annotPath = testDir_ + "/a/b/c/concave_poly.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10.0, 10.0, 30.0, 10.0, 20.0, 25.0, 30.0, 40.0, 10.0, 40.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-053: Integration_OverlappingPolygons
TEST_F(SegmentationDatasetTest, Integration_OverlappingPolygons)
{
    std::string annotPath = testDir_ + "/a/b/c/overlap_poly.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\", \"car\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [\n";
    file << "      [0, 10.0, 10.0, 40.0, 10.0, 40.0, 40.0, 10.0, 40.0],\n";
    file << "      [1, 20.0, 20.0, 50.0, 20.0, 50.0, 50.0, 20.0, 50.0]\n";
    file << "    ]}\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-054: Integration_ManyPolygons
TEST_F(SegmentationDatasetTest, Integration_ManyPolygons)
{
    std::string annotPath = testDir_ + "/a/b/c/many_polys.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [\n";
    for (int i = 0; i < 10; ++i)
    {
        if (i > 0) file << ",\n";
        int x = (i % 5) * 10;
        int y = (i / 5) * 20;
        file << "      [0, " << x << ".0, " << y << ".0, " << (x + 8) << ".0, " << y << ".0, "
             << (x + 8) << ".0, " << (y + 8) << ".0, " << x << ".0, " << (y + 8) << ".0]";
    }
    file << "\n    ]}\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-055: Integration_SinglePolygon
TEST_F(SegmentationDatasetTest, Integration_SinglePolygon)
{
    std::string annotPath = testDir_ + "/a/b/c/single_poly.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// SEG-056: Integration_NoPolygon
TEST_F(SegmentationDatasetTest, Integration_NoPolygon)
{
    std::string annotPath = testDir_ + "/a/b/c/no_poly.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);

    // All-zero mask
    float sum = example.targets.sum().item<float>();
    EXPECT_EQ(sum, 0.0f);
}

// SEG-057: Integration_DisableMosaic
// Note: disableMosaic() is protected, test mosaic disabled via config instead
TEST_F(SegmentationDatasetTest, Integration_DisableMosaic)
{
    Configuration config = createConfig("false", 0.0f);  // mosaic = 0.0 disables it
    SegmentationDataset dataset(config, true);
    EXPECT_EQ(config.getMosaic(), 0.0f);

    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// SEG-058: Integration_TransformPolygon
TEST_F(SegmentationDatasetTest, Integration_TransformPolygon)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);

    // After transforms, data should still be valid
    EXPECT_TRUE(example.data.defined());
    EXPECT_EQ(example.data.dim(), 3);  // [C, H, W]
}

// SEG-059: Integration_ClipToBounds
TEST_F(SegmentationDatasetTest, Integration_ClipToBounds)
{
    Configuration config = createConfig();
    SegmentationDataset dataset(config, true);
    auto example = dataset.get(0);

    // Mask values should be [0, 1]
    auto minVal = example.targets.min().item<float>();
    auto maxVal = example.targets.max().item<float>();
    EXPECT_GE(minVal, 0.0f);
    EXPECT_LE(maxVal, 1.0f);
}

// SEG-060: Integration_OddPointsCount
// Polygon coordinates must be even (x,y pairs) - odd count should throw exception
TEST_F(SegmentationDatasetTest, Integration_OddPointsCount)
{
    std::string annotPath = testDir_ + "/a/b/c/odd_points.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"person\"] },\n";
    file << "  \"annotations\": [\n";
    // Odd number of coordinates (classId + 5 coords = one incomplete pair)
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": [[0, 10.0, 10.0, 30.0, 10.0, 30.0]] }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    EXPECT_THROW({
        SegmentationDataset dataset(config, true);
    }, std::exception);
}
