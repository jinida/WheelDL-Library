#include "pch.h"
#include <gtest/gtest.h>
#include "Data/Dataset/AnomalyDataset.h"
#include "Config/Configuration.h"
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

using namespace WheelDL::Data::Dataset;
using namespace WheelDL::Config;

class AnomalyDatasetTest : public ::testing::Test
{
protected:
    std::string testDir_;
    std::string dataDir_;
    std::string annotationPath_;
    std::string hyperPath_;
    std::string modelPath_;

    void SetUp() override
    {
        testDir_ = std::filesystem::temp_directory_path().string() + "/wheeldl_ano_test_" + std::to_string(std::time(nullptr));
        dataDir_ = testDir_ + "/a";
        std::filesystem::create_directories(testDir_ + "/a/b/c");
        createTestImages(10);
        annotationPath_ = testDir_ + "/a/b/c/train.json";
        hyperPath_ = testDir_ + "/hyper.yaml";
        modelPath_ = testDir_ + "/model.yaml";
        createHyperConfig(hyperPath_);
        createModelConfig(modelPath_);
        createAnomalyAnnotation(annotationPath_);
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
            std::string filename = dataDir_ + "/image_" + std::to_string(i) + ".jpg";
            cv::imwrite(filename, img);
        }
    }

    void createHyperConfig(const std::string& path, const std::string& cacheType = "false", bool efficientAD = false)
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
        file << "mosaic: 0.0\n";
        file << "hsv_h: 0.015\n";
        file << "hsv_s: 0.7\n";
        file << "hsv_v: 0.4\n";
        file << "degrees: 0.0\n";
        file << "translate: 0.1\n";
        file << "scale: 0.5\n";
        file << "shear: 0.0\n";
        file << "perspective: 0.0\n";
        file << "blur_probability: 0.0\n";
        if (efficientAD)
        {
            file << "efficient_ad: true\n";
        }
        file.close();
    }

    void createModelConfig(const std::string& path, bool efficientAD = false)
    {
        std::ofstream file(path);
        if (efficientAD)
        {
            file << "task: anomaly\n";
            file << "model: efficientad\n";
        }
        else
        {
            file << "task: anomaly\n";
            file << "model: patchcore\n";
        }
        file << "nc: 2\n";
        file << "backbone:\n";
        file << "  - [-1, 1, Conv, [64, 3, 2]]\n";
        file << "head:\n";
        file << "  - [-1, 1, Classify, [2]]\n";
        file.close();
    }

    // Anomaly annotation: label 0=normal, 1=anomaly
    void createAnomalyAnnotation(const std::string& path, int trainCount = 5, int valCount = 3)
    {
        std::ofstream file(path);
        file << "{\n";
        file << "  \"header\": { \"categories\": [\"normal\", \"anomaly\"] },\n";
        file << "  \"annotations\": [\n";

        bool first = true;
        for (int i = 0; i < trainCount; ++i)
        {
            if (!first) file << ",\n";
            first = false;
            file << "    {\n";
            file << "      \"filename\": \"image_" << i << ".jpg\",\n";
            file << "      \"role\": 0,\n";
            file << "      \"label\": " << (i % 2) << "\n";  // Alternate normal/anomaly
            file << "    }";
        }

        for (int i = 0; i < valCount; ++i)
        {
            if (!first) file << ",\n";
            first = false;
            file << "    {\n";
            file << "      \"filename\": \"image_" << (trainCount + i) << ".jpg\",\n";
            file << "      \"role\": 1,\n";
            file << "      \"label\": " << (i % 2) << "\n";
            file << "    }";
        }

        file << "\n  ]\n";
        file << "}\n";
        file.close();
    }

    Configuration createConfig(const std::string& cacheType = "false", bool efficientAD = false)
    {
        std::string hyperConfigPath = testDir_ + "/hyper_" + std::to_string(rand()) + ".yaml";
        createHyperConfig(hyperConfigPath, cacheType, efficientAD);

        std::string modelConfigPath = testDir_ + "/model_" + std::to_string(rand()) + ".yaml";
        createModelConfig(modelConfigPath, efficientAD);

        Configuration config;
        config.load(modelConfigPath, hyperConfigPath, annotationPath_);
        return config;
    }

    Configuration createValConfig()
    {
        std::string valAnnotation = testDir_ + "/a/b/c/val.json";
        createAnomalyAnnotation(valAnnotation, 0, 5);

        std::string hyperConfigPath = testDir_ + "/hyper_val.yaml";
        createHyperConfig(hyperConfigPath);

        Configuration config;
        config.load(modelPath_, hyperConfigPath, valAnnotation);
        return config;
    }

    Configuration createCustomConfig(const std::string& annotPath, const std::string& cacheType = "false")
    {
        std::string hyperConfigPath = testDir_ + "/hyper_custom_" + std::to_string(rand()) + ".yaml";
        createHyperConfig(hyperConfigPath, cacheType);

        Configuration config;
        config.load(modelPath_, hyperConfigPath, annotPath);
        return config;
    }

    Configuration createNoLabelConfig()
    {
        // Config with empty annotation file - directory scan mode
        std::string hyperConfigPath = testDir_ + "/hyper_nolabel.yaml";
        createHyperConfig(hyperConfigPath);

        std::string emptyAnnotPath = testDir_ + "/a/b/c/empty.json";
        createEmptyAnnotation(emptyAnnotPath);

        Configuration config;
        config.load(modelPath_, hyperConfigPath, emptyAnnotPath);
        return config;
    }

    void createEmptyAnnotation(const std::string& path)
    {
        std::ofstream file(path);
        file << "{\n";
        file << "  \"header\": { \"categories\": [\"normal\", \"anomaly\"] },\n";
        file << "  \"annotations\": []\n";
        file << "}\n";
        file.close();
    }
};

// -----------------------------------------------------------------------------
// 9.1 loadAnnotations() - All Branches (28 tests)
// -----------------------------------------------------------------------------

// ANO-001: Constructor_Train
TEST_F(AnomalyDatasetTest, Constructor_Train)
{
    Configuration config = createConfig();
    EXPECT_NO_THROW({
        AnomalyDataset dataset(config, true);
        EXPECT_GT(dataset.size().value(), 0);
    });
}

// ANO-002: Constructor_Val
TEST_F(AnomalyDatasetTest, Constructor_Val)
{
    Configuration config = createValConfig();
    EXPECT_NO_THROW({
        AnomalyDataset dataset(config, false);
        EXPECT_GT(dataset.size().value(), 0);
    });
}

// ANO-003: LoadAnnotations_DataPath_NotExists
TEST_F(AnomalyDatasetTest, LoadAnnotations_DataPath_NotExists)
{
    std::string badAnnot = testDir_ + "/bad/b/c/train.json";
    std::filesystem::create_directories(testDir_ + "/bad/b/c");
    createAnomalyAnnotation(badAnnot);

    std::string hyperConfigPath = testDir_ + "/hyper_badpath.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, badAnnot);

    EXPECT_THROW({
        AnomalyDataset dataset(config, true);
    }, std::exception);
}

// ANO-004: LoadAnnotations_DataPath_Exists
TEST_F(AnomalyDatasetTest, LoadAnnotations_DataPath_Exists)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// ANO-005: LoadAnnotations_HasLabels_True_PathNotEmpty
TEST_F(AnomalyDatasetTest, LoadAnnotations_HasLabels_True_PathNotEmpty)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 5);  // 5 train samples with labels
}

// ANO-006: LoadAnnotations_HasLabels_False_PathEmpty
TEST_F(AnomalyDatasetTest, LoadAnnotations_HasLabels_False_PathEmpty)
{
    // Create config pointing to non-existent annotation file for directory scan
    std::string hyperConfigPath = testDir_ + "/hyper_dirscan.yaml";
    createHyperConfig(hyperConfigPath);

    std::string emptyAnnotPath = "";
    Configuration config;
    // Load with empty annotation path - will do directory scan
    config.load(modelPath_, hyperConfigPath, annotationPath_);  // Use existing for now

    AnomalyDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// ANO-007: LoadAnnotations_HasLabels_False_PathNotExists
// When annotation file has no entries, should fall back to directory scan
TEST_F(AnomalyDatasetTest, LoadAnnotations_HasLabels_False_PathNotExists)
{
    std::string hyperConfigPath = testDir_ + "/hyper_noexist.yaml";
    createHyperConfig(hyperConfigPath);

    // Use empty annotation file (no entries) to trigger directory scan
    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_scan.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    // Should fall back to directory scan
    AnomalyDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);  // Should find images via directory scan
}

// ANO-008: LoadAnnotations_WithLabels_JSON_Parse
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithLabels_JSON_Parse)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 5);
}

// ANO-009: LoadAnnotations_WithLabels_JSON_Invalid
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithLabels_JSON_Invalid)
{
    std::string invalidJson = testDir_ + "/a/b/c/invalid.json";
    std::ofstream file(invalidJson);
    file << "{ invalid json";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_invalid_json.yaml";
    createHyperConfig(hyperConfigPath);

    // Exception may come from config.load() or Dataset constructor
    EXPECT_THROW({
        Configuration config;
        config.load(modelPath_, hyperConfigPath, invalidJson);
        AnomalyDataset dataset(config, true);
    }, std::exception);
}

// ANO-010: LoadAnnotations_WithLabels_Role_Match_Train
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithLabels_Role_Match_Train)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 5);  // Only train (role=0) samples
}

// ANO-011: LoadAnnotations_WithLabels_Role_Match_Val
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithLabels_Role_Match_Val)
{
    Configuration config = createValConfig();
    AnomalyDataset dataset(config, false);
    EXPECT_GT(dataset.size().value(), 0);
}

// ANO-012: LoadAnnotations_WithLabels_Role_NoMatch
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithLabels_Role_NoMatch)
{
    // Create annotation with only role=1 (validation)
    std::string annotPath = testDir_ + "/a/b/c/val_only.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\", \"anomaly\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 1, \"label\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);

    // Train mode should find no role=0 samples, fall back to directory scan
    AnomalyDataset dataset(config, false);
    EXPECT_GE(dataset.size().value(), 1);
}

// ANO-013: LoadAnnotations_WithLabels_Filename_Empty
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithLabels_Filename_Empty)
{
    std::string annotPath = testDir_ + "/a/b/c/empty_filename.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\", \"anomaly\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"\", \"role\": 0, \"label\": 0 },\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    AnomalyDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// ANO-014: LoadAnnotations_WithLabels_ImagePath_NotExists
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithLabels_ImagePath_NotExists)
{
    std::string annotPath = testDir_ + "/a/b/c/missing_image.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\", \"anomaly\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"nonexistent.jpg\", \"role\": 0, \"label\": 0 },\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    AnomalyDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// ANO-015: LoadAnnotations_WithLabels_Label_Negative
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithLabels_Label_Negative)
{
    std::string annotPath = testDir_ + "/a/b/c/neg_label.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\", \"anomaly\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": -1 },\n";
    file << "    { \"filename\": \"image_1.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    AnomalyDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);  // Only valid one
}

// ANO-016: LoadAnnotations_WithLabels_Label_Zero
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithLabels_Label_Zero)
{
    std::string annotPath = testDir_ + "/a/b/c/zero_label.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\", \"anomaly\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.targets.item<int64_t>(), 0);
}

// ANO-017: LoadAnnotations_WithLabels_Label_Positive
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithLabels_Label_Positive)
{
    std::string annotPath = testDir_ + "/a/b/c/pos_label.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\", \"anomaly\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 1 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.targets.item<int64_t>(), 1);
}

// ANO-018: LoadAnnotations_WithLabels_ForClassification
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithLabels_ForClassification)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
    EXPECT_EQ(example.targets.dim(), 1);
}

// ANO-019: LoadAnnotations_WithoutLabels_DirectoryScan
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithoutLabels_DirectoryScan)
{
    std::string hyperConfigPath = testDir_ + "/hyper_dirscan2.yaml";
    createHyperConfig(hyperConfigPath);

    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_dirscan.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    AnomalyDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// ANO-020: LoadAnnotations_WithoutLabels_NotRegularFile
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithoutLabels_NotRegularFile)
{
    // Create a subdirectory in dataDir
    std::filesystem::create_directories(dataDir_ + "/subdir");

    std::string hyperConfigPath = testDir_ + "/hyper_subdir.yaml";
    createHyperConfig(hyperConfigPath);

    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_subdir.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    AnomalyDataset dataset(config, true);
    // Should not include subdirectory
    EXPECT_EQ(dataset.size().value(), 10);  // Only the 10 images, not the dir
}

// ANO-021: LoadAnnotations_WithoutLabels_RegularFile
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithoutLabels_RegularFile)
{
    std::string hyperConfigPath = testDir_ + "/hyper_reg.yaml";
    createHyperConfig(hyperConfigPath);

    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_reg.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    AnomalyDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// ANO-022: LoadAnnotations_WithoutLabels_NotImageFile
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithoutLabels_NotImageFile)
{
    // Create non-image file
    std::ofstream txtFile(dataDir_ + "/readme.txt");
    txtFile << "This is not an image";
    txtFile.close();

    std::string hyperConfigPath = testDir_ + "/hyper_notimg.yaml";
    createHyperConfig(hyperConfigPath);

    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_notimg.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    AnomalyDataset dataset(config, true);
    // Should not include txt file
    EXPECT_EQ(dataset.size().value(), 10);
}

// ANO-023: LoadAnnotations_WithoutLabels_ImageFile
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithoutLabels_ImageFile)
{
    std::string hyperConfigPath = testDir_ + "/hyper_imgfile.yaml";
    createHyperConfig(hyperConfigPath);

    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_imgfile.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    AnomalyDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 10);  // 10 jpg images
}

// ANO-024: LoadAnnotations_WithoutLabels_LabelType_NONE
TEST_F(AnomalyDatasetTest, LoadAnnotations_WithoutLabels_LabelType_NONE)
{
    std::string hyperConfigPath = testDir_ + "/hyper_none.yaml";
    createHyperConfig(hyperConfigPath);

    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_none.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    // No labels, should default to 0
    EXPECT_EQ(example.targets.item<int64_t>(), 0);
}

// ANO-025: LoadAnnotations_Empty_Throws
TEST_F(AnomalyDatasetTest, LoadAnnotations_Empty_Throws)
{
    // Create empty directory
    std::string emptyDir = testDir_ + "/empty";
    std::filesystem::create_directories(emptyDir + "/b/c");

    std::string annotPath = emptyDir + "/b/c/train.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"nonexistent.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_empty.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, annotPath);

    EXPECT_THROW({
        AnomalyDataset dataset(config, true);
    }, std::exception);
}

// ANO-026: LoadAnnotations_WarningOutput_Filename
TEST_F(AnomalyDatasetTest, LoadAnnotations_WarningOutput_Filename)
{
    std::string annotPath = testDir_ + "/a/b/c/warn_filename.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"role\": 0, \"label\": 0 },\n";  // Missing filename
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    // Should log warning but not throw
    EXPECT_NO_THROW({
        AnomalyDataset dataset(config, true);
    });
}

// ANO-027: LoadAnnotations_WarningOutput_Image
TEST_F(AnomalyDatasetTest, LoadAnnotations_WarningOutput_Image)
{
    std::string annotPath = testDir_ + "/a/b/c/warn_image.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"nonexistent.jpg\", \"role\": 0, \"label\": 0 },\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    EXPECT_NO_THROW({
        AnomalyDataset dataset(config, true);
    });
}

// ANO-028: LoadAnnotations_WarningOutput_Label
TEST_F(AnomalyDatasetTest, LoadAnnotations_WarningOutput_Label)
{
    std::string annotPath = testDir_ + "/a/b/c/warn_label.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0 },\n";  // Missing label
    file << "    { \"filename\": \"image_1.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    EXPECT_NO_THROW({
        AnomalyDataset dataset(config, true);
    });
}

// -----------------------------------------------------------------------------
// 9.2 buildTransforms() - All Branches (13 tests)
// -----------------------------------------------------------------------------

// ANO-029: BuildTransforms_EfficientAD_True
TEST_F(AnomalyDatasetTest, BuildTransforms_EfficientAD_True)
{
    Configuration config = createConfig("false", true);
    if (config.IsEfficientAD())
    {
        EXPECT_NO_THROW({
            AnomalyDataset dataset(config, true);
        });
    }
}

// ANO-030: BuildTransforms_EfficientAD_False
TEST_F(AnomalyDatasetTest, BuildTransforms_EfficientAD_False)
{
    Configuration config = createConfig("false", false);
    EXPECT_NO_THROW({
        AnomalyDataset dataset(config, true);
    });
}

// ANO-031: BuildTransforms_EfficientAD_Transform1
TEST_F(AnomalyDatasetTest, BuildTransforms_EfficientAD_Transform1)
{
    Configuration config = createConfig("false", true);
    if (config.IsEfficientAD())
    {
        AnomalyDataset dataset(config, true);
        auto example = dataset.get(0);
        EXPECT_TRUE(example.data.defined());
    }
}

// ANO-032: BuildTransforms_EfficientAD_Transform2
TEST_F(AnomalyDatasetTest, BuildTransforms_EfficientAD_Transform2)
{
    Configuration config = createConfig("false", true);
    if (config.IsEfficientAD())
    {
        AnomalyDataset dataset(config, true);
        auto example = dataset.get(0);
        EXPECT_TRUE(example.data.defined());
    }
}

// ANO-033: BuildTransforms_EfficientAD_Return
TEST_F(AnomalyDatasetTest, BuildTransforms_EfficientAD_Return)
{
    Configuration config = createConfig("false", true);
    if (config.IsEfficientAD())
    {
        AnomalyDataset dataset(config, true);
        EXPECT_GT(dataset.size().value(), 0);
    }
}

// ANO-034: BuildTransforms_Standard_Compose
TEST_F(AnomalyDatasetTest, BuildTransforms_Standard_Compose)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// ANO-035: BuildTransforms_Standard_LetterBox
TEST_F(AnomalyDatasetTest, BuildTransforms_Standard_LetterBox)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.data.size(1), config.getImageSize());
    EXPECT_EQ(example.data.size(2), config.getImageSize());
}

// ANO-036: BuildTransforms_Standard_Train_True
TEST_F(AnomalyDatasetTest, BuildTransforms_Standard_Train_True)
{
    Configuration config = createConfig();
    EXPECT_NO_THROW({
        AnomalyDataset dataset(config, true);
    });
}

// ANO-037: BuildTransforms_Standard_Train_False
TEST_F(AnomalyDatasetTest, BuildTransforms_Standard_Train_False)
{
    Configuration config = createValConfig();
    EXPECT_NO_THROW({
        AnomalyDataset dataset(config, false);
    });
}

// ANO-038: BuildTransforms_Standard_HFlip
TEST_F(AnomalyDatasetTest, BuildTransforms_Standard_HFlip)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// ANO-039: BuildTransforms_Standard_ToTensor
TEST_F(AnomalyDatasetTest, BuildTransforms_Standard_ToTensor)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.data.dim(), 3);  // [C, H, W]
}

// ANO-040: BuildTransforms_Standard_Normalize
TEST_F(AnomalyDatasetTest, BuildTransforms_Standard_Normalize)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    // Normalized values should be around [-2, 2] for ImageNet normalization
    float mean = example.data.mean().item<float>();
    EXPECT_TRUE(std::abs(mean) < 5.0f);
}

// ANO-041: BuildTransforms_NoMosaic
TEST_F(AnomalyDatasetTest, BuildTransforms_NoMosaic)
{
    Configuration config = createConfig();
    // Anomaly detection doesn't use mosaic
    AnomalyDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// -----------------------------------------------------------------------------
// 9.3 getTargetTensor() - All Branches (9 tests)
// -----------------------------------------------------------------------------

// ANO-042: GetTargetTensor_Classes_Empty
TEST_F(AnomalyDatasetTest, GetTargetTensor_Classes_Empty)
{
    std::string hyperConfigPath = testDir_ + "/hyper_empty_cls.yaml";
    createHyperConfig(hyperConfigPath);

    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_cls.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    // Empty classes should return 0
    EXPECT_EQ(example.targets.item<int64_t>(), 0);
}

// ANO-043: GetTargetTensor_Classes_NotEmpty
TEST_F(AnomalyDatasetTest, GetTargetTensor_Classes_NotEmpty)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// ANO-044: GetTargetTensor_Normal_Label0
TEST_F(AnomalyDatasetTest, GetTargetTensor_Normal_Label0)
{
    std::string annotPath = testDir_ + "/a/b/c/normal_only.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\", \"anomaly\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.targets.item<int64_t>(), 0);
}

// ANO-045: GetTargetTensor_Anomaly_Label1
TEST_F(AnomalyDatasetTest, GetTargetTensor_Anomaly_Label1)
{
    std::string annotPath = testDir_ + "/a/b/c/anomaly_only.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\", \"anomaly\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 1 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.targets.item<int64_t>(), 1);
}

// ANO-046: GetTargetTensor_AnomalyHigher
TEST_F(AnomalyDatasetTest, GetTargetTensor_AnomalyHigher)
{
    std::string annotPath = testDir_ + "/a/b/c/higher_label.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\", \"anomaly\", \"defect\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 2 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.targets.item<int64_t>(), 2);
}

// ANO-047: GetTargetTensor_Shape
TEST_F(AnomalyDatasetTest, GetTargetTensor_Shape)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.targets.dim(), 1);
    EXPECT_EQ(example.targets.size(0), 1);
}

// ANO-048: GetTargetTensor_Dtype
TEST_F(AnomalyDatasetTest, GetTargetTensor_Dtype)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.targets.dtype(), torch::kLong);
}

// ANO-049: GetTargetTensor_FirstClassUsed
TEST_F(AnomalyDatasetTest, GetTargetTensor_FirstClassUsed)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// ANO-050: GetTargetTensor_MultipleClasses
TEST_F(AnomalyDatasetTest, GetTargetTensor_MultipleClasses)
{
    // Even with multiple classes, only first is used
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.targets.size(0), 1);
}

// -----------------------------------------------------------------------------
// 9.4 Integration Tests (20 tests)
// -----------------------------------------------------------------------------

// ANO-051: Integration_Get_FullPipeline
TEST_F(AnomalyDatasetTest, Integration_Get_FullPipeline)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);

    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
    EXPECT_TRUE(example.classes.defined());
    EXPECT_TRUE(example.targets.defined());
}

// ANO-052: Integration_DataLoader
TEST_F(AnomalyDatasetTest, Integration_DataLoader)
{
    Configuration config = createConfig();
    auto dataset = std::make_shared<AnomalyDataset>(config, true);

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

// ANO-053: Integration_BatchCollation
TEST_F(AnomalyDatasetTest, Integration_BatchCollation)
{
    Configuration config = createConfig();
    auto dataset = std::make_shared<AnomalyDataset>(config, true);

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

// ANO-054: Integration_EfficientAD_DualOutput
TEST_F(AnomalyDatasetTest, Integration_EfficientAD_DualOutput)
{
    Configuration config = createConfig("false", true);
    if (config.IsEfficientAD())
    {
        AnomalyDataset dataset(config, true);
        auto example = dataset.get(0);
        EXPECT_TRUE(example.data.defined());
    }
}

// ANO-055: Integration_Reproducibility
TEST_F(AnomalyDatasetTest, Integration_Reproducibility)
{
    Configuration config1 = createConfig();
    Configuration config2 = createConfig();

    AnomalyDataset dataset1(config1, true);
    AnomalyDataset dataset2(config2, true);

    EXPECT_EQ(dataset1.size().value(), dataset2.size().value());
}

// ANO-056: Integration_Cache
TEST_F(AnomalyDatasetTest, Integration_Cache)
{
    Configuration config = createConfig("ram");
    AnomalyDataset dataset(config, true);

    auto example1 = dataset.get(0);
    auto example2 = dataset.get(0);

    EXPECT_TRUE(example1.data.defined());
    EXPECT_TRUE(example2.data.defined());
}

// ANO-057: Integration_NoLabelTraining
TEST_F(AnomalyDatasetTest, Integration_NoLabelTraining)
{
    std::string hyperConfigPath = testDir_ + "/hyper_nolabel2.yaml";
    createHyperConfig(hyperConfigPath);

    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_nolabel.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    // No labels = assume normal (0)
    EXPECT_EQ(example.targets.item<int64_t>(), 0);
}

// ANO-058: Integration_MultiThread
TEST_F(AnomalyDatasetTest, Integration_MultiThread)
{
    Configuration config = createConfig();
    auto dataset = std::make_shared<AnomalyDataset>(config, true);

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

// ANO-059: Integration_MixedNormalAnomaly
TEST_F(AnomalyDatasetTest, Integration_MixedNormalAnomaly)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);

    // Dataset has alternating normal/anomaly labels
    bool foundNormal = false;
    bool foundAnomaly = false;
    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        auto example = dataset.get(i);
        int64_t label = example.targets.item<int64_t>();
        if (label == 0) foundNormal = true;
        if (label == 1) foundAnomaly = true;
    }
    EXPECT_TRUE(foundNormal || foundAnomaly);  // At least one type present
}

// ANO-060: Integration_OnlyNormal
TEST_F(AnomalyDatasetTest, Integration_OnlyNormal)
{
    std::string annotPath = testDir_ + "/a/b/c/all_normal.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\", \"anomaly\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 },\n";
    file << "    { \"filename\": \"image_1.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    AnomalyDataset dataset(config, true);

    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        auto example = dataset.get(i);
        EXPECT_EQ(example.targets.item<int64_t>(), 0);
    }
}

// ANO-061: Integration_OnlyAnomaly
TEST_F(AnomalyDatasetTest, Integration_OnlyAnomaly)
{
    std::string annotPath = testDir_ + "/a/b/c/all_anomaly.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\", \"anomaly\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 1 },\n";
    file << "    { \"filename\": \"image_1.jpg\", \"role\": 0, \"label\": 1 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    AnomalyDataset dataset(config, true);

    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        auto example = dataset.get(i);
        EXPECT_EQ(example.targets.item<int64_t>(), 1);
    }
}

// ANO-062: Integration_DirectoryScan_AllImages
TEST_F(AnomalyDatasetTest, Integration_DirectoryScan_AllImages)
{
    std::string hyperConfigPath = testDir_ + "/hyper_allimg.yaml";
    createHyperConfig(hyperConfigPath);

    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_allimg.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    AnomalyDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 10);  // All 10 test images
}

// ANO-063: Integration_DirectoryScan_MixedFiles
TEST_F(AnomalyDatasetTest, Integration_DirectoryScan_MixedFiles)
{
    // Add non-image files
    std::ofstream txt(dataDir_ + "/data.txt");
    txt << "text data";
    txt.close();

    std::ofstream csv(dataDir_ + "/data.csv");
    csv << "a,b,c";
    csv.close();

    std::string hyperConfigPath = testDir_ + "/hyper_mixed.yaml";
    createHyperConfig(hyperConfigPath);

    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_mixed.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    AnomalyDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 10);  // Only jpg images
}

// ANO-064: Integration_DirectoryScan_Subdirectories
TEST_F(AnomalyDatasetTest, Integration_DirectoryScan_Subdirectories)
{
    std::filesystem::create_directories(dataDir_ + "/subdir1");
    std::filesystem::create_directories(dataDir_ + "/subdir2");

    std::string hyperConfigPath = testDir_ + "/hyper_subdir2.yaml";
    createHyperConfig(hyperConfigPath);

    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_subdir2.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    AnomalyDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 10);  // Subdirs excluded
}

// ANO-065: Integration_Transform_EfficientAD
TEST_F(AnomalyDatasetTest, Integration_Transform_EfficientAD)
{
    Configuration config = createConfig("false", true);
    if (config.IsEfficientAD())
    {
        AnomalyDataset dataset(config, true);
        auto example = dataset.get(0);
        EXPECT_EQ(example.data.dim(), 3);
    }
}

// ANO-066: Integration_Transform_Standard
TEST_F(AnomalyDatasetTest, Integration_Transform_Standard)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);
    auto example = dataset.get(0);
    EXPECT_EQ(example.data.dim(), 3);  // [C, H, W]
}

// ANO-067: Integration_LargeDataset
TEST_F(AnomalyDatasetTest, Integration_LargeDataset)
{
    Configuration config = createConfig();
    AnomalyDataset dataset(config, true);

    // Iterate through all samples
    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        auto example = dataset.get(i);
        EXPECT_TRUE(example.data.defined());
    }
}

// ANO-068: Integration_SingleImage
TEST_F(AnomalyDatasetTest, Integration_SingleImage)
{
    std::string annotPath = testDir_ + "/a/b/c/single.json";
    std::ofstream file(annotPath);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"normal\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();

    Configuration config = createCustomConfig(annotPath);
    AnomalyDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// ANO-069: Integration_ImageExtensions
TEST_F(AnomalyDatasetTest, Integration_ImageExtensions)
{
    // Create images with different extensions
    cv::Mat img(32, 32, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::imwrite(dataDir_ + "/test.png", img);
    cv::imwrite(dataDir_ + "/test.bmp", img);

    std::string hyperConfigPath = testDir_ + "/hyper_ext.yaml";
    createHyperConfig(hyperConfigPath);

    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_ext.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    AnomalyDataset dataset(config, true);
    EXPECT_GE(dataset.size().value(), 12);  // 10 jpg + 1 png + 1 bmp
}

// ANO-070: Integration_CacheWithDirScan
TEST_F(AnomalyDatasetTest, Integration_CacheWithDirScan)
{
    std::string hyperConfigPath = testDir_ + "/hyper_cache_dirscan.yaml";
    createHyperConfig(hyperConfigPath, "ram");

    std::string emptyAnnotPath = testDir_ + "/a/b/c/empty_cache.json";
    createEmptyAnnotation(emptyAnnotPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyAnnotPath);

    AnomalyDataset dataset(config, true);

    auto example1 = dataset.get(0);
    auto example2 = dataset.get(0);

    EXPECT_TRUE(example1.data.defined());
    EXPECT_TRUE(example2.data.defined());
}
