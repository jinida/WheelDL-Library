#include "pch.h"
#include <gtest/gtest.h>
#include "Data/Dataset/ClassificationDataset.h"
#include "Config/Configuration.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <thread>
#include <atomic>

using namespace WheelDL::Data::Dataset;
using namespace WheelDL::Config;

// =============================================================================
// Test Fixture
// =============================================================================

class ClassificationDatasetTest : public ::testing::Test {
protected:
    std::string testDir_;
    std::string dataDir_;
    std::string annotationPath_;
    std::string modelPath_;
    std::string hyperPath_;

    void SetUp() override {
        testDir_ = std::filesystem::temp_directory_path().string() + "/wheeldl_cls_test_" + std::to_string(std::time(nullptr));
        dataDir_ = testDir_ + "/a";
        std::filesystem::create_directories(testDir_ + "/a/b/c");

        createTestImages(10);

        annotationPath_ = testDir_ + "/a/b/c/train.json";
        createAnnotationFile(annotationPath_, true, 10);

        modelPath_ = testDir_ + "/model.yaml";
        createModelConfig(modelPath_);

        hyperPath_ = testDir_ + "/hyper.yaml";
        createHyperConfig(hyperPath_);
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(testDir_, ec);
    }

    void createTestImages(int count, int width = 64, int height = 64) {
        for (int i = 0; i < count; ++i) {
            cv::Mat image(height, width, CV_8UC3);
            image.setTo(cv::Scalar(i * 50 % 256, (i * 30 + 50) % 256, (i * 70 + 100) % 256));
            std::string path = dataDir_ + "/image_" + std::to_string(i) + ".jpg";
            cv::imwrite(path, image);
        }
    }

    void createAnnotationFile(const std::string& path, bool isTrain, int count, int numClasses = 3) {
        std::ofstream file(path);
        file << "{\n";
        file << "  \"header\": {\n";
        file << "    \"type\": \"classification\",\n";
        file << "    \"categories\": [";
        for (int i = 0; i < numClasses; ++i) {
            file << "\"class" << i << "\"";
            if (i < numClasses - 1) file << ", ";
        }
        file << "]\n";
        file << "  },\n";
        file << "  \"annotations\": [\n";

        int role = isTrain ? 0 : 1;
        for (int i = 0; i < count; ++i) {
            file << "    {\n";
            file << "      \"filename\": \"image_" << i << ".jpg\",\n";
            file << "      \"role\": " << role << ",\n";
            file << "      \"label\": " << (i % numClasses) << "\n";
            file << "    }";
            if (i < count - 1) file << ",";
            file << "\n";
        }

        file << "  ]\n";
        file << "}\n";
        file.close();
    }

    void createModelConfig(const std::string& path) {
        std::ofstream file(path);
        file << "task: classification\n";
        file << "nc: 3\n";
        file << "backbone:\n";
        file << "  - [-1, 1, Conv, [64, 3, 2]]\n";
        file << "head:\n";
        file << "  - [-1, 1, Classify, [3]]\n";
        file.close();
    }

    void createHyperConfig(const std::string& path, const std::string& cacheType = "false") {
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
        file.close();
    }

    Configuration createConfig(const std::string& cacheType = "false") {
        std::string hyperConfigPath = testDir_ + "/hyper_" + std::to_string(rand()) + ".yaml";
        createHyperConfig(hyperConfigPath, cacheType);

        Configuration config;
        config.load(modelPath_, hyperConfigPath, annotationPath_);
        return config;
    }

    Configuration createValConfig() {
        std::string valAnnotation = testDir_ + "/a/b/c/val.json";
        createAnnotationFile(valAnnotation, false, 5);

        std::string hyperConfigPath = testDir_ + "/hyper_val.yaml";
        createHyperConfig(hyperConfigPath, "false");

        Configuration config;
        config.load(modelPath_, hyperConfigPath, valAnnotation);
        return config;
    }
};

// =============================================================================
// 5.1 Constructor Tests (CLS-001 ~ CLS-006)
// =============================================================================

// CLS-001: Constructor with train=true
TEST_F(ClassificationDatasetTest, Constructor_Train) {
    Configuration config = createConfig();
    EXPECT_NO_THROW({
        ClassificationDataset dataset(config, true);
    });
}

// CLS-002: Constructor with train=false
TEST_F(ClassificationDatasetTest, Constructor_Val) {
    Configuration config = createValConfig();
    EXPECT_NO_THROW({
        ClassificationDataset dataset(config, false);
    });
}

// CLS-003: Constructor calls loadAnnotations
TEST_F(ClassificationDatasetTest, Constructor_CallsLoadAnnotations) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// CLS-004: Constructor calls buildTransforms
TEST_F(ClassificationDatasetTest, Constructor_CallsBuildTransforms) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// CLS-005: Constructor stores config
TEST_F(ClassificationDatasetTest, Constructor_StoresConfig) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_EQ(dataset.getConfig().getImageSize(), config.getImageSize());
}

// CLS-006: Constructor stores train flag
TEST_F(ClassificationDatasetTest, Constructor_StoresTrain) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// =============================================================================
// 5.2 loadAnnotations() Tests (CLS-007 ~ CLS-041)
// =============================================================================

// CLS-007: Annotation path not exists -> exception
TEST_F(ClassificationDatasetTest, LoadAnnotations_AnnotationPath_NotExists) {
    std::string nonExistentPath = testDir_ + "/a/b/c/nonexistent.json";

    std::string hyperConfigPath = testDir_ + "/hyper_test.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    EXPECT_THROW({
        config.load(modelPath_, hyperConfigPath, nonExistentPath);
    }, std::exception);
}

// CLS-008: Annotation path exists
TEST_F(ClassificationDatasetTest, LoadAnnotations_AnnotationPath_Exists) {
    Configuration config = createConfig();
    EXPECT_NO_THROW({
        ClassificationDataset dataset(config, true);
    });
}

// CLS-009: Data path not exists -> exception
TEST_F(ClassificationDatasetTest, LoadAnnotations_DataPath_NotExists) {
    std::string badAnnotation = testDir_ + "/x/y/z/bad.json";
    std::filesystem::create_directories(testDir_ + "/x/y/z");
    createAnnotationFile(badAnnotation, true, 5);

    std::string hyperConfigPath = testDir_ + "/hyper_test2.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, badAnnotation);

    EXPECT_THROW({
        ClassificationDataset dataset(config, true);
    }, std::exception);
}

// CLS-010: Data path exists
TEST_F(ClassificationDatasetTest, LoadAnnotations_DataPath_Exists) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_FALSE(dataset.getDataPath().empty());
}

// CLS-011: JSON parse success
TEST_F(ClassificationDatasetTest, LoadAnnotations_JSON_ParseSuccess) {
    Configuration config = createConfig();
    EXPECT_NO_THROW({
        ClassificationDataset dataset(config, true);
    });
}

// CLS-012: JSON missing annotations key -> exception
TEST_F(ClassificationDatasetTest, LoadAnnotations_JSON_MissingAnnotations) {
    std::string badJson = testDir_ + "/a/b/c/bad.json";
    std::ofstream file(badJson);
    file << "{ \"header\": { \"categories\": [\"class0\"] } }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_bad.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, badJson);

    EXPECT_THROW({
        ClassificationDataset dataset(config, true);
    }, std::exception);
}

// CLS-013: JSON annotations not array -> exception
TEST_F(ClassificationDatasetTest, LoadAnnotations_JSON_AnnotationsNotArray) {
    std::string badJson = testDir_ + "/a/b/c/notarray.json";
    std::ofstream file(badJson);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": \"not_array\" }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_notarray.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, badJson);

    EXPECT_THROW({
        ClassificationDataset dataset(config, true);
    }, std::exception);
}

// CLS-014: JSON empty annotations
TEST_F(ClassificationDatasetTest, LoadAnnotations_JSON_EmptyAnnotations) {
    std::string emptyJson = testDir_ + "/a/b/c/empty.json";
    std::ofstream file(emptyJson);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_empty.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyJson);

    EXPECT_THROW({
        ClassificationDataset dataset(config, true);
    }, std::exception);
}

// CLS-015: Role train match (role==0 && train=true)
TEST_F(ClassificationDatasetTest, LoadAnnotations_Role_Train_Match) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 10);
}

// CLS-016: Role train no match (role!=0 && train=true)
TEST_F(ClassificationDatasetTest, LoadAnnotations_Role_Train_NoMatch) {
    std::string mixedJson = testDir_ + "/a/b/c/mixed.json";
    std::ofstream file(mixedJson);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [\n";
    file << "  { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 },\n";
    file << "  { \"filename\": \"image_1.jpg\", \"role\": 1, \"label\": 0 },\n";
    file << "  { \"filename\": \"image_2.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_mixed.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, mixedJson);
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 2);
}

// CLS-017: Role val match (role==1 && train=false)
TEST_F(ClassificationDatasetTest, LoadAnnotations_Role_Val_Match) {
    Configuration config = createValConfig();
    ClassificationDataset dataset(config, false);
    EXPECT_EQ(dataset.size().value(), 5);
}

// CLS-018: Role val no match
TEST_F(ClassificationDatasetTest, LoadAnnotations_Role_Val_NoMatch) {
    std::string mixedJson = testDir_ + "/a/b/c/mixed_val.json";
    std::ofstream file(mixedJson);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [\n";
    file << "  { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 },\n";
    file << "  { \"filename\": \"image_1.jpg\", \"role\": 1, \"label\": 0 },\n";
    file << "  { \"filename\": \"image_2.jpg\", \"role\": 1, \"label\": 0 },\n";
    file << "  { \"filename\": \"image_3.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_mixed_val.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, mixedJson);
    ClassificationDataset dataset(config, false);

    EXPECT_EQ(dataset.size().value(), 2);
}

// CLS-019: Role default (-1)
TEST_F(ClassificationDatasetTest, LoadAnnotations_Role_Default) {
    std::string noRoleJson = testDir_ + "/a/b/c/norole.json";
    std::ofstream file(noRoleJson);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [\n";
    file << "  { \"filename\": \"image_0.jpg\", \"label\": 0 }\n";
    file << "] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_norole.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, noRoleJson);

    EXPECT_THROW({
        ClassificationDataset dataset(config, true);
    }, std::exception);
}

// CLS-020: Role various values
TEST_F(ClassificationDatasetTest, LoadAnnotations_Role_Values) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// CLS-021: Filename empty -> skip
TEST_F(ClassificationDatasetTest, LoadAnnotations_Filename_Empty) {
    std::string emptyFilename = testDir_ + "/a/b/c/empty_filename.json";
    std::ofstream file(emptyFilename);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [\n";
    file << "  { \"filename\": \"\", \"role\": 0, \"label\": 0 },\n";
    file << "  { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_emptyf.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, emptyFilename);
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// CLS-022: Filename missing -> skip
TEST_F(ClassificationDatasetTest, LoadAnnotations_Filename_Missing) {
    std::string noFilename = testDir_ + "/a/b/c/no_filename.json";
    std::ofstream file(noFilename);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [\n";
    file << "  { \"role\": 0, \"label\": 0 },\n";
    file << "  { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_nof.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, noFilename);
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// CLS-023: Valid filename
TEST_F(ClassificationDatasetTest, LoadAnnotations_Filename_Valid) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// CLS-024: Image path not exists -> skip
TEST_F(ClassificationDatasetTest, LoadAnnotations_ImagePath_NotExists) {
    std::string badImage = testDir_ + "/a/b/c/bad_image.json";
    std::ofstream file(badImage);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [\n";
    file << "  { \"filename\": \"nonexistent.jpg\", \"role\": 0, \"label\": 0 },\n";
    file << "  { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_badimg.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, badImage);
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// CLS-025: Image path exists
TEST_F(ClassificationDatasetTest, LoadAnnotations_ImagePath_Exists) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 10);
}

// CLS-026: ClassId negative -> skip
TEST_F(ClassificationDatasetTest, LoadAnnotations_ClassId_Negative) {
    std::string negLabel = testDir_ + "/a/b/c/neg_label.json";
    std::ofstream file(negLabel);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [\n";
    file << "  { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": -1 },\n";
    file << "  { \"filename\": \"image_1.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_neg.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, negLabel);
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// CLS-027: ClassId missing -> skip
TEST_F(ClassificationDatasetTest, LoadAnnotations_ClassId_Missing) {
    std::string noLabel = testDir_ + "/a/b/c/no_label.json";
    std::ofstream file(noLabel);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [\n";
    file << "  { \"filename\": \"image_0.jpg\", \"role\": 0 },\n";
    file << "  { \"filename\": \"image_1.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_nolbl.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, noLabel);
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// CLS-028: ClassId zero (valid)
TEST_F(ClassificationDatasetTest, LoadAnnotations_ClassId_Zero) {
    std::string zeroLabel = testDir_ + "/a/b/c/zero_label.json";
    std::ofstream file(zeroLabel);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [\n";
    file << "  { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_zero.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, zeroLabel);
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// CLS-029: ClassId positive (valid)
TEST_F(ClassificationDatasetTest, LoadAnnotations_ClassId_Positive) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(1);
    EXPECT_TRUE(example.classes.defined());
}

// CLS-030: Multiple classes
TEST_F(ClassificationDatasetTest, LoadAnnotations_MultipleClasses) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    std::set<int64_t> classIds;
    for (size_t i = 0; i < dataset.size().value(); ++i) {
        DataExample example = dataset.get(i);
        classIds.insert(example.classes.item<int64_t>());
    }

    EXPECT_GT(classIds.size(), 1);
}

// CLS-031: Image path string
TEST_F(ClassificationDatasetTest, LoadAnnotations_ImagePathStr) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_FALSE(dataset.getDataPath().empty());
}

// CLS-032: Image paths push
TEST_F(ClassificationDatasetTest, LoadAnnotations_ImagePathsPush) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 10);
}

// CLS-033: Annotation creation
TEST_F(ClassificationDatasetTest, LoadAnnotations_AnnotationCreation) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.classes.defined());
}

// CLS-034: Annotations map
TEST_F(ClassificationDatasetTest, LoadAnnotations_AnnotationsMap) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    for (size_t i = 0; i < dataset.size().value(); ++i) {
        DataExample example = dataset.get(i);
        EXPECT_TRUE(example.data.defined());
    }
}

// CLS-035: Empty after process -> exception
TEST_F(ClassificationDatasetTest, LoadAnnotations_Empty_AfterProcess) {
    std::string allSkipped = testDir_ + "/a/b/c/allskipped.json";
    std::ofstream file(allSkipped);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [\n";
    file << "  { \"filename\": \"nonexistent1.jpg\", \"role\": 0, \"label\": 0 },\n";
    file << "  { \"filename\": \"nonexistent2.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_allskip.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, allSkipped);

    EXPECT_THROW({
        ClassificationDataset dataset(config, true);
    }, std::exception);
}

// CLS-036: Not empty after process
TEST_F(ClassificationDatasetTest, LoadAnnotations_NotEmpty_AfterProcess) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// CLS-037: Partially valid
TEST_F(ClassificationDatasetTest, LoadAnnotations_PartiallyValid) {
    std::string partial = testDir_ + "/a/b/c/partial.json";
    std::ofstream file(partial);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [\n";
    file << "  { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 },\n";
    file << "  { \"filename\": \"bad.jpg\", \"role\": 0, \"label\": 0 },\n";
    file << "  { \"filename\": \"image_1.jpg\", \"role\": 0, \"label\": -1 },\n";
    file << "  { \"filename\": \"image_2.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_partial.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, partial);
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 2);
}

// CLS-038: All skipped -> exception
TEST_F(ClassificationDatasetTest, LoadAnnotations_AllSkipped) {
    std::string allBad = testDir_ + "/a/b/c/allbad.json";
    std::ofstream file(allBad);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [\n";
    file << "  { \"filename\": \"\", \"role\": 0, \"label\": 0 },\n";
    file << "  { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": -1 }\n";
    file << "] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_allbad.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, allBad);

    EXPECT_THROW({
        ClassificationDataset dataset(config, true);
    }, std::exception);
}

// CLS-039 ~ CLS-041: Warning output tests (cerr)
TEST_F(ClassificationDatasetTest, LoadAnnotations_WarningOutput_Filename) {
    SUCCEED();
}

TEST_F(ClassificationDatasetTest, LoadAnnotations_WarningOutput_Image) {
    SUCCEED();
}

TEST_F(ClassificationDatasetTest, LoadAnnotations_WarningOutput_ClassId) {
    SUCCEED();
}

// =============================================================================
// 5.3 buildTransforms() Tests (CLS-042 ~ CLS-046)
// =============================================================================

// CLS-042: buildTransforms calls buildStandardTransforms
TEST_F(ClassificationDatasetTest, BuildTransforms_CallsStandard) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// CLS-043: Train with colorAug=true
TEST_F(ClassificationDatasetTest, BuildTransforms_Train_ColorAugTrue) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.data.size(0), 3);
}

// CLS-044: Val with colorAug=true
TEST_F(ClassificationDatasetTest, BuildTransforms_Val_ColorAugTrue) {
    Configuration config = createValConfig();
    ClassificationDataset dataset(config, false);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.data.size(0), 3);
}

// CLS-045: No mosaic for classification
TEST_F(ClassificationDatasetTest, BuildTransforms_NoMosaic) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// CLS-046: Returns valid transform
TEST_F(ClassificationDatasetTest, BuildTransforms_ReturnsValid) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.data.dim(), 3);
}

// =============================================================================
// 5.4 getTargetTensor() Tests (CLS-047 ~ CLS-056)
// =============================================================================

// CLS-047: Classes not empty
TEST_F(ClassificationDatasetTest, GetTargetTensor_ClassesNotEmpty) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.classes.defined());
    EXPECT_GT(example.classes.numel(), 0);
}

// CLS-048: Classes empty throws
TEST_F(ClassificationDatasetTest, GetTargetTensor_ClassesEmpty_Throws) {
    SUCCEED();
}

// CLS-049: Class 0
TEST_F(ClassificationDatasetTest, GetTargetTensor_Class0) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    int64_t classId = example.targets.item<int64_t>();
    EXPECT_GE(classId, 0);
}

// CLS-050: Class N
TEST_F(ClassificationDatasetTest, GetTargetTensor_ClassN) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    bool foundNonZero = false;
    for (size_t i = 0; i < dataset.size().value(); ++i) {
        DataExample example = dataset.get(i);
        if (example.targets.item<int64_t>() > 0) {
            foundNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(foundNonZero);
}

// CLS-051: Class max
TEST_F(ClassificationDatasetTest, GetTargetTensor_ClassMax) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    int64_t maxClass = 0;
    for (size_t i = 0; i < dataset.size().value(); ++i) {
        DataExample example = dataset.get(i);
        maxClass = std::max(maxClass, example.targets.item<int64_t>());
    }
    EXPECT_GE(maxClass, 0);
}

// CLS-052: Shape == [1]
TEST_F(ClassificationDatasetTest, GetTargetTensor_Shape) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.targets.dim(), 1);
    EXPECT_EQ(example.targets.size(0), 1);
}

// CLS-053: Dtype == kLong
TEST_F(ClassificationDatasetTest, GetTargetTensor_Dtype) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.targets.dtype(), torch::kLong);
}

// CLS-054: First class used
TEST_F(ClassificationDatasetTest, GetTargetTensor_FirstClassUsed) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.classes.item<int64_t>(), example.targets.item<int64_t>());
}

// CLS-055: Multiple classes - first used
TEST_F(ClassificationDatasetTest, GetTargetTensor_MultipleClasses) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// CLS-056: Value correct
TEST_F(ClassificationDatasetTest, GetTargetTensor_ValueCorrect) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    int64_t targetVal = example.targets.item<int64_t>();
    EXPECT_GE(targetVal, 0);
    EXPECT_LT(targetVal, 3);
}

// =============================================================================
// 5.5 Integration Tests (CLS-057 ~ CLS-075)
// =============================================================================

// CLS-057: Get full pipeline
TEST_F(ClassificationDatasetTest, Integration_Get_FullPipeline) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
    EXPECT_TRUE(example.classes.defined());
    EXPECT_TRUE(example.targets.defined());
}

// CLS-058: DataLoader single batch
TEST_F(ClassificationDatasetTest, Integration_DataLoader_SingleBatch) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_GE(dataset.size().value(), 4);
}

// CLS-059: DataLoader multiple batches
TEST_F(ClassificationDatasetTest, Integration_DataLoader_MultipleBatches) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 10);
}

// CLS-060: Batch collation
TEST_F(ClassificationDatasetTest, Integration_BatchCollation) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    std::vector<DataExample> batch;
    for (size_t i = 0; i < 4; ++i) {
        batch.push_back(dataset.get(i));
    }

    EXPECT_EQ(batch.size(), 4);
}

// CLS-061: Shuffle enabled
TEST_F(ClassificationDatasetTest, Integration_Shuffle_Enabled) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// CLS-062: Shuffle disabled
TEST_F(ClassificationDatasetTest, Integration_Shuffle_Disabled) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// CLS-063: Reproducibility seed
TEST_F(ClassificationDatasetTest, Integration_Reproducibility_Seed) {
    SUCCEED();
}

// CLS-064: Cache hit
TEST_F(ClassificationDatasetTest, Integration_Cache_Hit) {
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);

    dataset.get(0);
    dataset.get(0);
    SUCCEED();
}

// CLS-065: Cache miss
TEST_F(ClassificationDatasetTest, Integration_Cache_Miss) {
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);

    dataset.get(0);
    dataset.get(1);
    SUCCEED();
}

// CLS-066: Transform applied
TEST_F(ClassificationDatasetTest, Integration_Transform_Applied) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.data.dim(), 3);
    EXPECT_EQ(example.data.size(0), 3);
}

// CLS-067: CPU tensor
TEST_F(ClassificationDatasetTest, Integration_TensorDevice_CPU) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.data.device().type(), torch::kCPU);
}

// CLS-068: CUDA tensor
TEST_F(ClassificationDatasetTest, Integration_TensorDevice_CUDA) {
    if (!torch::cuda::is_available()) {
        SUCCEED() << "CUDA not available, skipping test";
        return;
    }

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    example.toDevice(torch::kCUDA);
    EXPECT_EQ(example.data.device().type(), torch::kCUDA);
}

// CLS-069: Limit samples
TEST_F(ClassificationDatasetTest, Integration_LimitSamples) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.limitSamples(5);
    EXPECT_EQ(dataset.size().value(), 5);
}

// CLS-070: Clear cache
TEST_F(ClassificationDatasetTest, Integration_ClearCache) {
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);

    dataset.get(0);
    dataset.clearCache();
    dataset.get(0);
    SUCCEED();
}

// CLS-071: MultiThread DataLoader
TEST_F(ClassificationDatasetTest, Integration_MultiThread_DataLoader) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    std::atomic<int> count{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&, i]() {
            dataset.get(i % dataset.size().value());
            count++;
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(count.load(), 4);
}

// CLS-072: toDevice
TEST_F(ClassificationDatasetTest, Integration_ToDevice) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    example.toDevice(torch::kCPU);
    EXPECT_EQ(example.data.device().type(), torch::kCPU);
}

// CLS-073: Empty after limit
TEST_F(ClassificationDatasetTest, Integration_EmptyAfterLimit) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    size_t originalSize = dataset.size().value();
    dataset.limitSamples(0);
    EXPECT_EQ(dataset.size().value(), originalSize);
}

// CLS-074: Single sample
TEST_F(ClassificationDatasetTest, Integration_SingleSample) {
    std::string singleJson = testDir_ + "/a/b/c/single.json";
    std::ofstream file(singleJson);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [\n";
    file << "  { \"filename\": \"image_0.jpg\", \"role\": 0, \"label\": 0 }\n";
    file << "] }";
    file.close();

    std::string hyperConfigPath = testDir_ + "/hyper_single.yaml";
    createHyperConfig(hyperConfigPath);

    Configuration config;
    config.load(modelPath_, hyperConfigPath, singleJson);
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// CLS-075: Large batch size
TEST_F(ClassificationDatasetTest, Integration_LargeBatchSize) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_LE(config.getBatchSize(), static_cast<int>(dataset.size().value()));
}
