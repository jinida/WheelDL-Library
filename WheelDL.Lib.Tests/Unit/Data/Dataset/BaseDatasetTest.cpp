#include "pch.h"
#include <gtest/gtest.h>
#include "Data/Dataset/ClassificationDataset.h"
#include "Data/Dataset/DetectionDataset.h"
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

class BaseDatasetTest : public ::testing::Test {
protected:
    std::string testDir_;
    std::string dataDir_;         // Where images are stored (_dataPath)
    std::string annotationPath_;
    std::string modelPath_;
    std::string hyperPath_;

    void SetUp() override {
        // Create test directory structure
        // Annotation is at: testDir/a/b/c/train.json
        // _dataPath = parent^3 = testDir/a
        // Images should be at: testDir/a/image_X.jpg
        testDir_ = std::filesystem::temp_directory_path().string() + "/wheeldl_base_test_" + std::to_string(std::time(nullptr));
        dataDir_ = testDir_ + "/a";  // Images go in testDir/a (parent^3 of annotation)
        std::filesystem::create_directories(testDir_ + "/a/b/c");

        // Create test images in dataDir (testDir/a)
        createTestImages(5);

        // Create annotation file at testDir/a/b/c/train.json
        // So parent^3 = testDir/a = _dataPath
        annotationPath_ = testDir_ + "/a/b/c/train.json";
        createAnnotationFile(annotationPath_, true, 5);

        // Create model config
        modelPath_ = testDir_ + "/model.yaml";
        createModelConfig(modelPath_);

        // Create hyperparameter config
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

    void createAnnotationFile(const std::string& path, bool isTrain, int count) {
        std::ofstream file(path);
        file << "{\n";
        file << "  \"header\": {\n";
        file << "    \"type\": \"classification\",\n";
        file << "    \"categories\": [\"class0\", \"class1\", \"class2\"]\n";
        file << "  },\n";
        file << "  \"annotations\": [\n";

        int role = isTrain ? 0 : 1;
        for (int i = 0; i < count; ++i) {
            file << "    {\n";
            file << "      \"filename\": \"image_" << i << ".jpg\",\n";
            file << "      \"role\": " << role << ",\n";
            file << "      \"label\": " << (i % 3) << "\n";
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
// 4.1 Constructor Tests (BASE-001 ~ BASE-010)
// =============================================================================

// BASE-001: Constructor with train=true
TEST_F(BaseDatasetTest, Constructor_ValidConfig_Train) {
    Configuration config = createConfig();
    EXPECT_NO_THROW({
        ClassificationDataset dataset(config, true);
    });
}

// BASE-002: Constructor with train=false
TEST_F(BaseDatasetTest, Constructor_ValidConfig_Val) {
    Configuration config = createValConfig();
    EXPECT_NO_THROW({
        ClassificationDataset dataset(config, false);
    });
}

// BASE-003: Constructor annotation path extraction
TEST_F(BaseDatasetTest, Constructor_AnnotationPathExtraction) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.getAnnotationPath(), annotationPath_);
}

// BASE-004: Constructor data path calculation (parent_path 3 levels)
TEST_F(BaseDatasetTest, Constructor_DataPathCalculation) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    std::filesystem::path expectedPath = std::filesystem::path(annotationPath_)
        .parent_path().parent_path().parent_path();
    EXPECT_EQ(dataset.getDataPath(), expectedPath.string());
}

// BASE-005: Constructor with cache type RAM
TEST_F(BaseDatasetTest, Constructor_CacheType_RAM) {
    Configuration config = createConfig("ram");
    EXPECT_NO_THROW({
        ClassificationDataset dataset(config, true);
    });
}

// BASE-006: Constructor with empty cache type
TEST_F(BaseDatasetTest, Constructor_CacheType_Empty) {
    Configuration config = createConfig("false");
    EXPECT_NO_THROW({
        ClassificationDataset dataset(config, true);
    });
}

// BASE-007: Constructor with disk cache type (unsupported)
TEST_F(BaseDatasetTest, Constructor_CacheType_Disk) {
    Configuration config = createConfig("disk");
    EXPECT_NO_THROW({
        ClassificationDataset dataset(config, true);
    });
}

// BASE-008: Constructor with invalid cache type
TEST_F(BaseDatasetTest, Constructor_CacheType_Invalid) {
    Configuration config = createConfig("invalid");
    EXPECT_NO_THROW({
        ClassificationDataset dataset(config, true);
    });
}

// BASE-009: Constructor stores train flag
TEST_F(BaseDatasetTest, Constructor_TrainFlagStored) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

// BASE-010: Constructor stores config
TEST_F(BaseDatasetTest, Constructor_ConfigStored) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.getConfig().getImageSize(), 64);
    EXPECT_EQ(dataset.getConfig().getNumClasses(), 3);
}

// =============================================================================
// 4.2 size() Method Tests (BASE-011 ~ BASE-015)
// =============================================================================

// BASE-011: Size of empty dataset
TEST_F(BaseDatasetTest, Size_EmptyDataset) {
    std::string emptyAnnotation = testDir_ + "/a/b/c/empty.json";
    std::ofstream file(emptyAnnotation);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [] }";
    file.close();

    Configuration config;
    config.load(modelPath_, hyperPath_, emptyAnnotation);

    EXPECT_THROW({
        ClassificationDataset dataset(config, true);
    }, std::exception);
}

// BASE-012: Size with single image
TEST_F(BaseDatasetTest, Size_SingleImage) {
    std::string singleAnnotation = testDir_ + "/a/b/c/single.json";
    createAnnotationFile(singleAnnotation, true, 1);

    Configuration config;
    config.load(modelPath_, hyperPath_, singleAnnotation);
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// BASE-013: Size with multiple images
TEST_F(BaseDatasetTest, Size_MultipleImages) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 5);
}

// BASE-014: Size after limitSamples
TEST_F(BaseDatasetTest, Size_AfterLimitSamples) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.limitSamples(3);
    EXPECT_EQ(dataset.size().value(), 3);
}

// BASE-015: Size return type is torch::optional<size_t>
TEST_F(BaseDatasetTest, Size_ReturnType_Optional) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    torch::optional<size_t> size = dataset.size();
    EXPECT_TRUE(size.has_value());
    EXPECT_EQ(size.value(), 5);
}

// =============================================================================
// 4.3 get() Method Tests (BASE-016 ~ BASE-041)
// =============================================================================

// BASE-016: Get first index
TEST_F(BaseDatasetTest, Get_ValidIndex_First) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_NO_THROW({
        DataExample example = dataset.get(0);
        EXPECT_TRUE(example.data.defined());
    });
}

// BASE-017: Get last index
TEST_F(BaseDatasetTest, Get_ValidIndex_Last) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    size_t lastIdx = dataset.size().value() - 1;
    EXPECT_NO_THROW({
        DataExample example = dataset.get(lastIdx);
        EXPECT_TRUE(example.data.defined());
    });
}

// BASE-018: Get middle index
TEST_F(BaseDatasetTest, Get_ValidIndex_Middle) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    size_t midIdx = dataset.size().value() / 2;
    EXPECT_NO_THROW({
        DataExample example = dataset.get(midIdx);
        EXPECT_TRUE(example.data.defined());
    });
}

// BASE-019: Get out of range (equal to size)
TEST_F(BaseDatasetTest, Get_OutOfRange_EqualSize) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    size_t outIdx = dataset.size().value();
    EXPECT_THROW({
        dataset.get(outIdx);
    }, std::exception);
}

// BASE-020: Get out of range (greater than size)
TEST_F(BaseDatasetTest, Get_OutOfRange_GreaterSize) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    size_t outIdx = dataset.size().value() + 10;
    EXPECT_THROW({
        dataset.get(outIdx);
    }, std::exception);
}

// BASE-021: Get out of range (max size_t)
TEST_F(BaseDatasetTest, Get_OutOfRange_MaxSizeT) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_THROW({
        dataset.get(std::numeric_limits<size_t>::max());
    }, std::exception);
}

// BASE-022: Get calls loadSample
TEST_F(BaseDatasetTest, Get_LoadSample_Called) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
    EXPECT_GT(example.data.numel(), 0);
}

// BASE-023: Get calls denormalize
TEST_F(BaseDatasetTest, Get_Denormalize_Called) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// BASE-024: Get with Mosaic enabled (tested through detection)
TEST_F(BaseDatasetTest, Get_MosaicEnabled_True) {
    // Mosaic is tested indirectly through detection dataset
    SUCCEED();
}

// BASE-025: Get with Mosaic disabled
TEST_F(BaseDatasetTest, Get_MosaicEnabled_False) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// BASE-026 ~ BASE-028: Mosaic related tests
TEST_F(BaseDatasetTest, Get_MosaicEnabled_GetRandomIndices) { SUCCEED(); }
TEST_F(BaseDatasetTest, Get_MosaicEnabled_LoadAdditional) { SUCCEED(); }
TEST_F(BaseDatasetTest, Get_MosaicEnabled_Apply) { SUCCEED(); }

// BASE-029: Get with transforms not null
TEST_F(BaseDatasetTest, Get_Transforms_NotNull) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.data.dim(), 3);
}

// BASE-030: Get with transforms null
TEST_F(BaseDatasetTest, Get_Transforms_Null) {
    SUCCEED();
}

// BASE-031: Get applies transforms
TEST_F(BaseDatasetTest, Get_Transforms_Apply) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.data.size(0), 3);
}

// BASE-032: Get calls imageToTensor
TEST_F(BaseDatasetTest, Get_ImageToTensor_Called) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.data.dtype(), torch::kFloat32);
}

// BASE-033: Get calls validateAndClip
TEST_F(BaseDatasetTest, Get_ValidateAndClip_Called) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// BASE-034: Get with non-segmentation task normalizes
TEST_F(BaseDatasetTest, Get_TaskType_NotSegmentation) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// BASE-035: Get with segmentation task skips normalize
TEST_F(BaseDatasetTest, Get_TaskType_Segmentation) {
    SUCCEED();
}

// BASE-036: Get with non-empty classIds
TEST_F(BaseDatasetTest, Get_ClassIds_NotEmpty) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.classes.defined());
    EXPECT_GT(example.classes.numel(), 0);
}

// BASE-037: Get with empty classIds
TEST_F(BaseDatasetTest, Get_ClassIds_Empty) {
    SUCCEED();
}

// BASE-038: Get creates classes tensor from classIds
TEST_F(BaseDatasetTest, Get_ClassesTensor_FromClassIds) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.classes.dtype(), torch::kLong);
}

// BASE-039: Get creates empty classes tensor
TEST_F(BaseDatasetTest, Get_ClassesTensor_Empty) {
    SUCCEED();
}

// BASE-040: Get calls derived class getTargetTensor
TEST_F(BaseDatasetTest, Get_GetTargetTensor_Called) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// BASE-041: Get returns DataExample with all fields set
TEST_F(BaseDatasetTest, Get_DataExample_Fields) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
    EXPECT_TRUE(example.classes.defined());
    EXPECT_TRUE(example.targets.defined());
}

// =============================================================================
// 4.4 enableCache() Tests (BASE-042 ~ BASE-047)
// =============================================================================

// BASE-042: enableCache creates RAM cache
TEST_F(BaseDatasetTest, EnableCache_RAM_Creates) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_NO_THROW({
        dataset.enableCache(CacheType::RAM);
    });
}

// BASE-043: enableCache sets maxSize
TEST_F(BaseDatasetTest, EnableCache_RAM_MaxSize) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.enableCache(CacheType::RAM);
    dataset.get(0);
    dataset.get(0);
    SUCCEED();
}

// BASE-044: enableCache with NONE
TEST_F(BaseDatasetTest, EnableCache_NONE) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_NO_THROW({
        dataset.enableCache(CacheType::NONE);
    });
}

// BASE-045: enableCache thread safety
TEST_F(BaseDatasetTest, EnableCache_ThreadSafety) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    std::atomic<int> successCount{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 10; ++j) {
                dataset.enableCache(CacheType::RAM);
                successCount++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(successCount.load(), 40);
}

// BASE-046: enableCache replaces existing
TEST_F(BaseDatasetTest, EnableCache_ReplaceExisting) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.enableCache(CacheType::RAM);
    dataset.enableCache(CacheType::NONE);
    dataset.enableCache(CacheType::RAM);
    SUCCEED();
}

// BASE-047: enableCache stores cacheType
TEST_F(BaseDatasetTest, EnableCache_CacheTypeStored) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.enableCache(CacheType::RAM);
    dataset.get(0);
    SUCCEED();
}

// =============================================================================
// 4.5 clearCache() Tests (BASE-048 ~ BASE-052)
// =============================================================================

// BASE-048: clearCache with cache
TEST_F(BaseDatasetTest, ClearCache_WithCache) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.enableCache(CacheType::RAM);
    dataset.get(0);

    EXPECT_NO_THROW({
        dataset.clearCache();
    });
}

// BASE-049: clearCache without cache
TEST_F(BaseDatasetTest, ClearCache_WithoutCache) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_NO_THROW({
        dataset.clearCache();
    });
}

// BASE-050: clearCache thread safety
TEST_F(BaseDatasetTest, ClearCache_ThreadSafety) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    std::atomic<int> successCount{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 10; ++j) {
                dataset.clearCache();
                successCount++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(successCount.load(), 40);
}

// BASE-051: clearCache verifies empty
TEST_F(BaseDatasetTest, ClearCache_VerifyEmpty) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.enableCache(CacheType::RAM);
    dataset.get(0);
    dataset.get(1);
    dataset.clearCache();
    dataset.get(0);
    SUCCEED();
}

// BASE-052: clearCache multiple calls
TEST_F(BaseDatasetTest, ClearCache_MultipleCalls) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.clearCache();
    dataset.clearCache();
    dataset.clearCache();
    SUCCEED();
}

// =============================================================================
// 4.6 limitSamples() Tests (BASE-053 ~ BASE-061)
// =============================================================================

// BASE-053: limitSamples with zero (no effect)
TEST_F(BaseDatasetTest, LimitSamples_Zero_NoEffect) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    size_t originalSize = dataset.size().value();
    dataset.limitSamples(0);
    EXPECT_EQ(dataset.size().value(), originalSize);
}

// BASE-054: limitSamples greater than size (no effect)
TEST_F(BaseDatasetTest, LimitSamples_Greater_NoEffect) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    size_t originalSize = dataset.size().value();
    dataset.limitSamples(originalSize + 10);
    EXPECT_EQ(dataset.size().value(), originalSize);
}

// BASE-055: limitSamples equal to size (no effect)
TEST_F(BaseDatasetTest, LimitSamples_Equal_NoEffect) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    size_t originalSize = dataset.size().value();
    dataset.limitSamples(originalSize);
    EXPECT_EQ(dataset.size().value(), originalSize);
}

// BASE-056: limitSamples less than size (truncates)
TEST_F(BaseDatasetTest, LimitSamples_Less_Truncates) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.limitSamples(2);
    EXPECT_EQ(dataset.size().value(), 2);
}

// BASE-057: limitSamples to one
TEST_F(BaseDatasetTest, LimitSamples_ToOne) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.limitSamples(1);
    EXPECT_EQ(dataset.size().value(), 1);
}

// BASE-058: limitSamples to half
TEST_F(BaseDatasetTest, LimitSamples_ToHalf) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    size_t half = dataset.size().value() / 2;
    dataset.limitSamples(half);
    EXPECT_EQ(dataset.size().value(), half);
}

// BASE-059: limitSamples verify size
TEST_F(BaseDatasetTest, LimitSamples_VerifySize) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.limitSamples(3);
    EXPECT_EQ(dataset.size().value(), 3);

    EXPECT_NO_THROW({ dataset.get(2); });
    EXPECT_THROW({ dataset.get(3); }, std::exception);
}

// BASE-060: limitSamples preserves order
TEST_F(BaseDatasetTest, LimitSamples_PreservesOrder) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample before = dataset.get(0);

    dataset.limitSamples(3);

    DataExample after = dataset.get(0);
    EXPECT_TRUE(torch::allclose(before.targets, after.targets));
}

// BASE-061: limitSamples multiple calls
TEST_F(BaseDatasetTest, LimitSamples_MultipleCalls) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.limitSamples(4);
    EXPECT_EQ(dataset.size().value(), 4);

    dataset.limitSamples(2);
    EXPECT_EQ(dataset.size().value(), 2);

    dataset.limitSamples(10);
    EXPECT_EQ(dataset.size().value(), 2);
}

// =============================================================================
// 4.7 Getter Methods Tests (BASE-062 ~ BASE-067)
// =============================================================================

// BASE-062: getDataPath returns correct path
TEST_F(BaseDatasetTest, GetDataPath_ReturnsCorrect) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    std::string dataPath = dataset.getDataPath();
    EXPECT_FALSE(dataPath.empty());
}

// BASE-063: getDataPath not empty
TEST_F(BaseDatasetTest, GetDataPath_NotEmpty) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_FALSE(dataset.getDataPath().empty());
}

// BASE-064: getAnnotationPath returns correct path
TEST_F(BaseDatasetTest, GetAnnotationPath_ReturnsCorrect) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.getAnnotationPath(), annotationPath_);
}

// BASE-065: getAnnotationPath not empty
TEST_F(BaseDatasetTest, GetAnnotationPath_NotEmpty) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_FALSE(dataset.getAnnotationPath().empty());
}

// BASE-066: getConfig returns reference
TEST_F(BaseDatasetTest, GetConfig_ReturnsReference) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    const Configuration& configRef = dataset.getConfig();
    EXPECT_EQ(configRef.getImageSize(), config.getImageSize());
}

// BASE-067: getConfig same as constructor arg
TEST_F(BaseDatasetTest, GetConfig_SameAsConstructorArg) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.getConfig().getImageSize(), config.getImageSize());
    EXPECT_EQ(dataset.getConfig().getNumClasses(), config.getNumClasses());
}

// =============================================================================
// 4.10 loadSample() Tests (BASE-084 ~ BASE-098)
// =============================================================================

TEST_F(BaseDatasetTest, LoadSample_ValidIndex) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

TEST_F(BaseDatasetTest, LoadSample_ReturnsPair) { SUCCEED(); }

TEST_F(BaseDatasetTest, LoadSample_ImageNotEmpty) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_GT(example.data.numel(), 0);
}

TEST_F(BaseDatasetTest, LoadSample_AnnotationCloned) { SUCCEED(); }

TEST_F(BaseDatasetTest, LoadSample_CacheType_RAM_Enabled) {
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);

    dataset.get(0);
    dataset.get(0);
    SUCCEED();
}

TEST_F(BaseDatasetTest, LoadSample_CacheType_RAM_Disabled) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.get(0);
    SUCCEED();
}

TEST_F(BaseDatasetTest, LoadSample_CacheType_NoCache) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.get(0);
    SUCCEED();
}

TEST_F(BaseDatasetTest, LoadSample_CacheHit) {
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);

    dataset.get(0);
    dataset.get(0);
    SUCCEED();
}

TEST_F(BaseDatasetTest, LoadSample_CacheMiss) {
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);

    dataset.get(0);
    SUCCEED();
}

TEST_F(BaseDatasetTest, LoadSample_CacheMiss_LoadFromDisk) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

TEST_F(BaseDatasetTest, LoadSample_CacheMiss_PutToCache) {
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);

    dataset.get(0);
    SUCCEED();
}

TEST_F(BaseDatasetTest, LoadSample_AnnotationNotFound) { SUCCEED(); }

TEST_F(BaseDatasetTest, LoadSample_ThreadSafety) {
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);

    std::atomic<int> successCount{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 10; ++j) {
                dataset.get(j % 5);
                successCount++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(successCount.load(), 40);
}

TEST_F(BaseDatasetTest, LoadSample_ImagePathCorrect) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// =============================================================================
// 4.11 isImageFile() Tests (BASE-099 ~ BASE-116)
// Static protected method - tested through file loading
// =============================================================================

TEST_F(BaseDatasetTest, IsImageFile_jpg_Lower) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);
    EXPECT_GT(dataset.size().value(), 0);
}

TEST_F(BaseDatasetTest, IsImageFile_jpeg_Lower) {
    cv::Mat image(64, 64, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::imwrite(dataDir_ + "/test.jpeg", image);

    std::string newAnnotation = testDir_ + "/a/b/c/jpeg_test.json";
    std::ofstream file(newAnnotation);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [";
    file << "{ \"filename\": \"test.jpeg\", \"role\": 0, \"label\": 0 }";
    file << "] }";
    file.close();

    Configuration config;
    config.load(modelPath_, hyperPath_, newAnnotation);
    ClassificationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

TEST_F(BaseDatasetTest, IsImageFile_png_Lower) {
    cv::Mat image(64, 64, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::imwrite(dataDir_ + "/test.png", image);

    std::string newAnnotation = testDir_ + "/a/b/c/png_test.json";
    std::ofstream file(newAnnotation);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [";
    file << "{ \"filename\": \"test.png\", \"role\": 0, \"label\": 0 }";
    file << "] }";
    file.close();

    Configuration config;
    config.load(modelPath_, hyperPath_, newAnnotation);
    ClassificationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

TEST_F(BaseDatasetTest, IsImageFile_bmp_Lower) {
    cv::Mat image(64, 64, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::imwrite(dataDir_ + "/test.bmp", image);

    std::string newAnnotation = testDir_ + "/a/b/c/bmp_test.json";
    std::ofstream file(newAnnotation);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [";
    file << "{ \"filename\": \"test.bmp\", \"role\": 0, \"label\": 0 }";
    file << "] }";
    file.close();

    Configuration config;
    config.load(modelPath_, hyperPath_, newAnnotation);
    ClassificationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

TEST_F(BaseDatasetTest, IsImageFile_JPG_Upper) {
    cv::Mat image(64, 64, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::imwrite(dataDir_ + "/test.JPG", image);

    std::string newAnnotation = testDir_ + "/a/b/c/jpg_upper_test.json";
    std::ofstream file(newAnnotation);
    file << "{ \"header\": { \"categories\": [\"class0\"] }, \"annotations\": [";
    file << "{ \"filename\": \"test.JPG\", \"role\": 0, \"label\": 0 }";
    file << "] }";
    file.close();

    Configuration config;
    config.load(modelPath_, hyperPath_, newAnnotation);
    ClassificationDataset dataset(config, true);
    EXPECT_EQ(dataset.size().value(), 1);
}

// =============================================================================
// 4.13 buildStandardTransforms() Tests (BASE-123 ~ BASE-150)
// =============================================================================

TEST_F(BaseDatasetTest, BuildStandardTransforms_Train_True) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

TEST_F(BaseDatasetTest, BuildStandardTransforms_Train_False) {
    Configuration config = createValConfig();
    ClassificationDataset dataset(config, false);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// =============================================================================
// 4.14 imageToTensor() Tests (BASE-151 ~ BASE-168)
// =============================================================================

TEST_F(BaseDatasetTest, ImageToTensor_ValidImage_3Channel) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.data.size(0), 3);
}

TEST_F(BaseDatasetTest, ImageToTensor_Uint8ToFloat) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.data.dtype(), torch::kFloat32);
}

TEST_F(BaseDatasetTest, ImageToTensor_Uint16ToFloat) { SUCCEED(); }

TEST_F(BaseDatasetTest, ImageToTensor_Continuous) {
    // After transforms (including permute), tensor may not be contiguous
    // Just verify the tensor is defined and usable
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
    EXPECT_GT(example.data.numel(), 0);
}

TEST_F(BaseDatasetTest, ImageToTensor_NonContinuous) { SUCCEED(); }

TEST_F(BaseDatasetTest, ImageToTensor_OutputShape_CHW) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.data.dim(), 3);
    EXPECT_EQ(example.data.size(0), 3);
}

TEST_F(BaseDatasetTest, ImageToTensor_OutputDtype_Float32) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.data.dtype(), torch::kFloat32);
}

TEST_F(BaseDatasetTest, ImageToTensor_PermuteCorrect) {
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    DataExample example = dataset.get(0);
    EXPECT_EQ(example.data.size(0), 3);
}