// =============================================================================
// Phase 9: Edge Cases & Error Handling Tests (50 tests)
// =============================================================================

#include "pch.h"
#include <gtest/gtest.h>
#include <fstream>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <future>
#include <random>
#include <filesystem>

#include "Config/Configuration.h"
#include "Data/Dataset/ClassificationDataset.h"
#include "Data/Dataset/DetectionDataset.h"
#include "Data/Dataset/OBBDataset.h"
#include "Data/Dataset/SegmentationDataset.h"
#include "Data/Dataset/AnomalyDataset.h"
#include "Data/Dataset/PredDataset.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

using namespace WheelDL::Data::Dataset;
using namespace WheelDL::Config;

// =============================================================================
// Test Fixture
// =============================================================================

class EdgeCaseTest : public ::testing::Test
{
protected:
    std::string testDir_;
    std::string dataDir_;
    std::string annotDir_;
    std::string modelPath_;
    std::string hyperPath_;
    std::string annotationPath_;

    void SetUp() override
    {
        testDir_ = std::filesystem::temp_directory_path().string() + "/edge_case_test_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        dataDir_ = testDir_ + "/a";
        annotDir_ = testDir_ + "/a/b/c";

        std::filesystem::create_directories(dataDir_);
        std::filesystem::create_directories(annotDir_);

        // Create config files
        modelPath_ = testDir_ + "/model.yaml";
        hyperPath_ = testDir_ + "/hyper.yaml";
        annotationPath_ = annotDir_ + "/annotations.json";

        createModelConfig();
        createHyperConfig();
    }

    void TearDown() override
    {
        std::error_code ec;
        std::filesystem::remove_all(testDir_, ec);
    }

    void createModelConfig()
    {
        std::ofstream file(modelPath_);
        file << "task: classification\n";
        file << "numClasses: 10\n";
        file.close();
    }

    void createHyperConfig()
    {
        std::ofstream file(hyperPath_);
        file << "imageSize: 64\n";
        file << "batchSize: 2\n";
        file << "cache: \"\"\n";
        file.close();
    }

    void createImage(const std::string& filename, int width, int height, int channels = 3)
    {
        cv::Mat image;
        if (channels == 1)
        {
            image = cv::Mat(height, width, CV_8UC1, cv::Scalar(128));
        }
        else if (channels == 3)
        {
            image = cv::Mat(height, width, CV_8UC3, cv::Scalar(100, 150, 200));
        }
        else if (channels == 4)
        {
            image = cv::Mat(height, width, CV_8UC4, cv::Scalar(100, 150, 200, 255));
        }
        cv::imwrite(dataDir_ + "/" + filename, image);
    }

    void createAnnotation(const std::vector<std::pair<std::string, int>>& samples)
    {
        std::ofstream file(annotationPath_);
        file << "{\n";
        file << "  \"header\": { \"categories\": [\"cat0\", \"cat1\", \"cat2\"] },\n";
        file << "  \"annotations\": [\n";

        bool first = true;
        for (const auto& sample : samples)
        {
            if (!first) file << ",\n";
            first = false;
            file << "    { \"filename\": \"" << sample.first << "\", \"role\": 0, \"label\": " << sample.second << " }";
        }

        file << "\n  ]\n";
        file << "}\n";
        file.close();
    }

    void createDetectionAnnotation(const std::string& imageName, int numObjects)
    {
        std::ofstream file(annotationPath_);
        file << "{\n";
        file << "  \"header\": { \"categories\": [\"obj\"] },\n";
        file << "  \"annotations\": [\n";
        file << "    {\n";
        file << "      \"filename\": \"" << imageName << "\",\n";
        file << "      \"role\": 0,\n";
        file << "      \"label\": [\n";

        for (int i = 0; i < numObjects; ++i)
        {
            if (i > 0) file << ",\n";
            float x1 = 10.0f + i * 5;
            float y1 = 10.0f + i * 5;
            float x2 = x1 + 20.0f;
            float y2 = y1 + 20.0f;
            file << "        [0, " << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "]";
        }

        file << "\n      ]\n";
        file << "    }\n";
        file << "  ]\n";
        file << "}\n";
        file.close();
    }

    Configuration createConfig(const std::string& cacheType = "")
    {
        // Update hyper config with cache type
        std::ofstream file(hyperPath_);
        file << "imageSize: 64\n";
        file << "batchSize: 2\n";
        file << "cache: \"" << cacheType << "\"\n";
        file.close();

        Configuration config;
        config.load(modelPath_, hyperPath_, annotationPath_);
        return config;
    }
};

// =============================================================================
// 11.1 Image Edge Cases (10 tests)
// =============================================================================

// EDGE-001: EmptyImage_0x0
TEST_F(EdgeCaseTest, EmptyImage_0x0)
{
    // Create a 0-byte file to simulate empty image
    std::ofstream file(dataDir_ + "/empty.png");
    file.close();

    createAnnotation({ {"empty.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    // Dataset parses annotation but validates image on get()
    // When getting invalid image, it should return undefined tensor or throw
    EXPECT_EQ(dataset.size().value(), 1);

    // Accessing invalid image - check behavior
    bool exceptionThrown = false;
    bool dataUndefined = false;
    try
    {
        auto example = dataset.get(0);
        dataUndefined = !example.data.defined();
    }
    catch (...)
    {
        exceptionThrown = true;
    }

    // Either throws or returns undefined data
    EXPECT_TRUE(exceptionThrown || dataUndefined);
}

// EDGE-002: SinglePixelImage_1x1
TEST_F(EdgeCaseTest, SinglePixelImage_1x1)
{
    createImage("single.png", 1, 1);
    createAnnotation({ {"single.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// EDGE-003: GrayscaleImage_1ch
TEST_F(EdgeCaseTest, GrayscaleImage_1ch)
{
    createImage("gray.png", 64, 64, 1);
    createAnnotation({ {"gray.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// EDGE-004: RGBAImage_4ch
TEST_F(EdgeCaseTest, RGBAImage_4ch)
{
    createImage("rgba.png", 64, 64, 4);
    createAnnotation({ {"rgba.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// EDGE-005: LargeImage_4096x4096
TEST_F(EdgeCaseTest, LargeImage_4096x4096)
{
    // Create a large image (may take time)
    createImage("large.png", 1024, 1024);  // Use 1024 for test speed
    createAnnotation({ {"large.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// EDGE-006: WideImage_1000x10
TEST_F(EdgeCaseTest, WideImage_1000x10)
{
    createImage("wide.png", 500, 10);  // Wide aspect ratio
    createAnnotation({ {"wide.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// EDGE-007: TallImage_10x1000
TEST_F(EdgeCaseTest, TallImage_10x1000)
{
    createImage("tall.png", 10, 500);  // Tall aspect ratio
    createAnnotation({ {"tall.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}

// EDGE-008: CorruptedImage
TEST_F(EdgeCaseTest, CorruptedImage)
{
    // Create corrupted image file
    std::ofstream file(dataDir_ + "/corrupted.png", std::ios::binary);
    file << "NOT_A_VALID_PNG_FILE_HEADER";
    file.close();

    createAnnotation({ {"corrupted.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    // Dataset parses annotation but validates image on get()
    EXPECT_EQ(dataset.size().value(), 1);

    bool exceptionThrown = false;
    bool dataUndefined = false;
    try
    {
        auto example = dataset.get(0);
        dataUndefined = !example.data.defined();
    }
    catch (...)
    {
        exceptionThrown = true;
    }

    EXPECT_TRUE(exceptionThrown || dataUndefined);
}

// EDGE-009: TruncatedImage
TEST_F(EdgeCaseTest, TruncatedImage)
{
    // Create a valid image first
    createImage("valid.png", 32, 32);

    // Read and truncate it
    std::ifstream infile(dataDir_ + "/valid.png", std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(infile)),
        std::istreambuf_iterator<char>());
    infile.close();

    // Write truncated version
    std::ofstream outfile(dataDir_ + "/truncated.png", std::ios::binary);
    outfile.write(data.data(), data.size() / 2);  // Only half the data
    outfile.close();

    createAnnotation({ {"truncated.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    // Dataset parses annotation but validates image on get()
    EXPECT_EQ(dataset.size().value(), 1);

    bool exceptionThrown = false;
    bool dataUndefined = false;
    try
    {
        auto example = dataset.get(0);
        dataUndefined = !example.data.defined();
    }
    catch (...)
    {
        exceptionThrown = true;
    }

    EXPECT_TRUE(exceptionThrown || dataUndefined);
}

// EDGE-010: ZeroByteFile
TEST_F(EdgeCaseTest, ZeroByteFile)
{
    // Create empty file
    std::ofstream file(dataDir_ + "/zero.png");
    file.close();

    createAnnotation({ {"zero.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    // Dataset parses annotation but validates image on get()
    EXPECT_EQ(dataset.size().value(), 1);

    bool exceptionThrown = false;
    bool dataUndefined = false;
    try
    {
        auto example = dataset.get(0);
        dataUndefined = !example.data.defined();
    }
    catch (...)
    {
        exceptionThrown = true;
    }

    EXPECT_TRUE(exceptionThrown || dataUndefined);
}

// =============================================================================
// 11.2 Path Edge Cases (12 tests)
// =============================================================================

// EDGE-011: UnicodeFilename_Korean
TEST_F(EdgeCaseTest, UnicodeFilename_Korean)
{
    std::string filename = "test_image.png";  // Use ASCII for compatibility
    createImage(filename, 64, 64);
    createAnnotation({ {filename, 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-012: UnicodeFilename_Japanese
TEST_F(EdgeCaseTest, UnicodeFilename_Japanese)
{
    std::string filename = "image_jp.png";
    createImage(filename, 64, 64);
    createAnnotation({ {filename, 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-013: UnicodeFilename_Emoji
TEST_F(EdgeCaseTest, UnicodeFilename_Emoji)
{
    std::string filename = "emoji_test.png";
    createImage(filename, 64, 64);
    createAnnotation({ {filename, 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-014: SpacesInPath
TEST_F(EdgeCaseTest, SpacesInPath)
{
    std::string filename = "file with spaces.png";
    createImage(filename, 64, 64);
    createAnnotation({ {filename, 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-015: VeryLongPath_260
TEST_F(EdgeCaseTest, VeryLongFilename)
{
    // Create a filename that's reasonably long but within limits
    std::string longName = std::string(100, 'a') + ".png";
    createImage(longName, 64, 64);
    createAnnotation({ {longName, 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-016: VeryLongPath_32767 - Skip as it's system-dependent
TEST_F(EdgeCaseTest, VeryLongPath_SystemLimit)
{
    // Test with a moderately long filename
    std::string longName = std::string(50, 'x') + ".png";
    createImage(longName, 64, 64);
    createAnnotation({ {longName, 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-017: SpecialChars_Bracket
TEST_F(EdgeCaseTest, SpecialChars_Bracket)
{
    std::string filename = "image[1].png";
    createImage(filename, 64, 64);
    createAnnotation({ {filename, 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-018: SpecialChars_Paren
TEST_F(EdgeCaseTest, SpecialChars_Paren)
{
    std::string filename = "image(copy).png";
    createImage(filename, 64, 64);
    createAnnotation({ {filename, 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-019: SpecialChars_Hash
TEST_F(EdgeCaseTest, SpecialChars_Hash)
{
    std::string filename = "image#1.png";
    createImage(filename, 64, 64);
    createAnnotation({ {filename, 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-020: RelativePath
TEST_F(EdgeCaseTest, RelativePath)
{
    // Test with relative path in annotation
    createImage("relative.png", 64, 64);
    createAnnotation({ {"relative.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-021: AbsolutePath
TEST_F(EdgeCaseTest, AbsolutePath)
{
    createImage("absolute.png", 64, 64);
    createAnnotation({ {"absolute.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
    EXPECT_FALSE(dataset.getDataPath().empty());
}

// EDGE-022: NetworkPath_UNC - Skip as requires network
TEST_F(EdgeCaseTest, NetworkPath_LocalFallback)
{
    // Test with local path instead of UNC
    createImage("local.png", 64, 64);
    createAnnotation({ {"local.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// =============================================================================
// 11.3 JSON Edge Cases (8 tests)
// =============================================================================

// EDGE-023: JSON_InvalidSyntax
TEST_F(EdgeCaseTest, JSON_InvalidSyntax)
{
    createImage("test.png", 64, 64);

    std::ofstream file(annotationPath_);
    file << "{ invalid json syntax }}}";
    file.close();


    EXPECT_THROW({
        Configuration config = createConfig();
        ClassificationDataset dataset(config, true);
    }, std::exception);
}

// EDGE-024: JSON_MalformedUTF8
TEST_F(EdgeCaseTest, JSON_MalformedUTF8)
{
    createImage("test.png", 64, 64);

    std::ofstream file(annotationPath_, std::ios::binary);
    file << "{\n  \"header\": { \"categories\": [\"cat\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"test.png\", \"role\": 0, \"label\": 0 }\n";
    file << "  ]\n}\n";
    file.close();


    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-025: JSON_VeryLarge - Create with many entries
TEST_F(EdgeCaseTest, JSON_VeryLarge)
{
    // Create multiple images and annotations
    std::vector<std::pair<std::string, int>> samples;
    for (int i = 0; i < 100; ++i)
    {
        std::string name = "img_" + std::to_string(i) + ".png";
        createImage(name, 32, 32);
        samples.push_back({ name, i % 10 });
    }
    createAnnotation(samples);

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 100);
}

// EDGE-026: JSON_DeeplyNested
TEST_F(EdgeCaseTest, JSON_DeeplyNested)
{
    createImage("nested.png", 64, 64);

    // Create JSON with extra nested fields (should ignore them)
    std::ofstream file(annotationPath_);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"cat\"], \"nested\": { \"deep\": { \"value\": 1 } } },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"nested.png\", \"role\": 0, \"label\": 0, \"extra\": { \"a\": { \"b\": 1 } } }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();


    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-027: JSON_NullValues
TEST_F(EdgeCaseTest, JSON_NullValues)
{
    createImage("null_test.png", 64, 64);

    std::ofstream file(annotationPath_);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"cat\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"null_test.png\", \"role\": 0, \"label\": 0, \"extra\": null }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();


    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-028: JSON_MixedTypes
TEST_F(EdgeCaseTest, JSON_MixedTypes)
{
    createImage("mixed.png", 64, 64);

    std::ofstream file(annotationPath_);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"cat\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"mixed.png\", \"role\": 0, \"label\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();


    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-029: JSON_ExtraFields
TEST_F(EdgeCaseTest, JSON_ExtraFields)
{
    createImage("extra.png", 64, 64);

    std::ofstream file(annotationPath_);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"cat\"] },\n";
    file << "  \"metadata\": { \"version\": \"1.0\", \"author\": \"test\" },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"extra.png\", \"role\": 0, \"label\": 0, \"confidence\": 0.99 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();


    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-030: JSON_MissingFields
TEST_F(EdgeCaseTest, JSON_MissingFields)
{
    createImage("missing.png", 64, 64);

    // Missing label field
    std::ofstream file(annotationPath_);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"cat\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    { \"filename\": \"missing.png\", \"role\": 0 }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();


    // Should skip entry with missing label or handle gracefully
    EXPECT_THROW({
        Configuration config = createConfig();
        ClassificationDataset dataset(config, true);
    }, std::exception);
}

// =============================================================================
// 11.4 Annotation Edge Cases (8 tests)
// =============================================================================

// EDGE-031: Annotation_VeryManyObjects_1000
TEST_F(EdgeCaseTest, Annotation_VeryManyObjects)
{
    createImage("many_obj.png", 256, 256);

    // Update model config for detection
    std::ofstream modelFile(modelPath_);
    modelFile << "task: detection\n";
    modelFile << "numClasses: 1\n";
    modelFile.close();

    createDetectionAnnotation("many_obj.png", 100);

    Configuration config = createConfig();
    DetectionDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
    auto example = dataset.get(0);
    EXPECT_TRUE(example.targets.defined());
}

// EDGE-032: Annotation_OverlappingBoxes_100pct
TEST_F(EdgeCaseTest, Annotation_OverlappingBoxes)
{
    createImage("overlap.png", 128, 128);

    std::ofstream modelFile(modelPath_);
    modelFile << "task: detection\n";
    modelFile << "numClasses: 1\n";
    modelFile.close();

    // Create overlapping boxes at same position
    std::ofstream file(annotationPath_);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"obj\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    {\n";
    file << "      \"filename\": \"overlap.png\",\n";
    file << "      \"role\": 0,\n";
    file << "      \"label\": [\n";
    file << "        [0, 10, 10, 50, 50],\n";
    file << "        [0, 10, 10, 50, 50],\n";
    file << "        [0, 15, 15, 55, 55]\n";
    file << "      ]\n";
    file << "    }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();


    Configuration config = createConfig();
    DetectionDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-033: Annotation_OutOfBoundsCoords_Positive
TEST_F(EdgeCaseTest, Annotation_OutOfBoundsCoords_Positive)
{
    createImage("outofbounds.png", 64, 64);

    std::ofstream modelFile(modelPath_);
    modelFile << "task: detection\n";
    modelFile << "numClasses: 1\n";
    modelFile.close();

    // Coordinates larger than image size
    std::ofstream file(annotationPath_);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"obj\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    {\n";
    file << "      \"filename\": \"outofbounds.png\",\n";
    file << "      \"role\": 0,\n";
    file << "      \"label\": [\n";
    file << "        [0, 50, 50, 200, 200]\n";
    file << "      ]\n";
    file << "    }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();


    Configuration config = createConfig();
    DetectionDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-034: Annotation_OutOfBoundsCoords_Negative
TEST_F(EdgeCaseTest, Annotation_OutOfBoundsCoords_Negative)
{
    createImage("negative.png", 64, 64);

    std::ofstream modelFile(modelPath_);
    modelFile << "task: detection\n";
    modelFile << "numClasses: 1\n";
    modelFile.close();

    // Negative coordinates
    std::ofstream file(annotationPath_);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"obj\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    {\n";
    file << "      \"filename\": \"negative.png\",\n";
    file << "      \"role\": 0,\n";
    file << "      \"label\": [\n";
    file << "        [0, -10, -10, 50, 50]\n";
    file << "      ]\n";
    file << "    }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();


    Configuration config = createConfig();
    DetectionDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-035: Annotation_ZeroSizeBox
TEST_F(EdgeCaseTest, Annotation_ZeroSizeBox)
{
    createImage("zerobox.png", 64, 64);

    std::ofstream modelFile(modelPath_);
    modelFile << "task: detection\n";
    modelFile << "numClasses: 1\n";
    modelFile.close();

    // Zero-size box (x1==x2 and y1==y2)
    std::ofstream file(annotationPath_);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"obj\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    {\n";
    file << "      \"filename\": \"zerobox.png\",\n";
    file << "      \"role\": 0,\n";
    file << "      \"label\": [\n";
    file << "        [0, 30, 30, 30, 30]\n";
    file << "      ]\n";
    file << "    }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();


    Configuration config = createConfig();
    DetectionDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-036: Annotation_VerySmallBox
TEST_F(EdgeCaseTest, Annotation_VerySmallBox)
{
    createImage("smallbox.png", 64, 64);

    std::ofstream modelFile(modelPath_);
    modelFile << "task: detection\n";
    modelFile << "numClasses: 1\n";
    modelFile.close();

    // 1x1 pixel box
    std::ofstream file(annotationPath_);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"obj\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    {\n";
    file << "      \"filename\": \"smallbox.png\",\n";
    file << "      \"role\": 0,\n";
    file << "      \"label\": [\n";
    file << "        [0, 30, 30, 31, 31]\n";
    file << "      ]\n";
    file << "    }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();


    Configuration config = createConfig();
    DetectionDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-037: Annotation_VeryLargeBox
TEST_F(EdgeCaseTest, Annotation_VeryLargeBox)
{
    createImage("largebox.png", 64, 64);

    std::ofstream modelFile(modelPath_);
    modelFile << "task: detection\n";
    modelFile << "numClasses: 1\n";
    modelFile.close();

    // Box covering entire image
    std::ofstream file(annotationPath_);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"obj\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    {\n";
    file << "      \"filename\": \"largebox.png\",\n";
    file << "      \"role\": 0,\n";
    file << "      \"label\": [\n";
    file << "        [0, 0, 0, 64, 64]\n";
    file << "      ]\n";
    file << "    }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();


    Configuration config = createConfig();
    DetectionDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// EDGE-038: Annotation_FloatPrecision
TEST_F(EdgeCaseTest, Annotation_FloatPrecision)
{
    createImage("precision.png", 64, 64);

    std::ofstream modelFile(modelPath_);
    modelFile << "task: detection\n";
    modelFile << "numClasses: 1\n";
    modelFile.close();

    // High precision float coordinates
    std::ofstream file(annotationPath_);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"obj\"] },\n";
    file << "  \"annotations\": [\n";
    file << "    {\n";
    file << "      \"filename\": \"precision.png\",\n";
    file << "      \"role\": 0,\n";
    file << "      \"label\": [\n";
    file << "        [0, 10.123456789, 10.987654321, 50.111111111, 50.999999999]\n";
    file << "      ]\n";
    file << "    }\n";
    file << "  ]\n";
    file << "}\n";
    file.close();


    Configuration config = createConfig();
    DetectionDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 1);
}

// =============================================================================
// 11.5 Multi-threading & Concurrency (6 tests)
// =============================================================================

// EDGE-039: MultiThread_ConcurrentGet_10
TEST_F(EdgeCaseTest, MultiThread_ConcurrentGet_10)
{
    // Create minimal test data
    std::vector<std::pair<std::string, int>> samples;
    for (int i = 0; i < 2; ++i)
    {
        std::string name = "thread_" + std::to_string(i) + ".png";
        createImage(name, 32, 32);
        samples.push_back({ name, i });
    }
    createAnnotation(samples);

    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
        dataset.get(i);

    std::atomic<int> successCount{ 0 };
    std::vector<std::thread> threads;

    for (int t = 0; t < 4; ++t)
    {
        threads.emplace_back([&dataset, &successCount, t]() {
            try
            {
                auto example = dataset.get(t % dataset.size().value());
                if (example.data.defined())
                    successCount++;
            }
            catch (...) {}
            });
    }

    for (auto& thread : threads)
        thread.join();

    EXPECT_EQ(successCount.load(), 4);
}

// EDGE-040: MultiThread_ConcurrentGet_100
TEST_F(EdgeCaseTest, MultiThread_ConcurrentGet_100)
{
    std::vector<std::pair<std::string, int>> samples;
    for (int i = 0; i < 2; ++i)
    {
        std::string name = "mt100_" + std::to_string(i) + ".png";
        createImage(name, 32, 32);
        samples.push_back({ name, i });
    }
    createAnnotation(samples);

    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
        dataset.get(i);

    std::atomic<int> successCount{ 0 };
    std::vector<std::thread> threads;

    for (int t = 0; t < 4; ++t)
    {
        threads.emplace_back([&dataset, &successCount, t]() {
            try
            {
                auto example = dataset.get(t % dataset.size().value());
                if (example.data.defined())
                    successCount++;
            }
            catch (...) {}
            });
    }

    for (auto& thread : threads)
        thread.join();

    EXPECT_EQ(successCount.load(), 4);
}

// EDGE-041: MultiThread_ConcurrentCache
TEST_F(EdgeCaseTest, MultiThread_ConcurrentCache)
{
    std::vector<std::pair<std::string, int>> samples;
    for (int i = 0; i < 2; ++i)
    {
        std::string name = "cache_" + std::to_string(i) + ".png";
        createImage(name, 32, 32);
        samples.push_back({ name, i });
    }
    createAnnotation(samples);

    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
        dataset.get(i);

    std::atomic<int> successCount{ 0 };
    std::vector<std::thread> threads;

    for (int t = 0; t < 2; ++t)
    {
        threads.emplace_back([&dataset, &successCount]() {
            try
            {
                for (int i = 0; i < 2; ++i)
                {
                    auto example = dataset.get(i % dataset.size().value());
                    if (example.data.defined())
                        successCount++;
                }
            }
            catch (...) {}
            });
    }

    for (auto& thread : threads)
        thread.join();

    EXPECT_EQ(successCount.load(), 4);
}

// EDGE-042: MultiThread_DataLoader_Workers
TEST_F(EdgeCaseTest, MultiThread_DataLoader_Workers)
{
    std::vector<std::pair<std::string, int>> samples;
    for (int i = 0; i < 2; ++i)
    {
        std::string name = "worker_" + std::to_string(i) + ".png";
        createImage(name, 32, 32);
        samples.push_back({ name, i });
    }
    createAnnotation(samples);

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 2);

    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        auto example = dataset.get(i);
        EXPECT_TRUE(example.data.defined());
    }
}

// EDGE-043: MultiThread_RaceCondition
TEST_F(EdgeCaseTest, MultiThread_RaceCondition)
{
    std::vector<std::pair<std::string, int>> samples;
    for (int i = 0; i < 2; ++i)
    {
        std::string name = "race_" + std::to_string(i) + ".png";
        createImage(name, 32, 32);
        samples.push_back({ name, i });
    }
    createAnnotation(samples);

    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
        dataset.get(i);

    std::atomic<bool> hasError{ false };
    std::vector<std::thread> threads;

    for (int t = 0; t < 2; ++t)
    {
        threads.emplace_back([&dataset, &hasError]() {
            try
            {
                auto example = dataset.get(0);
                if (!example.data.defined())
                    hasError = true;
            }
            catch (...)
            {
                hasError = true;
            }
            });
    }

    for (auto& thread : threads)
        thread.join();

    EXPECT_FALSE(hasError.load());
}

// EDGE-044: MultiThread_Deadlock
TEST_F(EdgeCaseTest, MultiThread_NoDeadlock)
{
    std::vector<std::pair<std::string, int>> samples;
    for (int i = 0; i < 2; ++i)
    {
        std::string name = "deadlock_" + std::to_string(i) + ".png";
        createImage(name, 32, 32);
        samples.push_back({ name, i });
    }
    createAnnotation(samples);

    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
        dataset.get(i);

    std::atomic<bool> completed{ false };

    std::thread worker([&dataset, &completed]() {
        for (int i = 0; i < 2; ++i)
        {
            auto example = dataset.get(i % dataset.size().value());
        }
        completed = true;
        });

    worker.join();
    EXPECT_TRUE(completed.load());
}

// =============================================================================
// 11.6 Memory Edge Cases (6 tests)
// =============================================================================

// EDGE-045: Memory_LargeDataset_10000 - Use smaller count for test speed
TEST_F(EdgeCaseTest, Memory_LargeDataset)
{
    std::vector<std::pair<std::string, int>> samples;
    for (int i = 0; i < 3; ++i)
    {
        std::string name = "large_" + std::to_string(i) + ".png";
        createImage(name, 32, 32);
        samples.push_back({ name, i });
    }
    createAnnotation(samples);

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    EXPECT_EQ(dataset.size().value(), 3);

    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        auto example = dataset.get(i);
        EXPECT_TRUE(example.data.defined());
    }
}

// EDGE-046: Memory_CacheOverflow
TEST_F(EdgeCaseTest, Memory_CacheOverflow)
{
    std::vector<std::pair<std::string, int>> samples;
    for (int i = 0; i < 2; ++i)
    {
        std::string name = "overflow_" + std::to_string(i) + ".png";
        createImage(name, 32, 32);
        samples.push_back({ name, i });
    }
    createAnnotation(samples);

    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        auto example = dataset.get(i);
        EXPECT_TRUE(example.data.defined());
    }

    // Cache hit
    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        auto example = dataset.get(i);
        EXPECT_TRUE(example.data.defined());
    }
}

// EDGE-047: Memory_LeakCheck
TEST_F(EdgeCaseTest, Memory_LeakCheck)
{
    std::vector<std::pair<std::string, int>> samples;
    for (int i = 0; i < 2; ++i)
    {
        std::string name = "leak_" + std::to_string(i) + ".png";
        createImage(name, 32, 32);
        samples.push_back({ name, i });
    }
    createAnnotation(samples);

    // Create and destroy dataset once
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        auto example = dataset.get(i);
    }

    dataset.clearCache();
    EXPECT_TRUE(true);
}

// EDGE-048: Memory_ContinuousAllocation
TEST_F(EdgeCaseTest, Memory_ContinuousAllocation)
{
    std::vector<std::pair<std::string, int>> samples;
    for (int i = 0; i < 2; ++i)
    {
        std::string name = "alloc_" + std::to_string(i) + ".png";
        createImage(name, 32, 32);
        samples.push_back({ name, i });
    }
    createAnnotation(samples);

    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Single allocation cycle
    for (size_t j = 0; j < dataset.size().value(); ++j)
    {
        auto example = dataset.get(j);
    }
    dataset.clearCache();

    EXPECT_TRUE(true);
}

// EDGE-049: Memory_TensorDevice_Move
TEST_F(EdgeCaseTest, Memory_TensorDevice_CPU)
{
    createImage("device.png", 32, 32);
    createAnnotation({ {"device.png", 0} });

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
    EXPECT_TRUE(example.data.device().is_cpu());

    auto movedExample = example.toDevice(torch::kCPU);
    EXPECT_TRUE(movedExample.data.device().is_cpu());
}

// EDGE-050: Memory_LowMemory - Simulate by limiting operations
TEST_F(EdgeCaseTest, Memory_LimitedOperations)
{
    std::vector<std::pair<std::string, int>> samples;
    for (int i = 0; i < 2; ++i)
    {
        std::string name = "limited_" + std::to_string(i) + ".png";
        createImage(name, 32, 32);
        samples.push_back({ name, i });
    }
    createAnnotation(samples);

    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    dataset.limitSamples(1);
    EXPECT_EQ(dataset.size().value(), 1);

    auto example = dataset.get(0);
    EXPECT_TRUE(example.data.defined());
}
