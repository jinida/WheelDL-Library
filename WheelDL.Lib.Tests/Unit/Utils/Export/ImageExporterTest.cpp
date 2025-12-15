#include "pch.h"
#include <gtest/gtest.h>
#include "Utils/Export/ImageExporter.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>

using namespace WheelDL::Utils::Export;
namespace fs = std::filesystem;

// =============================================================================
// Test Fixture
// =============================================================================

class ImageExporterTest : public ::testing::Test {
protected:
    std::string testBaseDir;
    std::vector<std::string> createdFiles;
    std::vector<std::string> createdDirs;

    void SetUp() override {
        testBaseDir = (fs::temp_directory_path() / "WheelDL_ImageExporter_Test").string();
        fs::remove_all(testBaseDir);
        fs::create_directories(testBaseDir);
    }

    void TearDown() override {
        try {
            fs::remove_all(testBaseDir);
            for (const auto& dir : createdDirs) {
                fs::remove_all(dir);
            }
        }
        catch (...) {}
    }

    std::string getTestPath(const std::string& filename) {
        return (fs::path(testBaseDir) / filename).string();
    }

    cv::Mat createTestImage(int width = 100, int height = 100, int channels = 3) {
        return cv::Mat(height, width, channels == 3 ? CV_8UC3 : CV_8UC1, cv::Scalar(128, 64, 32));
    }

    void trackDir(const std::string& dir) {
        createdDirs.push_back(dir);
    }
};

// =============================================================================
// Save Tests (IE-001 ~ IE-006)
// =============================================================================

// IE-001: Save valid image
TEST_F(ImageExporterTest, Save_ValidImage) {
    cv::Mat image = createTestImage();
    std::string filepath = getTestPath("valid_image.png");

    bool result = ImageExporter::save(image, filepath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(filepath));
}

// IE-002: Save empty image returns false
TEST_F(ImageExporterTest, Save_EmptyImage) {
    cv::Mat emptyImage;
    std::string filepath = getTestPath("empty_image.png");

    bool result = ImageExporter::save(emptyImage, filepath);

    EXPECT_FALSE(result);
    EXPECT_FALSE(fs::exists(filepath));
}

// IE-003: Save with invalid path returns false
TEST_F(ImageExporterTest, Save_InvalidPath) {
    cv::Mat image = createTestImage();
    // Invalid path with invalid characters
    std::string filepath = "Z:\\<invalid>\\path|test\\image.png";

    bool result = ImageExporter::save(image, filepath);

    EXPECT_FALSE(result);
}

// IE-004: Save creates directory if not exists
TEST_F(ImageExporterTest, Save_CreateDirectory) {
    cv::Mat image = createTestImage();
    std::string subdir = getTestPath("new_subdir/nested/image.png");

    bool result = ImageExporter::save(image, subdir);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(subdir));
}

// IE-005: Save PNG format
TEST_F(ImageExporterTest, Save_PNG) {
    cv::Mat image = createTestImage();
    std::string filepath = getTestPath("test.png");

    bool result = ImageExporter::save(image, filepath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(filepath));

    // Verify it's a valid PNG by reading it back
    cv::Mat loaded = cv::imread(filepath);
    EXPECT_FALSE(loaded.empty());
    EXPECT_EQ(image.rows, loaded.rows);
    EXPECT_EQ(image.cols, loaded.cols);
}

// IE-006: Save JPG format
TEST_F(ImageExporterTest, Save_JPG) {
    cv::Mat image = createTestImage();
    std::string filepath = getTestPath("test.jpg");

    bool result = ImageExporter::save(image, filepath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(filepath));

    // Verify it's a valid JPEG by reading it back
    cv::Mat loaded = cv::imread(filepath);
    EXPECT_FALSE(loaded.empty());
}

// =============================================================================
// SaveBatch Tests (IE-007 ~ IE-013)
// =============================================================================

// IE-007: SaveBatch multiple images
TEST_F(ImageExporterTest, SaveBatch_Multiple) {
    std::vector<cv::Mat> images;
    std::vector<std::string> imagePaths;

    for (int i = 0; i < 5; ++i) {
        images.push_back(createTestImage());
        imagePaths.push_back("folder/image" + std::to_string(i) + ".png");
    }

    std::string outputDir = getTestPath("batch_output");

    size_t count = ImageExporter::saveBatch(images, imagePaths, outputDir);

    EXPECT_EQ(5u, count);
    EXPECT_TRUE(fs::exists(outputDir));
}

// IE-008: SaveBatch partial failure (some empty images)
TEST_F(ImageExporterTest, SaveBatch_PartialFailure) {
    std::vector<cv::Mat> images;
    std::vector<std::string> imagePaths;

    images.push_back(createTestImage());
    imagePaths.push_back("folder/image1.png");

    images.push_back(cv::Mat());  // Empty image
    imagePaths.push_back("folder/image2.png");

    images.push_back(createTestImage());
    imagePaths.push_back("folder/image3.png");

    std::string outputDir = getTestPath("partial_output");

    size_t count = ImageExporter::saveBatch(images, imagePaths, outputDir);

    // Should save 2 out of 3 (skipping empty)
    EXPECT_EQ(2u, count);
}

// IE-009: SaveBatch with suffix
TEST_F(ImageExporterTest, SaveBatch_Suffix) {
    std::vector<cv::Mat> images = {createTestImage()};
    std::vector<std::string> imagePaths = {"folder/myimage.png"};

    std::string outputDir = getTestPath("suffix_output");

    size_t count = ImageExporter::saveBatch(images, imagePaths, outputDir, "_result");

    EXPECT_EQ(1u, count);

    // Check filename contains suffix
    bool foundWithSuffix = false;
    for (const auto& entry : fs::directory_iterator(outputDir)) {
        std::string filename = entry.path().filename().string();
        if (filename.find("_result") != std::string::npos) {
            foundWithSuffix = true;
            break;
        }
    }
    EXPECT_TRUE(foundWithSuffix);
}

// IE-010: SaveBatch empty list
TEST_F(ImageExporterTest, SaveBatch_EmptyList) {
    std::vector<cv::Mat> images;
    std::vector<std::string> imagePaths;

    std::string outputDir = getTestPath("empty_output");

    size_t count = ImageExporter::saveBatch(images, imagePaths, outputDir);

    EXPECT_EQ(0u, count);
}

// IE-011: SaveBatch mismatched sizes
TEST_F(ImageExporterTest, SaveBatch_MismatchedSizes) {
    std::vector<cv::Mat> images = {createTestImage(), createTestImage()};
    std::vector<std::string> imagePaths = {"image1.png"};  // Only 1 path for 2 images

    std::string outputDir = getTestPath("mismatch_output");

    size_t count = ImageExporter::saveBatch(images, imagePaths, outputDir);

    EXPECT_EQ(0u, count);
}

// IE-012: SaveBatch skips empty images
TEST_F(ImageExporterTest, SaveBatch_SkipsEmptyImages) {
    std::vector<cv::Mat> images;
    std::vector<std::string> imagePaths;

    images.push_back(cv::Mat());  // Empty
    imagePaths.push_back("folder/empty1.png");

    images.push_back(createTestImage());  // Valid
    imagePaths.push_back("folder/valid.png");

    images.push_back(cv::Mat());  // Empty
    imagePaths.push_back("folder/empty2.png");

    std::string outputDir = getTestPath("skip_empty_output");

    size_t count = ImageExporter::saveBatch(images, imagePaths, outputDir);

    EXPECT_EQ(1u, count);
}

// IE-013: SaveBatch unique filenames with parent folder prefix
TEST_F(ImageExporterTest, SaveBatch_UniqueFilenames) {
    std::vector<cv::Mat> images = {createTestImage(), createTestImage()};
    std::vector<std::string> imagePaths = {
        "folder1/image.png",
        "folder2/image.png"
    };

    std::string outputDir = getTestPath("unique_output");

    size_t count = ImageExporter::saveBatch(images, imagePaths, outputDir);

    EXPECT_EQ(2u, count);

    // Count files in output directory
    int fileCount = 0;
    for (const auto& entry : fs::directory_iterator(outputDir)) {
        if (entry.is_regular_file()) {
            fileCount++;
        }
    }
    EXPECT_EQ(2, fileCount);
}

// =============================================================================
// Exception Handling Tests (IE-014 ~ IE-015)
// =============================================================================

// IE-014: Save exception handled
TEST_F(ImageExporterTest, Save_ExceptionHandled) {
    cv::Mat image = createTestImage();

    // Should not throw, returns false on failure
    EXPECT_NO_THROW({
        bool result = ImageExporter::save(image, "Z:\\nonexistent\\<invalid>\\image.png");
        EXPECT_FALSE(result);
    });
}

// IE-015: SaveBatch exception handled
TEST_F(ImageExporterTest, SaveBatch_ExceptionHandled) {
    std::vector<cv::Mat> images = {createTestImage()};
    std::vector<std::string> imagePaths = {"folder/image.png"};

    // Should not throw
    EXPECT_NO_THROW({
        size_t count = ImageExporter::saveBatch(images, imagePaths, "Z:\\nonexistent\\<invalid>");
        EXPECT_EQ(0u, count);
    });
}

// =============================================================================
// Additional Tests
// =============================================================================

TEST_F(ImageExporterTest, Save_GrayscaleImage) {
    cv::Mat grayImage = createTestImage(100, 100, 1);
    std::string filepath = getTestPath("grayscale.png");

    bool result = ImageExporter::save(grayImage, filepath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(filepath));

    cv::Mat loaded = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    EXPECT_FALSE(loaded.empty());
    EXPECT_EQ(1, loaded.channels());
}

TEST_F(ImageExporterTest, Save_LargeImage) {
    cv::Mat largeImage = createTestImage(1920, 1080);
    std::string filepath = getTestPath("large_image.png");

    bool result = ImageExporter::save(largeImage, filepath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(filepath));
}

TEST_F(ImageExporterTest, Save_SmallImage) {
    cv::Mat smallImage = createTestImage(1, 1);
    std::string filepath = getTestPath("tiny.png");

    bool result = ImageExporter::save(smallImage, filepath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(filepath));
}

TEST_F(ImageExporterTest, Save_OverwriteExisting) {
    cv::Mat image1 = cv::Mat(100, 100, CV_8UC3, cv::Scalar(255, 0, 0));  // Blue
    cv::Mat image2 = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 255, 0));  // Green
    std::string filepath = getTestPath("overwrite.png");

    // Save first image
    EXPECT_TRUE(ImageExporter::save(image1, filepath));

    // Overwrite with second image
    EXPECT_TRUE(ImageExporter::save(image2, filepath));

    // Verify overwritten
    cv::Mat loaded = cv::imread(filepath);
    EXPECT_FALSE(loaded.empty());
    // Check it's green (BGR format)
    cv::Vec3b pixel = loaded.at<cv::Vec3b>(50, 50);
    EXPECT_EQ(0, pixel[0]);    // B
    EXPECT_EQ(255, pixel[1]);  // G
    EXPECT_EQ(0, pixel[2]);    // R
}

TEST_F(ImageExporterTest, SaveBatch_DefaultSuffix) {
    std::vector<cv::Mat> images = {createTestImage()};
    std::vector<std::string> imagePaths = {"folder/test.png"};

    std::string outputDir = getTestPath("default_suffix");

    size_t count = ImageExporter::saveBatch(images, imagePaths, outputDir);

    EXPECT_EQ(1u, count);

    // Default suffix is "_output"
    bool foundDefault = false;
    for (const auto& entry : fs::directory_iterator(outputDir)) {
        std::string filename = entry.path().filename().string();
        if (filename.find("_output") != std::string::npos) {
            foundDefault = true;
            break;
        }
    }
    EXPECT_TRUE(foundDefault);
}
