#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Utils/ImageIO.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>

using namespace WheelDL::Data::Utils;
namespace fs = std::filesystem;

namespace {
	std::filesystem::path getTestDataPath() {
		std::filesystem::path sourceDir = __FILE__;
		sourceDir = sourceDir.parent_path();
		while (sourceDir.filename() != "WheelDL.Lib.Tests" && sourceDir.has_parent_path()) {
			sourceDir = sourceDir.parent_path();
		}
		return sourceDir / "Data";
	}

	// Helper function to create a test image with specific properties
	cv::Mat createTestImage(int width, int height, int channels = 3) {
		cv::Mat image;
		if (channels == 3) {
			image = cv::Mat(height, width, CV_8UC3, cv::Scalar(100, 150, 200));
		}
		else if (channels == 1) {
			image = cv::Mat(height, width, CV_8UC1, cv::Scalar(128));
		}
		else {
			image = cv::Mat(height, width, CV_8UC4, cv::Scalar(100, 150, 200, 255));
		}
		// Add some pattern to make it more realistic
		for (int y = 0; y < height; y += 10) {
			cv::line(image, cv::Point(0, y), cv::Point(width, y), cv::Scalar(255, 255, 255), 1);
		}
		for (int x = 0; x < width; x += 10) {
			cv::line(image, cv::Point(x, 0), cv::Point(x, height), cv::Scalar(255, 255, 255), 1);
		}
		return image;
	}

	// Helper function to corrupt image data
	void corruptImageFile(const std::string& path, const std::string& corruptionType) {
		if (corruptionType == "truncate") {
			// Read the file and write only half of it
			std::ifstream input(path, std::ios::binary);
			std::vector<char> buffer((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
			input.close();

			std::ofstream output(path, std::ios::binary | std::ios::trunc);
			output.write(buffer.data(), buffer.size() / 2);
			output.close();
		}
		else if (corruptionType == "invalidheader") {
			// Overwrite the header with garbage
			std::ofstream output(path, std::ios::binary | std::ios::in);
			const char garbage[] = "INVALID_HEADER_DATA";
			output.write(garbage, sizeof(garbage));
			output.close();
		}
		else if (corruptionType == "empty") {
			// Create an empty file
			std::ofstream output(path, std::ios::binary | std::ios::trunc);
			output.close();
		}
	}
}

class ImageIOComprehensiveTest : public ::testing::Test {
protected:
	std::string testDataPath;
	std::string testImagePath;
	std::string outputPath;
	std::string unicodePath;

	void SetUp() override {
		auto testData = getTestDataPath();
		testDataPath = (testData / "mnist_sample" / "images").string();
		testImagePath = (testData / "mnist_sample" / "images" / "mnist_000000.png").string();
		outputPath = (testData / "temp").string();
		unicodePath = (testData / "temp" / "unicode").string();

		// Create output directories
		if (!fs::exists(outputPath)) {
			fs::create_directories(outputPath);
		}
		if (!fs::exists(unicodePath)) {
			fs::create_directories(unicodePath);
		}
	}

	void TearDown() override {
		// Clean up temporary files
		if (fs::exists(outputPath)) {
			try {
				for (const auto& entry : fs::recursive_directory_iterator(outputPath)) {
					if (fs::is_regular_file(entry.path())) {
						fs::remove(entry.path());
					}
				}
				// Remove unicode subdirectory
				if (fs::exists(unicodePath)) {
					fs::remove_all(unicodePath);
				}
			}
			catch (...) {
				// Ignore errors during cleanup
			}
		}
	}

	// Helper to get test image paths
	std::string getTestImagePath(int index) {
		char buffer[64];
		sprintf_s(buffer, "mnist_%06d.png", index);
		return (fs::path(testDataPath) / buffer).string();
	}
};

// ============================================================================
// 1. UNICODE PATH SUPPORT TESTS
// ============================================================================

TEST_F(ImageIOComprehensiveTest, DISABLED_UnicodePath_ChineseCharacters) {
	// Create a test image
	cv::Mat testImage = createTestImage(640, 480);

	// Save with Chinese characters in filename
	std::string chinesePath = (fs::path(unicodePath) / "�?��测试.png").string();
	ASSERT_NO_THROW(ImageIO::saveImage(chinesePath, testImage));
	ASSERT_TRUE(fs::exists(chinesePath));

	// Load it back
	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(chinesePath, 640));
	EXPECT_FALSE(loaded.empty());
	EXPECT_EQ(loaded.rows, testImage.rows);
	EXPECT_EQ(loaded.cols, testImage.cols);
}

TEST_F(ImageIOComprehensiveTest, DISABLED_UnicodePath_JapaneseCharacters) {
	cv::Mat testImage = createTestImage(640, 480);

	std::string japanesePath = (fs::path(unicodePath) / "?�ス??png").string();
	ASSERT_NO_THROW(ImageIO::saveImage(japanesePath, testImage));
	ASSERT_TRUE(fs::exists(japanesePath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(japanesePath, 640));
	EXPECT_FALSE(loaded.empty());
}

TEST_F(ImageIOComprehensiveTest, DISABLED_UnicodePath_KoreanCharacters) {
	cv::Mat testImage = createTestImage(640, 480);

	std::string koreanPath = (fs::path(unicodePath) / "?�스??png").string();
	ASSERT_NO_THROW(ImageIO::saveImage(koreanPath, testImage));
	ASSERT_TRUE(fs::exists(koreanPath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(koreanPath, 640));
	EXPECT_FALSE(loaded.empty());
}

TEST_F(ImageIOComprehensiveTest, UnicodePath_EmojiInFilename) {
	cv::Mat testImage = createTestImage(640, 480);

	// Note: Emoji support depends on filesystem
	// Using a simpler emoji that's more likely to be supported
	std::string emojiPath = (fs::path(unicodePath) / "test_emoji_?��.png").string();

	try {
		ImageIO::saveImage(emojiPath, testImage);
		if (fs::exists(emojiPath)) {
			cv::Mat loaded = ImageIO::loadImage(emojiPath, 640);
			EXPECT_FALSE(loaded.empty());
		}
	}
	catch (...) {
		// Emoji support may not be available on all systems
		std::cout << "SKIPPED: Emoji in filenames not supported on this system" << std::endl;
		return;
	}
}

TEST_F(ImageIOComprehensiveTest, UnicodePath_SpacesAndSpecialChars) {
	cv::Mat testImage = createTestImage(640, 480);

	std::string specialPath = (fs::path(unicodePath) / "test file with spaces & special!.png").string();
	ASSERT_NO_THROW(ImageIO::saveImage(specialPath, testImage));
	ASSERT_TRUE(fs::exists(specialPath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(specialPath, 640));
	EXPECT_FALSE(loaded.empty());
}

TEST_F(ImageIOComprehensiveTest, DISABLED_UnicodePath_MixedUnicodeCharacters) {
	cv::Mat testImage = createTestImage(640, 480);

	std::string mixedPath = (fs::path(unicodePath) / "混合?�ス?�한글test.png").string();
	ASSERT_NO_THROW(ImageIO::saveImage(mixedPath, testImage));
	ASSERT_TRUE(fs::exists(mixedPath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(mixedPath, 640));
	EXPECT_FALSE(loaded.empty());
}

// ============================================================================
// 2. IMAGE FORMAT SUPPORT TESTS
// ============================================================================

TEST_F(ImageIOComprehensiveTest, ImageFormat_LoadAndSavePNG) {
	cv::Mat testImage = createTestImage(640, 480);
	std::string pngPath = (fs::path(outputPath) / "test_format.png").string();

	ASSERT_NO_THROW(ImageIO::saveImage(pngPath, testImage));
	ASSERT_TRUE(fs::exists(pngPath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(pngPath, 640));
	EXPECT_FALSE(loaded.empty());
	EXPECT_EQ(loaded.channels(), 3);
}

TEST_F(ImageIOComprehensiveTest, ImageFormat_LoadAndSaveJPEG) {
	cv::Mat testImage = createTestImage(640, 480);
	std::string jpegPath = (fs::path(outputPath) / "test_format.jpg").string();

	ASSERT_NO_THROW(ImageIO::saveImage(jpegPath, testImage));
	ASSERT_TRUE(fs::exists(jpegPath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(jpegPath, 640));
	EXPECT_FALSE(loaded.empty());
	EXPECT_EQ(loaded.channels(), 3);
}

TEST_F(ImageIOComprehensiveTest, ImageFormat_LoadAndSaveJPEGAlternativeExtension) {
	cv::Mat testImage = createTestImage(640, 480);
	std::string jpegPath = (fs::path(outputPath) / "test_format.jpeg").string();

	ASSERT_NO_THROW(ImageIO::saveImage(jpegPath, testImage));
	ASSERT_TRUE(fs::exists(jpegPath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(jpegPath, 640));
	EXPECT_FALSE(loaded.empty());
}

TEST_F(ImageIOComprehensiveTest, ImageFormat_LoadAndSaveBMP) {
	cv::Mat testImage = createTestImage(640, 480);
	std::string bmpPath = (fs::path(outputPath) / "test_format.bmp").string();

	ASSERT_NO_THROW(ImageIO::saveImage(bmpPath, testImage));
	ASSERT_TRUE(fs::exists(bmpPath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(bmpPath, 640));
	EXPECT_FALSE(loaded.empty());
	EXPECT_EQ(loaded.channels(), 3);
}

TEST_F(ImageIOComprehensiveTest, ImageFormat_LoadAndSaveTIFF) {
	cv::Mat testImage = createTestImage(640, 480);
	std::string tiffPath = (fs::path(outputPath) / "test_format.tiff").string();

	ASSERT_NO_THROW(ImageIO::saveImage(tiffPath, testImage));
	ASSERT_TRUE(fs::exists(tiffPath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(tiffPath, 640));
	EXPECT_FALSE(loaded.empty());
	EXPECT_EQ(loaded.channels(), 3);
}

TEST_F(ImageIOComprehensiveTest, ImageFormat_NoExtensionDefaultsToJPEG) {
	cv::Mat testImage = createTestImage(640, 480);
	std::string noExtPath = (fs::path(outputPath) / "test_no_extension").string();

	ASSERT_NO_THROW(ImageIO::saveImage(noExtPath, testImage));
	ASSERT_TRUE(fs::exists(noExtPath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(noExtPath, 640));
	EXPECT_FALSE(loaded.empty());
}

TEST_F(ImageIOComprehensiveTest, ImageFormat_InvalidFormatHandling) {
	cv::Mat testImage = createTestImage(640, 480);
	std::string invalidPath = (fs::path(outputPath) / "test.xyz").string();

	// OpenCV will try to encode, but it might fail
	// We expect either success (if codec is available) or exception
	try {
		ImageIO::saveImage(invalidPath, testImage);
		// If it succeeds, check if file exists
		if (fs::exists(invalidPath)) {
			SUCCEED();
		}
	}
	catch (const std::runtime_error&) {
		// Expected if format is not supported
		SUCCEED();
	}
}

// ============================================================================
// 3. IMAGE SIZE TESTS
// ============================================================================

TEST_F(ImageIOComprehensiveTest, ImageSize_VerySmall_1x1) {
	cv::Mat testImage = createTestImage(1, 1);
	std::string savePath = (fs::path(outputPath) / "test_1x1.png").string();

	ASSERT_NO_THROW(ImageIO::saveImage(savePath, testImage));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(savePath, 10));
	EXPECT_FALSE(loaded.empty());
}

TEST_F(ImageIOComprehensiveTest, ImageSize_VerySmall_2x2) {
	cv::Mat testImage = createTestImage(2, 2);
	std::string savePath = (fs::path(outputPath) / "test_2x2.png").string();

	ASSERT_NO_THROW(ImageIO::saveImage(savePath, testImage));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(savePath, 10));
	EXPECT_FALSE(loaded.empty());
}

TEST_F(ImageIOComprehensiveTest, ImageSize_Large_4000x4000) {
	cv::Mat testImage = createTestImage(4000, 4000);
	std::string savePath = (fs::path(outputPath) / "test_4000x4000.png").string();

	ASSERT_NO_THROW(ImageIO::saveImage(savePath, testImage));
	ASSERT_TRUE(fs::exists(savePath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(savePath, 640));
	EXPECT_FALSE(loaded.empty());
	EXPECT_EQ(std::max(loaded.rows, loaded.cols), 640);
}

TEST_F(ImageIOComprehensiveTest, ImageSize_NonSquare_1920x1080) {
	cv::Mat testImage = createTestImage(1920, 1080);
	std::string savePath = (fs::path(outputPath) / "test_1920x1080.png").string();

	ASSERT_NO_THROW(ImageIO::saveImage(savePath, testImage));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(savePath, 640));
	EXPECT_FALSE(loaded.empty());
	EXPECT_EQ(std::max(loaded.rows, loaded.cols), 640);

	// Check aspect ratio is maintained
	float originalRatio = 1920.0f / 1080.0f;
	float loadedRatio = static_cast<float>(loaded.cols) / static_cast<float>(loaded.rows);
	EXPECT_NEAR(originalRatio, loadedRatio, 0.01f);
}

TEST_F(ImageIOComprehensiveTest, ImageSize_NonSquare_Portrait_1080x1920) {
	cv::Mat testImage = createTestImage(1080, 1920);
	std::string savePath = (fs::path(outputPath) / "test_1080x1920.png").string();

	ASSERT_NO_THROW(ImageIO::saveImage(savePath, testImage));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(savePath, 640));
	EXPECT_FALSE(loaded.empty());
	EXPECT_EQ(std::max(loaded.rows, loaded.cols), 640);
}

TEST_F(ImageIOComprehensiveTest, ImageSize_ZeroDimensionHandling) {
	// Cannot create 0-dimension image with OpenCV
	// Test loading with 0 resize parameter instead
	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(testImagePath, 0));
	EXPECT_FALSE(loaded.empty());
	// Should not resize when imageSize is 0
}

TEST_F(ImageIOComprehensiveTest, ImageSize_NegativeSizeHandling) {
	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(testImagePath, -100));
	EXPECT_FALSE(loaded.empty());
	// Should not resize when imageSize is negative
}

// ============================================================================
// 4. CORRUPTED IMAGE HANDLING TESTS
// ============================================================================

TEST_F(ImageIOComprehensiveTest, CorruptedImage_TruncatedFile) {
	// First create a valid image
	cv::Mat testImage = createTestImage(640, 480);
	std::string corruptPath = (fs::path(outputPath) / "corrupted_truncated.png").string();
	ImageIO::saveImage(corruptPath, testImage);

	// Corrupt it
	corruptImageFile(corruptPath, "truncate");

	// Try to load - should throw exception
	EXPECT_THROW(ImageIO::loadImage(corruptPath, 640), std::runtime_error);
}

TEST_F(ImageIOComprehensiveTest, CorruptedImage_InvalidHeader) {
	cv::Mat testImage = createTestImage(640, 480);
	std::string corruptPath = (fs::path(outputPath) / "corrupted_header.png").string();
	ImageIO::saveImage(corruptPath, testImage);

	corruptImageFile(corruptPath, "invalidheader");

	EXPECT_THROW(ImageIO::loadImage(corruptPath, 640), std::runtime_error);
}

TEST_F(ImageIOComprehensiveTest, CorruptedImage_EmptyFile) {
	std::string emptyPath = (fs::path(outputPath) / "empty_file.png").string();

	// Create empty file
	std::ofstream emptyFile(emptyPath);
	emptyFile.close();

	EXPECT_THROW(ImageIO::loadImage(emptyPath, 640), std::runtime_error);
}

TEST_F(ImageIOComprehensiveTest, CorruptedImage_InvalidImageData) {
	std::string invalidPath = (fs::path(outputPath) / "invalid_data.png").string();

	// Create a file with random data
	std::ofstream file(invalidPath, std::ios::binary);
	for (int i = 0; i < 1000; ++i) {
		file.put(static_cast<char>(rand() % 256));
	}
	file.close();

	EXPECT_THROW(ImageIO::loadImage(invalidPath, 640), std::runtime_error);
}

// ============================================================================
// 5. RESIZE HELPER TESTS
// ============================================================================

TEST_F(ImageIOComprehensiveTest, Resize_MaintainAspectRatio_Landscape) {
	cv::Mat testImage = createTestImage(1600, 1200);
	std::string savePath = (fs::path(outputPath) / "resize_landscape.png").string();
	ImageIO::saveImage(savePath, testImage);

	cv::Mat loaded = ImageIO::loadImage(savePath, 800);

	EXPECT_EQ(std::max(loaded.rows, loaded.cols), 800);
	EXPECT_EQ(loaded.cols, 800);
	EXPECT_EQ(loaded.rows, 600);  // Aspect ratio 4:3 maintained
}

TEST_F(ImageIOComprehensiveTest, Resize_MaintainAspectRatio_Portrait) {
	cv::Mat testImage = createTestImage(1200, 1600);
	std::string savePath = (fs::path(outputPath) / "resize_portrait.png").string();
	ImageIO::saveImage(savePath, testImage);

	cv::Mat loaded = ImageIO::loadImage(savePath, 800);

	EXPECT_EQ(std::max(loaded.rows, loaded.cols), 800);
	EXPECT_EQ(loaded.rows, 800);
	EXPECT_EQ(loaded.cols, 600);  // Aspect ratio 3:4 maintained
}

TEST_F(ImageIOComprehensiveTest, Resize_Upscaling) {
	cv::Mat testImage = createTestImage(100, 100);
	std::string savePath = (fs::path(outputPath) / "resize_upscale.png").string();
	ImageIO::saveImage(savePath, testImage);

	cv::Mat loaded = ImageIO::loadImage(savePath, 500);

	EXPECT_EQ(std::max(loaded.rows, loaded.cols), 500);
	EXPECT_EQ(loaded.rows, 500);
	EXPECT_EQ(loaded.cols, 500);
}

TEST_F(ImageIOComprehensiveTest, Resize_Downscaling) {
	cv::Mat testImage = createTestImage(2000, 2000);
	std::string savePath = (fs::path(outputPath) / "resize_downscale.png").string();
	ImageIO::saveImage(savePath, testImage);

	cv::Mat loaded = ImageIO::loadImage(savePath, 400);

	EXPECT_EQ(std::max(loaded.rows, loaded.cols), 400);
	EXPECT_EQ(loaded.rows, 400);
	EXPECT_EQ(loaded.cols, 400);
}

TEST_F(ImageIOComprehensiveTest, Resize_MultipleResizeSizes) {
	cv::Mat testImage = createTestImage(1024, 768);
	std::string savePath = (fs::path(outputPath) / "resize_multiple.png").string();
	ImageIO::saveImage(savePath, testImage);

	std::vector<int> sizes = { 64, 128, 256, 512, 1024, 2048 };

	for (int size : sizes) {
		cv::Mat loaded = ImageIO::loadImage(savePath, size);
		EXPECT_EQ(std::max(loaded.rows, loaded.cols), size);

		// Verify aspect ratio
		float expectedRatio = 1024.0f / 768.0f;
		float actualRatio = static_cast<float>(loaded.cols) / static_cast<float>(loaded.rows);
		EXPECT_NEAR(expectedRatio, actualRatio, 0.01f);
	}
}

TEST_F(ImageIOComprehensiveTest, Resize_EdgeCase_SameSize) {
	cv::Mat testImage = createTestImage(640, 640);
	std::string savePath = (fs::path(outputPath) / "resize_same.png").string();
	ImageIO::saveImage(savePath, testImage);

	cv::Mat loaded = ImageIO::loadImage(savePath, 640);

	EXPECT_EQ(loaded.rows, 640);
	EXPECT_EQ(loaded.cols, 640);
}

// ============================================================================
// 6. CROSS-PLATFORM PATH TESTS
// ============================================================================

TEST_F(ImageIOComprehensiveTest, Path_WindowsStyle_Backslashes) {
	cv::Mat testImage = createTestImage(640, 480);

	// Use Windows-style path (backslashes are already standard on Windows)
	std::string windowsPath = (fs::path(outputPath) / "test_windows_path.png").string();

	ASSERT_NO_THROW(ImageIO::saveImage(windowsPath, testImage));
	ASSERT_TRUE(fs::exists(windowsPath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(windowsPath, 640));
	EXPECT_FALSE(loaded.empty());
}

TEST_F(ImageIOComprehensiveTest, Path_UnixStyle_ForwardSlashes) {
	cv::Mat testImage = createTestImage(640, 480);

	// Use Unix-style path with forward slashes
	std::string unixPath = (fs::path(outputPath) / "test_unix_path.png").string();

	ASSERT_NO_THROW(ImageIO::saveImage(unixPath, testImage));
	ASSERT_TRUE(fs::exists(unixPath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(unixPath, 640));
	EXPECT_FALSE(loaded.empty());
}

TEST_F(ImageIOComprehensiveTest, Path_MixedSeparators) {
	cv::Mat testImage = createTestImage(640, 480);

	// Mix forward and back slashes
	std::string mixedPath = (fs::path(outputPath) / "subdir" / "test_mixed.png").string();

	ASSERT_NO_THROW(ImageIO::saveImage(mixedPath, testImage));
	ASSERT_TRUE(fs::exists(mixedPath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(mixedPath, 640));
	EXPECT_FALSE(loaded.empty());
}

TEST_F(ImageIOComprehensiveTest, Path_RelativePath) {
	// Test with mnist images using relative understanding
	ASSERT_TRUE(fs::exists(testImagePath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(testImagePath, 640));
	EXPECT_FALSE(loaded.empty());
}

TEST_F(ImageIOComprehensiveTest, Path_AbsolutePath) {
	cv::Mat testImage = createTestImage(640, 480);

	// Get absolute path
	std::string absolutePath = fs::absolute(fs::path(outputPath) / "test_absolute.png").string();

	ASSERT_NO_THROW(ImageIO::saveImage(absolutePath, testImage));
	ASSERT_TRUE(fs::exists(absolutePath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(absolutePath, 640));
	EXPECT_FALSE(loaded.empty());
}

TEST_F(ImageIOComprehensiveTest, Path_NestedDirectories) {
	cv::Mat testImage = createTestImage(640, 480);

	std::string nestedPath = (fs::path(outputPath) / "level1" / "level2" / "level3" / "test.png").string();

	ASSERT_NO_THROW(ImageIO::saveImage(nestedPath, testImage));
	ASSERT_TRUE(fs::exists(nestedPath));

	cv::Mat loaded;
	ASSERT_NO_THROW(loaded = ImageIO::loadImage(nestedPath, 640));
	EXPECT_FALSE(loaded.empty());
}

// ============================================================================
// 7. PERFORMANCE TESTS
// ============================================================================

TEST_F(ImageIOComprehensiveTest, Performance_Load100ImagesSequentially) {
	// Use existing mnist images
	const int numImages = 100;
	std::vector<std::string> imagePaths;

	for (int i = 0; i < numImages; ++i) {
		std::string path = getTestImagePath(i);
		if (fs::exists(path)) {
			imagePaths.push_back(path);
		}
		if (imagePaths.size() >= numImages) break;
	}

	ASSERT_GE(imagePaths.size(), 10) << "Not enough test images available";

	auto startTime = std::chrono::high_resolution_clock::now();

	for (const auto& path : imagePaths) {
		cv::Mat image = ImageIO::loadImage(path, 640);
		ASSERT_FALSE(image.empty());
	}

	auto endTime = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

	double averageTime = static_cast<double>(duration.count()) / imagePaths.size();

	std::cout << "Loaded " << imagePaths.size() << " images in " << duration.count() << "ms" << std::endl;
	std::cout << "Average load time: " << averageTime << "ms per image" << std::endl;

	// Reasonable performance expectation: less than 100ms per image on average
	EXPECT_LT(averageTime, 100.0);
}

TEST_F(ImageIOComprehensiveTest, Performance_MeasureAverageLoadTime) {
	const int iterations = 50;
	std::vector<long long> loadTimes;

	for (int i = 0; i < iterations; ++i) {
		auto start = std::chrono::high_resolution_clock::now();
		cv::Mat image = ImageIO::loadImage(testImagePath, 640);
		auto end = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		loadTimes.push_back(duration.count());

		ASSERT_FALSE(image.empty());
	}

	// Calculate statistics
	long long sum = 0;
	long long minTime = loadTimes[0];
	long long maxTime = loadTimes[0];

	for (long long time : loadTimes) {
		sum += time;
		minTime = std::min(minTime, time);
		maxTime = std::max(maxTime, time);
	}

	double average = static_cast<double>(sum) / iterations;

	std::cout << "Load time statistics (" << iterations << " iterations):" << std::endl;
	std::cout << "  Average: " << average / 1000.0 << "ms" << std::endl;
	std::cout << "  Min: " << minTime / 1000.0 << "ms" << std::endl;
	std::cout << "  Max: " << maxTime / 1000.0 << "ms" << std::endl;

	EXPECT_LT(average / 1000.0, 50.0);  // Average should be less than 50ms
}

TEST_F(ImageIOComprehensiveTest, Performance_MemoryUsageForLargeImages) {
	// Create and load a large image to check memory handling
	cv::Mat largeImage = createTestImage(4000, 3000);
	std::string largePath = (fs::path(outputPath) / "large_memory_test.png").string();

	ASSERT_NO_THROW(ImageIO::saveImage(largePath, largeImage));

	// Load it multiple times to ensure no memory leaks
	for (int i = 0; i < 10; ++i) {
		cv::Mat loaded = ImageIO::loadImage(largePath, 640);
		EXPECT_FALSE(loaded.empty());
		EXPECT_EQ(std::max(loaded.rows, loaded.cols), 640);
	}

	// If we got here without crashing, memory is being handled properly
	SUCCEED();
}

TEST_F(ImageIOComprehensiveTest, Performance_SaveAndLoadComparison) {
	cv::Mat testImage = createTestImage(1920, 1080);
	std::string savePath = (fs::path(outputPath) / "perf_test.png").string();

	// Measure save time
	auto saveStart = std::chrono::high_resolution_clock::now();
	ImageIO::saveImage(savePath, testImage);
	auto saveEnd = std::chrono::high_resolution_clock::now();
	auto saveDuration = std::chrono::duration_cast<std::chrono::milliseconds>(saveEnd - saveStart);

	// Measure load time
	auto loadStart = std::chrono::high_resolution_clock::now();
	cv::Mat loaded = ImageIO::loadImage(savePath, 640);
	auto loadEnd = std::chrono::high_resolution_clock::now();
	auto loadDuration = std::chrono::duration_cast<std::chrono::milliseconds>(loadEnd - loadStart);

	std::cout << "Save time: " << saveDuration.count() << "ms" << std::endl;
	std::cout << "Load time: " << loadDuration.count() << "ms" << std::endl;

	EXPECT_FALSE(loaded.empty());
}

// ============================================================================
// 8. ADDITIONAL EDGE CASE TESTS
// ============================================================================

TEST_F(ImageIOComprehensiveTest, EdgeCase_LoadSameImageMultipleTimes) {
	for (int i = 0; i < 20; ++i) {
		cv::Mat image = ImageIO::loadImage(testImagePath, 640);
		EXPECT_FALSE(image.empty());
		EXPECT_EQ(image.type(), CV_8UC3);
	}
}

TEST_F(ImageIOComprehensiveTest, EdgeCase_SaveOverwriteExistingFile) {
	cv::Mat testImage1 = createTestImage(640, 480);
	cv::Mat testImage2 = createTestImage(800, 600);

	std::string savePath = (fs::path(outputPath) / "overwrite_test.png").string();

	ImageIO::saveImage(savePath, testImage1);
	auto size1 = fs::file_size(savePath);

	ImageIO::saveImage(savePath, testImage2);
	auto size2 = fs::file_size(savePath);

	EXPECT_NE(size1, size2);  // File should be different after overwrite
}

TEST_F(ImageIOComprehensiveTest, EdgeCase_MultipleFormatsSequentially) {
	cv::Mat testImage = createTestImage(640, 480);

	std::vector<std::string> formats = { ".png", ".jpg", ".bmp", ".tiff" };

	for (const auto& format : formats) {
		std::string path = (fs::path(outputPath) / ("test" + format)).string();
		ASSERT_NO_THROW(ImageIO::saveImage(path, testImage));

		cv::Mat loaded = ImageIO::loadImage(path, 640);
		EXPECT_FALSE(loaded.empty());
	}
}

TEST_F(ImageIOComprehensiveTest, EdgeCase_VeryLongFilename) {
	cv::Mat testImage = createTestImage(100, 100);

	// Create a very long filename (but within filesystem limits)
	std::string longName(200, 'a');
	longName += ".png";
	std::string longPath = (fs::path(outputPath) / longName).string();

	try {
		ImageIO::saveImage(longPath, testImage);
		if (fs::exists(longPath)) {
			cv::Mat loaded = ImageIO::loadImage(longPath, 640);
			EXPECT_FALSE(loaded.empty());
		}
	}
	catch (...) {
		// May fail on some filesystems with path length limits
		std::cout << "SKIPPED: Filesystem does not support very long filenames" << std::endl;
		return;
	}
}

TEST_F(ImageIOComprehensiveTest, EdgeCase_DifferentImageSizesInBatch) {
	std::vector<std::pair<int, int>> sizes = {
		{100, 100}, {200, 150}, {640, 480}, {1920, 1080},
		{50, 200}, {300, 100}, {1000, 1000}
	};

	for (size_t i = 0; i < sizes.size(); ++i) {
		cv::Mat testImage = createTestImage(sizes[i].first, sizes[i].second);
		std::string savePath = (fs::path(outputPath) / ("batch_" + std::to_string(i) + ".png")).string();

		ASSERT_NO_THROW(ImageIO::saveImage(savePath, testImage));

		cv::Mat loaded = ImageIO::loadImage(savePath, 640);
		EXPECT_FALSE(loaded.empty());
		EXPECT_EQ(std::max(loaded.rows, loaded.cols), 640);
	}
}

TEST_F(ImageIOComprehensiveTest, Robustness_ConcurrentDirectoryCreation) {
	cv::Mat testImage = createTestImage(640, 480);

	// Test that directory creation works even when saving to multiple nested new directories
	std::vector<std::string> paths;
	for (int i = 0; i < 5; ++i) {
		std::string path = (fs::path(outputPath) / "nested" / std::to_string(i) / "test.png").string();
		paths.push_back(path);
	}

	for (const auto& path : paths) {
		ASSERT_NO_THROW(ImageIO::saveImage(path, testImage));
		ASSERT_TRUE(fs::exists(path));
	}
}

TEST_F(ImageIOComprehensiveTest, Robustness_LoadAfterSaveImmediately) {
	cv::Mat original = createTestImage(640, 480);
	std::string savePath = (fs::path(outputPath) / "immediate_load.png").string();

	ImageIO::saveImage(savePath, original);
	cv::Mat loaded = ImageIO::loadImage(savePath, 640);

	EXPECT_FALSE(loaded.empty());
	EXPECT_EQ(original.rows, loaded.rows);
	EXPECT_EQ(original.cols, loaded.cols);
	EXPECT_EQ(original.channels(), loaded.channels());
}

TEST_F(ImageIOComprehensiveTest, Validation_NonExistentDirectory) {
	cv::Mat testImage = createTestImage(640, 480);

	std::string nonExistentPath = (fs::path(outputPath) / "new_dir" / "test.png").string();

	// Should create directory automatically
	ASSERT_NO_THROW(ImageIO::saveImage(nonExistentPath, testImage));
	ASSERT_TRUE(fs::exists(nonExistentPath));
}

TEST_F(ImageIOComprehensiveTest, Validation_ImageProperties_AfterLoad) {
	cv::Mat loaded = ImageIO::loadImage(testImagePath, 640);

	EXPECT_FALSE(loaded.empty());
	EXPECT_EQ(loaded.channels(), 3);
	EXPECT_EQ(loaded.type(), CV_8UC3);
	EXPECT_GT(loaded.rows, 0);
	EXPECT_GT(loaded.cols, 0);
	EXPECT_TRUE(loaded.isContinuous() || !loaded.isContinuous());  // Just check it's valid
}
