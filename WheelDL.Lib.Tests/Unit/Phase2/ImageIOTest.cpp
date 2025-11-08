#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Utils/ImageIO.h"
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace WheelDL::Data::Utils;
namespace fs = std::filesystem;

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

class ImageIOTest : public ::testing::Test
{
protected:
	std::string testDataPath;
	std::string testImagePath;
	std::string outputPath;

	void SetUp() override
	{
		auto testData = getTestDataPath();
		testDataPath = (testData / "mnist_sample" / "images").string();
		testImagePath = (testData / "mnist_sample" / "images" / "mnist_000000.png").string();
		outputPath = (testData / "temp").string();

		if (!fs::exists(outputPath)) {
			fs::create_directories(outputPath);
		}
	}

	void TearDown() override
	{
		if (fs::exists(outputPath)) {
			try {
				fs::remove_all(outputPath);
				fs::create_directories(outputPath);
			}
			catch (...) {
				// Ignore cleanup errors
			}
		}
	}
};

TEST_F(ImageIOTest, LoadImage_ValidPath_Success)
{
	ASSERT_TRUE(fs::exists(testImagePath));
	cv::Mat image = ImageIO::loadImage(testImagePath, 640);
	EXPECT_FALSE(image.empty());
	EXPECT_EQ(image.type(), CV_8UC3);
}

TEST_F(ImageIOTest, LoadImage_InvalidPath_ThrowsException)
{
	std::string invalidPath = (fs::path(testDataPath) / "nonexistent.png").string();
	EXPECT_THROW(ImageIO::loadImage(invalidPath, 640), std::runtime_error);
}

TEST_F(ImageIOTest, LoadImage_EmptyPath_ThrowsException)
{
	EXPECT_THROW(ImageIO::loadImage("", 640), std::runtime_error);
}

TEST_F(ImageIOTest, LoadImage_Resize640_CorrectDimensions)
{
	cv::Mat image = ImageIO::loadImage(testImagePath, 640);
	EXPECT_FALSE(image.empty());
	EXPECT_EQ(std::max(image.rows, image.cols), 640);
}

TEST_F(ImageIOTest, LoadImage_Resize1024_CorrectDimensions)
{
	cv::Mat image = ImageIO::loadImage(testImagePath, 1024);
	EXPECT_FALSE(image.empty());
	EXPECT_EQ(std::max(image.rows, image.cols), 1024);
}

TEST_F(ImageIOTest, LoadImage_Resize256_CorrectDimensions)
{
	cv::Mat image = ImageIO::loadImage(testImagePath, 256);
	EXPECT_FALSE(image.empty());
	EXPECT_EQ(std::max(image.rows, image.cols), 256);
}

TEST_F(ImageIOTest, SaveImage_ValidImage_Success)
{
	cv::Mat image = ImageIO::loadImage(testImagePath, 640);
	std::string savePath = (fs::path(outputPath) / "saved_test.png").string();
	EXPECT_NO_THROW(ImageIO::saveImage(savePath, image));
	EXPECT_TRUE(fs::exists(savePath));
}

TEST_F(ImageIOTest, SaveImage_EmptyImage_ThrowsException)
{
	cv::Mat emptyImage;
	std::string savePath = (fs::path(outputPath) / "empty_test.png").string();
	EXPECT_THROW(ImageIO::saveImage(savePath, emptyImage), std::runtime_error);
}

TEST_F(ImageIOTest, SaveAndLoadRoundTrip_PreservesData)
{
	cv::Mat original = ImageIO::loadImage(testImagePath, 640);
	std::string savePath = (fs::path(outputPath) / "roundtrip_test.png").string();
	ImageIO::saveImage(savePath, original);
	cv::Mat loaded = cv::imread(savePath, cv::IMREAD_COLOR);
	ASSERT_FALSE(loaded.empty());
	EXPECT_EQ(original.size(), loaded.size());
	EXPECT_EQ(original.type(), loaded.type());
}

TEST_F(ImageIOTest, LoadImage_ColorChannels_BGRFormat)
{
	cv::Mat image = ImageIO::loadImage(testImagePath, 640);
	EXPECT_EQ(image.channels(), 3);
	EXPECT_EQ(image.type(), CV_8UC3);
}
