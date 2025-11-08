#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Utils/ImageIO.h"
#include <opencv2/opencv.hpp>
#include <filesystem>

using WheelDL::Data::Utils::ImageIO;
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

class ImageIOTestExtended : public ::testing::Test
{
protected:
	fs::path testDataPath;
	fs::path testImagePath;
	fs::path outputPath;

	void SetUp() override
	{
		testDataPath = getTestDataPath() / "mnist_sample" / "images";
		testImagePath = testDataPath / "mnist_000000.png";
		outputPath = getTestDataPath() / "temp";
		if (!fs::exists(outputPath)) fs::create_directories(outputPath);
	}
};

TEST_F(ImageIOTestExtended, LoadImage_Size32)
{
	cv::Mat img = ImageIO::loadImage(testImagePath.string(), 32); EXPECT_EQ(std::max(img.rows, img.cols), 32);
}
TEST_F(ImageIOTestExtended, LoadImage_Size64)
{
	cv::Mat img = ImageIO::loadImage(testImagePath.string(), 64); EXPECT_EQ(std::max(img.rows, img.cols), 64);
}
TEST_F(ImageIOTestExtended, LoadImage_Size128)
{
	cv::Mat img = ImageIO::loadImage(testImagePath.string(), 128); EXPECT_EQ(std::max(img.rows, img.cols), 128);
}
TEST_F(ImageIOTestExtended, LoadImage_Size256)
{
	cv::Mat img = ImageIO::loadImage(testImagePath.string(), 256); EXPECT_EQ(std::max(img.rows, img.cols), 256);
}
TEST_F(ImageIOTestExtended, LoadImage_Size512)
{
	cv::Mat img = ImageIO::loadImage(testImagePath.string(), 512); EXPECT_EQ(std::max(img.rows, img.cols), 512);
}
TEST_F(ImageIOTestExtended, LoadImage_Size1024)
{
	cv::Mat img = ImageIO::loadImage(testImagePath.string(), 1024); EXPECT_EQ(std::max(img.rows, img.cols), 1024);
}
TEST_F(ImageIOTestExtended, LoadImage_Size2048)
{
	cv::Mat img = ImageIO::loadImage(testImagePath.string(), 2048); EXPECT_EQ(std::max(img.rows, img.cols), 2048);
}
TEST_F(ImageIOTestExtended, LoadImage_Image1)
{
	cv::Mat img = ImageIO::loadImage((testDataPath / "mnist_000001.png").string(), 640); EXPECT_FALSE(img.empty());
}
TEST_F(ImageIOTestExtended, LoadImage_Image2)
{
	cv::Mat img = ImageIO::loadImage((testDataPath / "mnist_000002.png").string(), 640); EXPECT_FALSE(img.empty());
}
TEST_F(ImageIOTestExtended, LoadImage_Image3)
{
	cv::Mat img = ImageIO::loadImage((testDataPath / "mnist_000003.png").string(), 640); EXPECT_FALSE(img.empty());
}
TEST_F(ImageIOTestExtended, LoadImage_Image4)
{
	cv::Mat img = ImageIO::loadImage((testDataPath / "mnist_000004.png").string(), 640); EXPECT_FALSE(img.empty());
}
TEST_F(ImageIOTestExtended, LoadImage_Image5)
{
	cv::Mat img = ImageIO::loadImage((testDataPath / "mnist_000005.png").string(), 640); EXPECT_FALSE(img.empty());
}
TEST_F(ImageIOTestExtended, LoadImage_Image6)
{
	cv::Mat img = ImageIO::loadImage((testDataPath / "mnist_000006.png").string(), 640); EXPECT_FALSE(img.empty());
}
TEST_F(ImageIOTestExtended, LoadImage_Image7)
{
	cv::Mat img = ImageIO::loadImage((testDataPath / "mnist_000007.png").string(), 640); EXPECT_FALSE(img.empty());
}
TEST_F(ImageIOTestExtended, LoadImage_Image8)
{
	cv::Mat img = ImageIO::loadImage((testDataPath / "mnist_000008.png").string(), 640); EXPECT_FALSE(img.empty());
}
TEST_F(ImageIOTestExtended, LoadImage_Image9)
{
	cv::Mat img = ImageIO::loadImage((testDataPath / "mnist_000009.png").string(), 640); EXPECT_FALSE(img.empty());
}
TEST_F(ImageIOTestExtended, SaveLoad_PNG)
{
	cv::Mat img = ImageIO::loadImage(testImagePath.string(), 640); ImageIO::saveImage((outputPath / "test.png").string(), img); EXPECT_TRUE(fs::exists(outputPath / "test.png"));
}
TEST_F(ImageIOTestExtended, SaveLoad_JPG)
{
	cv::Mat img = ImageIO::loadImage(testImagePath.string(), 640); ImageIO::saveImage((outputPath / "test.jpg").string(), img); EXPECT_TRUE(fs::exists(outputPath / "test.jpg"));
}
TEST_F(ImageIOTestExtended, SaveLoad_BMP)
{
	cv::Mat img = ImageIO::loadImage(testImagePath.string(), 640); ImageIO::saveImage((outputPath / "test.bmp").string(), img); EXPECT_TRUE(fs::exists(outputPath / "test.bmp"));
}
TEST_F(ImageIOTestExtended, Channels_3)
{
	cv::Mat img = ImageIO::loadImage(testImagePath.string(), 640); EXPECT_EQ(img.channels(), 3);
}
