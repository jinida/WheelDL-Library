#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Transforms/GeometricTransforms.h"
#include "WheelDL.Lib/Data/Transforms/ColorTransforms.h"
#include "WheelDL.Lib/Data/Common/Annotation.h"
#include <opencv2/opencv.hpp>

using namespace WheelDL::Data;
using namespace WheelDL::Data::Transforms;

class TransformStressTest : public ::testing::Test
{
protected:
	static cv::Mat testImage;
	static Annotation testAnnotation;

	static void SetUpTestSuite()
	{
		testImage = cv::Mat(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
		testAnnotation = Annotation(LabelType::XYWH);
		testAnnotation.addObject(0, { 100,100,50,50 });
	}

	cv::Mat img()
	{
		return testImage.clone();
	}
	Annotation ann()
	{
		return testAnnotation;
	}
};

cv::Mat TransformStressTest::testImage;
Annotation TransformStressTest::testAnnotation;

TEST_F(TransformStressTest, LetterBox640)
{
	LetterBox t(640, 640);
	cv::Mat i = img();
	Annotation a = ann();
	EXPECT_NO_THROW(t.apply(i, a));
}
TEST_F(TransformStressTest, LetterBox1024)
{
	LetterBox t(1024, 1024);
	cv::Mat i = img();
	Annotation a = ann();
	EXPECT_NO_THROW(t.apply(i, a));
}
TEST_F(TransformStressTest, HFlip)
{
	RandomHorizontalFlip t(1.0f);
	cv::Mat i = img();
	Annotation a = ann();
	EXPECT_NO_THROW(t.apply(i, a));
}
TEST_F(TransformStressTest, VFlip)
{
	RandomVerticalFlip t(1.0f);
	cv::Mat i = img();
	Annotation a = ann();
	EXPECT_NO_THROW(t.apply(i, a));
}
TEST_F(TransformStressTest, Perspective)
{
	RandomPerspective t(10, 0.1f, 0.5f, 1, 0.001f, 640, 640);
	cv::Mat i = img();
	Annotation a = ann();
	EXPECT_NO_THROW(t.apply(i, a));
}
TEST_F(TransformStressTest, ColorJitter)
{
	ColorJitter t(0.3f, 0.3f, 0.3f, 0.01f);
	cv::Mat i = img();
	Annotation a = ann();
	EXPECT_NO_THROW(t.apply(i, a));
}
TEST_F(TransformStressTest, Blur)
{
	GaussianBlur t(3, 5, 1.0f);
	cv::Mat i = img();
	Annotation a = ann();
	EXPECT_NO_THROW(t.apply(i, a));
}
TEST_F(TransformStressTest, ToTensor)
{
	ToTensor t(true);
	cv::Mat i = img();
	Annotation a = ann();
	EXPECT_NO_THROW(t.apply(i, a));
}
TEST_F(TransformStressTest, Compose1)
{
	Compose c;
	c.addTransform(std::make_unique<LetterBox>(640, 640));
	cv::Mat i = img();
	Annotation a = ann();
	EXPECT_NO_THROW(c.apply(i, a));
}
TEST_F(TransformStressTest, Compose2)
{
	Compose c;
	c.addTransform(std::make_unique<LetterBox>(640, 640));
	c.addTransform(std::make_unique<RandomHorizontalFlip>(0.5f));
	cv::Mat i = img();
	Annotation a = ann();
	EXPECT_NO_THROW(c.apply(i, a));
}
TEST_F(TransformStressTest, Compose3)
{
	Compose c;
	c.addTransform(std::make_unique<LetterBox>(640, 640));
	c.addTransform(std::make_unique<RandomHorizontalFlip>(0.5f));
	c.addTransform(std::make_unique<ColorJitter>(0.3f, 0.3f, 0.3f, 0.01f));
	cv::Mat i = img();
	Annotation a = ann();
	EXPECT_NO_THROW(c.apply(i, a));
}
TEST_F(TransformStressTest, Many1)
{
	for (int k = 0; k < 10; ++k)
	{
		LetterBox t(640, 640);
		cv::Mat i = img();
		Annotation a = ann();
		t.apply(i, a);
	}
	EXPECT_TRUE(true);
}
TEST_F(TransformStressTest, Many2)
{
	for (int k = 0; k < 10; ++k)
	{
		RandomHorizontalFlip t(0.5f);
		cv::Mat i = img();
		Annotation a = ann();
		t.apply(i, a);
	}
	EXPECT_TRUE(true);
}
TEST_F(TransformStressTest, Many3)
{
	for (int k = 0; k < 10; ++k)
	{
		ColorJitter t(0.3f, 0.3f, 0.3f, 0.01f);
		cv::Mat i = img();
		Annotation a = ann();
		t.apply(i, a);
	}
	EXPECT_TRUE(true);
}
TEST_F(TransformStressTest, Seed1)
{
	RandomHorizontalFlip t(1.0f);
	t.setSeed(42);
	cv::Mat i = img();
	Annotation a = ann();
	EXPECT_NO_THROW(t.apply(i, a));
}
TEST_F(TransformStressTest, Seed2)
{
	ColorJitter t(0.3f, 0.3f, 0.3f, 0.01f);
	t.setSeed(42);
	cv::Mat i = img();
	Annotation a = ann();
	EXPECT_NO_THROW(t.apply(i, a));
}
