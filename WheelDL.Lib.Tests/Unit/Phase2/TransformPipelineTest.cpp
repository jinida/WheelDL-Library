#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Transforms/Transform.h"
#include "WheelDL.Lib/Data/Transforms/GeometricTransforms.h"
#include "WheelDL.Lib/Data/Transforms/ColorTransforms.h"
#include "WheelDL.Lib/Data/Common/Annotation.h"
#include <opencv2/opencv.hpp>

using namespace WheelDL::Data;
using namespace WheelDL::Data::Transforms;

class TransformPipelineTest : public ::testing::Test
{
protected:
	cv::Mat createTestImage(int rows, int cols)
	{
		return cv::Mat(rows, cols, CV_8UC3, cv::Scalar(128, 128, 128));
	}

	Annotation createTestAnnotation()
	{
		Annotation ann(LabelType::XYWH);
		ann.addObject(0, { 100.0f, 100.0f, 50.0f, 50.0f });
		return ann;
	}
};

TEST_F(TransformPipelineTest, LetterBox_Square)
{
	LetterBox transform(640, 640);
	cv::Mat image = createTestImage(480, 640);
	Annotation ann = createTestAnnotation();
	transform.apply(image, ann);
	EXPECT_EQ(image.rows, 640);
	EXPECT_EQ(image.cols, 640);
}

TEST_F(TransformPipelineTest, LetterBox_Rect)
{
	LetterBox transform(640, 480);
	cv::Mat image = createTestImage(480, 640);
	Annotation ann = createTestAnnotation();
	transform.apply(image, ann);
	EXPECT_TRUE(image.rows <= 640 && image.cols <= 480);
}

TEST_F(TransformPipelineTest, RandomHorizontalFlip_Apply)
{
	RandomHorizontalFlip transform(1.0f);
	cv::Mat image = createTestImage(100, 100);
	Annotation ann = createTestAnnotation();
	EXPECT_NO_THROW(transform.apply(image, ann));
}

TEST_F(TransformPipelineTest, RandomVerticalFlip_Apply)
{
	RandomVerticalFlip transform(1.0f);
	cv::Mat image = createTestImage(100, 100);
	Annotation ann = createTestAnnotation();
	EXPECT_NO_THROW(transform.apply(image, ann));
}

TEST_F(TransformPipelineTest, RandomPerspective_Apply)
{
	RandomPerspective transform(10.0f, 0.1f, 0.5f, 2.0f, 0.001f, 640, 640);
	cv::Mat image = createTestImage(480, 640);
	Annotation ann = createTestAnnotation();
	EXPECT_NO_THROW(transform.apply(image, ann));
}

TEST_F(TransformPipelineTest, ColorJitter_Apply)
{
	ColorJitter transform(0.3f, 0.3f, 0.3f, 0.01f);
	cv::Mat image = createTestImage(100, 100);
	Annotation ann = createTestAnnotation();
	EXPECT_NO_THROW(transform.apply(image, ann));
}

TEST_F(TransformPipelineTest, GaussianBlur_Apply)
{
	GaussianBlur transform(3, 5, 1.0f);
	cv::Mat image = createTestImage(100, 100);
	Annotation ann = createTestAnnotation();
	EXPECT_NO_THROW(transform.apply(image, ann));
}

TEST_F(TransformPipelineTest, ToTensor_Apply)
{
	ToTensor transform(true);
	cv::Mat image = createTestImage(100, 100);
	Annotation ann = createTestAnnotation();
	EXPECT_NO_THROW(transform.apply(image, ann));
}

TEST_F(TransformPipelineTest, Compose_Empty)
{
	Compose compose;
	cv::Mat image = createTestImage(100, 100);
	Annotation ann = createTestAnnotation();
	EXPECT_NO_THROW(compose.apply(image, ann));
}

TEST_F(TransformPipelineTest, Compose_Single)
{
	Compose compose;
	compose.addTransform(std::make_unique<LetterBox>(640, 640));
	cv::Mat image = createTestImage(480, 640);
	Annotation ann = createTestAnnotation();
	compose.apply(image, ann);
	EXPECT_EQ(image.rows, 640);
}

TEST_F(TransformPipelineTest, Compose_Multiple)
{
	Compose compose;
	compose.addTransform(std::make_unique<LetterBox>(640, 640));
	compose.addTransform(std::make_unique<RandomHorizontalFlip>(0.5f));
	compose.addTransform(std::make_unique<ColorJitter>(0.3f, 0.3f, 0.3f, 0.01f));
	cv::Mat image = createTestImage(480, 640);
	Annotation ann = createTestAnnotation();
	EXPECT_NO_THROW(compose.apply(image, ann));
}

TEST_F(TransformPipelineTest, Compose_FullPipeline)
{
	Compose compose;
	compose.addTransform(std::make_unique<RandomPerspective>(10.0f, 0.1f, 0.5f, 1.0f, 0.001f, 640, 640));
	compose.addTransform(std::make_unique<RandomHorizontalFlip>(0.5f));
	compose.addTransform(std::make_unique<ColorJitter>(0.3f, 0.3f, 0.3f, 0.01f));
	compose.addTransform(std::make_unique<GaussianBlur>(3, 5, 0.5f));
	compose.addTransform(std::make_unique<ToTensor>(true));
	cv::Mat image = createTestImage(480, 640);
	Annotation ann = createTestAnnotation();
	EXPECT_NO_THROW(compose.apply(image, ann));
}

TEST_F(TransformPipelineTest, Seed_Reproducibility)
{
	RandomHorizontalFlip transform(1.0f);
	transform.setSeed(42);
	cv::Mat img1 = createTestImage(100, 100);
	cv::Mat img2 = img1.clone();
	Annotation ann1 = createTestAnnotation();
	Annotation ann2 = createTestAnnotation();
	transform.apply(img1, ann1);
	transform.setSeed(42);
	transform.apply(img2, ann2);
	EXPECT_EQ(cv::norm(img1, img2, cv::NORM_L2), 0.0);
}

TEST_F(TransformPipelineTest, LetterBox_CreatesSquareOutput)
{
	LetterBox transform(640, 640, 114, 114, 114, true, true);
	cv::Mat image = createTestImage(480, 640);
	Annotation ann = createTestAnnotation();
	transform.apply(image, ann);
	// LetterBox should create square output (640x640)
	EXPECT_EQ(image.rows, 640);
	EXPECT_EQ(image.cols, 640);
}

TEST_F(TransformPipelineTest, ColorJitter_ModifiesImage)
{
	ColorJitter transform(0.5f, 0.5f, 0.5f, 0.05f);
	cv::Mat original = createTestImage(100, 100);
	cv::Mat modified = original.clone();
	Annotation ann = createTestAnnotation();
	transform.apply(modified, ann);
	double diff = cv::norm(original, modified, cv::NORM_L2);
	EXPECT_GT(diff, 0.0);
}

TEST_F(TransformPipelineTest, GaussianBlur_Probability)
{
	GaussianBlur transform(3, 5, 0.0f);
	cv::Mat original = createTestImage(100, 100);
	cv::Mat noBlur = original.clone();
	Annotation ann = createTestAnnotation();
	transform.apply(noBlur, ann);
	EXPECT_EQ(cv::norm(original, noBlur, cv::NORM_L2), 0.0);
}

TEST_F(TransformPipelineTest, StressTest_ManyTransforms)
{
	for (int i = 0; i < 100; ++i)
	{
		Compose compose;
		compose.addTransform(std::make_unique<LetterBox>(640, 640));
		compose.addTransform(std::make_unique<RandomHorizontalFlip>(0.5f));
		cv::Mat image = createTestImage(480, 640);
		Annotation ann = createTestAnnotation();
		EXPECT_NO_THROW(compose.apply(image, ann));
	}
}
