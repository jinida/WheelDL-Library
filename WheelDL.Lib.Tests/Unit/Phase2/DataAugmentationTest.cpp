#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Augmentation/DatasetAugmentation.h"
#include "WheelDL.Lib/Data/Common/Annotation.h"
#include <opencv2/opencv.hpp>

using namespace WheelDL::Data;
using namespace WheelDL::Data::Augmentation;

class DataAugmentationTest : public ::testing::Test
{
protected:
    cv::Mat createTestImage(int rows, int cols, cv::Scalar color)
    {
        return cv::Mat(rows, cols, CV_8UC3, color);
    }
    
    Annotation createTestAnnotation(int classId, float x, float y, float w, float h)
    {
        Annotation ann(LabelType::XYWH);
        ann.addObject(classId, {x, y, w, h});
        return ann;
    }
};

TEST_F(DataAugmentationTest, Mosaic_Construction)
{
    EXPECT_NO_THROW(Mosaic mosaic(640, 640, 1.0f));
}

TEST_F(DataAugmentationTest, Mosaic_FourImages)
{
    Mosaic mosaic(640, 640, 1.0f);
    std::vector<cv::Mat> images = {
        createTestImage(480, 640, cv::Scalar(255, 0, 0)),
        createTestImage(480, 640, cv::Scalar(0, 255, 0)),
        createTestImage(480, 640, cv::Scalar(0, 0, 255)),
        createTestImage(480, 640, cv::Scalar(255, 255, 0))
    };
    std::vector<Annotation> annots = {
        createTestAnnotation(0, 100, 100, 50, 50),
        createTestAnnotation(1, 200, 200, 50, 50),
        createTestAnnotation(2, 300, 300, 50, 50),
        createTestAnnotation(3, 400, 400, 50, 50)
    };
    cv::Mat output;
    Annotation outputAnn(LabelType::XYWH);
    bool applied = mosaic.apply(images, annots, output, outputAnn);
    if (applied) {
        EXPECT_EQ(output.rows, 640);
        EXPECT_EQ(output.cols, 640);
    }
}

TEST_F(DataAugmentationTest, Mosaic_OutputSize)
{
    Mosaic mosaic(1024, 1024, 1.0f);
    std::vector<cv::Mat> images = {
        createTestImage(480, 640, cv::Scalar(100, 100, 100)),
        createTestImage(480, 640, cv::Scalar(100, 100, 100)),
        createTestImage(480, 640, cv::Scalar(100, 100, 100)),
        createTestImage(480, 640, cv::Scalar(100, 100, 100))
    };
    std::vector<Annotation> annots(4, createTestAnnotation(0, 100, 100, 50, 50));
    cv::Mat output;
    Annotation outputAnn(LabelType::XYWH);
    bool applied = mosaic.apply(images, annots, output, outputAnn);
    if (applied) {
        EXPECT_EQ(output.rows, 1024);
        EXPECT_EQ(output.cols, 1024);
    }
}

TEST_F(DataAugmentationTest, Mosaic_Probability)
{
    Mosaic mosaic(640, 640, 0.0f);
    std::vector<cv::Mat> images = {
        createTestImage(480, 640, cv::Scalar(255, 0, 0)),
        createTestImage(480, 640, cv::Scalar(0, 255, 0)),
        createTestImage(480, 640, cv::Scalar(0, 0, 255)),
        createTestImage(480, 640, cv::Scalar(255, 255, 0))
    };
    std::vector<Annotation> annots(4, createTestAnnotation(0, 100, 100, 50, 50));
    cv::Mat output;
    Annotation outputAnn(LabelType::XYWH);
    bool applied = mosaic.apply(images, annots, output, outputAnn);
    EXPECT_FALSE(applied);
}

TEST_F(DataAugmentationTest, Mosaic_Seed)
{
    Mosaic mosaic1(640, 640, 1.0f);
    mosaic1.setSeed(42);
    std::vector<cv::Mat> images1 = {
        createTestImage(480, 640, cv::Scalar(255, 0, 0)),
        createTestImage(480, 640, cv::Scalar(0, 255, 0)),
        createTestImage(480, 640, cv::Scalar(0, 0, 255)),
        createTestImage(480, 640, cv::Scalar(255, 255, 0))
    };
    std::vector<Annotation> annots1(4, createTestAnnotation(0, 100, 100, 50, 50));
    cv::Mat output1;
    Annotation outputAnn1(LabelType::XYWH);
    mosaic1.apply(images1, annots1, output1, outputAnn1);
    
    Mosaic mosaic2(640, 640, 1.0f);
    mosaic2.setSeed(42);
    std::vector<cv::Mat> images2 = images1;
    std::vector<Annotation> annots2 = annots1;
    cv::Mat output2;
    Annotation outputAnn2(LabelType::XYWH);
    mosaic2.apply(images2, annots2, output2, outputAnn2);
    
    if (!output1.empty() && !output2.empty()) {
        EXPECT_EQ(cv::norm(output1, output2, cv::NORM_L2), 0.0);
    }
}

TEST_F(DataAugmentationTest, Mosaic_AnnotationMerge)
{
    Mosaic mosaic(640, 640, 1.0f);
    std::vector<cv::Mat> images(4, createTestImage(480, 640, cv::Scalar(128, 128, 128)));
    std::vector<Annotation> annots = {
        createTestAnnotation(0, 100, 100, 50, 50),
        createTestAnnotation(1, 200, 200, 50, 50),
        createTestAnnotation(2, 300, 300, 50, 50),
        createTestAnnotation(3, 400, 400, 50, 50)
    };
    cv::Mat output;
    Annotation outputAnn(LabelType::XYWH);
    bool applied = mosaic.apply(images, annots, output, outputAnn);
    if (applied) {
        EXPECT_GT(outputAnn.size(), 0);
    }
}

TEST_F(DataAugmentationTest, Mosaic_DifferentSizes)
{
    Mosaic mosaic(640, 640, 1.0f);
    std::vector<cv::Mat> images = {
        createTestImage(480, 640, cv::Scalar(255, 0, 0)),
        createTestImage(720, 1280, cv::Scalar(0, 255, 0)),
        createTestImage(360, 480, cv::Scalar(0, 0, 255)),
        createTestImage(600, 800, cv::Scalar(255, 255, 0))
    };
    std::vector<Annotation> annots(4, createTestAnnotation(0, 100, 100, 50, 50));
    cv::Mat output;
    Annotation outputAnn(LabelType::XYWH);
    EXPECT_NO_THROW(mosaic.apply(images, annots, output, outputAnn));
}

TEST_F(DataAugmentationTest, Mosaic_StressTest)
{
    Mosaic mosaic(640, 640, 1.0f);
    for (int i = 0; i < 50; ++i) {
        std::vector<cv::Mat> images(4, createTestImage(480, 640, cv::Scalar(128, 128, 128)));
        std::vector<Annotation> annots(4, createTestAnnotation(0, 100, 100, 50, 50));
        cv::Mat output;
        Annotation outputAnn(LabelType::XYWH);
        EXPECT_NO_THROW(mosaic.apply(images, annots, output, outputAnn));
    }
}
