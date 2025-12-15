#include "pch.h"
#include "Data/Transforms/GeometricTransforms.h"
#include "Data/Transforms/Transform.h"
#include "Data/Common/Annotation.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <cmath>

using namespace WheelDL::Data;
using namespace WheelDL::Data::Transforms;

// =============================================================================
// Test Fixture
// =============================================================================

class GeometricTransformsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper: Create test image
    cv::Mat createTestImage(int width = 100, int height = 100, int channels = 3) {
        cv::Mat image(height, width, channels == 3 ? CV_8UC3 : CV_8UC1);
        image.setTo(cv::Scalar(128, 128, 128));
        return image;
    }

    // Helper: Create test annotation with XYWH format
    Annotation createTestAnnotation() {
        Annotation ann(LabelType::XYWH);
        ann.addObject(0, {50.0f, 50.0f, 20.0f, 20.0f});  // Center at (50,50), size 20x20
        return ann;
    }

    // Helper: Create annotation with multiple objects
    Annotation createMultiObjectAnnotation() {
        Annotation ann(LabelType::XYWH);
        ann.addObject(0, {25.0f, 25.0f, 10.0f, 10.0f});
        ann.addObject(1, {75.0f, 75.0f, 10.0f, 10.0f});
        return ann;
    }

    // Helper: Compare floats
    bool floatNear(float a, float b, float tol = 1e-3f) {
        return std::abs(a - b) < tol;
    }
};

// =============================================================================
// 8.1 Resize Tests (GEO-001 ~ GEO-010)
// =============================================================================

// GEO-001: Resize constructor
TEST_F(GeometricTransformsTest, Resize_Constructor) {
    Resize resize(640, 480);

    EXPECT_EQ("Resize", resize.getName());
}

// GEO-002: Resize constructor invalid
TEST_F(GeometricTransformsTest, Resize_Constructor_Invalid) {
    EXPECT_THROW(Resize(0, 480), std::invalid_argument);
    EXPECT_THROW(Resize(640, 0), std::invalid_argument);
    EXPECT_THROW(Resize(-1, 480), std::invalid_argument);
    EXPECT_THROW(Resize(640, -1), std::invalid_argument);
}

// GEO-003: Resize apply empty image
TEST_F(GeometricTransformsTest, Resize_Apply_EmptyImage) {
    Resize resize(640, 480);
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    // Should not throw, just return early
    EXPECT_NO_THROW(resize.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// GEO-004: Resize apply keepAspectRatio false (stretch)
TEST_F(GeometricTransformsTest, Resize_Apply_Stretch) {
    Resize resize(200, 100, false);  // targetHeight=200, targetWidth=100
    cv::Mat image = createTestImage(100, 100);
    Annotation ann = createTestAnnotation();

    resize.apply(image, ann);

    EXPECT_EQ(100, image.cols);
    EXPECT_EQ(200, image.rows);
}

// GEO-005: Resize apply keepAspectRatio true (padding)
TEST_F(GeometricTransformsTest, Resize_Apply_KeepAspectRatio) {
    Resize resize(200, 200, true);
    cv::Mat image = createTestImage(100, 50);  // Wide image
    Annotation ann = createTestAnnotation();

    resize.apply(image, ann);

    EXPECT_EQ(200, image.cols);
    EXPECT_EQ(200, image.rows);
}

// GEO-006: Resize annotation scale
TEST_F(GeometricTransformsTest, Resize_AnnotationScale) {
    Resize resize(200, 200, false);  // 2x scale
    cv::Mat image = createTestImage(100, 100);
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {50.0f, 50.0f, 20.0f, 20.0f});

    resize.apply(image, ann);

    const auto& points = ann.getPoints()[0];
    EXPECT_TRUE(floatNear(points[0], 100.0f));  // cx scaled
    EXPECT_TRUE(floatNear(points[1], 100.0f));  // cy scaled
    EXPECT_TRUE(floatNear(points[2], 40.0f));   // w scaled
    EXPECT_TRUE(floatNear(points[3], 40.0f));   // h scaled
}

// GEO-007: Resize annotation translate (with keepAspectRatio)
TEST_F(GeometricTransformsTest, Resize_AnnotationTranslate) {
    Resize resize(200, 200, true);
    cv::Mat image = createTestImage(100, 50);  // 100x50 -> scale 2.0 -> 200x100, offset y=50
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {50.0f, 25.0f, 10.0f, 10.0f});

    resize.apply(image, ann);

    const auto& points = ann.getPoints()[0];
    // After scale: (100, 50), after translate: (100, 100)
    EXPECT_TRUE(floatNear(points[0], 100.0f));  // cx
    EXPECT_TRUE(floatNear(points[1], 100.0f));  // cy (50*2 + 50 offset)
}

// GEO-008: Resize clone
TEST_F(GeometricTransformsTest, Resize_Clone) {
    Resize original(640, 480, true);

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("Resize", cloned->getName());
}

// GEO-009: Resize getName
TEST_F(GeometricTransformsTest, Resize_GetName) {
    Resize resize(640, 480);

    EXPECT_EQ("Resize", resize.getName());
}

// GEO-010: Resize large to small
TEST_F(GeometricTransformsTest, Resize_LargeToSmall) {
    Resize resize(50, 50, false);
    cv::Mat image = createTestImage(200, 200);
    Annotation ann = createTestAnnotation();

    resize.apply(image, ann);

    EXPECT_EQ(50, image.cols);
    EXPECT_EQ(50, image.rows);
}

// =============================================================================
// 8.2 LetterBox Tests (GEO-011 ~ GEO-022)
// =============================================================================

// GEO-011: LetterBox constructor
TEST_F(GeometricTransformsTest, LetterBox_Constructor) {
    LetterBox letterbox(640, 640, 114, 114, 114, true, true);

    EXPECT_EQ("LetterBox", letterbox.getName());
}

// GEO-012: LetterBox constructor invalid
TEST_F(GeometricTransformsTest, LetterBox_Constructor_Invalid) {
    EXPECT_THROW(LetterBox(0, 640), std::invalid_argument);
    EXPECT_THROW(LetterBox(640, 0), std::invalid_argument);
    EXPECT_THROW(LetterBox(-1, 640), std::invalid_argument);
}

// GEO-013: LetterBox apply empty image
TEST_F(GeometricTransformsTest, LetterBox_Apply_EmptyImage) {
    LetterBox letterbox(640, 640);
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(letterbox.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// GEO-014: LetterBox apply scaleUp true
TEST_F(GeometricTransformsTest, LetterBox_Apply_ScaleUpTrue) {
    LetterBox letterbox(200, 200, 0, 0, 0, true, true);  // scaleUp=true
    cv::Mat image = createTestImage(100, 100);
    Annotation ann = createTestAnnotation();

    letterbox.apply(image, ann);

    EXPECT_EQ(200, image.cols);
    EXPECT_EQ(200, image.rows);
}

// GEO-015: LetterBox apply scaleUp false
TEST_F(GeometricTransformsTest, LetterBox_Apply_ScaleUpFalse) {
    LetterBox letterbox(200, 200, 0, 0, 0, true, false);  // scaleUp=false
    cv::Mat image = createTestImage(100, 100);
    Annotation ann = createTestAnnotation();

    letterbox.apply(image, ann);

    // Image should fit within 200x200 but not upscale
    EXPECT_EQ(200, image.cols);
    EXPECT_EQ(200, image.rows);
}

// GEO-016: LetterBox apply center true
TEST_F(GeometricTransformsTest, LetterBox_Apply_CenterTrue) {
    LetterBox letterbox(200, 200, 0, 0, 0, true, true);  // center=true
    cv::Mat image = createTestImage(100, 50);  // Wide image
    Annotation ann = createTestAnnotation();

    letterbox.apply(image, ann);

    // Image should be centered - check that padding exists on top/bottom
    EXPECT_EQ(200, image.cols);
    EXPECT_EQ(200, image.rows);
}

// GEO-017: LetterBox apply center false
TEST_F(GeometricTransformsTest, LetterBox_Apply_CenterFalse) {
    LetterBox letterbox(200, 200, 128, 128, 128, false, true);  // center=false
    cv::Mat image = createTestImage(100, 50);
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {50.0f, 25.0f, 10.0f, 10.0f});

    letterbox.apply(image, ann);

    // With center=false, image should be at top-left
    const auto& points = ann.getPoints()[0];
    // Scale = 2.0, no offset
    EXPECT_TRUE(floatNear(points[0], 100.0f));  // cx * 2
    EXPECT_TRUE(floatNear(points[1], 50.0f));   // cy * 2 (no y offset)
}

// GEO-018: LetterBox fill color
TEST_F(GeometricTransformsTest, LetterBox_FillColor) {
    LetterBox letterbox(200, 200, 255, 128, 64, true, true);  // RGB fill
    cv::Mat image = createTestImage(100, 50);
    Annotation ann = createTestAnnotation();

    letterbox.apply(image, ann);

    // Check padding color (OpenCV uses BGR order)
    cv::Vec3b paddingColor = image.at<cv::Vec3b>(0, 0);  // Top-left should be padding
    EXPECT_EQ(64, paddingColor[0]);   // B
    EXPECT_EQ(128, paddingColor[1]);  // G
    EXPECT_EQ(255, paddingColor[2]);  // R
}

// GEO-019: LetterBox MIN_SCALE (min 10%)
TEST_F(GeometricTransformsTest, LetterBox_MinScale) {
    // Test that MIN_SCALE (0.1) is applied
    // 100x100 target with 500x500 image -> scale = 0.2, which is above MIN_SCALE
    LetterBox letterbox(100, 100, 0, 0, 0, true, true);
    cv::Mat image = createTestImage(500, 500);
    Annotation ann = createTestAnnotation();

    letterbox.apply(image, ann);

    // Should produce valid output
    EXPECT_EQ(100, image.cols);
    EXPECT_EQ(100, image.rows);
}

// GEO-020: LetterBox min dimension
TEST_F(GeometricTransformsTest, LetterBox_MinDimension) {
    LetterBox letterbox(100, 100, 0, 0, 0, true, false);  // scaleUp=false
    cv::Mat image = createTestImage(10, 10);  // Small image
    Annotation ann = createTestAnnotation();

    letterbox.apply(image, ann);

    // Should produce valid output
    EXPECT_EQ(100, image.cols);
    EXPECT_EQ(100, image.rows);
}

// GEO-021: LetterBox clone
TEST_F(GeometricTransformsTest, LetterBox_Clone) {
    LetterBox original(640, 480, 114, 114, 114, true, false);

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("LetterBox", cloned->getName());
}

// GEO-022: LetterBox annotation update
TEST_F(GeometricTransformsTest, LetterBox_AnnotationUpdate) {
    LetterBox letterbox(200, 200, 0, 0, 0, true, true);
    cv::Mat image = createTestImage(100, 100);
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {50.0f, 50.0f, 20.0f, 20.0f});

    letterbox.apply(image, ann);

    const auto& points = ann.getPoints()[0];
    // Scale = 2.0, center offset = 0 for square image
    EXPECT_TRUE(floatNear(points[0], 100.0f));  // cx * 2
    EXPECT_TRUE(floatNear(points[1], 100.0f));  // cy * 2
    EXPECT_TRUE(floatNear(points[2], 40.0f));   // w * 2
    EXPECT_TRUE(floatNear(points[3], 40.0f));   // h * 2
}

// =============================================================================
// 8.3 RandomHorizontalFlip Tests (GEO-023 ~ GEO-030)
// =============================================================================

// GEO-023: RandomHorizontalFlip constructor
TEST_F(GeometricTransformsTest, RandomHorizontalFlip_Constructor) {
    RandomHorizontalFlip flip(0.5f);

    EXPECT_EQ("RandomHorizontalFlip", flip.getName());
    EXPECT_TRUE(flip.isRandom());
}

// GEO-024: RandomHorizontalFlip invalid prob
TEST_F(GeometricTransformsTest, RandomHorizontalFlip_InvalidProbability) {
    EXPECT_THROW(RandomHorizontalFlip(-0.1f), std::invalid_argument);
    EXPECT_THROW(RandomHorizontalFlip(1.1f), std::invalid_argument);
}

// GEO-025: RandomHorizontalFlip apply empty
TEST_F(GeometricTransformsTest, RandomHorizontalFlip_Apply_Empty) {
    RandomHorizontalFlip flip(1.0f);
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(flip.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// GEO-026: RandomHorizontalFlip apply skip (probability 0)
TEST_F(GeometricTransformsTest, RandomHorizontalFlip_Apply_Skip) {
    RandomHorizontalFlip flip(0.0f);  // Never flip
    cv::Mat image = createTestImage(100, 100);
    cv::Mat originalImage = image.clone();
    Annotation ann = createTestAnnotation();

    flip.apply(image, ann);

    // Image should be unchanged
    cv::Mat diff;
    cv::absdiff(image, originalImage, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// GEO-027: RandomHorizontalFlip apply flip
TEST_F(GeometricTransformsTest, RandomHorizontalFlip_Apply_Flip) {
    RandomHorizontalFlip flip(1.0f);  // Always flip
    cv::Mat image = createTestImage(100, 100);

    // Draw distinctive pattern on left side
    cv::rectangle(image, cv::Point(0, 0), cv::Point(10, 100), cv::Scalar(255, 0, 0), -1);

    Annotation ann = createTestAnnotation();

    flip.apply(image, ann);

    // Pattern should now be on right side
    cv::Vec3b leftPixel = image.at<cv::Vec3b>(50, 5);
    cv::Vec3b rightPixel = image.at<cv::Vec3b>(50, 95);

    EXPECT_NE(255, leftPixel[0]);  // Left no longer has blue
    EXPECT_EQ(255, rightPixel[0]); // Right now has blue
}

// GEO-028: RandomHorizontalFlip annotation flip
TEST_F(GeometricTransformsTest, RandomHorizontalFlip_AnnotationFlip) {
    RandomHorizontalFlip flip(1.0f);
    cv::Mat image = createTestImage(100, 100);
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {25.0f, 50.0f, 10.0f, 10.0f});  // Left side

    flip.apply(image, ann);

    const auto& points = ann.getPoints()[0];
    // cx should be flipped: 100 - 25 = 75
    EXPECT_TRUE(floatNear(points[0], 75.0f));
    EXPECT_TRUE(floatNear(points[1], 50.0f));  // cy unchanged
}

// GEO-029: RandomHorizontalFlip clone
TEST_F(GeometricTransformsTest, RandomHorizontalFlip_Clone) {
    RandomHorizontalFlip original(0.7f);

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("RandomHorizontalFlip", cloned->getName());
}

// GEO-030: RandomHorizontalFlip reproducibility
TEST_F(GeometricTransformsTest, RandomHorizontalFlip_Reproducibility) {
    RandomHorizontalFlip flip1(0.5f);
    RandomHorizontalFlip flip2(0.5f);

    flip1.setSeed(42);
    flip2.setSeed(42);

    cv::Mat image1 = createTestImage(100, 100);
    cv::Mat image2 = createTestImage(100, 100);
    Annotation ann1 = createTestAnnotation();
    Annotation ann2 = createTestAnnotation();

    flip1.apply(image1, ann1);
    flip2.apply(image2, ann2);

    // Same seed should produce same result
    cv::Mat diff;
    cv::absdiff(image1, image2, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// =============================================================================
// 8.4 RandomVerticalFlip Tests (GEO-031 ~ GEO-038)
// =============================================================================

// GEO-031: RandomVerticalFlip constructor
TEST_F(GeometricTransformsTest, RandomVerticalFlip_Constructor) {
    RandomVerticalFlip flip(0.5f);

    EXPECT_EQ("RandomVerticalFlip", flip.getName());
    EXPECT_TRUE(flip.isRandom());
}

// GEO-032: RandomVerticalFlip invalid prob
TEST_F(GeometricTransformsTest, RandomVerticalFlip_InvalidProbability) {
    EXPECT_THROW(RandomVerticalFlip(-0.1f), std::invalid_argument);
    EXPECT_THROW(RandomVerticalFlip(1.1f), std::invalid_argument);
}

// GEO-033: RandomVerticalFlip apply empty
TEST_F(GeometricTransformsTest, RandomVerticalFlip_Apply_Empty) {
    RandomVerticalFlip flip(1.0f);
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(flip.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// GEO-034: RandomVerticalFlip apply skip (probability 0)
TEST_F(GeometricTransformsTest, RandomVerticalFlip_Apply_Skip) {
    RandomVerticalFlip flip(0.0f);  // Never flip
    cv::Mat image = createTestImage(100, 100);
    cv::Mat originalImage = image.clone();
    Annotation ann = createTestAnnotation();

    flip.apply(image, ann);

    // Image should be unchanged
    cv::Mat diff;
    cv::absdiff(image, originalImage, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// GEO-035: RandomVerticalFlip apply flip
TEST_F(GeometricTransformsTest, RandomVerticalFlip_Apply_Flip) {
    RandomVerticalFlip flip(1.0f);  // Always flip
    cv::Mat image = createTestImage(100, 100);

    // Draw distinctive pattern on top
    cv::rectangle(image, cv::Point(0, 0), cv::Point(100, 10), cv::Scalar(0, 255, 0), -1);

    Annotation ann = createTestAnnotation();

    flip.apply(image, ann);

    // Pattern should now be on bottom
    cv::Vec3b topPixel = image.at<cv::Vec3b>(5, 50);
    cv::Vec3b bottomPixel = image.at<cv::Vec3b>(95, 50);

    EXPECT_NE(255, topPixel[1]);     // Top no longer has green
    EXPECT_EQ(255, bottomPixel[1]);  // Bottom now has green
}

// GEO-036: RandomVerticalFlip annotation flip
TEST_F(GeometricTransformsTest, RandomVerticalFlip_AnnotationFlip) {
    RandomVerticalFlip flip(1.0f);
    cv::Mat image = createTestImage(100, 100);
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {50.0f, 25.0f, 10.0f, 10.0f});  // Top side

    flip.apply(image, ann);

    const auto& points = ann.getPoints()[0];
    // cy should be flipped: 100 - 25 = 75
    EXPECT_TRUE(floatNear(points[0], 50.0f));  // cx unchanged
    EXPECT_TRUE(floatNear(points[1], 75.0f));
}

// GEO-037: RandomVerticalFlip clone
TEST_F(GeometricTransformsTest, RandomVerticalFlip_Clone) {
    RandomVerticalFlip original(0.7f);

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("RandomVerticalFlip", cloned->getName());
}

// GEO-038: RandomVerticalFlip reproducibility
TEST_F(GeometricTransformsTest, RandomVerticalFlip_Reproducibility) {
    RandomVerticalFlip flip1(0.5f);
    RandomVerticalFlip flip2(0.5f);

    flip1.setSeed(42);
    flip2.setSeed(42);

    cv::Mat image1 = createTestImage(100, 100);
    cv::Mat image2 = createTestImage(100, 100);
    Annotation ann1 = createTestAnnotation();
    Annotation ann2 = createTestAnnotation();

    flip1.apply(image1, ann1);
    flip2.apply(image2, ann2);

    // Same seed should produce same result
    cv::Mat diff;
    cv::absdiff(image1, image2, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// =============================================================================
// 8.5 RandomPerspective Tests (GEO-039 ~ GEO-054)
// =============================================================================

// GEO-039: RandomPerspective constructor
TEST_F(GeometricTransformsTest, RandomPerspective_Constructor) {
    RandomPerspective perspective(10.0f, 0.1f, 0.1f, 10.0f, 0.0001f,
                                  640, 640, 0.5f, 114, 114, 114, 5.0f);

    EXPECT_EQ("RandomPerspective", perspective.getName());
    EXPECT_TRUE(perspective.isRandom());
}

// GEO-040: RandomPerspective invalid prob
TEST_F(GeometricTransformsTest, RandomPerspective_InvalidProbability) {
    EXPECT_THROW(RandomPerspective(0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                   640, 640, -0.1f), std::invalid_argument);
    EXPECT_THROW(RandomPerspective(0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                   640, 640, 1.1f), std::invalid_argument);
}

// GEO-041: RandomPerspective apply empty
TEST_F(GeometricTransformsTest, RandomPerspective_Apply_Empty) {
    RandomPerspective perspective(10.0f, 0.1f, 0.1f, 10.0f, 0.0f,
                                  640, 640, 1.0f);
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(perspective.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// GEO-042: RandomPerspective apply skip (probability 0)
TEST_F(GeometricTransformsTest, RandomPerspective_Apply_Skip) {
    RandomPerspective perspective(10.0f, 0.1f, 0.1f, 10.0f, 0.0f,
                                  100, 100, 0.0f);  // probability = 0
    cv::Mat image = createTestImage(100, 100);
    cv::Mat originalImage = image.clone();
    Annotation ann = createTestAnnotation();

    perspective.apply(image, ann);

    // Image should be unchanged
    cv::Mat diff;
    cv::absdiff(image, originalImage, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// GEO-043: RandomPerspective apply success
TEST_F(GeometricTransformsTest, RandomPerspective_Apply_Success) {
    RandomPerspective perspective(10.0f, 0.1f, 0.1f, 10.0f, 0.0001f,
                                  100, 100, 1.0f);
    cv::Mat image = createTestImage(100, 100);
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(perspective.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// GEO-044: RandomPerspective MAX_PIXELS check
TEST_F(GeometricTransformsTest, RandomPerspective_MaxPixels) {
    // Create perspective with huge target size
    RandomPerspective perspective(0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  50000, 50000, 1.0f);  // 2.5 billion pixels
    cv::Mat image = createTestImage(100, 100);
    Annotation ann = createTestAnnotation();

    EXPECT_THROW(perspective.apply(image, ann), std::runtime_error);
}

// GEO-045: RandomPerspective rotation
TEST_F(GeometricTransformsTest, RandomPerspective_Rotation) {
    RandomPerspective perspective(45.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  100, 100, 1.0f);  // Only rotation
    perspective.setSeed(42);

    cv::Mat image = createTestImage(100, 100);
    cv::rectangle(image, cv::Point(40, 40), cv::Point(60, 60), cv::Scalar(255, 0, 0), -1);
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(perspective.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// GEO-046: RandomPerspective scale
TEST_F(GeometricTransformsTest, RandomPerspective_Scale) {
    RandomPerspective perspective(0.0f, 0.0f, 0.3f, 0.0f, 0.0f,
                                  100, 100, 1.0f);  // Only scale
    perspective.setSeed(42);

    cv::Mat image = createTestImage(100, 100);
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(perspective.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// GEO-047: RandomPerspective shear
TEST_F(GeometricTransformsTest, RandomPerspective_Shear) {
    RandomPerspective perspective(0.0f, 0.0f, 0.0f, 20.0f, 0.0f,
                                  100, 100, 1.0f);  // Only shear
    perspective.setSeed(42);

    cv::Mat image = createTestImage(100, 100);
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(perspective.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// GEO-048: RandomPerspective perspective
TEST_F(GeometricTransformsTest, RandomPerspective_PerspectiveParam) {
    RandomPerspective perspective(0.0f, 0.0f, 0.0f, 0.0f, 0.001f,
                                  100, 100, 1.0f);  // Only perspective
    perspective.setSeed(42);

    cv::Mat image = createTestImage(100, 100);
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(perspective.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// GEO-049: RandomPerspective translate
TEST_F(GeometricTransformsTest, RandomPerspective_Translate) {
    RandomPerspective perspective(0.0f, 0.2f, 0.0f, 0.0f, 0.0f,
                                  100, 100, 1.0f);  // Only translate
    perspective.setSeed(42);

    cv::Mat image = createTestImage(100, 100);
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(perspective.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// GEO-050: RandomPerspective centering (needsCentering condition)
TEST_F(GeometricTransformsTest, RandomPerspective_Centering) {
    // All params > 0 should trigger centering
    RandomPerspective perspective(10.0f, 0.1f, 0.1f, 10.0f, 0.0f,
                                  100, 100, 1.0f);
    perspective.setSeed(42);

    cv::Mat image = createTestImage(100, 100);
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(perspective.apply(image, ann));
}

// GEO-051: RandomPerspective MIN_SCALE
TEST_F(GeometricTransformsTest, RandomPerspective_MinScale) {
    RandomPerspective perspective(0.0f, 0.0f, 0.95f, 0.0f, 0.0f,
                                  100, 100, 1.0f);  // scale could go to 0.05
    perspective.setSeed(12345);

    cv::Mat image = createTestImage(100, 100);
    Annotation ann = createTestAnnotation();

    // Should not crash due to MIN_SCALE protection
    EXPECT_NO_THROW(perspective.apply(image, ann));
}

// GEO-052: RandomPerspective clone
TEST_F(GeometricTransformsTest, RandomPerspective_Clone) {
    RandomPerspective original(10.0f, 0.1f, 0.1f, 10.0f, 0.0001f,
                               640, 640, 0.5f, 114, 114, 114, 5.0f);

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("RandomPerspective", cloned->getName());
}

// GEO-053: RandomPerspective annotation.transform
TEST_F(GeometricTransformsTest, RandomPerspective_AnnotationTransform) {
    RandomPerspective perspective(10.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  100, 100, 1.0f);
    perspective.setSeed(42);

    cv::Mat image = createTestImage(100, 100);
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {50.0f, 50.0f, 20.0f, 20.0f});

    perspective.apply(image, ann);

    // Annotation should be updated (transformed)
    EXPECT_EQ(1, ann.size());
}

// GEO-054: RandomPerspective reproducibility
TEST_F(GeometricTransformsTest, RandomPerspective_Reproducibility) {
    RandomPerspective perspective1(10.0f, 0.1f, 0.1f, 10.0f, 0.0f,
                                   100, 100, 1.0f);
    RandomPerspective perspective2(10.0f, 0.1f, 0.1f, 10.0f, 0.0f,
                                   100, 100, 1.0f);

    perspective1.setSeed(42);
    perspective2.setSeed(42);

    cv::Mat image1 = createTestImage(100, 100);
    cv::Mat image2 = createTestImage(100, 100);
    Annotation ann1 = createTestAnnotation();
    Annotation ann2 = createTestAnnotation();

    perspective1.apply(image1, ann1);
    perspective2.apply(image2, ann2);

    // Same seed should produce same result
    cv::Mat diff;
    cv::absdiff(image1, image2, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// =============================================================================
// 8.6 EfficientADTransform Tests (GEO-055 ~ GEO-060)
// =============================================================================

// GEO-055: EfficientADTransform constructor
TEST_F(GeometricTransformsTest, EfficientADTransform_Constructor) {
    auto branchA = std::make_unique<Compose>();
    auto branchB = std::make_unique<Compose>();

    branchA->addTransform(std::make_unique<Resize>(100, 100));
    branchB->addTransform(std::make_unique<Resize>(100, 100));

    EXPECT_NO_THROW({
        EfficientADTransform transform(std::move(branchA), std::move(branchB));
        EXPECT_EQ("EfficientADTransform", transform.getName());
    });
}

// GEO-056: EfficientADTransform null branch
TEST_F(GeometricTransformsTest, EfficientADTransform_NullBranch) {
    auto branchA = std::make_unique<Compose>();
    branchA->addTransform(std::make_unique<Resize>(100, 100));

    EXPECT_THROW({
        EfficientADTransform transform(std::move(branchA), nullptr);
    }, std::invalid_argument);

    auto branchB = std::make_unique<Compose>();
    branchB->addTransform(std::make_unique<Resize>(100, 100));

    EXPECT_THROW({
        EfficientADTransform transform(nullptr, std::move(branchB));
    }, std::invalid_argument);
}

// GEO-057: EfficientADTransform apply empty
TEST_F(GeometricTransformsTest, EfficientADTransform_Apply_Empty) {
    auto branchA = std::make_unique<Compose>();
    auto branchB = std::make_unique<Compose>();

    branchA->addTransform(std::make_unique<Resize>(100, 100));
    branchB->addTransform(std::make_unique<Resize>(100, 100));

    EfficientADTransform transform(std::move(branchA), std::move(branchB));

    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_THROW(transform.apply(emptyImage, ann), std::runtime_error);
}

// GEO-058: EfficientADTransform apply success (6-channel output)
TEST_F(GeometricTransformsTest, EfficientADTransform_Apply_Success) {
    auto branchA = std::make_unique<Compose>();
    auto branchB = std::make_unique<Compose>();

    branchA->addTransform(std::make_unique<Resize>(100, 100));
    branchB->addTransform(std::make_unique<Resize>(100, 100));

    EfficientADTransform transform(std::move(branchA), std::move(branchB));

    cv::Mat image = createTestImage(100, 100, 3);  // 3-channel input
    Annotation ann = createTestAnnotation();

    transform.apply(image, ann);

    // Output should be 6-channel (3 from branchA + 3 from branchB)
    EXPECT_EQ(6, image.channels());
    EXPECT_EQ(100, image.cols);
    EXPECT_EQ(100, image.rows);
}

// GEO-059: EfficientADTransform clone
TEST_F(GeometricTransformsTest, EfficientADTransform_Clone) {
    auto branchA = std::make_unique<Compose>();
    auto branchB = std::make_unique<Compose>();

    branchA->addTransform(std::make_unique<Resize>(100, 100));
    branchB->addTransform(std::make_unique<Resize>(100, 100));

    EfficientADTransform original(std::move(branchA), std::move(branchB));

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("EfficientADTransform", cloned->getName());
}

// GEO-060: EfficientADTransform getName
TEST_F(GeometricTransformsTest, EfficientADTransform_GetName) {
    auto branchA = std::make_unique<Compose>();
    auto branchB = std::make_unique<Compose>();

    branchA->addTransform(std::make_unique<Resize>(100, 100));
    branchB->addTransform(std::make_unique<Resize>(100, 100));

    EfficientADTransform transform(std::move(branchA), std::move(branchB));

    EXPECT_EQ("EfficientADTransform", transform.getName());
}
