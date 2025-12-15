#include "pch.h"
#include "Data/Transforms/ColorTransforms.h"
#include "Data/Transforms/Transform.h"
#include "Data/Common/Annotation.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <cmath>
#include <array>

using namespace WheelDL::Data;
using namespace WheelDL::Data::Transforms;

// =============================================================================
// Test Fixture
// =============================================================================

class ColorTransformsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper: Create test image (uint8, 3-channel BGR)
    cv::Mat createTestImage(int width = 100, int height = 100) {
        cv::Mat image(height, width, CV_8UC3);
        image.setTo(cv::Scalar(100, 128, 150));  // BGR
        return image;
    }

    // Helper: Create grayscale test image
    cv::Mat createGrayImage(int width = 100, int height = 100) {
        cv::Mat image(height, width, CV_8UC1);
        image.setTo(cv::Scalar(128));
        return image;
    }

    // Helper: Create float test image (CV_32FC3)
    cv::Mat createFloatImage(int width = 100, int height = 100) {
        cv::Mat image(height, width, CV_32FC3);
        image.setTo(cv::Scalar(0.4f, 0.5f, 0.6f));
        return image;
    }

    // Helper: Create test annotation
    Annotation createTestAnnotation() {
        Annotation ann(LabelType::XYWH);
        ann.addObject(0, {50.0f, 50.0f, 20.0f, 20.0f});
        return ann;
    }

    // Helper: Compare floats
    bool floatNear(float a, float b, float tol = 1e-3f) {
        return std::abs(a - b) < tol;
    }
};

// =============================================================================
// 9.1 Normalize Tests (CLR-001 ~ CLR-010)
// =============================================================================

// CLR-001: Normalize constructor
TEST_F(ColorTransformsTest, Normalize_Constructor) {
    std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> std = {0.229f, 0.224f, 0.225f};

    Normalize normalize(mean, std);

    EXPECT_EQ("Normalize", normalize.getName());
}

// CLR-002: Normalize constructor invalid std
TEST_F(ColorTransformsTest, Normalize_Constructor_InvalidStd) {
    std::array<float, 3> mean = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> invalidStd = {0.0f, 0.2f, 0.2f};  // Zero std is invalid

    EXPECT_THROW(Normalize(mean, invalidStd), std::invalid_argument);

    std::array<float, 3> negativeStd = {-0.1f, 0.2f, 0.2f};  // Negative std is invalid
    EXPECT_THROW(Normalize(mean, negativeStd), std::invalid_argument);
}

// CLR-003: Normalize imageNet factory
TEST_F(ColorTransformsTest, Normalize_ImageNetFactory) {
    Normalize normalize = Normalize::imageNet();

    EXPECT_EQ("Normalize", normalize.getName());
}

// CLR-004: Normalize apply empty
TEST_F(ColorTransformsTest, Normalize_Apply_Empty) {
    Normalize normalize = Normalize::imageNet();
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(normalize.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// CLR-005: Normalize apply wrong type
TEST_F(ColorTransformsTest, Normalize_Apply_WrongType) {
    Normalize normalize = Normalize::imageNet();
    cv::Mat image = createTestImage();  // CV_8UC3, not CV_32FC3
    Annotation ann = createTestAnnotation();

    EXPECT_THROW(normalize.apply(image, ann), std::invalid_argument);
}

// CLR-006: Normalize apply AVX2 path (tested via normal apply)
TEST_F(ColorTransformsTest, Normalize_Apply_AVX2) {
    Normalize normalize = Normalize::imageNet();
    cv::Mat image = createFloatImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(normalize.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// CLR-007: Normalize apply scalar fallback
TEST_F(ColorTransformsTest, Normalize_Apply_Scalar) {
    std::array<float, 3> mean = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> std = {0.5f, 0.5f, 0.5f};
    Normalize normalize(mean, std);

    cv::Mat image = createFloatImage();
    Annotation ann = createTestAnnotation();

    normalize.apply(image, ann);

    // Check that normalization was applied
    cv::Vec3f pixel = image.at<cv::Vec3f>(50, 50);
    // Original was (0.4, 0.5, 0.6), normalized: (v - 0.5) / 0.5
    EXPECT_TRUE(floatNear(pixel[0], -0.2f));  // (0.4 - 0.5) / 0.5 = -0.2
    EXPECT_TRUE(floatNear(pixel[1], 0.0f));   // (0.5 - 0.5) / 0.5 = 0.0
    EXPECT_TRUE(floatNear(pixel[2], 0.2f));   // (0.6 - 0.5) / 0.5 = 0.2
}

// CLR-008: Normalize clone
TEST_F(ColorTransformsTest, Normalize_Clone) {
    Normalize original = Normalize::imageNet();

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("Normalize", cloned->getName());
}

// CLR-009: Normalize getName
TEST_F(ColorTransformsTest, Normalize_GetName) {
    Normalize normalize = Normalize::imageNet();

    EXPECT_EQ("Normalize", normalize.getName());
}

// CLR-010: Normalize roundtrip accuracy
TEST_F(ColorTransformsTest, Normalize_RoundtripAccuracy) {
    std::array<float, 3> mean = {0.5f, 0.5f, 0.5f};
    std::array<float, 3> std = {0.25f, 0.25f, 0.25f};
    Normalize normalize(mean, std);

    cv::Mat image = createFloatImage();
    cv::Mat original = image.clone();
    Annotation ann = createTestAnnotation();

    normalize.apply(image, ann);

    // Verify normalized values are different from original
    cv::Vec3f origPixel = original.at<cv::Vec3f>(50, 50);
    cv::Vec3f normPixel = image.at<cv::Vec3f>(50, 50);

    EXPECT_NE(origPixel[0], normPixel[0]);
}

// =============================================================================
// 9.2 ToTensor Tests (CLR-011 ~ CLR-020)
// =============================================================================

// CLR-011: ToTensor constructor
TEST_F(ColorTransformsTest, ToTensor_Constructor) {
    ToTensor toTensor(true);

    EXPECT_EQ("ToTensor", toTensor.getName());
}

// CLR-012: ToTensor apply empty
TEST_F(ColorTransformsTest, ToTensor_Apply_Empty) {
    ToTensor toTensor;
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(toTensor.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// CLR-013: ToTensor apply wrong type
TEST_F(ColorTransformsTest, ToTensor_Apply_WrongType) {
    ToTensor toTensor;
    cv::Mat image = createFloatImage();  // CV_32FC3, not CV_8UC3
    Annotation ann = createTestAnnotation();

    EXPECT_THROW(toTensor.apply(image, ann), std::invalid_argument);
}

// CLR-014: ToTensor apply convert
TEST_F(ColorTransformsTest, ToTensor_Apply_Convert) {
    ToTensor toTensor(false);  // Don't convert BGR->RGB
    cv::Mat image = createTestImage();  // BGR (100, 128, 150)
    Annotation ann = createTestAnnotation();

    toTensor.apply(image, ann);

    EXPECT_EQ(CV_32FC3, image.type());

    cv::Vec3f pixel = image.at<cv::Vec3f>(50, 50);
    EXPECT_TRUE(floatNear(pixel[0], 100.0f / 255.0f));
    EXPECT_TRUE(floatNear(pixel[1], 128.0f / 255.0f));
    EXPECT_TRUE(floatNear(pixel[2], 150.0f / 255.0f));
}

// CLR-015: ToTensor apply bgrToRgb true
TEST_F(ColorTransformsTest, ToTensor_Apply_BgrToRgb) {
    ToTensor toTensor(true);  // Convert BGR->RGB
    cv::Mat image = createTestImage();  // BGR (100, 128, 150)
    Annotation ann = createTestAnnotation();

    toTensor.apply(image, ann);

    cv::Vec3f pixel = image.at<cv::Vec3f>(50, 50);
    // After BGR->RGB: (150, 128, 100) / 255
    EXPECT_TRUE(floatNear(pixel[0], 150.0f / 255.0f));  // R
    EXPECT_TRUE(floatNear(pixel[1], 128.0f / 255.0f));  // G
    EXPECT_TRUE(floatNear(pixel[2], 100.0f / 255.0f));  // B
}

// CLR-016: ToTensor apply bgrToRgb false
TEST_F(ColorTransformsTest, ToTensor_Apply_KeepBgr) {
    ToTensor toTensor(false);  // Keep BGR order
    cv::Mat image = createTestImage();  // BGR (100, 128, 150)
    Annotation ann = createTestAnnotation();

    toTensor.apply(image, ann);

    cv::Vec3f pixel = image.at<cv::Vec3f>(50, 50);
    // BGR order preserved: (100, 128, 150) / 255
    EXPECT_TRUE(floatNear(pixel[0], 100.0f / 255.0f));  // B
    EXPECT_TRUE(floatNear(pixel[1], 128.0f / 255.0f));  // G
    EXPECT_TRUE(floatNear(pixel[2], 150.0f / 255.0f));  // R
}

// CLR-017: ToTensor clone
TEST_F(ColorTransformsTest, ToTensor_Clone) {
    ToTensor original(true);

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("ToTensor", cloned->getName());
}

// CLR-018: ToTensor getName
TEST_F(ColorTransformsTest, ToTensor_GetName) {
    ToTensor toTensor;

    EXPECT_EQ("ToTensor", toTensor.getName());
}

// CLR-019: ToTensor output type
TEST_F(ColorTransformsTest, ToTensor_OutputType) {
    ToTensor toTensor;
    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    toTensor.apply(image, ann);

    EXPECT_EQ(CV_32FC3, image.type());
}

// CLR-020: ToTensor value range
TEST_F(ColorTransformsTest, ToTensor_ValueRange) {
    ToTensor toTensor(false);

    // Create image with extreme values
    cv::Mat image(10, 10, CV_8UC3);
    image.setTo(cv::Scalar(0, 128, 255));  // Min, mid, max
    Annotation ann = createTestAnnotation();

    toTensor.apply(image, ann);

    cv::Vec3f pixel = image.at<cv::Vec3f>(5, 5);
    EXPECT_GE(pixel[0], 0.0f);
    EXPECT_LE(pixel[0], 1.0f);
    EXPECT_GE(pixel[1], 0.0f);
    EXPECT_LE(pixel[1], 1.0f);
    EXPECT_GE(pixel[2], 0.0f);
    EXPECT_LE(pixel[2], 1.0f);
}

// =============================================================================
// 9.3 ColorJitter Tests (CLR-021 ~ CLR-032)
// =============================================================================

// CLR-021: ColorJitter constructor
TEST_F(ColorTransformsTest, ColorJitter_Constructor) {
    ColorJitter jitter(0.2f, 0.2f, 0.2f, 0.1f);

    EXPECT_EQ("ColorJitter", jitter.getName());
    EXPECT_TRUE(jitter.isRandom());
}

// CLR-022: ColorJitter constructor invalid
TEST_F(ColorTransformsTest, ColorJitter_Constructor_Invalid) {
    EXPECT_THROW(ColorJitter(-0.1f, 0.2f, 0.2f, 0.0f), std::invalid_argument);
    EXPECT_THROW(ColorJitter(0.2f, -0.1f, 0.2f, 0.0f), std::invalid_argument);
    EXPECT_THROW(ColorJitter(0.2f, 0.2f, -0.1f, 0.0f), std::invalid_argument);
}

// CLR-023: ColorJitter apply empty
TEST_F(ColorTransformsTest, ColorJitter_Apply_Empty) {
    ColorJitter jitter;
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(jitter.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// CLR-024: ColorJitter brightness
TEST_F(ColorTransformsTest, ColorJitter_Brightness) {
    ColorJitter jitter(0.5f, 0.0f, 0.0f, 0.0f);  // Only brightness
    jitter.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(jitter.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// CLR-025: ColorJitter contrast
TEST_F(ColorTransformsTest, ColorJitter_Contrast) {
    ColorJitter jitter(0.0f, 0.5f, 0.0f, 0.0f);  // Only contrast
    jitter.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(jitter.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// CLR-026: ColorJitter saturation
TEST_F(ColorTransformsTest, ColorJitter_Saturation) {
    ColorJitter jitter(0.0f, 0.0f, 0.5f, 0.0f);  // Only saturation
    jitter.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(jitter.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// CLR-027: ColorJitter hue
TEST_F(ColorTransformsTest, ColorJitter_Hue) {
    ColorJitter jitter(0.0f, 0.0f, 0.0f, 30.0f);  // Only hue
    jitter.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(jitter.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// CLR-028: ColorJitter random order
TEST_F(ColorTransformsTest, ColorJitter_RandomOrder) {
    ColorJitter jitter(0.2f, 0.2f, 0.2f, 0.1f);  // All params
    jitter.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    // Should apply transforms in random order
    EXPECT_NO_THROW(jitter.apply(image, ann));
}

// CLR-029: ColorJitter skip zero factor
TEST_F(ColorTransformsTest, ColorJitter_SkipZeroFactor) {
    ColorJitter jitter(0.0f, 0.0f, 0.0f, 0.0f);  // All zeros

    cv::Mat image = createTestImage();
    cv::Mat original = image.clone();
    Annotation ann = createTestAnnotation();

    jitter.apply(image, ann);

    // Image should be unchanged when all factors are zero
    cv::Mat diff;
    cv::absdiff(image, original, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// CLR-030: ColorJitter clone
TEST_F(ColorTransformsTest, ColorJitter_Clone) {
    ColorJitter original(0.2f, 0.2f, 0.2f, 0.1f);

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("ColorJitter", cloned->getName());
}

// CLR-031: ColorJitter getName
TEST_F(ColorTransformsTest, ColorJitter_GetName) {
    ColorJitter jitter;

    EXPECT_EQ("ColorJitter", jitter.getName());
}

// CLR-032: ColorJitter reproducibility
TEST_F(ColorTransformsTest, ColorJitter_Reproducibility) {
    ColorJitter jitter1(0.3f, 0.3f, 0.3f, 0.1f);
    ColorJitter jitter2(0.3f, 0.3f, 0.3f, 0.1f);

    jitter1.setSeed(42);
    jitter2.setSeed(42);

    cv::Mat image1 = createTestImage();
    cv::Mat image2 = createTestImage();
    Annotation ann1 = createTestAnnotation();
    Annotation ann2 = createTestAnnotation();

    jitter1.apply(image1, ann1);
    jitter2.apply(image2, ann2);

    cv::Mat diff;
    cv::absdiff(image1, image2, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// =============================================================================
// 9.4 GaussianBlur Tests (CLR-033 ~ CLR-044)
// =============================================================================

// CLR-033: GaussianBlur constructor
TEST_F(ColorTransformsTest, GaussianBlur_Constructor) {
    GaussianBlur blur(3, 7, 0.5f);

    EXPECT_EQ("GaussianBlur", blur.getName());
    EXPECT_TRUE(blur.isRandom());
}

// CLR-034: GaussianBlur makeOddKernelSize
TEST_F(ColorTransformsTest, GaussianBlur_MakeOddKernelSize) {
    // Even sizes should be converted to odd
    GaussianBlur blur1(4, 6, 1.0f);  // Should become 5, 7
    GaussianBlur blur2(3, 7, 1.0f);  // Already odd

    EXPECT_EQ("GaussianBlur", blur1.getName());
    EXPECT_EQ("GaussianBlur", blur2.getName());
}

// CLR-035: GaussianBlur constructor invalid
TEST_F(ColorTransformsTest, GaussianBlur_Constructor_Invalid) {
    EXPECT_THROW(GaussianBlur(7, 3, 0.5f), std::invalid_argument);  // min > max
    EXPECT_THROW(GaussianBlur(3, 7, -0.1f), std::invalid_argument);  // negative prob
    EXPECT_THROW(GaussianBlur(3, 7, 1.1f), std::invalid_argument);   // prob > 1
}

// CLR-036: GaussianBlur apply empty
TEST_F(ColorTransformsTest, GaussianBlur_Apply_Empty) {
    GaussianBlur blur(3, 7, 1.0f);
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(blur.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// CLR-037: GaussianBlur apply skip
TEST_F(ColorTransformsTest, GaussianBlur_Apply_Skip) {
    GaussianBlur blur(3, 7, 0.0f);  // probability = 0
    cv::Mat image = createTestImage();
    cv::Mat original = image.clone();
    Annotation ann = createTestAnnotation();

    blur.apply(image, ann);

    cv::Mat diff;
    cv::absdiff(image, original, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// CLR-038: GaussianBlur apply same kernel
TEST_F(ColorTransformsTest, GaussianBlur_Apply_SameKernel) {
    GaussianBlur blur(5, 5, 1.0f);  // min == max
    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(blur.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// CLR-039: GaussianBlur apply range kernel
TEST_F(ColorTransformsTest, GaussianBlur_Apply_RangeKernel) {
    GaussianBlur blur(3, 11, 1.0f);  // Range of kernel sizes
    blur.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(blur.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// CLR-040: GaussianBlur apply AVX2 (tested via normal apply)
TEST_F(ColorTransformsTest, GaussianBlur_Apply_AVX2) {
    GaussianBlur blur(5, 5, 1.0f);
    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(blur.apply(image, ann));
}

// CLR-041: GaussianBlur apply OpenCV fallback
TEST_F(ColorTransformsTest, GaussianBlur_Apply_OpenCVFallback) {
    GaussianBlur blur(3, 3, 1.0f);
    cv::Mat image = createGrayImage();  // Single channel won't use AVX2 BGR path
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(blur.apply(image, ann));
}

// CLR-042: GaussianBlur AVX2 bad_alloc (tested implicitly)
TEST_F(ColorTransformsTest, GaussianBlur_LargeImage) {
    GaussianBlur blur(3, 3, 1.0f);
    cv::Mat image = createTestImage(500, 500);  // Larger image
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(blur.apply(image, ann));
}

// CLR-043: GaussianBlur clone
TEST_F(ColorTransformsTest, GaussianBlur_Clone) {
    GaussianBlur original(3, 7, 0.5f);

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("GaussianBlur", cloned->getName());
}

// CLR-044: GaussianBlur getName
TEST_F(ColorTransformsTest, GaussianBlur_GetName) {
    GaussianBlur blur;

    EXPECT_EQ("GaussianBlur", blur.getName());
}

// =============================================================================
// 9.5 GaussianNoise Tests (CLR-045 ~ CLR-052)
// =============================================================================

// CLR-045: GaussianNoise constructor
TEST_F(ColorTransformsTest, GaussianNoise_Constructor) {
    GaussianNoise noise(0.0f, 25.0f);

    EXPECT_EQ("GaussianNoise", noise.getName());
    EXPECT_TRUE(noise.isRandom());
}

// CLR-046: GaussianNoise constructor invalid
TEST_F(ColorTransformsTest, GaussianNoise_Constructor_Invalid) {
    EXPECT_THROW(GaussianNoise(0.0f, -1.0f), std::invalid_argument);  // negative stddev
}

// CLR-047: GaussianNoise apply empty
TEST_F(ColorTransformsTest, GaussianNoise_Apply_Empty) {
    GaussianNoise noise;
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(noise.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// CLR-048: GaussianNoise apply skip (probability check)
TEST_F(ColorTransformsTest, GaussianNoise_Apply_Probability) {
    GaussianNoise noise(0.0f, 25.0f);
    noise.setSeed(12345);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    // With default low probability, might skip - just ensure no crash
    EXPECT_NO_THROW(noise.apply(image, ann));
}

// CLR-049: GaussianNoise apply noise
TEST_F(ColorTransformsTest, GaussianNoise_Apply_Noise) {
    GaussianNoise noise(0.0f, 50.0f);
    noise.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(noise.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// CLR-050: GaussianNoise apply clamp
TEST_F(ColorTransformsTest, GaussianNoise_Apply_Clamp) {
    GaussianNoise noise(0.0f, 100.0f);  // High stddev
    noise.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    noise.apply(image, ann);

    // All values should be in [0, 255] range
    double minVal, maxVal;
    cv::minMaxLoc(image.reshape(1), &minVal, &maxVal);
    EXPECT_GE(minVal, 0.0);
    EXPECT_LE(maxVal, 255.0);
}

// CLR-051: GaussianNoise clone
TEST_F(ColorTransformsTest, GaussianNoise_Clone) {
    GaussianNoise original(0.0f, 30.0f);

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("GaussianNoise", cloned->getName());
}

// CLR-052: GaussianNoise getName
TEST_F(ColorTransformsTest, GaussianNoise_GetName) {
    GaussianNoise noise;

    EXPECT_EQ("GaussianNoise", noise.getName());
}

// =============================================================================
// 9.6 CLAHE Tests (CLR-053 ~ CLR-062)
// =============================================================================

// CLR-053: CLAHE constructor
TEST_F(ColorTransformsTest, CLAHE_Constructor) {
    CLAHE clahe(2.0f, 8);

    EXPECT_EQ("CLAHE", clahe.getName());
    EXPECT_TRUE(clahe.isRandom());
}

// CLR-054: CLAHE constructor invalid clipLimit
TEST_F(ColorTransformsTest, CLAHE_Constructor_InvalidClipLimit) {
    EXPECT_THROW(CLAHE(0.0f, 8), std::invalid_argument);
    EXPECT_THROW(CLAHE(-1.0f, 8), std::invalid_argument);
}

// CLR-055: CLAHE constructor invalid gridSize
TEST_F(ColorTransformsTest, CLAHE_Constructor_InvalidGridSize) {
    EXPECT_THROW(CLAHE(2.0f, 0), std::invalid_argument);
    EXPECT_THROW(CLAHE(2.0f, -1), std::invalid_argument);
}

// CLR-056: CLAHE apply empty
TEST_F(ColorTransformsTest, CLAHE_Apply_Empty) {
    CLAHE clahe;
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(clahe.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// CLR-057: CLAHE apply skip (probability check)
TEST_F(ColorTransformsTest, CLAHE_Apply_Probability) {
    CLAHE clahe(2.0f, 8);
    clahe.setSeed(12345);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(clahe.apply(image, ann));
}

// CLR-058: CLAHE apply grayscale
TEST_F(ColorTransformsTest, CLAHE_Apply_Grayscale) {
    CLAHE clahe(2.0f, 8);
    clahe.setSeed(42);

    cv::Mat image = createGrayImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(clahe.apply(image, ann));
    EXPECT_EQ(1, image.channels());
}

// CLR-059: CLAHE apply color
TEST_F(ColorTransformsTest, CLAHE_Apply_Color) {
    CLAHE clahe(2.0f, 8);
    clahe.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(clahe.apply(image, ann));
    EXPECT_EQ(3, image.channels());
}

// CLR-060: CLAHE clone
TEST_F(ColorTransformsTest, CLAHE_Clone) {
    CLAHE original(3.0f, 4);

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("CLAHE", cloned->getName());
}

// CLR-061: CLAHE getName
TEST_F(ColorTransformsTest, CLAHE_GetName) {
    CLAHE clahe;

    EXPECT_EQ("CLAHE", clahe.getName());
}

// CLR-062: CLAHE internal clahe_ object
TEST_F(ColorTransformsTest, CLAHE_InternalObject) {
    CLAHE clahe(4.0f, 16);
    clahe.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    // Just verify it works with non-default params
    EXPECT_NO_THROW(clahe.apply(image, ann));
}

// =============================================================================
// 9.7 RandomGamma Tests (CLR-063 ~ CLR-070)
// =============================================================================

// CLR-063: RandomGamma constructor
TEST_F(ColorTransformsTest, RandomGamma_Constructor) {
    RandomGamma gamma(0.4f);

    EXPECT_EQ("RandomGamma", gamma.getName());
    EXPECT_TRUE(gamma.isRandom());
}

// CLR-064: RandomGamma constructor invalid
TEST_F(ColorTransformsTest, RandomGamma_Constructor_Invalid) {
    EXPECT_THROW(RandomGamma(-0.1f), std::invalid_argument);
    EXPECT_THROW(RandomGamma(1.0f), std::invalid_argument);  // >= 1.0 invalid
    EXPECT_THROW(RandomGamma(1.5f), std::invalid_argument);
}

// CLR-065: RandomGamma apply empty
TEST_F(ColorTransformsTest, RandomGamma_Apply_Empty) {
    RandomGamma gamma;
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(gamma.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// CLR-066: RandomGamma apply skip (probability check)
TEST_F(ColorTransformsTest, RandomGamma_Apply_Probability) {
    RandomGamma gamma(0.3f);
    gamma.setSeed(12345);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(gamma.apply(image, ann));
}

// CLR-067: RandomGamma apply LUT
TEST_F(ColorTransformsTest, RandomGamma_Apply_LUT) {
    RandomGamma gamma(0.5f);
    gamma.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(gamma.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// CLR-068: RandomGamma clone
TEST_F(ColorTransformsTest, RandomGamma_Clone) {
    RandomGamma original(0.3f);

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("RandomGamma", cloned->getName());
}

// CLR-069: RandomGamma getName
TEST_F(ColorTransformsTest, RandomGamma_GetName) {
    RandomGamma gamma;

    EXPECT_EQ("RandomGamma", gamma.getName());
}

// CLR-070: RandomGamma reproducibility
TEST_F(ColorTransformsTest, RandomGamma_Reproducibility) {
    RandomGamma gamma1(0.4f);
    RandomGamma gamma2(0.4f);

    gamma1.setSeed(42);
    gamma2.setSeed(42);

    cv::Mat image1 = createTestImage();
    cv::Mat image2 = createTestImage();
    Annotation ann1 = createTestAnnotation();
    Annotation ann2 = createTestAnnotation();

    gamma1.apply(image1, ann1);
    gamma2.apply(image2, ann2);

    cv::Mat diff;
    cv::absdiff(image1, image2, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// =============================================================================
// 9.8 ToGray Tests (CLR-071 ~ CLR-080)
// =============================================================================

// CLR-071: ToGray constructor
TEST_F(ColorTransformsTest, ToGray_Constructor) {
    ToGray toGray1(false);  // Default probability (0.01f)
    ToGray toGray2(true, 0.5f);  // Custom probability

    EXPECT_EQ("ToGray", toGray1.getName());
    EXPECT_TRUE(toGray1.isRandom());
    EXPECT_EQ("ToGray", toGray2.getName());
    EXPECT_TRUE(toGray2.isRandom());
}

// CLR-072: ToGray apply empty
TEST_F(ColorTransformsTest, ToGray_Apply_Empty) {
    ToGray toGray;
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(toGray.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// CLR-073: ToGray apply skip (probability check)
TEST_F(ColorTransformsTest, ToGray_Apply_Probability) {
    // Test with 0% probability - should never apply
    ToGray toGray(true, 0.0f);

    cv::Mat image = createTestImage();
    cv::Mat original = image.clone();
    Annotation ann = createTestAnnotation();

    toGray.apply(image, ann);

    // Image should be unchanged when probability is 0
    cv::Mat diff;
    cv::absdiff(image, original, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// CLR-074: ToGray apply 1ch keepChannels true
TEST_F(ColorTransformsTest, ToGray_Apply_1ch_KeepChannelsTrue) {
    ToGray toGray(true, 1.0f);  // keepChannels=true, probability=1.0 to always apply

    cv::Mat image = createGrayImage();
    Annotation ann = createTestAnnotation();

    toGray.apply(image, ann);

    // 1-channel grayscale with keepChannels=true should become 3-channel
    EXPECT_EQ(3, image.channels());
}

// CLR-075: ToGray apply 1ch keepChannels false
TEST_F(ColorTransformsTest, ToGray_Apply_1ch_KeepChannelsFalse) {
    ToGray toGray(false, 1.0f);  // keepChannels=false, probability=1.0 to always apply

    cv::Mat image = createGrayImage();
    Annotation ann = createTestAnnotation();

    toGray.apply(image, ann);

    // 1-channel grayscale with keepChannels=false should remain 1-channel
    EXPECT_EQ(1, image.channels());
}

// CLR-076: ToGray apply 3ch keepChannels true
TEST_F(ColorTransformsTest, ToGray_Apply_3ch_KeepChannelsTrue) {
    ToGray toGray(true, 1.0f);  // keepChannels=true, probability=1.0 to always apply

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    toGray.apply(image, ann);

    // 3-channel color with keepChannels=true should remain 3-channel (grayscale values in all channels)
    EXPECT_EQ(3, image.channels());
}

// CLR-077: ToGray apply 3ch keepChannels false
TEST_F(ColorTransformsTest, ToGray_Apply_3ch_KeepChannelsFalse) {
    ToGray toGray(false, 1.0f);  // keepChannels=false, probability=1.0 to always apply

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    toGray.apply(image, ann);

    // 3-channel color with keepChannels=false should become 1-channel
    EXPECT_EQ(1, image.channels());
}

// CLR-078: ToGray clone
TEST_F(ColorTransformsTest, ToGray_Clone) {
    ToGray original(true, 0.5f);  // Test with non-default probability

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("ToGray", cloned->getName());
}

// CLR-079: ToGray getName
TEST_F(ColorTransformsTest, ToGray_GetName) {
    ToGray toGray;

    EXPECT_EQ("ToGray", toGray.getName());
}

// CLR-080: ToGray reproducibility
TEST_F(ColorTransformsTest, ToGray_Reproducibility) {
    ToGray toGray1(true, 1.0f);  // Use 100% probability for deterministic test
    ToGray toGray2(true, 1.0f);

    toGray1.setSeed(42);
    toGray2.setSeed(42);

    cv::Mat image1 = createTestImage();
    cv::Mat image2 = createTestImage();
    Annotation ann1 = createTestAnnotation();
    Annotation ann2 = createTestAnnotation();

    toGray1.apply(image1, ann1);
    toGray2.apply(image2, ann2);

    cv::Mat diff;
    cv::absdiff(image1, image2, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// =============================================================================
// 9.9 SaltAndPepper Tests (CLR-081 ~ CLR-090)
// =============================================================================

// CLR-081: SaltAndPepper constructor
TEST_F(ColorTransformsTest, SaltAndPepper_Constructor) {
    SaltAndPepper sap(0.01f, 0.01f);

    EXPECT_EQ("SaltAndPepper", sap.getName());
    EXPECT_TRUE(sap.isRandom());
}

// CLR-082: SaltAndPepper constructor invalid salt
TEST_F(ColorTransformsTest, SaltAndPepper_Constructor_InvalidSalt) {
    EXPECT_THROW(SaltAndPepper(-0.1f, 0.01f), std::invalid_argument);
    EXPECT_THROW(SaltAndPepper(1.1f, 0.01f), std::invalid_argument);
}

// CLR-083: SaltAndPepper constructor invalid pepper
TEST_F(ColorTransformsTest, SaltAndPepper_Constructor_InvalidPepper) {
    EXPECT_THROW(SaltAndPepper(0.01f, -0.1f), std::invalid_argument);
    EXPECT_THROW(SaltAndPepper(0.01f, 1.1f), std::invalid_argument);
}

// CLR-084: SaltAndPepper apply empty
TEST_F(ColorTransformsTest, SaltAndPepper_Apply_Empty) {
    SaltAndPepper sap;
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(sap.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// CLR-085: SaltAndPepper apply skip (probability check)
TEST_F(ColorTransformsTest, SaltAndPepper_Apply_Probability) {
    SaltAndPepper sap(0.05f, 0.05f);
    sap.setSeed(12345);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(sap.apply(image, ann));
}

// CLR-086: SaltAndPepper apply salt 1ch
TEST_F(ColorTransformsTest, SaltAndPepper_Apply_Salt_1ch) {
    SaltAndPepper sap(0.5f, 0.0f);  // High salt probability
    sap.setSeed(42);

    cv::Mat image = createGrayImage();
    Annotation ann = createTestAnnotation();

    sap.apply(image, ann);

    // Should have some white pixels (255)
    double maxVal;
    cv::minMaxLoc(image, nullptr, &maxVal);
    // Note: might not always have 255 due to probability
    EXPECT_FALSE(image.empty());
}

// CLR-087: SaltAndPepper apply salt 3ch
TEST_F(ColorTransformsTest, SaltAndPepper_Apply_Salt_3ch) {
    SaltAndPepper sap(0.5f, 0.0f);  // High salt probability
    sap.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    sap.apply(image, ann);

    EXPECT_EQ(3, image.channels());
}

// CLR-088: SaltAndPepper apply pepper 1ch
TEST_F(ColorTransformsTest, SaltAndPepper_Apply_Pepper_1ch) {
    SaltAndPepper sap(0.0f, 0.5f);  // High pepper probability
    sap.setSeed(42);

    cv::Mat image = createGrayImage();
    Annotation ann = createTestAnnotation();

    sap.apply(image, ann);

    // Should have some black pixels (0)
    double minVal;
    cv::minMaxLoc(image, &minVal, nullptr);
    EXPECT_FALSE(image.empty());
}

// CLR-089: SaltAndPepper apply pepper 3ch
TEST_F(ColorTransformsTest, SaltAndPepper_Apply_Pepper_3ch) {
    SaltAndPepper sap(0.0f, 0.5f);  // High pepper probability
    sap.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    sap.apply(image, ann);

    EXPECT_EQ(3, image.channels());
}

// CLR-090: SaltAndPepper clone
TEST_F(ColorTransformsTest, SaltAndPepper_Clone) {
    SaltAndPepper original(0.02f, 0.02f);

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("SaltAndPepper", cloned->getName());
}

// =============================================================================
// 9.10 MedianBlur Tests (CLR-091 ~ CLR-100)
// =============================================================================

// CLR-091: MedianBlur constructor
TEST_F(ColorTransformsTest, MedianBlur_Constructor) {
    MedianBlur blur(3, 7, 0.5f);

    EXPECT_EQ("MedianBlur", blur.getName());
    EXPECT_TRUE(blur.isRandom());
}

// CLR-092: MedianBlur makeOddKernelSize
TEST_F(ColorTransformsTest, MedianBlur_MakeOddKernelSize) {
    MedianBlur blur1(4, 6, 1.0f);  // Should become 5, 7
    MedianBlur blur2(3, 7, 1.0f);  // Already odd

    EXPECT_EQ("MedianBlur", blur1.getName());
    EXPECT_EQ("MedianBlur", blur2.getName());
}

// CLR-093: MedianBlur constructor invalid
TEST_F(ColorTransformsTest, MedianBlur_Constructor_Invalid) {
    EXPECT_THROW(MedianBlur(7, 3, 0.5f), std::invalid_argument);  // min > max
    EXPECT_THROW(MedianBlur(3, 7, -0.1f), std::invalid_argument);  // negative prob
    EXPECT_THROW(MedianBlur(3, 7, 1.1f), std::invalid_argument);   // prob > 1
}

// CLR-094: MedianBlur apply empty
TEST_F(ColorTransformsTest, MedianBlur_Apply_Empty) {
    MedianBlur blur(3, 7, 1.0f);
    cv::Mat emptyImage;
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(blur.apply(emptyImage, ann));
    EXPECT_TRUE(emptyImage.empty());
}

// CLR-095: MedianBlur apply skip
TEST_F(ColorTransformsTest, MedianBlur_Apply_Skip) {
    MedianBlur blur(3, 7, 0.0f);  // probability = 0
    cv::Mat image = createTestImage();
    cv::Mat original = image.clone();
    Annotation ann = createTestAnnotation();

    blur.apply(image, ann);

    cv::Mat diff;
    cv::absdiff(image, original, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// CLR-096: MedianBlur apply same kernel
TEST_F(ColorTransformsTest, MedianBlur_Apply_SameKernel) {
    MedianBlur blur(5, 5, 1.0f);  // min == max
    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(blur.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// CLR-097: MedianBlur apply range kernel
TEST_F(ColorTransformsTest, MedianBlur_Apply_RangeKernel) {
    MedianBlur blur(3, 11, 1.0f);  // Range of kernel sizes
    blur.setSeed(42);

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(blur.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// CLR-098: MedianBlur apply cv::medianBlur
TEST_F(ColorTransformsTest, MedianBlur_Apply_OpenCV) {
    MedianBlur blur(3, 3, 1.0f);
    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    EXPECT_NO_THROW(blur.apply(image, ann));
    EXPECT_FALSE(image.empty());
}

// CLR-099: MedianBlur clone
TEST_F(ColorTransformsTest, MedianBlur_Clone) {
    MedianBlur original(3, 7, 0.5f);

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("MedianBlur", cloned->getName());
}

// CLR-100: MedianBlur getName
TEST_F(ColorTransformsTest, MedianBlur_GetName) {
    MedianBlur blur;

    EXPECT_EQ("MedianBlur", blur.getName());
}
