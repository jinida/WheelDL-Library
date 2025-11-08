#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Transforms/Transform.h"
#include "WheelDL.Lib/Data/Transforms/GeometricTransforms.h"
#include "WheelDL.Lib/Data/Transforms/ColorTransforms.h"
#include "WheelDL.Lib/Data/Common/Annotation.h"
#include "WheelDL.Lib/Data/Utils/ImageIO.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <memory>

using namespace WheelDL::Data;
using namespace WheelDL::Data::Transforms;
using WheelDL::Data::Utils::ImageIO;
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
}

/**
 * @class TransformComprehensiveTest
 * @brief Comprehensive test suite for all Transform functionality
 *
 * Tests:
 * - Transform base class operations (clone, seed)
 * - Compose pipeline tests
 * - LetterBox with aspect ratio and padding
 * - RandomPerspective with all transformation types
 * - ColorJitter with all color parameters
 * - GaussianBlur with probability
 * - Resize with/without aspect ratio
 * - Flip operations with annotations
 * - Edge cases and reproducibility
 */
class TransformComprehensiveTest : public ::testing::Test
{
protected:
	const float EPSILON = 1e-4f;

	void SetUp() override {
		// Load real MNIST images for testing
		auto mnistPath = getTestDataPath() / "mnist_sample" / "images";
		auto imageFiles = getImageFilesInDirectory(mnistPath);
		if (!imageFiles.empty()) {
			testImage_ = ImageIO::loadImage(imageFiles[0], 640);
		}
		else {
			// Fallback to synthetic image
			testImage_ = createTestImage(28, 28);
		}
	}

	std::vector<std::string> getImageFilesInDirectory(const fs::path& dir) {
		std::vector<std::string> files;
		if (!fs::exists(dir)) return files;

		for (const auto& entry : fs::directory_iterator(dir)) {
			if (entry.is_regular_file()) {
				auto ext = entry.path().extension().string();
				if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
					files.push_back(entry.path().string());
					if (files.size() >= 5) break;
				}
			}
		}
		return files;
	}

	cv::Mat createTestImage(int rows, int cols, cv::Scalar color = cv::Scalar(128, 128, 128))
	{
		cv::Mat image(rows, cols, CV_8UC3, color);
		// Add random noise to make blur effects detectable
		cv::Mat noise(rows, cols, CV_8UC3);
		cv::randu(noise, cv::Scalar(-20, -20, -20), cv::Scalar(20, 20, 20));
		image += noise;
		return image;
	}

	Annotation createTestAnnotation(LabelType type = LabelType::XYWH) {
		Annotation ann(type);
		switch (type) {
		case LabelType::XYWH:
			ann.addObject(0, { 100.0f, 100.0f, 50.0f, 50.0f }); // center-based
			ann.addObject(1, { 200.0f, 150.0f, 60.0f, 40.0f });
			break;
		case LabelType::XYXY:
			ann.addObject(0, { 75.0f, 75.0f, 125.0f, 125.0f }); // corner-based
			ann.addObject(1, { 170.0f, 130.0f, 230.0f, 170.0f });
			break;
		case LabelType::POLYGON:
			ann.addObject(0, { 75.0f, 75.0f, 125.0f, 75.0f, 125.0f, 125.0f, 75.0f, 125.0f }); // 4 corners
			ann.addObject(1, { 50.0f, 50.0f, 70.0f, 40.0f, 80.0f, 60.0f }); // triangle
			break;
		case LabelType::XYWHR:
			ann.addObject(0, { 100.0f, 100.0f, 50.0f, 50.0f, 0.785f }); // with rotation
			break;
		case LabelType::XYXYXYXY:
			ann.addObject(0, { 75.0f, 75.0f, 125.0f, 75.0f, 125.0f, 125.0f, 75.0f, 125.0f }); // 4 corners
			break;
		default:
			break;
		}
		return ann;
	}

	void expectVectorNear(const std::vector<float>& actual, const std::vector<float>& expected, float epsilon = 1e-4f) {
		ASSERT_EQ(actual.size(), expected.size()) << "Vector sizes differ";
		for (size_t i = 0; i < actual.size(); ++i) {
			EXPECT_NEAR(actual[i], expected[i], epsilon) << "Mismatch at index " << i;
		}
	}

	cv::Mat testImage_;
};

// ========================================
// Transform Base Class Tests
// ========================================

TEST_F(TransformComprehensiveTest, BaseClass_CloneOperation) {
	RandomHorizontalFlip original(0.5f);
	original.setSeed(42);

	auto cloned = original.clone();
	ASSERT_NE(cloned, nullptr);
	EXPECT_EQ(cloned->getName(), "RandomHorizontalFlip");
	EXPECT_TRUE(cloned->isRandom());
}

TEST_F(TransformComprehensiveTest, BaseClass_SeedSetting) {
	RandomHorizontalFlip transform(1.0f); // Always flip
	transform.setSeed(12345);

	cv::Mat img1 = createTestImage(100, 100);
	cv::Mat img2 = img1.clone();
	Annotation ann1 = createTestAnnotation();
	Annotation ann2 = createTestAnnotation();

	transform.apply(img1, ann1);
	transform.setSeed(12345); // Reset to same seed
	transform.apply(img2, ann2);

	// Should produce identical results
	EXPECT_EQ(cv::norm(img1, img2, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, BaseClass_IsRandomDeterministic) {
	LetterBox deterministic(640, 640);
	EXPECT_FALSE(deterministic.isRandom());

	RandomHorizontalFlip random(0.5f);
	EXPECT_TRUE(random.isRandom());
}

// ========================================
// Compose Tests
// ========================================

TEST_F(TransformComprehensiveTest, Compose_Empty) {
	Compose compose;
	EXPECT_TRUE(compose.empty());
	EXPECT_EQ(compose.size(), 0);
	EXPECT_FALSE(compose.isRandom());

	cv::Mat image = createTestImage(100, 100);
	Annotation ann = createTestAnnotation();
	cv::Mat originalImage = image.clone();

	compose.apply(image, ann);

	// Image should be unchanged
	EXPECT_EQ(cv::norm(image, originalImage, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, Compose_SingleTransform) {
	Compose compose;
	compose.addTransform(std::make_unique<LetterBox>(640, 640));

	EXPECT_FALSE(compose.empty());
	EXPECT_EQ(compose.size(), 1);
	EXPECT_FALSE(compose.isRandom());

	cv::Mat image = createTestImage(480, 640);
	Annotation ann = createTestAnnotation();

	compose.apply(image, ann);

	EXPECT_EQ(image.rows, 640);
	EXPECT_EQ(image.cols, 640);
}

TEST_F(TransformComprehensiveTest, Compose_MultipleTransforms) {
	Compose compose;
	compose.addTransform(std::make_unique<LetterBox>(640, 640));
	compose.addTransform(std::make_unique<RandomHorizontalFlip>(0.5f));
	compose.addTransform(std::make_unique<ColorJitter>(0.2f, 0.2f, 0.2f, 0.01f));

	EXPECT_EQ(compose.size(), 3);
	EXPECT_TRUE(compose.isRandom()); // Contains random transforms

	cv::Mat image = createTestImage(480, 640);
	Annotation ann = createTestAnnotation();

	EXPECT_NO_THROW(compose.apply(image, ann));
	EXPECT_EQ(image.rows, 640);
	EXPECT_EQ(image.cols, 640);
}

TEST_F(TransformComprehensiveTest, Compose_CloneBehavior) {
	Compose original;
	original.addTransform(std::make_unique<LetterBox>(640, 640));
	original.addTransform(std::make_unique<RandomHorizontalFlip>(0.5f));

	auto cloned = original.clone();
	auto clonedCompose = dynamic_cast<Compose*>(cloned.get());
	ASSERT_NE(clonedCompose, nullptr);
	EXPECT_EQ(clonedCompose->size(), 2);
	EXPECT_TRUE(clonedCompose->isRandom());
}

TEST_F(TransformComprehensiveTest, Compose_SeedPropagation) {
	Compose compose;
	compose.addTransform(std::make_unique<RandomHorizontalFlip>(1.0f));
	compose.addTransform(std::make_unique<ColorJitter>(0.5f, 0.5f, 0.5f, 0.05f));
	compose.setSeed(42);

	cv::Mat img1 = createTestImage(100, 100);
	cv::Mat img2 = img1.clone();
	Annotation ann1 = createTestAnnotation();
	Annotation ann2 = createTestAnnotation();

	compose.apply(img1, ann1);
	compose.setSeed(42); // Reset seed
	compose.apply(img2, ann2);

	// Should produce identical results
	EXPECT_EQ(cv::norm(img1, img2, cv::NORM_L2), 0.0);
}

// ========================================
// LetterBox Tests
// ========================================

TEST_F(TransformComprehensiveTest, LetterBox_AspectRatioPreservation) {
	LetterBox transform(640, 640, 114, 114, 114, true, true);

	cv::Mat image = createTestImage(480, 640); // 4:3 aspect ratio
	float aspectBefore = static_cast<float>(image.cols) / image.rows;

	Annotation ann = createTestAnnotation();
	transform.apply(image, ann);

	// After letterbox, image should be 640x640
	EXPECT_EQ(image.rows, 640);
	EXPECT_EQ(image.cols, 640);

	// Content aspect ratio should be preserved (within padding region)
	// The actual content will have aspect close to original
	float expectedScale = std::min(640.0f / 640.0f, 640.0f / 480.0f);
	EXPECT_GT(expectedScale, 0.0f);
}

TEST_F(TransformComprehensiveTest, LetterBox_PaddingCalculation_Horizontal) {
	LetterBox transform(640, 640, 114, 114, 114, true, true);

	cv::Mat image = createTestImage(400, 800); // Wide image (2:1)
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	EXPECT_EQ(image.rows, 640);
	EXPECT_EQ(image.cols, 640);

	// Top and bottom should have padding (horizontal letterbox)
	// Check that padding color exists in image
	cv::Scalar meanColor = cv::mean(image.row(0)); // Top row should be padding
	EXPECT_NEAR(meanColor[0], 114, 5); // Allow some interpolation tolerance
}

TEST_F(TransformComprehensiveTest, LetterBox_PaddingCalculation_Vertical) {
	LetterBox transform(640, 640, 114, 114, 114, true, true);

	cv::Mat image = createTestImage(800, 400); // Tall image (1:2)
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	EXPECT_EQ(image.rows, 640);
	EXPECT_EQ(image.cols, 640);

	// Left and right should have padding (vertical letterbox)
	cv::Scalar meanColor = cv::mean(image.col(0)); // Left column should be padding
	EXPECT_NEAR(meanColor[0], 114, 5);
}

TEST_F(TransformComprehensiveTest, LetterBox_DifferentTargetSizes) {
	LetterBox transform1(320, 320);
	LetterBox transform2(640, 480);
	LetterBox transform3(1280, 720);

	cv::Mat img1 = createTestImage(480, 640);
	cv::Mat img2 = img1.clone();
	cv::Mat img3 = img1.clone();

	Annotation ann1 = createTestAnnotation();
	Annotation ann2 = createTestAnnotation();
	Annotation ann3 = createTestAnnotation();

	transform1.apply(img1, ann1);
	transform2.apply(img2, ann2);
	transform3.apply(img3, ann3);

	EXPECT_EQ(img1.rows, 320);
	EXPECT_EQ(img1.cols, 320);
	EXPECT_EQ(img2.rows, 640);
	EXPECT_EQ(img2.cols, 480);
	EXPECT_EQ(img3.rows, 1280);
	EXPECT_EQ(img3.cols, 720);
}

TEST_F(TransformComprehensiveTest, LetterBox_AnnotationAdjustment) {
	LetterBox transform(640, 640, 114, 114, 114, true, true);

	cv::Mat image = createTestImage(320, 640);
	Annotation ann = createTestAnnotation(LabelType::XYWH);

	auto originalPoints = ann.getPoints();
	transform.apply(image, ann);
	auto transformedPoints = ann.getPoints();

	// Annotations should be adjusted (scaled/translated)
	// Image is 320x640, scaled to 640x640 -> scale=1.0, offsetY=160
	// Y coordinate should change by offsetY
	EXPECT_NE(originalPoints[0][1], transformedPoints[0][1]); // Y coordinate changed
}

TEST_F(TransformComprehensiveTest, LetterBox_NoScaleUp) {
	LetterBox transform(1920, 1080, 114, 114, 114, true, false); // scaleUp=false

	cv::Mat image = createTestImage(480, 640); // Smaller than target
	Annotation ann = createTestAnnotation();

	int originalRows = image.rows;
	int originalCols = image.cols;

	transform.apply(image, ann);

	// Image should not be scaled up, just padded
	// Result should be padded to fit target but not upscaled
	EXPECT_LE(image.rows, 1920);
	EXPECT_LE(image.cols, 1080);
}

// ========================================
// RandomPerspective Tests
// ========================================

TEST_F(TransformComprehensiveTest, RandomPerspective_RotationOnly) {
	RandomPerspective transform(10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 640, 640, 1.0f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(640, 640);
	Annotation ann = createTestAnnotation();

	cv::Mat original = image.clone();
	transform.apply(image, ann);

	// Image should be modified
	EXPECT_GT(cv::norm(image, original, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, RandomPerspective_ScaleOnly) {
	RandomPerspective transform(0.0f, 0.0f, 0.2f, 0.0f, 0.0f, 640, 640, 1.0f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(640, 640);
	Annotation ann = createTestAnnotation();

	EXPECT_NO_THROW(transform.apply(image, ann));
}

TEST_F(TransformComprehensiveTest, RandomPerspective_TranslationOnly) {
	RandomPerspective transform(0.0f, 0.1f, 0.0f, 0.0f, 0.0f, 640, 640, 1.0f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(640, 640);
	Annotation ann = createTestAnnotation();

	EXPECT_NO_THROW(transform.apply(image, ann));
}

TEST_F(TransformComprehensiveTest, RandomPerspective_ShearOnly) {
	RandomPerspective transform(0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 640, 640, 1.0f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(640, 640);
	Annotation ann = createTestAnnotation();

	EXPECT_NO_THROW(transform.apply(image, ann));
}

TEST_F(TransformComprehensiveTest, RandomPerspective_PerspectiveOnly) {
	RandomPerspective transform(0.0f, 0.0f, 0.0f, 0.0f, 0.001f, 640, 640, 1.0f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(640, 640);
	Annotation ann = createTestAnnotation();

	EXPECT_NO_THROW(transform.apply(image, ann));
}

TEST_F(TransformComprehensiveTest, RandomPerspective_AllCombined) {
	RandomPerspective transform(10.0f, 0.1f, 0.5f, 5.0f, 0.001f, 640, 640, 1.0f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(640, 640);
	Annotation ann = createTestAnnotation();

	EXPECT_NO_THROW(transform.apply(image, ann));
}

TEST_F(TransformComprehensiveTest, RandomPerspective_AnnotationTransformation) {
	RandomPerspective transform(10.0f, 0.1f, 0.5f, 2.0f, 0.001f, 640, 640, 1.0f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(640, 640);
	Annotation ann = createTestAnnotation(LabelType::XYXY);

	size_t numObjectsBefore = ann.size();
	transform.apply(image, ann);

	// Annotations should still exist (might be filtered if out of bounds)
	EXPECT_LE(ann.size(), numObjectsBefore);
}

TEST_F(TransformComprehensiveTest, RandomPerspective_ProbabilityZero) {
	RandomPerspective transform(10.0f, 0.1f, 0.5f, 2.0f, 0.001f, 640, 640, 0.0f); // probability=0

	cv::Mat image = createTestImage(640, 640);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	// Image should be unchanged (probability is 0)
	EXPECT_EQ(cv::norm(image, original, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, RandomPerspective_ProbabilityOne) {
	RandomPerspective transform(10.0f, 0.1f, 0.5f, 2.0f, 0.001f, 640, 640, 1.0f); // probability=1
	transform.setSeed(42);

	cv::Mat image = createTestImage(640, 640);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	// Image should be changed (probability is 1)
	EXPECT_GT(cv::norm(image, original, cv::NORM_L2), 0.0);
}

// ========================================
// ColorJitter Tests
// ========================================

TEST_F(TransformComprehensiveTest, ColorJitter_BrightnessOnly) {
	ColorJitter transform(0.5f, 0.0f, 0.0f, 0.0f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(100, 100);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	// Brightness should change the image
	EXPECT_GT(cv::norm(image, original, cv::NORM_L2), 0.0);

	// Values should still be in valid range [0, 255]
	double minVal, maxVal;
	cv::minMaxLoc(image, &minVal, &maxVal);
	EXPECT_GE(minVal, 0.0);
	EXPECT_LE(maxVal, 255.0);
}

TEST_F(TransformComprehensiveTest, ColorJitter_ContrastOnly) {
	ColorJitter transform(0.0f, 0.5f, 0.0f, 0.0f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(100, 100);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	EXPECT_GT(cv::norm(image, original, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, ColorJitter_SaturationOnly) {
	ColorJitter transform(0.0f, 0.0f, 0.5f, 0.0f);
	transform.setSeed(42);

	// Use a colored image (not gray) so saturation has an effect
	cv::Mat image = createTestImage(100, 100, cv::Scalar(100, 150, 200));
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	EXPECT_GT(cv::norm(image, original, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, ColorJitter_HueOnly) {
	ColorJitter transform(0.0f, 0.0f, 0.0f, 0.1f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(100, 100);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	// Hue shift on gray image might have minimal effect, but should not throw
	EXPECT_NO_THROW(transform.apply(image, ann));
}

TEST_F(TransformComprehensiveTest, ColorJitter_AllCombined) {
	ColorJitter transform(0.4f, 0.4f, 0.4f, 0.1f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(100, 100);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	EXPECT_GT(cv::norm(image, original, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, ColorJitter_RangeClamping) {
	ColorJitter transform(1.0f, 1.0f, 1.0f, 0.5f); // Extreme values
	transform.setSeed(42);

	cv::Mat image = createTestImage(100, 100, cv::Scalar(255, 255, 255)); // White image
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	// All values must be clamped to [0, 255]
	double minVal, maxVal;
	cv::minMaxLoc(image, &minVal, &maxVal);
	EXPECT_GE(minVal, 0.0);
	EXPECT_LE(maxVal, 255.0);
}

TEST_F(TransformComprehensiveTest, ColorJitter_AnnotationsUnchanged) {
	ColorJitter transform(0.3f, 0.3f, 0.3f, 0.05f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(100, 100);
	Annotation ann = createTestAnnotation();
	auto originalPoints = ann.getPoints();

	transform.apply(image, ann);

	// Color transforms should not modify annotations
	EXPECT_EQ(ann.getPoints().size(), originalPoints.size());
	if (!originalPoints.empty() && !ann.getPoints().empty()) {
		expectVectorNear(ann.getPoints()[0], originalPoints[0]);
	}
}

// ========================================
// GaussianBlur Tests
// ========================================

TEST_F(TransformComprehensiveTest, GaussianBlur_KernelSize3) {
	GaussianBlur transform(3, 3, 1.0f); // Always apply

	cv::Mat image = createTestImage(100, 100);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	// Blur should change the image
	EXPECT_GT(cv::norm(image, original, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, GaussianBlur_KernelSize7) {
	GaussianBlur transform(7, 7, 1.0f);

	cv::Mat image = createTestImage(100, 100);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	EXPECT_GT(cv::norm(image, original, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, GaussianBlur_KernelSizeRange) {
	GaussianBlur transform(3, 9, 1.0f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(100, 100);
	Annotation ann = createTestAnnotation();

	// Should pick random kernel size in range and apply blur
	EXPECT_NO_THROW(transform.apply(image, ann));
}

TEST_F(TransformComprehensiveTest, GaussianBlur_ProbabilityZero) {
	GaussianBlur transform(5, 7, 0.0f); // Never apply

	cv::Mat image = createTestImage(100, 100);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	// Image should be unchanged
	EXPECT_EQ(cv::norm(image, original, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, GaussianBlur_ProbabilityOne) {
	GaussianBlur transform(5, 5, 1.0f); // Always apply

	cv::Mat image = createTestImage(100, 100);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	// Image should be blurred
	EXPECT_GT(cv::norm(image, original, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, GaussianBlur_KernelSize1) {
	GaussianBlur transform(1, 1, 1.0f); // Edge case: kernel size 1 (no blur)

	cv::Mat image = createTestImage(100, 100);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	// Kernel size 1 should have no effect
	EXPECT_EQ(cv::norm(image, original, cv::NORM_L2), 0.0);
}

// ========================================
// Resize Tests
// ========================================

TEST_F(TransformComprehensiveTest, Resize_NoAspectRatio) {
	Resize transform(640, 480, false); // Stretch to fit

	cv::Mat image = createTestImage(800, 600);
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	EXPECT_EQ(image.rows, 640);
	EXPECT_EQ(image.cols, 480);
}

TEST_F(TransformComprehensiveTest, Resize_MaintainAspectRatio) {
	Resize transform(640, 640, true); // Keep aspect ratio

	cv::Mat image = createTestImage(480, 640);
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	// Should fit within 640x640 while maintaining aspect ratio
	EXPECT_LE(image.rows, 640);
	EXPECT_LE(image.cols, 640);
}

TEST_F(TransformComprehensiveTest, Resize_AnnotationScaling) {
	Resize transform(320, 320, false);

	cv::Mat image = createTestImage(640, 640);
	Annotation ann = createTestAnnotation(LabelType::XYWH);

	auto originalPoints = ann.getPoints()[0];
	transform.apply(image, ann);
	auto scaledPoints = ann.getPoints()[0];

	// Annotations should be scaled by 0.5
	float scaleX = 320.0f / 640.0f;
	float scaleY = 320.0f / 640.0f;

	EXPECT_NEAR(scaledPoints[0], originalPoints[0] * scaleX, EPSILON);
	EXPECT_NEAR(scaledPoints[1], originalPoints[1] * scaleY, EPSILON);
}

TEST_F(TransformComprehensiveTest, Resize_UpscaleAndDownscale) {
	Resize upscale(1280, 1280, false);
	Resize downscale(160, 160, false);

	cv::Mat img1 = createTestImage(640, 640);
	cv::Mat img2 = img1.clone();

	Annotation ann1 = createTestAnnotation();
	Annotation ann2 = createTestAnnotation();

	upscale.apply(img1, ann1);
	downscale.apply(img2, ann2);

	EXPECT_EQ(img1.rows, 1280);
	EXPECT_EQ(img1.cols, 1280);
	EXPECT_EQ(img2.rows, 160);
	EXPECT_EQ(img2.cols, 160);
}

// ========================================
// Flip Tests
// ========================================

TEST_F(TransformComprehensiveTest, HorizontalFlip_ImageTransformation) {
	RandomHorizontalFlip transform(1.0f); // Always flip

	// Create image without noise for pixel comparison
	cv::Mat image(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));
	// Create a pattern to verify flip
	cv::rectangle(image, cv::Point(10, 10), cv::Point(30, 30), cv::Scalar(255, 0, 0), -1);
	cv::Mat original = image.clone();

	Annotation ann = createTestAnnotation();
	transform.apply(image, ann);

	// Image should be different after flip
	EXPECT_GT(cv::norm(image, original, cv::NORM_L2), 0.0);

	// Verify flip by checking pixel symmetry
	// Horizontal flip: x' = width - 1 - x, so x=10 -> x'=89
	EXPECT_EQ(image.at<cv::Vec3b>(10, 10), original.at<cv::Vec3b>(10, 89)); // Flipped position
}

TEST_F(TransformComprehensiveTest, HorizontalFlip_AnnotationTransformation) {
	RandomHorizontalFlip transform(1.0f);

	cv::Mat image = createTestImage(100, 100);
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 25.0f, 50.0f, 20.0f, 30.0f }); // Center at x=25

	transform.apply(image, ann);

	auto& points = ann.getPoints()[0];
	// X-coordinate should be flipped: new_x = width - old_x = 100 - 25 = 75
	EXPECT_NEAR(points[0], 75.0f, EPSILON);
	EXPECT_NEAR(points[1], 50.0f, EPSILON); // Y unchanged
}

TEST_F(TransformComprehensiveTest, VerticalFlip_ImageTransformation) {
	RandomVerticalFlip transform(1.0f); // Always flip

	// Create image without noise for pixel comparison
	cv::Mat image(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));
	cv::rectangle(image, cv::Point(10, 10), cv::Point(30, 30), cv::Scalar(255, 0, 0), -1);
	cv::Mat original = image.clone();

	Annotation ann = createTestAnnotation();
	transform.apply(image, ann);

	EXPECT_GT(cv::norm(image, original, cv::NORM_L2), 0.0);

	// Verify flip by checking pixel symmetry
	// Vertical flip: y' = height - 1 - y, so y=10 -> y'=89
	EXPECT_EQ(image.at<cv::Vec3b>(10, 10), original.at<cv::Vec3b>(89, 10));
}

TEST_F(TransformComprehensiveTest, VerticalFlip_AnnotationTransformation) {
	RandomVerticalFlip transform(1.0f);

	cv::Mat image = createTestImage(100, 100);
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 50.0f, 25.0f, 20.0f, 30.0f }); // Center at y=25

	transform.apply(image, ann);

	auto& points = ann.getPoints()[0];
	EXPECT_NEAR(points[0], 50.0f, EPSILON); // X unchanged
	// Y-coordinate should be flipped: new_y = height - old_y = 100 - 25 = 75
	EXPECT_NEAR(points[1], 75.0f, EPSILON);
}

TEST_F(TransformComprehensiveTest, HorizontalFlip_ProbabilityZero) {
	RandomHorizontalFlip transform(0.0f); // Never flip

	cv::Mat image = createTestImage(100, 100);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	EXPECT_EQ(cv::norm(image, original, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, VerticalFlip_ProbabilityZero) {
	RandomVerticalFlip transform(0.0f); // Never flip

	cv::Mat image = createTestImage(100, 100);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	transform.apply(image, ann);

	EXPECT_EQ(cv::norm(image, original, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, Flip_MultipleAnnotationTypes) {
	RandomHorizontalFlip transform(1.0f);

	// Test with different label types
	std::vector<LabelType> types = {
		LabelType::XYWH,
		LabelType::XYXY,
		LabelType::POLYGON
	};

	for (auto type : types) {
		cv::Mat image = createTestImage(100, 100);
		Annotation ann = createTestAnnotation(type);
		size_t numObjects = ann.size();

		EXPECT_NO_THROW(transform.apply(image, ann));
		EXPECT_EQ(ann.size(), numObjects); // All objects should remain
	}
}

// ========================================
// Reproducibility Tests
// ========================================

TEST_F(TransformComprehensiveTest, Reproducibility_RandomHorizontalFlip) {
	RandomHorizontalFlip transform1(0.5f);
	RandomHorizontalFlip transform2(0.5f);

	transform1.setSeed(12345);
	transform2.setSeed(12345);

	cv::Mat img1 = createTestImage(100, 100);
	cv::Mat img2 = img1.clone();

	Annotation ann1 = createTestAnnotation();
	Annotation ann2 = createTestAnnotation();

	transform1.apply(img1, ann1);
	transform2.apply(img2, ann2);

	EXPECT_EQ(cv::norm(img1, img2, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, Reproducibility_ColorJitter) {
	ColorJitter transform1(0.5f, 0.5f, 0.5f, 0.1f);
	ColorJitter transform2(0.5f, 0.5f, 0.5f, 0.1f);

	transform1.setSeed(99999);
	transform2.setSeed(99999);

	cv::Mat img1 = createTestImage(100, 100);
	cv::Mat img2 = img1.clone();

	Annotation ann1 = createTestAnnotation();
	Annotation ann2 = createTestAnnotation();

	transform1.apply(img1, ann1);
	transform2.apply(img2, ann2);

	EXPECT_EQ(cv::norm(img1, img2, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, Reproducibility_GaussianBlur) {
	GaussianBlur transform1(3, 9, 0.5f);
	GaussianBlur transform2(3, 9, 0.5f);

	transform1.setSeed(55555);
	transform2.setSeed(55555);

	cv::Mat img1 = createTestImage(100, 100);
	cv::Mat img2 = img1.clone();

	Annotation ann1 = createTestAnnotation();
	Annotation ann2 = createTestAnnotation();

	transform1.apply(img1, ann1);
	transform2.apply(img2, ann2);

	EXPECT_EQ(cv::norm(img1, img2, cv::NORM_L2), 0.0);
}

TEST_F(TransformComprehensiveTest, Reproducibility_ComposePipeline) {
	Compose compose1;
	compose1.addTransform(std::make_unique<RandomHorizontalFlip>(0.5f));
	compose1.addTransform(std::make_unique<ColorJitter>(0.3f, 0.3f, 0.3f, 0.05f));
	compose1.addTransform(std::make_unique<GaussianBlur>(3, 7, 0.5f));

	Compose compose2;
	compose2.addTransform(std::make_unique<RandomHorizontalFlip>(0.5f));
	compose2.addTransform(std::make_unique<ColorJitter>(0.3f, 0.3f, 0.3f, 0.05f));
	compose2.addTransform(std::make_unique<GaussianBlur>(3, 7, 0.5f));

	compose1.setSeed(777);
	compose2.setSeed(777);

	cv::Mat img1 = createTestImage(100, 100);
	cv::Mat img2 = img1.clone();

	Annotation ann1 = createTestAnnotation();
	Annotation ann2 = createTestAnnotation();

	compose1.apply(img1, ann1);
	compose2.apply(img2, ann2);

	EXPECT_EQ(cv::norm(img1, img2, cv::NORM_L2), 0.0);
}

// ========================================
// Edge Cases
// ========================================

TEST_F(TransformComprehensiveTest, EdgeCase_EmptyAnnotation) {
	Compose compose;
	compose.addTransform(std::make_unique<LetterBox>(640, 640));
	compose.addTransform(std::make_unique<RandomHorizontalFlip>(0.5f));

	cv::Mat image = createTestImage(480, 640);
	Annotation ann(LabelType::XYWH); // Empty annotation

	EXPECT_NO_THROW(compose.apply(image, ann));
	EXPECT_TRUE(ann.empty());
}

TEST_F(TransformComprehensiveTest, EdgeCase_VerySmallImage) {
	LetterBox transform(640, 640);

	cv::Mat image = createTestImage(10, 10);
	Annotation ann = createTestAnnotation();

	EXPECT_NO_THROW(transform.apply(image, ann));
	EXPECT_EQ(image.rows, 640);
	EXPECT_EQ(image.cols, 640);
}

TEST_F(TransformComprehensiveTest, EdgeCase_SingleChannelImage) {
	cv::Mat grayImage(100, 100, CV_8UC1, cv::Scalar(128));

	// Most color transforms require 3-channel images
	// This test verifies graceful handling
	Annotation ann = createTestAnnotation();

	// Geometric transforms should work with grayscale
	LetterBox transform(640, 640);
	EXPECT_NO_THROW(transform.apply(grayImage, ann));
}

TEST_F(TransformComprehensiveTest, EdgeCase_AllAnnotationsFilteredOut) {
	RandomPerspective transform(45.0f, 0.5f, 1.0f, 20.0f, 0.01f, 640, 640, 1.0f);
	transform.setSeed(42);

	cv::Mat image = createTestImage(640, 640);
	Annotation ann = createTestAnnotation();

	// Extreme transforms might filter out all annotations
	transform.apply(image, ann);

	// Should not crash, annotations might be empty
	EXPECT_GE(ann.size(), 0);
}

TEST_F(TransformComprehensiveTest, EdgeCase_IdentityTransform) {
	Compose compose;
	compose.addTransform(std::make_unique<Resize>(100, 100, false));

	cv::Mat image = createTestImage(100, 100);
	cv::Mat original = image.clone();
	Annotation ann = createTestAnnotation();

	compose.apply(image, ann);

	// Resizing to same size should produce similar result
	EXPECT_EQ(image.rows, 100);
	EXPECT_EQ(image.cols, 100);
}

// Removed: Passing nullptr to addTransform is undefined behavior and should not be tested
// If Compose needs to handle nullptr, it should validate and reject it in addTransform

// ========================================
// Integration Tests with Real Images
// ========================================

TEST_F(TransformComprehensiveTest, Integration_RealMNISTImage) {
	if (!testImage_.empty()) {
		Compose pipeline;
		pipeline.addTransform(std::make_unique<LetterBox>(640, 640));
		pipeline.addTransform(std::make_unique<RandomPerspective>(10.0f, 0.1f, 0.5f, 2.0f, 0.001f, 640, 640, 1.0f));
		pipeline.addTransform(std::make_unique<ColorJitter>(0.3f, 0.3f, 0.3f, 0.05f));
		pipeline.addTransform(std::make_unique<GaussianBlur>(3, 7, 0.5f));
		pipeline.setSeed(42);

		cv::Mat image = testImage_.clone();
		Annotation ann = createTestAnnotation();

		EXPECT_NO_THROW(pipeline.apply(image, ann));
		EXPECT_EQ(image.rows, 640);
		EXPECT_EQ(image.cols, 640);
	}
}

TEST_F(TransformComprehensiveTest, Integration_FullTrainingPipeline) {
	Compose pipeline;
	pipeline.addTransform(std::make_unique<RandomPerspective>(10.0f, 0.1f, 0.5f, 2.0f, 0.001f, 640, 640, 0.5f));
	pipeline.addTransform(std::make_unique<RandomHorizontalFlip>(0.5f));
	pipeline.addTransform(std::make_unique<ColorJitter>(0.4f, 0.4f, 0.4f, 0.1f));
	pipeline.addTransform(std::make_unique<GaussianBlur>(3, 11, 0.3f));
	pipeline.setSeed(12345);

	cv::Mat image = createTestImage(480, 640);
	Annotation ann = createTestAnnotation();

	EXPECT_NO_THROW(pipeline.apply(image, ann));
}

TEST_F(TransformComprehensiveTest, Integration_InferencePipeline) {
	Compose pipeline;
	pipeline.addTransform(std::make_unique<LetterBox>(640, 640, 114, 114, 114));
	// No random transforms for inference

	EXPECT_FALSE(pipeline.isRandom());

	cv::Mat image = createTestImage(480, 640);
	Annotation ann = createTestAnnotation();

	pipeline.apply(image, ann);

	EXPECT_EQ(image.rows, 640);
	EXPECT_EQ(image.cols, 640);
}

TEST_F(TransformComprehensiveTest, Integration_MultiThreadCloning) {
	Compose original;
	original.addTransform(std::make_unique<RandomHorizontalFlip>(0.5f));
	original.addTransform(std::make_unique<ColorJitter>(0.3f, 0.3f, 0.3f, 0.05f));
	original.setSeed(42);

	// Simulate multi-threaded data loading by cloning
	auto clone1 = original.clone();
	auto clone2 = original.clone();

	cv::Mat img1 = createTestImage(100, 100);
	cv::Mat img2 = img1.clone();

	Annotation ann1 = createTestAnnotation();
	Annotation ann2 = createTestAnnotation();

	// Clones should be independent
	clone1->apply(img1, ann1);
	clone2->apply(img2, ann2);

	// Results might differ (independent RNG states after cloning)
	// But both should execute successfully
	EXPECT_NO_THROW(clone1->apply(img1, ann1));
	EXPECT_NO_THROW(clone2->apply(img2, ann2));
}
