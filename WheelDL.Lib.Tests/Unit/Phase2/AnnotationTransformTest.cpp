#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Common/Annotation.h"
#include "WheelDL.Lib/Data/Utils/ImageIO.h"
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace WheelDL::Data;
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

/**
 * @class AnnotationTransformTest
 * @brief Comprehensive tests for Annotation transformations
 *
 * Tests all annotation format conversions, coordinate transformations,
 * edge cases, and Mosaic-specific operations.
 */
class AnnotationTransformTest : public ::testing::Test
{
protected:
	const float EPSILON = 1e-4f;

	void SetUp() override
	{
		// Load a test MNIST image for use in transform tests
		auto mnistPath = getTestDataPath() / "mnist_sample" / "images";
		auto imageFiles = getImageFilesInDirectory(mnistPath);
		if (!imageFiles.empty()) {
			testImage_ = ImageIO::loadImage(imageFiles[0], 640);
		}
		else {
			// Create a synthetic test image if MNIST not available
			testImage_ = cv::Mat(28, 28, CV_8UC3, cv::Scalar(128, 128, 128));
		}
	}

	std::vector<std::string> getImageFilesInDirectory(const fs::path& dir)
	{
		std::vector<std::string> files;
		if (!fs::exists(dir)) return files;

		for (const auto& entry : fs::directory_iterator(dir)) {
			if (entry.is_regular_file()) {
				auto ext = entry.path().extension().string();
				if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
					files.push_back(entry.path().string());
					if (files.size() >= 10) break; // Only need a few
				}
			}
		}
		return files;
	}

	void expectVectorNear(const std::vector<float>& actual, const std::vector<float>& expected, float epsilon = 1e-4f)
	{
		ASSERT_EQ(actual.size(), expected.size()) << "Vector sizes differ";
		for (size_t i = 0; i < actual.size(); ++i) {
			EXPECT_NEAR(actual[i], expected[i], epsilon) << "Mismatch at index " << i;
		}
	}

	cv::Mat testImage_;
};

// ========== Format Conversion Tests: XYXY ??XYWH ==========

TEST_F(AnnotationTransformTest, ConvertXYXYToXYWH)
{
	Annotation ann(LabelType::XYXY);
	ann.addObject(0, { 50.0f, 60.0f, 150.0f, 160.0f }); // x1, y1, x2, y2

	ann.convertTo(LabelType::XYWH);

	EXPECT_EQ(ann.getLabelType(), LabelType::XYWH);
	// Expected: center=(100, 110), size=(100, 100)
	expectVectorNear(ann.getPoints()[0], { 100.0f, 110.0f, 100.0f, 100.0f });
}

TEST_F(AnnotationTransformTest, ConvertXYWHToXYXY)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 100.0f, 110.0f, 100.0f, 100.0f }); // cx, cy, w, h

	ann.convertTo(LabelType::XYXY);

	EXPECT_EQ(ann.getLabelType(), LabelType::XYXY);
	expectVectorNear(ann.getPoints()[0], { 50.0f, 60.0f, 150.0f, 160.0f });
}

TEST_F(AnnotationTransformTest, ConvertXYXYToXYWH_RoundTrip)
{
	Annotation ann(LabelType::XYXY);
	std::vector<float> original = { 10.0f, 20.0f, 90.0f, 80.0f };
	ann.addObject(0, original);

	ann.convertTo(LabelType::XYWH);
	ann.convertTo(LabelType::XYXY);

	expectVectorNear(ann.getPoints()[0], original);
}

// ========== Format Conversion Tests: XYXY ??POLYGON ==========

TEST_F(AnnotationTransformTest, ConvertXYXYToPolygon)
{
	Annotation ann(LabelType::XYXY);
	ann.addObject(0, { 10.0f, 20.0f, 100.0f, 200.0f });

	ann.convertTo(LabelType::POLYGON);

	EXPECT_EQ(ann.getLabelType(), LabelType::POLYGON);
	// Should have 4 corners (8 values): top-left, top-right, bottom-right, bottom-left
	auto& points = ann.getPoints()[0];
	EXPECT_EQ(points.size(), 8);
	expectVectorNear(points, { 10.0f, 20.0f, 100.0f, 20.0f, 100.0f, 200.0f, 10.0f, 200.0f });
}

TEST_F(AnnotationTransformTest, ConvertPolygonToXYXY)
{
	Annotation ann(LabelType::POLYGON);
	// Irregular polygon
	ann.addObject(0, { 10.0f, 20.0f, 100.0f, 30.0f, 90.0f, 200.0f, 5.0f, 180.0f });

	ann.convertTo(LabelType::XYXY);

	EXPECT_EQ(ann.getLabelType(), LabelType::XYXY);
	// Bounding box: min=(5, 20), max=(100, 200)
	expectVectorNear(ann.getPoints()[0], { 5.0f, 20.0f, 100.0f, 200.0f });
}

TEST_F(AnnotationTransformTest, ConvertPolygonToXYXY_Triangle)
{
	Annotation ann(LabelType::POLYGON);
	// Triangle: 3 points (6 values)
	ann.addObject(0, { 50.0f, 10.0f, 10.0f, 90.0f, 90.0f, 90.0f });

	ann.convertTo(LabelType::XYXY);

	// Bounding box: min=(10, 10), max=(90, 90)
	expectVectorNear(ann.getPoints()[0], { 10.0f, 10.0f, 90.0f, 90.0f });
}

// ========== Format Conversion Tests: XYWH ??POLYGON ==========

TEST_F(AnnotationTransformTest, ConvertXYWHToPolygon)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 50.0f, 50.0f, 40.0f, 60.0f }); // center=(50,50), size=40x60

	ann.convertTo(LabelType::POLYGON);

	auto& points = ann.getPoints()[0];
	EXPECT_EQ(points.size(), 8);
	// corners: (30,20), (70,20), (70,80), (30,80)
	expectVectorNear(points, { 30.0f, 20.0f, 70.0f, 20.0f, 70.0f, 80.0f, 30.0f, 80.0f });
}

TEST_F(AnnotationTransformTest, ConvertPolygonToXYWH)
{
	Annotation ann(LabelType::POLYGON);
	ann.addObject(0, { 30.0f, 20.0f, 70.0f, 20.0f, 70.0f, 80.0f, 30.0f, 80.0f });

	ann.convertTo(LabelType::XYWH);

	// Bounding box: (30,20) to (70,80) -> center=(50,50), size=40x60
	expectVectorNear(ann.getPoints()[0], { 50.0f, 50.0f, 40.0f, 60.0f });
}

// ========== Format Conversion Tests: XYXYXYXY ??XYWHR ==========

TEST_F(AnnotationTransformTest, ConvertXYXYXYXYToXYWHR_AxisAligned)
{
	Annotation ann(LabelType::XYXYXYXY);
	// Axis-aligned rectangle corners
	ann.addObject(0, { 10.0f, 20.0f, 50.0f, 20.0f, 50.0f, 60.0f, 10.0f, 60.0f });

	ann.convertTo(LabelType::XYWHR);

	auto& points = ann.getPoints()[0];
	EXPECT_EQ(points.size(), 5);
	// Center should be (30, 40), width=40, height=40
	EXPECT_NEAR(points[0], 30.0f, EPSILON);
	EXPECT_NEAR(points[1], 40.0f, EPSILON);
	EXPECT_NEAR(points[2], 40.0f, EPSILON);
	EXPECT_NEAR(points[3], 40.0f, EPSILON);
	// Angle should be ~0 (axis-aligned)
	EXPECT_NEAR(points[4], 0.0f, 0.1f);
}

TEST_F(AnnotationTransformTest, ConvertXYWHRToXYXYXYXY_NoRotation)
{
	Annotation ann(LabelType::XYWHR);
	// Center=(50,50), size=40x60, rotation=0
	ann.addObject(0, { 50.0f, 50.0f, 40.0f, 60.0f, 0.0f });

	ann.convertTo(LabelType::XYXYXYXY);

	auto& points = ann.getPoints()[0];
	EXPECT_EQ(points.size(), 8);
	// At rotation=0, corners should be at (±w/2, ±h/2) relative to center
	// Expected: (30,20), (70,20), (70,80), (30,80)
	EXPECT_NEAR(points[0], 30.0f, EPSILON);
	EXPECT_NEAR(points[1], 20.0f, EPSILON);
	EXPECT_NEAR(points[2], 70.0f, EPSILON);
	EXPECT_NEAR(points[3], 20.0f, EPSILON);
}

TEST_F(AnnotationTransformTest, ConvertXYWHRToXYXYXYXY_WithRotation)
{
	Annotation ann(LabelType::XYWHR);
	// Center=(50,50), size=40x40, rotation=?/4 (45 degrees)
	ann.addObject(0, { 50.0f, 50.0f, 40.0f, 40.0f, 0.785398f });

	ann.convertTo(LabelType::XYXYXYXY);

	auto& points = ann.getPoints()[0];
	EXPECT_EQ(points.size(), 8);
	// Rotated corners should be different from axis-aligned
	// Just verify we got 8 values and center is still roughly (50, 50)
	float centerX = (points[0] + points[2] + points[4] + points[6]) / 4.0f;
	float centerY = (points[1] + points[3] + points[5] + points[7]) / 4.0f;
	EXPECT_NEAR(centerX, 50.0f, 1.0f);
	EXPECT_NEAR(centerY, 50.0f, 1.0f);
}

// ========== Format Conversion Tests: POLYGON ??XYWHR ==========

TEST_F(AnnotationTransformTest, ConvertPolygonToXYWHR_AxisAligned)
{
	Annotation ann(LabelType::POLYGON);
	// Axis-aligned rectangle
	ann.addObject(0, { 10.0f, 10.0f, 50.0f, 10.0f, 50.0f, 30.0f, 10.0f, 30.0f });

	ann.convertTo(LabelType::XYWHR);

	auto& points = ann.getPoints()[0];
	EXPECT_EQ(points.size(), 5);
	// For axis-aligned rectangle, center=(30,20), w=40, h=20, angle??
	EXPECT_NEAR(points[0], 30.0f, 1.0f); // cx
	EXPECT_NEAR(points[1], 20.0f, 1.0f); // cy
	EXPECT_GT(points[2], 0.0f);          // w > 0
	EXPECT_GT(points[3], 0.0f);          // h > 0
}

TEST_F(AnnotationTransformTest, ConvertPolygonToXYXYXYXY)
{
	Annotation ann(LabelType::POLYGON);
	ann.addObject(0, { 10.0f, 10.0f, 50.0f, 10.0f, 50.0f, 30.0f, 10.0f, 30.0f });

	ann.convertTo(LabelType::XYXYXYXY);

	// Should produce 4 corners of minimum rotated box
	auto& points = ann.getPoints()[0];
	EXPECT_EQ(points.size(), 8);
}

// ========== Coordinate Transformation Tests ==========

TEST_F(AnnotationTransformTest, Scale_XYWH)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 100.0f, 200.0f, 50.0f, 100.0f });

	ann.scale(2.0f, 0.5f);

	// Both center and size are scaled
	expectVectorNear(ann.getPoints()[0], { 200.0f, 100.0f, 100.0f, 50.0f });
}

TEST_F(AnnotationTransformTest, Scale_XYXY)
{
	Annotation ann(LabelType::XYXY);
	ann.addObject(0, { 10.0f, 20.0f, 110.0f, 220.0f });

	ann.scale(2.0f, 0.5f);

	// All coordinates are scaled
	expectVectorNear(ann.getPoints()[0], { 20.0f, 10.0f, 220.0f, 110.0f });
}

TEST_F(AnnotationTransformTest, Scale_XYWHR_NonUniform)
{
	Annotation ann(LabelType::XYWHR);
	ann.addObject(0, { 100.0f, 100.0f, 40.0f, 60.0f, 0.5f });

	ann.scale(2.0f, 3.0f);

	auto& points = ann.getPoints()[0];
	// Center and size are scaled, rotation is recalculated
	EXPECT_NEAR(points[0], 200.0f, EPSILON); // cx scaled by 2.0
	EXPECT_NEAR(points[1], 300.0f, EPSILON); // cy scaled by 3.0
	// Width and height scaled, but rotation may change due to non-uniform scaling
}

TEST_F(AnnotationTransformTest, Translate_XYWH)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 100.0f, 200.0f, 50.0f, 100.0f });

	ann.translate(10.0f, -20.0f);

	// Only center is translated, not size
	expectVectorNear(ann.getPoints()[0], { 110.0f, 180.0f, 50.0f, 100.0f });
}

TEST_F(AnnotationTransformTest, Translate_XYXY)
{
	Annotation ann(LabelType::XYXY);
	ann.addObject(0, { 10.0f, 20.0f, 110.0f, 220.0f });

	ann.translate(5.0f, -10.0f);

	// All coordinates are translated
	expectVectorNear(ann.getPoints()[0], { 15.0f, 10.0f, 115.0f, 210.0f });
}

TEST_F(AnnotationTransformTest, Translate_POLYGON)
{
	Annotation ann(LabelType::POLYGON);
	ann.addObject(0, { 10.0f, 20.0f, 50.0f, 30.0f, 40.0f, 60.0f });

	ann.translate(100.0f, 200.0f);

	expectVectorNear(ann.getPoints()[0], { 110.0f, 220.0f, 150.0f, 230.0f, 140.0f, 260.0f });
}

TEST_F(AnnotationTransformTest, FlipHorizontal_XYWH)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 100.0f, 100.0f, 50.0f, 50.0f });

	ann.flipHorizontal(640);

	// cx should be flipped: 640 - 100 = 540
	expectVectorNear(ann.getPoints()[0], { 540.0f, 100.0f, 50.0f, 50.0f });
}

TEST_F(AnnotationTransformTest, FlipHorizontal_XYXY)
{
	Annotation ann(LabelType::XYXY);
	ann.addObject(0, { 100.0f, 100.0f, 200.0f, 200.0f });

	ann.flipHorizontal(640);

	// x coordinates should be flipped
	auto& points = ann.getPoints()[0];
	EXPECT_NEAR(points[0], 540.0f, EPSILON); // 640 - 100
	EXPECT_NEAR(points[2], 440.0f, EPSILON); // 640 - 200
	// y coordinates unchanged
	EXPECT_NEAR(points[1], 100.0f, EPSILON);
	EXPECT_NEAR(points[3], 200.0f, EPSILON);
}

TEST_F(AnnotationTransformTest, FlipVertical_XYWH)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 100.0f, 100.0f, 50.0f, 50.0f });

	ann.flipVertical(480);

	// cy should be flipped: 480 - 100 = 380
	expectVectorNear(ann.getPoints()[0], { 100.0f, 380.0f, 50.0f, 50.0f });
}

TEST_F(AnnotationTransformTest, FlipVertical_XYXY)
{
	Annotation ann(LabelType::XYXY);
	ann.addObject(0, { 100.0f, 100.0f, 200.0f, 200.0f });

	ann.flipVertical(480);

	// y coordinates should be flipped
	auto& points = ann.getPoints()[0];
	EXPECT_NEAR(points[0], 100.0f, EPSILON); // x unchanged
	EXPECT_NEAR(points[2], 200.0f, EPSILON); // x unchanged
	EXPECT_NEAR(points[1], 380.0f, EPSILON); // 480 - 100
	EXPECT_NEAR(points[3], 280.0f, EPSILON); // 480 - 200
}

TEST_F(AnnotationTransformTest, FlipHorizontal_XYWHR_NegatesRotation)
{
	Annotation ann(LabelType::XYWHR);
	ann.addObject(0, { 100.0f, 100.0f, 50.0f, 50.0f, 0.5f });

	ann.flipHorizontal(640);

	auto& points = ann.getPoints()[0];
	EXPECT_NEAR(points[0], 540.0f, EPSILON);  // cx flipped
	EXPECT_NEAR(points[1], 100.0f, EPSILON);  // cy unchanged
	EXPECT_NEAR(points[4], -0.5f, EPSILON);   // rotation negated
}

// ========== Normalize/Denormalize Tests ==========

TEST_F(AnnotationTransformTest, NormalizeDenormalize_Idempotency_XYWH)
{
	Annotation ann(LabelType::XYWH);
	std::vector<float> original = { 320.0f, 240.0f, 100.0f, 80.0f };
	ann.addObject(0, original);

	ann.normalize(640, 480);
	ann.denormalize(640, 480);

	expectVectorNear(ann.getPoints()[0], original);
}

TEST_F(AnnotationTransformTest, NormalizeDenormalize_Idempotency_XYXY)
{
	Annotation ann(LabelType::XYXY);
	std::vector<float> original = { 100.0f, 80.0f, 540.0f, 400.0f };
	ann.addObject(0, original);

	ann.normalize(640, 480);
	ann.denormalize(640, 480);

	expectVectorNear(ann.getPoints()[0], original);
}

TEST_F(AnnotationTransformTest, NormalizeDenormalize_Idempotency_POLYGON)
{
	Annotation ann(LabelType::POLYGON);
	std::vector<float> original = { 100.0f, 80.0f, 540.0f, 80.0f, 540.0f, 400.0f, 100.0f, 400.0f };
	ann.addObject(0, original);

	ann.normalize(640, 480);
	ann.denormalize(640, 480);

	expectVectorNear(ann.getPoints()[0], original);
}

TEST_F(AnnotationTransformTest, Normalize_IsIdempotent)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 320.0f, 240.0f, 100.0f, 80.0f });

	ann.normalize(640, 480);
	auto afterFirst = ann.getPoints()[0];

	ann.normalize(640, 480); // Second normalize should do nothing
	auto afterSecond = ann.getPoints()[0];

	expectVectorNear(afterFirst, afterSecond);
}

TEST_F(AnnotationTransformTest, Denormalize_IsIdempotent)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 0.5f, 0.5f, 0.15625f, 0.16667f });
	ann.normalize(640, 480); // Mark as normalized

	ann.denormalize(640, 480);
	auto afterFirst = ann.getPoints()[0];

	ann.denormalize(640, 480); // Second denormalize should do nothing
	auto afterSecond = ann.getPoints()[0];

	expectVectorNear(afterFirst, afterSecond);
}

// ========== clipToBounds Tests (Mosaic specific) ==========

TEST_F(AnnotationTransformTest, ClipToBounds_PartiallyOutside)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 50.0f, 50.0f, 100.0f, 100.0f }); // extends beyond (0, 0, 100, 100)

	ann.clipToBounds(0.0f, 0.0f, 100.0f, 100.0f);

	// Object should be clipped to fit within bounds
	EXPECT_GT(ann.size(), 0); // Should not be removed
	auto& points = ann.getPoints()[0];
	// Center should be adjusted to fit clipped box
	EXPECT_GT(points[0], 0.0f);
	EXPECT_LT(points[0], 100.0f);
}

TEST_F(AnnotationTransformTest, ClipToBounds_CompletelyOutside)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 200.0f, 200.0f, 50.0f, 50.0f }); // completely outside (0, 0, 100, 100)

	ann.clipToBounds(0.0f, 0.0f, 100.0f, 100.0f);

	// Object should be removed
	EXPECT_EQ(ann.size(), 0);
}

TEST_F(AnnotationTransformTest, ClipToBounds_CompletelyInside)
{
	Annotation ann(LabelType::XYWH);
	std::vector<float> original = { 50.0f, 50.0f, 20.0f, 20.0f };
	ann.addObject(0, original); // completely inside (0, 0, 100, 100)

	ann.clipToBounds(0.0f, 0.0f, 100.0f, 100.0f);

	// Object should remain unchanged
	EXPECT_EQ(ann.size(), 1);
	expectVectorNear(ann.getPoints()[0], original);
}

TEST_F(AnnotationTransformTest, ClipToBounds_MultipleObjects_MixedClipping)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 50.0f, 50.0f, 20.0f, 20.0f });   // inside
	ann.addObject(1, { 200.0f, 200.0f, 50.0f, 50.0f }); // outside
	ann.addObject(2, { 90.0f, 90.0f, 40.0f, 40.0f });   // partially outside

	ann.clipToBounds(0.0f, 0.0f, 100.0f, 100.0f);

	// First should remain, second should be removed, third should be clipped
	EXPECT_EQ(ann.size(), 2);
}

TEST_F(AnnotationTransformTest, ClipToBounds_XYXY_Format)
{
	Annotation ann(LabelType::XYXY);
	ann.addObject(0, { 10.0f, 10.0f, 150.0f, 150.0f }); // extends beyond (0, 0, 100, 100)

	ann.clipToBounds(0.0f, 0.0f, 100.0f, 100.0f);

	EXPECT_EQ(ann.size(), 1);
	auto& points = ann.getPoints()[0];
	// Should be clipped to bounds
	EXPECT_GE(points[0], 0.0f);
	EXPECT_LE(points[2], 100.0f);
}

TEST_F(AnnotationTransformTest, ClipToBounds_MinAreaFilter)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 100.0f, 100.0f, 50.0f, 50.0f }); // will be clipped to tiny area

	ann.clipToBounds(95.0f, 95.0f, 105.0f, 105.0f, 200.0f); // minArea = 200 pixels (clipped box is 10x10=100, less than 200)

	// Object should be removed due to small area after clipping
	EXPECT_EQ(ann.size(), 0);
}

// ========== Edge Case Tests ==========

TEST_F(AnnotationTransformTest, EmptyAnnotation_Operations)
{
	Annotation ann(LabelType::XYWH);

	// Should not crash on empty
	EXPECT_NO_THROW(ann.normalize(640, 480));
	EXPECT_NO_THROW(ann.denormalize(640, 480));
	EXPECT_NO_THROW(ann.scale(2.0f, 2.0f));
	EXPECT_NO_THROW(ann.translate(10.0f, 10.0f));
	EXPECT_NO_THROW(ann.flipHorizontal(640));
	EXPECT_NO_THROW(ann.flipVertical(480));
	EXPECT_NO_THROW(ann.convertTo(LabelType::XYXY));
	EXPECT_NO_THROW(ann.clipToBounds(0.0f, 0.0f, 640.0f, 480.0f));
}

TEST_F(AnnotationTransformTest, SinglePointPolygon)
{
	Annotation ann(LabelType::POLYGON);
	ann.addObject(0, { 50.0f, 50.0f }); // Single point

	// Should handle gracefully
	EXPECT_NO_THROW(ann.scale(2.0f, 2.0f));
	EXPECT_NO_THROW(ann.translate(10.0f, 10.0f));
}

TEST_F(AnnotationTransformTest, DegeneratePolygon_Line)
{
	Annotation ann(LabelType::POLYGON);
	ann.addObject(0, { 50.0f, 50.0f, 100.0f, 50.0f }); // Horizontal line

	// Should handle gracefully
	EXPECT_NO_THROW(ann.convertTo(LabelType::XYXY));
	if (!ann.empty()) {
		auto& points = ann.getPoints()[0];
		EXPECT_EQ(points.size(), 4); // XYXY
		// Height should be 0
		EXPECT_NEAR(points[1], points[3], EPSILON);
	}
}

TEST_F(AnnotationTransformTest, DegenerateBox_ZeroWidth)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 50.0f, 50.0f, 0.0f, 50.0f }); // Zero width

	ann.validateAndClip(640, 480, 1.0f);

	// Should be removed due to zero area
	EXPECT_EQ(ann.size(), 0);
}

TEST_F(AnnotationTransformTest, DegenerateBox_ZeroHeight)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 50.0f, 50.0f, 50.0f, 0.0f }); // Zero height

	ann.validateAndClip(640, 480, 1.0f);

	// Should be removed due to zero area
	EXPECT_EQ(ann.size(), 0);
}

TEST_F(AnnotationTransformTest, NegativeDimensions_XYWH)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 50.0f, 50.0f, -50.0f, 50.0f }); // Negative width

	ann.validateAndClip(640, 480, 1.0f);

	// Should be removed
	EXPECT_EQ(ann.size(), 0);
}

TEST_F(AnnotationTransformTest, MultipleObjects_AllOperations)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 100.0f, 100.0f, 50.0f, 50.0f });
	ann.addObject(1, { 200.0f, 200.0f, 60.0f, 60.0f });
	ann.addObject(2, { 300.0f, 300.0f, 70.0f, 70.0f });

	// Apply various operations
	ann.scale(0.5f, 0.5f);
	ann.translate(10.0f, 20.0f);
	ann.normalize(640, 480);
	ann.denormalize(640, 480);
	ann.flipHorizontal(640);

	// Should maintain all objects
	EXPECT_EQ(ann.size(), 3);
	EXPECT_EQ(ann.getClasses().size(), 3);
	EXPECT_EQ(ann.getPoints().size(), 3);
}

TEST_F(AnnotationTransformTest, VeryLargeCoordinates)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 10000.0f, 10000.0f, 1000.0f, 1000.0f });

	// Should handle large values
	EXPECT_NO_THROW(ann.scale(2.0f, 2.0f));
	EXPECT_NO_THROW(ann.translate(5000.0f, 5000.0f));
}

TEST_F(AnnotationTransformTest, VerySmallCoordinates)
{
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 0.001f, 0.001f, 0.001f, 0.001f });

	// Should handle small values
	EXPECT_NO_THROW(ann.scale(2.0f, 2.0f));
	EXPECT_NO_THROW(ann.normalize(640, 480));
}

// ========== Perspective Transform Tests ==========

TEST_F(AnnotationTransformTest, PerspectiveTransform_Identity)
{
	if (testImage_.empty()) {
		std::cout << "SKIPPED: Test image not available" << std::endl;
		return;
	}

	cv::Mat image = testImage_.clone();
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 14.0f, 14.0f, 10.0f, 10.0f }); // Center of 28x28 image

	// Identity transform
	cv::Mat transform = cv::Mat::eye(3, 3, CV_64F);

	EXPECT_NO_THROW(ann.transform(image, transform));

	// Image and annotation should be roughly unchanged
	EXPECT_GT(ann.size(), 0);
}

TEST_F(AnnotationTransformTest, PerspectiveTransform_Translation)
{
	if (testImage_.empty()) {
		std::cout << "SKIPPED: Test image not available" << std::endl;
		return;
	}

	cv::Mat image = testImage_.clone();
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 14.0f, 14.0f, 10.0f, 10.0f });

	// Translation matrix
	cv::Mat transform = (cv::Mat_<double>(3, 3) <<
		1, 0, 10,
		0, 1, 20,
		0, 0, 1);

	EXPECT_NO_THROW(ann.transform(image, transform));
}

TEST_F(AnnotationTransformTest, PerspectiveTransform_Scale)
{
	if (testImage_.empty()) {
		std::cout << "SKIPPED: Test image not available" << std::endl;
		return;
	}

	cv::Mat image = testImage_.clone();
	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 14.0f, 14.0f, 10.0f, 10.0f });

	// Scale matrix (2x)
	cv::Mat transform = (cv::Mat_<double>(3, 3) <<
		2, 0, 0,
		0, 2, 0,
		0, 0, 1);

	EXPECT_NO_THROW(ann.transform(image, transform, cv::Size(56, 56)));

	EXPECT_EQ(image.rows, 56);
	EXPECT_EQ(image.cols, 56);
}

TEST_F(AnnotationTransformTest, PerspectiveTransform_FiltersInvalidObjects)
{
	// Use a small fixed-size image to ensure consistent behavior
	cv::Mat image(28, 28, CV_8UC3, cv::Scalar(128, 128, 128));

	Annotation ann(LabelType::XYWH);
	ann.addObject(0, { 14.0f, 14.0f, 10.0f, 10.0f });  // center - inside 28x28
	ann.addObject(1, { 50.0f, 50.0f, 10.0f, 10.0f });  // outside 28x28 image

	cv::Mat transform = cv::Mat::eye(3, 3, CV_64F);

	ann.transform(image, transform);

	// Second object should be filtered out because it's outside the image
	EXPECT_EQ(ann.size(), 1);
}

// ========== Clone Tests ==========

TEST_F(AnnotationTransformTest, Clone_DeepCopy)
{
	Annotation original(LabelType::XYWH);
	original.addObject(0, { 100.0f, 100.0f, 50.0f, 50.0f });
	original.addObject(1, { 200.0f, 200.0f, 60.0f, 60.0f });

	Annotation cloned = original.clone();

	// Modify clone
	cloned.scale(2.0f, 2.0f);

	// Original should be unchanged
	EXPECT_EQ(original.size(), 2);
	expectVectorNear(original.getPoints()[0], { 100.0f, 100.0f, 50.0f, 50.0f });

	// Clone should be modified
	EXPECT_EQ(cloned.size(), 2);
	expectVectorNear(cloned.getPoints()[0], { 200.0f, 200.0f, 100.0f, 100.0f });
}

// ========== Integration Test: Complex Transformation Chain ==========

TEST_F(AnnotationTransformTest, ComplexTransformationChain)
{
	Annotation ann(LabelType::XYXY);
	ann.addObject(0, { 100.0f, 100.0f, 200.0f, 200.0f });
	ann.addObject(1, { 300.0f, 300.0f, 400.0f, 400.0f });

	// Convert to XYWH
	ann.convertTo(LabelType::XYWH);
	EXPECT_EQ(ann.getLabelType(), LabelType::XYWH);

	// Scale
	ann.scale(0.5f, 0.5f);

	// Translate
	ann.translate(50.0f, 50.0f);

	// Normalize
	ann.normalize(640, 480);

	// Flip
	ann.flipHorizontal(1);

	// Denormalize
	ann.denormalize(640, 480);

	// Convert back to XYXY
	ann.convertTo(LabelType::XYXY);
	EXPECT_EQ(ann.getLabelType(), LabelType::XYXY);

	// Should still have both objects
	EXPECT_EQ(ann.size(), 2);
}
