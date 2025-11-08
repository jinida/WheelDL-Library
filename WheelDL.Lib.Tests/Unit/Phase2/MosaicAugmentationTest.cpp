#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Augmentation/DatasetAugmentation.h"
#include "WheelDL.Lib/Data/Common/Annotation.h"
#include "WheelDL.Lib/Data/Utils/ImageIO.h"
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace WheelDL::Data;
using namespace WheelDL::Data::Augmentation;
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
 * @class MosaicAugmentationTest
 * @brief Comprehensive tests for Mosaic augmentation
 *
 * Tests Mosaic with different image sizes, annotation types, probability,
 * center point randomization, and all dataset types (detection, OBB, segmentation).
 */
class MosaicAugmentationTest : public ::testing::Test
{
protected:
	std::vector<cv::Mat> testImages_;
	const float EPSILON = 1e-4f;

	void SetUp() override
	{
		// Load MNIST images for testing
		auto mnistPath = getTestDataPath() / "mnist_sample" / "images";
		auto imageFiles = getImageFilesInDirectory(mnistPath);

		// Load at least 4 images
		for (size_t i = 0; i < std::min(size_t(4), imageFiles.size()); ++i) {
			cv::Mat img = ImageIO::loadImage(imageFiles[i], 640);
			if (!img.empty()) {
				testImages_.push_back(img);
			}
		}

		// If we don't have enough MNIST images, create synthetic ones
		while (testImages_.size() < 4) {
			cv::Mat syntheticImage = createColoredImage(28, 28,
				cv::Scalar(100 + testImages_.size() * 30,
					128,
					200 - testImages_.size() * 30));
			testImages_.push_back(syntheticImage);
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
					if (files.size() >= 10) break;
				}
			}
		}
		return files;
	}

	cv::Mat createColoredImage(int rows, int cols, cv::Scalar color)
	{
		return cv::Mat(rows, cols, CV_8UC3, color);
	}

	Annotation createTestAnnotation(LabelType type, int classId, float x, float y, float w, float h)
	{
		Annotation ann(type);
		switch (type) {
		case LabelType::XYWH:
			ann.addObject(classId, { x, y, w, h });
			break;
		case LabelType::XYXY:
			// Convert XYWH to XYXY
			ann.addObject(classId, { x - w / 2, y - h / 2, x + w / 2, y + h / 2 });
			break;
		case LabelType::POLYGON:
			// Create a simple rectangle polygon
			ann.addObject(classId, { x - w / 2, y - h / 2, x + w / 2, y - h / 2,
									x + w / 2, y + h / 2, x - w / 2, y + h / 2 });
			break;
		case LabelType::XYWHR:
			ann.addObject(classId, { x, y, w, h, 0.0f }); // No rotation
			break;
		case LabelType::XYXYXYXY:
			// Create a simple rectangle with 4 corners
			ann.addObject(classId, { x - w / 2, y - h / 2, x + w / 2, y - h / 2,
									x + w / 2, y + h / 2, x - w / 2, y + h / 2 });
			break;
		default:
			break;
		}
		return ann;
	}
};

// ========== Basic Construction and Configuration ==========

TEST_F(MosaicAugmentationTest, Construction_Default)
{
	EXPECT_NO_THROW(Mosaic mosaic());
}

TEST_F(MosaicAugmentationTest, Construction_CustomSize)
{
	EXPECT_NO_THROW(Mosaic mosaic(1024, 768, 1.0f));
}

TEST_F(MosaicAugmentationTest, Construction_CustomBorder)
{
	EXPECT_NO_THROW(Mosaic mosaic(640, 640, 1.0f, 114, 114, 114));
}

TEST_F(MosaicAugmentationTest, GetProbability)
{
	Mosaic mosaic(640, 640, 0.75f);
	EXPECT_FLOAT_EQ(mosaic.getProbability(), 0.75f);
}

// ========== Basic Mosaic Application ==========

TEST_F(MosaicAugmentationTest, Apply_FourImages_SameSize)
{
	Mosaic mosaic(640, 640, 1.0f);

	std::vector<cv::Mat> images = {
		createColoredImage(480, 640, cv::Scalar(255, 0, 0)),
		createColoredImage(480, 640, cv::Scalar(0, 255, 0)),
		createColoredImage(480, 640, cv::Scalar(0, 0, 255)),
		createColoredImage(480, 640, cv::Scalar(255, 255, 0))
	};

	std::vector<Annotation> annots = {
		createTestAnnotation(LabelType::XYWH, 0, 320, 240, 100, 100),
		createTestAnnotation(LabelType::XYWH, 1, 320, 240, 100, 100),
		createTestAnnotation(LabelType::XYWH, 2, 320, 240, 100, 100),
		createTestAnnotation(LabelType::XYWH, 3, 320, 240, 100, 100)
	};

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);
	EXPECT_EQ(output.rows, 640);
	EXPECT_EQ(output.cols, 640);
	EXPECT_EQ(output.type(), CV_8UC3);
}

TEST_F(MosaicAugmentationTest, Apply_FourImages_DifferentSizes)
{
	Mosaic mosaic(640, 640, 1.0f);

	std::vector<cv::Mat> images = {
		createColoredImage(480, 640, cv::Scalar(255, 0, 0)),
		createColoredImage(720, 1280, cv::Scalar(0, 255, 0)),
		createColoredImage(360, 480, cv::Scalar(0, 0, 255)),
		createColoredImage(600, 800, cv::Scalar(255, 255, 0))
	};

	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	EXPECT_NO_THROW(mosaic.apply(images, annots, output, outputAnn));

	EXPECT_EQ(output.rows, 640);
	EXPECT_EQ(output.cols, 640);
}

TEST_F(MosaicAugmentationTest, Apply_UsingMNISTImages)
{
	if (testImages_.size() < 4) {
		std::cout << "SKIPPED: Not enough test images loaded" << std::endl;
		return;
	}

	Mosaic mosaic(640, 640, 1.0f);

	std::vector<Annotation> annots;
	for (int i = 0; i < 4; ++i) {
		annots.push_back(createTestAnnotation(LabelType::XYWH, i, 14, 14, 10, 10));
	}

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	bool applied = mosaic.apply(testImages_, annots, output, outputAnn);

	EXPECT_TRUE(applied);
	EXPECT_EQ(output.rows, 640);
	EXPECT_EQ(output.cols, 640);
}

// ========== Output Size Tests ==========

TEST_F(MosaicAugmentationTest, OutputSize_Square_640x640)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));
	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	mosaic.apply(images, annots, output, outputAnn);

	EXPECT_EQ(output.rows, 640);
	EXPECT_EQ(output.cols, 640);
}

TEST_F(MosaicAugmentationTest, OutputSize_Square_1024x1024)
{
	Mosaic mosaic(1024, 1024, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));
	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	mosaic.apply(images, annots, output, outputAnn);

	EXPECT_EQ(output.rows, 1024);
	EXPECT_EQ(output.cols, 1024);
}

TEST_F(MosaicAugmentationTest, OutputSize_Rectangle_1280x720)
{
	Mosaic mosaic(1280, 720, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));
	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	mosaic.apply(images, annots, output, outputAnn);

	EXPECT_EQ(output.rows, 720);
	EXPECT_EQ(output.cols, 1280);
}

// ========== Probability Tests ==========

TEST_F(MosaicAugmentationTest, Probability_AlwaysApply)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));
	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);
}

TEST_F(MosaicAugmentationTest, Probability_NeverApply)
{
	Mosaic mosaic(640, 640, 0.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));
	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_FALSE(applied);
}

TEST_F(MosaicAugmentationTest, Probability_HalfChance)
{
	Mosaic mosaic(640, 640, 0.5f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));
	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	int appliedCount = 0;
	int totalRuns = 100;

	for (int i = 0; i < totalRuns; ++i) {
		cv::Mat output;
		Annotation outputAnn(LabelType::XYWH);
		if (mosaic.apply(images, annots, output, outputAnn)) {
			appliedCount++;
		}
	}

	// Should be roughly 50% (allow 30-70% range for randomness)
	EXPECT_GT(appliedCount, 30);
	EXPECT_LT(appliedCount, 70);
}

// ========== Annotation Coordinate Transformation Tests ==========

TEST_F(MosaicAugmentationTest, AnnotationTransform_CoordinatesAdjusted)
{
	Mosaic mosaic(640, 640, 1.0f);
	mosaic.setSeed(42); // For reproducibility

	std::vector<cv::Mat> images(4, createColoredImage(320, 320, cv::Scalar(128, 128, 128)));

	// Add annotations at center of each image
	std::vector<Annotation> annots = {
		createTestAnnotation(LabelType::XYWH, 0, 160, 160, 50, 50),
		createTestAnnotation(LabelType::XYWH, 1, 160, 160, 50, 50),
		createTestAnnotation(LabelType::XYWH, 2, 160, 160, 50, 50),
		createTestAnnotation(LabelType::XYWH, 3, 160, 160, 50, 50)
	};

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);
	EXPECT_GT(outputAnn.size(), 0);

	// Annotations should have different coordinates than input
	// (scaled and translated to their quadrants)
	for (size_t i = 0; i < outputAnn.size(); ++i) {
		auto& points = outputAnn.getPoints()[i];
		// Coordinates should be within output image bounds
		EXPECT_GE(points[0], 0.0f);
		EXPECT_LT(points[0], 640.0f);
		EXPECT_GE(points[1], 0.0f);
		EXPECT_LT(points[1], 640.0f);
	}
}

TEST_F(MosaicAugmentationTest, AnnotationTransform_MergesAllAnnotations)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(320, 320, cv::Scalar(128, 128, 128)));

	std::vector<Annotation> annots = {
		createTestAnnotation(LabelType::XYWH, 0, 160, 160, 50, 50),
		createTestAnnotation(LabelType::XYWH, 1, 160, 160, 50, 50),
		createTestAnnotation(LabelType::XYWH, 2, 160, 160, 50, 50),
		createTestAnnotation(LabelType::XYWH, 3, 160, 160, 50, 50)
	};

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);
	// Should have annotations from all 4 images (some may be clipped)
	EXPECT_GT(outputAnn.size(), 0);
	EXPECT_LE(outputAnn.size(), 4);
}

TEST_F(MosaicAugmentationTest, AnnotationTransform_PreservesClassIDs)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(320, 320, cv::Scalar(128, 128, 128)));

	std::vector<Annotation> annots = {
		createTestAnnotation(LabelType::XYWH, 0, 160, 160, 50, 50),
		createTestAnnotation(LabelType::XYWH, 1, 160, 160, 50, 50),
		createTestAnnotation(LabelType::XYWH, 2, 160, 160, 50, 50),
		createTestAnnotation(LabelType::XYWH, 3, 160, 160, 50, 50)
	};

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);

	// Check that class IDs are preserved
	auto& classes = outputAnn.getClasses();
	for (int classId : classes) {
		EXPECT_GE(classId, 0);
		EXPECT_LE(classId, 3);
	}
}

TEST_F(MosaicAugmentationTest, AnnotationTransform_ClipsOutOfBounds)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(320, 320, cv::Scalar(128, 128, 128)));

	// Add annotations near edges (will be clipped)
	std::vector<Annotation> annots = {
		createTestAnnotation(LabelType::XYWH, 0, 10, 10, 50, 50),    // Near corner
		createTestAnnotation(LabelType::XYWH, 1, 310, 10, 50, 50),   // Near edge
		createTestAnnotation(LabelType::XYWH, 2, 10, 310, 50, 50),   // Near edge
		createTestAnnotation(LabelType::XYWH, 3, 310, 310, 50, 50)   // Near corner
	};

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);

	// Some annotations may be removed due to clipping
	// Just verify we don't crash and output is valid
	EXPECT_GE(outputAnn.size(), 0);
	EXPECT_LE(outputAnn.size(), 4);
}

// ========== Center Point Randomization Tests ==========

TEST_F(MosaicAugmentationTest, CenterPoint_Randomization)
{
	Mosaic mosaic1(640, 640, 1.0f);
	Mosaic mosaic2(640, 640, 1.0f);

	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));
	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	cv::Mat output1, output2;
	Annotation outputAnn1(LabelType::XYWH), outputAnn2(LabelType::XYWH);

	mosaic1.apply(images, annots, output1, outputAnn1);
	mosaic2.apply(images, annots, output2, outputAnn2);

	// Different runs should produce different results (different center points)
	// Compare images - they should differ
	double diff = cv::norm(output1, output2, cv::NORM_L2);
	EXPECT_GT(diff, 0.0);
}

TEST_F(MosaicAugmentationTest, CenterPoint_WithSeed_Reproducible)
{
	Mosaic mosaic1(640, 640, 1.0f);
	mosaic1.setSeed(42);

	Mosaic mosaic2(640, 640, 1.0f);
	mosaic2.setSeed(42);

	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));
	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	cv::Mat output1, output2;
	Annotation outputAnn1(LabelType::XYWH), outputAnn2(LabelType::XYWH);

	mosaic1.apply(images, annots, output1, outputAnn1);
	mosaic2.apply(images, annots, output2, outputAnn2);

	// Same seed should produce identical results
	double diff = cv::norm(output1, output2, cv::NORM_L2);
	EXPECT_EQ(diff, 0.0);
}

TEST_F(MosaicAugmentationTest, CenterPoint_InValidRange)
{
	// Center should be in 0.25-0.75 range
	Mosaic mosaic(640, 640, 1.0f);
	mosaic.setSeed(12345);

	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));
	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	// Run multiple times to check center point range
	for (int i = 0; i < 10; ++i) {
		cv::Mat output;
		Annotation outputAnn(LabelType::XYWH);
		mosaic.apply(images, annots, output, outputAnn);

		// Can't directly verify center point, but verify output is valid
		EXPECT_EQ(output.rows, 640);
		EXPECT_EQ(output.cols, 640);
	}
}

// ========== Dataset Type Tests ==========

TEST_F(MosaicAugmentationTest, DetectionDataset_XYWH)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));

	std::vector<Annotation> annots = {
		createTestAnnotation(LabelType::XYWH, 0, 320, 240, 100, 100),
		createTestAnnotation(LabelType::XYWH, 1, 320, 240, 120, 120),
		createTestAnnotation(LabelType::XYWH, 2, 320, 240, 80, 80),
		createTestAnnotation(LabelType::XYWH, 3, 320, 240, 150, 150)
	};

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);
	EXPECT_EQ(outputAnn.getLabelType(), LabelType::XYWH);
}

TEST_F(MosaicAugmentationTest, DetectionDataset_XYXY)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));

	std::vector<Annotation> annots = {
		createTestAnnotation(LabelType::XYXY, 0, 320, 240, 100, 100),
		createTestAnnotation(LabelType::XYXY, 1, 320, 240, 120, 120),
		createTestAnnotation(LabelType::XYXY, 2, 320, 240, 80, 80),
		createTestAnnotation(LabelType::XYXY, 3, 320, 240, 150, 150)
	};

	cv::Mat output;
	Annotation outputAnn(LabelType::XYXY);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);
	EXPECT_EQ(outputAnn.getLabelType(), LabelType::XYXY);
}

TEST_F(MosaicAugmentationTest, OBBDataset_XYWHR)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));

	std::vector<Annotation> annots = {
		createTestAnnotation(LabelType::XYWHR, 0, 320, 240, 100, 100),
		createTestAnnotation(LabelType::XYWHR, 1, 320, 240, 120, 120),
		createTestAnnotation(LabelType::XYWHR, 2, 320, 240, 80, 80),
		createTestAnnotation(LabelType::XYWHR, 3, 320, 240, 150, 150)
	};

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWHR);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);
	EXPECT_EQ(outputAnn.getLabelType(), LabelType::XYWHR);
}

TEST_F(MosaicAugmentationTest, OBBDataset_XYXYXYXY)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));

	std::vector<Annotation> annots = {
		createTestAnnotation(LabelType::XYXYXYXY, 0, 320, 240, 100, 100),
		createTestAnnotation(LabelType::XYXYXYXY, 1, 320, 240, 120, 120),
		createTestAnnotation(LabelType::XYXYXYXY, 2, 320, 240, 80, 80),
		createTestAnnotation(LabelType::XYXYXYXY, 3, 320, 240, 150, 150)
	};

	cv::Mat output;
	Annotation outputAnn(LabelType::XYXYXYXY);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);
	EXPECT_EQ(outputAnn.getLabelType(), LabelType::XYXYXYXY);
}

TEST_F(MosaicAugmentationTest, SegmentationDataset_POLYGON)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));

	std::vector<Annotation> annots = {
		createTestAnnotation(LabelType::POLYGON, 0, 320, 240, 100, 100),
		createTestAnnotation(LabelType::POLYGON, 1, 320, 240, 120, 120),
		createTestAnnotation(LabelType::POLYGON, 2, 320, 240, 80, 80),
		createTestAnnotation(LabelType::POLYGON, 3, 320, 240, 150, 150)
	};

	cv::Mat output;
	Annotation outputAnn(LabelType::POLYGON);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);
	EXPECT_EQ(outputAnn.getLabelType(), LabelType::POLYGON);
}

TEST_F(MosaicAugmentationTest, SegmentationDataset_ComplexPolygons)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));

	// Create complex polygons (5-sided)
	std::vector<Annotation> annots;
	for (int i = 0; i < 4; ++i) {
		Annotation ann(LabelType::POLYGON);
		ann.addObject(i, { 200.0f, 150.0f, 250.0f, 180.0f, 230.0f, 250.0f,
						 170.0f, 250.0f, 150.0f, 180.0f });
		annots.push_back(ann);
	}

	cv::Mat output;
	Annotation outputAnn(LabelType::POLYGON);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);
	EXPECT_EQ(outputAnn.getLabelType(), LabelType::POLYGON);
}

// ========== Multiple Annotations per Image ==========

TEST_F(MosaicAugmentationTest, MultipleAnnotationsPerImage)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));

	std::vector<Annotation> annots;
	for (int i = 0; i < 4; ++i) {
		Annotation ann(LabelType::XYWH);
		// Add 3 objects per image
		ann.addObject(i, { 100.0f, 100.0f, 50.0f, 50.0f });
		ann.addObject(i, { 300.0f, 100.0f, 50.0f, 50.0f });
		ann.addObject(i, { 200.0f, 300.0f, 50.0f, 50.0f });
		annots.push_back(ann);
	}

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);
	// Should have up to 12 objects (3 per image * 4 images), some may be clipped
	EXPECT_GT(outputAnn.size(), 0);
	EXPECT_LE(outputAnn.size(), 12);
}

// ========== Edge Cases ==========

TEST_F(MosaicAugmentationTest, EmptyAnnotations)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));

	std::vector<Annotation> annots(4, Annotation(LabelType::XYWH)); // Empty annotations

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	bool applied = mosaic.apply(images, annots, output, outputAnn);

	EXPECT_TRUE(applied);
	EXPECT_EQ(outputAnn.size(), 0);
}

TEST_F(MosaicAugmentationTest, VerySmallImages)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(10, 10, cv::Scalar(128, 128, 128)));

	std::vector<Annotation> annots = {
		createTestAnnotation(LabelType::XYWH, 0, 5, 5, 3, 3),
		createTestAnnotation(LabelType::XYWH, 1, 5, 5, 3, 3),
		createTestAnnotation(LabelType::XYWH, 2, 5, 5, 3, 3),
		createTestAnnotation(LabelType::XYWH, 3, 5, 5, 3, 3)
	};

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	EXPECT_NO_THROW(mosaic.apply(images, annots, output, outputAnn));
}

TEST_F(MosaicAugmentationTest, VeryLargeImages)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(2000, 3000, cv::Scalar(128, 128, 128)));

	std::vector<Annotation> annots = {
		createTestAnnotation(LabelType::XYWH, 0, 1500, 1000, 500, 500),
		createTestAnnotation(LabelType::XYWH, 1, 1500, 1000, 500, 500),
		createTestAnnotation(LabelType::XYWH, 2, 1500, 1000, 500, 500),
		createTestAnnotation(LabelType::XYWH, 3, 1500, 1000, 500, 500)
	};

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	EXPECT_NO_THROW(mosaic.apply(images, annots, output, outputAnn));

	// Output should still be 640x640
	EXPECT_EQ(output.rows, 640);
	EXPECT_EQ(output.cols, 640);
}

TEST_F(MosaicAugmentationTest, ErrorHandling_WrongNumberOfImages)
{
	Mosaic mosaic(640, 640, 1.0f);

	// Only 3 images
	std::vector<cv::Mat> images(3, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));
	std::vector<Annotation> annots(3, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	EXPECT_THROW(mosaic.apply(images, annots, output, outputAnn), std::invalid_argument);
}

TEST_F(MosaicAugmentationTest, ErrorHandling_MismatchedVectorSizes)
{
	Mosaic mosaic(640, 640, 1.0f);

	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));
	std::vector<Annotation> annots(3, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	EXPECT_THROW(mosaic.apply(images, annots, output, outputAnn), std::invalid_argument);
}

// ========== Border Color Tests ==========

TEST_F(MosaicAugmentationTest, BorderColor_Default)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(100, 100, cv::Scalar(255, 255, 255)));
	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 50, 50, 20, 20));

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	mosaic.apply(images, annots, output, outputAnn);

	// Border areas should exist (images are small, so large border)
	EXPECT_EQ(output.rows, 640);
	EXPECT_EQ(output.cols, 640);
}

TEST_F(MosaicAugmentationTest, BorderColor_Custom)
{
	Mosaic mosaic(640, 640, 1.0f, 114, 114, 114);
	std::vector<cv::Mat> images(4, createColoredImage(100, 100, cv::Scalar(255, 255, 255)));
	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 50, 50, 20, 20));

	cv::Mat output;
	Annotation outputAnn(LabelType::XYWH);

	mosaic.apply(images, annots, output, outputAnn);

	EXPECT_EQ(output.rows, 640);
	EXPECT_EQ(output.cols, 640);
}

// ========== Stress Test ==========

TEST_F(MosaicAugmentationTest, StressTest_MultipleIterations)
{
	Mosaic mosaic(640, 640, 1.0f);
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));
	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	for (int i = 0; i < 50; ++i) {
		cv::Mat output;
		Annotation outputAnn(LabelType::XYWH);
		EXPECT_NO_THROW(mosaic.apply(images, annots, output, outputAnn));
	}
}

TEST_F(MosaicAugmentationTest, StressTest_DifferentConfigurations)
{
	std::vector<cv::Mat> images(4, createColoredImage(480, 640, cv::Scalar(128, 128, 128)));
	std::vector<Annotation> annots(4, createTestAnnotation(LabelType::XYWH, 0, 100, 100, 50, 50));

	// Test various configurations
	std::vector<std::pair<int, int>> sizes = { {640, 640}, {1024, 1024}, {1280, 720} };

	for (auto [width, height] : sizes) {
		Mosaic mosaic(width, height, 1.0f);
		cv::Mat output;
		Annotation outputAnn(LabelType::XYWH);

		bool applied = mosaic.apply(images, annots, output, outputAnn);
		EXPECT_TRUE(applied);
		EXPECT_EQ(output.rows, height);
		EXPECT_EQ(output.cols, width);
	}
}
