#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Common/Annotation.h"

using namespace WheelDL::Data;

/**
 * @class AnnotationTest
 * @brief Test suite for Annotation class
 */
class AnnotationTest : public ::testing::Test
{
protected:
    const float EPSILON = 1e-5f;

    void expectVectorNear(const std::vector<float>& actual, const std::vector<float>& expected)
    {
        ASSERT_EQ(actual.size(), expected.size()) << "Vector sizes differ";
        for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_NEAR(actual[i], expected[i], EPSILON) << "Mismatch at index " << i;
        }
    }
};

// ========== Object Management Tests ==========

TEST_F(AnnotationTest, Construction)
{
    Annotation ann(LabelType::XYWH);
    EXPECT_EQ(ann.getLabelType(), LabelType::XYWH);
    EXPECT_TRUE(ann.empty());
    EXPECT_EQ(ann.size(), 0);
}

TEST_F(AnnotationTest, AddObject)
{
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 100.0f, 50.0f, 50.0f});
    ann.addObject(1, {200.0f, 200.0f, 60.0f, 60.0f});

    EXPECT_EQ(ann.size(), 2);
    EXPECT_FALSE(ann.empty());
    EXPECT_EQ(ann.getClasses().size(), 2);
    EXPECT_EQ(ann.getPoints().size(), 2);
}

TEST_F(AnnotationTest, Clear)
{
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 100.0f, 50.0f, 50.0f});
    ann.clear();

    EXPECT_TRUE(ann.empty());
    EXPECT_EQ(ann.size(), 0);
}

// ========== Normalization Tests ==========

TEST_F(AnnotationTest, Normalize)
{
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 200.0f, 50.0f, 100.0f});

    ann.normalize(1000, 1000);

    auto& points = ann.getPoints();
    expectVectorNear(points[0], {0.1f, 0.2f, 0.05f, 0.1f});
}

TEST_F(AnnotationTest, Denormalize)
{
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 200.0f, 50.0f, 100.0f});

    // First normalize, then denormalize back
    ann.normalize(1000, 1000);  // {0.1f, 0.2f, 0.05f, 0.1f}
    ann.denormalize(1000, 1000);  // Back to pixel coordinates

    auto& points = ann.getPoints();
    expectVectorNear(points[0], {100.0f, 200.0f, 50.0f, 100.0f});
}

TEST_F(AnnotationTest, NormalizeDenormalize_RoundTrip)
{
    Annotation ann(LabelType::XYWH);
    std::vector<float> original = {100.0f, 200.0f, 50.0f, 100.0f};
    ann.addObject(0, original);

    ann.normalize(1000, 1000);
    ann.denormalize(1000, 1000);

    expectVectorNear(ann.getPoints()[0], original);
}

// ========== Transformation Tests ==========

TEST_F(AnnotationTest, Scale)
{
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 200.0f, 50.0f, 100.0f});

    ann.scale(2.0f, 0.5f);

    expectVectorNear(ann.getPoints()[0], {200.0f, 100.0f, 100.0f, 50.0f});
}

TEST_F(AnnotationTest, Translate)
{
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 200.0f, 50.0f, 100.0f});

    ann.translate(10.0f, -20.0f);

    // Translate only affects center (x,y), not width/height
    expectVectorNear(ann.getPoints()[0], {110.0f, 180.0f, 50.0f, 100.0f});
}

TEST_F(AnnotationTest, FlipHorizontal)
{
    Annotation ann(LabelType::XYXY);
    ann.addObject(0, {100.0f, 100.0f, 200.0f, 200.0f});

    ann.flipHorizontal(1000);

    auto& points = ann.getPoints()[0];
    // x coordinates should be flipped: 1000 - x
    EXPECT_NEAR(points[0], 900.0f, EPSILON);  // 1000 - 100
    EXPECT_NEAR(points[2], 800.0f, EPSILON);  // 1000 - 200
    // y coordinates unchanged
    EXPECT_NEAR(points[1], 100.0f, EPSILON);
    EXPECT_NEAR(points[3], 200.0f, EPSILON);
}

// ========== Type Conversion Tests ==========

TEST_F(AnnotationTest, ConvertTo_XYWHToXYXY)
{
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 100.0f, 50.0f, 50.0f});  // center=(100,100), size=50x50

    ann.convertTo(LabelType::XYXY);

    EXPECT_EQ(ann.getLabelType(), LabelType::XYXY);
    expectVectorNear(ann.getPoints()[0], {75.0f, 75.0f, 125.0f, 125.0f});
}

TEST_F(AnnotationTest, ConvertTo_XYXYToXYWH)
{
    Annotation ann(LabelType::XYXY);
    ann.addObject(0, {75.0f, 75.0f, 125.0f, 125.0f});

    ann.convertTo(LabelType::XYWH);

    EXPECT_EQ(ann.getLabelType(), LabelType::XYWH);
    expectVectorNear(ann.getPoints()[0], {100.0f, 100.0f, 50.0f, 50.0f});
}

TEST_F(AnnotationTest, GetAs_CopyConversion)
{
    Annotation original(LabelType::XYWH);
    original.addObject(0, {100.0f, 100.0f, 50.0f, 50.0f});

    Annotation converted = original.getAs(LabelType::XYXY);

    // Original unchanged
    EXPECT_EQ(original.getLabelType(), LabelType::XYWH);
    expectVectorNear(original.getPoints()[0], {100.0f, 100.0f, 50.0f, 50.0f});

    // Converted is different
    EXPECT_EQ(converted.getLabelType(), LabelType::XYXY);
    expectVectorNear(converted.getPoints()[0], {75.0f, 75.0f, 125.0f, 125.0f});
}

TEST_F(AnnotationTest, ConvertTo_XYXYToPolygon)
{
    Annotation ann(LabelType::XYXY);
    ann.addObject(0, {10.0f, 20.0f, 100.0f, 200.0f});

    ann.convertTo(LabelType::POLYGON);

    EXPECT_EQ(ann.getLabelType(), LabelType::POLYGON);
    // Should have 4 corners (8 values)
    auto& points = ann.getPoints()[0];
    EXPECT_EQ(points.size(), 8);
    expectVectorNear(points, {10.0f, 20.0f, 100.0f, 20.0f, 100.0f, 200.0f, 10.0f, 200.0f});
}

TEST_F(AnnotationTest, ConvertTo_PolygonToXYXY)
{
    Annotation ann(LabelType::POLYGON);
    ann.addObject(0, {10.0f, 20.0f, 100.0f, 30.0f, 90.0f, 200.0f, 5.0f, 180.0f});

    ann.convertTo(LabelType::XYXY);

    EXPECT_EQ(ann.getLabelType(), LabelType::XYXY);
    // Should compute bounding box: min=(5, 20), max=(100, 200)
    expectVectorNear(ann.getPoints()[0], {5.0f, 20.0f, 100.0f, 200.0f});
}

TEST_F(AnnotationTest, ConvertTo_XYWHToPolygon)
{
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {50.0f, 50.0f, 40.0f, 60.0f});  // center=(50,50), size=40x60

    ann.convertTo(LabelType::POLYGON);

    // Should be 4 corners
    auto& points = ann.getPoints()[0];
    EXPECT_EQ(points.size(), 8);
    // corners: (30,20), (70,20), (70,80), (30,80)
    expectVectorNear(points, {30.0f, 20.0f, 70.0f, 20.0f, 70.0f, 80.0f, 30.0f, 80.0f});
}

TEST_F(AnnotationTest, ConvertTo_PolygonToXYWH)
{
    Annotation ann(LabelType::POLYGON);
    ann.addObject(0, {30.0f, 20.0f, 70.0f, 20.0f, 70.0f, 80.0f, 30.0f, 80.0f});

    ann.convertTo(LabelType::XYWH);

    // Bounding box: (30,20) to (70,80) -> center=(50,50), size=40x60
    expectVectorNear(ann.getPoints()[0], {50.0f, 50.0f, 40.0f, 60.0f});
}

TEST_F(AnnotationTest, ConvertTo_XYXYXYXYToXYWHR)
{
    Annotation ann(LabelType::XYXYXYXY);
    // Axis-aligned rectangle corners
    ann.addObject(0, {10.0f, 20.0f, 50.0f, 20.0f, 50.0f, 60.0f, 10.0f, 60.0f});

    ann.convertTo(LabelType::XYWHR);

    auto& points = ann.getPoints()[0];
    EXPECT_EQ(points.size(), 5);
    // Center should be (30, 40)
    EXPECT_NEAR(points[0], 30.0f, EPSILON);
    EXPECT_NEAR(points[1], 40.0f, EPSILON);
    // Width = 40, Height = 40
    EXPECT_NEAR(points[2], 40.0f, EPSILON);
    EXPECT_NEAR(points[3], 40.0f, EPSILON);
    // Angle should be ~0 (axis-aligned)
    EXPECT_NEAR(points[4], 0.0f, 0.1f);
}

TEST_F(AnnotationTest, ConvertTo_XYWHRToXYXYXYXY)
{
    Annotation ann(LabelType::XYWHR);
    // Center=(50,50), size=40x60, rotation=0
    ann.addObject(0, {50.0f, 50.0f, 40.0f, 60.0f, 0.0f});

    ann.convertTo(LabelType::XYXYXYXY);

    // Should produce 4 corners
    auto& points = ann.getPoints()[0];
    EXPECT_EQ(points.size(), 8);
    // At rotation=0, corners should be at (±w/2, ±h/2) relative to center
    // Expected: (30,20), (70,20), (70,80), (30,80)
    EXPECT_NEAR(points[0], 30.0f, EPSILON);
    EXPECT_NEAR(points[1], 20.0f, EPSILON);
}

TEST_F(AnnotationTest, ConvertTo_PolygonToXYWHR_MinimumRotatedBox)
{
    Annotation ann(LabelType::POLYGON);
    // Axis-aligned rectangle
    ann.addObject(0, {10.0f, 10.0f, 50.0f, 10.0f, 50.0f, 30.0f, 10.0f, 30.0f});

    ann.convertTo(LabelType::XYWHR);

    auto& points = ann.getPoints()[0];
    EXPECT_EQ(points.size(), 5);

    // For axis-aligned rectangle, center=(30,20), w=40, h=20, angle??
    EXPECT_NEAR(points[0], 30.0f, 1.0f);  // cx
    EXPECT_NEAR(points[1], 20.0f, 1.0f);  // cy
    EXPECT_GT(points[2], 0.0f);           // w > 0
    EXPECT_GT(points[3], 0.0f);           // h > 0
}

TEST_F(AnnotationTest, ConvertTo_PolygonToXYXYXYXY_MinimumRotatedBox)
{
    Annotation ann(LabelType::POLYGON);
    ann.addObject(0, {10.0f, 10.0f, 50.0f, 10.0f, 50.0f, 30.0f, 10.0f, 30.0f});

    ann.convertTo(LabelType::XYXYXYXY);

    // Should produce 4 corners of minimum rotated box
    auto& points = ann.getPoints()[0];
    EXPECT_EQ(points.size(), 8);
}

// ========== Edge Cases ==========

TEST_F(AnnotationTest, EmptyAnnotation_Operations)
{
    Annotation ann(LabelType::XYWH);

    // Should not crash on empty
    EXPECT_NO_THROW(ann.normalize(1000, 1000));
    EXPECT_NO_THROW(ann.scale(2.0f, 2.0f));
    EXPECT_NO_THROW(ann.convertTo(LabelType::XYXY));
}

TEST_F(AnnotationTest, MultipleObjects_Conversion)
{
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 100.0f, 50.0f, 50.0f});
    ann.addObject(1, {200.0f, 200.0f, 60.0f, 60.0f});

    ann.convertTo(LabelType::XYXY);

    EXPECT_EQ(ann.size(), 2);
    expectVectorNear(ann.getPoints()[0], {75.0f, 75.0f, 125.0f, 125.0f});
    expectVectorNear(ann.getPoints()[1], {170.0f, 170.0f, 230.0f, 230.0f});
}

TEST_F(AnnotationTest, InvalidConversion_ThrowsException)
{
    Annotation ann(LabelType::NONE);
    ann.addObject(0, {1.0f, 2.0f});

    // NONE type doesn't support conversions
    EXPECT_THROW(ann.convertTo(LabelType::XYWH), std::invalid_argument);
}
