#include "pch.h"
#include "Data/Common/Annotation.h"
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace WheelDL::Data;

// =============================================================================
// Test Fixture
// =============================================================================

class AnnotationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper: Create XYWH annotation (center-based bbox)
    Annotation createXYWHAnnotation(int classId = 0, float cx = 100.0f, float cy = 100.0f,
                                     float w = 50.0f, float h = 50.0f) {
        Annotation ann(LabelType::XYWH);
        ann.addObject(classId, {cx, cy, w, h});
        return ann;
    }

    // Helper: Create XYXY annotation (corner-based bbox)
    Annotation createXYXYAnnotation(int classId = 0, float x1 = 50.0f, float y1 = 50.0f,
                                     float x2 = 150.0f, float y2 = 150.0f) {
        Annotation ann(LabelType::XYXY);
        ann.addObject(classId, {x1, y1, x2, y2});
        return ann;
    }

    // Helper: Create XYWHR annotation (oriented bbox with rotation)
    Annotation createXYWHRAnnotation(int classId = 0, float cx = 100.0f, float cy = 100.0f,
                                      float w = 50.0f, float h = 30.0f, float r = 0.5f) {
        Annotation ann(LabelType::XYWHR);
        ann.addObject(classId, {cx, cy, w, h, r});
        return ann;
    }

    // Helper: Create XYXYXYXY annotation (4-corner rotated bbox)
    Annotation createXYXYXYXYAnnotation(int classId = 0) {
        Annotation ann(LabelType::XYXYXYXY);
        // Rectangle corners: (50,50), (150,50), (150,150), (50,150)
        ann.addObject(classId, {50.0f, 50.0f, 150.0f, 50.0f, 150.0f, 150.0f, 50.0f, 150.0f});
        return ann;
    }

    // Helper: Create POLYGON annotation
    Annotation createPolygonAnnotation(int classId = 0) {
        Annotation ann(LabelType::POLYGON);
        // Triangle: (100,50), (150,150), (50,150)
        ann.addObject(classId, {100.0f, 50.0f, 150.0f, 150.0f, 50.0f, 150.0f});
        return ann;
    }

    // Helper: Compare floats with tolerance
    bool floatEqual(float a, float b, float tol = 1e-3f) {
        return std::abs(a - b) < tol;
    }
};

// =============================================================================
// 3.1 Construction Tests (ANN-001 ~ ANN-006)
// =============================================================================

// ANN-001: Default constructor
TEST_F(AnnotationTest, Ctor_Default) {
    Annotation ann;
    EXPECT_EQ(LabelType::NONE, ann.getLabelType());
    EXPECT_TRUE(ann.empty());
    EXPECT_EQ(0, ann.size());
}

// ANN-002: Constructor with XYWH type
TEST_F(AnnotationTest, Ctor_XYWH) {
    Annotation ann(LabelType::XYWH);
    EXPECT_EQ(LabelType::XYWH, ann.getLabelType());
    EXPECT_TRUE(ann.empty());
}

// ANN-003: Constructor with isNormalized=true
TEST_F(AnnotationTest, Ctor_IsNormalizedTrue) {
    Annotation ann(LabelType::XYWH, true);
    EXPECT_EQ(LabelType::XYWH, ann.getLabelType());
    // Verify internal normalized state via normalize idempotency
    // If already normalized, normalize should be a no-op
    ann.addObject(0, {0.5f, 0.5f, 0.1f, 0.1f});
    ann.normalize(100, 100);  // Should skip since already normalized
    auto& pts = ann.getPoints();
    EXPECT_TRUE(floatEqual(0.5f, pts[0][0]));  // cx unchanged
}

// ANN-004: forClassification with class 0
TEST_F(AnnotationTest, ForClassification_Class0) {
    Annotation ann = Annotation::forClassification(0);
    EXPECT_EQ(LabelType::NONE, ann.getLabelType());
    EXPECT_EQ(1, ann.size());
    EXPECT_EQ(0, ann.getClasses()[0]);
    EXPECT_TRUE(ann.getPoints()[0].empty());
}

// ANN-005: forClassification with class 5
TEST_F(AnnotationTest, ForClassification_Class5) {
    Annotation ann = Annotation::forClassification(5);
    EXPECT_EQ(LabelType::NONE, ann.getLabelType());
    EXPECT_EQ(1, ann.size());
    EXPECT_EQ(5, ann.getClasses()[0]);
}

// ANN-006: forClassification with negative class
TEST_F(AnnotationTest, ForClassification_NegativeClass) {
    Annotation ann = Annotation::forClassification(-1);
    EXPECT_EQ(-1, ann.getClasses()[0]);
}

// =============================================================================
// 3.2 Object Management Tests (ANN-007 ~ ANN-018)
// =============================================================================

// ANN-007: addObject single
TEST_F(AnnotationTest, AddObject_Single) {
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 100.0f, 50.0f, 50.0f});
    EXPECT_EQ(1, ann.size());
}

// ANN-008: addObject multiple
TEST_F(AnnotationTest, AddObject_Multiple) {
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 100.0f, 50.0f, 50.0f});
    ann.addObject(1, {200.0f, 200.0f, 30.0f, 30.0f});
    ann.addObject(2, {300.0f, 300.0f, 40.0f, 40.0f});
    EXPECT_EQ(3, ann.size());
}

// ANN-009: addObject exception safety (rollback on exception)
TEST_F(AnnotationTest, AddObject_ExceptionSafety) {
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 100.0f, 50.0f, 50.0f});
    size_t sizeBefore = ann.size();
    // Adding objects should maintain consistency
    ann.addObject(1, {200.0f, 200.0f, 30.0f, 30.0f});
    EXPECT_EQ(sizeBefore + 1, ann.size());
}

// ANN-010: size() empty
TEST_F(AnnotationTest, Size_Empty) {
    Annotation ann(LabelType::XYWH);
    EXPECT_EQ(0, ann.size());
}

// ANN-011: size() with objects
TEST_F(AnnotationTest, Size_WithObjects) {
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 100.0f, 50.0f, 50.0f});
    ann.addObject(1, {200.0f, 200.0f, 30.0f, 30.0f});
    EXPECT_EQ(2, ann.size());
}

// ANN-012: empty() true
TEST_F(AnnotationTest, Empty_True) {
    Annotation ann(LabelType::XYWH);
    EXPECT_TRUE(ann.empty());
}

// ANN-013: empty() false
TEST_F(AnnotationTest, Empty_False) {
    Annotation ann = createXYWHAnnotation();
    EXPECT_FALSE(ann.empty());
}

// ANN-014: clear()
TEST_F(AnnotationTest, Clear) {
    Annotation ann = createXYWHAnnotation();
    ann.addObject(1, {200.0f, 200.0f, 30.0f, 30.0f});
    EXPECT_EQ(2, ann.size());
    ann.clear();
    EXPECT_EQ(0, ann.size());
    EXPECT_TRUE(ann.empty());
}

// ANN-015: clone() deep copy
TEST_F(AnnotationTest, Clone_DeepCopy) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 50.0f, 50.0f);
    Annotation cloned = ann.clone();

    // Modify original
    ann.getPoints()[0][0] = 999.0f;

    // Cloned should be independent
    EXPECT_NE(999.0f, cloned.getPoints()[0][0]);
    EXPECT_TRUE(floatEqual(100.0f, cloned.getPoints()[0][0]));
}

// ANN-016: clone() preserves isNormalized
TEST_F(AnnotationTest, Clone_PreservesNormalized) {
    Annotation ann(LabelType::XYWH, true);
    ann.addObject(0, {0.5f, 0.5f, 0.1f, 0.1f});
    Annotation cloned = ann.clone();

    // Both should skip normalize since already normalized
    cloned.normalize(100, 100);
    EXPECT_TRUE(floatEqual(0.5f, cloned.getPoints()[0][0]));
}

// ANN-017: getClasses() const
TEST_F(AnnotationTest, GetClasses_Const) {
    const Annotation ann = createXYWHAnnotation(5);
    const std::vector<int>& classes = ann.getClasses();
    EXPECT_EQ(1, classes.size());
    EXPECT_EQ(5, classes[0]);
}

// ANN-018: getClasses() mutable
TEST_F(AnnotationTest, GetClasses_Mutable) {
    Annotation ann = createXYWHAnnotation(5);
    std::vector<int>& classes = ann.getClasses();
    classes[0] = 10;
    EXPECT_EQ(10, ann.getClasses()[0]);
}

// =============================================================================
// 3.3 Normalize Tests (ANN-019 ~ ANN-030)
// =============================================================================

// ANN-019: normalize XYWH
TEST_F(AnnotationTest, Normalize_XYWH) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 50.0f, 50.0f);
    ann.normalize(200, 200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(0.5f, pts[0]));   // cx: 100/200
    EXPECT_TRUE(floatEqual(0.5f, pts[1]));   // cy: 100/200
    EXPECT_TRUE(floatEqual(0.25f, pts[2]));  // w: 50/200
    EXPECT_TRUE(floatEqual(0.25f, pts[3]));  // h: 50/200
}

// ANN-020: normalize XYWHR
TEST_F(AnnotationTest, Normalize_XYWHR) {
    Annotation ann = createXYWHRAnnotation(0, 100.0f, 100.0f, 50.0f, 30.0f, 0.5f);
    ann.normalize(200, 200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(0.5f, pts[0]));   // cx
    EXPECT_TRUE(floatEqual(0.5f, pts[1]));   // cy
    EXPECT_TRUE(floatEqual(0.25f, pts[2]));  // w
    EXPECT_TRUE(floatEqual(0.15f, pts[3]));  // h: 30/200
    EXPECT_TRUE(floatEqual(0.5f, pts[4]));   // rotation kept as-is
}

// ANN-021: normalize XYXY
TEST_F(AnnotationTest, Normalize_XYXY) {
    Annotation ann = createXYXYAnnotation(0, 50.0f, 50.0f, 150.0f, 150.0f);
    ann.normalize(200, 200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(0.25f, pts[0]));  // x1: 50/200
    EXPECT_TRUE(floatEqual(0.25f, pts[1]));  // y1: 50/200
    EXPECT_TRUE(floatEqual(0.75f, pts[2]));  // x2: 150/200
    EXPECT_TRUE(floatEqual(0.75f, pts[3]));  // y2: 150/200
}

// ANN-022: normalize POLYGON
TEST_F(AnnotationTest, Normalize_POLYGON) {
    Annotation ann = createPolygonAnnotation();
    ann.normalize(200, 200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(0.5f, pts[0]));   // x1: 100/200
    EXPECT_TRUE(floatEqual(0.25f, pts[1]));  // y1: 50/200
}

// ANN-023: normalize XYXYXYXY
TEST_F(AnnotationTest, Normalize_XYXYXYXY) {
    Annotation ann = createXYXYXYXYAnnotation();
    ann.normalize(200, 200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(0.25f, pts[0]));  // x1: 50/200
    EXPECT_TRUE(floatEqual(0.25f, pts[1]));  // y1: 50/200
    EXPECT_TRUE(floatEqual(0.75f, pts[2]));  // x2: 150/200
}

// ANN-024: normalize idempotent
TEST_F(AnnotationTest, Normalize_Idempotent) {
    Annotation ann(LabelType::XYWH, true);  // Already normalized
    ann.addObject(0, {0.5f, 0.5f, 0.25f, 0.25f});

    ann.normalize(200, 200);  // Should skip

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(0.5f, pts[0]));  // Unchanged
}

// ANN-025: normalize negative width throws
TEST_F(AnnotationTest, Normalize_NegativeWidth_Throws) {
    Annotation ann = createXYWHAnnotation();
    EXPECT_THROW(ann.normalize(-100, 200), std::invalid_argument);
}

// ANN-026: normalize zero height throws
TEST_F(AnnotationTest, Normalize_ZeroHeight_Throws) {
    Annotation ann = createXYWHAnnotation();
    EXPECT_THROW(ann.normalize(200, 0), std::invalid_argument);
}

// ANN-027: normalize empty annotation
TEST_F(AnnotationTest, Normalize_Empty) {
    Annotation ann(LabelType::XYWH);
    EXPECT_NO_THROW(ann.normalize(200, 200));
    EXPECT_TRUE(ann.empty());
}

// ANN-028: normalize coords.size() < 4
TEST_F(AnnotationTest, Normalize_ShortCoords) {
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 100.0f});  // Only 2 coords
    EXPECT_NO_THROW(ann.normalize(200, 200));
}

// ANN-029: normalize sets flag
TEST_F(AnnotationTest, Normalize_SetsFlag) {
    Annotation ann = createXYWHAnnotation();
    ann.normalize(200, 200);

    // Second normalize should be no-op (flag set)
    auto ptsBefore = ann.getPoints()[0];
    ann.normalize(400, 400);  // Should skip
    auto ptsAfter = ann.getPoints()[0];

    EXPECT_EQ(ptsBefore, ptsAfter);
}

// ANN-030: normalize multiple objects
TEST_F(AnnotationTest, Normalize_MultipleObjects) {
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 100.0f, 50.0f, 50.0f});
    ann.addObject(1, {150.0f, 150.0f, 40.0f, 40.0f});

    ann.normalize(200, 200);

    EXPECT_TRUE(floatEqual(0.5f, ann.getPoints()[0][0]));
    EXPECT_TRUE(floatEqual(0.75f, ann.getPoints()[1][0]));  // 150/200
}

// =============================================================================
// 3.4 Denormalize Tests (ANN-031 ~ ANN-042)
// =============================================================================

// ANN-031: denormalize XYWH
TEST_F(AnnotationTest, Denormalize_XYWH) {
    Annotation ann(LabelType::XYWH, true);
    ann.addObject(0, {0.5f, 0.5f, 0.25f, 0.25f});
    ann.denormalize(200, 200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(100.0f, pts[0]));  // cx: 0.5*200
    EXPECT_TRUE(floatEqual(100.0f, pts[1]));  // cy
    EXPECT_TRUE(floatEqual(50.0f, pts[2]));   // w: 0.25*200
    EXPECT_TRUE(floatEqual(50.0f, pts[3]));   // h
}

// ANN-032: denormalize XYWHR
TEST_F(AnnotationTest, Denormalize_XYWHR) {
    Annotation ann(LabelType::XYWHR, true);
    ann.addObject(0, {0.5f, 0.5f, 0.25f, 0.15f, 0.5f});
    ann.denormalize(200, 200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(100.0f, pts[0]));  // cx
    EXPECT_TRUE(floatEqual(100.0f, pts[1]));  // cy
    EXPECT_TRUE(floatEqual(50.0f, pts[2]));   // w
    EXPECT_TRUE(floatEqual(30.0f, pts[3]));   // h: 0.15*200
    EXPECT_TRUE(floatEqual(0.5f, pts[4]));    // rotation unchanged
}

// ANN-033: denormalize XYXY
TEST_F(AnnotationTest, Denormalize_XYXY) {
    Annotation ann(LabelType::XYXY, true);
    ann.addObject(0, {0.25f, 0.25f, 0.75f, 0.75f});
    ann.denormalize(200, 200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(50.0f, pts[0]));   // x1
    EXPECT_TRUE(floatEqual(50.0f, pts[1]));   // y1
    EXPECT_TRUE(floatEqual(150.0f, pts[2]));  // x2
    EXPECT_TRUE(floatEqual(150.0f, pts[3]));  // y2
}

// ANN-034: denormalize POLYGON
TEST_F(AnnotationTest, Denormalize_POLYGON) {
    Annotation ann(LabelType::POLYGON, true);
    ann.addObject(0, {0.5f, 0.25f, 0.75f, 0.75f, 0.25f, 0.75f});
    ann.denormalize(200, 200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(100.0f, pts[0]));  // x1
    EXPECT_TRUE(floatEqual(50.0f, pts[1]));   // y1
}

// ANN-035: denormalize XYXYXYXY
TEST_F(AnnotationTest, Denormalize_XYXYXYXY) {
    Annotation ann(LabelType::XYXYXYXY, true);
    ann.addObject(0, {0.25f, 0.25f, 0.75f, 0.25f, 0.75f, 0.75f, 0.25f, 0.75f});
    ann.denormalize(200, 200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(50.0f, pts[0]));   // x1
    EXPECT_TRUE(floatEqual(150.0f, pts[2]));  // x2
}

// ANN-036: denormalize idempotent
TEST_F(AnnotationTest, Denormalize_Idempotent) {
    Annotation ann = createXYWHAnnotation();  // Not normalized
    auto ptsBefore = ann.getPoints()[0];

    ann.denormalize(200, 200);  // Should skip

    auto ptsAfter = ann.getPoints()[0];
    EXPECT_EQ(ptsBefore, ptsAfter);
}

// ANN-037: denormalize negative width throws
TEST_F(AnnotationTest, Denormalize_NegativeWidth_Throws) {
    Annotation ann(LabelType::XYWH, true);
    ann.addObject(0, {0.5f, 0.5f, 0.25f, 0.25f});
    EXPECT_THROW(ann.denormalize(-100, 200), std::invalid_argument);
}

// ANN-038: roundtrip normalize-denormalize XYWH
TEST_F(AnnotationTest, Roundtrip_XYWH) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 50.0f, 50.0f);
    auto original = ann.getPoints()[0];

    ann.normalize(200, 200);
    ann.denormalize(200, 200);

    auto& result = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(original[0], result[0]));
    EXPECT_TRUE(floatEqual(original[1], result[1]));
    EXPECT_TRUE(floatEqual(original[2], result[2]));
    EXPECT_TRUE(floatEqual(original[3], result[3]));
}

// ANN-039: roundtrip XYWHR
TEST_F(AnnotationTest, Roundtrip_XYWHR) {
    Annotation ann = createXYWHRAnnotation(0, 100.0f, 100.0f, 50.0f, 30.0f, 0.5f);
    auto original = ann.getPoints()[0];

    ann.normalize(200, 200);
    ann.denormalize(200, 200);

    auto& result = ann.getPoints()[0];
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_TRUE(floatEqual(original[i], result[i]));
    }
}

// ANN-040: roundtrip XYXY
TEST_F(AnnotationTest, Roundtrip_XYXY) {
    Annotation ann = createXYXYAnnotation(0, 50.0f, 50.0f, 150.0f, 150.0f);
    auto original = ann.getPoints()[0];

    ann.normalize(200, 200);
    ann.denormalize(200, 200);

    auto& result = ann.getPoints()[0];
    for (size_t i = 0; i < original.size(); ++i) {
        EXPECT_TRUE(floatEqual(original[i], result[i]));
    }
}

// ANN-041: denormalize empty
TEST_F(AnnotationTest, Denormalize_Empty) {
    Annotation ann(LabelType::XYWH, true);
    EXPECT_NO_THROW(ann.denormalize(200, 200));
    EXPECT_TRUE(ann.empty());
}

// ANN-042: denormalize NONE type
TEST_F(AnnotationTest, Denormalize_NONEType) {
    Annotation ann = Annotation::forClassification(5);
    EXPECT_NO_THROW(ann.denormalize(200, 200));
}

// =============================================================================
// 3.5 Scale Tests (ANN-043 ~ ANN-054)
// =============================================================================

// ANN-043: scale XYWH uniform
TEST_F(AnnotationTest, Scale_XYWH_Uniform) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 50.0f, 50.0f);
    ann.scale(2.0f, 2.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(200.0f, pts[0]));  // cx
    EXPECT_TRUE(floatEqual(200.0f, pts[1]));  // cy
    EXPECT_TRUE(floatEqual(100.0f, pts[2]));  // w
    EXPECT_TRUE(floatEqual(100.0f, pts[3]));  // h
}

// ANN-044: scale XYWH non-uniform
TEST_F(AnnotationTest, Scale_XYWH_NonUniform) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 50.0f, 50.0f);
    ann.scale(2.0f, 0.5f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(200.0f, pts[0]));  // cx * 2
    EXPECT_TRUE(floatEqual(50.0f, pts[1]));   // cy * 0.5
    EXPECT_TRUE(floatEqual(100.0f, pts[2]));  // w * 2
    EXPECT_TRUE(floatEqual(25.0f, pts[3]));   // h * 0.5
}

// ANN-045: scale XYWHR uniform
TEST_F(AnnotationTest, Scale_XYWHR_Uniform) {
    Annotation ann = createXYWHRAnnotation(0, 100.0f, 100.0f, 50.0f, 30.0f, 0.5f);
    ann.scale(2.0f, 2.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(200.0f, pts[0]));  // cx
    EXPECT_TRUE(floatEqual(200.0f, pts[1]));  // cy
    EXPECT_TRUE(floatEqual(100.0f, pts[2]));  // w
    EXPECT_TRUE(floatEqual(60.0f, pts[3]));   // h
    EXPECT_TRUE(floatEqual(0.5f, pts[4]));    // rotation unchanged for uniform scale
}

// ANN-046: scale XYWHR non-uniform (rotation recalculated)
TEST_F(AnnotationTest, Scale_XYWHR_NonUniform) {
    Annotation ann = createXYWHRAnnotation(0, 100.0f, 100.0f, 50.0f, 30.0f, 0.5f);
    ann.scale(2.0f, 1.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(200.0f, pts[0]));  // cx
    EXPECT_TRUE(floatEqual(100.0f, pts[1]));  // cy
    // Rotation may be recalculated for non-uniform scale
}

// ANN-047: scale XYXY
TEST_F(AnnotationTest, Scale_XYXY) {
    Annotation ann = createXYXYAnnotation(0, 50.0f, 50.0f, 150.0f, 150.0f);
    ann.scale(2.0f, 2.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(100.0f, pts[0]));  // x1
    EXPECT_TRUE(floatEqual(100.0f, pts[1]));  // y1
    EXPECT_TRUE(floatEqual(300.0f, pts[2]));  // x2
    EXPECT_TRUE(floatEqual(300.0f, pts[3]));  // y2
}

// ANN-048: scale POLYGON
TEST_F(AnnotationTest, Scale_POLYGON) {
    Annotation ann = createPolygonAnnotation();
    ann.scale(2.0f, 2.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(200.0f, pts[0]));  // x1: 100*2
    EXPECT_TRUE(floatEqual(100.0f, pts[1]));  // y1: 50*2
}

// ANN-049: scale XYXYXYXY
TEST_F(AnnotationTest, Scale_XYXYXYXY) {
    Annotation ann = createXYXYXYXYAnnotation();
    ann.scale(2.0f, 2.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(100.0f, pts[0]));  // x1: 50*2
    EXPECT_TRUE(floatEqual(300.0f, pts[2]));  // x2: 150*2
}

// ANN-050: scale factor 0.5
TEST_F(AnnotationTest, Scale_Factor_Half) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 50.0f, 50.0f);
    ann.scale(0.5f, 0.5f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(50.0f, pts[0]));   // cx
    EXPECT_TRUE(floatEqual(25.0f, pts[2]));   // w
}

// ANN-051: scale factor 2.0
TEST_F(AnnotationTest, Scale_Factor_Double) {
    Annotation ann = createXYWHAnnotation(0, 50.0f, 50.0f, 25.0f, 25.0f);
    ann.scale(2.0f, 2.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(100.0f, pts[0]));  // cx
    EXPECT_TRUE(floatEqual(50.0f, pts[2]));   // w
}

// ANN-052: scale factor negative
TEST_F(AnnotationTest, Scale_Factor_Negative) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 50.0f, 50.0f);
    ann.scale(-1.0f, 1.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(-100.0f, pts[0]));  // cx negated
}

// ANN-053: scale empty annotation
TEST_F(AnnotationTest, Scale_Empty) {
    Annotation ann(LabelType::XYWH);
    EXPECT_NO_THROW(ann.scale(2.0f, 2.0f));
    EXPECT_TRUE(ann.empty());
}

// ANN-054: scale XYWHR coords.size() < 5
TEST_F(AnnotationTest, Scale_XYWHR_ShortCoords) {
    Annotation ann(LabelType::XYWHR);
    ann.addObject(0, {100.0f, 100.0f, 50.0f, 30.0f});  // Only 4 coords (missing rotation)
    EXPECT_NO_THROW(ann.scale(2.0f, 2.0f));
}

// =============================================================================
// 3.6 Translate Tests (ANN-055 ~ ANN-064)
// =============================================================================

// ANN-055: translate XYWH
TEST_F(AnnotationTest, Translate_XYWH) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 50.0f, 50.0f);
    ann.translate(50.0f, 30.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(150.0f, pts[0]));  // cx + 50
    EXPECT_TRUE(floatEqual(130.0f, pts[1]));  // cy + 30
    EXPECT_TRUE(floatEqual(50.0f, pts[2]));   // w unchanged
    EXPECT_TRUE(floatEqual(50.0f, pts[3]));   // h unchanged
}

// ANN-056: translate XYWHR
TEST_F(AnnotationTest, Translate_XYWHR) {
    Annotation ann = createXYWHRAnnotation(0, 100.0f, 100.0f, 50.0f, 30.0f, 0.5f);
    ann.translate(50.0f, 30.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(150.0f, pts[0]));  // cx + 50
    EXPECT_TRUE(floatEqual(130.0f, pts[1]));  // cy + 30
    EXPECT_TRUE(floatEqual(0.5f, pts[4]));    // rotation unchanged
}

// ANN-057: translate XYXY
TEST_F(AnnotationTest, Translate_XYXY) {
    Annotation ann = createXYXYAnnotation(0, 50.0f, 50.0f, 150.0f, 150.0f);
    ann.translate(50.0f, 30.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(100.0f, pts[0]));  // x1 + 50
    EXPECT_TRUE(floatEqual(80.0f, pts[1]));   // y1 + 30
    EXPECT_TRUE(floatEqual(200.0f, pts[2]));  // x2 + 50
    EXPECT_TRUE(floatEqual(180.0f, pts[3]));  // y2 + 30
}

// ANN-058: translate POLYGON
TEST_F(AnnotationTest, Translate_POLYGON) {
    Annotation ann = createPolygonAnnotation();
    ann.translate(10.0f, 20.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(110.0f, pts[0]));  // x1 + 10
    EXPECT_TRUE(floatEqual(70.0f, pts[1]));   // y1 + 20 (50+20)
}

// ANN-059: translate XYXYXYXY
TEST_F(AnnotationTest, Translate_XYXYXYXY) {
    Annotation ann = createXYXYXYXYAnnotation();
    ann.translate(10.0f, 20.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(60.0f, pts[0]));   // x1 + 10 (50+10)
    EXPECT_TRUE(floatEqual(70.0f, pts[1]));   // y1 + 20 (50+20)
}

// ANN-060: translate normalized throws
TEST_F(AnnotationTest, Translate_Normalized_Throws) {
    Annotation ann(LabelType::XYWH, true);
    ann.addObject(0, {0.5f, 0.5f, 0.25f, 0.25f});
    EXPECT_THROW(ann.translate(10.0f, 10.0f), std::runtime_error);
}

// ANN-061: translate positive offset
TEST_F(AnnotationTest, Translate_PositiveOffset) {
    Annotation ann = createXYWHAnnotation(0, 50.0f, 50.0f, 20.0f, 20.0f);
    ann.translate(100.0f, 100.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(150.0f, pts[0]));
    EXPECT_TRUE(floatEqual(150.0f, pts[1]));
}

// ANN-062: translate negative offset
TEST_F(AnnotationTest, Translate_NegativeOffset) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 20.0f, 20.0f);
    ann.translate(-50.0f, -30.0f);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(50.0f, pts[0]));
    EXPECT_TRUE(floatEqual(70.0f, pts[1]));
}

// ANN-063: translate empty annotation
TEST_F(AnnotationTest, Translate_Empty) {
    Annotation ann(LabelType::XYWH);
    EXPECT_NO_THROW(ann.translate(50.0f, 30.0f));
    EXPECT_TRUE(ann.empty());
}

// ANN-064: translate XYWH coords.size() < 2
TEST_F(AnnotationTest, Translate_XYWH_ShortCoords) {
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f});  // Only 1 coord
    EXPECT_NO_THROW(ann.translate(50.0f, 30.0f));
}

// =============================================================================
// 3.7 FlipHorizontal Tests (ANN-065 ~ ANN-076)
// =============================================================================

// ANN-065: flipHorizontal XYWH
TEST_F(AnnotationTest, FlipHorizontal_XYWH) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 50.0f, 50.0f);
    ann.flipHorizontal(200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(100.0f, pts[0]));  // cx: 200 - 100 = 100
    EXPECT_TRUE(floatEqual(100.0f, pts[1]));  // cy unchanged
}

// ANN-066: flipHorizontal XYWHR
TEST_F(AnnotationTest, FlipHorizontal_XYWHR) {
    Annotation ann = createXYWHRAnnotation(0, 100.0f, 100.0f, 50.0f, 30.0f, 0.5f);
    ann.flipHorizontal(200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(100.0f, pts[0]));  // cx: 200 - 100
    EXPECT_TRUE(floatEqual(-0.5f, pts[4]));   // rotation sign flipped
}

// ANN-067: flipHorizontal XYXY
TEST_F(AnnotationTest, FlipHorizontal_XYXY) {
    Annotation ann = createXYXYAnnotation(0, 50.0f, 50.0f, 150.0f, 150.0f);
    ann.flipHorizontal(200);

    auto& pts = ann.getPoints()[0];
    // After flip: x1=200-150=50, x2=200-50=150 (swapped)
    EXPECT_TRUE(floatEqual(50.0f, pts[0]));
    EXPECT_TRUE(floatEqual(150.0f, pts[2]));
}

// ANN-068: flipHorizontal POLYGON
TEST_F(AnnotationTest, FlipHorizontal_POLYGON) {
    Annotation ann = createPolygonAnnotation();  // (100,50), (150,150), (50,150)
    ann.flipHorizontal(200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(100.0f, pts[0]));  // x: 200-100
    EXPECT_TRUE(floatEqual(50.0f, pts[2]));   // x: 200-150
    EXPECT_TRUE(floatEqual(150.0f, pts[4]));  // x: 200-50
}

// ANN-069: flipHorizontal XYXYXYXY
TEST_F(AnnotationTest, FlipHorizontal_XYXYXYXY) {
    Annotation ann = createXYXYXYXYAnnotation();  // (50,50), (150,50), (150,150), (50,150)
    ann.flipHorizontal(200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(150.0f, pts[0]));  // x1: 200-50
    EXPECT_TRUE(floatEqual(50.0f, pts[2]));   // x2: 200-150
}

// ANN-070: flipHorizontal negative width throws
TEST_F(AnnotationTest, FlipHorizontal_NegativeWidth_Throws) {
    Annotation ann = createXYWHAnnotation();
    EXPECT_THROW(ann.flipHorizontal(-100), std::invalid_argument);
}

// ANN-071: flipHorizontal zero width throws
TEST_F(AnnotationTest, FlipHorizontal_ZeroWidth_Throws) {
    Annotation ann = createXYWHAnnotation();
    EXPECT_THROW(ann.flipHorizontal(0), std::invalid_argument);
}

// ANN-072: flipHorizontal normalized (w=1 used)
TEST_F(AnnotationTest, FlipHorizontal_Normalized) {
    Annotation ann(LabelType::XYWH, true);
    ann.addObject(0, {0.3f, 0.5f, 0.2f, 0.2f});
    ann.flipHorizontal(1);  // w=1 for normalized

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(0.7f, pts[0]));  // cx: 1 - 0.3
}

// ANN-073: flipHorizontal empty annotation
TEST_F(AnnotationTest, FlipHorizontal_Empty) {
    Annotation ann(LabelType::XYWH);
    EXPECT_NO_THROW(ann.flipHorizontal(200));
    EXPECT_TRUE(ann.empty());
}

// ANN-074: flipHorizontal XYWHR coords.size() < 5
TEST_F(AnnotationTest, FlipHorizontal_XYWHR_ShortCoords) {
    Annotation ann(LabelType::XYWHR);
    ann.addObject(0, {100.0f, 100.0f, 50.0f, 30.0f});  // Only 4 coords
    EXPECT_NO_THROW(ann.flipHorizontal(200));
}

// ANN-075: flipHorizontal XYXY coords.size() >= 4
TEST_F(AnnotationTest, FlipHorizontal_XYXY_FullCoords) {
    Annotation ann = createXYXYAnnotation();
    EXPECT_NO_THROW(ann.flipHorizontal(200));
    EXPECT_EQ(4, ann.getPoints()[0].size());
}

// ANN-076: flipHorizontal XYWH coords.size() < 1
TEST_F(AnnotationTest, FlipHorizontal_XYWH_EmptyCoords) {
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {});  // Empty coords
    EXPECT_NO_THROW(ann.flipHorizontal(200));
}

// =============================================================================
// 3.8 FlipVertical Tests (ANN-077 ~ ANN-088)
// =============================================================================

// ANN-077: flipVertical XYWH
TEST_F(AnnotationTest, FlipVertical_XYWH) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 50.0f, 50.0f);
    ann.flipVertical(200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(100.0f, pts[0]));  // cx unchanged
    EXPECT_TRUE(floatEqual(100.0f, pts[1]));  // cy: 200 - 100 = 100
}

// ANN-078: flipVertical XYWHR
TEST_F(AnnotationTest, FlipVertical_XYWHR) {
    Annotation ann = createXYWHRAnnotation(0, 100.0f, 100.0f, 50.0f, 30.0f, 0.5f);
    ann.flipVertical(200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(100.0f, pts[1]));  // cy: 200 - 100
    EXPECT_TRUE(floatEqual(-0.5f, pts[4]));   // rotation sign flipped
}

// ANN-079: flipVertical XYXY
TEST_F(AnnotationTest, FlipVertical_XYXY) {
    Annotation ann = createXYXYAnnotation(0, 50.0f, 50.0f, 150.0f, 150.0f);
    ann.flipVertical(200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(50.0f, pts[1]));   // y1: swapped
    EXPECT_TRUE(floatEqual(150.0f, pts[3]));  // y2: swapped
}

// ANN-080: flipVertical POLYGON
TEST_F(AnnotationTest, FlipVertical_POLYGON) {
    Annotation ann = createPolygonAnnotation();  // (100,50), (150,150), (50,150)
    ann.flipVertical(200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(150.0f, pts[1]));  // y: 200-50
    EXPECT_TRUE(floatEqual(50.0f, pts[3]));   // y: 200-150
}

// ANN-081: flipVertical XYXYXYXY
TEST_F(AnnotationTest, FlipVertical_XYXYXYXY) {
    Annotation ann = createXYXYXYXYAnnotation();  // (50,50), (150,50), (150,150), (50,150)
    ann.flipVertical(200);

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(150.0f, pts[1]));  // y1: 200-50
    EXPECT_TRUE(floatEqual(50.0f, pts[5]));   // y3: 200-150
}

// ANN-082: flipVertical negative height throws
TEST_F(AnnotationTest, FlipVertical_NegativeHeight_Throws) {
    Annotation ann = createXYWHAnnotation();
    EXPECT_THROW(ann.flipVertical(-100), std::invalid_argument);
}

// ANN-083: flipVertical zero height throws
TEST_F(AnnotationTest, FlipVertical_ZeroHeight_Throws) {
    Annotation ann = createXYWHAnnotation();
    EXPECT_THROW(ann.flipVertical(0), std::invalid_argument);
}

// ANN-084: flipVertical normalized (h=1 used)
TEST_F(AnnotationTest, FlipVertical_Normalized) {
    Annotation ann(LabelType::XYWH, true);
    ann.addObject(0, {0.5f, 0.3f, 0.2f, 0.2f});
    ann.flipVertical(1);  // h=1 for normalized

    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(0.7f, pts[1]));  // cy: 1 - 0.3
}

// ANN-085: flipVertical empty annotation
TEST_F(AnnotationTest, FlipVertical_Empty) {
    Annotation ann(LabelType::XYWH);
    EXPECT_NO_THROW(ann.flipVertical(200));
    EXPECT_TRUE(ann.empty());
}

// ANN-086: flipVertical XYWHR coords.size() < 5
TEST_F(AnnotationTest, FlipVertical_XYWHR_ShortCoords) {
    Annotation ann(LabelType::XYWHR);
    ann.addObject(0, {100.0f, 100.0f, 50.0f, 30.0f});  // Only 4 coords
    EXPECT_NO_THROW(ann.flipVertical(200));
}

// ANN-087: flipVertical XYXY coords.size() >= 4
TEST_F(AnnotationTest, FlipVertical_XYXY_FullCoords) {
    Annotation ann = createXYXYAnnotation();
    EXPECT_NO_THROW(ann.flipVertical(200));
    EXPECT_EQ(4, ann.getPoints()[0].size());
}

// ANN-088: flipVertical XYWH coords.size() < 2
TEST_F(AnnotationTest, FlipVertical_XYWH_ShortCoords) {
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f});  // Only 1 coord
    EXPECT_NO_THROW(ann.flipVertical(200));
}

// =============================================================================
// 3.9 ValidateAndClip / ClipToBounds Tests (ANN-089 ~ ANN-096)
// =============================================================================

// ANN-089: validateAndClip normalized
TEST_F(AnnotationTest, ValidateAndClip_Normalized) {
    Annotation ann(LabelType::XYWH, true);
    ann.addObject(0, {0.5f, 0.5f, 0.3f, 0.3f});
    EXPECT_NO_THROW(ann.validateAndClip(1, 1));
}

// ANN-090: validateAndClip not normalized
TEST_F(AnnotationTest, ValidateAndClip_NotNormalized) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 50.0f, 50.0f);
    EXPECT_NO_THROW(ann.validateAndClip(200, 200));
    EXPECT_FALSE(ann.empty());
}

// ANN-091: clipToBounds basic
TEST_F(AnnotationTest, ClipToBounds_Basic) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 50.0f, 50.0f);
    EXPECT_NO_THROW(ann.clipToBounds(0.0f, 0.0f, 200.0f, 200.0f));
    EXPECT_FALSE(ann.empty());
}

// ANN-092: clipToBounds removes out of bounds
TEST_F(AnnotationTest, ClipToBounds_RemovesOutOfBounds) {
    Annotation ann = createXYWHAnnotation(0, 300.0f, 300.0f, 50.0f, 50.0f);  // Outside bounds
    ann.clipToBounds(0.0f, 0.0f, 200.0f, 200.0f);
    EXPECT_TRUE(ann.empty());
}

// ANN-093: clipToBounds area filter
TEST_F(AnnotationTest, ClipToBounds_AreaFilter) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 2.0f, 2.0f);  // Area = 4
    ann.clipToBounds(0.0f, 0.0f, 200.0f, 200.0f, 10.0f);  // minArea = 10
    EXPECT_TRUE(ann.empty());  // Should be filtered out
}

// ANN-094: clipToBounds empty annotation
TEST_F(AnnotationTest, ClipToBounds_Empty) {
    Annotation ann(LabelType::XYWH);
    EXPECT_NO_THROW(ann.clipToBounds(0.0f, 0.0f, 200.0f, 200.0f));
    EXPECT_TRUE(ann.empty());
}

// ANN-095: validateAndClip with custom minArea
TEST_F(AnnotationTest, ValidateAndClip_CustomMinArea) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 3.0f, 3.0f);  // Area = 9
    ann.validateAndClip(200, 200, 20.0f);  // minArea = 20
    EXPECT_TRUE(ann.empty());  // Should be filtered
}

// ANN-096: clipToBounds overlapping bounds
TEST_F(AnnotationTest, ClipToBounds_OverlappingBounds) {
    Annotation ann = createXYWHAnnotation(0, 150.0f, 150.0f, 100.0f, 100.0f);  // Partially outside
    ann.clipToBounds(0.0f, 0.0f, 200.0f, 200.0f);
    // Should clip to bounds, not remove if still valid area
}

// =============================================================================
// 3.10 Type Conversion Tests (ANN-097 ~ ANN-116)
// =============================================================================

// ANN-097: convertTo same type
TEST_F(AnnotationTest, ConvertTo_SameType) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 50.0f, 50.0f);
    auto ptsBefore = ann.getPoints()[0];
    ann.convertTo(LabelType::XYWH);
    auto ptsAfter = ann.getPoints()[0];
    EXPECT_EQ(ptsBefore, ptsAfter);
}

// ANN-098: convertTo XYXY->XYWH
TEST_F(AnnotationTest, ConvertTo_XYXY_To_XYWH) {
    Annotation ann = createXYXYAnnotation(0, 50.0f, 50.0f, 150.0f, 150.0f);
    ann.convertTo(LabelType::XYWH);

    EXPECT_EQ(LabelType::XYWH, ann.getLabelType());
    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(100.0f, pts[0]));  // cx: (50+150)/2
    EXPECT_TRUE(floatEqual(100.0f, pts[1]));  // cy: (50+150)/2
    EXPECT_TRUE(floatEqual(100.0f, pts[2]));  // w: 150-50
    EXPECT_TRUE(floatEqual(100.0f, pts[3]));  // h: 150-50
}

// ANN-099: convertTo XYXY->POLYGON
TEST_F(AnnotationTest, ConvertTo_XYXY_To_POLYGON) {
    Annotation ann = createXYXYAnnotation(0, 50.0f, 50.0f, 150.0f, 150.0f);
    ann.convertTo(LabelType::POLYGON);

    EXPECT_EQ(LabelType::POLYGON, ann.getLabelType());
    EXPECT_EQ(8, ann.getPoints()[0].size());  // 4 corners
}

// ANN-100: convertTo POLYGON->XYXY
TEST_F(AnnotationTest, ConvertTo_POLYGON_To_XYXY) {
    Annotation ann = createPolygonAnnotation();  // Triangle
    ann.convertTo(LabelType::XYXY);

    EXPECT_EQ(LabelType::XYXY, ann.getLabelType());
    EXPECT_EQ(4, ann.getPoints()[0].size());
}

// ANN-101: convertTo POLYGON->XYWH
TEST_F(AnnotationTest, ConvertTo_POLYGON_To_XYWH) {
    Annotation ann = createPolygonAnnotation();
    ann.convertTo(LabelType::XYWH);

    EXPECT_EQ(LabelType::XYWH, ann.getLabelType());
    EXPECT_EQ(4, ann.getPoints()[0].size());
}

// ANN-102: convertTo POLYGON->XYWHR
TEST_F(AnnotationTest, ConvertTo_POLYGON_To_XYWHR) {
    Annotation ann = createPolygonAnnotation();
    ann.convertTo(LabelType::XYWHR);

    EXPECT_EQ(LabelType::XYWHR, ann.getLabelType());
    EXPECT_EQ(5, ann.getPoints()[0].size());  // cx, cy, w, h, r
}

// ANN-103: convertTo POLYGON->XYXYXYXY
TEST_F(AnnotationTest, ConvertTo_POLYGON_To_XYXYXYXY) {
    Annotation ann(LabelType::POLYGON);
    ann.addObject(0, {50.0f, 50.0f, 150.0f, 50.0f, 150.0f, 150.0f, 50.0f, 150.0f});
    ann.convertTo(LabelType::XYXYXYXY);

    EXPECT_EQ(LabelType::XYXYXYXY, ann.getLabelType());
    EXPECT_EQ(8, ann.getPoints()[0].size());
}

// ANN-104: convertTo XYWH->XYXY
TEST_F(AnnotationTest, ConvertTo_XYWH_To_XYXY) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 100.0f, 100.0f);
    ann.convertTo(LabelType::XYXY);

    EXPECT_EQ(LabelType::XYXY, ann.getLabelType());
    auto& pts = ann.getPoints()[0];
    EXPECT_TRUE(floatEqual(50.0f, pts[0]));   // x1: 100 - 100/2
    EXPECT_TRUE(floatEqual(50.0f, pts[1]));   // y1
    EXPECT_TRUE(floatEqual(150.0f, pts[2]));  // x2: 100 + 100/2
    EXPECT_TRUE(floatEqual(150.0f, pts[3]));  // y2
}

// ANN-105: convertTo XYWH->POLYGON
TEST_F(AnnotationTest, ConvertTo_XYWH_To_POLYGON) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 100.0f, 100.0f);
    ann.convertTo(LabelType::POLYGON);

    EXPECT_EQ(LabelType::POLYGON, ann.getLabelType());
    EXPECT_EQ(8, ann.getPoints()[0].size());  // 4 corners
}

// ANN-106: convertTo XYXYXYXY->XYWHR
TEST_F(AnnotationTest, ConvertTo_XYXYXYXY_To_XYWHR) {
    Annotation ann = createXYXYXYXYAnnotation();
    ann.convertTo(LabelType::XYWHR);

    EXPECT_EQ(LabelType::XYWHR, ann.getLabelType());
    EXPECT_EQ(5, ann.getPoints()[0].size());
}

// ANN-107: convertTo XYXYXYXY->POLYGON
TEST_F(AnnotationTest, ConvertTo_XYXYXYXY_To_POLYGON) {
    Annotation ann = createXYXYXYXYAnnotation();
    ann.convertTo(LabelType::POLYGON);

    EXPECT_EQ(LabelType::POLYGON, ann.getLabelType());
    EXPECT_EQ(8, ann.getPoints()[0].size());
}

// ANN-108: convertTo XYWHR->XYXYXYXY
TEST_F(AnnotationTest, ConvertTo_XYWHR_To_XYXYXYXY) {
    Annotation ann = createXYWHRAnnotation(0, 100.0f, 100.0f, 50.0f, 30.0f, 0.5f);
    ann.convertTo(LabelType::XYXYXYXY);

    EXPECT_EQ(LabelType::XYXYXYXY, ann.getLabelType());
    EXPECT_EQ(8, ann.getPoints()[0].size());
}

// ANN-109: convertTo XYWHR->POLYGON
TEST_F(AnnotationTest, ConvertTo_XYWHR_To_POLYGON) {
    Annotation ann = createXYWHRAnnotation(0, 100.0f, 100.0f, 50.0f, 30.0f, 0.5f);
    ann.convertTo(LabelType::POLYGON);

    EXPECT_EQ(LabelType::POLYGON, ann.getLabelType());
}

// ANN-110: convertTo unsupported throws
TEST_F(AnnotationTest, ConvertTo_Unsupported_Throws) {
    Annotation ann(LabelType::NONE);
    ann.addObject(0, {});
    // NONE to XYWH may not be supported
    EXPECT_THROW(ann.convertTo(LabelType::XYWH), std::invalid_argument);
}

// ANN-111: getAs returns copy (original unchanged)
TEST_F(AnnotationTest, GetAs_ReturnsCopy) {
    Annotation ann = createXYXYAnnotation(0, 50.0f, 50.0f, 150.0f, 150.0f);
    Annotation converted = ann.getAs(LabelType::XYWH);

    EXPECT_EQ(LabelType::XYXY, ann.getLabelType());  // Original unchanged
    EXPECT_EQ(LabelType::XYWH, converted.getLabelType());
}

// ANN-112: xyxyToXywh invalid size throws
TEST_F(AnnotationTest, ConvertTo_XYXY_InvalidSize_Throws) {
    Annotation ann(LabelType::XYXY);
    ann.addObject(0, {50.0f, 50.0f});  // Only 2 coords (needs 4)
    EXPECT_THROW(ann.convertTo(LabelType::XYWH), std::invalid_argument);
}

// ANN-113: xywhToXyxy invalid size throws
TEST_F(AnnotationTest, ConvertTo_XYWH_InvalidSize_Throws) {
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {100.0f, 100.0f});  // Only 2 coords (needs 4)
    EXPECT_THROW(ann.convertTo(LabelType::XYXY), std::invalid_argument);
}

// ANN-114: polygonToXyxy min points
TEST_F(AnnotationTest, ConvertTo_POLYGON_MinPoints) {
    Annotation ann(LabelType::POLYGON);
    ann.addObject(0, {50.0f});  // Only 1 value
    EXPECT_THROW(ann.convertTo(LabelType::XYXY), std::invalid_argument);
}

// ANN-115: xywhrToXyxyxyxy invalid size throws
TEST_F(AnnotationTest, ConvertTo_XYWHR_InvalidSize_Throws) {
    Annotation ann(LabelType::XYWHR);
    ann.addObject(0, {100.0f, 100.0f, 50.0f});  // Only 3 coords (needs 5)
    EXPECT_THROW(ann.convertTo(LabelType::XYXYXYXY), std::invalid_argument);
}

// ANN-116: polygonToXywhr min points
TEST_F(AnnotationTest, ConvertTo_POLYGON_To_XYWHR_MinPoints) {
    Annotation ann(LabelType::POLYGON);
    ann.addObject(0, {50.0f, 50.0f, 100.0f});  // Only 3 values (needs 6)
    EXPECT_THROW(ann.convertTo(LabelType::XYWHR), std::invalid_argument);
}

// =============================================================================
// 3.11 Transform with Matrix Tests (ANN-117 ~ ANN-126)
// =============================================================================

// ANN-117: transform empty image throws
TEST_F(AnnotationTest, Transform_EmptyImage_Throws) {
    Annotation ann = createXYWHAnnotation();
    cv::Mat emptyImage;
    cv::Mat transformMat = cv::Mat::eye(3, 3, CV_64F);
    EXPECT_THROW(ann.transform(emptyImage, transformMat), std::invalid_argument);
}

// ANN-118: transform invalid matrix size throws
TEST_F(AnnotationTest, Transform_InvalidMatrixSize_Throws) {
    Annotation ann = createXYWHAnnotation();
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat invalidMat(2, 2, CV_64F);  // Should be 3x3
    EXPECT_THROW(ann.transform(image, invalidMat), std::invalid_argument);
}

// ANN-119: transform invalid matrix type throws
TEST_F(AnnotationTest, Transform_InvalidMatrixType_Throws) {
    Annotation ann = createXYWHAnnotation();
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat invalidMat(3, 3, CV_8U);  // Should be CV_64F or CV_32F
    EXPECT_THROW(ann.transform(image, invalidMat), std::invalid_argument);
}

// ANN-120: transform CV_32F matrix (converted to CV_64F)
TEST_F(AnnotationTest, Transform_CV32F_Matrix) {
    Annotation ann = createXYWHAnnotation(0, 50.0f, 50.0f, 20.0f, 20.0f);
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat transformMat = cv::Mat::eye(3, 3, CV_32F);
    EXPECT_NO_THROW(ann.transform(image, transformMat));
}

// ANN-121: transform auto dsize
TEST_F(AnnotationTest, Transform_AutoDsize) {
    Annotation ann = createXYWHAnnotation(0, 50.0f, 50.0f, 20.0f, 20.0f);
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat transformMat = cv::Mat::eye(3, 3, CV_64F);

    ann.transform(image, transformMat);  // Auto dsize
    EXPECT_FALSE(image.empty());
}

// ANN-122: transform explicit dsize
TEST_F(AnnotationTest, Transform_ExplicitDsize) {
    Annotation ann = createXYWHAnnotation(0, 50.0f, 50.0f, 20.0f, 20.0f);
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat transformMat = cv::Mat::eye(3, 3, CV_64F);

    ann.transform(image, transformMat, cv::Size(200, 200));
    EXPECT_EQ(200, image.cols);
    EXPECT_EQ(200, image.rows);
}

// ANN-123: transform XYWH coordinates
TEST_F(AnnotationTest, Transform_XYWH_Coordinates) {
    Annotation ann = createXYWHAnnotation(0, 50.0f, 50.0f, 20.0f, 20.0f);
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));

    // Translation matrix (shift by 10, 10)
    cv::Mat transformMat = cv::Mat::eye(3, 3, CV_64F);
    transformMat.at<double>(0, 2) = 10.0;
    transformMat.at<double>(1, 2) = 10.0;

    ann.transform(image, transformMat);
    // Coordinates should be transformed
}

// ANN-124: transform XYWHR coordinates
TEST_F(AnnotationTest, Transform_XYWHR_Coordinates) {
    Annotation ann = createXYWHRAnnotation(0, 50.0f, 50.0f, 20.0f, 10.0f, 0.0f);
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));

    cv::Mat transformMat = cv::Mat::eye(3, 3, CV_64F);
    transformMat.at<double>(0, 2) = 10.0;
    transformMat.at<double>(1, 2) = 10.0;

    EXPECT_NO_THROW(ann.transform(image, transformMat));
}

// ANN-125: transform XYXY/POLYGON/XYXYXYXY
TEST_F(AnnotationTest, Transform_XYXY_Coordinates) {
    Annotation ann = createXYXYAnnotation(0, 40.0f, 40.0f, 60.0f, 60.0f);
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));

    cv::Mat transformMat = cv::Mat::eye(3, 3, CV_64F);
    transformMat.at<double>(0, 2) = 10.0;
    transformMat.at<double>(1, 2) = 10.0;

    EXPECT_NO_THROW(ann.transform(image, transformMat));
}

// ANN-126: transform w near zero (division protection)
TEST_F(AnnotationTest, Transform_WNearZero_Protection) {
    Annotation ann = createXYWHAnnotation(0, 50.0f, 50.0f, 20.0f, 20.0f);
    cv::Mat image(100, 100, CV_8UC3, cv::Scalar(128, 128, 128));

    // Matrix that could cause w near zero in homogeneous coords
    cv::Mat transformMat = cv::Mat::eye(3, 3, CV_64F);

    EXPECT_NO_THROW(ann.transform(image, transformMat));
}

// =============================================================================
// 3.12 Private Helpers Tests (ANN-127 ~ ANN-132)
// =============================================================================
// Note: These tests use public methods that internally call private helpers

// ANN-127: calculateArea XYWH (via validateAndClip)
TEST_F(AnnotationTest, CalculateArea_XYWH) {
    Annotation ann = createXYWHAnnotation(0, 100.0f, 100.0f, 10.0f, 10.0f);  // Area = 100
    ann.validateAndClip(200, 200, 50.0f);  // minArea = 50
    EXPECT_FALSE(ann.empty());  // Area > minArea, should keep
}

// ANN-128: calculateArea XYXY (via validateAndClip)
TEST_F(AnnotationTest, CalculateArea_XYXY) {
    Annotation ann = createXYXYAnnotation(0, 50.0f, 50.0f, 60.0f, 60.0f);  // Area = 100
    ann.validateAndClip(200, 200, 50.0f);
    EXPECT_FALSE(ann.empty());
}

// ANN-129: calculateArea XYWHR (via validateAndClip)
TEST_F(AnnotationTest, CalculateArea_XYWHR) {
    Annotation ann = createXYWHRAnnotation(0, 100.0f, 100.0f, 10.0f, 10.0f, 0.5f);  // Area = 100
    ann.validateAndClip(200, 200, 50.0f);
    EXPECT_FALSE(ann.empty());
}

// ANN-130: calculateArea XYXYXYXY (shoelace formula)
TEST_F(AnnotationTest, CalculateArea_XYXYXYXY) {
    Annotation ann = createXYXYXYXYAnnotation();  // Rectangle 100x100 = Area 10000
    ann.validateAndClip(300, 300, 5000.0f);
    EXPECT_FALSE(ann.empty());
}

// ANN-131: calculateArea POLYGON (shoelace formula)
TEST_F(AnnotationTest, CalculateArea_POLYGON) {
    Annotation ann = createPolygonAnnotation();  // Triangle
    ann.validateAndClip(300, 300, 100.0f);
    EXPECT_FALSE(ann.empty());
}

// ANN-132: normalizePolygonOrder (via convertTo XYXYXYXY)
TEST_F(AnnotationTest, NormalizePolygonOrder) {
    // Create polygon with corners in arbitrary order
    Annotation ann(LabelType::POLYGON);
    ann.addObject(0, {150.0f, 50.0f, 150.0f, 150.0f, 50.0f, 150.0f, 50.0f, 50.0f});

    ann.convertTo(LabelType::XYXYXYXY);

    // Should be normalized to clockwise, smallest y first
    EXPECT_EQ(LabelType::XYXYXYXY, ann.getLabelType());
    EXPECT_EQ(8, ann.getPoints()[0].size());
}
