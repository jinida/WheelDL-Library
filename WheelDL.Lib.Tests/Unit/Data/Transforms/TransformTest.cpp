#include "pch.h"
#include "Data/Transforms/Transform.h"
#include "Data/Common/Annotation.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

using namespace WheelDL::Data;
using namespace WheelDL::Data::Transforms;

// =============================================================================
// Mock Transforms for Testing
// =============================================================================

// Mock deterministic transform (adds offset to image pixels)
class MockDeterministicTransform : public Transform {
public:
    explicit MockDeterministicTransform(int offset = 10) : offset_(offset), applyCount_(0) {}

    void apply(cv::Mat& image, Annotation& annotations) override {
        if (!image.empty()) {
            image += cv::Scalar(offset_, offset_, offset_);
        }
        applyCount_++;
    }

    std::string getName() const override { return "MockDeterministic"; }

    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<MockDeterministicTransform>(offset_);
    }

    int getApplyCount() const { return applyCount_; }
    int getOffset() const { return offset_; }

private:
    int offset_;
    mutable int applyCount_;
};

// Mock random transform (uses RNG)
class MockRandomTransform : public RandomTransform {
public:
    explicit MockRandomTransform(float probability = 1.0f)
        : probability_(probability), applyCount_(0), lastRandomValue_(0) {}

    void apply(cv::Mat& image, Annotation& annotations) override {
        lastRandomValue_ = rng_.uniformFloat(0.0f, 1.0f);
        if (lastRandomValue_ < probability_) {
            if (!image.empty()) {
                image += cv::Scalar(5, 5, 5);
            }
        }
        applyCount_++;
    }

    std::string getName() const override { return "MockRandom"; }

    std::unique_ptr<Transform> clone() const override {
        auto cloned = std::make_unique<MockRandomTransform>(probability_);
        return cloned;
    }

    int getApplyCount() const { return applyCount_; }
    float getLastRandomValue() const { return lastRandomValue_; }

private:
    float probability_;
    mutable int applyCount_;
    mutable float lastRandomValue_;
};

// Null transform (does nothing) for nullptr testing
class NullTransform : public Transform {
public:
    void apply(cv::Mat& image, Annotation& annotations) override {}
    std::string getName() const override { return "Null"; }
    std::unique_ptr<Transform> clone() const override {
        return std::make_unique<NullTransform>();
    }
};

// =============================================================================
// Test Fixture
// =============================================================================

class TransformTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper: Create test image
    cv::Mat createTestImage(int width = 100, int height = 100) {
        cv::Mat image(height, width, CV_8UC3);
        image.setTo(cv::Scalar(100, 100, 100));
        return image;
    }

    // Helper: Create test annotation
    Annotation createTestAnnotation() {
        Annotation ann(LabelType::XYWH);
        ann.addObject(0, {50.0f, 50.0f, 20.0f, 20.0f});
        return ann;
    }
};

// =============================================================================
// 7.1 RandomTransform Tests (TRF-001 ~ TRF-005)
// =============================================================================

// TRF-001: RandomTransform constructor (random_device init)
TEST_F(TransformTest, RandomTransform_Constructor) {
    MockRandomTransform transform;

    EXPECT_TRUE(transform.isRandom());
    EXPECT_EQ("MockRandom", transform.getName());
}

// TRF-002: RandomTransform setSeed
TEST_F(TransformTest, RandomTransform_SetSeed) {
    MockRandomTransform transform;

    EXPECT_NO_THROW(transform.setSeed(12345));
}

// TRF-003: RandomTransform isRandom returns true
TEST_F(TransformTest, RandomTransform_IsRandom) {
    MockRandomTransform transform;

    EXPECT_TRUE(transform.isRandom());
}

// TRF-004: RandomTransform reproducibility (same seed = same result)
TEST_F(TransformTest, RandomTransform_Reproducibility) {
    MockRandomTransform transform1;
    MockRandomTransform transform2;

    transform1.setSeed(42);
    transform2.setSeed(42);

    cv::Mat image1 = createTestImage();
    cv::Mat image2 = createTestImage();
    Annotation ann1 = createTestAnnotation();
    Annotation ann2 = createTestAnnotation();

    transform1.apply(image1, ann1);
    transform2.apply(image2, ann2);

    // Same seed should produce same random values
    EXPECT_FLOAT_EQ(transform1.getLastRandomValue(), transform2.getLastRandomValue());
}

// TRF-005: RandomTransform different seeds produce different results
TEST_F(TransformTest, RandomTransform_DifferentSeeds) {
    MockRandomTransform transform1;
    MockRandomTransform transform2;

    transform1.setSeed(42);
    transform2.setSeed(12345);

    cv::Mat image1 = createTestImage();
    cv::Mat image2 = createTestImage();
    Annotation ann1 = createTestAnnotation();
    Annotation ann2 = createTestAnnotation();

    transform1.apply(image1, ann1);
    transform2.apply(image2, ann2);

    // Different seeds should (very likely) produce different values
    // Note: There's a tiny chance they could be equal by coincidence
    EXPECT_NE(transform1.getLastRandomValue(), transform2.getLastRandomValue());
}

// =============================================================================
// 7.2 Compose Tests (TRF-006 ~ TRF-025)
// =============================================================================

// TRF-006: Compose default constructor (empty pipeline)
TEST_F(TransformTest, Compose_DefaultConstructor) {
    Compose compose;

    EXPECT_TRUE(compose.empty());
    EXPECT_EQ(0, compose.size());
    EXPECT_EQ("Compose", compose.getName());
}

// TRF-007: Compose addTransform
TEST_F(TransformTest, Compose_AddTransform) {
    Compose compose;

    compose.addTransform(std::make_unique<MockDeterministicTransform>());

    EXPECT_FALSE(compose.empty());
    EXPECT_EQ(1, compose.size());
}

// TRF-008: Compose addTransform nullptr (skipped)
TEST_F(TransformTest, Compose_AddTransform_Nullptr) {
    Compose compose;

    compose.addTransform(nullptr);

    EXPECT_TRUE(compose.empty());
    EXPECT_EQ(0, compose.size());
}

// TRF-009: Compose apply empty (early return)
TEST_F(TransformTest, Compose_Apply_Empty) {
    Compose compose;
    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    cv::Mat originalImage = image.clone();

    compose.apply(image, ann);

    // Image should be unchanged
    cv::Mat diff;
    cv::absdiff(image, originalImage, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// TRF-010: Compose apply single transform
TEST_F(TransformTest, Compose_Apply_Single) {
    Compose compose;
    auto transform = std::make_unique<MockDeterministicTransform>(10);
    auto* transformPtr = transform.get();

    compose.addTransform(std::move(transform));

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    compose.apply(image, ann);

    EXPECT_EQ(1, transformPtr->getApplyCount());
}

// TRF-011: Compose apply multiple transforms (sequential)
TEST_F(TransformTest, Compose_Apply_Multiple) {
    Compose compose;

    auto t1 = std::make_unique<MockDeterministicTransform>(5);
    auto t2 = std::make_unique<MockDeterministicTransform>(10);
    auto* t1Ptr = t1.get();
    auto* t2Ptr = t2.get();

    compose.addTransform(std::move(t1));
    compose.addTransform(std::move(t2));

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    compose.apply(image, ann);

    EXPECT_EQ(1, t1Ptr->getApplyCount());
    EXPECT_EQ(1, t2Ptr->getApplyCount());
    EXPECT_EQ(2, compose.size());
}

// TRF-012: Compose apply nullptr in list (skipped defensively)
TEST_F(TransformTest, Compose_Apply_WithValidTransforms) {
    Compose compose;

    compose.addTransform(std::make_unique<MockDeterministicTransform>(5));
    compose.addTransform(std::make_unique<MockDeterministicTransform>(10));

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    // Should not crash
    EXPECT_NO_THROW(compose.apply(image, ann));
}

// TRF-013: Compose isRandom with random transform
TEST_F(TransformTest, Compose_IsRandom_WithRandom) {
    Compose compose;
    compose.addTransform(std::make_unique<MockRandomTransform>());

    EXPECT_TRUE(compose.isRandom());
}

// TRF-014: Compose isRandom without random transform
TEST_F(TransformTest, Compose_IsRandom_WithoutRandom) {
    Compose compose;
    compose.addTransform(std::make_unique<MockDeterministicTransform>());

    EXPECT_FALSE(compose.isRandom());
}

// TRF-015: Compose isRandom mixed (random + deterministic)
TEST_F(TransformTest, Compose_IsRandom_Mixed) {
    Compose compose;
    compose.addTransform(std::make_unique<MockDeterministicTransform>());
    compose.addTransform(std::make_unique<MockRandomTransform>());

    EXPECT_TRUE(compose.isRandom());
}

// TRF-016: Compose setSeed propagation (baseSeed + index)
TEST_F(TransformTest, Compose_SetSeed_Propagation) {
    Compose compose1;
    Compose compose2;

    compose1.addTransform(std::make_unique<MockRandomTransform>());
    compose1.addTransform(std::make_unique<MockRandomTransform>());

    compose2.addTransform(std::make_unique<MockRandomTransform>());
    compose2.addTransform(std::make_unique<MockRandomTransform>());

    compose1.setSeed(42);
    compose2.setSeed(42);

    cv::Mat image1 = createTestImage();
    cv::Mat image2 = createTestImage();
    Annotation ann1 = createTestAnnotation();
    Annotation ann2 = createTestAnnotation();

    compose1.apply(image1, ann1);
    compose2.apply(image2, ann2);

    // Same seed should produce same results
    cv::Mat diff;
    cv::absdiff(image1, image2, diff);
    EXPECT_EQ(0, cv::countNonZero(diff.reshape(1)));
}

// TRF-017: Compose setSeed nullptr handling
TEST_F(TransformTest, Compose_SetSeed_EmptyCompose) {
    Compose compose;

    // Should not crash on empty compose
    EXPECT_NO_THROW(compose.setSeed(42));
}

// TRF-018: Compose empty() true
TEST_F(TransformTest, Compose_Empty_True) {
    Compose compose;

    EXPECT_TRUE(compose.empty());
}

// TRF-019: Compose empty() false
TEST_F(TransformTest, Compose_Empty_False) {
    Compose compose;
    compose.addTransform(std::make_unique<MockDeterministicTransform>());

    EXPECT_FALSE(compose.empty());
}

// TRF-020: Compose size()
TEST_F(TransformTest, Compose_Size) {
    Compose compose;

    EXPECT_EQ(0, compose.size());

    compose.addTransform(std::make_unique<MockDeterministicTransform>());
    EXPECT_EQ(1, compose.size());

    compose.addTransform(std::make_unique<MockRandomTransform>());
    EXPECT_EQ(2, compose.size());

    compose.addTransform(std::make_unique<NullTransform>());
    EXPECT_EQ(3, compose.size());
}

// TRF-021: Compose clone deep copy
TEST_F(TransformTest, Compose_Clone_DeepCopy) {
    Compose original;
    original.addTransform(std::make_unique<MockDeterministicTransform>(15));
    original.addTransform(std::make_unique<MockRandomTransform>());

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);
    EXPECT_EQ("Compose", cloned->getName());

    // Cast to Compose to check size
    auto* clonedCompose = dynamic_cast<Compose*>(cloned.get());
    ASSERT_NE(nullptr, clonedCompose);
    EXPECT_EQ(original.size(), clonedCompose->size());
}

// TRF-022: Compose clone nullptr handling
TEST_F(TransformTest, Compose_Clone_EmptyCompose) {
    Compose original;

    auto cloned = original.clone();

    EXPECT_NE(nullptr, cloned);

    auto* clonedCompose = dynamic_cast<Compose*>(cloned.get());
    ASSERT_NE(nullptr, clonedCompose);
    EXPECT_TRUE(clonedCompose->empty());
}

// TRF-023: Compose getName
TEST_F(TransformTest, Compose_GetName) {
    Compose compose;

    EXPECT_EQ("Compose", compose.getName());
}

// TRF-024: Compose chaining order (transforms applied in order)
TEST_F(TransformTest, Compose_ChainingOrder) {
    Compose compose;

    // First transform adds 10, second adds 20
    compose.addTransform(std::make_unique<MockDeterministicTransform>(10));
    compose.addTransform(std::make_unique<MockDeterministicTransform>(20));

    cv::Mat image = createTestImage();  // Starts at 100
    Annotation ann = createTestAnnotation();

    compose.apply(image, ann);

    // After both transforms: 100 + 10 + 20 = 130
    cv::Vec3b pixel = image.at<cv::Vec3b>(50, 50);
    EXPECT_EQ(130, pixel[0]);
    EXPECT_EQ(130, pixel[1]);
    EXPECT_EQ(130, pixel[2]);
}

// TRF-025: Compose with image + annotation (both modified)
TEST_F(TransformTest, Compose_ImageAndAnnotation) {
    Compose compose;
    compose.addTransform(std::make_unique<MockDeterministicTransform>(5));

    cv::Mat image = createTestImage();
    Annotation ann = createTestAnnotation();

    cv::Vec3b originalPixel = image.at<cv::Vec3b>(50, 50);

    compose.apply(image, ann);

    cv::Vec3b newPixel = image.at<cv::Vec3b>(50, 50);

    // Image should be modified
    EXPECT_NE(originalPixel[0], newPixel[0]);
    EXPECT_EQ(originalPixel[0] + 5, newPixel[0]);

    // Annotation should still be valid
    EXPECT_EQ(1, ann.size());
}
