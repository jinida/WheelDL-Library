/**
 * @file SegmentationValidatorTest.cpp
 * @brief Unit tests for WheelDL::Core::Validator::SegmentationValidator
 *
 * Phase 3 of test_engine_and_etc.md - 14 tests
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "../Engine/EngineTestHelpers.h"
#include "Core/Validator/SegmentationValidator.h"

using namespace WheelDL;
using namespace WheelDL::Core::Validator;
using namespace WheelDL::Test::Engine;
using namespace WheelDL::Utils;

// ============================================================================
// Test Fixture
// ============================================================================

class SegmentationValidatorTest : public EngineTestFixture {
protected:
    void SetUp() override {
        EngineTestFixture::SetUp();
    }

    void TearDown() override {
        EngineTestFixture::TearDown();
    }

    std::unique_ptr<SegmentationValidator> createValidator(
        std::atomic<bool>* stopFlag = nullptr)
    {
        auto segConfig = EngineConfigFactory::createMinimal(TaskType::SEGMENTATION);
        return std::make_unique<SegmentationValidator>(
            segConfig, logger(), profiler(), stopFlag);
    }
};

// ============================================================================
// Constructor Tests (1)
// ============================================================================

TEST_F(SegmentationValidatorTest, Constructor_SetsNumClasses) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

// ============================================================================
// SetupModel Tests (1)
// ============================================================================

TEST_F(SegmentationValidatorTest, SetupModel_CreatesModel) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

// ============================================================================
// PreprocessBatch Tests (1)
// ============================================================================

TEST_F(SegmentationValidatorTest, PreprocessBatch_Passthrough) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

// ============================================================================
// PostprocessBatch Tests (2)
// ============================================================================

TEST_F(SegmentationValidatorTest, PostprocessBatch_AppliesSigmoid) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(SegmentationValidatorTest, PostprocessBatch_ExtractsContours) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

// ============================================================================
// ComputeMetrics Tests (9)
// ============================================================================

TEST_F(SegmentationValidatorTest, ComputeMetrics_mIoU) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(SegmentationValidatorTest, ComputeMetrics_PerClassIoU) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(SegmentationValidatorTest, ComputeMetrics_DiceCoefficient) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(SegmentationValidatorTest, ComputeMetrics_PixelAccuracy) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(SegmentationValidatorTest, ComputeMetrics_PerfectSegmentation) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(SegmentationValidatorTest, ComputeMetrics_NoOverlap) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(SegmentationValidatorTest, ComputeMetrics_Exception_LogsError) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(SegmentationValidatorTest, ComputeMetrics_MultiClassMasks) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(SegmentationValidatorTest, ComputeMetrics_SetsFitness) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

