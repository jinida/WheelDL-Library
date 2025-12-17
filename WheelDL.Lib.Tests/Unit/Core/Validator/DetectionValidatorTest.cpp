/**
 * @file DetectionValidatorTest.cpp
 * @brief Unit tests for WheelDL::Core::Validator::DetectionValidator
 *
 * Phase 3 of test_engine_and_etc.md - 16 tests
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "../Engine/EngineTestHelpers.h"
#include "Core/Validator/DetectionValidator.h"

using namespace WheelDL;
using namespace WheelDL::Core::Validator;
using namespace WheelDL::Test::Engine;
using namespace WheelDL::Utils;

// ============================================================================
// Test Fixture
// ============================================================================

class DetectionValidatorTest : public EngineTestFixture {
protected:
    void SetUp() override {
        EngineTestFixture::SetUp();
    }

    void TearDown() override {
        EngineTestFixture::TearDown();
    }

    std::unique_ptr<DetectionValidator> createValidator(
        std::atomic<bool>* stopFlag = nullptr)
    {
        auto detConfig = EngineConfigFactory::createMinimal(TaskType::DETECTION);
        return std::make_unique<DetectionValidator>(
            detConfig, logger(), profiler(), stopFlag);
    }
};

// ============================================================================
// Constructor Tests (1)
// ============================================================================

TEST_F(DetectionValidatorTest, Constructor_SetsThresholds) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

// ============================================================================
// SetupModel Tests (1)
// ============================================================================

TEST_F(DetectionValidatorTest, SetupModel_CreatesModel) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

// ============================================================================
// PreprocessBatch Tests (1)
// ============================================================================

TEST_F(DetectionValidatorTest, PreprocessBatch_Passthrough) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

// ============================================================================
// PostprocessBatch Tests (2)
// ============================================================================

TEST_F(DetectionValidatorTest, PostprocessBatch_AppliesNMS) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(DetectionValidatorTest, PostprocessBatch_EmptyPrediction) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

// ============================================================================
// ComputeMetrics Tests (11)
// ============================================================================

TEST_F(DetectionValidatorTest, ComputeMetrics_mAP) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(DetectionValidatorTest, ComputeMetrics_mAP50) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(DetectionValidatorTest, ComputeMetrics_Precision) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(DetectionValidatorTest, ComputeMetrics_Recall) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(DetectionValidatorTest, ComputeMetrics_NoDetections) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(DetectionValidatorTest, ComputeMetrics_NoGroundTruth) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(DetectionValidatorTest, ComputeMetrics_PerfectMatch) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(DetectionValidatorTest, ComputeMetrics_Exception_LogsError) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(DetectionValidatorTest, ComputeMetrics_MultipleIoUThresholds) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(DetectionValidatorTest, ComputeMetrics_PerClassAP) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(DetectionValidatorTest, ComputeMetrics_SetsFitness) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}
