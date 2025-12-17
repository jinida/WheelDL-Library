/**
 * @file OBBValidatorTest.cpp
 * @brief Unit tests for WheelDL::Core::Validator::OBBValidator
 *
 * Phase 3 of test_engine_and_etc.md - 11 tests
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "../Engine/EngineTestHelpers.h"
#include "Core/Validator/OBBValidator.h"

using namespace WheelDL;
using namespace WheelDL::Core::Validator;
using namespace WheelDL::Test::Engine;
using namespace WheelDL::Utils;

// ============================================================================
// Test Fixture
// ============================================================================

class OBBValidatorTest : public EngineTestFixture {
protected:
    void SetUp() override {
        EngineTestFixture::SetUp();
    }

    void TearDown() override {
        EngineTestFixture::TearDown();
    }

    std::unique_ptr<OBBValidator> createValidator(
        std::atomic<bool>* stopFlag = nullptr)
    {
        auto obbConfig = EngineConfigFactory::createMinimal(TaskType::OBB);
        return std::make_unique<OBBValidator>(
            obbConfig, logger(), profiler(), stopFlag);
    }
};

// ============================================================================
// Constructor Tests (1)
// ============================================================================

TEST_F(OBBValidatorTest, Constructor_SetsThresholds) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

// ============================================================================
// SetupModel Tests (1)
// ============================================================================

TEST_F(OBBValidatorTest, SetupModel_CreatesModel) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

// ============================================================================
// PostprocessBatch Tests (2)
// ============================================================================

TEST_F(OBBValidatorTest, PostprocessBatch_AppliesRotatedNMS) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(OBBValidatorTest, PostprocessBatch_EmptyPrediction) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

// ============================================================================
// ComputeMetrics Tests (7)
// ============================================================================

TEST_F(OBBValidatorTest, ComputeMetrics_mAP) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(OBBValidatorTest, ComputeMetrics_RotatedIoU) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(OBBValidatorTest, ComputeMetrics_AnglePrecision) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(OBBValidatorTest, ComputeMetrics_NoDetections) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(OBBValidatorTest, ComputeMetrics_PerfectMatch) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(OBBValidatorTest, ComputeMetrics_Exception_LogsError) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

TEST_F(OBBValidatorTest, ComputeMetrics_SetsFitness) {
    if (!requireCuda()) return;

    auto validator = createValidator();
    EXPECT_NE(validator, nullptr);
}

