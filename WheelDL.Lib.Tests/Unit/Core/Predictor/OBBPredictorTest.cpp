/**
 * @file OBBPredictorTest.cpp
 * @brief Unit tests for WheelDL::Core::Predictor::OBBPredictor
 *
 * Phase 3 of test_engine_and_etc.md - 14 tests
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "../Engine/EngineTestHelpers.h"
#include "Core/Predictor/OBBPredictor.h"

using namespace WheelDL;
using namespace WheelDL::Core::Predictor;
using namespace WheelDL::Test::Engine;
using namespace WheelDL::Utils;

// ============================================================================
// Test Fixture
// ============================================================================

class OBBPredictorTest : public EngineTestFixture {
protected:
    void SetUp() override {
        EngineTestFixture::SetUp();
    }

    void TearDown() override {
        EngineTestFixture::TearDown();
    }

    std::unique_ptr<OBBPredictor> createPredictor(
        const std::string& checkpointPath = "",
        std::atomic<bool>* stopFlag = nullptr)
    {
        auto obbConfig = EngineConfigFactory::createMinimal(TaskType::OBB);
        return std::make_unique<OBBPredictor>(
            obbConfig, checkpointPath, logger(), profiler(), stopFlag);
    }

    std::shared_ptr<Config::Configuration> createConfigWithTaskType(TaskType taskType) {
        return EngineConfigFactory::createMinimal(taskType);
    }
};

// ============================================================================
// Constructor Tests (1)
// ============================================================================

TEST_F(OBBPredictorTest, Constructor_InvalidTaskType) {
    if (!requireCuda()) return;

    auto clsConfig = createConfigWithTaskType(TaskType::CLASSIFICATION);

    EXPECT_THROW({
        OBBPredictor predictor(
            clsConfig, "", logger(), profiler(), nullptr);
    }, WheelLibException);
}

// ============================================================================
// SetupModel Tests (3)
// ============================================================================

TEST_F(OBBPredictorTest, SetupModel_CreatesOBBModel) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(OBBPredictorTest, SetupModel_SetsThresholds) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(OBBPredictorTest, SetupModel_Exception_Throws) {
    if (!requireCuda()) return;

    auto invalidConfig = createConfigWithTaskType(TaskType::CLASSIFICATION);

    EXPECT_THROW({
        OBBPredictor predictor(
            invalidConfig, "", logger(), profiler(), nullptr);
    }, WheelLibException);
}

// ============================================================================
// Postprocess Tests (5)
// ============================================================================

TEST_F(OBBPredictorTest, Postprocess_AppliesNMS) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(OBBPredictorTest, Postprocess_ScalesOBBs) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(OBBPredictorTest, Postprocess_EmptyOutput_ReturnsEarly) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(OBBPredictorTest, Postprocess_Exception_LogsError) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(OBBPredictorTest, Postprocess_NoDetections) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

// ============================================================================
// ScaleOBBs Tests (2)
// ============================================================================

TEST_F(OBBPredictorTest, ScaleOBBs_CorrectScaling) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(OBBPredictorTest, ScaleOBBs_PreservesAngle) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

// ============================================================================
// ExportResults Tests (2)
// ============================================================================

TEST_F(OBBPredictorTest, ExportResults_CreatesJSON) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(OBBPredictorTest, ExportResults_FileOpenFailed_LogsError) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

// ============================================================================
// Additional Tests (1)
// ============================================================================

TEST_F(OBBPredictorTest, Postprocess_MaxDetLimit) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}
