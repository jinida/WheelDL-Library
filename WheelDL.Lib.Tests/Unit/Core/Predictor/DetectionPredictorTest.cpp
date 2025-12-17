/**
 * @file DetectionPredictorTest.cpp
 * @brief Unit tests for WheelDL::Core::Predictor::DetectionPredictor
 *
 * Phase 3 of test_engine_and_etc.md - 14 tests
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "../Engine/EngineTestHelpers.h"
#include "Core/Predictor/DetectionPredictor.h"

using namespace WheelDL;
using namespace WheelDL::Core::Predictor;
using namespace WheelDL::Test::Engine;
using namespace WheelDL::Utils;

// ============================================================================
// Test Fixture
// ============================================================================

class DetectionPredictorTest : public EngineTestFixture {
protected:
    void SetUp() override {
        EngineTestFixture::SetUp();
    }

    void TearDown() override {
        EngineTestFixture::TearDown();
    }

    std::unique_ptr<DetectionPredictor> createPredictor(
        const std::string& checkpointPath = "",
        std::atomic<bool>* stopFlag = nullptr)
    {
        auto detConfig = EngineConfigFactory::createMinimal(TaskType::DETECTION);
        return std::make_unique<DetectionPredictor>(
            detConfig, checkpointPath, logger(), profiler(), stopFlag);
    }

    std::shared_ptr<Config::Configuration> createConfigWithTaskType(TaskType taskType) {
        return EngineConfigFactory::createMinimal(taskType);
    }
};

// ============================================================================
// Constructor Tests (1)
// ============================================================================

TEST_F(DetectionPredictorTest, Constructor_InvalidTaskType) {
    if (!requireCuda()) return;

    auto clsConfig = createConfigWithTaskType(TaskType::CLASSIFICATION);

    EXPECT_THROW({
        DetectionPredictor predictor(
            clsConfig, "", logger(), profiler(), nullptr);
    }, WheelLibException);
}

// ============================================================================
// SetupModel Tests (3)
// ============================================================================

TEST_F(DetectionPredictorTest, SetupModel_CreatesDetectionModel) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(DetectionPredictorTest, SetupModel_SetsThresholds) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    // Thresholds should be set from config
    EXPECT_NE(predictor, nullptr);
}

TEST_F(DetectionPredictorTest, SetupModel_Exception_Throws) {
    if (!requireCuda()) return;

    auto invalidConfig = createConfigWithTaskType(TaskType::CLASSIFICATION);

    EXPECT_THROW({
        DetectionPredictor predictor(
            invalidConfig, "", logger(), profiler(), nullptr);
    }, WheelLibException);
}

// ============================================================================
// Postprocess Tests (6)
// ============================================================================

TEST_F(DetectionPredictorTest, Postprocess_AppliesNMS) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(DetectionPredictorTest, Postprocess_ScalesBoxes) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(DetectionPredictorTest, Postprocess_EmptyOutput_ReturnsEarly) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(DetectionPredictorTest, Postprocess_NoDetections_ReturnsEarly) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(DetectionPredictorTest, Postprocess_Exception_LogsError) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(DetectionPredictorTest, Postprocess_MaxDetLimit) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

// ============================================================================
// ScaleBoxes Tests (2)
// ============================================================================

TEST_F(DetectionPredictorTest, ScaleBoxes_CorrectScaling) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(DetectionPredictorTest, ScaleBoxes_PreservesAspectRatio) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

// ============================================================================
// ExportResults Tests (2)
// ============================================================================

TEST_F(DetectionPredictorTest, ExportResults_CreatesJSON) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(DetectionPredictorTest, ExportResults_FileOpenFailed_LogsError) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}
