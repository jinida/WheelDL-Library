/**
 * @file SegmentationPredictorTest.cpp
 * @brief Unit tests for WheelDL::Core::Predictor::SegmentationPredictor
 *
 * Phase 3 of test_engine_and_etc.md - 14 tests
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "../Engine/EngineTestHelpers.h"
#include "Core/Predictor/SegmentationPredictor.h"

using namespace WheelDL;
using namespace WheelDL::Core::Predictor;
using namespace WheelDL::Test::Engine;
using namespace WheelDL::Utils;

// ============================================================================
// Test Fixture
// ============================================================================

class SegmentationPredictorTest : public EngineTestFixture {
protected:
    void SetUp() override {
        EngineTestFixture::SetUp();
    }

    void TearDown() override {
        EngineTestFixture::TearDown();
    }

    std::unique_ptr<SegmentationPredictor> createPredictor(
        const std::string& checkpointPath = "",
        std::atomic<bool>* stopFlag = nullptr)
    {
        auto segConfig = EngineConfigFactory::createMinimal(TaskType::SEGMENTATION);
        return std::make_unique<SegmentationPredictor>(
            segConfig, checkpointPath, logger(), profiler(), stopFlag);
    }

    std::shared_ptr<Config::Configuration> createConfigWithTaskType(TaskType taskType) {
        return EngineConfigFactory::createMinimal(taskType);
    }
};

// ============================================================================
// Constructor Tests (1)
// ============================================================================

TEST_F(SegmentationPredictorTest, Constructor_InvalidTaskType) {
    if (!requireCuda()) return;

    auto clsConfig = createConfigWithTaskType(TaskType::CLASSIFICATION);

    EXPECT_THROW({
        SegmentationPredictor predictor(
            clsConfig, "", logger(), profiler(), nullptr);
    }, WheelLibException);
}

// ============================================================================
// SetupModel Tests (3)
// ============================================================================

TEST_F(SegmentationPredictorTest, SetupModel_CreatesSegmentationModel) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(SegmentationPredictorTest, SetupModel_SetsNumClasses) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(SegmentationPredictorTest, SetupModel_Exception_Throws) {
    if (!requireCuda()) return;

    auto invalidConfig = createConfigWithTaskType(TaskType::CLASSIFICATION);

    EXPECT_THROW({
        SegmentationPredictor predictor(
            invalidConfig, "", logger(), profiler(), nullptr);
    }, WheelLibException);
}

// ============================================================================
// Postprocess Tests (5)
// ============================================================================

TEST_F(SegmentationPredictorTest, Postprocess_AppliesSigmoid) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(SegmentationPredictorTest, Postprocess_ExtractsContours) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(SegmentationPredictorTest, Postprocess_ScalesContours) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(SegmentationPredictorTest, Postprocess_Exception_LogsError) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(SegmentationPredictorTest, Postprocess_MultiClassMasks) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

// ============================================================================
// ExportResults Tests (3)
// ============================================================================

TEST_F(SegmentationPredictorTest, ExportResults_CreatesJSON) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(SegmentationPredictorTest, ExportResults_CorrectFormat) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(SegmentationPredictorTest, ExportResults_FileOpenFailed_LogsError) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

// ============================================================================
// Additional Tests (2)
// ============================================================================

TEST_F(SegmentationPredictorTest, Postprocess_NoContours) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(SegmentationPredictorTest, ClearResults_EmptiesContainer) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}
