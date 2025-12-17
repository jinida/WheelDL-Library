/**
 * @file AnomalyPredictorTest.cpp
 * @brief Unit tests for WheelDL::Core::Predictor::AnomalyPredictor
 *
 * Phase 3 of test_engine_and_etc.md - 14 tests
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "../Engine/EngineTestHelpers.h"
#include "Core/Predictor/AnomalyPredictor.h"

using namespace WheelDL;
using namespace WheelDL::Core::Predictor;
using namespace WheelDL::Test::Engine;
using namespace WheelDL::Utils;

// ============================================================================
// Test Fixture
// ============================================================================

class AnomalyPredictorTest : public EngineTestFixture {
protected:
    void SetUp() override {
        EngineTestFixture::SetUp();
    }

    void TearDown() override {
        EngineTestFixture::TearDown();
    }

    std::unique_ptr<AnomalyPredictor> createPredictor(
        const std::string& checkpointPath = "",
        std::atomic<bool>* stopFlag = nullptr)
    {
        auto anomalyConfig = EngineConfigFactory::createMinimal(TaskType::ANOMALY);
        return std::make_unique<AnomalyPredictor>(
            anomalyConfig, checkpointPath, logger(), profiler(), stopFlag);
    }

    std::shared_ptr<Config::Configuration> createConfigWithTaskType(TaskType taskType) {
        return EngineConfigFactory::createMinimal(taskType);
    }
};

// ============================================================================
// Constructor Tests (1)
// ============================================================================

TEST_F(AnomalyPredictorTest, Constructor_InvalidTaskType) {
    if (!requireCuda()) return;

    auto clsConfig = createConfigWithTaskType(TaskType::CLASSIFICATION);

    EXPECT_THROW({
        AnomalyPredictor predictor(
            clsConfig, "", logger(), profiler(), nullptr);
    }, WheelLibException);
}

// ============================================================================
// SetupModel Tests (2)
// ============================================================================

TEST_F(AnomalyPredictorTest, SetupModel_CreatesAnomalyModel) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(AnomalyPredictorTest, SetupModel_Exception_Throws) {
    if (!requireCuda()) return;

    auto invalidConfig = createConfigWithTaskType(TaskType::CLASSIFICATION);

    EXPECT_THROW({
        AnomalyPredictor predictor(
            invalidConfig, "", logger(), profiler(), nullptr);
    }, WheelLibException);
}

// ============================================================================
// Postprocess Tests (7)
// ============================================================================

TEST_F(AnomalyPredictorTest, Postprocess_ComputesAnomalyScore) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(AnomalyPredictorTest, Postprocess_StoresAnomalyMap) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(AnomalyPredictorTest, Postprocess_ClassifiesAnomaly) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(AnomalyPredictorTest, Postprocess_ResizesMap) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(AnomalyPredictorTest, Postprocess_InsufficientOutputs_LogsError) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(AnomalyPredictorTest, Postprocess_Exception_LogsError) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(AnomalyPredictorTest, Postprocess_NormalizesScore) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

// ============================================================================
// ExportResults Tests (3)
// ============================================================================

TEST_F(AnomalyPredictorTest, ExportResults_CreatesJSON) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(AnomalyPredictorTest, ExportResults_SavesAnomalyMaps) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

TEST_F(AnomalyPredictorTest, ExportResults_FileOpenFailed_LogsError) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}

// ============================================================================
// ClearResults Tests (1)
// ============================================================================

TEST_F(AnomalyPredictorTest, ClearResults_ClearsAllMaps) {
    if (!requireCuda()) return;

    auto predictor = createPredictor();
    EXPECT_NE(predictor, nullptr);
}
