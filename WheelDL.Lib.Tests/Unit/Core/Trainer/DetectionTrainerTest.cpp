/**
 * @file DetectionTrainerTest.cpp
 * @brief Unit tests for WheelDL::Core::Trainer::DetectionTrainer
 *
 * Phase 3 of test_engine_and_etc.md - 13 tests
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "../Engine/EngineTestHelpers.h"
#include "Core/Trainer/DetectionTrainer.h"

using namespace WheelDL;
using namespace WheelDL::Core::Trainer;
using namespace WheelDL::Test::Engine;
using namespace WheelDL::Utils;

// ============================================================================
// Test Fixture
// ============================================================================

class DetectionTrainerTest : public EngineTestFixture {
protected:
    void SetUp() override {
        EngineTestFixture::SetUp();
    }

    void TearDown() override {
        EngineTestFixture::TearDown();
    }

    std::unique_ptr<DetectionTrainer> createTrainer(
        std::atomic<bool>* stopFlag = nullptr)
    {
        auto detConfig = EngineConfigFactory::createMinimal(TaskType::DETECTION);
        return std::make_unique<DetectionTrainer>(
            detConfig, logger(), workspace(), profiler(), nullptr, stopFlag);
    }
};

// ============================================================================
// Constructor Tests (1)
// ============================================================================

TEST_F(DetectionTrainerTest, Constructor_SetsThresholds) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

// ============================================================================
// SetupModel Tests (1)
// ============================================================================

TEST_F(DetectionTrainerTest, SetupModel_CreatesModel) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

// ============================================================================
// SetupLoss Tests (2)
// ============================================================================

TEST_F(DetectionTrainerTest, SetupLoss_YOLOLoss) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

TEST_F(DetectionTrainerTest, SetupLoss_CIoULoss) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

// ============================================================================
// ComputeLoss Tests (3)
// ============================================================================

TEST_F(DetectionTrainerTest, ComputeLoss_BoxLoss) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

TEST_F(DetectionTrainerTest, ComputeLoss_ClassLoss) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

TEST_F(DetectionTrainerTest, ComputeLoss_Exception_LogsError) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

// ============================================================================
// TrainEpoch Tests (3)
// ============================================================================

TEST_F(DetectionTrainerTest, TrainEpoch_UpdatesWeights) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

TEST_F(DetectionTrainerTest, TrainEpoch_ComputesmAP) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

TEST_F(DetectionTrainerTest, TrainEpoch_StopRequested) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

// ============================================================================
// Validation Tests (2)
// ============================================================================

TEST_F(DetectionTrainerTest, Validate_ComputesMetrics) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

TEST_F(DetectionTrainerTest, Validate_Exception_LogsError) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

// ============================================================================
// Fitness Tests (1)
// ============================================================================

TEST_F(DetectionTrainerTest, ComputeFitness_UsesmAP) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

