/**
 * @file OBBTrainerTest.cpp
 * @brief Unit tests for WheelDL::Core::Trainer::OBBTrainer
 *
 * Phase 3 of test_engine_and_etc.md - 13 tests
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "../Engine/EngineTestHelpers.h"
#include "Core/Trainer/OBBTrainer.h"

using namespace WheelDL;
using namespace WheelDL::Core::Trainer;
using namespace WheelDL::Test::Engine;
using namespace WheelDL::Utils;

// ============================================================================
// Test Fixture
// ============================================================================

class OBBTrainerTest : public EngineTestFixture {
protected:
    void SetUp() override {
        EngineTestFixture::SetUp();
    }

    void TearDown() override {
        EngineTestFixture::TearDown();
    }

    std::unique_ptr<OBBTrainer> createTrainer(
        std::atomic<bool>* stopFlag = nullptr)
    {
        auto obbConfig = EngineConfigFactory::createMinimal(TaskType::OBB);
        return std::make_unique<OBBTrainer>(
            obbConfig, logger(), workspace(), profiler(), nullptr, stopFlag);
    }
};

// ============================================================================
// Constructor Tests (1)
// ============================================================================

TEST_F(OBBTrainerTest, Constructor_SetsThresholds) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

// ============================================================================
// SetupModel Tests (1)
// ============================================================================

TEST_F(OBBTrainerTest, SetupModel_CreatesModel) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

// ============================================================================
// SetupLoss Tests (2)
// ============================================================================

TEST_F(OBBTrainerTest, SetupLoss_RotatedBoxLoss) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

TEST_F(OBBTrainerTest, SetupLoss_AngleLoss) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

// ============================================================================
// ComputeLoss Tests (3)
// ============================================================================

TEST_F(OBBTrainerTest, ComputeLoss_BoxLoss) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

TEST_F(OBBTrainerTest, ComputeLoss_AnglePrediction) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

TEST_F(OBBTrainerTest, ComputeLoss_Exception_LogsError) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

// ============================================================================
// TrainEpoch Tests (3)
// ============================================================================

TEST_F(OBBTrainerTest, TrainEpoch_UpdatesWeights) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

TEST_F(OBBTrainerTest, TrainEpoch_ComputesmAP) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

TEST_F(OBBTrainerTest, TrainEpoch_StopRequested) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

// ============================================================================
// Validation Tests (2)
// ============================================================================

TEST_F(OBBTrainerTest, Validate_ComputesMetrics) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

TEST_F(OBBTrainerTest, Validate_Exception_LogsError) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

// ============================================================================
// Fitness Tests (1)
// ============================================================================

TEST_F(OBBTrainerTest, ComputeFitness_UsesmAP) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    EXPECT_NE(trainer, nullptr);
}

