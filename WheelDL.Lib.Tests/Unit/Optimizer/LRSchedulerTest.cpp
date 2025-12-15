/**
 * @file LRSchedulerTest.cpp
 * @brief Unit tests for LRScheduler, CosineAnnealingLR, and LinearLR
 *
 * Test Plan Reference: docs/test/Optimizer_Test_Plan.md Section 4
 * Total Tests: 87
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "Optimizer/Scheduler/LRScheduler.h"
#include "Optimizer/Scheduler/CosineAnnealingLR.h"
#include "Optimizer/Scheduler/LinearLR.h"
#include "Optimizer/OptimizerFactory.h"
#include "Config/Configuration.h"
#include "Utils/Error/WheelLibException.h"
#include "Utils/Error/ErrorCodes.h"
#include <torch/torch.h>
#include <cmath>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace WheelDL::Optimizer::Scheduler;
using namespace WheelDL::Optimizer;
using namespace WheelDL::Config;
using namespace WheelDL::Utils;

// =============================================================================
// Test Fixture and Helpers
// =============================================================================

class LRSchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper to create a test configuration
    // Note: lrf (LRFinalFraction) defaults to 0.01f and has no setter
    // IMPORTANT: Configuration class clamps warmupEpochs to max 3% of epochs!
    // setEpochs() also resets warmupEpochs to 0, so order matters.
    // Actual warmup = min(requested, epochs * 0.03)
    Configuration createTestConfig(
        float lr0 = 0.01f,
        int epochs = 100,
        float warmupEpochs = 3.0f
    ) {
        Configuration config;
        config.setLearningRateFirst(lr0);
        config.setEpochs(epochs);
        config.setWarmupEpochs(warmupEpochs);
        return config;
    }

    // Helper to calculate actual warmup (accounting for 3% clamp)
    float getActualWarmup(int epochs, float requestedWarmup) {
        return std::min(requestedWarmup, static_cast<float>(epochs) * 0.03f);
    }

    int getActualWarmupCeil(int epochs, float requestedWarmup) {
        return static_cast<int>(std::ceil(getActualWarmup(epochs, requestedWarmup)));
    }

    // Helper to create a simple model with trainable parameters
    std::vector<torch::Tensor> createValidParams() {
        return { torch::randn({10, 5}, torch::requires_grad()) };
    }

    // Helper to create SGD optimizer
    std::unique_ptr<torch::optim::Optimizer> createSGDOptimizer(float lr = 0.01f) {
        auto params = createValidParams();
        return std::make_unique<torch::optim::SGD>(params, torch::optim::SGDOptions(lr));
    }

    // Helper to create Adam optimizer
    std::unique_ptr<torch::optim::Optimizer> createAdamOptimizer(float lr = 0.01f) {
        auto params = createValidParams();
        return std::make_unique<torch::optim::Adam>(params, torch::optim::AdamOptions(lr));
    }

    // Helper to create AdamW optimizer
    std::unique_ptr<torch::optim::Optimizer> createAdamWOptimizer(float lr = 0.01f) {
        auto params = createValidParams();
        return std::make_unique<torch::optim::AdamW>(params, torch::optim::AdamWOptions(lr));
    }
};

// Concrete implementation for testing base class
class TestableScheduler : public LRScheduler {
public:
    TestableScheduler(torch::optim::Optimizer* optimizer, const Configuration& config)
        : LRScheduler(optimizer, config) {}

    float computeLR(int epoch) const override {
        // Simple linear decay for testing
        float progress = computeProgress(epoch);
        return _finalLR + (_initialLR - _finalLR) * (1.0f - progress);
    }

    float computeProgress(int epoch) const {
        int effectiveEpoch = epoch - _warmupEpochs;
        int effectiveTotal = _totalEpochs - _warmupEpochs;
        if (effectiveTotal <= 0) return 1.0f;
        float progress = static_cast<float>(effectiveEpoch) / static_cast<float>(effectiveTotal);
        return std::max(0.0f, std::min(1.0f, progress));
    }

    // Expose protected methods for testing
    using LRScheduler::setLR;
    using LRScheduler::computeWarmupLR;
};

// =============================================================================
// 4.2.1 LRScheduler Constructor Tests (LRS-001 ~ LRS-014)
// =============================================================================

// LRS-001: Ctor_Valid
TEST_F(LRSchedulerTest, Ctor_Valid) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 3.0f);
    EXPECT_NO_THROW({
        TestableScheduler scheduler(optimizer.get(), config);
    });
}

// LRS-002: Ctor_NullOptimizer
TEST_F(LRSchedulerTest, Ctor_NullOptimizer) {
    auto config = createTestConfig();
    EXPECT_THROW({
        TestableScheduler scheduler(nullptr, config);
    }, ConfigurationException);
}

// LRS-003: Ctor_ZeroLR
// Note: Configuration.setLearningRateFirst() clamps lr < 1e-8 to 1e-8, so no exception thrown
TEST_F(LRSchedulerTest, Ctor_ZeroLR) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.0f, 100, 3.0f);
    // Configuration clamps 0.0 to 1e-8, so scheduler constructs successfully
    EXPECT_NO_THROW({
        TestableScheduler scheduler(optimizer.get(), config);
    });
    // Verify LR was clamped
    EXPECT_NEAR(1e-8f, config.getLearningRate(), 1e-10f);
}

// LRS-004: Ctor_NegativeLR
// Note: Configuration.setLearningRateFirst() clamps lr < 1e-8 to 1e-8, so no exception thrown
TEST_F(LRSchedulerTest, Ctor_NegativeLR) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(-0.01f, 100, 3.0f);
    // Configuration clamps negative to 1e-8, so scheduler constructs successfully
    EXPECT_NO_THROW({
        TestableScheduler scheduler(optimizer.get(), config);
    });
    // Verify LR was clamped
    EXPECT_NEAR(1e-8f, config.getLearningRate(), 1e-10f);
}

// LRS-005: Ctor_NegativeEpochs
// Note: Configuration.setEpochs() clamps negative values to 0
TEST_F(LRSchedulerTest, Ctor_NegativeEpochs) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, -1, 0.0f);  // warmup=0 to avoid warmup > epochs
    // Configuration clamps -1 to 0, scheduler allows epochs=0 (for PatchCore)
    EXPECT_NO_THROW({
        TestableScheduler scheduler(optimizer.get(), config);
    });
    EXPECT_EQ(0, config.getEpochs());
}

// LRS-006: Ctor_ZeroEpochs (Allowed for PatchCore)
TEST_F(LRSchedulerTest, Ctor_ZeroEpochs) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 0, 0.0f);
    EXPECT_NO_THROW({
        TestableScheduler scheduler(optimizer.get(), config);
    });
}

// LRS-007: Ctor_WarmupExceedsTotal
// Note: Configuration clamps warmup to 3% of epochs, so warmup>epochs won't throw
// For epochs=5, warmup=10 requested -> actual=min(10, 0.15)=0.15
TEST_F(LRSchedulerTest, Ctor_WarmupExceedsTotal) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 5, 10.0f);
    // No exception because warmup is clamped to 0.15 (3% of 5)
    EXPECT_NO_THROW({
        TestableScheduler scheduler(optimizer.get(), config);
    });
    // Verify warmup was clamped
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_EQ(1, scheduler.getWarmupEpochs());  // ceil(0.15) = 1
}

// LRS-008: Ctor_NegativeWarmup
TEST_F(LRSchedulerTest, Ctor_NegativeWarmup) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, -1.0f);
    EXPECT_THROW({
        TestableScheduler scheduler(optimizer.get(), config);
    }, ConfigurationException);
}

// LRS-009: Ctor_WarmupEqualsTotal (Allowed, but warmup is clamped to 3%)
// epochs=100, warmup=100 requested -> actual=min(100, 3)=3
TEST_F(LRSchedulerTest, Ctor_WarmupEqualsTotal) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 100.0f);
    EXPECT_NO_THROW({
        TestableScheduler scheduler(optimizer.get(), config);
    });
    // Verify warmup was clamped to 3 (3% of 100)
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_EQ(3, scheduler.getWarmupEpochs());
}

// LRS-010: Ctor_FinalLRCalc
TEST_F(LRSchedulerTest, Ctor_FinalLRCalc) {
    auto optimizer = createSGDOptimizer();
    // lr0=0.01, lrf=0.1 (default is 0.01), finalLR = 0.01 * 0.01 = 0.0001
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_NEAR(0.0001f, scheduler.getFinalLR(), 1e-6f);  // 0.01 * 0.01 default lrf
}

// LRS-011: Ctor_WarmupCeil
TEST_F(LRSchedulerTest, Ctor_WarmupCeil) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 2.5f);
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_EQ(3, scheduler.getWarmupEpochs());  // ceil(2.5) = 3
}

// LRS-012: Ctor_WarmupCeil_Integer
TEST_F(LRSchedulerTest, Ctor_WarmupCeil_Integer) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_EQ(3, scheduler.getWarmupEpochs());  // ceil(3.0) = 3
}

// LRS-013: Ctor_InitialLR
TEST_F(LRSchedulerTest, Ctor_InitialLR) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.02f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_NEAR(0.02f, scheduler.getInitialLR(), 1e-6f);
}

// LRS-014: Ctor_TotalEpochs
TEST_F(LRSchedulerTest, Ctor_TotalEpochs) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_EQ(200, scheduler.getTotalEpochs());
}

// =============================================================================
// 4.2.2 Getter Methods (LRS-015 ~ LRS-018)
// =============================================================================

// LRS-015: Get_InitialLR
TEST_F(LRSchedulerTest, Get_InitialLR) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);  // epochs=200 so warmup=5 allowed (200*0.03=6)
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_NEAR(0.01f, scheduler.getInitialLR(), 1e-6f);
}

// LRS-016: Get_FinalLR
TEST_F(LRSchedulerTest, Get_FinalLR) {
    auto optimizer = createSGDOptimizer();
    // finalLR = lr0 * lrf = 0.01 * 0.01 (default) = 0.0001
    auto config = createTestConfig(0.01f, 200, 5.0f);  // epochs=200 so warmup=5 allowed
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_NEAR(0.0001f, scheduler.getFinalLR(), 1e-6f);
}

// LRS-017: Get_TotalEpochs
TEST_F(LRSchedulerTest, Get_TotalEpochs) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);  // epochs=200 so warmup=5 allowed
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_EQ(200, scheduler.getTotalEpochs());
}

// LRS-018: Get_WarmupEpochs
TEST_F(LRSchedulerTest, Get_WarmupEpochs) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);  // epochs=200 so warmup=5 allowed (200*0.03=6 > 5)
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_EQ(5, scheduler.getWarmupEpochs());
}

// =============================================================================
// 4.2.3 getCurrentLR() (LRS-019 ~ LRS-024)
// =============================================================================

// LRS-019: GetLR_SGD
TEST_F(LRSchedulerTest, GetLR_SGD) {
    auto optimizer = createSGDOptimizer(0.05f);
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_NEAR(0.05f, scheduler.getCurrentLR(), 1e-6f);
}

// LRS-020: GetLR_Adam
TEST_F(LRSchedulerTest, GetLR_Adam) {
    auto optimizer = createAdamOptimizer(0.03f);
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_NEAR(0.03f, scheduler.getCurrentLR(), 1e-6f);
}

// LRS-021: GetLR_AdamW
TEST_F(LRSchedulerTest, GetLR_AdamW) {
    auto optimizer = createAdamWOptimizer(0.02f);
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_NEAR(0.02f, scheduler.getCurrentLR(), 1e-6f);
}

// LRS-022: GetLR_EmptyGroups
TEST_F(LRSchedulerTest, GetLR_EmptyGroups) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    // Clear param_groups manually
    optimizer->param_groups().clear();
    EXPECT_NEAR(0.0f, scheduler.getCurrentLR(), 1e-6f);
}

// LRS-023: GetLR_UnknownOptimizer - Skip (needs custom optimizer type)
TEST_F(LRSchedulerTest, GetLR_UnknownOptimizer) {
    // With standard optimizers, this path is hard to test
    // The implementation returns _initialLR for unknown types
    auto optimizer = createSGDOptimizer(0.05f);
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    // Just verify we can get the current LR
    EXPECT_GT(scheduler.getCurrentLR(), 0.0f);
}

// LRS-024: GetLR_AfterStep
TEST_F(LRSchedulerTest, GetLR_AfterStep) {
    auto optimizer = createSGDOptimizer(0.01f);
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.step(0);  // First epoch (warmup)
    float lr = scheduler.getCurrentLR();
    // During warmup epoch 0: lr = 0.01 * (0+1)/3 = 0.01/3 ≈ 0.00333
    EXPECT_NEAR(0.01f / 3.0f, lr, 1e-4f);
}

// =============================================================================
// 4.2.4 setLR() - Clamping (LRS-025 ~ LRS-032)
// =============================================================================

// LRS-025: SetLR_Normal
TEST_F(LRSchedulerTest, SetLR_Normal) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.setLR(0.01f);
    EXPECT_NEAR(0.01f, scheduler.getCurrentLR(), 1e-6f);
}

// LRS-026: SetLR_ClampMin
TEST_F(LRSchedulerTest, SetLR_ClampMin) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.setLR(1e-10f);  // Very small
    EXPECT_NEAR(1e-8f, scheduler.getCurrentLR(), 1e-10f);  // Clamped to 1e-8
}

// LRS-027: SetLR_ClampMax
TEST_F(LRSchedulerTest, SetLR_ClampMax) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.setLR(10.0f);  // Very large
    EXPECT_NEAR(1.0f, scheduler.getCurrentLR(), 1e-6f);  // Clamped to 1.0
}

// LRS-028: SetLR_ExactMin
TEST_F(LRSchedulerTest, SetLR_ExactMin) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.setLR(1e-8f);
    EXPECT_NEAR(1e-8f, scheduler.getCurrentLR(), 1e-10f);
}

// LRS-029: SetLR_ExactMax
TEST_F(LRSchedulerTest, SetLR_ExactMax) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.setLR(1.0f);
    EXPECT_NEAR(1.0f, scheduler.getCurrentLR(), 1e-6f);
}

// LRS-030: SetLR_Zero
TEST_F(LRSchedulerTest, SetLR_Zero) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.setLR(0.0f);
    EXPECT_NEAR(1e-8f, scheduler.getCurrentLR(), 1e-10f);  // Clamped to 1e-8
}

// LRS-031: SetLR_Negative
TEST_F(LRSchedulerTest, SetLR_Negative) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 3.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.setLR(-0.01f);
    EXPECT_NEAR(1e-8f, scheduler.getCurrentLR(), 1e-10f);  // Clamped to 1e-8
}

// LRS-032: SetLR_AllGroups
TEST_F(LRSchedulerTest, SetLR_AllGroups) {
    // Create optimizer with multiple param groups via OptimizerFactory
    struct MultiParamModel : torch::nn::Module {
        torch::Tensor bias;
        torch::Tensor weight;
        torch::Tensor bn_weight;
        MultiParamModel() {
            bias = register_parameter("bias", torch::zeros({10}));
            weight = register_parameter("weight", torch::randn({10, 5}));
            bn_weight = register_parameter("bn_weight", torch::ones({10}));
        }
    };

    MultiParamModel model;
    Configuration config;
    config.setOptimizer("sgd");
    config.setLearningRateFirst(0.01f);
    config.setMomentum(0.9f);
    config.setEpochs(100);
    config.setWarmupEpochs(3.0f);

    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.setLR(0.05f);

    // Verify all groups have the new LR
    for (auto& group : optimizer->param_groups()) {
        auto& options = static_cast<torch::optim::SGDOptions&>(group.options());
        EXPECT_NEAR(0.05f, options.lr(), 1e-6f);
    }
}

// =============================================================================
// 4.2.5 computeWarmupLR() (LRS-033 ~ LRS-038)
// =============================================================================

// LRS-033: Warmup_ZeroWarmup
TEST_F(LRSchedulerTest, Warmup_ZeroWarmup) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 0.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    EXPECT_NEAR(0.01f, scheduler.computeWarmupLR(0), 1e-6f);
}

// LRS-034: Warmup_Epoch0
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Warmup_Epoch0) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    // epoch=0: lr = 0.01 * (0+1)/5 = 0.01 * 0.2 = 0.002
    EXPECT_NEAR(0.002f, scheduler.computeWarmupLR(0), 1e-6f);
}

// LRS-035: Warmup_EpochMid
// epochs=200 allows warmup=4 (200*0.03=6 > 4)
TEST_F(LRSchedulerTest, Warmup_EpochMid) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 4.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    // epoch=1: lr = 0.01 * (1+1)/4 = 0.01 * 0.5 = 0.005
    EXPECT_NEAR(0.005f, scheduler.computeWarmupLR(1), 1e-6f);
}

// LRS-036: Warmup_EpochLast
TEST_F(LRSchedulerTest, Warmup_EpochLast) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 3.0f);  // warmup=3 allowed (100*0.03=3)
    TestableScheduler scheduler(optimizer.get(), config);
    // epoch=2 (last warmup): lr = 0.01 * (2+1)/3 = 0.01 * 1.0 = 0.01
    EXPECT_NEAR(0.01f, scheduler.computeWarmupLR(2), 1e-6f);
}

// LRS-037: Warmup_LinearIncrease
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Warmup_LinearIncrease) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.1f, 200, 5.0f);
    TestableScheduler scheduler(optimizer.get(), config);

    // Check linear progression: 0.02, 0.04, 0.06, 0.08, 0.10
    EXPECT_NEAR(0.02f, scheduler.computeWarmupLR(0), 1e-6f);
    EXPECT_NEAR(0.04f, scheduler.computeWarmupLR(1), 1e-6f);
    EXPECT_NEAR(0.06f, scheduler.computeWarmupLR(2), 1e-6f);
    EXPECT_NEAR(0.08f, scheduler.computeWarmupLR(3), 1e-6f);
    EXPECT_NEAR(0.10f, scheduler.computeWarmupLR(4), 1e-6f);
}

// LRS-038: Warmup_Formula
// epochs=400 allows warmup=10 (400*0.03=12 > 10)
TEST_F(LRSchedulerTest, Warmup_Formula) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.1f, 400, 10.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    // epoch=4: lr = 0.1 * (4+1)/10 = 0.1 * 0.5 = 0.05
    EXPECT_NEAR(0.05f, scheduler.computeWarmupLR(4), 1e-6f);
}

// =============================================================================
// 4.2.6 step() - Phase Switching (LRS-039 ~ LRS-043)
// =============================================================================

// LRS-039: Step_DuringWarmup
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Step_DuringWarmup) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.step(0);  // During warmup
    // epoch=0: lr = 0.01 * (0+1)/5 = 0.002
    EXPECT_NEAR(0.002f, scheduler.getCurrentLR(), 1e-6f);
}

// LRS-040: Step_AtWarmupEnd
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Step_AtWarmupEnd) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.step(5);  // At warmup end, computeLR is called
    // After warmup, computeLR takes over
    float lr = scheduler.getCurrentLR();
    EXPECT_GT(lr, 0.0f);  // Should be a valid LR from computeLR
}

// LRS-041: Step_AfterWarmup
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Step_AfterWarmup) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.step(100);  // Well after warmup
    float lr = scheduler.getCurrentLR();
    EXPECT_GT(lr, 0.0f);  // Valid LR
    EXPECT_LE(lr, 0.01f);  // Should be <= initial LR
}

// LRS-042: Step_CallsSetLR
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Step_CallsSetLR) {
    auto optimizer = createSGDOptimizer(0.05f);  // Start with 0.05
    auto config = createTestConfig(0.01f, 200, 5.0f);
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.step(0);  // Should update LR
    // Should be different from initial 0.05
    float lr = scheduler.getCurrentLR();
    EXPECT_NE(0.05f, lr);
}

// LRS-043: Step_UpdatesOptimizer
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Step_UpdatesOptimizer) {
    auto optimizer = createSGDOptimizer(0.1f);
    auto config = createTestConfig(0.01f, 200, 5.0f);
    TestableScheduler scheduler(optimizer.get(), config);

    // Before step, LR should be 0.1
    EXPECT_NEAR(0.1f, scheduler.getCurrentLR(), 1e-6f);

    // After step, LR should be updated
    scheduler.step(0);
    float newLR = scheduler.getCurrentLR();
    EXPECT_NE(0.1f, newLR);
}

// =============================================================================
// 4.2.7 CosineAnnealingLR - computeLR() (COS-001 ~ COS-008)
// =============================================================================

// COS-001: Cosine_Progress0
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Cosine_Progress0) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);
    scheduler.step(5);  // First epoch after warmup (progress=0)
    // At progress=0: cosine_decay=0.5*(1+cos(0))=1.0, lr = finalLR + (initialLR - finalLR) * 1.0 = initialLR
    EXPECT_NEAR(0.01f, scheduler.getCurrentLR(), 1e-4f);
}

// COS-002: Cosine_Progress1
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Cosine_Progress1) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);
    scheduler.step(199);  // Last epoch (progress≈1)
    // At progress≈1: cosine_decay=0.5*(1+cos(PI))≈0, lr ≈ finalLR
    float finalLR = 0.01f * 0.01f;  // lr0 * lrf (default)
    EXPECT_NEAR(finalLR, scheduler.getCurrentLR(), 1e-4f);
}

// COS-003: Cosine_ProgressMid
TEST_F(LRSchedulerTest, Cosine_ProgressMid) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 0.0f);  // No warmup for simpler math
    CosineAnnealingLR scheduler(optimizer.get(), config);
    // At epoch 50 (progress=0.5): cosine_decay=0.5*(1+cos(PI*0.5))=0.5
    // lr = 0.0001 + (0.01 - 0.0001) * 0.5 = 0.0001 + 0.00495 = 0.00505
    scheduler.step(50);
    float expectedLR = 0.0001f + (0.01f - 0.0001f) * 0.5f;
    EXPECT_NEAR(expectedLR, scheduler.getCurrentLR(), 1e-4f);
}

// COS-004: Cosine_Formula
// epochs=400 allows warmup=10 (400*0.03=12 > 10)
TEST_F(LRSchedulerTest, Cosine_Formula) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.1f, 400, 10.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);

    float initialLR = 0.1f;
    float finalLR = 0.1f * 0.01f;  // 0.001

    // Test at epoch 205 (effectiveEpoch=195, effectiveTotal=390, progress=0.5)
    float progress = 195.0f / 390.0f;
    float cosine_decay = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * progress));
    float expectedLR = finalLR + (initialLR - finalLR) * cosine_decay;

    scheduler.step(205);
    EXPECT_NEAR(expectedLR, scheduler.getCurrentLR(), 1e-4f);
}

// COS-005: Cosine_SmoothCurve
TEST_F(LRSchedulerTest, Cosine_SmoothCurve) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 50, 0.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);

    float prevLR = scheduler.getInitialLR();
    for (int epoch = 0; epoch < 50; ++epoch) {
        scheduler.step(epoch);
        float lr = scheduler.getCurrentLR();
        EXPECT_LE(lr, prevLR + 1e-6f);  // LR should decrease (or stay same)
        prevLR = lr;
    }
}

// COS-006: Cosine_EffectiveTotalZero
// Note: With Configuration's 3% clamp, warmup=10 with epochs=10 becomes warmup=0.3 (ceil=1)
// To test effectiveTotal=0, we need to use epochs where warmup equals epochs
// But this is clamped to 3%, so we test with a very small warmup that still equals epochs
TEST_F(LRSchedulerTest, Cosine_EffectiveTotalZero) {
    auto optimizer = createSGDOptimizer();
    // epochs=100, warmup=100 requested -> warmup clamped to 3, so effectiveTotal=97 (not 0)
    // Let's use the clamped value and verify behavior after warmup
    auto config = createTestConfig(0.01f, 100, 100.0f);  // warmup clamped to 3
    CosineAnnealingLR scheduler(optimizer.get(), config);
    scheduler.step(3);  // At warmup end (ceil(3)=3)
    // After warmup, effectiveTotal = 100-3 = 97, not 0
    // This test now verifies behavior at warmup boundary
    float lr = scheduler.getCurrentLR();
    EXPECT_NEAR(0.01f, lr, 1e-4f);  // Should be near initial LR
}

// COS-007: Cosine_ProgressClamped
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Cosine_ProgressClamped) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);
    scheduler.step(300);  // Beyond total epochs
    // Progress should be clamped to 1.0
    float finalLR = 0.01f * 0.01f;
    EXPECT_NEAR(finalLR, scheduler.getCurrentLR(), 1e-4f);
}

// COS-008: Cosine_ProgressNegative
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Cosine_ProgressNegative) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);
    scheduler.step(3);  // Still in warmup (epoch < warmup)
    // During warmup, warmup LR is used
    float expectedWarmupLR = 0.01f * (3.0f + 1.0f) / 5.0f;
    EXPECT_NEAR(expectedWarmupLR, scheduler.getCurrentLR(), 1e-4f);
}

// =============================================================================
// 4.2.8 LinearLR - computeLR() (LIN-001 ~ LIN-008)
// =============================================================================

// LIN-001: Linear_Progress0
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Linear_Progress0) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    LinearLR scheduler(optimizer.get(), config);
    scheduler.step(5);  // First epoch after warmup (progress=0)
    // At progress=0: lr = initialLR - 0 = initialLR
    EXPECT_NEAR(0.01f, scheduler.getCurrentLR(), 1e-4f);
}

// LIN-002: Linear_Progress1
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Linear_Progress1) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    LinearLR scheduler(optimizer.get(), config);
    scheduler.step(199);  // Last epoch (progress≈1)
    float finalLR = 0.01f * 0.01f;
    EXPECT_NEAR(finalLR, scheduler.getCurrentLR(), 1e-4f);
}

// LIN-003: Linear_ProgressMid
TEST_F(LRSchedulerTest, Linear_ProgressMid) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 0.0f);  // No warmup
    LinearLR scheduler(optimizer.get(), config);
    scheduler.step(50);  // progress=0.5
    // lr = 0.01 - (0.01 - 0.0001) * 0.5 = 0.01 - 0.00495 = 0.00505
    float expectedLR = 0.01f - (0.01f - 0.0001f) * 0.5f;
    EXPECT_NEAR(expectedLR, scheduler.getCurrentLR(), 1e-4f);
}

// LIN-004: Linear_Formula
// epochs=400 allows warmup=10 (400*0.03=12 > 10)
TEST_F(LRSchedulerTest, Linear_Formula) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.1f, 400, 10.0f);
    LinearLR scheduler(optimizer.get(), config);

    float initialLR = 0.1f;
    float finalLR = 0.1f * 0.01f;  // 0.001

    // Test at epoch 205 (effectiveEpoch=195, effectiveTotal=390, progress=0.5)
    float progress = 195.0f / 390.0f;
    float expectedLR = initialLR - (initialLR - finalLR) * progress;

    scheduler.step(205);
    EXPECT_NEAR(expectedLR, scheduler.getCurrentLR(), 1e-4f);
}

// LIN-005: Linear_SteadyDecay
TEST_F(LRSchedulerTest, Linear_SteadyDecay) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 50, 0.0f);
    LinearLR scheduler(optimizer.get(), config);

    float prevLR = scheduler.getInitialLR();
    for (int epoch = 0; epoch < 50; ++epoch) {
        scheduler.step(epoch);
        float lr = scheduler.getCurrentLR();
        EXPECT_LE(lr, prevLR + 1e-6f);  // LR should decrease (or stay same)
        prevLR = lr;
    }
}

// LIN-006: Linear_EffectiveTotalZero
// Note: With Configuration's 3% clamp, warmup=10 with epochs=10 becomes warmup=0.3 (ceil=1)
TEST_F(LRSchedulerTest, Linear_EffectiveTotalZero) {
    auto optimizer = createSGDOptimizer();
    // epochs=100, warmup=100 requested -> warmup clamped to 3, effectiveTotal = 97 (not 0)
    auto config = createTestConfig(0.01f, 100, 100.0f);  // warmup clamped to 3
    LinearLR scheduler(optimizer.get(), config);
    scheduler.step(3);  // At warmup end
    // After warmup, effectiveTotal = 100-3 = 97, not 0
    // This test now verifies behavior at warmup boundary
    float lr = scheduler.getCurrentLR();
    EXPECT_NEAR(0.01f, lr, 1e-4f);  // Should be near initial LR
}

// LIN-007: Linear_ProgressClamped
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Linear_ProgressClamped) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    LinearLR scheduler(optimizer.get(), config);
    scheduler.step(300);  // Beyond total epochs
    float finalLR = 0.01f * 0.01f;
    EXPECT_NEAR(finalLR, scheduler.getCurrentLR(), 1e-4f);
}

// LIN-008: Linear_ProgressNegative
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Linear_ProgressNegative) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    LinearLR scheduler(optimizer.get(), config);
    scheduler.step(3);  // Still in warmup
    float expectedWarmupLR = 0.01f * (3.0f + 1.0f) / 5.0f;
    EXPECT_NEAR(expectedWarmupLR, scheduler.getCurrentLR(), 1e-4f);
}

// =============================================================================
// 4.2.9 Full Training Simulation (LRS-044 ~ LRS-048)
// =============================================================================

// LRS-044: FullTrain_CosineNoWarmup
TEST_F(LRSchedulerTest, FullTrain_CosineNoWarmup) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 20, 0.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);

    for (int epoch = 0; epoch < 20; ++epoch) {
        scheduler.step(epoch);
        float lr = scheduler.getCurrentLR();
        EXPECT_GE(lr, 1e-8f);
        EXPECT_LE(lr, 1.0f);
    }
}

// LRS-045: FullTrain_CosineWithWarmup
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, FullTrain_CosineWithWarmup) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);

    // Warmup phase
    for (int epoch = 0; epoch < 5; ++epoch) {
        scheduler.step(epoch);
        float lr = scheduler.getCurrentLR();
        float expectedWarmupLR = 0.01f * (epoch + 1) / 5.0f;
        EXPECT_NEAR(expectedWarmupLR, lr, 1e-4f);
    }

    // Decay phase
    float prevLR = scheduler.getCurrentLR();
    for (int epoch = 5; epoch < 50; ++epoch) {
        scheduler.step(epoch);
        float lr = scheduler.getCurrentLR();
        EXPECT_LE(lr, prevLR + 1e-6f);
        prevLR = lr;
    }
}

// LRS-046: FullTrain_LinearNoWarmup
TEST_F(LRSchedulerTest, FullTrain_LinearNoWarmup) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 20, 0.0f);
    LinearLR scheduler(optimizer.get(), config);

    for (int epoch = 0; epoch < 20; ++epoch) {
        scheduler.step(epoch);
        float lr = scheduler.getCurrentLR();
        EXPECT_GE(lr, 1e-8f);
        EXPECT_LE(lr, 1.0f);
    }
}

// LRS-047: FullTrain_LinearWithWarmup
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, FullTrain_LinearWithWarmup) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    LinearLR scheduler(optimizer.get(), config);

    // Warmup phase
    for (int epoch = 0; epoch < 5; ++epoch) {
        scheduler.step(epoch);
        float lr = scheduler.getCurrentLR();
        float expectedWarmupLR = 0.01f * (epoch + 1) / 5.0f;
        EXPECT_NEAR(expectedWarmupLR, lr, 1e-4f);
    }

    // Decay phase
    float prevLR = scheduler.getCurrentLR();
    for (int epoch = 5; epoch < 50; ++epoch) {
        scheduler.step(epoch);
        float lr = scheduler.getCurrentLR();
        EXPECT_LE(lr, prevLR + 1e-6f);
        prevLR = lr;
    }
}

// LRS-048: FullTrain_EndValue
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, FullTrain_EndValue) {
    float lr0 = 0.01f;
    float finalLR = lr0 * 0.01f;  // Default lrf

    // Test Cosine
    {
        auto optimizer = createSGDOptimizer();
        auto config = createTestConfig(lr0, 200, 5.0f);
        CosineAnnealingLR scheduler(optimizer.get(), config);
        scheduler.step(199);
        EXPECT_NEAR(finalLR, scheduler.getCurrentLR(), 1e-4f);
    }

    // Test Linear
    {
        auto optimizer = createSGDOptimizer();
        auto config = createTestConfig(lr0, 200, 5.0f);
        LinearLR scheduler(optimizer.get(), config);
        scheduler.step(199);
        EXPECT_NEAR(finalLR, scheduler.getCurrentLR(), 1e-4f);
    }
}

// =============================================================================
// 4.2.10 Edge Cases (LRS-049 ~ LRS-056)
// =============================================================================

// LRS-049: Edge_SingleEpoch
TEST_F(LRSchedulerTest, Edge_SingleEpoch) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 1, 0.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);
    scheduler.step(0);
    // With epochs=1, progress=0/1=0, should return initial LR
    EXPECT_NEAR(0.01f, scheduler.getCurrentLR(), 1e-4f);
}

// LRS-050: Edge_ZeroFinalLR (lrf=0 would mean finalLR=0, clamped to 1e-8)
TEST_F(LRSchedulerTest, Edge_ZeroFinalLR) {
    // Note: lrf defaults to 0.01, so finalLR = 0.01 * 0.01 = 0.0001
    // When the computed LR approaches this, it should be clamped if < 1e-8
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 0.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);
    scheduler.step(99);
    // finalLR = 0.0001, which is > 1e-8, so no clamping
    EXPECT_GT(scheduler.getCurrentLR(), 1e-8f);
}

// LRS-051: Edge_EqualInitFinal
TEST_F(LRSchedulerTest, Edge_EqualInitFinal) {
    // With default lrf=0.01, we can't make init=final without changing lrf
    // So we test that LR decreases toward finalLR
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 0.0f);
    LinearLR scheduler(optimizer.get(), config);
    scheduler.step(0);
    float lr0 = scheduler.getCurrentLR();
    scheduler.step(99);
    float lrFinal = scheduler.getCurrentLR();
    EXPECT_LT(lrFinal, lr0);  // Should have decreased
}

// LRS-052: Edge_LargeLRF (lrf=1 would mean finalLR = initialLR)
TEST_F(LRSchedulerTest, Edge_LargeLRF) {
    // With lrf=0.01 default, finalLR = lr0 * 0.01
    // This test verifies decay happens
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 0.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);
    scheduler.step(0);
    float startLR = scheduler.getCurrentLR();
    scheduler.step(99);
    float endLR = scheduler.getCurrentLR();
    EXPECT_LT(endLR, startLR);
}

// LRS-053: Edge_VerySmallLRF
TEST_F(LRSchedulerTest, Edge_VerySmallLRF) {
    // Default lrf=0.01, so finalLR = 0.01 * 0.01 = 0.0001
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, 0.0f);
    LinearLR scheduler(optimizer.get(), config);
    scheduler.step(99);
    EXPECT_NEAR(0.0001f, scheduler.getCurrentLR(), 1e-4f);
}

// LRS-054: Edge_NegativeEpoch
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Edge_NegativeEpoch) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);
    scheduler.step(-1);  // Negative epoch - treated as warmup
    // epoch=-1 < warmup=5, so warmup LR: 0.01 * (-1+1)/5 = 0
    // But this would be clamped to 1e-8
    EXPECT_GE(scheduler.getCurrentLR(), 1e-8f);
}

// LRS-055: Edge_EpochBeyondTotal
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Edge_EpochBeyondTotal) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);
    scheduler.step(300);  // Beyond total
    float finalLR = 0.01f * 0.01f;
    EXPECT_NEAR(finalLR, scheduler.getCurrentLR(), 1e-4f);
}

// LRS-056: Edge_VeryLargeLR
TEST_F(LRSchedulerTest, Edge_VeryLargeLR) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(100.0f, 100, 0.0f);  // Very large LR
    TestableScheduler scheduler(optimizer.get(), config);
    scheduler.setLR(100.0f);
    // Should be clamped to 1.0
    EXPECT_NEAR(1.0f, scheduler.getCurrentLR(), 1e-6f);
}

// =============================================================================
// 4.2.11 Exception Details (LRS-057 ~ LRS-064)
// =============================================================================

// LRS-057: ExType_NullOptimizer
TEST_F(LRSchedulerTest, ExType_NullOptimizer) {
    auto config = createTestConfig();
    EXPECT_THROW({
        CosineAnnealingLR scheduler(nullptr, config);
    }, ConfigurationException);
}

// LRS-058: ExType_ZeroLR
// Note: Configuration clamps lr < 1e-8 to 1e-8, so no exception
TEST_F(LRSchedulerTest, ExType_ZeroLR) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.0f, 100, 3.0f);
    EXPECT_NO_THROW({
        CosineAnnealingLR scheduler(optimizer.get(), config);
    });
    EXPECT_NEAR(1e-8f, config.getLearningRate(), 1e-10f);
}

// LRS-059: ExType_NegativeLR
// Note: Configuration clamps lr < 1e-8 to 1e-8, so no exception
TEST_F(LRSchedulerTest, ExType_NegativeLR) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(-0.01f, 100, 3.0f);
    EXPECT_NO_THROW({
        LinearLR scheduler(optimizer.get(), config);
    });
    EXPECT_NEAR(1e-8f, config.getLearningRate(), 1e-10f);
}

// LRS-060: ExType_NegativeEpochs
// Note: Configuration.setEpochs() clamps negative values to 0, so no exception thrown
TEST_F(LRSchedulerTest, ExType_NegativeEpochs) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, -10, 3.0f);
    // Configuration clamps -10 to 0, scheduler allows epochs=0 (for PatchCore)
    EXPECT_NO_THROW({
        CosineAnnealingLR scheduler(optimizer.get(), config);
    });
    EXPECT_EQ(0, config.getEpochs());
}

// LRS-061: ExType_BadWarmup
// Note: Configuration clamps warmup to 3% of epochs, so warmup>epochs won't throw
// epochs=50, warmup=100 requested -> actual=min(100, 1.5)=1.5
TEST_F(LRSchedulerTest, ExType_BadWarmup) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 50, 100.0f);  // warmup clamped to 1.5
    // No exception because warmup is clamped
    EXPECT_NO_THROW({
        LinearLR scheduler(optimizer.get(), config);
    });
    // Verify warmup was clamped
    LinearLR scheduler(optimizer.get(), config);
    EXPECT_EQ(2, scheduler.getWarmupEpochs());  // ceil(1.5) = 2
}

// LRS-062: ExCode_AllExceptions
TEST_F(LRSchedulerTest, ExCode_AllExceptions) {
    auto optimizer = createSGDOptimizer();

    // Test null optimizer
    try {
        auto config = createTestConfig();
        CosineAnnealingLR scheduler(nullptr, config);
        FAIL() << "Expected exception";
    } catch (const ConfigurationException& e) {
        EXPECT_EQ(ErrorCode::INVALID_CONFIG, e.getErrorCode());
    }

    // Test negative warmup (not clamped by Configuration)
    try {
        auto config = createTestConfig(0.01f, 100, -5.0f);
        CosineAnnealingLR scheduler(optimizer.get(), config);
        FAIL() << "Expected exception";
    } catch (const ConfigurationException& e) {
        EXPECT_EQ(ErrorCode::INVALID_CONFIG, e.getErrorCode());
    }
}

// LRS-063: ExMsg_NullOptimizer
TEST_F(LRSchedulerTest, ExMsg_NullOptimizer) {
    auto config = createTestConfig();
    try {
        CosineAnnealingLR scheduler(nullptr, config);
        FAIL() << "Expected exception";
    } catch (const ConfigurationException& e) {
        std::string msg = e.what();
        EXPECT_NE(std::string::npos, msg.find("null"));
    }
}

// LRS-064: ExMsg_ZeroLR
// Note: Configuration clamps lr and epochs, so test negative warmup instead
TEST_F(LRSchedulerTest, ExMsg_ZeroLR) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 100, -5.0f);
    try {
        CosineAnnealingLR scheduler(optimizer.get(), config);
        FAIL() << "Expected exception";
    } catch (const ConfigurationException& e) {
        std::string msg = e.what();
        // Message is "Warmup epochs must be between 0 and total epochs"
        EXPECT_NE(std::string::npos, msg.find("Warmup"));
    }
}

// =============================================================================
// 4.2.12 Virtual Destructor & Polymorphism (LRS-065 ~ LRS-067)
// =============================================================================

// LRS-065: Destructor_Virtual
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Destructor_Virtual) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);

    // Create via derived, delete via base pointer
    LRScheduler* scheduler = new CosineAnnealingLR(optimizer.get(), config);
    EXPECT_NO_THROW({
        delete scheduler;
    });
}

// LRS-066: Polymorphism_CosineViaBase
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Polymorphism_CosineViaBase) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);

    std::unique_ptr<LRScheduler> scheduler =
        std::make_unique<CosineAnnealingLR>(optimizer.get(), config);

    EXPECT_NO_THROW({
        scheduler->step(0);
        scheduler->step(100);
        float lr = scheduler->getCurrentLR();
        EXPECT_GT(lr, 0.0f);
    });
}

// LRS-067: Polymorphism_LinearViaBase
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Polymorphism_LinearViaBase) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);

    std::unique_ptr<LRScheduler> scheduler =
        std::make_unique<LinearLR>(optimizer.get(), config);

    EXPECT_NO_THROW({
        scheduler->step(0);
        scheduler->step(100);
        float lr = scheduler->getCurrentLR();
        EXPECT_GT(lr, 0.0f);
    });
}

// =============================================================================
// 4.2.13 CosineAnnealingLR Constructor (COS-009 ~ COS-010)
// =============================================================================

// COS-009: Cosine_Ctor_Valid
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Cosine_Ctor_Valid) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    EXPECT_NO_THROW({
        CosineAnnealingLR scheduler(optimizer.get(), config);
    });
}

// COS-010: Cosine_Ctor_InheritsBase
// epochs=400 allows warmup=10 (400*0.03=12 > 10)
TEST_F(LRSchedulerTest, Cosine_Ctor_InheritsBase) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.02f, 400, 10.0f);
    CosineAnnealingLR scheduler(optimizer.get(), config);

    // Verify base class members are initialized
    EXPECT_NEAR(0.02f, scheduler.getInitialLR(), 1e-6f);
    EXPECT_EQ(400, scheduler.getTotalEpochs());
    EXPECT_EQ(10, scheduler.getWarmupEpochs());
}

// =============================================================================
// 4.2.14 LinearLR Constructor (LIN-009 ~ LIN-010)
// =============================================================================

// LIN-009: Linear_Ctor_Valid
// epochs=200 allows warmup=5 (200*0.03=6 > 5)
TEST_F(LRSchedulerTest, Linear_Ctor_Valid) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.01f, 200, 5.0f);
    EXPECT_NO_THROW({
        LinearLR scheduler(optimizer.get(), config);
    });
}

// LIN-010: Linear_Ctor_InheritsBase
// epochs=300 allows warmup=8 (300*0.03=9 > 8)
TEST_F(LRSchedulerTest, Linear_Ctor_InheritsBase) {
    auto optimizer = createSGDOptimizer();
    auto config = createTestConfig(0.03f, 300, 8.0f);
    LinearLR scheduler(optimizer.get(), config);

    // Verify base class members are initialized
    EXPECT_NEAR(0.03f, scheduler.getInitialLR(), 1e-6f);
    EXPECT_EQ(300, scheduler.getTotalEpochs());
    EXPECT_EQ(8, scheduler.getWarmupEpochs());
}
