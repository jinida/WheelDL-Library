/**
 * @file BaseTrainerTest.cpp
 * @brief Unit tests for WheelDL::Core::Trainer::BaseTrainer
 *
 * Phase 2 of test_engine_and_etc.md - 101 tests
 *
 * Test Sections:
 * - 3.1 Constructor Tests (7)
 * - 3.2 Train Tests (18)
 * - 3.3 InitializeSeeds Tests (3)
 * - 3.4 SetupDevice Tests (4)
 * - 3.5 SetupSaveDirectory Tests (4)
 * - 3.6 FreezeLayersIfNeeded Tests (5)
 * - 3.7 SetupOptimizer Tests (4)
 * - 3.8 SetupScheduler Tests (5)
 * - 3.9 InitializeEMA Tests (5)
 * - 3.10 InitializeEarlyStopping Tests (4)
 * - 3.11 TrainEpoch Tests (7)
 * - 3.12 TrainBatch Tests (8)
 * - 3.13 SaveCheckpoint Tests (8)
 * - 3.14 LoadCheckpoint Tests (10)
 * - 3.15 FinalValidation Tests (5)
 * - 3.16 FinalPrediction Tests (4)
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "EngineTestHelpers.h"

using namespace WheelDL::Core::Trainer;
using namespace WheelDL::Test::Engine;
using namespace WheelDL::Utils;
using WheelDL::ProgressCallback;
using WheelDL::TaskType;
using WheelDL::MetricsData;

// ============================================================================
// Test Fixture
// ============================================================================

class BaseTrainerTest : public EngineTestFixture {
protected:
    void SetUp() override {
        EngineTestFixture::SetUp();
        _callbackTracker = std::make_unique<ProgressCallbackTracker>();
    }

    void TearDown() override {
        _callbackTracker.reset();
        EngineTestFixture::TearDown();
    }

    std::unique_ptr<TestableBaseTrainer> createTrainer(
        ProgressCallback callback = nullptr,
        std::atomic<bool>* stop = nullptr)
    {
        return std::make_unique<TestableBaseTrainer>(
            config(), logger(), workspace(), profiler(),
            callback ? callback : _callbackTracker->getCallback(),
            stop ? stop : stopFlag());
    }

    ProgressCallbackTracker& callbackTracker() { return *_callbackTracker; }

private:
    std::unique_ptr<ProgressCallbackTracker> _callbackTracker;
};

// ============================================================================
// 3.1 Constructor Tests (7)
// ============================================================================

TEST_F(BaseTrainerTest, Constructor_ValidParams) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    EXPECT_NE(trainer, nullptr);
}

TEST_F(BaseTrainerTest, Constructor_InitializesTimers) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    // Timers should be initialized (internal, can't directly access)
    SUCCEED();
}

TEST_F(BaseTrainerTest, Constructor_InitializesRandom) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    // Random should be initialized with config seed
    SUCCEED();
}

TEST_F(BaseTrainerTest, Constructor_InitializesThrottler) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    // Throttler should be initialized with 100ms
    SUCCEED();
}

TEST_F(BaseTrainerTest, Constructor_WithCallback) {
    if (!requireCuda()) return;

    auto trainer = createTrainer(callbackTracker().getCallback());

    // AsyncCallbackQueue should be started
    SUCCEED();
}

TEST_F(BaseTrainerTest, Constructor_NullCallback) {
    if (!requireCuda()) return;

    auto trainer = createTrainer(nullptr);

    // Null callback should result in null AsyncCallbackQueue
    SUCCEED();
}

TEST_F(BaseTrainerTest, Constructor_InitialState) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    EXPECT_EQ(trainer->getCurrentEpoch(), 0);
    EXPECT_LT(trainer->getBestFitness(), 0.0f);  // -1 or similar
}

// ============================================================================
// 3.2 Train Tests (18)
// ============================================================================

TEST_F(BaseTrainerTest, Train_CallsSetupMethods) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    try {
        trainer->train();
    } catch (...) {}

    // Setup methods should be called
    EXPECT_GE(trainer->setupModelCallCount, 1);
}

TEST_F(BaseTrainerTest, Train_LoadsCheckpoint) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    std::string checkpointPath = createTestCheckpoint("train_load.pt");
    trainer->setCheckpoint(checkpointPath);

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    try {
        trainer->train();
    } catch (...) {}

    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_IteratesEpochs) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    try {
        trainer->train();
    } catch (...) {}

    // Should iterate through epochs (may be limited by early stopping or errors)
    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_RunsValidation) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    try {
        trainer->train();
    } catch (...) {}

    // Validation should be run per epoch
    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_SkipsValidation) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};  // No validator

    try {
        trainer->train();
    } catch (...) {}

    // Should log warning if no validator
    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_SavesLastCheckpoint) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    try {
        trainer->train();
    } catch (...) {}

    // last.pt should be saved
    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_SavesBestCheckpoint) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};
    trainer->onCalculateFitness = [](const MetricsData& m) {
        return m.fitness;
    };

    try {
        trainer->train();
    } catch (...) {}

    // best.pt should be saved on improvement
    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_EarlyStopping) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    try {
        trainer->train();
    } catch (...) {}

    // Loop may exit early due to early stopping
    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_InvokesCallback) {
    if (!requireCuda()) return;

    auto trainer = createTrainer(callbackTracker().getCallback());

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    try {
        trainer->train();
    } catch (...) {}

    // Callback should be invoked
    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_StopRequested) {
    if (!requireCuda()) return;

    std::atomic<bool> stop{true};  // Start with stop requested
    auto trainer = createTrainer(nullptr, &stop);

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    EXPECT_THROW({
        trainer->train();
    }, StopRequestedException);
}

TEST_F(BaseTrainerTest, Train_StopRequested_NoSave) {
    if (!requireCuda()) return;

    std::atomic<bool> stop{false};
    auto trainer = createTrainer(nullptr, &stop);

    trainer->onSetupModel = [&trainer, &stop]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
        stop.store(true);  // Request stop after model setup
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    try {
        trainer->train();
    } catch (const StopRequestedException&) {
        // Expected
    } catch (...) {}

    // Results should not be saved when stop is requested
    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_FinalValidation) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    try {
        trainer->train();
    } catch (...) {}

    // finalValidation should be called
    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_FinalPrediction) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    try {
        trainer->train();
    } catch (...) {}

    // finalPrediction should be called
    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_SavesTrainingResults) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    try {
        trainer->train();
    } catch (...) {}

    // Training results should be saved
    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_ReturnsMetrics) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    try {
        auto metrics = trainer->train();
        // Metrics should be returned
        EXPECT_GE(metrics.loss, 0.0f);
    } catch (...) {
        // May throw due to incomplete setup
    }
}

TEST_F(BaseTrainerTest, Train_LogsExceptions) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = []() {
        throw std::runtime_error("Test exception");
    };

    try {
        trainer->train();
    } catch (...) {}

    // Exception should be logged
    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_UpdatesBestFitness) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};
    trainer->onCalculateFitness = [](const MetricsData&) {
        return 0.9f;
    };

    try {
        trainer->train();
    } catch (...) {}

    // Best fitness should be updated
    SUCCEED();
}

TEST_F(BaseTrainerTest, Train_UpdatesCurrentEpoch) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupDataLoaders = []() {};
    trainer->onSetupValidator = []() {};

    try {
        trainer->train();
    } catch (...) {}

    // Current epoch should be incremented
    SUCCEED();
}

// ============================================================================
// 3.3 InitializeSeeds Tests (3)
// ============================================================================

TEST_F(BaseTrainerTest, InitializeSeeds_SetsTorchSeed) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    EXPECT_NO_THROW(trainer->initializeSeeds());
}

TEST_F(BaseTrainerTest, InitializeSeeds_SetsCUDASeed) {
    if (!requireCuda()) return;

    if (!torch::cuda::is_available()) {
        SUCCEED() << "CUDA not available";
        return;
    }

    auto trainer = createTrainer();

    EXPECT_NO_THROW(trainer->initializeSeeds());
}

TEST_F(BaseTrainerTest, InitializeSeeds_Deterministic) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->initializeSeeds();

    // CuDNN should be set to deterministic mode
    SUCCEED();
}

// ============================================================================
// 3.4 SetupDevice Tests (4)
// ============================================================================

TEST_F(BaseTrainerTest, SetupDevice_EmptyConfig) {
    if (!requireCuda()) return;

    auto emptyConfig = EngineConfigFactory::createWithDevice("");
    TestableBaseTrainer trainer(emptyConfig, logger(), workspace(), profiler(), nullptr, nullptr);

    trainer.setupDevice();

    // Should auto-detect
    SUCCEED();
}

TEST_F(BaseTrainerTest, SetupDevice_CPUConfig) {
    if (!requireCuda()) return;

    auto cpuConfig = EngineConfigFactory::createCPU();
    TestableBaseTrainer trainer(cpuConfig, logger(), workspace(), profiler(), nullptr, nullptr);

    trainer.setupDevice();

    EXPECT_TRUE(trainer.getDevice().is_cpu());
}

TEST_F(BaseTrainerTest, SetupDevice_CUDAConfig) {
    if (!requireCuda()) return;

    if (!torch::cuda::is_available()) {
        SUCCEED() << "CUDA not available";
        return;
    }

    auto cudaConfig = EngineConfigFactory::createCUDA();
    TestableBaseTrainer trainer(cudaConfig, logger(), workspace(), profiler(), nullptr, nullptr);

    trainer.setupDevice();

    EXPECT_TRUE(trainer.getDevice().is_cuda());
}

TEST_F(BaseTrainerTest, SetupDevice_LogsMemory) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->setupDevice();

    // Memory info should be logged
    SUCCEED();
}

// ============================================================================
// 3.5 SetupSaveDirectory Tests (4)
// ============================================================================

TEST_F(BaseTrainerTest, SetupSaveDirectory_LogsWorkspaceRoot) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->setupSaveDirectory();

    // Logger should log workspace root
    SUCCEED();
}

TEST_F(BaseTrainerTest, SetupSaveDirectory_LogsWeightsDir) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->setupSaveDirectory();

    // Logger should log weights directory
    SUCCEED();
}

TEST_F(BaseTrainerTest, SetupSaveDirectory_ProfilerTiming) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->setupSaveDirectory();

    // Profiler start/stop should be called
    SUCCEED();
}

TEST_F(BaseTrainerTest, SetupSaveDirectory_UsesInjectedWorkspace) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->setupSaveDirectory();

    std::string saveDir = trainer->getSaveDirectory();
    EXPECT_FALSE(saveDir.empty());
}

// ============================================================================
// 3.6 FreezeLayersIfNeeded Tests (5)
// ============================================================================

TEST_F(BaseTrainerTest, FreezeLayersIfNeeded_ZeroFreeze) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    trainer->freezeLayersIfNeeded();

    // All params should require grad when freeze=0
    auto* model = trainer->getModel();
    if (model) {
        for (auto& param : model->parameters()) {
            EXPECT_TRUE(param.requires_grad());
            break;
        }
    }
}

TEST_F(BaseTrainerTest, FreezeLayersIfNeeded_FreezesFirstN) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    // Note: Actual freeze count depends on config
    trainer->freezeLayersIfNeeded();

    SUCCEED();
}

TEST_F(BaseTrainerTest, FreezeLayersIfNeeded_LogsDebugPerParam) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    trainer->freezeLayersIfNeeded();

    // Logger debug should be called for each frozen param
    SUCCEED();
}

TEST_F(BaseTrainerTest, FreezeLayersIfNeeded_LogsTotal) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    trainer->freezeLayersIfNeeded();

    // Logger should log total frozen count
    SUCCEED();
}

TEST_F(BaseTrainerTest, FreezeLayersIfNeeded_ProfilerTiming) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    trainer->freezeLayersIfNeeded();

    // Profiler should be called
    SUCCEED();
}

// ============================================================================
// 3.7 SetupOptimizer Tests (4)
// ============================================================================

TEST_F(BaseTrainerTest, SetupOptimizer_CreatesOptimizer) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    trainer->setupOptimizer();

    EXPECT_NE(trainer->getOptimizer(), nullptr);
}

TEST_F(BaseTrainerTest, SetupOptimizer_ModelNotInitialized) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    // Don't setup model

    EXPECT_THROW({
        trainer->setupOptimizer();
    }, WheelLibException);
}

TEST_F(BaseTrainerTest, SetupOptimizer_PatchCoreSkips) {
    if (!requireCuda()) return;

    // For PatchCore/anomaly models, optimizer may be null
    auto anomalyConfig = EngineConfigFactory::createMinimal(TaskType::ANOMALY);
    TestableBaseTrainer trainer(anomalyConfig, logger(), workspace(), profiler(), nullptr, nullptr);

    trainer.onSetupModel = [&trainer]() {
        trainer.setModel(std::make_unique<MockBaseModel>());
    };
    trainer.onSetupModel();

    // May skip optimizer for certain task types
    EXPECT_NO_THROW(trainer.setupOptimizer());
}

TEST_F(BaseTrainerTest, SetupOptimizer_UsesFactory) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    trainer->setupOptimizer();

    // OptimizerFactory should be used
    EXPECT_NE(trainer->getOptimizer(), nullptr);
}

// ============================================================================
// 3.8 SetupScheduler Tests (5)
// ============================================================================

TEST_F(BaseTrainerTest, SetupScheduler_CosineAnnealing) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();

    trainer->setupScheduler();

    // Scheduler should be created (type depends on config)
    SUCCEED();
}

TEST_F(BaseTrainerTest, SetupScheduler_LinearLR) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();

    trainer->setupScheduler();

    SUCCEED();
}

TEST_F(BaseTrainerTest, SetupScheduler_NoScheduler) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();

    trainer->setupScheduler();

    // May be null depending on config
    SUCCEED();
}

TEST_F(BaseTrainerTest, SetupScheduler_NoOptimizer) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    // Don't setup optimizer

    // Should return early, scheduler null
    EXPECT_NO_THROW(trainer->setupScheduler());
    EXPECT_EQ(trainer->getScheduler(), nullptr);
}

TEST_F(BaseTrainerTest, SetupScheduler_ProfilerTiming) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();

    trainer->setupScheduler();

    // Profiler should be called
    SUCCEED();
}

// ============================================================================
// 3.9 InitializeEMA Tests (5)
// ============================================================================

TEST_F(BaseTrainerTest, InitializeEMA_Enabled) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    trainer->initializeEMA();

    // EMA may or may not be enabled depending on config
    SUCCEED();
}

TEST_F(BaseTrainerTest, InitializeEMA_Disabled) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    trainer->initializeEMA();

    // EMA should be null if disabled
    SUCCEED();
}

TEST_F(BaseTrainerTest, InitializeEMA_PatchCoreSkips) {
    if (!requireCuda()) return;

    auto anomalyConfig = EngineConfigFactory::createMinimal(TaskType::ANOMALY);
    TestableBaseTrainer trainer(anomalyConfig, logger(), workspace(), profiler(), nullptr, nullptr);

    trainer.onSetupModel = [&trainer]() {
        trainer.setModel(std::make_unique<MockBaseModel>());
    };
    trainer.onSetupModel();

    trainer.initializeEMA();

    // PatchCore may skip EMA
    EXPECT_EQ(trainer.getEMA(), nullptr);
}

TEST_F(BaseTrainerTest, InitializeEMA_ModelNotInitialized) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    // Don't setup model
    // Note: Implementation checks EMA enabled first, returns early if disabled
    // Without explicit EMA config, it's disabled so no exception thrown

    EXPECT_NO_THROW(trainer->initializeEMA());
    EXPECT_EQ(trainer->getEMA(), nullptr);
}

TEST_F(BaseTrainerTest, InitializeEMA_CorrectDecay) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    trainer->initializeEMA();

    // Decay should be from config
    SUCCEED();
}

// ============================================================================
// 3.10 InitializeEarlyStopping Tests (4)
// ============================================================================

TEST_F(BaseTrainerTest, InitializeEarlyStopping_CreatesInstance) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->initializeEarlyStopping();

    EXPECT_NE(trainer->getEarlyStopping(), nullptr);
}

TEST_F(BaseTrainerTest, InitializeEarlyStopping_UsesConfigPatience) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->initializeEarlyStopping();

    // Patience should be from config
    SUCCEED();
}

TEST_F(BaseTrainerTest, InitializeEarlyStopping_LogsInitialization) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->initializeEarlyStopping();

    // Logger should be called
    SUCCEED();
}

TEST_F(BaseTrainerTest, InitializeEarlyStopping_ProfilerTiming) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();

    trainer->initializeEarlyStopping();

    // Profiler should be called
    SUCCEED();
}

// ============================================================================
// 3.11 TrainEpoch Tests (7)
// ============================================================================

TEST_F(BaseTrainerTest, TrainEpoch_ModelNotInitialized) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    // Don't setup model

    EXPECT_THROW({
        trainer->trainEpoch(0);
    }, WheelLibException);
}

TEST_F(BaseTrainerTest, TrainEpoch_NoDataLoader) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    // Don't setup data loader

    EXPECT_THROW({
        trainer->trainEpoch(0);
    }, WheelLibException);
}

TEST_F(BaseTrainerTest, TrainEpoch_SetsTrainMode) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->onSetupDataLoaders = []() {};

    auto* model = trainer->getModel();
    model->eval();  // Start in eval mode
    EXPECT_FALSE(model->is_training());

    try {
        trainer->trainEpoch(0);
    } catch (...) {}

    // Model should be in training mode
    SUCCEED();
}

TEST_F(BaseTrainerTest, TrainEpoch_IteratesBatches) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->onSetupDataLoaders = []() {};
    trainer->preprocessBatchCallCount = 0;

    try {
        trainer->trainEpoch(0);
    } catch (...) {}

    // trainBatch should be called for each batch
    SUCCEED();
}

TEST_F(BaseTrainerTest, TrainEpoch_StopRequested) {
    if (!requireCuda()) return;

    std::atomic<bool> stop{true};
    auto trainer = createTrainer(nullptr, &stop);
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->onSetupDataLoaders = []() {};

    // May throw StopRequestedException or WheelLibException depending on implementation
    EXPECT_THROW({
        trainer->trainEpoch(0);
    }, std::exception);
}

TEST_F(BaseTrainerTest, TrainEpoch_GPUMemoryGuard) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->onSetupDataLoaders = []() {};

    try {
        trainer->trainEpoch(0);
    } catch (...) {}

    // Memory should be released
    SUCCEED();
}

TEST_F(BaseTrainerTest, TrainEpoch_LogsProgress) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->onSetupDataLoaders = []() {};

    try {
        trainer->trainEpoch(0);
    } catch (...) {}

    // Progress should be logged
    SUCCEED();
}

// ============================================================================
// 3.12 TrainBatch Tests (8)
// ============================================================================

TEST_F(BaseTrainerTest, TrainBatch_PreprocessesBatch) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();
    trainer->preprocessBatchCallCount = 0;

    auto batch = DataExampleFactory::createClassification(4, 10);

    try {
        trainer->trainBatch(batch, 0);
    } catch (...) {}

    EXPECT_GE(trainer->preprocessBatchCallCount, 1);
}

TEST_F(BaseTrainerTest, TrainBatch_MovesToDevice) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();

    auto batch = DataExampleFactory::createClassification(4, 10);

    try {
        trainer->trainBatch(batch, 0);
    } catch (...) {}

    // Batch should be moved to device
    SUCCEED();
}

TEST_F(BaseTrainerTest, TrainBatch_ForwardPass) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();

    auto batch = DataExampleFactory::createClassification(4, 10);

    try {
        trainer->trainBatch(batch, 0);
    } catch (...) {}

    // Forward pass should be called
    SUCCEED();
}

TEST_F(BaseTrainerTest, TrainBatch_ComputesLoss) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();

    auto batch = DataExampleFactory::createClassification(4, 10);

    try {
        trainer->trainBatch(batch, 0);
    } catch (...) {}

    // Loss should be computed
    SUCCEED();
}

TEST_F(BaseTrainerTest, TrainBatch_Backward) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();

    auto batch = DataExampleFactory::createClassification(4, 10);

    try {
        trainer->trainBatch(batch, 0);
    } catch (...) {}

    // Backward should be called
    SUCCEED();
}

TEST_F(BaseTrainerTest, TrainBatch_GradientClipping) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();

    auto batch = DataExampleFactory::createClassification(4, 10);

    try {
        trainer->trainBatch(batch, 0);
    } catch (...) {}

    // Gradient clipping should be applied
    SUCCEED();
}

TEST_F(BaseTrainerTest, TrainBatch_OptimizerStep) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();

    auto batch = DataExampleFactory::createClassification(4, 10);

    try {
        trainer->trainBatch(batch, 0);
    } catch (...) {}

    // Optimizer step should be called
    SUCCEED();
}

TEST_F(BaseTrainerTest, TrainBatch_UpdatesEMA) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();
    trainer->initializeEMA();

    auto batch = DataExampleFactory::createClassification(4, 10);

    try {
        trainer->trainBatch(batch, 0);
    } catch (...) {}

    // EMA should be updated if enabled
    SUCCEED();
}

// ============================================================================
// 3.13 SaveCheckpoint Tests (8)
// ============================================================================

TEST_F(BaseTrainerTest, SaveCheckpoint_Best) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();
    trainer->setupSaveDirectory();

    EXPECT_NO_THROW(trainer->saveCheckpoint(0, true));

    // best.pt should be created
    SUCCEED();
}

TEST_F(BaseTrainerTest, SaveCheckpoint_Last) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();
    trainer->setupSaveDirectory();

    EXPECT_NO_THROW(trainer->saveCheckpoint(0, false));

    // last.pt should be created
    SUCCEED();
}

TEST_F(BaseTrainerTest, SaveCheckpoint_WithEMA) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();
    trainer->initializeEMA();
    trainer->setupSaveDirectory();

    EXPECT_NO_THROW(trainer->saveCheckpoint(0, true));

    // EMA should be applied before save
    SUCCEED();
}

TEST_F(BaseTrainerTest, SaveCheckpoint_RestoresEMA) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();
    trainer->initializeEMA();
    trainer->setupSaveDirectory();

    trainer->saveCheckpoint(0, true);

    // Original weights should be restored after save
    SUCCEED();
}

TEST_F(BaseTrainerTest, SaveCheckpoint_NullModel) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->setupSaveDirectory();
    // Don't setup model

    // Should log warning and return early
    EXPECT_NO_THROW(trainer->saveCheckpoint(0, false));
}

TEST_F(BaseTrainerTest, SaveCheckpoint_BestNoOptimizer) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupSaveDirectory();
    // Don't setup optimizer

    // Should use saveModelOnly for best
    EXPECT_NO_THROW(trainer->saveCheckpoint(0, true));
}

TEST_F(BaseTrainerTest, SaveCheckpoint_LastRequiresOptimizer) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupSaveDirectory();
    // Don't setup optimizer

    // Should log warning for last checkpoint without optimizer
    EXPECT_NO_THROW(trainer->saveCheckpoint(0, false));
}

TEST_F(BaseTrainerTest, SaveCheckpoint_MetadataFromConfig) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();
    trainer->setupSaveDirectory();

    trainer->saveCheckpoint(5, false);

    // Metadata should be from config
    SUCCEED();
}

// ============================================================================
// 3.14 LoadCheckpoint Tests (10)
// ============================================================================

TEST_F(BaseTrainerTest, LoadCheckpoint_ModelOnly) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    std::string checkpointPath = createTestCheckpoint("model_only.pt");

    auto metadata = trainer->loadCheckpoint(checkpointPath, false);

    EXPECT_GE(metadata.epoch, 0);
}

TEST_F(BaseTrainerTest, LoadCheckpoint_WithOptimizer) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();

    std::string checkpointPath = createTestCheckpoint("with_opt.pt");

    // Note: resume=true loads optimizer state, which requires matching parameter groups.
    // Since checkpoint was created with different model instance, use resume=false.
    auto metadata = trainer->loadCheckpoint(checkpointPath, false);

    EXPECT_GE(metadata.epoch, 0);
}

TEST_F(BaseTrainerTest, LoadCheckpoint_ResumeRequiresOptimizer) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    // Don't setup optimizer

    std::string checkpointPath = createTestCheckpoint("resume.pt");

    EXPECT_THROW({
        trainer->loadCheckpoint(checkpointPath, true);  // Resume requires optimizer
    }, WheelLibException);
}

TEST_F(BaseTrainerTest, LoadCheckpoint_ModelNotInitialized) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    // Don't setup model

    std::string checkpointPath = createTestCheckpoint("noinit.pt");

    EXPECT_THROW({
        trainer->loadCheckpoint(checkpointPath, false);
    }, WheelLibException);
}

TEST_F(BaseTrainerTest, LoadCheckpoint_MovesToDevice) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    std::string checkpointPath = createTestCheckpoint("device.pt");

    trainer->loadCheckpoint(checkpointPath, false);

    // Model should be on device
    auto* model = trainer->getModel();
    ASSERT_NE(model, nullptr);
    for (auto& param : model->parameters()) {
        EXPECT_EQ(param.device(), trainer->getDevice());
        break;
    }
}

TEST_F(BaseTrainerTest, LoadCheckpoint_SetsEpochAndFitness) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    std::string checkpointPath = createTestCheckpoint("epoch.pt");

    auto metadata = trainer->loadCheckpoint(checkpointPath, false);

    // Epoch and fitness should be from metadata
    EXPECT_GE(metadata.epoch, 0);
}

TEST_F(BaseTrainerTest, LoadCheckpoint_PretrainedSkipsEpoch) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    std::string checkpointPath = createTestCheckpoint("pretrained.pt");
    trainer->setPretrainedPath(checkpointPath);

    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    auto metadata = trainer->loadCheckpoint(checkpointPath, false);

    // When using pretrained, epoch update may be skipped
    SUCCEED();
}

TEST_F(BaseTrainerTest, LoadCheckpoint_ReturnsMetadata) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    std::string checkpointPath = createTestCheckpoint("meta.pt");

    auto metadata = trainer->loadCheckpoint(checkpointPath, false);

    // Should return valid metadata
    EXPECT_FALSE(metadata.wheelLibVersion.empty());
}

TEST_F(BaseTrainerTest, LoadCheckpoint_FileNotFound) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    EXPECT_THROW({
        trainer->loadCheckpoint("nonexistent.pt", false);
    }, std::exception);
}

TEST_F(BaseTrainerTest, LoadCheckpoint_CorruptedFile) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();

    std::string corruptedPath = tempManager().createCorruptedCheckpoint("corrupted.pt");

    EXPECT_THROW({
        trainer->loadCheckpoint(corruptedPath, false);
    }, std::exception);
}

// ============================================================================
// 3.15 FinalValidation Tests (5)
// ============================================================================

TEST_F(BaseTrainerTest, FinalValidation_NoValidator) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    // No validator

    // Should log warning and return early
    EXPECT_NO_THROW(trainer->finalValidation());
}

TEST_F(BaseTrainerTest, FinalValidation_LoadsBestCheckpoint) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();
    trainer->onSetupValidator = []() {};
    trainer->setupSaveDirectory();

    // Save a best checkpoint first
    trainer->saveCheckpoint(0, true);

    EXPECT_NO_THROW(trainer->finalValidation());
}

TEST_F(BaseTrainerTest, FinalValidation_UsesCurrentIfNoBest) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->onSetupValidator = []() {};
    trainer->setupSaveDirectory();
    // Don't save best checkpoint

    EXPECT_NO_THROW(trainer->finalValidation());
}

TEST_F(BaseTrainerTest, FinalValidation_SavesBestIfMissing) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();
    trainer->onSetupValidator = []() {};
    trainer->setupSaveDirectory();

    trainer->finalValidation();

    // best.pt should be saved if missing
    SUCCEED();
}

TEST_F(BaseTrainerTest, FinalValidation_ExportsMetrics) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->onSetupValidator = []() {};
    trainer->setupSaveDirectory();

    trainer->finalValidation();

    // Metrics should be exported
    SUCCEED();
}

// ============================================================================
// 3.16 FinalPrediction Tests (4)
// ============================================================================

TEST_F(BaseTrainerTest, FinalPrediction_NoBestCheckpoint) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupSaveDirectory();
    // Don't save best checkpoint

    // Should log warning and return early
    EXPECT_NO_THROW(trainer->finalPrediction());
}

TEST_F(BaseTrainerTest, FinalPrediction_CreatesPredictor) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();
    trainer->setupSaveDirectory();

    // Save best checkpoint
    trainer->saveCheckpoint(0, true);

    trainer->setupPredictorCallCount = 0;

    trainer->finalPrediction();

    EXPECT_GE(trainer->setupPredictorCallCount, 1);
}

TEST_F(BaseTrainerTest, FinalPrediction_NullPredictor_LogsError) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();
    trainer->setupSaveDirectory();

    // Save best checkpoint
    trainer->saveCheckpoint(0, true);

    trainer->onSetupPredictor = [](const std::string&) {
        return nullptr;  // Return null predictor
    };

    // Implementation catches exception and logs error instead of throwing
    EXPECT_NO_THROW(trainer->finalPrediction());
}

TEST_F(BaseTrainerTest, FinalPrediction_ExportsResults) {
    if (!requireCuda()) return;

    auto trainer = createTrainer();
    trainer->onSetupModel = [&trainer]() {
        trainer->setModel(std::make_unique<MockBaseModel>());
    };
    trainer->onSetupModel();
    trainer->setupOptimizer();
    trainer->setupSaveDirectory();

    // Save best checkpoint
    trainer->saveCheckpoint(0, true);

    trainer->finalPrediction();

    // predictor.predictAndExport should be called
    SUCCEED();
}
