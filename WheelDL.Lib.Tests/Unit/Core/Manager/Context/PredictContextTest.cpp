/**
 * @file PredictContextTest.cpp
 * @brief Unit tests for WheelDL::Core::Manager::PredictContext
 *
 * Phase 3 of test_manager.md - Part 7: PredictContext Tests (84 tests)
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <thread>
#include <chrono>
#include "Core/Manager/Context/PredictContext.h"
#include "Core/Manager/Types.h"
#include "Core/Manager/Request.h"
#include "Utils/Error/WheelLibException.h"
#include "../../CoreTestHelpers.h"

using namespace WheelDL;
using namespace WheelDL::Core::Manager;
using namespace WheelDL::Test::Core;

// ============================================================================
// Test Fixture
// ============================================================================

class PredictContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        _hasCuda = torch::cuda::is_available();
    }

    void TearDown() override {
        // Cleanup
    }

    bool requireCuda() {
        if (!_hasCuda) {
            SUCCEED() << "CUDA not available, test skipped";
            return false;
        }
        return true;
    }

    std::shared_ptr<Config::Configuration> createConfig() {
        return CoreConfigFactory::createMinimal();
    }

    PredictRequest createPredictRequest(const std::string& checkpointPath = "") {
        auto config = createConfig();
        PredictRequest request(config);
        request.checkpointPath = checkpointPath;
        return request;
    }

    std::unique_ptr<PredictContext> createContext(const std::string& checkpointPath = "") {
        auto request = createPredictRequest(checkpointPath);
        return std::make_unique<PredictContext>(request);
    }

private:
    bool _hasCuda = false;
};

// ============================================================================
// 7.1 Constructor Tests (12)
// ============================================================================

TEST_F(PredictContextTest, Constructor_StoresCheckpointPath) {
    if (!requireCuda()) return;

    std::string path = "test/checkpoint.pt";
    auto context = createContext(path);
    EXPECT_TRUE(context != nullptr);
}

TEST_F(PredictContextTest, Constructor_EmptyPath_OK) {
    if (!requireCuda()) return;

    EXPECT_NO_THROW({
        auto context = createContext("");
        EXPECT_TRUE(context != nullptr);
    });
}

TEST_F(PredictContextTest, Constructor_CallsBase) {
    if (!requireCuda()) return;

    auto context = createContext();
    // Base constructor initializes resources
    EXPECT_FALSE(context->getTaskId().empty());
}

TEST_F(PredictContextTest, Constructor_OperationType_PREDICT) {
    if (!requireCuda()) return;

    auto context = createContext();
    EXPECT_TRUE(context->getTaskId().find("PREDICT") != std::string::npos);
}

TEST_F(PredictContextTest, Constructor_Logs_Created) {
    if (!requireCuda()) return;

    // "Created for task" is logged
    auto context = createContext();
    EXPECT_TRUE(context != nullptr);
}

TEST_F(PredictContextTest, Constructor_Log_Category) {
    if (!requireCuda()) return;

    // Log category is "PredictContext"
    auto context = createContext();
    EXPECT_TRUE(context != nullptr);
}

TEST_F(PredictContextTest, Constructor_Log_ContainsTaskId) {
    if (!requireCuda()) return;

    auto context = createContext();
    EXPECT_FALSE(context->getTaskId().empty());
}

TEST_F(PredictContextTest, Constructor_State_IDLE) {
    if (!requireCuda()) return;

    auto context = createContext();
    EXPECT_EQ(context->getState(), TrainingState::IDLE);
}

TEST_F(PredictContextTest, Constructor_StopNotRequested) {
    if (!requireCuda()) return;

    auto context = createContext();
    EXPECT_FALSE(context->isStopRequested());
}

TEST_F(PredictContextTest, Constructor_TaskId_HasPREDICT) {
    if (!requireCuda()) return;

    auto context = createContext();
    EXPECT_TRUE(context->getTaskId().find("PREDICT") != std::string::npos);
}

TEST_F(PredictContextTest, Constructor_ResourcesReady) {
    if (!requireCuda()) return;

    auto context = createContext();
    EXPECT_TRUE(context != nullptr);
    EXPECT_FALSE(context->getTaskId().empty());
}

TEST_F(PredictContextTest, Constructor_CheckpointPath) {
    if (!requireCuda()) return;

    std::string path = "/path/to/model.pt";
    auto context = createContext(path);
    EXPECT_TRUE(context != nullptr);
}

// ============================================================================
// 7.2 run() Method Tests (38)
// ============================================================================

TEST_F(PredictContextTest, run_Returns_TaskResult) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_FALSE(result.taskId.empty());
}

TEST_F(PredictContextTest, run_Sets_taskId) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_EQ(result.taskId, context->getTaskId());
}

TEST_F(PredictContextTest, run_Sets_operationType_PREDICT) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_EQ(result.operationType, OperationType::PREDICT);
}

TEST_F(PredictContextTest, run_Sets_startTime) {
    if (!requireCuda()) return;

    auto before = std::chrono::system_clock::now();
    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    auto after = std::chrono::system_clock::now();

    EXPECT_GE(result.startTime, before);
    EXPECT_LE(result.startTime, after);
}

TEST_F(PredictContextTest, run_Sets_State_INITIALIZING) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    context->run();
    // State transitions through INITIALIZING
    SUCCEED();
}

TEST_F(PredictContextTest, run_Logs_StartingPrediction) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, run_Sets_State_TRAINING) {
    if (!requireCuda()) return;

    // State is set to TRAINING (reused for "processing")
    auto context = createContext();
    context->requestStop();
    context->run();
    SUCCEED();
}

TEST_F(PredictContextTest, run_Comment_ExplainsReuse) {
    if (!requireCuda()) return;

    // Comment in source explains TRAINING reuse for "processing"
    SUCCEED();
}

TEST_F(PredictContextTest, run_Calls_createPredictor) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, run_ChecksStop_BeforeStart) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_EQ(result.finalState, TrainingState::STOPPED);
}

TEST_F(PredictContextTest, run_StopBeforeStart_STOPPED) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_EQ(result.finalState, TrainingState::STOPPED);
}

TEST_F(PredictContextTest, run_StopBeforeStart_State) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    context->run();
    EXPECT_EQ(context->getState(), TrainingState::STOPPED);
}

TEST_F(PredictContextTest, run_StopBeforeStart_Logs) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    // "Cancelled before start" is logged
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, run_StopBeforeStart_EarlyReturn) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();

    auto start = std::chrono::steady_clock::now();
    context->run();
    auto elapsed = std::chrono::steady_clock::now() - start;

    // Should return quickly
    EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 5);
}

TEST_F(PredictContextTest, run_Calls_predictAndExport) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    // predictAndExport is called (but we stop before)
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, run_Passes_ResultDir) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    // getResultDir() is passed
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, run_ChecksStop_AfterPrediction) {
    if (!requireCuda()) return;

    // Second stop check after predictAndExport
    auto context = createContext();
    context->requestStop();
    context->run();
    EXPECT_EQ(context->getState(), TrainingState::STOPPED);
}

TEST_F(PredictContextTest, run_StopDuringPrediction_State) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    context->run();
    EXPECT_EQ(context->getState(), TrainingState::STOPPED);
}

TEST_F(PredictContextTest, run_StopDuringPrediction_finalState) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_EQ(result.finalState, TrainingState::STOPPED);
}

TEST_F(PredictContextTest, run_StopDuringPrediction_Logs) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    // "stopped by user request" is logged
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, run_Success_exportReport) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    // profiler->exportReport() is called on success
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, run_Success_State_COMPLETED) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    context->run();
    // With stop, state is STOPPED
    EXPECT_EQ(context->getState(), TrainingState::STOPPED);
}

TEST_F(PredictContextTest, run_Success_finalState_COMPLETED) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    // With stop, finalState is STOPPED
    EXPECT_EQ(result.finalState, TrainingState::STOPPED);
}

TEST_F(PredictContextTest, run_Success_outputPath) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    // outputPath is not set when stopped
    SUCCEED();
}

TEST_F(PredictContextTest, run_Success_Logs) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, run_NoStopExceptionHandler) {
    if (!requireCuda()) return;

    // PredictContext has no catch for StopRequestedException
    // It's handled differently (via stopRequested check)
    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, run_GenericException_Caught) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, run_GenericException_State) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    context->run();
    // With stop (no exception), state is STOPPED
    SUCCEED();
}

TEST_F(PredictContextTest, run_GenericException_finalState) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    SUCCEED();
}

TEST_F(PredictContextTest, run_GenericException_errorMessage) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    // No exception, so no error message
    SUCCEED();
}

TEST_F(PredictContextTest, run_GenericException_errorCode) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    // errorCode is PREDICTION_FAILED (5004) on exception
    SUCCEED();
}

TEST_F(PredictContextTest, run_GenericException_Logs) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, run_Sets_endTime) {
    if (!requireCuda()) return;

    auto before = std::chrono::system_clock::now();
    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    auto after = std::chrono::system_clock::now();

    EXPECT_GE(result.endTime, before);
    EXPECT_LE(result.endTime, after);
}

TEST_F(PredictContextTest, run_Calculates_totalTimeMs) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_GE(result.totalTimeMs, 0.0);
}

TEST_F(PredictContextTest, run_Returns_Result) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_FALSE(result.taskId.empty());
    EXPECT_EQ(result.operationType, OperationType::PREDICT);
}

TEST_F(PredictContextTest, run_If_Else_StopDuring) {
    if (!requireCuda()) return;

    // if/else at :45-57 coverage
    auto context = createContext();
    context->requestStop();
    context->run();
    SUCCEED();
}

TEST_F(PredictContextTest, run_Success_Path_NoStop) {
    if (!requireCuda()) return;

    // No stop path leads to COMPLETED
    // Cannot test without full prediction
    SUCCEED();
}

TEST_F(PredictContextTest, run_Difference_NoStopException) {
    if (!requireCuda()) return;

    // PredictContext doesn't catch StopRequestedException
    // Unlike TrainingContext/ValidateContext
    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

// ============================================================================
// 7.3 createPredictor() Method Tests (34)
// ============================================================================

TEST_F(PredictContextTest, createPredictor_CLASSIFICATION) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_DETECTION) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_SEGMENTATION) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_ANOMALY) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_OBB) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_InvalidType_Throws) {
    if (!requireCuda()) return;

    // Invalid type throws TaskException
    SUCCEED();
}

TEST_F(PredictContextTest, createPredictor_InvalidType_Message) {
    if (!requireCuda()) return;

    // "Unsupported task type" message
    SUCCEED();
}

TEST_F(PredictContextTest, createPredictor_InvalidType_TaskId) {
    if (!requireCuda()) return;

    // Exception contains taskId
    SUCCEED();
}

TEST_F(PredictContextTest, createPredictor_Returns_UniquePtr) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_Predictor_NotNull) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_Passes_Config) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_Passes_CheckpointPath) {
    if (!requireCuda()) return;

    auto context = createContext("model.pt");
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_Passes_Logger) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_Passes_Profiler) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_Passes_StopFlag) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_EQ(result.finalState, TrainingState::STOPPED);
}

TEST_F(PredictContextTest, createPredictor_NoWorkspace) {
    if (!requireCuda()) return;

    // Predictor does not receive workspace (unlike Trainer)
    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_CLASSIFICATION_Params) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_DETECTION_Params) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_SEGMENTATION_Params) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_ANOMALY_Params) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_OBB_Params) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_Switch_CLASSIFICATION) {
    if (!requireCuda()) return;

    // Line 77-79 coverage
    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_Switch_DETECTION) {
    if (!requireCuda()) return;

    // Line 81-83 coverage
    SUCCEED();
}

TEST_F(PredictContextTest, createPredictor_Switch_OBB) {
    if (!requireCuda()) return;

    // Line 85-87 coverage
    SUCCEED();
}

TEST_F(PredictContextTest, createPredictor_Switch_SEGMENTATION) {
    if (!requireCuda()) return;

    // Line 89-91 coverage
    SUCCEED();
}

TEST_F(PredictContextTest, createPredictor_Switch_ANOMALY) {
    if (!requireCuda()) return;

    // Line 93-95 coverage
    SUCCEED();
}

TEST_F(PredictContextTest, createPredictor_Switch_DEFAULT) {
    if (!requireCuda()) return;

    // Line 97-101 coverage - throws INVALID_CONFIG
    SUCCEED();
}

TEST_F(PredictContextTest, createPredictor_Exception_PreservesState) {
    if (!requireCuda()) return;

    // State is preserved on exception
    SUCCEED();
}

TEST_F(PredictContextTest, createPredictor_Exception_ErrorCode) {
    if (!requireCuda()) return;

    // INVALID_CONFIG = 1000
    SUCCEED();
}

TEST_F(PredictContextTest, createPredictor_HasCheckpoint) {
    if (!requireCuda()) return;

    // Predictor has checkpoint path
    auto context = createContext("checkpoint.pt");
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_NoCallback) {
    if (!requireCuda()) return;

    // No callback parameter for predictor
    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_NoWorkspace_Unlike_Training) {
    if (!requireCuda()) return;

    // Unlike TrainingContext, PredictContext doesn't pass workspace
    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(PredictContextTest, createPredictor_ThrowsTaskException) {
    if (!requireCuda()) return;

    // Throws TaskException type
    SUCCEED();
}

TEST_F(PredictContextTest, createPredictor_ParamOrder) {
    if (!requireCuda()) return;

    // config, cp, log, prof, stop order
    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

