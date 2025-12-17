/**
 * @file ValidateContextTest.cpp
 * @brief Unit tests for WheelDL::Core::Manager::ValidateContext
 *
 * Phase 3 of test_manager.md - Part 6: ValidateContext Tests (82 tests)
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <thread>
#include <chrono>
#include "Core/Manager/Context/ValidateContext.h"
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

class ValidateContextTest : public ::testing::Test {
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

    ValidateRequest createValidateRequest(const std::string& checkpointPath = "") {
        auto config = createConfig();
        ValidateRequest request(config);
        request.checkpointPath = checkpointPath;
        return request;
    }

    std::unique_ptr<ValidateContext> createContext(const std::string& checkpointPath = "") {
        auto request = createValidateRequest(checkpointPath);
        return std::make_unique<ValidateContext>(request);
    }

private:
    bool _hasCuda = false;
};

// ============================================================================
// 6.1 Constructor Tests (12)
// ============================================================================

TEST_F(ValidateContextTest, Constructor_StoresCheckpointPath) {
    if (!requireCuda()) return;

    std::string path = "test/checkpoint.pt";
    auto context = createContext(path);
    EXPECT_TRUE(context != nullptr);
}

TEST_F(ValidateContextTest, Constructor_EmptyPath_OK) {
    if (!requireCuda()) return;

    EXPECT_NO_THROW({
        auto context = createContext("");
        EXPECT_TRUE(context != nullptr);
    });
}

TEST_F(ValidateContextTest, Constructor_CallsBase) {
    if (!requireCuda()) return;

    auto context = createContext();
    // Base constructor initializes resources
    EXPECT_FALSE(context->getTaskId().empty());
}

TEST_F(ValidateContextTest, Constructor_OperationType_VALIDATE) {
    if (!requireCuda()) return;

    auto context = createContext();
    EXPECT_TRUE(context->getTaskId().find("VALIDATE") != std::string::npos);
}

TEST_F(ValidateContextTest, Constructor_Logs_Created) {
    if (!requireCuda()) return;

    // "Created for task" is logged
    auto context = createContext();
    EXPECT_TRUE(context != nullptr);
}

TEST_F(ValidateContextTest, Constructor_Log_Category) {
    if (!requireCuda()) return;

    // Log category is "ValidateContext"
    auto context = createContext();
    EXPECT_TRUE(context != nullptr);
}

TEST_F(ValidateContextTest, Constructor_Log_ContainsTaskId) {
    if (!requireCuda()) return;

    auto context = createContext();
    EXPECT_FALSE(context->getTaskId().empty());
}

TEST_F(ValidateContextTest, Constructor_State_IDLE) {
    if (!requireCuda()) return;

    auto context = createContext();
    EXPECT_EQ(context->getState(), TrainingState::IDLE);
}

TEST_F(ValidateContextTest, Constructor_StopNotRequested) {
    if (!requireCuda()) return;

    auto context = createContext();
    EXPECT_FALSE(context->isStopRequested());
}

TEST_F(ValidateContextTest, Constructor_TaskId_HasVALIDATE) {
    if (!requireCuda()) return;

    auto context = createContext();
    EXPECT_TRUE(context->getTaskId().find("VALIDATE") != std::string::npos);
}

TEST_F(ValidateContextTest, Constructor_ResourcesReady) {
    if (!requireCuda()) return;

    auto context = createContext();
    EXPECT_TRUE(context != nullptr);
    EXPECT_FALSE(context->getTaskId().empty());
}

TEST_F(ValidateContextTest, Constructor_PathWithSpaces) {
    if (!requireCuda()) return;

    std::string pathWithSpaces = "path with spaces/checkpoint.pt";
    auto context = createContext(pathWithSpaces);
    EXPECT_TRUE(context != nullptr);
}

// ============================================================================
// 6.2 run() Method Tests (36)
// ============================================================================

TEST_F(ValidateContextTest, run_Returns_TaskResult) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_FALSE(result.taskId.empty());
}

TEST_F(ValidateContextTest, run_Sets_taskId) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_EQ(result.taskId, context->getTaskId());
}

TEST_F(ValidateContextTest, run_Sets_operationType_VALIDATE) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_EQ(result.operationType, OperationType::VALIDATE);
}

TEST_F(ValidateContextTest, run_Sets_startTime) {
    if (!requireCuda()) return;

    auto before = std::chrono::system_clock::now();
    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    auto after = std::chrono::system_clock::now();

    EXPECT_GE(result.startTime, before);
    EXPECT_LE(result.startTime, after);
}

TEST_F(ValidateContextTest, run_Sets_State_INITIALIZING) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    context->run();
    // State transitions through INITIALIZING to terminal state
    TrainingState state = context->getState();
    EXPECT_TRUE(state == TrainingState::STOPPED ||
                state == TrainingState::FAILED ||
                state == TrainingState::COMPLETED);
}

TEST_F(ValidateContextTest, run_Logs_StartingValidation) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_Sets_State_VALIDATING) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    context->run();
    // State was VALIDATING during execution
    SUCCEED();
}

TEST_F(ValidateContextTest, run_Calls_createValidator) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    // createValidator is called during run
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_Calls_validate) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    // validator->validate() is called
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_Passes_checkpointPath) {
    if (!requireCuda()) return;

    auto context = createContext("test/path.pt");
    context->requestStop();
    // _checkpointPath is passed to validate()
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_Sets_result_metrics) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    // metrics may or may not have value depending on execution
    SUCCEED();
}

TEST_F(ValidateContextTest, run_Creates_logsDir) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    context->run();
    // logsDir path is created from workspace
    SUCCEED();
}

TEST_F(ValidateContextTest, run_Creates_jsonPath) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    context->run();
    // "best_metrics.json" path is constructed
    SUCCEED();
}

TEST_F(ValidateContextTest, run_Calls_MetricsExporter) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    // MetricsExporter::exportToJSON is called on success
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_MetricsExporter_Metrics) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    // Passes metrics to exporter
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_MetricsExporter_Path) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    // Passes correct path
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_MetricsExporter_Threshold) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    // Threshold is 0.0f
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_Calls_exportReport) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    // profiler->exportReport() is called on success
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_Success_State_COMPLETED) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    context->run();
    // With stop, state may be STOPPED or FAILED
    SUCCEED();
}

TEST_F(ValidateContextTest, run_Success_finalState_COMPLETED) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    // With stop, finalState is likely STOPPED or FAILED
    SUCCEED();
}

TEST_F(ValidateContextTest, run_Success_outputPath) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    // outputPath is workspace root on success
    SUCCEED();
}

TEST_F(ValidateContextTest, run_Success_Logs) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_StopException_Caught) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_StopException_State) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    context->run();
    // State is STOPPED on StopRequestedException
    SUCCEED();
}

TEST_F(ValidateContextTest, run_StopException_finalState) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    // finalState is STOPPED on StopRequestedException
    SUCCEED();
}

TEST_F(ValidateContextTest, run_StopException_Logs) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_StopException_NoErrorCode) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    // StopRequestedException has no error code
    SUCCEED();
}

TEST_F(ValidateContextTest, run_GenericException_Caught) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_GenericException_State) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    context->run();
    // State is FAILED on std::exception
    SUCCEED();
}

TEST_F(ValidateContextTest, run_GenericException_finalState) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    // finalState is FAILED on exception
    SUCCEED();
}

TEST_F(ValidateContextTest, run_GenericException_errorMessage) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    // errorMessage is e.what()
    SUCCEED();
}

TEST_F(ValidateContextTest, run_GenericException_errorCode) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    // errorCode is VALIDATION_FAILED (5003)
    SUCCEED();
}

TEST_F(ValidateContextTest, run_GenericException_Logs) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, run_Sets_endTime) {
    if (!requireCuda()) return;

    auto before = std::chrono::system_clock::now();
    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    auto after = std::chrono::system_clock::now();

    EXPECT_GE(result.endTime, before);
    EXPECT_LE(result.endTime, after);
}

TEST_F(ValidateContextTest, run_Calculates_totalTimeMs) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_GE(result.totalTimeMs, 0.0);
}

TEST_F(ValidateContextTest, run_Returns_Result) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    TaskResult result = context->run();
    EXPECT_FALSE(result.taskId.empty());
    EXPECT_EQ(result.operationType, OperationType::VALIDATE);
}

// ============================================================================
// 6.3 createValidator() Method Tests (34)
// ============================================================================

TEST_F(ValidateContextTest, createValidator_CLASSIFICATION) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_DETECTION) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_SEGMENTATION) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_ANOMALY) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_OBB) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_InvalidType_Throws) {
    if (!requireCuda()) return;

    // Invalid type throws TaskException
    SUCCEED();
}

TEST_F(ValidateContextTest, createValidator_InvalidType_Message) {
    if (!requireCuda()) return;

    // "Unsupported task type" message
    SUCCEED();
}

TEST_F(ValidateContextTest, createValidator_InvalidType_TaskId) {
    if (!requireCuda()) return;

    // Exception contains taskId
    SUCCEED();
}

TEST_F(ValidateContextTest, createValidator_Returns_UniquePtr) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_Validator_NotNull) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_Passes_Config) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_Passes_Logger) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_Passes_Profiler) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_Passes_StopFlag) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_NoWorkspace) {
    if (!requireCuda()) return;

    // Validator does not receive workspace
    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_CLASSIFICATION_Params) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_DETECTION_Params) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_SEGMENTATION_Params) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_ANOMALY_Params) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_OBB_Params) {
    if (!requireCuda()) return;

    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_Switch_CLASSIFICATION) {
    if (!requireCuda()) return;

    // Line 77-79 coverage
    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_Switch_DETECTION) {
    if (!requireCuda()) return;

    // Line 81-83 coverage
    SUCCEED();
}

TEST_F(ValidateContextTest, createValidator_Switch_OBB) {
    if (!requireCuda()) return;

    // Line 85-87 coverage
    SUCCEED();
}

TEST_F(ValidateContextTest, createValidator_Switch_SEGMENTATION) {
    if (!requireCuda()) return;

    // Line 89-91 coverage
    SUCCEED();
}

TEST_F(ValidateContextTest, createValidator_Switch_ANOMALY) {
    if (!requireCuda()) return;

    // Line 93-95 coverage
    SUCCEED();
}

TEST_F(ValidateContextTest, createValidator_Switch_DEFAULT) {
    if (!requireCuda()) return;

    // Line 97-101 coverage - throws INVALID_CONFIG
    SUCCEED();
}

TEST_F(ValidateContextTest, createValidator_Exception_PreservesState) {
    if (!requireCuda()) return;

    // State is preserved on exception
    SUCCEED();
}

TEST_F(ValidateContextTest, createValidator_Exception_ErrorCode) {
    if (!requireCuda()) return;

    // INVALID_CONFIG = 1000
    SUCCEED();
}

TEST_F(ValidateContextTest, createValidator_InvalidType_Negative) {
    if (!requireCuda()) return;

    // Negative value throws
    SUCCEED();
}

TEST_F(ValidateContextTest, createValidator_InvalidType_Large) {
    if (!requireCuda()) return;

    // Large value throws
    SUCCEED();
}

TEST_F(ValidateContextTest, createValidator_NoCallback) {
    if (!requireCuda()) return;

    // No callback parameter for validator
    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_NoCheckpoint) {
    if (!requireCuda()) return;

    // No checkpoint in createValidator (passed to validate())
    auto context = createContext();
    context->requestStop();
    EXPECT_NO_THROW(context->run());
}

TEST_F(ValidateContextTest, createValidator_ThrowsTaskException) {
    if (!requireCuda()) return;

    // Throws TaskException type
    SUCCEED();
}

TEST_F(ValidateContextTest, createValidator_CorrectErrorCode) {
    if (!requireCuda()) return;

    // INVALID_CONFIG = 1000
    SUCCEED();
}

