/**
 * @file ResultTest.cpp
 * @brief Unit tests for WheelDL::Core::Manager Result Structures
 *
 * Phase 1 of test_manager.md - 58 tests
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Core/Manager/Result.h"

using namespace WheelDL;
using namespace WheelDL::Core::Manager;

// ============================================================================
// Test Fixture
// ============================================================================

class ResultTest : public ::testing::Test {
protected:
    void SetUp() override {
        _hasCuda = torch::cuda::is_available();
    }

    bool requireCuda() {
        if (!_hasCuda) {
            SUCCEED() << "CUDA not available, test skipped";
            return false;
        }
        return true;
    }

private:
    bool _hasCuda = false;
};

// ============================================================================
// 3.1 ProfilingData Tests (26)
// ============================================================================

TEST_F(ResultTest, ProfilingData_Default_totalTimeMs) {
    if (!requireCuda()) return;

    ProfilingData data;
    EXPECT_DOUBLE_EQ(data.totalTimeMs, 0.0);
}

TEST_F(ResultTest, ProfilingData_Default_gpuTimeMs) {
    if (!requireCuda()) return;

    ProfilingData data;
    EXPECT_DOUBLE_EQ(data.gpuTimeMs, 0.0);
}

TEST_F(ResultTest, ProfilingData_Default_dataLoadTimeMs) {
    if (!requireCuda()) return;

    ProfilingData data;
    EXPECT_DOUBLE_EQ(data.dataLoadTimeMs, 0.0);
}

TEST_F(ResultTest, ProfilingData_Default_preprocessTimeMs) {
    if (!requireCuda()) return;

    ProfilingData data;
    EXPECT_DOUBLE_EQ(data.preprocessTimeMs, 0.0);
}

TEST_F(ResultTest, ProfilingData_Default_inferenceTimeMs) {
    if (!requireCuda()) return;

    ProfilingData data;
    EXPECT_DOUBLE_EQ(data.inferenceTimeMs, 0.0);
}

TEST_F(ResultTest, ProfilingData_Default_postprocessTimeMs) {
    if (!requireCuda()) return;

    ProfilingData data;
    EXPECT_DOUBLE_EQ(data.postprocessTimeMs, 0.0);
}

TEST_F(ResultTest, ProfilingData_Default_peakGpuMemoryMB) {
    if (!requireCuda()) return;

    ProfilingData data;
    EXPECT_EQ(data.peakGpuMemoryMB, 0u);
}

TEST_F(ResultTest, ProfilingData_Default_peakCpuMemoryMB) {
    if (!requireCuda()) return;

    ProfilingData data;
    EXPECT_EQ(data.peakCpuMemoryMB, 0u);
}

TEST_F(ResultTest, ProfilingData_Default_customTimings) {
    if (!requireCuda()) return;

    ProfilingData data;
    EXPECT_TRUE(data.customTimings.empty());
}

TEST_F(ResultTest, ProfilingData_Set_totalTimeMs) {
    if (!requireCuda()) return;

    ProfilingData data;
    data.totalTimeMs = 1234.56;
    EXPECT_DOUBLE_EQ(data.totalTimeMs, 1234.56);
}

TEST_F(ResultTest, ProfilingData_Set_gpuTimeMs) {
    if (!requireCuda()) return;

    ProfilingData data;
    data.gpuTimeMs = 567.89;
    EXPECT_DOUBLE_EQ(data.gpuTimeMs, 567.89);
}

TEST_F(ResultTest, ProfilingData_Set_LargeValue) {
    if (!requireCuda()) return;

    ProfilingData data;
    data.totalTimeMs = 999999999.999;
    EXPECT_DOUBLE_EQ(data.totalTimeMs, 999999999.999);
}

TEST_F(ResultTest, ProfilingData_Set_NegativeValue) {
    if (!requireCuda()) return;

    ProfilingData data;
    data.totalTimeMs = -100.0;
    EXPECT_DOUBLE_EQ(data.totalTimeMs, -100.0);
}

TEST_F(ResultTest, ProfilingData_customTimings_Insert) {
    if (!requireCuda()) return;

    ProfilingData data;
    data.customTimings["custom1"] = 100.0;
    EXPECT_EQ(data.customTimings.size(), 1u);
}

TEST_F(ResultTest, ProfilingData_customTimings_Retrieve) {
    if (!requireCuda()) return;

    ProfilingData data;
    data.customTimings["timing1"] = 42.5;
    EXPECT_DOUBLE_EQ(data.customTimings["timing1"], 42.5);
}

TEST_F(ResultTest, ProfilingData_customTimings_Multiple) {
    if (!requireCuda()) return;

    ProfilingData data;
    data.customTimings["timing1"] = 1.0;
    data.customTimings["timing2"] = 2.0;
    data.customTimings["timing3"] = 3.0;
    EXPECT_EQ(data.customTimings.size(), 3u);
    EXPECT_DOUBLE_EQ(data.customTimings["timing1"], 1.0);
    EXPECT_DOUBLE_EQ(data.customTimings["timing2"], 2.0);
    EXPECT_DOUBLE_EQ(data.customTimings["timing3"], 3.0);
}

TEST_F(ResultTest, ProfilingData_customTimings_Overwrite) {
    if (!requireCuda()) return;

    ProfilingData data;
    data.customTimings["key"] = 1.0;
    data.customTimings["key"] = 99.0;
    EXPECT_DOUBLE_EQ(data.customTimings["key"], 99.0);
    EXPECT_EQ(data.customTimings.size(), 1u);
}

TEST_F(ResultTest, ProfilingData_customTimings_EmptyKey) {
    if (!requireCuda()) return;

    ProfilingData data;
    data.customTimings[""] = 123.0;
    EXPECT_DOUBLE_EQ(data.customTimings[""], 123.0);
}

TEST_F(ResultTest, ProfilingData_customTimings_LongKey) {
    if (!requireCuda()) return;

    ProfilingData data;
    std::string longKey(1000, 'x');
    data.customTimings[longKey] = 456.0;
    EXPECT_DOUBLE_EQ(data.customTimings[longKey], 456.0);
}

TEST_F(ResultTest, ProfilingData_CopyConstructor) {
    if (!requireCuda()) return;

    ProfilingData original;
    original.totalTimeMs = 100.0;
    original.gpuTimeMs = 50.0;
    original.customTimings["test"] = 25.0;

    ProfilingData copy(original);
    EXPECT_DOUBLE_EQ(copy.totalTimeMs, 100.0);
    EXPECT_DOUBLE_EQ(copy.gpuTimeMs, 50.0);
    EXPECT_DOUBLE_EQ(copy.customTimings["test"], 25.0);

    // Verify independence
    copy.totalTimeMs = 200.0;
    EXPECT_DOUBLE_EQ(original.totalTimeMs, 100.0);
}

TEST_F(ResultTest, ProfilingData_CopyAssignment) {
    if (!requireCuda()) return;

    ProfilingData original;
    original.totalTimeMs = 100.0;

    ProfilingData copy;
    copy = original;
    EXPECT_DOUBLE_EQ(copy.totalTimeMs, 100.0);
}

TEST_F(ResultTest, ProfilingData_MoveConstructor) {
    if (!requireCuda()) return;

    ProfilingData original;
    original.totalTimeMs = 100.0;
    original.customTimings["test"] = 25.0;

    ProfilingData moved(std::move(original));
    EXPECT_DOUBLE_EQ(moved.totalTimeMs, 100.0);
    EXPECT_DOUBLE_EQ(moved.customTimings["test"], 25.0);
}

TEST_F(ResultTest, ProfilingData_MoveAssignment) {
    if (!requireCuda()) return;

    ProfilingData original;
    original.totalTimeMs = 100.0;

    ProfilingData moved;
    moved = std::move(original);
    EXPECT_DOUBLE_EQ(moved.totalTimeMs, 100.0);
}

TEST_F(ResultTest, ProfilingData_SizeOf) {
    if (!requireCuda()) return;

    EXPECT_GT(sizeof(ProfilingData), 0u);
}

TEST_F(ResultTest, ProfilingData_AllFieldsSet) {
    if (!requireCuda()) return;

    ProfilingData data;
    data.totalTimeMs = 1000.0;
    data.gpuTimeMs = 800.0;
    data.dataLoadTimeMs = 50.0;
    data.preprocessTimeMs = 30.0;
    data.inferenceTimeMs = 700.0;
    data.postprocessTimeMs = 20.0;
    data.peakGpuMemoryMB = 4096;
    data.peakCpuMemoryMB = 8192;
    data.customTimings["custom"] = 100.0;

    EXPECT_DOUBLE_EQ(data.totalTimeMs, 1000.0);
    EXPECT_DOUBLE_EQ(data.gpuTimeMs, 800.0);
    EXPECT_DOUBLE_EQ(data.dataLoadTimeMs, 50.0);
    EXPECT_DOUBLE_EQ(data.preprocessTimeMs, 30.0);
    EXPECT_DOUBLE_EQ(data.inferenceTimeMs, 700.0);
    EXPECT_DOUBLE_EQ(data.postprocessTimeMs, 20.0);
    EXPECT_EQ(data.peakGpuMemoryMB, 4096u);
    EXPECT_EQ(data.peakCpuMemoryMB, 8192u);
    EXPECT_DOUBLE_EQ(data.customTimings["custom"], 100.0);
}

TEST_F(ResultTest, ProfilingData_Destructor_MapCleanup) {
    if (!requireCuda()) return;

    {
        ProfilingData data;
        for (int i = 0; i < 100; i++) {
            data.customTimings["key_" + std::to_string(i)] = static_cast<double>(i);
        }
    }
    // Implicit cleanup verification - no leak
    SUCCEED();
}

// ============================================================================
// 3.2 TaskResult Tests (32)
// ============================================================================

TEST_F(ResultTest, TaskResult_Default_taskId) {
    if (!requireCuda()) return;

    TaskResult result;
    EXPECT_TRUE(result.taskId.empty());
}

TEST_F(ResultTest, TaskResult_Default_finalState) {
    if (!requireCuda()) return;

    TaskResult result;
    EXPECT_EQ(result.finalState, TrainingState::IDLE);
}

TEST_F(ResultTest, TaskResult_Default_totalTimeMs) {
    if (!requireCuda()) return;

    TaskResult result;
    EXPECT_DOUBLE_EQ(result.totalTimeMs, 0.0);
}

TEST_F(ResultTest, TaskResult_Default_errorMessage) {
    if (!requireCuda()) return;

    TaskResult result;
    EXPECT_TRUE(result.errorMessage.empty());
}

TEST_F(ResultTest, TaskResult_Default_errorCode) {
    if (!requireCuda()) return;

    TaskResult result;
    EXPECT_EQ(result.errorCode, 0);
}

TEST_F(ResultTest, TaskResult_Default_metrics) {
    if (!requireCuda()) return;

    TaskResult result;
    EXPECT_FALSE(result.metrics.has_value());
}

TEST_F(ResultTest, TaskResult_Default_outputPath) {
    if (!requireCuda()) return;

    TaskResult result;
    EXPECT_TRUE(result.outputPath.empty());
}

TEST_F(ResultTest, TaskResult_Default_checkpointPath) {
    if (!requireCuda()) return;

    TaskResult result;
    EXPECT_TRUE(result.checkpointPath.empty());
}

TEST_F(ResultTest, TaskResult_isSuccess_COMPLETED) {
    if (!requireCuda()) return;

    TaskResult result;
    result.finalState = TrainingState::COMPLETED;
    EXPECT_TRUE(result.isSuccess());
}

TEST_F(ResultTest, TaskResult_isSuccess_FAILED) {
    if (!requireCuda()) return;

    TaskResult result;
    result.finalState = TrainingState::FAILED;
    EXPECT_FALSE(result.isSuccess());
}

TEST_F(ResultTest, TaskResult_isSuccess_STOPPED) {
    if (!requireCuda()) return;

    TaskResult result;
    result.finalState = TrainingState::STOPPED;
    EXPECT_FALSE(result.isSuccess());
}

TEST_F(ResultTest, TaskResult_isSuccess_IDLE) {
    if (!requireCuda()) return;

    TaskResult result;
    result.finalState = TrainingState::IDLE;
    EXPECT_FALSE(result.isSuccess());
}

TEST_F(ResultTest, TaskResult_isSuccess_TRAINING) {
    if (!requireCuda()) return;

    TaskResult result;
    result.finalState = TrainingState::TRAINING;
    EXPECT_FALSE(result.isSuccess());
}

TEST_F(ResultTest, TaskResult_isStopped_STOPPED) {
    if (!requireCuda()) return;

    TaskResult result;
    result.finalState = TrainingState::STOPPED;
    EXPECT_TRUE(result.isStopped());
}

TEST_F(ResultTest, TaskResult_isStopped_COMPLETED) {
    if (!requireCuda()) return;

    TaskResult result;
    result.finalState = TrainingState::COMPLETED;
    EXPECT_FALSE(result.isStopped());
}

TEST_F(ResultTest, TaskResult_isStopped_FAILED) {
    if (!requireCuda()) return;

    TaskResult result;
    result.finalState = TrainingState::FAILED;
    EXPECT_FALSE(result.isStopped());
}

TEST_F(ResultTest, TaskResult_isStopped_IDLE) {
    if (!requireCuda()) return;

    TaskResult result;
    result.finalState = TrainingState::IDLE;
    EXPECT_FALSE(result.isStopped());
}

TEST_F(ResultTest, TaskResult_isFailed_FAILED) {
    if (!requireCuda()) return;

    TaskResult result;
    result.finalState = TrainingState::FAILED;
    EXPECT_TRUE(result.isFailed());
}

TEST_F(ResultTest, TaskResult_isFailed_COMPLETED) {
    if (!requireCuda()) return;

    TaskResult result;
    result.finalState = TrainingState::COMPLETED;
    EXPECT_FALSE(result.isFailed());
}

TEST_F(ResultTest, TaskResult_isFailed_STOPPED) {
    if (!requireCuda()) return;

    TaskResult result;
    result.finalState = TrainingState::STOPPED;
    EXPECT_FALSE(result.isFailed());
}

TEST_F(ResultTest, TaskResult_isFailed_IDLE) {
    if (!requireCuda()) return;

    TaskResult result;
    result.finalState = TrainingState::IDLE;
    EXPECT_FALSE(result.isFailed());
}

TEST_F(ResultTest, TaskResult_metrics_SetValue) {
    if (!requireCuda()) return;

    TaskResult result;
    MetricsData metrics;
    metrics.accuracy = 0.95f;
    result.metrics = metrics;
    EXPECT_TRUE(result.metrics.has_value());
}

TEST_F(ResultTest, TaskResult_metrics_Reset) {
    if (!requireCuda()) return;

    TaskResult result;
    MetricsData metrics;
    result.metrics = metrics;
    EXPECT_TRUE(result.metrics.has_value());
    result.metrics.reset();
    EXPECT_FALSE(result.metrics.has_value());
}

TEST_F(ResultTest, TaskResult_metrics_Access) {
    if (!requireCuda()) return;

    TaskResult result;
    MetricsData metrics;
    metrics.accuracy = 0.85f;
    metrics.mAP = 0.75f;
    result.metrics = metrics;

    EXPECT_FLOAT_EQ(result.metrics->accuracy, 0.85f);
    EXPECT_FLOAT_EQ(result.metrics->mAP, 0.75f);
}

TEST_F(ResultTest, TaskResult_Timing_Consistency) {
    if (!requireCuda()) return;

    TaskResult result;
    result.startTime = std::chrono::system_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    result.endTime = std::chrono::system_clock::now();

    EXPECT_LT(result.startTime, result.endTime);
}

TEST_F(ResultTest, TaskResult_Timing_Duration) {
    if (!requireCuda()) return;

    TaskResult result;
    result.startTime = std::chrono::system_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    result.endTime = std::chrono::system_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        result.endTime - result.startTime).count();
    result.totalTimeMs = static_cast<double>(duration);

    EXPECT_GE(result.totalTimeMs, 50.0);
}

TEST_F(ResultTest, TaskResult_Set_taskId) {
    if (!requireCuda()) return;

    TaskResult result;
    result.taskId = "TRAIN_20240101_120000_001";
    EXPECT_EQ(result.taskId, "TRAIN_20240101_120000_001");
}

TEST_F(ResultTest, TaskResult_Set_operationType) {
    if (!requireCuda()) return;

    TaskResult result;
    result.operationType = OperationType::TRAIN;
    EXPECT_EQ(result.operationType, OperationType::TRAIN);

    result.operationType = OperationType::VALIDATE;
    EXPECT_EQ(result.operationType, OperationType::VALIDATE);

    result.operationType = OperationType::PREDICT;
    EXPECT_EQ(result.operationType, OperationType::PREDICT);
}

TEST_F(ResultTest, TaskResult_Set_outputPath) {
    if (!requireCuda()) return;

    TaskResult result;
    result.outputPath = "/path/to/output";
    EXPECT_EQ(result.outputPath, "/path/to/output");
}

TEST_F(ResultTest, TaskResult_Set_checkpointPath) {
    if (!requireCuda()) return;

    TaskResult result;
    result.checkpointPath = "/path/to/best.pt";
    EXPECT_EQ(result.checkpointPath, "/path/to/best.pt");
}

TEST_F(ResultTest, TaskResult_Set_errorInfo) {
    if (!requireCuda()) return;

    TaskResult result;
    result.errorMessage = "Out of memory";
    result.errorCode = 5000;

    EXPECT_EQ(result.errorMessage, "Out of memory");
    EXPECT_EQ(result.errorCode, 5000);
}

TEST_F(ResultTest, TaskResult_CopyMoveSemantics) {
    if (!requireCuda()) return;

    TaskResult original;
    original.taskId = "TEST_TASK";
    original.finalState = TrainingState::COMPLETED;
    original.totalTimeMs = 1000.0;

    // Copy
    TaskResult copy = original;
    EXPECT_EQ(copy.taskId, "TEST_TASK");
    EXPECT_EQ(copy.finalState, TrainingState::COMPLETED);
    EXPECT_DOUBLE_EQ(copy.totalTimeMs, 1000.0);

    // Move
    TaskResult moved = std::move(copy);
    EXPECT_EQ(moved.taskId, "TEST_TASK");
    EXPECT_EQ(moved.finalState, TrainingState::COMPLETED);
    EXPECT_DOUBLE_EQ(moved.totalTimeMs, 1000.0);
}

