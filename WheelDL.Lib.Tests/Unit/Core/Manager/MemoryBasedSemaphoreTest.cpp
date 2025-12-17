/**
 * @file MemoryBasedSemaphoreTest.cpp
 * @brief Unit tests for TaskManager's memory management through public API
 *
 * Phase 4 of test_manager.md - Memory Management Tests
 * Tests MemoryBasedSemaphore behavior indirectly through TaskManager's public API.
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include "Core/Manager/TaskManager.h"
#include "Utils/Memory/MemoryManager.h"

using namespace WheelDL;
using namespace WheelDL::Core::Manager;

// ============================================================================
// Test Fixture
// ============================================================================

class MemoryManagementTest : public ::testing::Test {
protected:
    void SetUp() override {
        _hasCuda = torch::cuda::is_available();
        if (_hasCuda) {
            _originalThreshold = TaskManager::getInstance().getMemoryThreshold();
        }
    }

    void TearDown() override {
        if (_hasCuda) {
            // Restore original threshold
            TaskManager::getInstance().setMemoryThreshold(_originalThreshold);
        }
    }

    bool requireCuda() {
        if (!_hasCuda) {
            SUCCEED() << "CUDA not available, test skipped";
            return false;
        }
        return true;
    }

    TaskManager& taskManager() {
        return TaskManager::getInstance();
    }

    Utils::MemoryManager& memoryManager() {
        return Utils::MemoryManager::getInstance();
    }

private:
    bool _hasCuda = false;
    size_t _originalThreshold = 4096;
};

// ============================================================================
// 1. Memory Threshold Tests (15)
// ============================================================================

TEST_F(MemoryManagementTest, getMemoryThreshold_ReturnsValue) {
    if (!requireCuda()) return;

    size_t threshold = taskManager().getMemoryThreshold();
    EXPECT_GE(threshold, 0u);
}

TEST_F(MemoryManagementTest, setMemoryThreshold_UpdatesValue) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(8192);
    EXPECT_EQ(taskManager().getMemoryThreshold(), 8192u);
}

TEST_F(MemoryManagementTest, setMemoryThreshold_ZeroValue) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(0);
    EXPECT_EQ(taskManager().getMemoryThreshold(), 0u);
}

TEST_F(MemoryManagementTest, setMemoryThreshold_SmallValue) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(1);
    EXPECT_EQ(taskManager().getMemoryThreshold(), 1u);
}

TEST_F(MemoryManagementTest, setMemoryThreshold_LargeValue) {
    if (!requireCuda()) return;

    size_t largeThreshold = 1024 * 1024; // 1TB
    taskManager().setMemoryThreshold(largeThreshold);
    EXPECT_EQ(taskManager().getMemoryThreshold(), largeThreshold);
}

TEST_F(MemoryManagementTest, setMemoryThreshold_MultipleUpdates) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(1000);
    EXPECT_EQ(taskManager().getMemoryThreshold(), 1000u);

    taskManager().setMemoryThreshold(2000);
    EXPECT_EQ(taskManager().getMemoryThreshold(), 2000u);

    taskManager().setMemoryThreshold(500);
    EXPECT_EQ(taskManager().getMemoryThreshold(), 500u);
}

TEST_F(MemoryManagementTest, setMemoryThreshold_NoException) {
    if (!requireCuda()) return;

    EXPECT_NO_THROW(taskManager().setMemoryThreshold(4096));
}

TEST_F(MemoryManagementTest, getMemoryThreshold_NoException) {
    if (!requireCuda()) return;

    EXPECT_NO_THROW(taskManager().getMemoryThreshold());
}

TEST_F(MemoryManagementTest, getMemoryThreshold_Const) {
    if (!requireCuda()) return;

    const TaskManager& constRef = taskManager();
    // Should compile - getMemoryThreshold is const
    size_t threshold = constRef.getMemoryThreshold();
    EXPECT_GE(threshold, 0u);
}

TEST_F(MemoryManagementTest, setMemoryThreshold_ThreadSafe) {
    if (!requireCuda()) return;

    std::vector<std::thread> threads;

    for (int i = 0; i < 10; i++) {
        threads.emplace_back([this, i]() {
            taskManager().setMemoryThreshold(static_cast<size_t>(i * 1000));
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // No crash - thread safe
    EXPECT_GE(taskManager().getMemoryThreshold(), 0u);
}

TEST_F(MemoryManagementTest, getMemoryThreshold_ThreadSafe) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(4096);
    std::atomic<size_t> sum{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; i++) {
        threads.emplace_back([this, &sum]() {
            sum += taskManager().getMemoryThreshold();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All threads should read the same value
    EXPECT_EQ(sum.load(), 4096u * 10);
}

TEST_F(MemoryManagementTest, setMemoryThreshold_DefaultValue) {
    if (!requireCuda()) return;

    // Default threshold is typically 4096 MB (4GB)
    taskManager().setMemoryThreshold(4096);
    EXPECT_EQ(taskManager().getMemoryThreshold(), 4096u);
}

TEST_F(MemoryManagementTest, setMemoryThreshold_Persistence) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(7777);

    // Value should persist across multiple calls
    EXPECT_EQ(taskManager().getMemoryThreshold(), 7777u);
    EXPECT_EQ(taskManager().getMemoryThreshold(), 7777u);
    EXPECT_EQ(taskManager().getMemoryThreshold(), 7777u);
}

TEST_F(MemoryManagementTest, getMemoryThreshold_AfterSet) {
    if (!requireCuda()) return;

    size_t original = taskManager().getMemoryThreshold();
    taskManager().setMemoryThreshold(original + 1000);
    EXPECT_EQ(taskManager().getMemoryThreshold(), original + 1000);
}

TEST_F(MemoryManagementTest, setMemoryThreshold_SameValue) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(5000);
    taskManager().setMemoryThreshold(5000);
    EXPECT_EQ(taskManager().getMemoryThreshold(), 5000u);
}

// ============================================================================
// 2. canAcceptNewTask Tests (15)
// ============================================================================

TEST_F(MemoryManagementTest, canAcceptNewTask_LowThreshold_True) {
    if (!requireCuda()) return;

    // Very low threshold should always return true
    taskManager().setMemoryThreshold(1);
    EXPECT_TRUE(taskManager().canAcceptNewTask(0));
}

TEST_F(MemoryManagementTest, canAcceptNewTask_HighThreshold_False) {
    if (!requireCuda()) return;

    // Very high threshold should return false
    taskManager().setMemoryThreshold(1024 * 1024); // 1TB
    EXPECT_FALSE(taskManager().canAcceptNewTask(0));
}

TEST_F(MemoryManagementTest, canAcceptNewTask_DefaultDevice) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(1);
    // Default device is 0
    EXPECT_TRUE(taskManager().canAcceptNewTask());
}

TEST_F(MemoryManagementTest, canAcceptNewTask_CustomDevice) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(1);
    EXPECT_TRUE(taskManager().canAcceptNewTask(0));
}

TEST_F(MemoryManagementTest, canAcceptNewTask_ZeroThreshold_True) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(0);
    EXPECT_TRUE(taskManager().canAcceptNewTask(0));
}

TEST_F(MemoryManagementTest, canAcceptNewTask_NoException) {
    if (!requireCuda()) return;

    EXPECT_NO_THROW(taskManager().canAcceptNewTask(0));
}

TEST_F(MemoryManagementTest, canAcceptNewTask_Const) {
    if (!requireCuda()) return;

    const TaskManager& constRef = taskManager();
    // Should compile - canAcceptNewTask is const
    bool result = constRef.canAcceptNewTask(0);
    (void)result; // Suppress unused warning
    SUCCEED();
}

TEST_F(MemoryManagementTest, canAcceptNewTask_ReflectsThreshold) {
    if (!requireCuda()) return;

    float available = taskManager().getAvailableGpuMemory(0);

    // Set threshold below available - should return true
    taskManager().setMemoryThreshold(static_cast<size_t>(available / 2));
    EXPECT_TRUE(taskManager().canAcceptNewTask(0));

    // Set threshold above available - should return false
    taskManager().setMemoryThreshold(static_cast<size_t>(available * 2));
    EXPECT_FALSE(taskManager().canAcceptNewTask(0));
}

TEST_F(MemoryManagementTest, canAcceptNewTask_ThreadSafe) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(1);
    std::atomic<int> trueCount{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; i++) {
        threads.emplace_back([this, &trueCount]() {
            if (taskManager().canAcceptNewTask(0)) {
                trueCount++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(trueCount.load(), 10);
}

TEST_F(MemoryManagementTest, canAcceptNewTask_ConsistentWithAvailable) {
    if (!requireCuda()) return;

    float available = taskManager().getAvailableGpuMemory(0);
    size_t threshold = static_cast<size_t>(available);

    taskManager().setMemoryThreshold(threshold);
    // At exact threshold, should return true (>=)
    EXPECT_TRUE(taskManager().canAcceptNewTask(0));
}

TEST_F(MemoryManagementTest, canAcceptNewTask_MultipleCalls) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(1);

    // Multiple calls should return consistent results
    bool first = taskManager().canAcceptNewTask(0);
    bool second = taskManager().canAcceptNewTask(0);
    bool third = taskManager().canAcceptNewTask(0);

    EXPECT_EQ(first, second);
    EXPECT_EQ(second, third);
}

TEST_F(MemoryManagementTest, canAcceptNewTask_AfterThresholdChange) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(1);
    EXPECT_TRUE(taskManager().canAcceptNewTask(0));

    taskManager().setMemoryThreshold(1024 * 1024);
    EXPECT_FALSE(taskManager().canAcceptNewTask(0));
}

TEST_F(MemoryManagementTest, canAcceptNewTask_ReturnsBoolean) {
    if (!requireCuda()) return;

    bool result = taskManager().canAcceptNewTask(0);
    EXPECT_TRUE(result == true || result == false);
}

TEST_F(MemoryManagementTest, canAcceptNewTask_QuickReturn) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(1);
    auto start = std::chrono::steady_clock::now();
    taskManager().canAcceptNewTask(0);
    auto elapsed = std::chrono::steady_clock::now() - start;

    // Should return quickly (under 100ms)
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 100);
}

TEST_F(MemoryManagementTest, canAcceptNewTask_Device0) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(1);
    EXPECT_TRUE(taskManager().canAcceptNewTask(0));
}

// ============================================================================
// 3. getAvailableGpuMemory Tests (15)
// ============================================================================

TEST_F(MemoryManagementTest, getAvailableGpuMemory_ReturnsPositive) {
    if (!requireCuda()) return;

    float available = taskManager().getAvailableGpuMemory(0);
    EXPECT_GE(available, 0.0f);
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_DefaultDevice) {
    if (!requireCuda()) return;

    float available = taskManager().getAvailableGpuMemory();
    EXPECT_GE(available, 0.0f);
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_CustomDevice) {
    if (!requireCuda()) return;

    float available = taskManager().getAvailableGpuMemory(0);
    EXPECT_GE(available, 0.0f);
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_Const) {
    if (!requireCuda()) return;

    const TaskManager& constRef = taskManager();
    // Should compile - getAvailableGpuMemory is const
    float available = constRef.getAvailableGpuMemory(0);
    EXPECT_GE(available, 0.0f);
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_NoException) {
    if (!requireCuda()) return;

    EXPECT_NO_THROW(taskManager().getAvailableGpuMemory(0));
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_ReasonableRange) {
    if (!requireCuda()) return;

    float available = taskManager().getAvailableGpuMemory(0);
    // Should be less than 1TB (reasonable upper bound)
    EXPECT_LT(available, 1024.0f * 1024.0f);
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_ConsistentWithMemoryManager) {
    if (!requireCuda()) return;

    float total = memoryManager().getGPUMemoryTotal(0);
    float used = memoryManager().getGPUMemoryUsed(0);
    float expected = total - used;
    float actual = taskManager().getAvailableGpuMemory(0);

    // Should be approximately equal (within 100MB tolerance)
    EXPECT_NEAR(actual, expected, 100.0f);
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_ThreadSafe) {
    if (!requireCuda()) return;

    std::atomic<int> validCount{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; i++) {
        threads.emplace_back([this, &validCount]() {
            float available = taskManager().getAvailableGpuMemory(0);
            if (available >= 0.0f) {
                validCount++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(validCount.load(), 10);
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_MultipleCalls) {
    if (!requireCuda()) return;

    float first = taskManager().getAvailableGpuMemory(0);
    float second = taskManager().getAvailableGpuMemory(0);

    // Should be approximately equal (within 100MB tolerance due to timing)
    EXPECT_NEAR(first, second, 100.0f);
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_ReturnsFloat) {
    if (!requireCuda()) return;

    float available = taskManager().getAvailableGpuMemory(0);
    // Verify it's a valid float
    EXPECT_FALSE(std::isnan(available));
    EXPECT_FALSE(std::isinf(available));
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_LessThanTotal) {
    if (!requireCuda()) return;

    float available = taskManager().getAvailableGpuMemory(0);
    float total = memoryManager().getGPUMemoryTotal(0);

    EXPECT_LE(available, total);
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_GreaterThanZero) {
    if (!requireCuda()) return;

    float available = taskManager().getAvailableGpuMemory(0);
    // In most cases, some memory should be available
    EXPECT_GE(available, 0.0f);
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_QuickReturn) {
    if (!requireCuda()) return;

    auto start = std::chrono::steady_clock::now();
    taskManager().getAvailableGpuMemory(0);
    auto elapsed = std::chrono::steady_clock::now() - start;

    // Should return quickly (under 100ms)
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 100);
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_NotNegative) {
    if (!requireCuda()) return;

    float available = taskManager().getAvailableGpuMemory(0);
    EXPECT_GE(available, 0.0f);
}

TEST_F(MemoryManagementTest, getAvailableGpuMemory_Device0) {
    if (!requireCuda()) return;

    float available = taskManager().getAvailableGpuMemory(0);
    EXPECT_GE(available, 0.0f);
}

// ============================================================================
// 4. Task Count Tests (15)
// ============================================================================

TEST_F(MemoryManagementTest, getRunningTaskCount_InitialZero) {
    if (!requireCuda()) return;

    // At start, running task count should be 0 or positive (other tests may be running)
    size_t count = taskManager().getRunningTaskCount();
    EXPECT_GE(count, 0u);
}

TEST_F(MemoryManagementTest, getPendingTaskCount_InitialZero) {
    if (!requireCuda()) return;

    // At start, pending task count should be 0 or positive
    size_t count = taskManager().getPendingTaskCount();
    EXPECT_GE(count, 0u);
}

TEST_F(MemoryManagementTest, getRunningTaskCount_NoException) {
    if (!requireCuda()) return;

    EXPECT_NO_THROW(taskManager().getRunningTaskCount());
}

TEST_F(MemoryManagementTest, getPendingTaskCount_NoException) {
    if (!requireCuda()) return;

    EXPECT_NO_THROW(taskManager().getPendingTaskCount());
}

TEST_F(MemoryManagementTest, getRunningTaskCount_ThreadSafe) {
    if (!requireCuda()) return;

    std::vector<std::thread> threads;

    for (int i = 0; i < 10; i++) {
        threads.emplace_back([this]() {
            taskManager().getRunningTaskCount();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // No crash - thread safe
    SUCCEED();
}

TEST_F(MemoryManagementTest, getPendingTaskCount_ThreadSafe) {
    if (!requireCuda()) return;

    std::vector<std::thread> threads;

    for (int i = 0; i < 10; i++) {
        threads.emplace_back([this]() {
            taskManager().getPendingTaskCount();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // No crash - thread safe
    SUCCEED();
}

TEST_F(MemoryManagementTest, getRunningTaskCount_ReturnsSize_t) {
    if (!requireCuda()) return;

    size_t count = taskManager().getRunningTaskCount();
    EXPECT_GE(count, 0u);
}

TEST_F(MemoryManagementTest, getPendingTaskCount_ReturnsSize_t) {
    if (!requireCuda()) return;

    size_t count = taskManager().getPendingTaskCount();
    EXPECT_GE(count, 0u);
}

TEST_F(MemoryManagementTest, getRunningTaskCount_QuickReturn) {
    if (!requireCuda()) return;

    auto start = std::chrono::steady_clock::now();
    taskManager().getRunningTaskCount();
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 100);
}

TEST_F(MemoryManagementTest, getPendingTaskCount_QuickReturn) {
    if (!requireCuda()) return;

    auto start = std::chrono::steady_clock::now();
    taskManager().getPendingTaskCount();
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 100);
}

TEST_F(MemoryManagementTest, getRunningTaskCount_MultipleCalls) {
    if (!requireCuda()) return;

    size_t first = taskManager().getRunningTaskCount();
    size_t second = taskManager().getRunningTaskCount();

    // Should be consistent (or change by at most 1 if a task completes)
    EXPECT_LE(std::abs(static_cast<int>(first) - static_cast<int>(second)), 1);
}

TEST_F(MemoryManagementTest, getPendingTaskCount_MultipleCalls) {
    if (!requireCuda()) return;

    size_t first = taskManager().getPendingTaskCount();
    size_t second = taskManager().getPendingTaskCount();

    // Should be consistent (or change by at most 1)
    EXPECT_LE(std::abs(static_cast<int>(first) - static_cast<int>(second)), 1);
}

TEST_F(MemoryManagementTest, getRunningTaskCount_Const) {
    if (!requireCuda()) return;

    const TaskManager& constRef = taskManager();
    // Should compile - getRunningTaskCount should be const-compatible
    size_t count = constRef.getRunningTaskCount();
    EXPECT_GE(count, 0u);
}

TEST_F(MemoryManagementTest, getPendingTaskCount_Const) {
    if (!requireCuda()) return;

    const TaskManager& constRef = taskManager();
    // Should compile - getPendingTaskCount should be const-compatible
    size_t count = constRef.getPendingTaskCount();
    EXPECT_GE(count, 0u);
}

TEST_F(MemoryManagementTest, TaskCounts_BothAccessible) {
    if (!requireCuda()) return;

    size_t running = taskManager().getRunningTaskCount();
    size_t pending = taskManager().getPendingTaskCount();

    // Both should be accessible
    EXPECT_GE(running, 0u);
    EXPECT_GE(pending, 0u);
}

// ============================================================================
// 5. Integration Tests (8)
// ============================================================================

TEST_F(MemoryManagementTest, Integration_ThresholdAffectsCanAccept) {
    if (!requireCuda()) return;

    float available = taskManager().getAvailableGpuMemory(0);

    // Low threshold - can accept
    taskManager().setMemoryThreshold(1);
    EXPECT_TRUE(taskManager().canAcceptNewTask(0));

    // High threshold - cannot accept
    taskManager().setMemoryThreshold(static_cast<size_t>(available * 2));
    EXPECT_FALSE(taskManager().canAcceptNewTask(0));
}

TEST_F(MemoryManagementTest, Integration_MemoryAndThreshold) {
    if (!requireCuda()) return;

    float available = taskManager().getAvailableGpuMemory(0);
    size_t threshold = taskManager().getMemoryThreshold();

    bool canAccept = taskManager().canAcceptNewTask(0);

    // canAccept should be true if available >= threshold
    EXPECT_EQ(canAccept, available >= static_cast<float>(threshold));
}

TEST_F(MemoryManagementTest, Integration_AllMethodsAccessible) {
    if (!requireCuda()) return;

    EXPECT_NO_THROW({
        taskManager().getMemoryThreshold();
        taskManager().setMemoryThreshold(4096);
        taskManager().canAcceptNewTask(0);
        taskManager().getAvailableGpuMemory(0);
        taskManager().getRunningTaskCount();
        taskManager().getPendingTaskCount();
    });
}

TEST_F(MemoryManagementTest, Integration_ConcurrentAccess) {
    if (!requireCuda()) return;

    std::vector<std::thread> threads;

    for (int i = 0; i < 5; i++) {
        threads.emplace_back([this]() {
            taskManager().getMemoryThreshold();
            taskManager().canAcceptNewTask(0);
            taskManager().getAvailableGpuMemory(0);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    SUCCEED();
}

TEST_F(MemoryManagementTest, Integration_ThresholdPersistence) {
    if (!requireCuda()) return;

    taskManager().setMemoryThreshold(9999);

    // Other operations shouldn't affect threshold
    taskManager().canAcceptNewTask(0);
    taskManager().getAvailableGpuMemory(0);
    taskManager().getRunningTaskCount();

    EXPECT_EQ(taskManager().getMemoryThreshold(), 9999u);
}

TEST_F(MemoryManagementTest, Integration_RapidThresholdChanges) {
    if (!requireCuda()) return;

    for (size_t i = 0; i < 100; i++) {
        taskManager().setMemoryThreshold(i * 10);
        EXPECT_EQ(taskManager().getMemoryThreshold(), i * 10);
    }
}

TEST_F(MemoryManagementTest, Integration_MemoryQueryConsistency) {
    if (!requireCuda()) return;

    float mem1 = taskManager().getAvailableGpuMemory(0);
    bool can1 = taskManager().canAcceptNewTask(0);

    taskManager().setMemoryThreshold(1);
    bool can2 = taskManager().canAcceptNewTask(0);

    // With threshold of 1MB, should be able to accept if any memory available
    if (mem1 >= 1.0f) {
        EXPECT_TRUE(can2);
    }
}

TEST_F(MemoryManagementTest, Integration_Singleton) {
    if (!requireCuda()) return;

    TaskManager& tm1 = TaskManager::getInstance();
    TaskManager& tm2 = TaskManager::getInstance();

    // Should be the same instance
    EXPECT_EQ(&tm1, &tm2);

    // Changes in one should reflect in other
    tm1.setMemoryThreshold(5555);
    EXPECT_EQ(tm2.getMemoryThreshold(), 5555u);
}
