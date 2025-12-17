/**
 * @file CallbackThrottlerTest.cpp
 * @brief Unit tests for WheelDL::Core::Callback::CallbackThrottler
 *
 * Phase 6 of test_core_utils.md - 31 tests
 *
 * Test Sections:
 * - 5.1 Constructor Tests (5)
 * - 5.2 ShouldInvoke Tests (5)
 * - 5.3 RecordInvocation Tests (4)
 * - 5.4 TryInvoke Tests (6)
 * - 5.5 SetMinInterval Tests (5)
 * - 5.6 GetMinInterval Tests (3)
 * - 5.7 Integration Tests (3)
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "../CoreTestHelpers.h"
#include "Core/Callback/CallbackThrottler.h"

#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

using namespace WheelDL::Core::Callback;
using namespace WheelDL::Test::Core;
using namespace WheelDL::Utils;

// ============================================================================
// Test Fixture
// ============================================================================

class CallbackThrottlerTest : public CUDATestFixture {
protected:
    void SetUp() override {
        CUDATestFixture::SetUp();
    }

    void TearDown() override {
        CUDATestFixture::TearDown();
    }

    // Helper to wait for specified milliseconds
    void waitMs(int ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
};

// ============================================================================
// 5.1 Constructor Tests (5)
// ============================================================================

// Default 100ms interval
TEST_F(CallbackThrottlerTest, Constructor_DefaultInterval) {
    if (!requireCuda()) return;

    CallbackThrottler throttler;

    EXPECT_EQ(throttler.getMinInterval(), 100);
}

// Custom interval
TEST_F(CallbackThrottlerTest, Constructor_CustomInterval) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(500);

    EXPECT_EQ(throttler.getMinInterval(), 500);
}

// Zero interval valid
TEST_F(CallbackThrottlerTest, Constructor_ZeroInterval) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(0);

    EXPECT_EQ(throttler.getMinInterval(), 0);
}

// Negative throws
TEST_F(CallbackThrottlerTest, Constructor_NegativeInterval) {
    if (!requireCuda()) return;

    EXPECT_THROW({
        CallbackThrottler throttler(-1);
    }, WheelLibException);

    try {
        CallbackThrottler throttler(-100);
        FAIL() << "Expected WheelLibException";
    }
    catch (const WheelLibException& e) {
        EXPECT_EQ(e.getErrorCode(), ErrorCode::INVALID_ARGUMENT);
    }
}

// Large interval valid
TEST_F(CallbackThrottlerTest, Constructor_LargeInterval) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(10000);

    EXPECT_EQ(throttler.getMinInterval(), 10000);
}

// ============================================================================
// 5.2 ShouldInvoke Tests (5)
// ============================================================================

// Initially true (timer starts at construction, elapsed time >= 0)
TEST_F(CallbackThrottlerTest, ShouldInvoke_Initially) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(0);

    // With 0ms interval, should always be true
    EXPECT_TRUE(throttler.shouldInvoke());
}

// True after interval wait
TEST_F(CallbackThrottlerTest, ShouldInvoke_AfterWait) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(50);
    throttler.recordInvocation();  // Reset timer

    // Wait longer than interval
    waitMs(60);

    EXPECT_TRUE(throttler.shouldInvoke());
}

// False before interval (too soon)
TEST_F(CallbackThrottlerTest, ShouldInvoke_TooSoon) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(200);
    throttler.recordInvocation();  // Reset timer

    // Check immediately (should be false)
    EXPECT_FALSE(throttler.shouldInvoke());
}

// Always true with zero interval
TEST_F(CallbackThrottlerTest, ShouldInvoke_ZeroInterval) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(0);
    throttler.recordInvocation();

    // Should always be true with 0 interval
    EXPECT_TRUE(throttler.shouldInvoke());
    EXPECT_TRUE(throttler.shouldInvoke());
    EXPECT_TRUE(throttler.shouldInvoke());
}

// Thread-safe check
TEST_F(CallbackThrottlerTest, ShouldInvoke_ThreadSafe) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(10);
    std::atomic<int> trueCount{0};
    std::atomic<int> falseCount{0};
    const int numThreads = 4;
    const int checksPerThread = 100;

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&throttler, &trueCount, &falseCount, checksPerThread]() {
            for (int i = 0; i < checksPerThread; ++i) {
                if (throttler.shouldInvoke()) {
                    trueCount++;
                }
                else {
                    falseCount++;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Just verify no crash and counts are reasonable
    EXPECT_EQ(trueCount + falseCount, numThreads * checksPerThread);
}

// ============================================================================
// 5.3 RecordInvocation Tests (4)
// ============================================================================

// Timer reset
TEST_F(CallbackThrottlerTest, RecordInvocation_ResetsTimer) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(100);

    // Wait some time
    waitMs(50);

    // Record invocation (resets timer)
    throttler.recordInvocation();

    // Should now need to wait full interval again
    EXPECT_FALSE(throttler.shouldInvoke());
}

// Record after shouldInvoke
TEST_F(CallbackThrottlerTest, RecordInvocation_AfterShouldInvoke) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(50);

    // Wait for interval
    waitMs(60);
    EXPECT_TRUE(throttler.shouldInvoke());

    // Record invocation
    throttler.recordInvocation();

    // Should be false immediately after
    EXPECT_FALSE(throttler.shouldInvoke());
}

// Multiple records
TEST_F(CallbackThrottlerTest, RecordInvocation_Multiple) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(50);

    for (int i = 0; i < 5; ++i) {
        throttler.recordInvocation();
        // Each record resets timer, so should be false
        EXPECT_FALSE(throttler.shouldInvoke());
        waitMs(10);  // Small wait between records
    }

    // Wait full interval after last record
    waitMs(60);
    EXPECT_TRUE(throttler.shouldInvoke());
}

// Thread-safe record
TEST_F(CallbackThrottlerTest, RecordInvocation_ThreadSafe) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(10);
    const int numThreads = 4;
    const int recordsPerThread = 50;

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&throttler, recordsPerThread]() {
            for (int i = 0; i < recordsPerThread; ++i) {
                EXPECT_NO_THROW(throttler.recordInvocation());
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // No crash means success
    SUCCEED();
}

// ============================================================================
// 5.4 TryInvoke Tests (6)
// ============================================================================

// Initially succeeds (with 0 interval for reliable test)
TEST_F(CallbackThrottlerTest, TryInvoke_Initially) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(0);

    // With 0 interval, should always succeed
    EXPECT_TRUE(throttler.tryInvoke());
}

// Fails if too soon
TEST_F(CallbackThrottlerTest, TryInvoke_TooSoon) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(200);

    // Wait for interval so first call can succeed
    waitMs(210);

    // First call succeeds (resets timer)
    bool first = throttler.tryInvoke();

    // Immediate second call should fail (not enough time passed)
    bool second = throttler.tryInvoke();

    // First should have succeeded, second should fail
    EXPECT_TRUE(first);
    EXPECT_FALSE(second);
}

// Succeeds after wait
TEST_F(CallbackThrottlerTest, TryInvoke_AfterInterval) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(50);

    // First call
    throttler.tryInvoke();

    // Wait for interval
    waitMs(60);

    // Should succeed again
    EXPECT_TRUE(throttler.tryInvoke());
}

// Atomic operation - only one succeeds per interval
TEST_F(CallbackThrottlerTest, TryInvoke_Atomic) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(100);
    std::atomic<int> successCount{0};
    const int numThreads = 10;

    // Reset to known state
    waitMs(110);

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&throttler, &successCount]() {
            if (throttler.tryInvoke()) {
                successCount++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Only one thread should have succeeded
    EXPECT_EQ(successCount.load(), 1);
}

// Always succeeds with zero interval
TEST_F(CallbackThrottlerTest, TryInvoke_ZeroInterval) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(0);

    // All calls should succeed with 0 interval
    int successCount = 0;
    for (int i = 0; i < 10; ++i) {
        if (throttler.tryInvoke()) {
            successCount++;
        }
    }

    EXPECT_EQ(successCount, 10);
}

// Thread-safe atomic
TEST_F(CallbackThrottlerTest, TryInvoke_ThreadSafe) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(20);
    std::atomic<int> totalSuccess{0};
    const int numThreads = 4;
    const int attemptsPerThread = 50;

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&throttler, &totalSuccess, attemptsPerThread]() {
            for (int i = 0; i < attemptsPerThread; ++i) {
                if (throttler.tryInvoke()) {
                    totalSuccess++;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Should have some successes, but not all attempts
    EXPECT_GT(totalSuccess.load(), 0);
    EXPECT_LT(totalSuccess.load(), numThreads * attemptsPerThread);
}

// ============================================================================
// 5.5 SetMinInterval Tests (5)
// ============================================================================

// Sets new interval
TEST_F(CallbackThrottlerTest, SetMinInterval_ValidValue) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(100);

    throttler.setMinInterval(200);

    EXPECT_EQ(throttler.getMinInterval(), 200);
}

// Zero valid
TEST_F(CallbackThrottlerTest, SetMinInterval_Zero) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(100);

    throttler.setMinInterval(0);

    EXPECT_EQ(throttler.getMinInterval(), 0);
}

// Negative throws
TEST_F(CallbackThrottlerTest, SetMinInterval_Negative) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(100);

    EXPECT_THROW({
        throttler.setMinInterval(-50);
    }, WheelLibException);

    try {
        throttler.setMinInterval(-1);
        FAIL() << "Expected WheelLibException";
    }
    catch (const WheelLibException& e) {
        EXPECT_EQ(e.getErrorCode(), ErrorCode::INVALID_ARGUMENT);
    }

    // Original value should be unchanged
    EXPECT_EQ(throttler.getMinInterval(), 100);
}

// Thread-safe set
TEST_F(CallbackThrottlerTest, SetMinInterval_ThreadSafe) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(100);
    const int numThreads = 4;
    const int setsPerThread = 50;

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&throttler, t, setsPerThread]() {
            for (int i = 0; i < setsPerThread; ++i) {
                EXPECT_NO_THROW(throttler.setMinInterval((t + 1) * 10 + i));
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Value should be one of the set values (no corruption)
    int finalInterval = throttler.getMinInterval();
    EXPECT_GE(finalInterval, 10);
    EXPECT_LT(finalInterval, 100);
}

// Affects next check
TEST_F(CallbackThrottlerTest, SetMinInterval_AffectsNextCheck) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(200);
    throttler.recordInvocation();

    // Initially false (200ms interval, just recorded)
    EXPECT_FALSE(throttler.shouldInvoke());

    // Change to 0ms interval
    throttler.setMinInterval(0);

    // Now should be true
    EXPECT_TRUE(throttler.shouldInvoke());
}

// ============================================================================
// 5.6 GetMinInterval Tests (3)
// ============================================================================

// Returns initial value
TEST_F(CallbackThrottlerTest, GetMinInterval_AfterConstruct) {
    if (!requireCuda()) return;

    CallbackThrottler throttler1(100);
    CallbackThrottler throttler2(500);
    CallbackThrottler throttler3;  // default

    EXPECT_EQ(throttler1.getMinInterval(), 100);
    EXPECT_EQ(throttler2.getMinInterval(), 500);
    EXPECT_EQ(throttler3.getMinInterval(), 100);  // default
}

// Returns updated value
TEST_F(CallbackThrottlerTest, GetMinInterval_AfterSet) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(100);

    throttler.setMinInterval(250);
    EXPECT_EQ(throttler.getMinInterval(), 250);

    throttler.setMinInterval(50);
    EXPECT_EQ(throttler.getMinInterval(), 50);

    throttler.setMinInterval(0);
    EXPECT_EQ(throttler.getMinInterval(), 0);
}

// Thread-safe get
TEST_F(CallbackThrottlerTest, GetMinInterval_ThreadSafe) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(100);
    std::atomic<bool> running{true};
    const int numReaders = 4;

    // Writer thread
    std::thread writer([&throttler, &running]() {
        int val = 100;
        while (running) {
            throttler.setMinInterval(val);
            val = (val + 10) % 1000;
            if (val == 0) val = 10;  // Avoid 0 to see changes
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    // Reader threads
    std::vector<std::thread> readers;
    for (int t = 0; t < numReaders; ++t) {
        readers.emplace_back([&throttler]() {
            for (int i = 0; i < 100; ++i) {
                int val = throttler.getMinInterval();
                EXPECT_GE(val, 0);
                EXPECT_LT(val, 1000);
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        });
    }

    for (auto& reader : readers) {
        reader.join();
    }

    running = false;
    writer.join();

    SUCCEED();
}

// ============================================================================
// 5.7 Integration Tests (3)
// ============================================================================

// Throttles rapid calls
TEST_F(CallbackThrottlerTest, Integration_ThrottleLoop) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(100);  // 100ms interval = ~10/second
    int invocationCount = 0;
    const int testDurationMs = 500;

    auto startTime = std::chrono::steady_clock::now();
    auto endTime = startTime + std::chrono::milliseconds(testDurationMs);

    while (std::chrono::steady_clock::now() < endTime) {
        if (throttler.tryInvoke()) {
            invocationCount++;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // With 100ms interval over 500ms, expect 5-6 invocations
    // Allow some tolerance for timing variations
    EXPECT_GE(invocationCount, 4);
    EXPECT_LE(invocationCount, 7);
}

// Handles burst traffic
TEST_F(CallbackThrottlerTest, Integration_BurstHandling) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(50);

    // Wait for initial interval so first call can succeed
    waitMs(60);

    int successCount = 0;
    const int burstSize = 100;

    // Rapid burst of calls
    for (int i = 0; i < burstSize; ++i) {
        if (throttler.tryInvoke()) {
            successCount++;
        }
    }

    // Only 1 should succeed in burst (too fast for interval)
    EXPECT_EQ(successCount, 1);

    // Wait and try again
    waitMs(60);

    if (throttler.tryInvoke()) {
        successCount++;
    }

    // Now should have 2 successes
    EXPECT_EQ(successCount, 2);
}

// Timing accuracy
TEST_F(CallbackThrottlerTest, Integration_RealTimeAccuracy) {
    if (!requireCuda()) return;

    CallbackThrottler throttler(100);
    std::vector<std::chrono::steady_clock::time_point> invocationTimes;

    // Collect invocation times over multiple intervals
    const int targetInvocations = 5;
    auto startTime = std::chrono::steady_clock::now();
    auto maxDuration = std::chrono::milliseconds(700);

    while (invocationTimes.size() < targetInvocations &&
           std::chrono::steady_clock::now() - startTime < maxDuration) {
        if (throttler.tryInvoke()) {
            invocationTimes.push_back(std::chrono::steady_clock::now());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Verify intervals between invocations
    ASSERT_GE(invocationTimes.size(), 3u);

    for (size_t i = 1; i < invocationTimes.size(); ++i) {
        auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(
            invocationTimes[i] - invocationTimes[i - 1]).count();

        // Allow 20% tolerance for timing variations
        EXPECT_GE(interval, 80);  // At least 80ms (100 - 20%)
        EXPECT_LE(interval, 150); // At most 150ms (accounting for sleep granularity)
    }
}
