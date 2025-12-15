#include "pch.h"
#include "Utils/Common/Timer.h"
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>
#include <cmath>
#include <gtest/gtest.h>

using namespace WheelDL::Utils;

// =============================================================================
// Timer Basic Tests (TMR-001 ~ TMR-005)
// =============================================================================

// TMR-001: Timer starts automatically on construction
TEST(TimerTest, Constructor_StartsTimer) {
    Timer timer;

    // Wait a small amount of time
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    double elapsed = timer.elapsedMilliseconds();
    EXPECT_GT(elapsed, 0.0);
}

// TMR-002: elapsedMilliseconds returns positive value after construction
TEST(TimerTest, ElapsedMilliseconds_PositiveAfterConstruction) {
    Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    double ms = timer.elapsedMilliseconds();
    EXPECT_GE(ms, 45.0);  // Allow some tolerance
    EXPECT_LT(ms, 150.0); // Should not be too long
}

// TMR-003: elapsedSeconds returns correct conversion from milliseconds
TEST(TimerTest, ElapsedSeconds_CorrectConversion) {
    Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    double seconds = timer.elapsedSeconds();
    double milliseconds = timer.elapsedMilliseconds();

    // Seconds should be approximately milliseconds / 1000
    EXPECT_NEAR(seconds, milliseconds / 1000.0, 0.01);
}

// TMR-004: elapsedMicroseconds returns correct conversion
TEST(TimerTest, ElapsedMicroseconds_CorrectConversion) {
    Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    double microseconds = timer.elapsedMicroseconds();
    double milliseconds = timer.elapsedMilliseconds();

    // Microseconds should be approximately milliseconds * 1000
    EXPECT_NEAR(microseconds / 1000.0, milliseconds, 5.0);
}

// TMR-005: reset() resets elapsed time to near zero
TEST(TimerTest, Reset_ResetsElapsedTime) {
    Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    double beforeReset = timer.elapsedMilliseconds();
    EXPECT_GT(beforeReset, 50.0);

    timer.reset();
    double afterReset = timer.elapsedMilliseconds();

    EXPECT_LT(afterReset, 10.0);  // Should be very small after reset
}

// =============================================================================
// Timer Precision Tests (TMR-006 ~ TMR-007)
// =============================================================================

// TMR-006: Timer measures time accurately (within 20% tolerance for short durations)
TEST(TimerTest, Accuracy_ShortDuration) {
    Timer timer;

    const int targetMs = 100;
    std::this_thread::sleep_for(std::chrono::milliseconds(targetMs));

    double elapsed = timer.elapsedMilliseconds();

    // Allow 20% tolerance due to OS scheduling
    EXPECT_GE(elapsed, targetMs * 0.8);
    EXPECT_LE(elapsed, targetMs * 1.5);
}

// TMR-007: Multiple elapsed calls return increasing values
TEST(TimerTest, MultipleElapsedCalls_IncreasingValues) {
    Timer timer;

    std::vector<double> readings;
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        readings.push_back(timer.elapsedMilliseconds());
    }

    // Each reading should be greater than the previous
    for (size_t i = 1; i < readings.size(); ++i) {
        EXPECT_GT(readings[i], readings[i - 1]);
    }
}

// =============================================================================
// Timer Thread Safety Tests (TMR-008 ~ TMR-009)
// =============================================================================

// TMR-008: Thread-safe concurrent reads
TEST(TimerTest, ThreadSafety_ConcurrentReads) {
    Timer timer;
    std::atomic<int> readCount{0};
    std::atomic<bool> hasError{false};

    const int numThreads = 4;
    const int readsPerThread = 100;
    std::vector<std::thread> threads;

    // Start multiple threads that read elapsed time concurrently
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&timer, &readCount, &hasError, readsPerThread]() {
            for (int i = 0; i < readsPerThread; ++i) {
                double ms = timer.elapsedMilliseconds();
                if (ms < 0) {
                    hasError.store(true);
                }
                readCount.fetch_add(1);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_FALSE(hasError.load());
    EXPECT_EQ(numThreads * readsPerThread, readCount.load());
}

// TMR-009: Thread-safe concurrent read and reset
TEST(TimerTest, ThreadSafety_ConcurrentReadAndReset) {
    Timer timer;
    std::atomic<bool> running{true};
    std::atomic<bool> hasError{false};
    std::atomic<int> resetCount{0};
    std::atomic<int> readCount{0};

    // Reader thread
    std::thread reader([&]() {
        while (running.load()) {
            double ms = timer.elapsedMilliseconds();
            if (ms < 0) {
                hasError.store(true);
            }
            readCount.fetch_add(1);
            std::this_thread::yield();
        }
    });

    // Resetter thread
    std::thread resetter([&]() {
        while (running.load()) {
            timer.reset();
            resetCount.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    // Let them run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    running.store(false);

    reader.join();
    resetter.join();

    EXPECT_FALSE(hasError.load());
    EXPECT_GT(readCount.load(), 0);
    EXPECT_GT(resetCount.load(), 0);
}

// =============================================================================
// Timer Edge Cases
// =============================================================================

// TMR-010: Immediate elapsed time after construction is very small
TEST(TimerTest, ImmediateElapsed_VerySmall) {
    Timer timer;
    double elapsed = timer.elapsedMilliseconds();

    // Should be less than 10ms even with some overhead
    EXPECT_LT(elapsed, 10.0);
}

// TMR-011: All time units are consistent
TEST(TimerTest, TimeUnits_Consistent) {
    Timer timer;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    double microsec = timer.elapsedMicroseconds();
    double millisec = timer.elapsedMilliseconds();
    double sec = timer.elapsedSeconds();

    // Check consistency (allow for small timing differences between calls)
    EXPECT_NEAR(microsec / 1000.0, millisec, 5.0);
    EXPECT_NEAR(millisec / 1000.0, sec, 0.01);
}

// TMR-012: Timer can be reset multiple times
TEST(TimerTest, MultipleResets) {
    Timer timer;

    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        double beforeReset = timer.elapsedMilliseconds();
        EXPECT_GE(beforeReset, 15.0);

        timer.reset();
        double afterReset = timer.elapsedMilliseconds();
        EXPECT_LT(afterReset, 10.0);
    }
}
