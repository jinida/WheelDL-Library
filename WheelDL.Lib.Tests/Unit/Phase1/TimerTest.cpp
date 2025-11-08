#include "pch.h"
#include "WheelDL.Lib/Utils/Common/Timer.h"
#include <thread>
#include <chrono>

using namespace WheelDL::Utils;

/**
 * @brief Timer basic functionality tests
 */
class TimerTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef _WIN32
        // Increase Windows timer resolution to 1ms for accurate sleep timing
        timeBeginPeriod(1);
#endif
    }

    void TearDown() override {
#ifdef _WIN32
        // Restore Windows timer resolution
        timeEndPeriod(1);
#endif
    }
};

TEST_F(TimerTest, Constructor) {
    // Timer should start immediately upon construction
    Timer timer;

    // Small delay to ensure some time has passed
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    double elapsed = timer.elapsedMilliseconds();
    EXPECT_GT(elapsed, 0.0);
    EXPECT_LT(elapsed, 50.0);  // Should be much less than 50ms
}

TEST_F(TimerTest, ElapsedMilliseconds) {
    Timer timer;

    // Sleep for a known duration
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    double elapsed = timer.elapsedMilliseconds();

    // Allow some tolerance for timing variations
    EXPECT_GE(elapsed, 14.0);  // At least 14ms
    EXPECT_LE(elapsed, 20.0);  // At most 20ms
}

TEST_F(TimerTest, ElapsedSeconds) {
    Timer timer;

    // Sleep for 50ms
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    double elapsed = timer.elapsedSeconds();

    // Should be around 0.05 seconds
    EXPECT_GE(elapsed, 0.045);  // At least 45ms
    EXPECT_LE(elapsed, 0.070);  // At most 70ms
}

TEST_F(TimerTest, ElapsedMicroseconds) {
    Timer timer;

    // Sleep for 10ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    double elapsed = timer.elapsedMicroseconds();

    // Should be around 10000 microseconds
    EXPECT_GE(elapsed, 9000.0);   // At least 9ms
    EXPECT_LE(elapsed, 15000.0);  // At most 15ms
}

TEST_F(TimerTest, Reset) {
    Timer timer;

    // Sleep and check elapsed time
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    double elapsed1 = timer.elapsedMilliseconds();
    EXPECT_GE(elapsed1, 18.0);

    // Reset timer
    timer.reset();

    // Immediately check - should be close to 0
    double elapsed2 = timer.elapsedMilliseconds();
    EXPECT_LT(elapsed2, 5.0);  // Should be very small

    // Sleep again and check
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    double elapsed3 = timer.elapsedMilliseconds();
    EXPECT_GE(elapsed3, 14.0);
    EXPECT_LE(elapsed3, 20.0);
}

TEST_F(TimerTest, MultipleReads) {
    Timer timer;

    // Sleep for 10ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    double elapsed1 = timer.elapsedMilliseconds();

    // Sleep for another 10ms
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    double elapsed2 = timer.elapsedMilliseconds();

    // Second reading should be larger
    EXPECT_GT(elapsed2, elapsed1);

    // Should be around 20ms total
    EXPECT_GE(elapsed2, 18.0);
    EXPECT_LE(elapsed2, 25.0);
}

TEST_F(TimerTest, Precision) {
    Timer timer;

    // Very short sleep (1ms)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    double elapsedMs = timer.elapsedMilliseconds();
    double elapsedUs = timer.elapsedMicroseconds();

    // Microseconds should be more precise
    EXPECT_GT(elapsedUs, 500.0);   // At least 0.5ms
    EXPECT_LT(elapsedUs, 5000.0);  // Less than 5ms

    // Milliseconds should match (within tolerance)
    EXPECT_NEAR(elapsedMs, elapsedUs / 1000.0, 0.1);
}

TEST_F(TimerTest, ConsistencyBetweenUnits) {
    Timer timer;

    // Sleep for 30ms
    std::this_thread::sleep_for(std::chrono::milliseconds(30));

    double ms = timer.elapsedMilliseconds();
    double sec = timer.elapsedSeconds();
    double us = timer.elapsedMicroseconds();

    // Check consistency: ms = sec * 1000 = us / 1000
    EXPECT_NEAR(ms, sec * 1000.0, 1.0);      // Within 1ms
    EXPECT_NEAR(ms, us / 1000.0, 1.0);       // Within 1ms
    EXPECT_NEAR(sec, us / 1000000.0, 0.001); // Within 1ms
}
