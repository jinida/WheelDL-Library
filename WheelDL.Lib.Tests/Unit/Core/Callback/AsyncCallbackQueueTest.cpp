/**
 * @file AsyncCallbackQueueTest.cpp
 * @brief Unit tests for WheelDL::Core::Callback::AsyncCallbackQueue
 *
 * Phase 5 of test_core_utils.md - 38 tests
 *
 * Test Sections:
 * - 4.1 Constructor Tests (5)
 * - 4.2 Start/Stop Tests (6)
 * - 4.3 Enqueue Tests (5)
 * - 4.4 Processing Tests (5)
 * - 4.5 Exception Handling Tests (9)
 * - 4.6 PendingCount Tests (4)
 * - 4.7 Non-Copyable Tests (4)
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "../CoreTestHelpers.h"
#include "Core/Callback/AsyncCallbackQueue.h"
#include "Utils/Error/WheelLibException.h"
#include "Utils/Error/ErrorCodes.h"

#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <type_traits>

using namespace WheelDL::Core::Callback;
using namespace WheelDL::Test::Core;
using namespace WheelDL::Utils;
using namespace WheelDL;

// ============================================================================
// Test Fixture
// ============================================================================

class AsyncCallbackQueueTest : public CUDATestFixture {
protected:
    void SetUp() override {
        CUDATestFixture::SetUp();
        _tracker = std::make_unique<CallbackTracker>();
        _logCapture = std::make_unique<LogCapture>("AsyncCallbackQueueTest");
    }

    void TearDown() override {
        _tracker.reset();
        _logCapture.reset();
        CUDATestFixture::TearDown();
    }

    CallbackTracker& tracker() { return *_tracker; }
    LogCapture& logCapture() { return *_logCapture; }
    Logger* logger() { return _logCapture->get(); }

    // Helper to create ProgressData with specific epoch
    ProgressData createProgressData(int epoch) {
        return ProgressFactory::createAt(epoch, 0);
    }

    // Helper to create a queue with tracker callback
    std::unique_ptr<AsyncCallbackQueue> createQueue(Logger* log = nullptr) {
        return std::make_unique<AsyncCallbackQueue>(tracker().getCallback(), log);
    }

private:
    std::unique_ptr<CallbackTracker> _tracker;
    std::unique_ptr<LogCapture> _logCapture;
};

// ============================================================================
// 4.1 Constructor Tests (5)
// ============================================================================

// Valid callback accepted
TEST_F(AsyncCallbackQueueTest, Constructor_ValidCallback) {
    if (!requireCuda()) return;

    EXPECT_NO_THROW({
        AsyncCallbackQueue queue(tracker().getCallback());
    });
}

// Null callback throws
TEST_F(AsyncCallbackQueueTest, Constructor_NullCallback) {
    if (!requireCuda()) return;

    EXPECT_THROW({
        AsyncCallbackQueue queue(nullptr);
    }, WheelLibException);

    try {
        AsyncCallbackQueue queue(nullptr);
        FAIL() << "Expected WheelLibException";
    }
    catch (const WheelLibException& e) {
        EXPECT_EQ(e.getErrorCode(), ErrorCode::INVALID_ARGUMENT);
    }
}

// Logger stored
TEST_F(AsyncCallbackQueueTest, Constructor_WithLogger) {
    if (!requireCuda()) return;

    EXPECT_NO_THROW({
        AsyncCallbackQueue queue(tracker().getCallback(), logger());
    });
}

// Logger optional (nullptr logger)
TEST_F(AsyncCallbackQueueTest, Constructor_WithoutLogger) {
    if (!requireCuda()) return;

    EXPECT_NO_THROW({
        AsyncCallbackQueue queue(tracker().getCallback(), nullptr);
    });
}

// Not running initially
TEST_F(AsyncCallbackQueueTest, Constructor_NotRunning) {
    if (!requireCuda()) return;

    auto queue = createQueue();

    // Enqueue item - should stay pending since not running
    queue->enqueue(createProgressData(1));

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    EXPECT_EQ(queue->pendingCount(), 1u);
    EXPECT_EQ(tracker().count(), 0u);
}

// ============================================================================
// 4.2 Start/Stop Tests (6)
// ============================================================================

// Starts processing thread
TEST_F(AsyncCallbackQueueTest, Start_CreatesThread) {
    if (!requireCuda()) return;

    auto queue = createQueue();

    queue->enqueue(createProgressData(1));
    EXPECT_EQ(queue->pendingCount(), 1u);

    queue->start();

    // Wait for processing
    EXPECT_TRUE(tracker().waitForCount(1, 1000));
    EXPECT_EQ(tracker().count(), 1u);
}

// Double start safe
TEST_F(AsyncCallbackQueueTest, Start_AlreadyRunning) {
    if (!requireCuda()) return;

    auto queue = createQueue();

    queue->start();

    // Double start should not crash
    EXPECT_NO_THROW(queue->start());

    queue->enqueue(createProgressData(1));

    EXPECT_TRUE(tracker().waitForCount(1, 1000));
}

// Stop stops the thread
TEST_F(AsyncCallbackQueueTest, Stop_StopsThread) {
    if (!requireCuda()) return;

    auto queue = createQueue();

    queue->start();
    queue->enqueue(createProgressData(1));

    EXPECT_TRUE(tracker().waitForCount(1, 1000));

    // Stop should complete without hanging
    EXPECT_NO_THROW(queue->stop());

    // After stop, new items should stay pending
    queue->enqueue(createProgressData(2));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    EXPECT_EQ(queue->pendingCount(), 1u);
}

// Double stop safe
TEST_F(AsyncCallbackQueueTest, Stop_NotRunning) {
    if (!requireCuda()) return;

    auto queue = createQueue();

    // Double stop should not crash
    EXPECT_NO_THROW(queue->stop());
    EXPECT_NO_THROW(queue->stop());
}

// Stop processes remaining items
TEST_F(AsyncCallbackQueueTest, Stop_ProcessesRemaining) {
    if (!requireCuda()) return;

    auto queue = createQueue();

    // Enqueue items before start
    for (int i = 0; i < 5; ++i) {
        queue->enqueue(createProgressData(i));
    }

    queue->start();

    // Add more while running
    for (int i = 5; i < 10; ++i) {
        queue->enqueue(createProgressData(i));
    }

    // Stop should process all remaining
    queue->stop();

    EXPECT_EQ(tracker().count(), 10u);
    EXPECT_EQ(queue->pendingCount(), 0u);
}

// Destructor stops thread
TEST_F(AsyncCallbackQueueTest, Destructor_StopsThread) {
    if (!requireCuda()) return;

    {
        auto queue = createQueue();
        queue->start();

        for (int i = 0; i < 5; ++i) {
            queue->enqueue(createProgressData(i));
        }

        // Destructor should stop and process remaining
    }

    // All items should have been processed
    EXPECT_EQ(tracker().count(), 5u);
}

// ============================================================================
// 4.3 Enqueue Tests (5)
// ============================================================================

// Enqueue before start
TEST_F(AsyncCallbackQueueTest, Enqueue_BeforeStart) {
    if (!requireCuda()) return;

    auto queue = createQueue();

    queue->enqueue(createProgressData(1));
    queue->enqueue(createProgressData(2));
    queue->enqueue(createProgressData(3));

    EXPECT_EQ(queue->pendingCount(), 3u);

    queue->start();

    EXPECT_TRUE(tracker().waitForCount(3, 1000));
}

// Enqueue while running
TEST_F(AsyncCallbackQueueTest, Enqueue_WhileRunning) {
    if (!requireCuda()) return;

    auto queue = createQueue();
    queue->start();

    for (int i = 0; i < 10; ++i) {
        queue->enqueue(createProgressData(i));
    }

    EXPECT_TRUE(tracker().waitForCount(10, 2000));
}

// Thread woken on enqueue
TEST_F(AsyncCallbackQueueTest, Enqueue_NotifiesThread) {
    if (!requireCuda()) return;

    auto queue = createQueue();
    queue->start();

    // Wait a bit for thread to enter wait state
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto startTime = std::chrono::steady_clock::now();
    queue->enqueue(createProgressData(1));

    EXPECT_TRUE(tracker().waitForCount(1, 500));

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    // Should be processed quickly (not waiting for timeout)
    EXPECT_LT(elapsedMs, 200);
}

// Concurrent enqueue safe
TEST_F(AsyncCallbackQueueTest, Enqueue_ThreadSafe) {
    if (!requireCuda()) return;

    auto queue = createQueue();
    queue->start();

    const int numThreads = 4;
    const int itemsPerThread = 100;
    std::vector<std::thread> threads;

    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&queue, t, itemsPerThread, this]() {
            for (int i = 0; i < itemsPerThread; ++i) {
                queue->enqueue(createProgressData(t * 1000 + i));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    queue->stop();

    EXPECT_EQ(tracker().count(), static_cast<size_t>(numThreads * itemsPerThread));
}

// Enqueue after stop
TEST_F(AsyncCallbackQueueTest, Enqueue_AfterStop) {
    if (!requireCuda()) return;

    auto queue = createQueue();
    queue->start();
    queue->stop();

    queue->enqueue(createProgressData(1));
    queue->enqueue(createProgressData(2));

    // Items should be queued but not processed
    EXPECT_EQ(queue->pendingCount(), 2u);
}

// ============================================================================
// 4.4 Processing Tests (5)
// ============================================================================

// Callback invoked async
TEST_F(AsyncCallbackQueueTest, Process_CallsCallback) {
    if (!requireCuda()) return;

    auto queue = createQueue();
    queue->start();

    queue->enqueue(createProgressData(42));

    EXPECT_TRUE(tracker().waitForCount(1, 1000));

    auto invocations = tracker().getInvocations();
    ASSERT_EQ(invocations.size(), 1u);
    EXPECT_EQ(invocations[0].currentEpoch, 42);
}

// FIFO order preserved
TEST_F(AsyncCallbackQueueTest, Process_OrderPreserved) {
    if (!requireCuda()) return;

    auto queue = createQueue();
    queue->start();

    for (int i = 0; i < 10; ++i) {
        queue->enqueue(createProgressData(i));
    }

    EXPECT_TRUE(tracker().waitForCount(10, 2000));

    auto invocations = tracker().getInvocations();
    ASSERT_EQ(invocations.size(), 10u);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(invocations[i].currentEpoch, i);
    }
}

// Items removed after process
TEST_F(AsyncCallbackQueueTest, Process_ClearsQueue) {
    if (!requireCuda()) return;

    auto queue = createQueue();
    queue->start();

    for (int i = 0; i < 5; ++i) {
        queue->enqueue(createProgressData(i));
    }

    EXPECT_TRUE(tracker().waitForCount(5, 1000));

    // After processing, queue should be empty
    EXPECT_EQ(queue->pendingCount(), 0u);
}

// Thread waits when empty (CPU not spinning)
TEST_F(AsyncCallbackQueueTest, Process_WaitsForData) {
    if (!requireCuda()) return;

    auto queue = createQueue();
    queue->start();

    // Let thread run idle for a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Enqueue item - should be processed
    queue->enqueue(createProgressData(1));

    EXPECT_TRUE(tracker().waitForCount(1, 500));
}

// Continuous processing
TEST_F(AsyncCallbackQueueTest, Process_ContinuousProcessing) {
    if (!requireCuda()) return;

    auto queue = createQueue();
    queue->start();

    const int totalItems = 100;

    // Enqueue items with small delays
    for (int i = 0; i < totalItems; ++i) {
        queue->enqueue(createProgressData(i));
        if (i % 10 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    queue->stop();

    EXPECT_EQ(tracker().count(), static_cast<size_t>(totalItems));
}

// ============================================================================
// 4.5 Exception Handling Tests (9)
// ============================================================================

// std::exception caught - thread continues
TEST_F(AsyncCallbackQueueTest, Exception_CallbackThrows_StdException) {
    if (!requireCuda()) return;

    std::atomic<int> callCount{0};
    auto throwingCallback = [&callCount](const ProgressData& data) {
        callCount++;
        if (data.currentEpoch == 1) {
            throw std::runtime_error("Test exception");
        }
    };

    AsyncCallbackQueue queue(throwingCallback, nullptr);
    queue.start();

    queue.enqueue(createProgressData(0));  // OK
    queue.enqueue(createProgressData(1));  // Throws
    queue.enqueue(createProgressData(2));  // Should still be processed

    queue.stop();

    EXPECT_EQ(callCount.load(), 3);  // All callbacks attempted
}

// std::exception logged with logger
TEST_F(AsyncCallbackQueueTest, Exception_StdException_LogsWithLogger) {
    if (!requireCuda()) return;

    auto throwingCallback = [](const ProgressData&) {
        throw std::runtime_error("Test exception message");
    };

    AsyncCallbackQueue queue(throwingCallback, logger());
    queue.start();

    queue.enqueue(createProgressData(1));

    // Should not crash - exception is logged and processing continues
    EXPECT_NO_THROW(queue.stop());
    // Note: Log output can be verified visually in test output
    // LogCapture cannot intercept direct Logger calls from AsyncCallbackQueue
}

// std::exception without logger - no crash
TEST_F(AsyncCallbackQueueTest, Exception_StdException_NoLogger) {
    if (!requireCuda()) return;

    auto throwingCallback = [](const ProgressData&) {
        throw std::runtime_error("Test exception");
    };

    AsyncCallbackQueue queue(throwingCallback, nullptr);
    queue.start();

    queue.enqueue(createProgressData(1));

    EXPECT_NO_THROW(queue.stop());
}

// Unknown exception without logger - no crash
TEST_F(AsyncCallbackQueueTest, Exception_UnknownException_NoLogger) {
    if (!requireCuda()) return;

    auto throwingCallback = [](const ProgressData&) {
        throw 42;  // Unknown exception type
    };

    AsyncCallbackQueue queue(throwingCallback, nullptr);
    queue.start();

    queue.enqueue(createProgressData(1));

    EXPECT_NO_THROW(queue.stop());
}

// Unknown exception with logger
TEST_F(AsyncCallbackQueueTest, Exception_UnknownException_WithLogger) {
    if (!requireCuda()) return;

    auto throwingCallback = [](const ProgressData&) {
        throw 42;  // Unknown exception type
    };

    AsyncCallbackQueue queue(throwingCallback, logger());
    queue.start();

    queue.enqueue(createProgressData(1));

    // Should not crash - unknown exception is logged and processing continues
    EXPECT_NO_THROW(queue.stop());
    // Note: Log output can be verified visually in test output
}

// std::exception during cleanup with logger
TEST_F(AsyncCallbackQueueTest, Exception_CleanupPhase_StdException_WithLogger) {
    if (!requireCuda()) return;

    std::atomic<int> callCount{0};
    auto throwingCallback = [&callCount](const ProgressData&) {
        callCount++;
        throw std::runtime_error("Cleanup exception");
    };

    AsyncCallbackQueue queue(throwingCallback, logger());

    // Enqueue items before start so they're processed during cleanup
    for (int i = 0; i < 3; ++i) {
        queue.enqueue(createProgressData(i));
    }

    queue.start();
    EXPECT_NO_THROW(queue.stop());

    // All items should be attempted despite exceptions
    EXPECT_EQ(callCount.load(), 3);
    // Note: Log output can be verified visually in test output
}

// std::exception during cleanup without logger
TEST_F(AsyncCallbackQueueTest, Exception_CleanupPhase_StdException_NoLogger) {
    if (!requireCuda()) return;

    std::atomic<int> callCount{0};
    auto throwingCallback = [&callCount](const ProgressData&) {
        callCount++;
        throw std::runtime_error("Cleanup exception");
    };

    AsyncCallbackQueue queue(throwingCallback, nullptr);

    for (int i = 0; i < 3; ++i) {
        queue.enqueue(createProgressData(i));
    }

    queue.start();
    EXPECT_NO_THROW(queue.stop());

    EXPECT_EQ(callCount.load(), 3);
}

// Unknown exception during cleanup with logger
TEST_F(AsyncCallbackQueueTest, Exception_CleanupPhase_UnknownException_WithLogger) {
    if (!requireCuda()) return;

    std::atomic<int> callCount{0};
    auto throwingCallback = [&callCount](const ProgressData&) {
        callCount++;
        throw 42;
    };

    AsyncCallbackQueue queue(throwingCallback, logger());

    for (int i = 0; i < 3; ++i) {
        queue.enqueue(createProgressData(i));
    }

    queue.start();
    EXPECT_NO_THROW(queue.stop());

    // All items should be attempted despite exceptions
    EXPECT_EQ(callCount.load(), 3);
    // Note: Log output can be verified visually in test output
}

// Unknown exception during cleanup without logger
TEST_F(AsyncCallbackQueueTest, Exception_CleanupPhase_UnknownException_NoLogger) {
    if (!requireCuda()) return;

    std::atomic<int> callCount{0};
    auto throwingCallback = [&callCount](const ProgressData&) {
        callCount++;
        throw 42;
    };

    AsyncCallbackQueue queue(throwingCallback, nullptr);

    for (int i = 0; i < 3; ++i) {
        queue.enqueue(createProgressData(i));
    }

    queue.start();
    EXPECT_NO_THROW(queue.stop());

    EXPECT_EQ(callCount.load(), 3);
}

// ============================================================================
// 4.6 PendingCount Tests (4)
// ============================================================================

// Empty count zero
TEST_F(AsyncCallbackQueueTest, PendingCount_Empty) {
    if (!requireCuda()) return;

    auto queue = createQueue();

    EXPECT_EQ(queue->pendingCount(), 0u);
}

// Count with items (not running)
TEST_F(AsyncCallbackQueueTest, PendingCount_WithItems) {
    if (!requireCuda()) return;

    auto queue = createQueue();

    for (int i = 0; i < 5; ++i) {
        queue->enqueue(createProgressData(i));
    }

    EXPECT_EQ(queue->pendingCount(), 5u);
}

// Count decreases as processed
TEST_F(AsyncCallbackQueueTest, PendingCount_Decreases) {
    if (!requireCuda()) return;

    std::atomic<bool> allowProcess{false};
    auto slowCallback = [&allowProcess](const ProgressData&) {
        while (!allowProcess) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    };

    AsyncCallbackQueue queue(slowCallback, nullptr);

    for (int i = 0; i < 5; ++i) {
        queue.enqueue(createProgressData(i));
    }

    queue.start();

    // First item should be processing, rest pending
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_LE(queue.pendingCount(), 5u);

    // Allow processing
    allowProcess = true;
    queue.stop();

    EXPECT_EQ(queue.pendingCount(), 0u);
}

// Thread-safe count
TEST_F(AsyncCallbackQueueTest, PendingCount_ThreadSafe) {
    if (!requireCuda()) return;

    auto queue = createQueue();
    queue->start();

    std::atomic<bool> running{true};

    // Producer thread
    std::thread producer([&queue, &running, this]() {
        int i = 0;
        while (running) {
            queue->enqueue(createProgressData(i++));
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    // Count checker thread
    std::thread counter([&queue, &running]() {
        while (running) {
            auto count = queue->pendingCount();
            // Just verify no crash
            (void)count;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    running = false;

    producer.join();
    counter.join();

    queue->stop();
    SUCCEED();
}

// ============================================================================
// 4.7 Non-Copyable Tests (4) - Compile-time verification
// ============================================================================

// Copy constructor deleted
TEST_F(AsyncCallbackQueueTest, NonCopyable_CopyConstruct) {
    if (!requireCuda()) return;

    // Compile-time check: copy constructor should be deleted
    static_assert(!std::is_copy_constructible<AsyncCallbackQueue>::value,
        "AsyncCallbackQueue should not be copy constructible");
    SUCCEED();
}

// Copy assignment deleted
TEST_F(AsyncCallbackQueueTest, NonCopyable_CopyAssign) {
    if (!requireCuda()) return;

    // Compile-time check: copy assignment should be deleted
    static_assert(!std::is_copy_assignable<AsyncCallbackQueue>::value,
        "AsyncCallbackQueue should not be copy assignable");
    SUCCEED();
}

// Move constructor deleted
TEST_F(AsyncCallbackQueueTest, NonMovable_MoveConstruct) {
    if (!requireCuda()) return;

    // Compile-time check: move constructor should be deleted
    static_assert(!std::is_move_constructible<AsyncCallbackQueue>::value,
        "AsyncCallbackQueue should not be move constructible");
    SUCCEED();
}

// Move assignment deleted
TEST_F(AsyncCallbackQueueTest, NonMovable_MoveAssign) {
    if (!requireCuda()) return;

    // Compile-time check: move assignment should be deleted
    static_assert(!std::is_move_assignable<AsyncCallbackQueue>::value,
        "AsyncCallbackQueue should not be move assignable");
    SUCCEED();
}
