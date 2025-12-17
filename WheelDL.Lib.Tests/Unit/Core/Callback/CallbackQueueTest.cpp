/**
 * @file CallbackQueueTest.cpp
 * @brief Unit tests for WheelDL::Core::Callback::CallbackQueue
 *
 * Phase 4 of test_core_utils.md - 23 tests
 *
 * Test Sections:
 * - 3.1 Enqueue Tests (5)
 * - 3.2 ProcessAll Tests (6)
 * - 3.3 Size/Empty Tests (6)
 * - 3.4 Clear Tests (3)
 * - 3.5 Thread Safety Tests (3)
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "../CoreTestHelpers.h"
#include "Core/Callback/CallbackQueue.h"

#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

using namespace WheelDL::Core::Callback;
using namespace WheelDL::Test::Core;
using namespace WheelDL;

// ============================================================================
// Test Fixture
// ============================================================================

class CallbackQueueTest : public CUDATestFixture {
protected:
    void SetUp() override {
        CUDATestFixture::SetUp();
        _queue = std::make_unique<CallbackQueue>();
        _tracker = std::make_unique<CallbackTracker>();
    }

    void TearDown() override {
        _queue.reset();
        _tracker.reset();
        CUDATestFixture::TearDown();
    }

    CallbackQueue& queue() { return *_queue; }
    CallbackTracker& tracker() { return *_tracker; }

    // Helper to create ProgressData with specific epoch
    ProgressData createProgressData(int epoch) {
        return ProgressFactory::createAt(epoch, 0);
    }

private:
    std::unique_ptr<CallbackQueue> _queue;
    std::unique_ptr<CallbackTracker> _tracker;
};

// ============================================================================
// 3.1 Enqueue Tests (5)
// ============================================================================

// Enqueue single item
TEST_F(CallbackQueueTest, Enqueue_Single) {
    if (!requireCuda()) return;

    auto data = createProgressData(1);
    queue().enqueue(data);

    EXPECT_EQ(queue().size(), 1u);
}

// Enqueue multiple items
TEST_F(CallbackQueueTest, Enqueue_Multiple) {
    if (!requireCuda()) return;

    queue().enqueue(createProgressData(1));
    queue().enqueue(createProgressData(2));
    queue().enqueue(createProgressData(3));

    EXPECT_EQ(queue().size(), 3u);
}

// Thread-safe enqueue
TEST_F(CallbackQueueTest, Enqueue_ThreadSafe) {
    if (!requireCuda()) return;

    const int numThreads = 4;
    const int itemsPerThread = 100;
    std::vector<std::thread> threads;

    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([this, t, itemsPerThread]() {
            for (int i = 0; i < itemsPerThread; ++i) {
                queue().enqueue(createProgressData(t * itemsPerThread + i));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(queue().size(), static_cast<size_t>(numThreads * itemsPerThread));
}

// FIFO order preserved
TEST_F(CallbackQueueTest, Enqueue_PreservesOrder) {
    if (!requireCuda()) return;

    queue().enqueue(createProgressData(1));
    queue().enqueue(createProgressData(2));
    queue().enqueue(createProgressData(3));

    std::vector<int> processedEpochs;
    queue().processAll([&processedEpochs](const ProgressData& data) {
        processedEpochs.push_back(data.currentEpoch);
    });

    ASSERT_EQ(processedEpochs.size(), 3u);
    EXPECT_EQ(processedEpochs[0], 1);
    EXPECT_EQ(processedEpochs[1], 2);
    EXPECT_EQ(processedEpochs[2], 3);
}

// Data is copied
TEST_F(CallbackQueueTest, Enqueue_CopiesData) {
    if (!requireCuda()) return;

    ProgressData data = createProgressData(42);
    queue().enqueue(data);

    // Modify original
    data.currentEpoch = 999;

    // Process and verify original value was preserved
    int processedEpoch = -1;
    queue().processAll([&processedEpoch](const ProgressData& d) {
        processedEpoch = d.currentEpoch;
    });

    EXPECT_EQ(processedEpoch, 42);
}

// ============================================================================
// 3.2 ProcessAll Tests (6)
// ============================================================================

// Callback invoked for each item
TEST_F(CallbackQueueTest, ProcessAll_CallsCallback) {
    if (!requireCuda()) return;

    queue().enqueue(createProgressData(1));
    queue().enqueue(createProgressData(2));
    queue().enqueue(createProgressData(3));

    queue().processAll(tracker().getCallback());

    EXPECT_EQ(tracker().count(), 3u);
}

// Queue empty after process
TEST_F(CallbackQueueTest, ProcessAll_ClearsQueue) {
    if (!requireCuda()) return;

    queue().enqueue(createProgressData(1));
    queue().enqueue(createProgressData(2));

    queue().processAll(tracker().getCallback());

    EXPECT_TRUE(queue().empty());
    EXPECT_EQ(queue().size(), 0u);
}

// Null callback safe
TEST_F(CallbackQueueTest, ProcessAll_NullCallback) {
    if (!requireCuda()) return;

    queue().enqueue(createProgressData(1));
    queue().enqueue(createProgressData(2));

    // Null callback should not crash
    EXPECT_NO_THROW(queue().processAll(nullptr));

    // Queue should be unchanged when callback is null
    EXPECT_EQ(queue().size(), 2u);
}

// Process empty queue
TEST_F(CallbackQueueTest, ProcessAll_EmptyQueue) {
    if (!requireCuda()) return;

    EXPECT_TRUE(queue().empty());

    queue().processAll(tracker().getCallback());

    EXPECT_EQ(tracker().count(), 0u);
}

// FIFO order in callbacks
TEST_F(CallbackQueueTest, ProcessAll_OrderPreserved) {
    if (!requireCuda()) return;

    for (int i = 0; i < 10; ++i) {
        queue().enqueue(createProgressData(i));
    }

    std::vector<int> order;
    queue().processAll([&order](const ProgressData& data) {
        order.push_back(data.currentEpoch);
    });

    ASSERT_EQ(order.size(), 10u);
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(order[i], i);
    }
}

// Thread-safe processing
TEST_F(CallbackQueueTest, ProcessAll_ThreadSafe) {
    if (!requireCuda()) return;

    // Fill queue
    for (int i = 0; i < 100; ++i) {
        queue().enqueue(createProgressData(i));
    }

    std::atomic<int> processedCount{0};

    // Process from multiple threads
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([this, &processedCount]() {
            queue().processAll([&processedCount](const ProgressData&) {
                processedCount++;
            });
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // All items should be processed exactly once (only one thread gets items)
    EXPECT_EQ(processedCount.load(), 100);
    EXPECT_TRUE(queue().empty());
}

// ============================================================================
// 3.3 Size/Empty Tests (6)
// ============================================================================

// Empty queue size
TEST_F(CallbackQueueTest, Size_Empty) {
    if (!requireCuda()) return;

    EXPECT_EQ(queue().size(), 0u);
}

// Queue with items
TEST_F(CallbackQueueTest, Size_WithItems) {
    if (!requireCuda()) return;

    for (int i = 0; i < 5; ++i) {
        queue().enqueue(createProgressData(i));
    }

    EXPECT_EQ(queue().size(), 5u);
}

// Size after processing
TEST_F(CallbackQueueTest, Size_AfterProcess) {
    if (!requireCuda()) return;

    for (int i = 0; i < 10; ++i) {
        queue().enqueue(createProgressData(i));
    }

    queue().processAll(tracker().getCallback());

    EXPECT_EQ(queue().size(), 0u);
}

// New queue is empty
TEST_F(CallbackQueueTest, Empty_NewQueue) {
    if (!requireCuda()) return;

    CallbackQueue newQueue;
    EXPECT_TRUE(newQueue.empty());
}

// Queue not empty with items
TEST_F(CallbackQueueTest, Empty_WithItems) {
    if (!requireCuda()) return;

    queue().enqueue(createProgressData(1));

    EXPECT_FALSE(queue().empty());
}

// Empty after clear
TEST_F(CallbackQueueTest, Empty_AfterClear) {
    if (!requireCuda()) return;

    queue().enqueue(createProgressData(1));
    queue().enqueue(createProgressData(2));

    queue().clear();

    EXPECT_TRUE(queue().empty());
}

// ============================================================================
// 3.4 Clear Tests (3)
// ============================================================================

// Clear removes all items
TEST_F(CallbackQueueTest, Clear_RemovesAll) {
    if (!requireCuda()) return;

    for (int i = 0; i < 10; ++i) {
        queue().enqueue(createProgressData(i));
    }

    EXPECT_EQ(queue().size(), 10u);

    queue().clear();

    EXPECT_TRUE(queue().empty());
    EXPECT_EQ(queue().size(), 0u);
}

// Clear empty queue safe
TEST_F(CallbackQueueTest, Clear_EmptyQueue) {
    if (!requireCuda()) return;

    EXPECT_TRUE(queue().empty());

    EXPECT_NO_THROW(queue().clear());

    EXPECT_TRUE(queue().empty());
}

// Thread-safe clear
TEST_F(CallbackQueueTest, Clear_ThreadSafe) {
    if (!requireCuda()) return;

    // Producer thread
    std::atomic<bool> running{true};
    std::thread producer([this, &running]() {
        int i = 0;
        while (running) {
            queue().enqueue(createProgressData(i++));
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    });

    // Clear multiple times from main thread
    for (int i = 0; i < 10; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        EXPECT_NO_THROW(queue().clear());
    }

    running = false;
    producer.join();

    // Final state should be consistent
    auto finalSize = queue().size();
    EXPECT_GE(finalSize, 0u);  // Just ensure no crash
}

// ============================================================================
// 3.5 Thread Safety Tests (3)
// ============================================================================

// Enqueue during process (no deadlock)
TEST_F(CallbackQueueTest, ThreadSafe_EnqueueWhileProcess) {
    if (!requireCuda()) return;

    // Fill initial items
    for (int i = 0; i < 50; ++i) {
        queue().enqueue(createProgressData(i));
    }

    std::atomic<bool> processingDone{false};

    // Process thread (slow)
    std::thread processor([this, &processingDone]() {
        queue().processAll([](const ProgressData&) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        });
        processingDone = true;
    });

    // Enqueue thread (concurrent)
    std::thread enqueuer([this, &processingDone]() {
        int i = 100;
        while (!processingDone) {
            queue().enqueue(createProgressData(i++));
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    });

    processor.join();
    enqueuer.join();

    // No deadlock, just verify completion
    SUCCEED();
}

// Multiple producers
TEST_F(CallbackQueueTest, ThreadSafe_MultipleProducers) {
    if (!requireCuda()) return;

    const int numProducers = 4;
    const int itemsPerProducer = 250;
    std::vector<std::thread> producers;

    for (int p = 0; p < numProducers; ++p) {
        producers.emplace_back([this, p, itemsPerProducer]() {
            for (int i = 0; i < itemsPerProducer; ++i) {
                queue().enqueue(createProgressData(p * 1000 + i));
            }
        });
    }

    for (auto& producer : producers) {
        producer.join();
    }

    EXPECT_EQ(queue().size(), static_cast<size_t>(numProducers * itemsPerProducer));
}

// Size accurate under load
TEST_F(CallbackQueueTest, ThreadSafe_SizeAccuracy) {
    if (!requireCuda()) return;

    const int numItems = 1000;
    std::atomic<int> enqueued{0};
    std::atomic<bool> done{false};

    // Producer thread
    std::thread producer([this, &enqueued, &done, numItems]() {
        for (int i = 0; i < numItems; ++i) {
            queue().enqueue(createProgressData(i));
            enqueued++;
        }
        done = true;
    });

    // Size checker thread
    std::thread sizeChecker([this, &done]() {
        while (!done) {
            auto s = queue().size();
            // Size should always be non-negative (unsigned)
            EXPECT_GE(s, 0u);
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    });

    producer.join();
    sizeChecker.join();

    // Final size should match enqueued count
    EXPECT_EQ(queue().size(), static_cast<size_t>(numItems));
}
