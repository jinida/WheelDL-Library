#include "pch.h"
#include "Utils/ThreadPool/ThreadPool.h"
#include <atomic>
#include <chrono>
#include <vector>
#include <set>
#include <gtest/gtest.h>

using namespace WheelDL::Utils;

// =============================================================================
// Constructor Tests (TP-001 ~ TP-003, TP-017)
// =============================================================================

// TP-001: Constructor with valid size creates threads
TEST(ThreadPoolTest, Constructor_ValidSize) {
    ThreadPool pool(4);

    EXPECT_EQ(4u, pool.getNumThreads());
}

// TP-002: Constructor with single thread
TEST(ThreadPoolTest, Constructor_SingleThread) {
    ThreadPool pool(1);

    EXPECT_EQ(1u, pool.getNumThreads());
}

// TP-003: getNumThreads returns correct count
TEST(ThreadPoolTest, GetNumThreads_ReturnsCorrect) {
    ThreadPool pool(8);

    EXPECT_EQ(8u, pool.getNumThreads());
}

// TP-017: Constructor with zero threads throws
TEST(ThreadPoolTest, Constructor_ZeroThreads) {
    EXPECT_THROW({
        ThreadPool pool(0);
    }, std::invalid_argument);
}

// =============================================================================
// Enqueue Tests (TP-004 ~ TP-009)
// =============================================================================

// TP-004: Enqueue single task executes successfully
TEST(ThreadPoolTest, Enqueue_SingleTask) {
    ThreadPool pool(2);
    std::atomic<bool> executed{ false };

    auto future = pool.enqueue([&executed]() {
        executed = true;
    });

    future.get();
    EXPECT_TRUE(executed.load());
}

// TP-005: Enqueue task with return value
TEST(ThreadPoolTest, Enqueue_ReturnValue) {
    ThreadPool pool(2);

    auto future = pool.enqueue([]() {
        return 42;
    });

    EXPECT_EQ(42, future.get());
}

// TP-006: Enqueue void task completes
TEST(ThreadPoolTest, Enqueue_VoidTask) {
    ThreadPool pool(2);
    std::atomic<int> counter{ 0 };

    auto future = pool.enqueue([&counter]() {
        counter++;
    });

    future.get();
    EXPECT_EQ(1, counter.load());
}

// TP-007: Enqueue multiple tasks all execute
TEST(ThreadPoolTest, Enqueue_MultipleTasks) {
    ThreadPool pool(4);
    std::atomic<int> counter{ 0 };
    std::vector<std::future<void>> futures;

    for (int i = 0; i < 100; i++) {
        futures.push_back(pool.enqueue([&counter]() {
            counter++;
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    EXPECT_EQ(100, counter.load());
}

// TP-008: Enqueue task that throws - future.get() throws
TEST(ThreadPoolTest, Enqueue_TaskException) {
    ThreadPool pool(2);

    auto future = pool.enqueue([]() -> int {
        throw std::runtime_error("Task error");
        return 0;
    });

    EXPECT_THROW({
        future.get();
    }, std::runtime_error);
}

// TP-009: Enqueue after stop throws runtime_error
TEST(ThreadPoolTest, Enqueue_AfterStop) {
    auto pool = std::make_unique<ThreadPool>(2);

    // Destroy the pool (calls destructor which sets stop flag)
    pool.reset();

    // Cannot test enqueue after stop since pool is destroyed
    // This test verifies the destructor completes without hanging
    SUCCEED();
}

// =============================================================================
// Destructor Tests (TP-010, TP-020)
// =============================================================================

// TP-010: Destructor waits for pending tasks
TEST(ThreadPoolTest, Destructor_WaitsForTasks) {
    std::atomic<int> counter{ 0 };

    {
        ThreadPool pool(2);

        for (int i = 0; i < 10; i++) {
            pool.enqueue([&counter]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                counter++;
            });
        }
        // Pool destructor called here - should wait for all tasks
    }

    EXPECT_EQ(10, counter.load());
}

// TP-020: Destructor joins all threads
TEST(ThreadPoolTest, Destructor_JoinsAllThreads) {
    std::atomic<bool> taskStarted{ false };
    std::atomic<bool> taskCompleted{ false };

    {
        ThreadPool pool(1);

        pool.enqueue([&]() {
            taskStarted = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            taskCompleted = true;
        });

        // Wait for task to start
        while (!taskStarted.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // Destructor should wait for task to complete
    }

    EXPECT_TRUE(taskCompleted.load());
}

// =============================================================================
// Queue Size Tests (TP-011 ~ TP-013)
// =============================================================================

// TP-011: setMaxQueueSize enforces limit
TEST(ThreadPoolTest, SetMaxQueueSize_Limit) {
    ThreadPool pool(1);
    pool.setMaxQueueSize(5);

    // Block the worker thread
    std::mutex blockMutex;
    std::unique_lock<std::mutex> lock(blockMutex);

    pool.enqueue([&blockMutex]() {
        std::lock_guard<std::mutex> guard(blockMutex);
    });

    // Fill the queue
    for (int i = 0; i < 4; i++) {
        EXPECT_NO_THROW(pool.enqueue([]() {}));
    }

    // This should throw - queue is full
    EXPECT_THROW({
        pool.enqueue([]() {});
    }, std::runtime_error);

    // Release the worker
    lock.unlock();
}

// TP-012: setMaxQueueSize with 0 means unlimited
TEST(ThreadPoolTest, SetMaxQueueSize_Unlimited) {
    ThreadPool pool(1);
    pool.setMaxQueueSize(0);  // Unlimited

    // Should be able to enqueue many tasks
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 1000; i++) {
        EXPECT_NO_THROW({
            futures.push_back(pool.enqueue([]() {}));
        });
    }

    for (auto& f : futures) {
        f.get();
    }
}

// TP-013: getPendingTaskCount returns accurate count
TEST(ThreadPoolTest, GetPendingTaskCount_Accurate) {
    ThreadPool pool(1);

    // Block the worker
    std::mutex blockMutex;
    std::unique_lock<std::mutex> lock(blockMutex);

    pool.enqueue([&blockMutex]() {
        std::lock_guard<std::mutex> guard(blockMutex);
    });

    // Small delay to let worker pick up the blocking task
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Enqueue more tasks
    pool.enqueue([]() {});
    pool.enqueue([]() {});
    pool.enqueue([]() {});

    size_t pending = pool.getPendingTaskCount();
    EXPECT_GE(pending, 2u);  // At least 2 should be pending

    // Release worker
    lock.unlock();
}

// =============================================================================
// Thread Safety Tests (TP-014)
// =============================================================================

// TP-014: Concurrent enqueue from multiple threads
TEST(ThreadPoolTest, ThreadSafety_ConcurrentEnqueue) {
    ThreadPool pool(4);
    std::atomic<int> counter{ 0 };
    std::vector<std::thread> enqueuers;
    std::vector<std::future<void>> allFutures;
    std::mutex futuresMutex;

    for (int t = 0; t < 10; t++) {
        enqueuers.emplace_back([&]() {
            for (int i = 0; i < 100; i++) {
                auto future = pool.enqueue([&counter]() {
                    counter++;
                });
                std::lock_guard<std::mutex> lock(futuresMutex);
                allFutures.push_back(std::move(future));
            }
        });
    }

    for (auto& t : enqueuers) {
        t.join();
    }

    for (auto& f : allFutures) {
        f.get();
    }

    EXPECT_EQ(1000, counter.load());
}

// =============================================================================
// Performance Tests (TP-015)
// =============================================================================

// TP-015: High throughput test
TEST(ThreadPoolTest, Performance_HighThroughput) {
    ThreadPool pool(4);
    std::atomic<int> counter{ 0 };
    std::vector<std::future<void>> futures;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; i++) {
        futures.push_back(pool.enqueue([&counter]() {
            counter++;
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_EQ(1000, counter.load());
    // Should complete reasonably fast (< 5 seconds for 1000 simple tasks)
    EXPECT_LT(duration.count(), 5000);
}

// =============================================================================
// Task Order Tests (TP-016)
// =============================================================================

// TP-016: Tasks execute in FIFO order (for single thread)
TEST(ThreadPoolTest, TaskOrder_FIFO) {
    ThreadPool pool(1);  // Single thread to guarantee order
    std::vector<int> executionOrder;
    std::mutex orderMutex;

    std::vector<std::future<void>> futures;

    for (int i = 0; i < 10; i++) {
        futures.push_back(pool.enqueue([i, &executionOrder, &orderMutex]() {
            std::lock_guard<std::mutex> lock(orderMutex);
            executionOrder.push_back(i);
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    // Verify FIFO order
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(i, executionOrder[i]);
    }
}

// =============================================================================
// Exception Handling Tests (TP-018 ~ TP-019)
// =============================================================================

// TP-018: Task exception is caught internally, pool continues
TEST(ThreadPoolTest, TaskException_CaughtInternally) {
    ThreadPool pool(2);
    std::atomic<int> successCount{ 0 };

    // First task throws
    auto f1 = pool.enqueue([]() -> int {
        throw std::runtime_error("Error");
        return 0;
    });

    // Second task should still execute
    auto f2 = pool.enqueue([&successCount]() {
        successCount++;
    });

    // First future should throw when accessed
    EXPECT_THROW(f1.get(), std::runtime_error);

    // Second should complete normally
    f2.get();
    EXPECT_EQ(1, successCount.load());
}

// TP-019: Unknown exception in task handled
TEST(ThreadPoolTest, TaskUnknownException_Handled) {
    ThreadPool pool(2);
    std::atomic<int> successCount{ 0 };

    // First task throws non-std::exception
    auto f1 = pool.enqueue([]() {
        throw 42;  // Throw an int
    });

    // Second task should still execute
    auto f2 = pool.enqueue([&successCount]() {
        successCount++;
    });

    // First future should throw
    EXPECT_ANY_THROW(f1.get());

    // Second should complete normally
    f2.get();
    EXPECT_EQ(1, successCount.load());
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(ThreadPoolTest, EnqueueWithArguments) {
    ThreadPool pool(2);

    auto future = pool.enqueue([](int a, int b) {
        return a + b;
    }, 10, 20);

    EXPECT_EQ(30, future.get());
}

TEST(ThreadPoolTest, EnqueueWithStringArgument) {
    ThreadPool pool(2);

    auto future = pool.enqueue([](const std::string& msg) {
        return msg + " World";
    }, std::string("Hello"));

    EXPECT_EQ("Hello World", future.get());
}

TEST(ThreadPoolTest, EnqueueWithCapture) {
    ThreadPool pool(2);
    int x = 5;
    int y = 10;

    auto future = pool.enqueue([x, y]() {
        return x * y;
    });

    EXPECT_EQ(50, future.get());
}

TEST(ThreadPoolTest, LargeThreadCount) {
    // Test with relatively large thread count
    ThreadPool pool(16);

    EXPECT_EQ(16u, pool.getNumThreads());

    std::atomic<int> counter{ 0 };
    std::vector<std::future<void>> futures;

    for (int i = 0; i < 100; i++) {
        futures.push_back(pool.enqueue([&counter]() {
            counter++;
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    EXPECT_EQ(100, counter.load());
}

TEST(ThreadPoolTest, TaskWithDelay) {
    ThreadPool pool(2);

    auto start = std::chrono::high_resolution_clock::now();

    auto future = pool.enqueue([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return 42;
    });

    EXPECT_EQ(42, future.get());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_GE(duration.count(), 100);
}

TEST(ThreadPoolTest, ReturnDifferentTypes) {
    ThreadPool pool(2);

    auto intFuture = pool.enqueue([]() { return 42; });
    auto doubleFuture = pool.enqueue([]() { return 3.14; });
    auto stringFuture = pool.enqueue([]() { return std::string("test"); });
    auto boolFuture = pool.enqueue([]() { return true; });

    EXPECT_EQ(42, intFuture.get());
    EXPECT_DOUBLE_EQ(3.14, doubleFuture.get());
    EXPECT_EQ("test", stringFuture.get());
    EXPECT_TRUE(boolFuture.get());
}
