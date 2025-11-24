#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Core/Callback/CallbackQueue.h"
#include <vector>
#include <thread>

using namespace WheelDL::Core::Callback;

class CallbackQueueTest : public ::testing::Test {
protected:
    CallbackQueue queue;
};

TEST_F(CallbackQueueTest, InitiallyEmpty) {
    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);
}

TEST_F(CallbackQueueTest, EnqueueSingleItem) {
    WheelDL::ProgressData data;
    data.stage = WheelDL::ProgressStage::TRAIN_BATCH;
    data.currentEpoch = 1;
    data.totalEpochs = 10;

    queue.enqueue(data);

    EXPECT_FALSE(queue.empty());
    EXPECT_EQ(queue.size(), 1);
}

TEST_F(CallbackQueueTest, EnqueueMultipleItems) {
    for (int i = 0; i < 5; ++i) {
        WheelDL::ProgressData data;
        data.currentEpoch = i;
        queue.enqueue(data);
    }

    EXPECT_EQ(queue.size(), 5);
}

TEST_F(CallbackQueueTest, ProcessAllInvokesCallback) {
    std::vector<int> processedEpochs;

    for (int i = 0; i < 3; ++i) {
        WheelDL::ProgressData data;
        data.currentEpoch = i;
        queue.enqueue(data);
    }

    queue.processAll([&processedEpochs](const WheelDL::ProgressData& data) {
        processedEpochs.push_back(data.currentEpoch);
    });

    EXPECT_EQ(processedEpochs.size(), 3);
    EXPECT_EQ(processedEpochs[0], 0);
    EXPECT_EQ(processedEpochs[1], 1);
    EXPECT_EQ(processedEpochs[2], 2);
}

TEST_F(CallbackQueueTest, ProcessAllClearsQueue) {
    for (int i = 0; i < 3; ++i) {
        WheelDL::ProgressData data;
        queue.enqueue(data);
    }

    queue.processAll([](const WheelDL::ProgressData&) {
        // Do nothing
    });

    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);
}

TEST_F(CallbackQueueTest, ProcessAllThrowsOnNullCallback) {
    WheelDL::ProgressData data;
    queue.enqueue(data);

    EXPECT_THROW(queue.processAll(nullptr), std::invalid_argument);
}

TEST_F(CallbackQueueTest, Clear) {
    for (int i = 0; i < 5; ++i) {
        WheelDL::ProgressData data;
        queue.enqueue(data);
    }

    EXPECT_EQ(queue.size(), 5);

    queue.clear();

    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);
}

TEST_F(CallbackQueueTest, ThreadSafety) {
    const int numThreads = 4;
    const int itemsPerThread = 100;

    std::vector<std::thread> threads;

    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([this, itemsPerThread]() {
            for (int i = 0; i < itemsPerThread; ++i) {
                WheelDL::ProgressData data;
                data.currentEpoch = i;
                queue.enqueue(data);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(queue.size(), numThreads * itemsPerThread);
}

TEST_F(CallbackQueueTest, ConcurrentProcessAndEnqueue) {
    std::atomic<int> processedCount{0};

    std::thread producer([this]() {
        for (int i = 0; i < 50; ++i) {
            WheelDL::ProgressData data;
            queue.enqueue(data);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    std::thread consumer([this, &processedCount]() {
        for (int i = 0; i < 10; ++i) {
            queue.processAll([&processedCount](const WheelDL::ProgressData&) {
                processedCount++;
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    producer.join();
    consumer.join();

    // Process any remaining items
    queue.processAll([&processedCount](const WheelDL::ProgressData&) {
        processedCount++;
    });

    EXPECT_EQ(processedCount, 50);
    EXPECT_TRUE(queue.empty());
}
