#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Core/Callback/AsyncCallbackQueue.h"
#include <atomic>
#include <thread>
#include <chrono>

using namespace WheelDL::Core::Callback;

class AsyncCallbackQueueTest : public ::testing::Test {
protected:
    void SetUp() override {
        callbackCount = 0;
    }

    std::atomic<int> callbackCount;
};

TEST_F(AsyncCallbackQueueTest, ConstructorThrowsOnNullCallback) {
    EXPECT_THROW(
        AsyncCallbackQueue(nullptr),
        std::invalid_argument
    );
}

TEST_F(AsyncCallbackQueueTest, StartAndStop) {
    auto callback = [](const WheelDL::ProgressData&) {};
    AsyncCallbackQueue queue(callback);

    EXPECT_NO_THROW(queue.start());
    EXPECT_NO_THROW(queue.stop());
}

TEST_F(AsyncCallbackQueueTest, EnqueueBeforeStart) {
    auto callback = [this](const WheelDL::ProgressData&) {
        callbackCount++;
    };

    AsyncCallbackQueue queue(callback);

    WheelDL::ProgressData data;
    data.currentEpoch = 1;

    // Enqueue before starting
    queue.enqueue(data);
    queue.enqueue(data);

    EXPECT_EQ(queue.pendingCount(), 2);

    // Start and wait for processing
    queue.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Should have processed both items
    EXPECT_EQ(callbackCount, 2);
    EXPECT_EQ(queue.pendingCount(), 0);

    queue.stop();
}

TEST_F(AsyncCallbackQueueTest, EnqueueAfterStart) {
    auto callback = [this](const WheelDL::ProgressData&) {
        callbackCount++;
    };

    AsyncCallbackQueue queue(callback);
    queue.start();

    WheelDL::ProgressData data;
    data.currentEpoch = 1;

    queue.enqueue(data);
    queue.enqueue(data);
    queue.enqueue(data);

    // Wait for async processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_EQ(callbackCount, 3);
    EXPECT_EQ(queue.pendingCount(), 0);

    queue.stop();
}

TEST_F(AsyncCallbackQueueTest, ProcessingOrder) {
    std::vector<int> processedEpochs;
    std::mutex mutex;

    auto callback = [&processedEpochs, &mutex](const WheelDL::ProgressData& data) {
        std::lock_guard<std::mutex> lock(mutex);
        processedEpochs.push_back(data.currentEpoch);
    };

    AsyncCallbackQueue queue(callback);
    queue.start();

    for (int i = 0; i < 10; ++i) {
        WheelDL::ProgressData data;
        data.currentEpoch = i;
        queue.enqueue(data);
    }

    // Wait for all items to be processed
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    EXPECT_EQ(processedEpochs.size(), 10);

    // Verify order is preserved
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(processedEpochs[i], i);
    }

    queue.stop();
}

TEST_F(AsyncCallbackQueueTest, StopProcessesRemainingItems) {
    std::atomic<int> processedCount{0};

    auto callback = [&processedCount](const WheelDL::ProgressData&) {
        processedCount++;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    };

    AsyncCallbackQueue queue(callback);
    queue.start();

    for (int i = 0; i < 5; ++i) {
        WheelDL::ProgressData data;
        queue.enqueue(data);
    }

    // Stop immediately - should still process all items
    queue.stop();

    EXPECT_EQ(processedCount, 5);
}

TEST_F(AsyncCallbackQueueTest, MultipleStartCallsAreIdempotent) {
    auto callback = [](const WheelDL::ProgressData&) {};
    AsyncCallbackQueue queue(callback);

    EXPECT_NO_THROW(queue.start());
    EXPECT_NO_THROW(queue.start());  // Should not throw or create duplicate thread
    EXPECT_NO_THROW(queue.start());

    queue.stop();
}

TEST_F(AsyncCallbackQueueTest, MultipleStopCallsAreIdempotent) {
    auto callback = [](const WheelDL::ProgressData&) {};
    AsyncCallbackQueue queue(callback);

    queue.start();

    EXPECT_NO_THROW(queue.stop());
    EXPECT_NO_THROW(queue.stop());  // Should not throw
    EXPECT_NO_THROW(queue.stop());
}

TEST_F(AsyncCallbackQueueTest, CallbackExceptionHandling) {
    std::atomic<int> successCount{0};

    auto callback = [&successCount](const WheelDL::ProgressData& data) {
        if (data.currentEpoch == 2) {
            throw std::runtime_error("Test exception");
        }
        successCount++;
    };

    AsyncCallbackQueue queue(callback);
    queue.start();

    for (int i = 0; i < 5; ++i) {
        WheelDL::ProgressData data;
        data.currentEpoch = i;
        queue.enqueue(data);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Should have processed all except the one that threw
    EXPECT_EQ(successCount, 4);

    queue.stop();
}

TEST_F(AsyncCallbackQueueTest, DestructorStopsThread) {
    std::atomic<bool> callbackInvoked{false};

    {
        auto callback = [&callbackInvoked](const WheelDL::ProgressData&) {
            callbackInvoked = true;
        };

        AsyncCallbackQueue queue(callback);
        queue.start();

        WheelDL::ProgressData data;
        queue.enqueue(data);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // Destructor called here
    }

    // Should have processed the item before destruction
    EXPECT_TRUE(callbackInvoked);
}

TEST_F(AsyncCallbackQueueTest, HighVolumeThroughput) {
    std::atomic<int> processedCount{0};
    const int itemCount = 1000;

    auto callback = [&processedCount](const WheelDL::ProgressData&) {
        processedCount++;
    };

    AsyncCallbackQueue queue(callback);
    queue.start();

    for (int i = 0; i < itemCount; ++i) {
        WheelDL::ProgressData data;
        data.currentEpoch = i;
        queue.enqueue(data);
    }

    // Wait for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    EXPECT_EQ(processedCount, itemCount);

    queue.stop();
}
