#pragma once

#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include "../../Utils/Common/Types.h"
#include "../../Utils/Error/WheelLibException.h"

namespace WheelDL {
    namespace Core {
        namespace Callback {

            /**
             * @class AsyncCallbackQueue
             * @brief Thread-safe asynchronous callback queue
             *
             * This class processes callbacks on a separate thread to prevent
             * blocking the main training loop. Callbacks are invoked in the
             * order they were enqueued.
             */
            class AsyncCallbackQueue {
            public:
                /**
                 * @brief Constructor
                 * @param callback The callback function to invoke for each progress update
                 */
                explicit AsyncCallbackQueue(ProgressCallback callback);

                /**
                 * @brief Destructor - stops the processing thread
                 */
                ~AsyncCallbackQueue();

                // Non-copyable, non-movable
                AsyncCallbackQueue(const AsyncCallbackQueue&) = delete;
                AsyncCallbackQueue& operator=(const AsyncCallbackQueue&) = delete;
                AsyncCallbackQueue(AsyncCallbackQueue&&) = delete;
                AsyncCallbackQueue& operator=(AsyncCallbackQueue&&) = delete;

                /**
                 * @brief Add progress data to the queue for asynchronous processing
                 * @param data The progress data to enqueue
                 */
                void enqueue(const ProgressData& data);

                /**
                 * @brief Start the processing thread
                 */
                void start();

                /**
                 * @brief Stop the processing thread
                 */
                void stop();

                /**
                 * @brief Get the number of pending callbacks
                 * @return Number of items waiting to be processed
                 */
                size_t pendingCount() const;

            private:
                /**
                 * @brief Processing loop that runs in a separate thread
                 */
                void processingLoop();

                ProgressCallback _callback;
                std::queue<ProgressData> _dataQueue;
                std::thread _processingThread;
                mutable std::mutex _mutex;
                std::condition_variable _condition;
                std::atomic<bool> _shouldStop;
                bool _isRunning;  // Protected by _mutex, not atomic
            };

        } // namespace Callback
    } // namespace Core
} // namespace WheelDL
