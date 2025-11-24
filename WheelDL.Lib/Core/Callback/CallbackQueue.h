#pragma once

#include <queue>
#include <mutex>
#include "../../Utils/Common/Types.h"

namespace WheelDL {
    namespace Core {
        namespace Callback {

            /**
             * @class CallbackQueue
             * @brief Synchronous callback queue for progress updates
             *
             * This is a simple synchronous version kept for reference.
             * For most use cases, prefer AsyncCallbackQueue which processes
             * callbacks on a separate thread.
             */
            class CallbackQueue {
            public:
                /**
                 * @brief Constructor
                 */
                CallbackQueue() = default;

                /**
                 * @brief Destructor
                 */
                ~CallbackQueue() = default;

                /**
                 * @brief Enqueue a progress data update
                 * @param data Progress data to enqueue
                 */
                void enqueue(const ProgressData& data);

                /**
                 * @brief Process all queued items with the given callback
                 * @param callback Callback function to invoke for each item
                 */
                void processAll(ProgressCallback callback);

                /**
                 * @brief Get the current queue size
                 * @return Number of items in the queue
                 */
                size_t size() const;

                /**
                 * @brief Check if the queue is empty
                 * @return true if empty, false otherwise
                 */
                bool empty() const;

                /**
                 * @brief Clear all items from the queue
                 */
                void clear();

            private:
                std::queue<ProgressData> _queue;
                mutable std::mutex _mutex;
            };

        } // namespace Callback
    } // namespace Core
} // namespace WheelDL
