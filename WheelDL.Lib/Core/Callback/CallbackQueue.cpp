#include "pch.h"
#include "CallbackQueue.h"

namespace WheelDL {
    namespace Core {
        namespace Callback {

            void CallbackQueue::enqueue(const ProgressData& data)
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _queue.push(data);
            }

            void CallbackQueue::processAll(ProgressCallback callback)
            {
                if (!callback) return;

                std::lock_guard<std::mutex> lock(_mutex);

                while (!_queue.empty()) {
                    const auto& data = _queue.front();
                    callback(data);
                    _queue.pop();
                }
            }

            size_t CallbackQueue::size() const
            {
                std::lock_guard<std::mutex> lock(_mutex);
                return _queue.size();
            }

            bool CallbackQueue::empty() const
            {
                std::lock_guard<std::mutex> lock(_mutex);
                return _queue.empty();
            }

            void CallbackQueue::clear()
            {
                std::lock_guard<std::mutex> lock(_mutex);
                while (!_queue.empty()) {
                    _queue.pop();
                }
            }

        } // namespace Callback
    } // namespace Core
} // namespace WheelDL
