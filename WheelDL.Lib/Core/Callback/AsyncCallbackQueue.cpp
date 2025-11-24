#include "pch.h"
#include "AsyncCallbackQueue.h"
#include "../../Utils/Logger/Logger.h"

namespace WheelDL {
    namespace Core {
        namespace Callback {

            AsyncCallbackQueue::AsyncCallbackQueue(ProgressCallback callback)
                : _callback(callback)
                , _shouldStop(false)
                , _isRunning(false)
            {
                if (!_callback) {
                    throw Utils::WheelLibException(
                        Utils::ErrorCode::INVALID_ARGUMENT,
                        "Callback function cannot be null"
                    );
                }
            }

            AsyncCallbackQueue::~AsyncCallbackQueue()
            {
                stop();
            }

            void AsyncCallbackQueue::enqueue(const ProgressData& data)
            {
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _dataQueue.push(data);
                }

                // Notify the processing thread that new data is available
                _condition.notify_one();
            }

            void AsyncCallbackQueue::start()
            {
                std::lock_guard<std::mutex> lock(_mutex);

                if (_isRunning) {
                    // Already running
                    return;
                }

                _shouldStop = false;
                _isRunning = true;
                _processingThread = std::thread(&AsyncCallbackQueue::processingLoop, this);
            }

            void AsyncCallbackQueue::stop()
            {
                std::unique_lock<std::mutex> lock(_mutex);

                if (!_isRunning) {
                    // Not running
                    return;
                }

                // Signal the thread to stop
                _shouldStop = true;
                _isRunning = false;

                // Unlock before notifying to allow the thread to proceed
                lock.unlock();
                _condition.notify_all();

                // Wait for the thread to finish
                if (_processingThread.joinable()) {
                    _processingThread.join();
                }
            }

            size_t AsyncCallbackQueue::pendingCount() const
            {
                std::lock_guard<std::mutex> lock(_mutex);
                return _dataQueue.size();
            }

            void AsyncCallbackQueue::processingLoop()
            {
                while (!_shouldStop) {
                    ProgressData data;
                    bool hasData = false;

                    {
                        std::unique_lock<std::mutex> lock(_mutex);

                        // Wait for data or stop signal
                        _condition.wait(lock, [this] {
                            return !_dataQueue.empty() || _shouldStop;
                        });

                        // Check if we should stop
                        if (_shouldStop && _dataQueue.empty()) {
                            break;
                        }

                        // Get data from queue
                        if (!_dataQueue.empty()) {
                            data = _dataQueue.front();
                            _dataQueue.pop();
                            hasData = true;
                        }
                    }

                    // Process the callback outside the lock to avoid blocking
                    if (hasData) {
                        try {
                            _callback(data);
                        }
                        catch (const std::exception& e) {
                            // Log error but continue processing
                            auto logger = Utils::Logger::getInstance();
                            logger->error("AsyncCallbackQueue",
                                         "Callback exception: " + std::string(e.what()));
                        }
                        catch (...) {
                            // Unknown exception, continue processing
                            auto logger = Utils::Logger::getInstance();
                            logger->error("AsyncCallbackQueue",
                                         "Unknown callback exception");
                        }
                    }
                }

                // Process any remaining items before exiting
                std::lock_guard<std::mutex> lock(_mutex);
                while (!_dataQueue.empty()) {
                    try {
                        _callback(_dataQueue.front());
                    }
                    catch (...) {
                        // Ignore exceptions during cleanup
                    }
                    _dataQueue.pop();
                }
            }

        } // namespace Callback
    } // namespace Core
} // namespace WheelDL
