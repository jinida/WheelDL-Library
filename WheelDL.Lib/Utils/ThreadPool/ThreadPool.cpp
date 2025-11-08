#include "pch.h"
#include "ThreadPool.h"
#include <iostream>

namespace WheelDL {
namespace Utils {

ThreadPool::ThreadPool(size_t threads)
    : _stop(false)
    , _maxQueueSize(0)  // Default: unlimited queue size
{
    // Validate thread count
    if (threads == 0) {
        throw std::invalid_argument("ThreadPool requires at least 1 thread");
    }

    // Create worker threads using member function instead of lambda
    for (size_t i = 0; i < threads; ++i) {
        _workers.emplace_back(&ThreadPool::workerThreadFunction, this);
    }
}

// Worker thread function (extracted from lambda for better readability)
void ThreadPool::workerThreadFunction() {
    // Worker thread loop
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(this->_queueMutex);

            // Wait for task or stop signal
            this->_condition.wait(lock, [this] {
                return this->_stop.load(std::memory_order_acquire) || !this->_tasks.empty();
            });

            // Exit if stop signal and no more tasks
            if (this->_stop.load(std::memory_order_acquire) && this->_tasks.empty()) {
                return;
            }

            // Safety check: ensure queue is not empty
            if (this->_tasks.empty()) {
                continue;
            }

            // Get next task
            task = std::move(this->_tasks.front());
            this->_tasks.pop();
        }

        // Execute task with exception handling
        try {
            task();
        } catch (const std::exception& e) {
            std::cerr << "ThreadPool: Task threw exception: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "ThreadPool: Task threw unknown exception" << std::endl;
        }
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(_queueMutex);
        _stop.store(true, std::memory_order_release);
    }

    // Notify all threads
    _condition.notify_all();

    // Wait for all threads to finish
    for (std::thread& worker : _workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

size_t ThreadPool::getPendingTaskCount() const {
    std::lock_guard<std::mutex> lock(_queueMutex);
    return _tasks.size();
}

} // namespace Utils
} // namespace WheelDL
