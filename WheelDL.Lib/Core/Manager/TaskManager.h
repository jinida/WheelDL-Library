#pragma once

#include "Types.h"
#include "Request.h"
#include "Result.h"
#include "Context/BaseContext.h"
#include "../../Utils/ThreadPool/ThreadPool.h"
#include "../../Utils/Memory/MemoryManager.h"
#include "../../Utils/Logger/Logger.h"
#include "../../Utils/Error/WheelLibException.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <shared_mutex>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>

namespace WheelDL {
namespace Core {
namespace Manager {

/**
 * @class TaskManager
 * @brief Asynchronous task management for training, validation, and prediction
 *
 * Features:
 * - Async task submission via ThreadPool
 * - GPU concurrency control via counting_semaphore
 * - Task lifecycle management (submit, query, stop)
 * - Memory availability check before execution
 * - Thread-safe task registry with shared_mutex
 *
 * Usage:
 * @code
 * auto& manager = TaskManager::getInstance();
 *
 * TrainRequest request(config);
 * request.progressCallback = [](const ProgressData& p) { ... };
 *
 * std::string taskId = manager.submitTask(request);
 *
 * // Query status
 * auto status = manager.getTaskStatus(taskId);
 *
 * // Wait for completion
 * auto result = manager.waitForTask(taskId);
 * @endcode
 */
class TaskManager {
public:
    /**
     * @brief Get singleton instance
     * @return TaskManager& Singleton instance
     */
    static TaskManager& getInstance();

    // Non-copyable, non-movable
    TaskManager(const TaskManager&) = delete;
    TaskManager& operator=(const TaskManager&) = delete;
    TaskManager(TaskManager&&) = delete;
    TaskManager& operator=(TaskManager&&) = delete;

    // ========== Task Submission ==========

    /**
     * @brief Submit training task
     * @param request Training request
     * @return std::string Task ID
     * @throws TaskException if submission fails
     */
    std::string submitTask(const TrainRequest& request);

    /**
     * @brief Submit validation task
     * @param request Validation request
     * @return std::string Task ID
     * @throws TaskException if submission fails
     */
    std::string submitTask(const ValidateRequest& request);

    /**
     * @brief Submit prediction task
     * @param request Prediction request
     * @return std::string Task ID
     * @throws TaskException if submission fails
     */
    std::string submitTask(const PredictRequest& request);

    // ========== Task Query ==========

    /**
     * @brief Get current status of a task
     * @param taskId Task ID
     * @return ProgressData Current progress (from Utils/Common/Types.h)
     * @throws TaskException if task not found
     */
    ProgressData getTaskStatus(const std::string& taskId) const;

    /**
     * @brief Get final result of a completed task
     * @param taskId Task ID
     * @return TaskResult Final result
     * @throws TaskException if task not found or not completed
     */
    TaskResult getTaskResult(const std::string& taskId) const;

    /**
     * @brief Check if task exists
     * @param taskId Task ID
     * @return bool True if task exists
     */
    bool taskExists(const std::string& taskId) const;

    /**
     * @brief Check if task is running
     * @param taskId Task ID
     * @return bool True if task is running
     */
    bool isTaskRunning(const std::string& taskId) const;

    // ========== Task Control ==========

    /**
     * @brief Request task to stop
     * @param taskId Task ID
     * @throws TaskException if task not found
     */
    void stopTask(const std::string& taskId);

    /**
     * @brief Wait for task completion
     * @param taskId Task ID
     * @param timeoutMs Timeout in milliseconds (0 = infinite)
     * @return TaskResult Final result
     * @throws TaskException if task not found or timeout
     */
    TaskResult waitForTask(const std::string& taskId, uint64_t timeoutMs = 0);

    // ========== Resource Management ==========

    /**
     * @brief Set minimum GPU memory threshold for accepting new tasks (in MB)
     * @param thresholdMB Minimum available GPU memory in MB (default: 4096 = 4GB)
     */
    void setMemoryThreshold(size_t thresholdMB);

    /**
     * @brief Get current memory threshold setting
     * @return size_t Memory threshold in MB
     */
    size_t getMemoryThreshold() const;

    /**
     * @brief Check if GPU has enough memory to accept new task
     * @param deviceIndex GPU device index (default: 0)
     * @return bool True if enough memory available
     */
    bool canAcceptNewTask(int deviceIndex = 0) const;

    /**
     * @brief Get current available GPU memory in MB
     * @param deviceIndex GPU device index (default: 0)
     * @return float Available memory in MB
     */
    float getAvailableGpuMemory(int deviceIndex = 0) const;

    /**
     * @brief Get number of currently running tasks
     * @return size_t Number of running tasks
     */
    size_t getRunningTaskCount() const;

    /**
     * @brief Get number of pending tasks
     * @return size_t Number of pending tasks
     */
    size_t getPendingTaskCount() const;

    // ========== Cleanup ==========

    /**
     * @brief Remove completed task from registry
     * @param taskId Task ID
     */
    void removeTask(const std::string& taskId);

    /**
     * @brief Clear all completed tasks
     */
    void clearCompletedTasks();

private:
    TaskManager();
    ~TaskManager();

    /**
     * @brief Generate unique task ID
     * @param type Operation type
     * @return std::string Unique task ID (timestamp-based)
     */
    std::string generateTaskId(OperationType type);

    /**
     * @brief Check if enough GPU memory is available
     * @param requiredMB Required memory in MB
     * @return bool True if enough memory available
     */
    bool checkGpuMemory(size_t requiredMB) const;

    /**
     * @brief Internal task entry
     */
    struct TaskEntry {
        std::string taskId;
        OperationType operationType;
        TrainingState state = TrainingState::IDLE;
        ProgressData progress;
        TaskResult result;
        std::unique_ptr<BaseContext> context;  // Owns resources (Logger, Workspace, Profiler)
        std::future<TaskResult> future;
        std::chrono::system_clock::time_point submitTime;
        std::chrono::system_clock::time_point startTime;
        std::chrono::system_clock::time_point endTime;
    };

    /**
     * @brief Update task state
     */
    void updateTaskState(const std::string& taskId, TrainingState state);

    /**
     * @brief Update task progress
     */
    void updateTaskProgress(const std::string& taskId, const ProgressData& progress);

    /**
     * @brief Wait for cooldown period since last task started
     */
    void waitForCooldown();

    /**
     * @class MemoryBasedSemaphore
     * @brief Memory-based semaphore that allows task execution when sufficient GPU memory is available
     */
    class MemoryBasedSemaphore {
    public:
        explicit MemoryBasedSemaphore(WheelDL::Utils::MemoryManager& memMgr, size_t thresholdMB = 4096)
            : _memoryManager(memMgr)
            , _thresholdMB(thresholdMB)
            , _runningTasks(0)
        {}

        /**
         * @brief Acquire permission to run a GPU task
         * Blocks until GPU memory is above threshold
         * @param deviceIndex GPU device index
         * @param timeoutMs Maximum wait time (0 = infinite)
         * @return bool True if acquired, false if timeout
         */
        bool acquire(int deviceIndex = 0, uint64_t timeoutMs = 0) {
            std::unique_lock<std::mutex> lock(_mutex);

            auto condition = [this, deviceIndex]() {
                return hasEnoughMemory(deviceIndex);
            };

            if (timeoutMs == 0) {
                // Infinite wait with periodic memory check
                while (!condition()) {
                    _cv.wait_for(lock, std::chrono::milliseconds(100));
                }
            } else {
                auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
                while (!condition()) {
                    if (_cv.wait_until(lock, deadline) == std::cv_status::timeout) {
                        if (!condition()) return false;
                    }
                }
            }

            ++_runningTasks;
            return true;
        }

        /**
         * @brief Release GPU task slot
         */
        void release() {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                if (_runningTasks > 0) --_runningTasks;
            }
            _cv.notify_all();
        }

        /**
         * @brief Set memory threshold
         * @param thresholdMB Threshold in MB
         */
        void setThreshold(size_t thresholdMB) {
            std::lock_guard<std::mutex> lock(_mutex);
            _thresholdMB = thresholdMB;
        }

        /**
         * @brief Get current threshold
         * @return size_t Threshold in MB
         */
        size_t getThreshold() const {
            std::lock_guard<std::mutex> lock(_mutex);
            return _thresholdMB;
        }

        /**
         * @brief Check if memory is above threshold
         * @param deviceIndex GPU device index
         * @return bool True if enough memory
         */
        bool hasEnoughMemory(int deviceIndex = 0) const {
            float usedMB = _memoryManager.getGPUMemoryUsed(deviceIndex);
            float totalMB = _memoryManager.getGPUMemoryTotal(deviceIndex);
            float availableMB = totalMB - usedMB;
            return availableMB >= static_cast<float>(_thresholdMB);
        }

        /**
         * @brief Get available memory
         * @param deviceIndex GPU device index
         * @return float Available memory in MB
         */
        float getAvailableMemory(int deviceIndex = 0) const {
            float usedMB = _memoryManager.getGPUMemoryUsed(deviceIndex);
            float totalMB = _memoryManager.getGPUMemoryTotal(deviceIndex);
            return totalMB - usedMB;
        }

        /**
         * @brief Get running task count
         * @return int Number of running tasks
         */
        int getRunningTasks() const {
            std::lock_guard<std::mutex> lock(_mutex);
            return _runningTasks;
        }

    private:
        WheelDL::Utils::MemoryManager& _memoryManager;
        size_t _thresholdMB;
        int _runningTasks;
        mutable std::mutex _mutex;
        std::condition_variable _cv;
    };

private:
    // Thread pool for async execution
    std::unique_ptr<WheelDL::Utils::ThreadPool> _threadPool;

    // Memory-based GPU scheduling (default threshold: 4GB)
    std::unique_ptr<MemoryBasedSemaphore> _memorySemaphore;

    // Task registry with reader-writer lock
    mutable std::shared_mutex _tasksMutex;
    std::unordered_map<std::string, std::shared_ptr<TaskEntry>> _tasks;

    // Mutex for serializing task submission (Context creation)
    std::mutex _submitMutex;

    // Task ID counter
    std::atomic<uint64_t> _taskIdCounter{0};

    // Memory manager reference
    WheelDL::Utils::MemoryManager& _memoryManager;

    // Logger
    WheelDL::Utils::Logger& _logger;

    // Last task activation time (for 30-second cooldown)
    std::chrono::steady_clock::time_point _lastTaskStartTime;
    static constexpr uint64_t TASK_COOLDOWN_MS = 30000;
};

} // namespace Manager
} // namespace Core
} // namespace WheelDL
