#include "pch.h"
#include "TaskManager.h"
#include "Context/BaseContext.h"
#include "Context/TrainingContext.h"
#include "Context/ValidateContext.h"
#include "Context/PredictContext.h"
#include <sstream>
#include <iomanip>

namespace WheelDL {
namespace Core {
namespace Manager {

TaskManager& TaskManager::getInstance()
{
    static TaskManager instance;
    return instance;
}

TaskManager::TaskManager()
    : _threadPool(std::make_unique<WheelDL::Utils::ThreadPool>(std::thread::hardware_concurrency()))
    , _memoryManager(WheelDL::Utils::MemoryManager::getInstance())
    , _memorySemaphore(std::make_unique<MemoryBasedSemaphore>(_memoryManager, 4096))
    , _logger(WheelDL::Utils::Logger::getDefault())
{
    _logger.info("TaskManager", "TaskManager initialized with " + std::to_string(std::thread::hardware_concurrency()) + " worker threads");
    _logger.info("TaskManager", "Memory-based scheduling enabled with 4GB threshold");
    _logger.info("TaskManager", "GPU Memory: " + std::to_string(_memorySemaphore->getAvailableMemory()) + "MB available");
}

TaskManager::~TaskManager() {
    _logger.info("TaskManager", "TaskManager shutting down");

    size_t stoppedCount = 0;
    {
        std::shared_lock<std::shared_mutex> lock(_tasksMutex);
        for (auto& [id, entry] : _tasks) {
            // Only stop tasks that are actually running
            if (entry->context) 
            {
                entry->context->requestStop();
                stoppedCount++;
            }
        }
    }

    if (stoppedCount > 0) {
        _logger.info("TaskManager", "Requested stop for " + std::to_string(stoppedCount) + " running tasks");
    }
}

std::string TaskManager::generateTaskId(OperationType type) {
    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << operationTypeToString(type) << "_";
    ss << std::put_time(std::localtime(&timeT), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
    ss << "_" << _taskIdCounter.fetch_add(1, std::memory_order_relaxed);

    return ss.str();
}

bool TaskManager::checkGpuMemory(size_t requiredMB) const {
    float usedMB = _memoryManager.getGPUMemoryUsed();
    float totalMB = _memoryManager.getGPUMemoryTotal();
    float availableMB = totalMB - usedMB;

    return availableMB >= static_cast<float>(requiredMB);
}

// ========== Task Submission ==========

std::string TaskManager::submitTask(const TrainRequest& request) {
    std::lock_guard<std::mutex> submitLock(_submitMutex);

    _logger.info("TaskManager", "submitTask(TrainRequest) - Epochs: " + std::to_string(request.config->getEpochs()) +
        ", BatchSize: " + std::to_string(request.config->getBatchSize()));

    waitForCooldown();
    _lastTaskStartTime = std::chrono::steady_clock::now();

    auto entry = std::make_shared<TaskEntry>();
    entry->operationType = OperationType::TRAIN;
    entry->state = TrainingState::IDLE;
    entry->submitTime = std::chrono::system_clock::now();
    entry->progress.totalEpochs = request.config->getEpochs();

    entry->context = std::make_unique<TrainingContext>(request);
    entry->taskId = entry->context->getTaskId();
    std::string taskId = entry->taskId;

    {
        std::unique_lock<std::shared_mutex> lock(_tasksMutex);
        _tasks[taskId] = entry;
    }

    _logger.info("TaskManager", "Task registered: " + taskId + " (TRAIN)");

    entry->future = _threadPool->enqueue([this, entry, taskId]() {
        _logger.info("TaskManager", "Task " + taskId + " waiting for GPU memory (threshold: " +
            std::to_string(_memorySemaphore->getThreshold()) + "MB, available: " +
            std::to_string(_memorySemaphore->getAvailableMemory()) + "MB)");

        _memorySemaphore->acquire();
        _logger.info("TaskManager", "Task " + taskId + " acquired GPU memory, starting execution");

        TaskResult result;
        try {
            updateTaskState(taskId, TrainingState::TRAINING);

            result = entry->context->run();

            updateTaskState(taskId, result.finalState);

            _logger.info("TaskManager", "Task " + taskId + " completed: " +
                (result.isSuccess() ? "SUCCESS" : "FAILED") +
                " | " + std::to_string(result.totalTimeMs) + "ms");

        } catch (const std::exception& e) {
            _logger.error("TaskManager", "Task " + taskId + " exception: " + std::string(e.what()));

            result.taskId = taskId;
            result.operationType = OperationType::TRAIN;
            result.finalState = TrainingState::FAILED;
            result.errorMessage = e.what();
            result.errorCode = static_cast<int>(WheelDL::Utils::ErrorCode::TRAINING_FAILED);

            updateTaskState(taskId, TrainingState::FAILED);
        }

        // Release resources
        _memorySemaphore->release();
        {
            std::unique_lock<std::shared_mutex> lock(_tasksMutex);
            entry->result = result;
            entry->context.reset();  // Release Context (Logger, Workspace, Profiler)
        }

        return result;
    });

    _logger.info("TaskManager", "Task submitted to thread pool: " + taskId);

    return taskId;
}

std::string TaskManager::submitTask(const ValidateRequest& request) {
    std::lock_guard<std::mutex> submitLock(_submitMutex);

    _logger.info("TaskManager", "submitTask(ValidateRequest) - Checkpoint: " + request.checkpointPath);

    waitForCooldown();
    _lastTaskStartTime = std::chrono::steady_clock::now();

    auto entry = std::make_shared<TaskEntry>();
    entry->operationType = OperationType::VALIDATE;
    entry->state = TrainingState::IDLE;
    entry->submitTime = std::chrono::system_clock::now();

    entry->context = std::make_unique<ValidateContext>(request);
    entry->taskId = entry->context->getTaskId();
    std::string taskId = entry->taskId;

    {
        std::unique_lock<std::shared_mutex> lock(_tasksMutex);
        _tasks[taskId] = entry;
    }

    _logger.info("TaskManager", "Task registered: " + taskId + " (VALIDATE)");

    entry->future = _threadPool->enqueue([this, entry, taskId]() {
        _logger.info("TaskManager", "Task " + taskId + " waiting for GPU memory (threshold: " +
            std::to_string(_memorySemaphore->getThreshold()) + "MB, available: " +
            std::to_string(_memorySemaphore->getAvailableMemory()) + "MB)");

        _memorySemaphore->acquire();
        _logger.info("TaskManager", "Task " + taskId + " acquired GPU memory, starting execution");

        TaskResult result;
        try {
            updateTaskState(taskId, TrainingState::VALIDATING);

            result = entry->context->run();

            updateTaskState(taskId, result.finalState);

            _logger.info("TaskManager", "Task " + taskId + " completed: " +
                (result.isSuccess() ? "SUCCESS" : "FAILED") +
                " | " + std::to_string(result.totalTimeMs) + "ms");

        } catch (const std::exception& e) {
            _logger.error("TaskManager", "Task " + taskId + " exception: " + std::string(e.what()));

            result.taskId = taskId;
            result.operationType = OperationType::VALIDATE;
            result.finalState = TrainingState::FAILED;
            result.errorMessage = e.what();
            result.errorCode = static_cast<int>(WheelDL::Utils::ErrorCode::VALIDATION_FAILED);

            updateTaskState(taskId, TrainingState::FAILED);
        }

        // Release resources
        _memorySemaphore->release();
        {
            std::unique_lock<std::shared_mutex> lock(_tasksMutex);
            entry->result = result;
            entry->context.reset();  // Release Context (Logger, Workspace, Profiler)
        }

        return result;
    });

    _logger.info("TaskManager", "Task submitted to thread pool: " + taskId);

    return taskId;
}

std::string TaskManager::submitTask(const PredictRequest& request) {
    std::lock_guard<std::mutex> submitLock(_submitMutex);

    _logger.info("TaskManager", "submitTask(PredictRequest) - Checkpoint: " + request.checkpointPath);

    waitForCooldown();
    _lastTaskStartTime = std::chrono::steady_clock::now();

    auto entry = std::make_shared<TaskEntry>();
    entry->operationType = OperationType::PREDICT;
    entry->state = TrainingState::IDLE;
    entry->submitTime = std::chrono::system_clock::now();

    entry->context = std::make_unique<PredictContext>(request);
    entry->taskId = entry->context->getTaskId();
    std::string taskId = entry->taskId;

    {
        std::unique_lock<std::shared_mutex> lock(_tasksMutex);
        _tasks[taskId] = entry;
    }

    _logger.info("TaskManager", "Task registered: " + taskId + " (PREDICT)");

    entry->future = _threadPool->enqueue([this, entry, taskId]() {
        _logger.info("TaskManager", "Task " + taskId + " waiting for GPU memory (threshold: " +
            std::to_string(_memorySemaphore->getThreshold()) + "MB, available: " +
            std::to_string(_memorySemaphore->getAvailableMemory()) + "MB)");

        _memorySemaphore->acquire();
        _logger.info("TaskManager", "Task " + taskId + " acquired GPU memory, starting execution");

        TaskResult result;
        try {
            updateTaskState(taskId, TrainingState::TRAINING);

            result = entry->context->run();

            updateTaskState(taskId, result.finalState);

            _logger.info("TaskManager", "Task " + taskId + " completed: " +
                (result.isSuccess() ? "SUCCESS" : "FAILED") +
                " | " + std::to_string(result.totalTimeMs) + "ms");

        } catch (const std::exception& e) {
            _logger.error("TaskManager", "Task " + taskId + " exception: " + std::string(e.what()));

            result.taskId = taskId;
            result.operationType = OperationType::PREDICT;
            result.finalState = TrainingState::FAILED;
            result.errorMessage = e.what();
            result.errorCode = static_cast<int>(WheelDL::Utils::ErrorCode::PREDICTION_FAILED);

            updateTaskState(taskId, TrainingState::FAILED);
        }

        // Release resources
        _memorySemaphore->release();
        {
            std::unique_lock<std::shared_mutex> lock(_tasksMutex);
            entry->result = result;
            entry->context.reset();  // Release Context (Logger, Workspace, Profiler)
        }

        return result;
    });

    _logger.info("TaskManager", "Task submitted to thread pool: " + taskId);

    return taskId;
}

// ========== Task Query ==========

ProgressData TaskManager::getTaskStatus(const std::string& taskId) const {
    std::shared_lock<std::shared_mutex> lock(_tasksMutex);

    auto it = _tasks.find(taskId);
    if (it == _tasks.end()) {
        throw WheelDL::Utils::TaskException(
            WheelDL::Utils::ErrorCode::TASK_NOT_FOUND,
            "Task not found: " + taskId,
            taskId);
    }

    if (it->second->context) {
        return it->second->context->getProgress();
    }

    return it->second->progress;
}

TaskResult TaskManager::getTaskResult(const std::string& taskId) const {
    std::shared_lock<std::shared_mutex> lock(_tasksMutex);

    auto it = _tasks.find(taskId);
    if (it == _tasks.end()) {
        throw WheelDL::Utils::TaskException(
            WheelDL::Utils::ErrorCode::TASK_NOT_FOUND,
            "Task not found: " + taskId,
            taskId);
    }

    if (!isTerminalState(it->second->state)) {
        throw WheelDL::Utils::TaskException(
            WheelDL::Utils::ErrorCode::TASK_INVALID_STATE,
            "Task not completed: " + taskId,
            taskId);
    }

    return it->second->result;
}

bool TaskManager::taskExists(const std::string& taskId) const {
    std::shared_lock<std::shared_mutex> lock(_tasksMutex);
    return _tasks.find(taskId) != _tasks.end();
}

bool TaskManager::isTaskRunning(const std::string& taskId) const {
    std::shared_lock<std::shared_mutex> lock(_tasksMutex);

    auto it = _tasks.find(taskId);
    if (it == _tasks.end()) {
        return false;
    }

    TrainingState state = it->second->state;
    return state == TrainingState::INITIALIZING ||
           state == TrainingState::TRAINING ||
           state == TrainingState::VALIDATING;
}

// ========== Task Control ==========

void TaskManager::stopTask(const std::string& taskId) {
    _logger.info("TaskManager", "stopTask requested: " + taskId);

    std::shared_lock<std::shared_mutex> lock(_tasksMutex);

    auto it = _tasks.find(taskId);
    if (it == _tasks.end()) {
        _logger.error("TaskManager", "stopTask failed - Task not found: " + taskId);
        throw WheelDL::Utils::TaskException(
            WheelDL::Utils::ErrorCode::TASK_NOT_FOUND,
            "Task not found: " + taskId,
            taskId);
    }

    if (it->second->context) {
        it->second->context->requestStop();
        _logger.info("TaskManager", "Stop signal sent to task: " + taskId);
    } else {
        _logger.warn("TaskManager", "stopTask: Task " + taskId + " has no context");
    }
}

TaskResult TaskManager::waitForTask(const std::string& taskId, uint64_t timeoutMs) {
    _logger.info("TaskManager", "waitForTask: " + taskId +
        (timeoutMs > 0 ? " (timeout: " + std::to_string(timeoutMs) + "ms)" : " (no timeout)"));

    std::shared_ptr<TaskEntry> entry;

    {
        std::shared_lock<std::shared_mutex> lock(_tasksMutex);
        auto it = _tasks.find(taskId);
        if (it == _tasks.end()) {
            _logger.error("TaskManager", "waitForTask failed - Task not found: " + taskId);
            throw WheelDL::Utils::TaskException(
                WheelDL::Utils::ErrorCode::TASK_NOT_FOUND,
                "Task not found: " + taskId,
                taskId);
        }
        entry = it->second;
    }

    TaskResult result;
    if (timeoutMs == 0) {
        result = entry->future.get();
    } else {
        auto status = entry->future.wait_for(std::chrono::milliseconds(timeoutMs));
        if (status == std::future_status::timeout) {
            _logger.error("TaskManager", "waitForTask failed - Timeout after " +
                std::to_string(timeoutMs) + "ms: " + taskId);
            throw WheelDL::Utils::TaskException(
                WheelDL::Utils::ErrorCode::TASK_TIMEOUT,
                "Task timeout: " + taskId,
                taskId);
        }
        result = entry->future.get();
    }

    _logger.info("TaskManager", "waitForTask completed: " + taskId +
        " | " + (result.isSuccess() ? "SUCCESS" : "FAILED") +
        " | " + std::to_string(result.totalTimeMs) + "ms");

    return result;
}

// ========== Resource Management ==========

void TaskManager::setMemoryThreshold(size_t thresholdMB) {
    _logger.info("TaskManager", "setMemoryThreshold: " + std::to_string(thresholdMB) + "MB");
    _memorySemaphore->setThreshold(thresholdMB);
}

size_t TaskManager::getMemoryThreshold() const {
    return _memorySemaphore->getThreshold();
}

bool TaskManager::canAcceptNewTask(int deviceIndex) const {
    return _memorySemaphore->hasEnoughMemory(deviceIndex);
}

float TaskManager::getAvailableGpuMemory(int deviceIndex) const {
    return _memorySemaphore->getAvailableMemory(deviceIndex);
}

size_t TaskManager::getRunningTaskCount() const {
    std::shared_lock<std::shared_mutex> lock(_tasksMutex);

    size_t count = 0;
    for (const auto& [id, entry] : _tasks) {
        if (entry->state == TrainingState::TRAINING ||
            entry->state == TrainingState::VALIDATING ||
            entry->state == TrainingState::INITIALIZING) {
            ++count;
        }
    }
    return count;
}

size_t TaskManager::getPendingTaskCount() const {
    std::shared_lock<std::shared_mutex> lock(_tasksMutex);

    size_t count = 0;
    for (const auto& [id, entry] : _tasks) {
        if (entry->state == TrainingState::IDLE) {
            ++count;
        }
    }
    return count;
}

// ========== Cleanup ==========

void TaskManager::removeTask(const std::string& taskId) {
    _logger.info("TaskManager", "removeTask: " + taskId);

    std::unique_lock<std::shared_mutex> lock(_tasksMutex);
    auto it = _tasks.find(taskId);
    if (it != _tasks.end()) {
        _tasks.erase(it);
        _logger.info("TaskManager", "Task removed: " + taskId);
    } else {
        _logger.warn("TaskManager", "removeTask: Task not found: " + taskId);
    }
}

void TaskManager::clearCompletedTasks() {
    _logger.info("TaskManager", "clearCompletedTasks");

    std::unique_lock<std::shared_mutex> lock(_tasksMutex);

    size_t removedCount = 0;
    for (auto it = _tasks.begin(); it != _tasks.end(); ) {
        if (isTerminalState(it->second->state)) {
            _logger.debug("TaskManager", "Removing completed task: " + it->first);
            it = _tasks.erase(it);
            removedCount++;
        } else {
            ++it;
        }
    }

    _logger.info("TaskManager", "Cleared " + std::to_string(removedCount) + " completed tasks");
}

// ========== Internal ==========

void TaskManager::updateTaskState(const std::string& taskId, TrainingState state) {
    std::unique_lock<std::shared_mutex> lock(_tasksMutex);

    auto it = _tasks.find(taskId);
    if (it != _tasks.end()) {
        TrainingState prevState = it->second->state;
        it->second->state = state;
        _logger.info("TaskManager", "Task " + taskId + " state changed: " +
            trainingStateToString(prevState) + " -> " + trainingStateToString(state));
    }
}

void TaskManager::updateTaskProgress(const std::string& taskId, const ProgressData& progress) {
    std::unique_lock<std::shared_mutex> lock(_tasksMutex);

    auto it = _tasks.find(taskId);
    if (it != _tasks.end()) {
        it->second->progress = progress;
    }
}

void TaskManager::waitForCooldown() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - _lastTaskStartTime);

    if (_lastTaskStartTime.time_since_epoch().count() > 0 && elapsed.count() < TASK_COOLDOWN_MS) {
        auto remaining = TASK_COOLDOWN_MS - elapsed.count();
        _logger.info("TaskManager", "Cooldown: waiting " + std::to_string(remaining) + "ms before next task");
        std::this_thread::sleep_for(std::chrono::milliseconds(remaining));
        _logger.info("TaskManager", "Cooldown completed");
    }
}

} // namespace Manager
} // namespace Core
} // namespace WheelDL
