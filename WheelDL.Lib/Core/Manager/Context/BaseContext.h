#pragma once

#include "../Types.h"
#include "../Result.h"
#include "../../../Config/Configuration.h"
#include "../../../Utils/Logger/Logger.h"
#include "../../../Utils/Workspace/Workspace.h"
#include "../../../Utils/Profiler/PerformanceProfiler.h"
#include <memory>
#include <string>
#include <atomic>
#include <chrono>

namespace WheelDL {
namespace Core {
namespace Manager {

/**
 * @class BaseContext
 * @brief Abstract base class for task execution context
 *
 * Owns task-specific resources (Logger, Workspace, Profiler) via unique_ptr.
 * Derived classes implement run() for specific operations.
 *
 * RAII: Resources automatically cleaned up on destruction.
 */
class BaseContext {
public:
    /**
     * @brief Constructor
     * @param config Configuration
     * @param operationType Type of operation
     */
    BaseContext(
        std::shared_ptr<Config::Configuration> config,
        OperationType operationType);

    virtual ~BaseContext() = default;

    // Non-copyable, non-movable
    BaseContext(const BaseContext&) = delete;
    BaseContext& operator=(const BaseContext&) = delete;
    BaseContext(BaseContext&&) = delete;
    BaseContext& operator=(BaseContext&&) = delete;

    /**
     * @brief Execute the task (implemented by derived classes)
     * @return TaskResult Result of execution
     */
    virtual TaskResult run() = 0;

    /**
     * @brief Request task to stop
     */
    void requestStop();

    /**
     * @brief Check if stop was requested
     */
    bool isStopRequested() const;

    /**
     * @brief Get task ID
     */
    std::string getTaskId() const { return _taskId; }

    /**
     * @brief Get current progress
     */
    ProgressData getProgress() const { return _progress; }

    /**
     * @brief Get current state
     */
    TrainingState getState() const { return _state; }

protected:
    /**
     * @brief Initialize resources (Logger, Workspace, Profiler)
     */
    void initializeResources();

    /**
     * @brief Generate unique task ID
     */
    std::string generateTaskId(OperationType type);

protected:
    // Task identification
    std::string _taskId;
    OperationType _operationType;

    // Configuration
    std::shared_ptr<Config::Configuration> _config;

    // Owned resources (RAII)
    std::unique_ptr<WheelDL::Utils::Logger> _logger;
    std::unique_ptr<WheelDL::Utils::Workspace> _workspace;
    std::unique_ptr<WheelDL::Utils::PerformanceProfiler> _profiler;

    // State
    std::atomic<bool> _stopRequested{false};
    ProgressData _progress;
    TrainingState _state{TrainingState::IDLE};

    // Static counter for unique task IDs
    static std::atomic<uint64_t> _taskIdCounter;
};

} // namespace Manager
} // namespace Core
} // namespace WheelDL
