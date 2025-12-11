#pragma once

#include "Types.h"
#include <string>
#include <chrono>
#include <optional>
#include <unordered_map>

namespace WheelDL {
namespace Core {
namespace Manager {

/**
 * @struct ProfilingData
 * @brief Performance profiling data for a task
 *
 * Note: ProgressData (Utils/Common/Types.h) contains runtime progress info.
 *       ProfilingData contains post-execution performance analysis.
 */
struct ProfilingData {
    double totalTimeMs = 0.0;                       ///< Total execution time in milliseconds
    double gpuTimeMs = 0.0;                         ///< GPU execution time in milliseconds
    double dataLoadTimeMs = 0.0;                    ///< Data loading time in milliseconds
    double preprocessTimeMs = 0.0;                  ///< Preprocessing time in milliseconds
    double inferenceTimeMs = 0.0;                   ///< Inference time in milliseconds
    double postprocessTimeMs = 0.0;                 ///< Postprocessing time in milliseconds

    size_t peakGpuMemoryMB = 0;                     ///< Peak GPU memory usage in MB
    size_t peakCpuMemoryMB = 0;                     ///< Peak CPU memory usage in MB

    std::unordered_map<std::string, double> customTimings;  ///< Custom timing measurements
};

/**
 * @struct TaskResult
 * @brief Final result of a completed task
 *
 * Uses:
 * - TrainingState from Utils/Common/Types.h (lifecycle state)
 * - MetricsData from Utils/Common/Types.h (evaluation metrics)
 * - OperationType from Core/Manager/Types.h (operation type)
 */
struct TaskResult {
    std::string taskId;                             ///< Unique task identifier
    TrainingState finalState = TrainingState::IDLE; ///< Final task state (from Utils/Common/Types.h)
    OperationType operationType;                    ///< Type of operation

    // Success metrics (populated on successful completion)
    std::optional<MetricsData> metrics;             ///< Training/validation metrics (from Utils/Common/Types.h)
    std::string outputPath;                         ///< Path to output directory
    std::string checkpointPath;                     ///< Path to saved checkpoint (training)

    // Timing and profiling
    std::chrono::system_clock::time_point startTime;    ///< Task start time
    std::chrono::system_clock::time_point endTime;      ///< Task end time
    double totalTimeMs = 0.0;                           ///< Total execution time
    ProfilingData profiling;                            ///< Detailed profiling data

    // Error information (if failed)
    std::string errorMessage;                       ///< Error message (if finalState == FAILED)
    int errorCode = 0;                              ///< Error code (if finalState == FAILED)

    /**
     * @brief Check if task completed successfully
     */
    bool isSuccess() const {
        return finalState == TrainingState::COMPLETED;
    }

    /**
     * @brief Check if task was stopped
     */
    bool isStopped() const {
        return finalState == TrainingState::STOPPED;
    }

    /**
     * @brief Check if task failed
     */
    bool isFailed() const {
        return finalState == TrainingState::FAILED;
    }
};

} // namespace Manager
} // namespace Core
} // namespace WheelDL
