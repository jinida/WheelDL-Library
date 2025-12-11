#pragma once

#include "../../Utils/Common/Types.h"
#include <string>
#include <cstdint>

namespace WheelDL {
namespace Core {
namespace Manager {

/**
 * @enum OperationType
 * @brief Types of operations that can be submitted to TaskManager
 *
 * Note: TaskType (in Utils/Common/Types.h) defines ML task types (Classification, Detection, etc.)
 *       OperationType defines what operation to perform (Train, Validate, Predict)
 */
enum class OperationType {
    TRAIN,      ///< Training operation
    VALIDATE,   ///< Validation operation
    PREDICT     ///< Prediction/Inference operation
};

/**
 * @brief Convert OperationType to string
 */
inline const char* operationTypeToString(OperationType type) {
    switch (type) {
        case OperationType::TRAIN: return "TRAIN";
        case OperationType::VALIDATE: return "VALIDATE";
        case OperationType::PREDICT: return "PREDICT";
        default: return "UNKNOWN";
    }
}

/**
 * @enum TaskPriority
 * @brief Priority levels for task scheduling
 */
enum class TaskPriority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3
};

// ============================================================
// Re-export from Utils/Common/Types.h for convenience
// ============================================================
// - TrainingState: Task lifecycle state (IDLE, TRAINING, COMPLETED, FAILED, etc.)
// - ProgressData: Progress information (epoch, batch, loss, metrics, etc.)
// - MetricsData: Evaluation metrics (accuracy, mAP, F1, etc.)
// - ProgressCallback: Callback for progress updates

/**
 * @brief Check if TrainingState is terminal (no more state transitions)
 */
inline bool isTerminalState(TrainingState state) {
    return state == TrainingState::COMPLETED ||
           state == TrainingState::FAILED ||
           state == TrainingState::STOPPED;
}

} // namespace Manager
} // namespace Core
} // namespace WheelDL
