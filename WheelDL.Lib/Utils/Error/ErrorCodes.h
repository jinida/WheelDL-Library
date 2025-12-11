#pragma once

namespace WheelDL {
namespace Utils {

/**
 * @enum ErrorCode
 * @brief Error codes for WheelDL.Lib exceptions
 */
enum class ErrorCode {
    // Success
    SUCCESS = 0,

    // Configuration errors (1000-1999)
    INVALID_CONFIG = 1000,
    CONFIG_PARSE_FAILED = 1001,
    CONFIG_FILE_NOT_FOUND = 1002,
    CONFIG_VALIDATION_FAILED = 1003,
    MISSING_CONFIG_KEY = 1004,
    INVALID_CONFIG_VALUE = 1005,

    // Model errors (2000-2999)
    MODEL_LOAD_FAILED = 2000,
    MODEL_SAVE_FAILED = 2001,
    MODEL_BUILD_FAILED = 2002,
    MODEL_INVALID_ARCHITECTURE = 2003,
    INVALID_MODEL_ARCHITECTURE = 2003,  // Alias for MODEL_INVALID_ARCHITECTURE
    MODEL_FORWARD_FAILED = 2004,
    MODEL_INFERENCE_FAILED = 2005,

    // Data errors (3000-3999)
    DATA_LOAD_FAILED = 3000,
    DATA_INVALID_FORMAT = 3001,
    INVALID_DATA_FORMAT = 3001,  // Alias for DATA_INVALID_FORMAT
    DATA_FILE_NOT_FOUND = 3002,
    DATA_ANNOTATION_INVALID = 3003,
    DATA_TRANSFORM_FAILED = 3004,
    DATA_CACHE_FAILED = 3005,
    DATA_PREPROCESSING_FAILED = 3006,
    DATASET_NOT_FOUND = 3007,
    INVALID_IMAGE_SIZE = 3008,

    // GPU errors (4000-4999)
    GPU_OUT_OF_MEMORY = 4000,
    GPU_NOT_AVAILABLE = 4001,
    GPU_INIT_FAILED = 4002,
    GPU_ALLOCATION_FAILED = 4003,
    GPU_CUDA_ERROR = 4004,
    CUDA_ERROR = 4004,  // Alias for GPU_CUDA_ERROR
    CUDNN_ERROR = 4005,
    GPU_DEVICE_MISMATCH = 4006,

    // Training errors (5000-5999)
    TRAINING_FAILED = 5000,
    TRAINING_INTERRUPTED = 5001,
    TRAINING_DIVERGED = 5002,
    VALIDATION_FAILED = 5003,
    PREDICTION_FAILED = 5004,
    CHECKPOINT_SAVE_FAILED = 5005,
    CHECKPOINT_LOAD_FAILED = 5006,
    LOSS_IS_NAN = 5007,
    GRADIENT_EXPLOSION = 5008,

    // Distributed training errors (6000-6999)
    DISTRIBUTED_INIT_FAILED = 6000,
    DISTRIBUTED_COMM_FAILED = 6001,
    DISTRIBUTED_SYNC_FAILED = 6002,
    DISTRIBUTED_RANK_ERROR = 6003,
    DISTRIBUTED_TIMEOUT = 6004,
    RANK_MISMATCH = 6005,
    WORLD_SIZE_MISMATCH = 6006,

    // Task management errors (7000-7999)
    TASK_NOT_FOUND = 7000,
    TASK_ALREADY_RUNNING = 7001,
    TASK_ALREADY_COMPLETED = 7002,
    TASK_CANCELLED = 7003,
    TASK_TIMEOUT = 7004,
    TASK_SUBMIT_FAILED = 7005,
    TASK_INVALID_STATE = 7006,
    TASK_DEPENDENCY_FAILED = 7007,
    TASK_RESOURCE_UNAVAILABLE = 7008,
    TASK_GPU_LIMIT_EXCEEDED = 7009,

    // General errors (9000-9999)
    UNKNOWN_ERROR = 9000,
    NOT_IMPLEMENTED = 9001,
    INVALID_ARGUMENT = 9002,
    OUT_OF_RANGE = 9003,
    FILE_IO_ERROR = 9004,
    MEMORY_ALLOCATION_FAILED = 9005
};

/**
 * @brief Convert error code to string
 * @param code Error code
 * @return const char* Error code name
 */
inline const char* errorCodeToString(ErrorCode code) {
    switch (code) {
        case ErrorCode::SUCCESS: return "SUCCESS";

        // Configuration
        case ErrorCode::INVALID_CONFIG: return "INVALID_CONFIG";
        case ErrorCode::CONFIG_PARSE_FAILED: return "CONFIG_PARSE_FAILED";
        case ErrorCode::CONFIG_FILE_NOT_FOUND: return "CONFIG_FILE_NOT_FOUND";
        case ErrorCode::CONFIG_VALIDATION_FAILED: return "CONFIG_VALIDATION_FAILED";
        case ErrorCode::MISSING_CONFIG_KEY: return "MISSING_CONFIG_KEY";
        case ErrorCode::INVALID_CONFIG_VALUE: return "INVALID_CONFIG_VALUE";

        // Model
        case ErrorCode::MODEL_LOAD_FAILED: return "MODEL_LOAD_FAILED";
        case ErrorCode::MODEL_SAVE_FAILED: return "MODEL_SAVE_FAILED";
        case ErrorCode::MODEL_BUILD_FAILED: return "MODEL_BUILD_FAILED";
        case ErrorCode::MODEL_INVALID_ARCHITECTURE: return "MODEL_INVALID_ARCHITECTURE";
        case ErrorCode::MODEL_FORWARD_FAILED: return "MODEL_FORWARD_FAILED";
        case ErrorCode::MODEL_INFERENCE_FAILED: return "MODEL_INFERENCE_FAILED";

        // Data
        case ErrorCode::DATA_LOAD_FAILED: return "DATA_LOAD_FAILED";
        case ErrorCode::DATA_INVALID_FORMAT: return "DATA_INVALID_FORMAT";
        case ErrorCode::DATA_FILE_NOT_FOUND: return "DATA_FILE_NOT_FOUND";
        case ErrorCode::DATA_ANNOTATION_INVALID: return "DATA_ANNOTATION_INVALID";
        case ErrorCode::DATA_TRANSFORM_FAILED: return "DATA_TRANSFORM_FAILED";
        case ErrorCode::DATA_CACHE_FAILED: return "DATA_CACHE_FAILED";
        case ErrorCode::DATA_PREPROCESSING_FAILED: return "DATA_PREPROCESSING_FAILED";
        case ErrorCode::DATASET_NOT_FOUND: return "DATASET_NOT_FOUND";
        case ErrorCode::INVALID_IMAGE_SIZE: return "INVALID_IMAGE_SIZE";

        // GPU
        case ErrorCode::GPU_OUT_OF_MEMORY: return "GPU_OUT_OF_MEMORY";
        case ErrorCode::GPU_NOT_AVAILABLE: return "GPU_NOT_AVAILABLE";
        case ErrorCode::GPU_INIT_FAILED: return "GPU_INIT_FAILED";
        case ErrorCode::GPU_ALLOCATION_FAILED: return "GPU_ALLOCATION_FAILED";
        case ErrorCode::GPU_CUDA_ERROR: return "GPU_CUDA_ERROR";
        case ErrorCode::CUDNN_ERROR: return "CUDNN_ERROR";
        case ErrorCode::GPU_DEVICE_MISMATCH: return "GPU_DEVICE_MISMATCH";

        // Training
        case ErrorCode::TRAINING_FAILED: return "TRAINING_FAILED";
        case ErrorCode::TRAINING_INTERRUPTED: return "TRAINING_INTERRUPTED";
        case ErrorCode::TRAINING_DIVERGED: return "TRAINING_DIVERGED";
        case ErrorCode::VALIDATION_FAILED: return "VALIDATION_FAILED";
        case ErrorCode::CHECKPOINT_SAVE_FAILED: return "CHECKPOINT_SAVE_FAILED";
        case ErrorCode::CHECKPOINT_LOAD_FAILED: return "CHECKPOINT_LOAD_FAILED";
        case ErrorCode::LOSS_IS_NAN: return "LOSS_IS_NAN";
        case ErrorCode::GRADIENT_EXPLOSION: return "GRADIENT_EXPLOSION";

        // Distributed
        case ErrorCode::DISTRIBUTED_INIT_FAILED: return "DISTRIBUTED_INIT_FAILED";
        case ErrorCode::DISTRIBUTED_COMM_FAILED: return "DISTRIBUTED_COMM_FAILED";
        case ErrorCode::DISTRIBUTED_SYNC_FAILED: return "DISTRIBUTED_SYNC_FAILED";
        case ErrorCode::DISTRIBUTED_RANK_ERROR: return "DISTRIBUTED_RANK_ERROR";
        case ErrorCode::DISTRIBUTED_TIMEOUT: return "DISTRIBUTED_TIMEOUT";
        case ErrorCode::RANK_MISMATCH: return "RANK_MISMATCH";
        case ErrorCode::WORLD_SIZE_MISMATCH: return "WORLD_SIZE_MISMATCH";

        // Task management
        case ErrorCode::TASK_NOT_FOUND: return "TASK_NOT_FOUND";
        case ErrorCode::TASK_ALREADY_RUNNING: return "TASK_ALREADY_RUNNING";
        case ErrorCode::TASK_ALREADY_COMPLETED: return "TASK_ALREADY_COMPLETED";
        case ErrorCode::TASK_CANCELLED: return "TASK_CANCELLED";
        case ErrorCode::TASK_TIMEOUT: return "TASK_TIMEOUT";
        case ErrorCode::TASK_SUBMIT_FAILED: return "TASK_SUBMIT_FAILED";
        case ErrorCode::TASK_INVALID_STATE: return "TASK_INVALID_STATE";
        case ErrorCode::TASK_DEPENDENCY_FAILED: return "TASK_DEPENDENCY_FAILED";
        case ErrorCode::TASK_RESOURCE_UNAVAILABLE: return "TASK_RESOURCE_UNAVAILABLE";
        case ErrorCode::TASK_GPU_LIMIT_EXCEEDED: return "TASK_GPU_LIMIT_EXCEEDED";

        // General
        case ErrorCode::UNKNOWN_ERROR: return "UNKNOWN_ERROR";
        case ErrorCode::NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
        case ErrorCode::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case ErrorCode::OUT_OF_RANGE: return "OUT_OF_RANGE";
        case ErrorCode::FILE_IO_ERROR: return "FILE_IO_ERROR";
        case ErrorCode::MEMORY_ALLOCATION_FAILED: return "MEMORY_ALLOCATION_FAILED";

        default: return "UNKNOWN";
    }
}

} // namespace Utils
} // namespace WheelDL
