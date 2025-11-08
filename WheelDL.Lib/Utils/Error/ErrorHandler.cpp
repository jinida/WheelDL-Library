#include "pch.h"
#include "ErrorHandler.h"
#include <cstring>
#include <algorithm>
#include <sstream>

namespace WheelDL {
namespace Utils {

int ErrorHandler::handleException(const std::exception& e,
                                   char* errorBuffer,
                                   size_t bufferSize) {
    // Try to cast to WheelLibException
    const WheelLibException* wheelException = dynamic_cast<const WheelLibException*>(&e);

    ErrorCode code;
    std::string message;

    if (wheelException != nullptr) {
        // WheelLibException - use its error code and message
        code = wheelException->getErrorCode();
        message = formatErrorMessage(code, wheelException->getMessage());
    } else {
        // Standard exception - use UNKNOWN_ERROR
        code = ErrorCode::UNKNOWN_ERROR;
        message = formatErrorMessage(code, e.what());
    }

    // Copy message to buffer if provided
    // Note: safeCopyToBuffer handles nullptr gracefully
    safeCopyToBuffer(message, errorBuffer, bufferSize);

    return static_cast<int>(code);
}

int ErrorHandler::handleUnknownException(char* errorBuffer, size_t bufferSize) {
    std::string message = formatErrorMessage(
        ErrorCode::UNKNOWN_ERROR,
        "An unknown exception occurred"
    );

    // Copy message to buffer if provided
    // Note: safeCopyToBuffer handles nullptr gracefully
    safeCopyToBuffer(message, errorBuffer, bufferSize);

    return static_cast<int>(ErrorCode::UNKNOWN_ERROR);
}

std::string ErrorHandler::getErrorMessage(ErrorCode code) {
    switch (code) {
        // Configuration errors
        case ErrorCode::INVALID_CONFIG:
            return "Invalid configuration";
        case ErrorCode::CONFIG_PARSE_FAILED:
            return "Failed to parse configuration file";
        case ErrorCode::CONFIG_FILE_NOT_FOUND:
            return "Configuration file not found";
        case ErrorCode::CONFIG_VALIDATION_FAILED:
            return "Configuration validation failed";
        case ErrorCode::MISSING_CONFIG_KEY:
            return "Required configuration key is missing";
        case ErrorCode::INVALID_CONFIG_VALUE:
            return "Invalid configuration value";

        // Model errors
        case ErrorCode::MODEL_LOAD_FAILED:
            return "Failed to load model";
        case ErrorCode::MODEL_SAVE_FAILED:
            return "Failed to save model";
        case ErrorCode::MODEL_BUILD_FAILED:
            return "Failed to build model";
        case ErrorCode::MODEL_INVALID_ARCHITECTURE:
            return "Invalid model architecture";
        case ErrorCode::MODEL_FORWARD_FAILED:
            return "Model forward pass failed";
        case ErrorCode::MODEL_INFERENCE_FAILED:
            return "Model inference failed";

        // Data errors
        case ErrorCode::DATA_LOAD_FAILED:
            return "Failed to load data";
        case ErrorCode::DATA_INVALID_FORMAT:
            return "Invalid data format";
        case ErrorCode::DATA_FILE_NOT_FOUND:
            return "Data file not found";
        case ErrorCode::DATA_ANNOTATION_INVALID:
            return "Invalid annotation format";
        case ErrorCode::DATA_TRANSFORM_FAILED:
            return "Data transformation failed";
        case ErrorCode::DATA_CACHE_FAILED:
            return "Data caching failed";
        case ErrorCode::DATA_PREPROCESSING_FAILED:
            return "Data preprocessing failed";
        case ErrorCode::DATASET_NOT_FOUND:
            return "Dataset not found";
        case ErrorCode::INVALID_IMAGE_SIZE:
            return "Invalid image size";

        // GPU errors
        case ErrorCode::GPU_OUT_OF_MEMORY:
            return "GPU out of memory";
        case ErrorCode::GPU_NOT_AVAILABLE:
            return "GPU not available";
        case ErrorCode::GPU_INIT_FAILED:
            return "GPU initialization failed";
        case ErrorCode::GPU_ALLOCATION_FAILED:
            return "GPU memory allocation failed";
        case ErrorCode::GPU_CUDA_ERROR:
            return "CUDA error occurred";
        case ErrorCode::CUDNN_ERROR:
            return "cuDNN error occurred";
        case ErrorCode::GPU_DEVICE_MISMATCH:
            return "GPU device mismatch";

        // Training errors
        case ErrorCode::TRAINING_FAILED:
            return "Training failed";
        case ErrorCode::TRAINING_INTERRUPTED:
            return "Training was interrupted";
        case ErrorCode::TRAINING_DIVERGED:
            return "Training diverged (loss is NaN or infinite)";
        case ErrorCode::VALIDATION_FAILED:
            return "Validation failed";
        case ErrorCode::CHECKPOINT_SAVE_FAILED:
            return "Failed to save checkpoint";
        case ErrorCode::CHECKPOINT_LOAD_FAILED:
            return "Failed to load checkpoint";
        case ErrorCode::LOSS_IS_NAN:
            return "Loss is NaN";
        case ErrorCode::GRADIENT_EXPLOSION:
            return "Gradient explosion detected";

        // Distributed errors
        case ErrorCode::DISTRIBUTED_INIT_FAILED:
            return "Distributed training initialization failed";
        case ErrorCode::DISTRIBUTED_COMM_FAILED:
            return "Distributed communication failed";
        case ErrorCode::DISTRIBUTED_SYNC_FAILED:
            return "Distributed synchronization failed";
        case ErrorCode::DISTRIBUTED_RANK_ERROR:
            return "Invalid distributed rank";
        case ErrorCode::DISTRIBUTED_TIMEOUT:
            return "Distributed operation timed out";
        case ErrorCode::RANK_MISMATCH:
            return "Rank mismatch in distributed training";
        case ErrorCode::WORLD_SIZE_MISMATCH:
            return "World size mismatch in distributed training";

        // General errors
        case ErrorCode::UNKNOWN_ERROR:
            return "Unknown error occurred";
        case ErrorCode::NOT_IMPLEMENTED:
            return "Feature not implemented";
        case ErrorCode::INVALID_ARGUMENT:
            return "Invalid argument";
        case ErrorCode::OUT_OF_RANGE:
            return "Value out of range";
        case ErrorCode::FILE_IO_ERROR:
            return "File I/O error";
        case ErrorCode::MEMORY_ALLOCATION_FAILED:
            return "Memory allocation failed";

        case ErrorCode::SUCCESS:
            return "Operation successful";

        default:
            return "Unknown error";
    }
}

void ErrorHandler::safeCopyToBuffer(const std::string& message,
                                     char* buffer,
                                     size_t size) {
    if (buffer == nullptr || size == 0) {
        return;
    }

    // Calculate how many characters we can copy (leave room for null terminator)
    size_t maxCopy = size - 1;
    size_t copyLength = (std::min)(message.length(), maxCopy);

    // Copy the string
    std::memcpy(buffer, message.c_str(), copyLength);

    // Null terminate
    buffer[copyLength] = '\0';
}

std::string ErrorHandler::formatErrorMessage(ErrorCode code,
                                              const std::string& message) {
    std::ostringstream oss;
    oss << "[" << errorCodeToString(code) << "] "
        << getErrorMessage(code);

    if (!message.empty()) {
        oss << ": " << message;
    }

    return oss.str();
}

} // namespace Utils
} // namespace WheelDL
