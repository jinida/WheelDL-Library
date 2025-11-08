#pragma once

#include "ErrorCodes.h"
#include <exception>
#include <string>
#include <memory>

namespace WheelDL {
namespace Utils {

/**
 * @class WheelLibException
 * @brief Base exception class for WheelDL.Lib
 *
 * Provides error code, message, and optional inner exception support.
 * All exceptions in WheelDL.Lib inherit from this class.
 *
 * @note Follows exception hierarchy pattern
 */
class WheelLibException : public std::exception {
public:
    /**
     * @brief Construct exception with error code and message
     * @param code Error code
     * @param message Error message
     */
    WheelLibException(ErrorCode code, const std::string& message);

    /**
     * @brief Construct exception with error code, message, and inner exception
     * @param code Error code
     * @param message Error message
     * @param innerException Inner exception pointer
     */
    WheelLibException(ErrorCode code, const std::string& message,
                      std::exception_ptr innerException);

    /**
     * @brief Virtual destructor
     */
    virtual ~WheelLibException() = default;

    /**
     * @brief Get error message
     * @return const char* Error message C-string
     */
    const char* what() const noexcept override;

    /**
     * @brief Get error code
     * @return ErrorCode Error code enum value
     */
    ErrorCode getErrorCode() const noexcept { return _errorCode; }

    /**
     * @brief Get error code as string
     * @return std::string Error code name
     */
    std::string getErrorCodeString() const noexcept;

    /**
     * @brief Get error message
     * @return std::string Error message
     */
    std::string getMessage() const noexcept { return _message; }

    /**
     * @brief Get inner exception
     * @return std::exception_ptr Inner exception pointer (may be null)
     */
    std::exception_ptr getInnerException() const noexcept { return _innerException; }

protected:
    ErrorCode _errorCode;
    std::string _message;
    std::exception_ptr _innerException;
};

/**
 * @class ConfigurationException
 * @brief Exception for configuration-related errors
 */
class ConfigurationException : public WheelLibException {
public:
    ConfigurationException(ErrorCode code, const std::string& message)
        : WheelLibException(code, message) {}

    ConfigurationException(ErrorCode code, const std::string& message,
                          std::exception_ptr innerException)
        : WheelLibException(code, message, innerException) {}
};

/**
 * @class ModelException
 * @brief Exception for model-related errors
 */
class ModelException : public WheelLibException {
public:
    ModelException(ErrorCode code, const std::string& message)
        : WheelLibException(code, message) {}

    ModelException(ErrorCode code, const std::string& message,
                  std::exception_ptr innerException)
        : WheelLibException(code, message, innerException) {}
};

/**
 * @class DataException
 * @brief Exception for data-related errors
 */
class DataException : public WheelLibException {
public:
    DataException(ErrorCode code, const std::string& message)
        : WheelLibException(code, message) {}

    DataException(ErrorCode code, const std::string& message,
                 std::exception_ptr innerException)
        : WheelLibException(code, message, innerException) {}
};

/**
 * @class GPUException
 * @brief Exception for GPU-related errors
 */
class GPUException : public WheelLibException {
public:
    GPUException(ErrorCode code, const std::string& message)
        : WheelLibException(code, message) {}

    GPUException(ErrorCode code, const std::string& message,
                std::exception_ptr innerException)
        : WheelLibException(code, message, innerException) {}
};

/**
 * @class TrainingException
 * @brief Exception for training-related errors
 */
class TrainingException : public WheelLibException {
public:
    TrainingException(ErrorCode code, const std::string& message)
        : WheelLibException(code, message) {}

    TrainingException(ErrorCode code, const std::string& message,
                     std::exception_ptr innerException)
        : WheelLibException(code, message, innerException) {}
};

/**
 * @class DistributedException
 * @brief Exception for distributed training errors
 */
class DistributedException : public WheelLibException {
public:
    DistributedException(ErrorCode code, const std::string& message)
        : WheelLibException(code, message) {}

    DistributedException(ErrorCode code, const std::string& message,
                        std::exception_ptr innerException)
        : WheelLibException(code, message, innerException) {}
};

} // namespace Utils
} // namespace WheelDL
