#pragma once

#include "ErrorCodes.h"
#include "WheelLibException.h"
#include <string>
#include <exception>

namespace WheelDL {
namespace Utils {

/**
 * @class ErrorHandler
 * @brief Static utility class for error handling across DLL boundaries
 *
 * Provides safe error handling and message formatting for DLL API.
 * All methods are static and thread-safe.
 */
class ErrorHandler {
public:
    /**
     * @brief Handle exception and write error message to buffer
     * @param e Exception to handle
     * @param errorBuffer Output buffer for error message
     * @param bufferSize Size of error buffer
     * @return int Error code (cast from ErrorCode enum)
     *
     * @note If exception is WheelLibException, returns its error code.
     *       Otherwise returns UNKNOWN_ERROR.
     */
    static int handleException(const std::exception& e,
                              char* errorBuffer,
                              size_t bufferSize);

    /**
     * @brief Handle current exception (for use in catch(...) blocks)
     * @param errorBuffer Output buffer for error message
     * @param bufferSize Size of error buffer
     * @return int Error code (UNKNOWN_ERROR)
     */
    static int handleUnknownException(char* errorBuffer, size_t bufferSize);

    /**
     * @brief Get descriptive error message for error code
     * @param code Error code
     * @return std::string Human-readable error message
     */
    static std::string getErrorMessage(ErrorCode code);

    /**
     * @brief Safely copy string to buffer with null termination
     * @param message Message to copy
     * @param buffer Destination buffer
     * @param size Buffer size
     *
     * @note Always null-terminates the buffer, even if message is truncated.
     */
    static void safeCopyToBuffer(const std::string& message,
                                 char* buffer,
                                 size_t size);

    /**
     * @brief Format error message with context
     * @param code Error code
     * @param message Additional context message
     * @return std::string Formatted error message
     */
    static std::string formatErrorMessage(ErrorCode code,
                                         const std::string& message);

private:
    // Static utility class - no instances allowed
    ErrorHandler() = delete;
    ~ErrorHandler() = delete;
    ErrorHandler(const ErrorHandler&) = delete;
    ErrorHandler& operator=(const ErrorHandler&) = delete;
};

} // namespace Utils
} // namespace WheelDL
