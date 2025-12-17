#pragma once

#include "ExportTypes.h"
#include "../Utils/Error/ErrorCodes.h"
#include "../Utils/Error/ErrorHandler.h"
#include "../Utils/Error/WheelLibException.h"
#include <cstring>

// ========== Export Macros ==========
#ifdef _WIN32
    #ifdef WHEELDL_STATIC
        // Static library build - no import/export needed
        #define WHEEL_API
    #elif defined(WHEELDL_EXPORTS)
        // DLL build - export symbols
        #define WHEEL_API __declspec(dllexport)
    #else
        // DLL consumer - import symbols
        #define WHEEL_API __declspec(dllimport)
    #endif
    #define WHEEL_CALL __stdcall
#else
    #define WHEEL_API __attribute__((visibility("default")))
    #define WHEEL_CALL
#endif

// ========== Error Handling Helper ==========
namespace WheelDL {
namespace Export {

/**
 * @brief Fill WheelError structure with error information
 * @param err Pointer to WheelError structure (may be null)
 * @param code Error code
 * @param category Exception category string
 * @param taskId Task ID (for TaskException)
 * @param message Error message
 */
inline void fillError(WheelError* err,
                      WheelDL::Utils::ErrorCode code,
                      const char* category,
                      const std::string& taskId,
                      const std::string& message) {
    if (!err) return;

    err->code = static_cast<int32_t>(code);
    WheelDL::Utils::ErrorHandler::safeCopyToBuffer(
        WheelDL::Utils::ErrorHandler::formatErrorMessage(code, message),
        err->message, sizeof(err->message));
    WheelDL::Utils::ErrorHandler::safeCopyToBuffer(category, err->category, sizeof(err->category));
    WheelDL::Utils::ErrorHandler::safeCopyToBuffer(taskId, err->taskId, sizeof(err->taskId));
}

/**
 * @brief Clear WheelError structure
 * @param err Pointer to WheelError structure
 */
inline void clearError(WheelError* err) {
    if (!err) return;
    err->code = 0;
    err->message[0] = '\0';
    err->category[0] = '\0';
    err->taskId[0] = '\0';
}

} // namespace Export
} // namespace WheelDL

// ========== Exception Handling Macros ==========

/**
 * @brief Begin safe block for exception handling
 */
#define WHEEL_SAFE_BEGIN() \
    try {

/**
 * @brief End safe block with exception handling
 * Uses existing ErrorHandler for consistent error formatting
 */
#define WHEEL_SAFE_END(errPtr) \
    } catch (const WheelDL::Utils::StopRequestedException&) { \
        WheelDL::Export::fillError(errPtr, WheelDL::Utils::ErrorCode::TASK_CANCELLED, "TASK", "", "Operation stopped by user"); \
        return WHEEL_RESULT_STOPPED; \
    } catch (const WheelDL::Utils::TaskException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "TASK", e.getTaskId(), e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::ConfigurationException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "CONFIG", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::ModelException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "MODEL", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::DataException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "DATA", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::GPUException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "GPU", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::TrainingException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "TRAINING", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::DistributedException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "DISTRIBUTED", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::WheelLibException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "GENERAL", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const std::exception& e) { \
        WheelDL::Export::fillError(errPtr, WheelDL::Utils::ErrorCode::UNKNOWN_ERROR, "UNKNOWN", "", e.what()); \
        return WHEEL_RESULT_ERROR; \
    } catch (...) { \
        WheelDL::Export::fillError(errPtr, WheelDL::Utils::ErrorCode::UNKNOWN_ERROR, "UNKNOWN", "", "Unknown exception"); \
        return WHEEL_RESULT_ERROR; \
    } \
    return WHEEL_RESULT_OK;

/**
 * @brief End safe block with custom success return
 */
#define WHEEL_SAFE_END_WITH_RESULT(errPtr, successResult) \
    } catch (const WheelDL::Utils::StopRequestedException&) { \
        WheelDL::Export::fillError(errPtr, WheelDL::Utils::ErrorCode::TASK_CANCELLED, "TASK", "", "Operation stopped by user"); \
        return WHEEL_RESULT_STOPPED; \
    } catch (const WheelDL::Utils::TaskException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "TASK", e.getTaskId(), e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::ConfigurationException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "CONFIG", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::ModelException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "MODEL", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::DataException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "DATA", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::GPUException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "GPU", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::TrainingException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "TRAINING", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::DistributedException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "DISTRIBUTED", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const WheelDL::Utils::WheelLibException& e) { \
        WheelDL::Export::fillError(errPtr, e.getErrorCode(), "GENERAL", "", e.getMessage()); \
        return WHEEL_RESULT_ERROR; \
    } catch (const std::exception& e) { \
        WheelDL::Export::fillError(errPtr, WheelDL::Utils::ErrorCode::UNKNOWN_ERROR, "UNKNOWN", "", e.what()); \
        return WHEEL_RESULT_ERROR; \
    } catch (...) { \
        WheelDL::Export::fillError(errPtr, WheelDL::Utils::ErrorCode::UNKNOWN_ERROR, "UNKNOWN", "", "Unknown exception"); \
        return WHEEL_RESULT_ERROR; \
    } \
    return successResult;
