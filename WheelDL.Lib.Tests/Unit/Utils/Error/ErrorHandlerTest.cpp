#include "pch.h"
#include "Utils/Error/ErrorHandler.h"
#include "Utils/Error/WheelLibException.h"
#include <cstring>
#include <gtest/gtest.h>

using namespace WheelDL::Utils;

// =============================================================================
// handleException Tests (EH-001 ~ EH-005)
// =============================================================================

// EH-001: handleException with WheelLibException returns correct error code
TEST(ErrorHandlerTest, HandleException_WheelLibException) {
    WheelLibException ex(ErrorCode::INVALID_CONFIG, "Config error");
    char buffer[256] = { 0 };

    int result = ErrorHandler::handleException(ex, buffer, sizeof(buffer));

    EXPECT_EQ(static_cast<int>(ErrorCode::INVALID_CONFIG), result);
    EXPECT_GT(strlen(buffer), 0u);
}

// EH-002: handleException with std::exception returns UNKNOWN_ERROR
TEST(ErrorHandlerTest, HandleException_StdException) {
    std::runtime_error ex("Runtime error");
    char buffer[256] = { 0 };

    int result = ErrorHandler::handleException(ex, buffer, sizeof(buffer));

    EXPECT_EQ(static_cast<int>(ErrorCode::UNKNOWN_ERROR), result);
    EXPECT_GT(strlen(buffer), 0u);
}

// EH-003: handleException fills buffer with message
TEST(ErrorHandlerTest, HandleException_BufferFilled) {
    WheelLibException ex(ErrorCode::MODEL_LOAD_FAILED, "Model not found");
    char buffer[256] = { 0 };

    ErrorHandler::handleException(ex, buffer, sizeof(buffer));

    EXPECT_GT(strlen(buffer), 0u);
    // Buffer should contain something related to the error
    std::string bufferStr(buffer);
    EXPECT_FALSE(bufferStr.empty());
}

// EH-004: handleException with null buffer doesn't crash
TEST(ErrorHandlerTest, HandleException_NullBuffer) {
    WheelLibException ex(ErrorCode::DATA_LOAD_FAILED, "Data error");

    EXPECT_NO_THROW({
        int result = ErrorHandler::handleException(ex, nullptr, 256);
        EXPECT_EQ(static_cast<int>(ErrorCode::DATA_LOAD_FAILED), result);
    });
}

// EH-005: handleException with zero buffer size doesn't crash
TEST(ErrorHandlerTest, HandleException_ZeroBufferSize) {
    WheelLibException ex(ErrorCode::GPU_OUT_OF_MEMORY, "OOM");
    char buffer[256] = { 0 };

    EXPECT_NO_THROW({
        int result = ErrorHandler::handleException(ex, buffer, 0);
        EXPECT_EQ(static_cast<int>(ErrorCode::GPU_OUT_OF_MEMORY), result);
    });
}

// =============================================================================
// handleUnknownException Tests (EH-006)
// =============================================================================

// EH-006: handleUnknownException returns UNKNOWN_ERROR
TEST(ErrorHandlerTest, HandleUnknownException_Returns) {
    char buffer[256] = { 0 };

    int result = ErrorHandler::handleUnknownException(buffer, sizeof(buffer));

    EXPECT_EQ(static_cast<int>(ErrorCode::UNKNOWN_ERROR), result);
    EXPECT_GT(strlen(buffer), 0u);
}

TEST(ErrorHandlerTest, HandleUnknownException_NullBuffer) {
    EXPECT_NO_THROW({
        int result = ErrorHandler::handleUnknownException(nullptr, 256);
        EXPECT_EQ(static_cast<int>(ErrorCode::UNKNOWN_ERROR), result);
    });
}

TEST(ErrorHandlerTest, HandleUnknownException_ZeroBufferSize) {
    char buffer[256] = { 0 };

    EXPECT_NO_THROW({
        int result = ErrorHandler::handleUnknownException(buffer, 0);
        EXPECT_EQ(static_cast<int>(ErrorCode::UNKNOWN_ERROR), result);
    });
}

// =============================================================================
// getErrorMessage Tests (EH-007 ~ EH-008)
// =============================================================================

// EH-007: getErrorMessage returns valid message for known code
TEST(ErrorHandlerTest, GetErrorMessage_ValidCode) {
    std::string msg = ErrorHandler::getErrorMessage(ErrorCode::INVALID_CONFIG);

    EXPECT_FALSE(msg.empty());
}

// EH-008: getErrorMessage returns message for unknown code
TEST(ErrorHandlerTest, GetErrorMessage_UnknownCode) {
    // Use a code that's unlikely to be defined
    std::string msg = ErrorHandler::getErrorMessage(static_cast<ErrorCode>(99999));

    EXPECT_FALSE(msg.empty());
}

TEST(ErrorHandlerTest, GetErrorMessage_SUCCESS) {
    std::string msg = ErrorHandler::getErrorMessage(ErrorCode::SUCCESS);

    EXPECT_FALSE(msg.empty());
}

TEST(ErrorHandlerTest, GetErrorMessage_VariousCodes) {
    // Test various error codes have non-empty messages
    EXPECT_FALSE(ErrorHandler::getErrorMessage(ErrorCode::CONFIG_PARSE_FAILED).empty());
    EXPECT_FALSE(ErrorHandler::getErrorMessage(ErrorCode::MODEL_LOAD_FAILED).empty());
    EXPECT_FALSE(ErrorHandler::getErrorMessage(ErrorCode::DATA_LOAD_FAILED).empty());
    EXPECT_FALSE(ErrorHandler::getErrorMessage(ErrorCode::GPU_OUT_OF_MEMORY).empty());
    EXPECT_FALSE(ErrorHandler::getErrorMessage(ErrorCode::TRAINING_FAILED).empty());
}

// =============================================================================
// safeCopyToBuffer Tests (EH-009 ~ EH-011)
// =============================================================================

// EH-009: safeCopyToBuffer normal copy
TEST(ErrorHandlerTest, SafeCopyToBuffer_Normal) {
    char buffer[256] = { 0 };
    std::string message = "Test message";

    ErrorHandler::safeCopyToBuffer(message, buffer, sizeof(buffer));

    EXPECT_STREQ("Test message", buffer);
    // Verify null termination
    EXPECT_EQ('\0', buffer[strlen(buffer)]);
}

// EH-010: safeCopyToBuffer truncation
TEST(ErrorHandlerTest, SafeCopyToBuffer_Truncation) {
    char buffer[10] = { 0 };
    std::string message = "This is a very long message that will be truncated";

    ErrorHandler::safeCopyToBuffer(message, buffer, sizeof(buffer));

    // Should be truncated but null-terminated
    EXPECT_EQ(9u, strlen(buffer));
    EXPECT_EQ('\0', buffer[9]);
}

// EH-011: safeCopyToBuffer exact fit
TEST(ErrorHandlerTest, SafeCopyToBuffer_ExactFit) {
    char buffer[12] = { 0 };
    std::string message = "Hello World";  // 11 chars + null = 12

    ErrorHandler::safeCopyToBuffer(message, buffer, sizeof(buffer));

    EXPECT_STREQ("Hello World", buffer);
}

TEST(ErrorHandlerTest, SafeCopyToBuffer_EmptyMessage) {
    char buffer[256] = "initial";
    std::string message = "";

    ErrorHandler::safeCopyToBuffer(message, buffer, sizeof(buffer));

    EXPECT_STREQ("", buffer);
}

TEST(ErrorHandlerTest, SafeCopyToBuffer_NullBuffer) {
    std::string message = "Test";

    EXPECT_NO_THROW({
        ErrorHandler::safeCopyToBuffer(message, nullptr, 256);
    });
}

TEST(ErrorHandlerTest, SafeCopyToBuffer_ZeroSize) {
    char buffer[256] = "initial";
    std::string message = "Test";

    EXPECT_NO_THROW({
        ErrorHandler::safeCopyToBuffer(message, buffer, 0);
    });
    // Buffer should be unchanged when size is 0
}

TEST(ErrorHandlerTest, SafeCopyToBuffer_SizeOne) {
    char buffer[256] = "initial";
    std::string message = "Test message";

    ErrorHandler::safeCopyToBuffer(message, buffer, 1);

    // Should only have null terminator
    EXPECT_EQ('\0', buffer[0]);
}

// =============================================================================
// formatErrorMessage Tests (EH-012)
// =============================================================================

// EH-012: formatErrorMessage returns formatted string
TEST(ErrorHandlerTest, FormatErrorMessage_Format) {
    std::string formatted = ErrorHandler::formatErrorMessage(
        ErrorCode::INVALID_CONFIG, "Configuration file not found");

    EXPECT_FALSE(formatted.empty());
    // Should contain the message
    EXPECT_NE(std::string::npos, formatted.find("Configuration file not found"));
}

TEST(ErrorHandlerTest, FormatErrorMessage_EmptyMessage) {
    std::string formatted = ErrorHandler::formatErrorMessage(ErrorCode::SUCCESS, "");

    // Should still contain error code information
    EXPECT_FALSE(formatted.empty());
}

TEST(ErrorHandlerTest, FormatErrorMessage_LongMessage) {
    std::string longMessage(5000, 'A');
    std::string formatted = ErrorHandler::formatErrorMessage(
        ErrorCode::UNKNOWN_ERROR, longMessage);

    EXPECT_FALSE(formatted.empty());
    EXPECT_NE(std::string::npos, formatted.find(longMessage));
}

// =============================================================================
// Integration Tests
// =============================================================================

TEST(ErrorHandlerTest, Integration_ExceptionToBuffer) {
    // Simulate DLL boundary error handling
    char errorBuffer[512] = { 0 };
    int errorCode = 0;

    try {
        throw WheelLibException(ErrorCode::MODEL_LOAD_FAILED, "Model file corrupted");
    }
    catch (const std::exception& e) {
        errorCode = ErrorHandler::handleException(e, errorBuffer, sizeof(errorBuffer));
    }

    EXPECT_EQ(static_cast<int>(ErrorCode::MODEL_LOAD_FAILED), errorCode);
    EXPECT_GT(strlen(errorBuffer), 0u);
}

TEST(ErrorHandlerTest, Integration_DerivedExceptionHandling) {
    char errorBuffer[512] = { 0 };

    ConfigurationException ex(ErrorCode::CONFIG_PARSE_FAILED, "YAML parse error");
    int errorCode = ErrorHandler::handleException(ex, errorBuffer, sizeof(errorBuffer));

    EXPECT_EQ(static_cast<int>(ErrorCode::CONFIG_PARSE_FAILED), errorCode);
}

// =============================================================================
// Static Utility Class Tests
// =============================================================================

TEST(ErrorHandlerStaticTest, CannotInstantiate) {
    // This is a compile-time check - the deleted constructor prevents instantiation
    // The following line would cause a compilation error if uncommented:
    // ErrorHandler handler;

    // Instead, verify all methods are static by calling without instance
    EXPECT_FALSE(ErrorHandler::getErrorMessage(ErrorCode::SUCCESS).empty());
}
