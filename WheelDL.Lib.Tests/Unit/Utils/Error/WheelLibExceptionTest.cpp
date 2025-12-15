#include "pch.h"
#include "Utils/Error/WheelLibException.h"
#include <gtest/gtest.h>

using namespace WheelDL::Utils;

// =============================================================================
// WheelLibException Base Class Tests (EX-001 ~ EX-009)
// =============================================================================

// EX-001: Basic construction with code and message
TEST(WheelLibExceptionTest, BasicConstruction) {
    WheelLibException ex(ErrorCode::INVALID_CONFIG, "Test error message");

    EXPECT_EQ(ErrorCode::INVALID_CONFIG, ex.getErrorCode());
    EXPECT_EQ("Test error message", ex.getMessage());
    EXPECT_EQ(nullptr, ex.getInnerException());
}

// EX-002: Construction with inner exception
TEST(WheelLibExceptionTest, WithInnerException) {
    std::exception_ptr inner;
    try {
        throw std::runtime_error("Inner error");
    }
    catch (...) {
        inner = std::current_exception();
    }

    WheelLibException ex(ErrorCode::MODEL_LOAD_FAILED, "Outer error", inner);

    EXPECT_EQ(ErrorCode::MODEL_LOAD_FAILED, ex.getErrorCode());
    EXPECT_EQ("Outer error", ex.getMessage());
    EXPECT_NE(nullptr, ex.getInnerException());
}

// EX-003: what() returns message
TEST(WheelLibExceptionTest, What_ReturnsMessage) {
    WheelLibException ex(ErrorCode::DATA_LOAD_FAILED, "Data loading failed");

    EXPECT_STREQ("Data loading failed", ex.what());
}

// EX-004: getErrorCode returns stored code
TEST(WheelLibExceptionTest, GetErrorCode_ReturnsCode) {
    WheelLibException ex(ErrorCode::GPU_OUT_OF_MEMORY, "Out of memory");

    EXPECT_EQ(ErrorCode::GPU_OUT_OF_MEMORY, ex.getErrorCode());
}

// EX-005: getErrorCodeString returns valid string
TEST(WheelLibExceptionTest, GetErrorCodeString_Valid) {
    WheelLibException ex(ErrorCode::INVALID_CONFIG, "Config error");

    std::string codeStr = ex.getErrorCodeString();
    EXPECT_FALSE(codeStr.empty());
    EXPECT_NE(std::string::npos, codeStr.find("INVALID_CONFIG"));
}

// EX-006: getMessage returns stored message
TEST(WheelLibExceptionTest, GetMessage_ReturnsMessage) {
    WheelLibException ex(ErrorCode::TRAINING_FAILED, "Training error occurred");

    EXPECT_EQ("Training error occurred", ex.getMessage());
}

// EX-007: getInnerException returns nullptr when no inner
TEST(WheelLibExceptionTest, GetInnerException_Null) {
    WheelLibException ex(ErrorCode::SUCCESS, "No inner");

    EXPECT_EQ(nullptr, ex.getInnerException());
}

// EX-008: getInnerException returns valid pointer when set
TEST(WheelLibExceptionTest, GetInnerException_Valid) {
    std::exception_ptr inner;
    try {
        throw std::logic_error("Logic error");
    }
    catch (...) {
        inner = std::current_exception();
    }

    WheelLibException ex(ErrorCode::UNKNOWN_ERROR, "With inner", inner);

    EXPECT_NE(nullptr, ex.getInnerException());
}

// EX-009: Inner exception can be rethrown
TEST(WheelLibExceptionTest, InnerException_Rethrow) {
    std::exception_ptr inner;
    try {
        throw std::runtime_error("Original error");
    }
    catch (...) {
        inner = std::current_exception();
    }

    WheelLibException ex(ErrorCode::UNKNOWN_ERROR, "Wrapper", inner);

    EXPECT_THROW({
        std::rethrow_exception(ex.getInnerException());
    }, std::runtime_error);
}

// =============================================================================
// Derived Exception Classes Tests (EX-010 ~ EX-017)
// =============================================================================

// EX-010: ConfigurationException construction
TEST(ConfigurationExceptionTest, Construction) {
    ConfigurationException ex(ErrorCode::CONFIG_PARSE_FAILED, "Parse failed");

    EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, ex.getErrorCode());
    EXPECT_EQ("Parse failed", ex.getMessage());
}

// EX-011: ModelException construction
TEST(ModelExceptionTest, Construction) {
    ModelException ex(ErrorCode::MODEL_LOAD_FAILED, "Model not found");

    EXPECT_EQ(ErrorCode::MODEL_LOAD_FAILED, ex.getErrorCode());
    EXPECT_EQ("Model not found", ex.getMessage());
}

// EX-012: DataException construction
TEST(DataExceptionTest, Construction) {
    DataException ex(ErrorCode::DATA_LOAD_FAILED, "Data corrupted");

    EXPECT_EQ(ErrorCode::DATA_LOAD_FAILED, ex.getErrorCode());
    EXPECT_EQ("Data corrupted", ex.getMessage());
}

// EX-013: GPUException construction
TEST(GPUExceptionTest, Construction) {
    GPUException ex(ErrorCode::GPU_OUT_OF_MEMORY, "CUDA OOM");

    EXPECT_EQ(ErrorCode::GPU_OUT_OF_MEMORY, ex.getErrorCode());
    EXPECT_EQ("CUDA OOM", ex.getMessage());
}

// EX-014: TrainingException construction
TEST(TrainingExceptionTest, Construction) {
    TrainingException ex(ErrorCode::TRAINING_FAILED, "Gradient explosion");

    EXPECT_EQ(ErrorCode::TRAINING_FAILED, ex.getErrorCode());
    EXPECT_EQ("Gradient explosion", ex.getMessage());
}

// EX-015: DistributedException construction
TEST(DistributedExceptionTest, Construction) {
    DistributedException ex(ErrorCode::DISTRIBUTED_INIT_FAILED, "NCCL init failed");

    EXPECT_EQ(ErrorCode::DISTRIBUTED_INIT_FAILED, ex.getErrorCode());
    EXPECT_EQ("NCCL init failed", ex.getMessage());
}

// EX-016: TaskException with taskId
TEST(TaskExceptionTest, WithTaskId) {
    TaskException ex(ErrorCode::TASK_NOT_FOUND, "Task missing", "task-123");

    EXPECT_EQ(ErrorCode::TASK_NOT_FOUND, ex.getErrorCode());
    EXPECT_EQ("Task missing", ex.getMessage());
    EXPECT_EQ("task-123", ex.getTaskId());
}

// EX-017: TaskException without taskId
TEST(TaskExceptionTest, EmptyTaskId) {
    TaskException ex(ErrorCode::TASK_NOT_FOUND, "Task error");

    EXPECT_EQ("", ex.getTaskId());
}

// TaskException with inner exception
TEST(TaskExceptionTest, WithInnerException) {
    std::exception_ptr inner;
    try {
        throw std::runtime_error("Inner task error");
    }
    catch (...) {
        inner = std::current_exception();
    }

    TaskException ex(ErrorCode::TASK_NOT_FOUND, "Outer error", "task-456", inner);

    EXPECT_EQ("task-456", ex.getTaskId());
    EXPECT_NE(nullptr, ex.getInnerException());
}

// =============================================================================
// StopRequestedException Tests (EX-018)
// =============================================================================

// EX-018: StopRequestedException what() returns message
TEST(StopRequestedExceptionTest, What) {
    StopRequestedException ex;

    EXPECT_STREQ("Operation stopped by user request", ex.what());
}

// =============================================================================
// Polymorphism and Catch Tests (EX-019 ~ EX-022)
// =============================================================================

// EX-019: Exception caught as std::exception
TEST(WheelLibExceptionTest, CatchAsStdException) {
    bool caught = false;
    try {
        throw WheelLibException(ErrorCode::UNKNOWN_ERROR, "Test");
    }
    catch (const std::exception& e) {
        caught = true;
        EXPECT_STREQ("Test", e.what());
    }
    EXPECT_TRUE(caught);
}

// EX-020: Polymorphism test - derived caught as base
TEST(WheelLibExceptionTest, Polymorphism) {
    bool caught = false;
    try {
        throw ConfigurationException(ErrorCode::INVALID_CONFIG, "Config error");
    }
    catch (const WheelLibException& e) {
        caught = true;
        EXPECT_EQ(ErrorCode::INVALID_CONFIG, e.getErrorCode());
    }
    EXPECT_TRUE(caught);
}

// EX-021: getErrorCodeString format includes code name and value
TEST(WheelLibExceptionTest, GetErrorCodeString_Format) {
    WheelLibException ex(ErrorCode::INVALID_CONFIG, "Test");

    std::string codeStr = ex.getErrorCodeString();
    // Should contain the code name
    EXPECT_NE(std::string::npos, codeStr.find("INVALID_CONFIG"));
    // Should contain the numeric value (1000)
    EXPECT_NE(std::string::npos, codeStr.find("1000"));
}

// EX-022: what() returns _message.c_str()
TEST(WheelLibExceptionTest, What_ReturnsMessageCStr) {
    std::string message = "This is a test message";
    WheelLibException ex(ErrorCode::SUCCESS, message);

    // Verify what() returns the same content as getMessage()
    EXPECT_STREQ(ex.getMessage().c_str(), ex.what());
}

// =============================================================================
// Derived Exceptions with Inner Exception Tests
// =============================================================================

TEST(ConfigurationExceptionTest, WithInnerException) {
    std::exception_ptr inner;
    try {
        throw std::runtime_error("File not found");
    }
    catch (...) {
        inner = std::current_exception();
    }

    ConfigurationException ex(ErrorCode::CONFIG_FILE_NOT_FOUND, "Config file missing", inner);

    EXPECT_NE(nullptr, ex.getInnerException());
}

TEST(ModelExceptionTest, WithInnerException) {
    std::exception_ptr inner;
    try {
        throw std::runtime_error("Serialization error");
    }
    catch (...) {
        inner = std::current_exception();
    }

    ModelException ex(ErrorCode::MODEL_SAVE_FAILED, "Failed to save model", inner);

    EXPECT_NE(nullptr, ex.getInnerException());
}

TEST(DataExceptionTest, WithInnerException) {
    std::exception_ptr inner;
    try {
        throw std::runtime_error("IO error");
    }
    catch (...) {
        inner = std::current_exception();
    }

    DataException ex(ErrorCode::DATA_LOAD_FAILED, "Data load error", inner);

    EXPECT_NE(nullptr, ex.getInnerException());
}

TEST(GPUExceptionTest, WithInnerException) {
    std::exception_ptr inner;
    try {
        throw std::runtime_error("CUDA error");
    }
    catch (...) {
        inner = std::current_exception();
    }

    GPUException ex(ErrorCode::CUDA_ERROR, "CUDA operation failed", inner);

    EXPECT_NE(nullptr, ex.getInnerException());
}

TEST(TrainingExceptionTest, WithInnerException) {
    std::exception_ptr inner;
    try {
        throw std::runtime_error("NaN detected");
    }
    catch (...) {
        inner = std::current_exception();
    }

    TrainingException ex(ErrorCode::TRAINING_FAILED, "Training diverged", inner);

    EXPECT_NE(nullptr, ex.getInnerException());
}

TEST(DistributedExceptionTest, WithInnerException) {
    std::exception_ptr inner;
    try {
        throw std::runtime_error("Network error");
    }
    catch (...) {
        inner = std::current_exception();
    }

    DistributedException ex(ErrorCode::DISTRIBUTED_INIT_FAILED, "Distributed init failed", inner);

    EXPECT_NE(nullptr, ex.getInnerException());
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(WheelLibExceptionTest, EmptyMessage) {
    WheelLibException ex(ErrorCode::SUCCESS, "");

    EXPECT_EQ("", ex.getMessage());
    EXPECT_STREQ("", ex.what());
}

TEST(WheelLibExceptionTest, LongMessage) {
    std::string longMessage(10000, 'A');
    WheelLibException ex(ErrorCode::UNKNOWN_ERROR, longMessage);

    EXPECT_EQ(longMessage, ex.getMessage());
    EXPECT_STREQ(longMessage.c_str(), ex.what());
}

TEST(WheelLibExceptionTest, UnicodeMessage) {
    std::string unicodeMessage = "Error: Korean";
    WheelLibException ex(ErrorCode::UNKNOWN_ERROR, unicodeMessage);

    EXPECT_EQ(unicodeMessage, ex.getMessage());
}
