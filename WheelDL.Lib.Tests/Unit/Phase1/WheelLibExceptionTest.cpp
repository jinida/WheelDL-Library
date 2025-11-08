#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include "WheelDL.Lib/Utils/Error/ErrorCodes.h"
#include <stdexcept>

using namespace WheelDL::Utils;

/**
 * @class WheelLibExceptionTest
 * @brief Unit tests for WheelLibException hierarchy
 *
 * Tests cover:
 * - Base exception class functionality
 * - All derived exception types
 * - Error code handling
 * - Nested exception support
 * - Error code to string conversion
 * - Exception message formatting
 */
class WheelLibExceptionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // No setup needed
    }

    void TearDown() override {
        // No cleanup needed
    }
};

// ============================================================================
// Base Exception Tests
// ============================================================================

TEST_F(WheelLibExceptionTest, BaseException_Constructor_WithCode) {
    // Arrange & Act
    WheelLibException ex(ErrorCode::UNKNOWN_ERROR, "Test error message");

    // Assert
    EXPECT_EQ(ex.getErrorCode(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_STREQ(ex.what(), "Test error message");
}

TEST_F(WheelLibExceptionTest, BaseException_WhatReturnsMessage) {
    // Arrange
    std::string message = "Detailed error description";
    WheelLibException ex(ErrorCode::UNKNOWN_ERROR, message);

    // Act
    const char* result = ex.what();

    // Assert
    EXPECT_STREQ(result, message.c_str());
}

TEST_F(WheelLibExceptionTest, BaseException_GetErrorCodeString) {
    // Arrange
    WheelLibException ex(ErrorCode::INVALID_CONFIG, "Config error");

    // Act
    std::string codeStr = ex.getErrorCodeString();

    // Assert
    EXPECT_FALSE(codeStr.empty());
    EXPECT_NE(codeStr.find("1000"), std::string::npos);  // INVALID_CONFIG = 1000
}

TEST_F(WheelLibExceptionTest, BaseException_EmptyMessage) {
    // Arrange & Act
    WheelLibException ex(ErrorCode::SUCCESS, "");

    // Assert
    EXPECT_STREQ(ex.what(), "");
}

TEST_F(WheelLibExceptionTest, BaseException_LongMessage) {
    // Arrange
    std::string longMessage(10000, 'x');

    // Act & Assert
    EXPECT_NO_THROW({
        WheelLibException ex(ErrorCode::UNKNOWN_ERROR, longMessage);
        EXPECT_EQ(std::string(ex.what()), longMessage);
    });
}

// ============================================================================
// Nested Exception Tests
// ============================================================================

TEST_F(WheelLibExceptionTest, NestedException_Constructor) {
    // Arrange
    std::exception_ptr innerEx;
    try {
        throw std::runtime_error("Inner exception");
    } catch (...) {
        innerEx = std::current_exception();
    }

    // Act
    WheelLibException ex(ErrorCode::UNKNOWN_ERROR, "Outer exception", innerEx);

    // Assert
    EXPECT_EQ(ex.getErrorCode(), ErrorCode::UNKNOWN_ERROR);
    EXPECT_STREQ(ex.what(), "Outer exception");
    EXPECT_NE(ex.getInnerException(), nullptr);
}

TEST_F(WheelLibExceptionTest, NestedException_RethrowInner) {
    // Arrange
    std::exception_ptr innerEx;
    try {
        throw std::runtime_error("Inner error");
    } catch (...) {
        innerEx = std::current_exception();
    }

    WheelLibException ex(ErrorCode::UNKNOWN_ERROR, "Outer error", innerEx);

    // Act & Assert
    EXPECT_THROW({
        if (ex.getInnerException()) {
            std::rethrow_exception(ex.getInnerException());
        }
    }, std::runtime_error);
}

TEST_F(WheelLibExceptionTest, NestedException_MultipleNesting) {
    // Arrange - Create nested chain
    std::exception_ptr level1Ex;
    try {
        throw std::runtime_error("Level 1");
    } catch (...) {
        level1Ex = std::current_exception();
    }

    std::exception_ptr level2Ex;
    try {
        throw WheelLibException(ErrorCode::DATA_LOAD_FAILED, "Level 2", level1Ex);
    } catch (...) {
        level2Ex = std::current_exception();
    }

    // Act
    WheelLibException level3Ex(ErrorCode::TRAINING_FAILED, "Level 3", level2Ex);

    // Assert
    EXPECT_EQ(level3Ex.getErrorCode(), ErrorCode::TRAINING_FAILED);
    EXPECT_NE(level3Ex.getInnerException(), nullptr);
}

// ============================================================================
// ConfigurationException Tests
// ============================================================================

TEST_F(WheelLibExceptionTest, ConfigurationException_BasicUsage) {
    // Act
    ConfigurationException ex(ErrorCode::INVALID_CONFIG, "Invalid config file");

    // Assert
    EXPECT_EQ(ex.getErrorCode(), ErrorCode::INVALID_CONFIG);
    EXPECT_STREQ(ex.what(), "Invalid config file");
}

TEST_F(WheelLibExceptionTest, ConfigurationException_AllErrorCodes) {
    // Test all configuration-related error codes
    EXPECT_NO_THROW({
        ConfigurationException(ErrorCode::INVALID_CONFIG, "Test");
        ConfigurationException(ErrorCode::CONFIG_PARSE_FAILED, "Test");
        ConfigurationException(ErrorCode::MISSING_CONFIG_KEY, "Test");
        ConfigurationException(ErrorCode::INVALID_CONFIG_VALUE, "Test");
    });
}

TEST_F(WheelLibExceptionTest, ConfigurationException_CatchAsBase) {
    // Act & Assert
    try {
        throw ConfigurationException(ErrorCode::INVALID_CONFIG, "Config error");
    } catch (const WheelLibException& ex) {
        EXPECT_EQ(ex.getErrorCode(), ErrorCode::INVALID_CONFIG);
        SUCCEED();
        return;
    }
    FAIL() << "Exception not caught as base class";
}

// ============================================================================
// ModelException Tests
// ============================================================================

TEST_F(WheelLibExceptionTest, ModelException_BasicUsage) {
    // Act
    ModelException ex(ErrorCode::MODEL_LOAD_FAILED, "Failed to load model");

    // Assert
    EXPECT_EQ(ex.getErrorCode(), ErrorCode::MODEL_LOAD_FAILED);
    EXPECT_STREQ(ex.what(), "Failed to load model");
}

TEST_F(WheelLibExceptionTest, ModelException_AllErrorCodes) {
    // Test all model-related error codes
    EXPECT_NO_THROW({
        ModelException(ErrorCode::MODEL_LOAD_FAILED, "Test");
        ModelException(ErrorCode::MODEL_SAVE_FAILED, "Test");
        ModelException(ErrorCode::INVALID_MODEL_ARCHITECTURE, "Test");
        ModelException(ErrorCode::MODEL_INFERENCE_FAILED, "Test");
    });
}

// ============================================================================
// DataException Tests
// ============================================================================

TEST_F(WheelLibExceptionTest, DataException_BasicUsage) {
    // Act
    DataException ex(ErrorCode::DATA_LOAD_FAILED, "Failed to load dataset");

    // Assert
    EXPECT_EQ(ex.getErrorCode(), ErrorCode::DATA_LOAD_FAILED);
    EXPECT_STREQ(ex.what(), "Failed to load dataset");
}

TEST_F(WheelLibExceptionTest, DataException_AllErrorCodes) {
    // Test all data-related error codes
    EXPECT_NO_THROW({
        DataException(ErrorCode::DATA_LOAD_FAILED, "Test");
        DataException(ErrorCode::INVALID_DATA_FORMAT, "Test");
        DataException(ErrorCode::DATA_PREPROCESSING_FAILED, "Test");
        DataException(ErrorCode::DATASET_NOT_FOUND, "Test");
        DataException(ErrorCode::INVALID_IMAGE_SIZE, "Test");
    });
}

// ============================================================================
// GPUException Tests
// ============================================================================

TEST_F(WheelLibExceptionTest, GPUException_BasicUsage) {
    // Act
    GPUException ex(ErrorCode::GPU_OUT_OF_MEMORY, "GPU memory exhausted");

    // Assert
    EXPECT_EQ(ex.getErrorCode(), ErrorCode::GPU_OUT_OF_MEMORY);
    EXPECT_STREQ(ex.what(), "GPU memory exhausted");
}

TEST_F(WheelLibExceptionTest, GPUException_AllErrorCodes) {
    // Test all GPU-related error codes
    EXPECT_NO_THROW({
        GPUException(ErrorCode::GPU_OUT_OF_MEMORY, "Test");
        GPUException(ErrorCode::GPU_NOT_AVAILABLE, "Test");
        GPUException(ErrorCode::CUDA_ERROR, "Test");
        GPUException(ErrorCode::CUDNN_ERROR, "Test");
        GPUException(ErrorCode::GPU_DEVICE_MISMATCH, "Test");
    });
}

// ============================================================================
// TrainingException Tests
// ============================================================================

TEST_F(WheelLibExceptionTest, TrainingException_BasicUsage) {
    // Act
    TrainingException ex(ErrorCode::TRAINING_FAILED, "Training interrupted");

    // Assert
    EXPECT_EQ(ex.getErrorCode(), ErrorCode::TRAINING_FAILED);
    EXPECT_STREQ(ex.what(), "Training interrupted");
}

TEST_F(WheelLibExceptionTest, TrainingException_AllErrorCodes) {
    // Test all training-related error codes
    EXPECT_NO_THROW({
        TrainingException(ErrorCode::TRAINING_FAILED, "Test");
        TrainingException(ErrorCode::VALIDATION_FAILED, "Test");
        TrainingException(ErrorCode::CHECKPOINT_SAVE_FAILED, "Test");
        TrainingException(ErrorCode::CHECKPOINT_LOAD_FAILED, "Test");
        TrainingException(ErrorCode::LOSS_IS_NAN, "Test");
        TrainingException(ErrorCode::GRADIENT_EXPLOSION, "Test");
    });
}

// ============================================================================
// DistributedException Tests
// ============================================================================

TEST_F(WheelLibExceptionTest, DistributedException_BasicUsage) {
    // Act
    DistributedException ex(ErrorCode::DISTRIBUTED_INIT_FAILED, "Failed to initialize");

    // Assert
    EXPECT_EQ(ex.getErrorCode(), ErrorCode::DISTRIBUTED_INIT_FAILED);
    EXPECT_STREQ(ex.what(), "Failed to initialize");
}

TEST_F(WheelLibExceptionTest, DistributedException_AllErrorCodes) {
    // Test all distributed-related error codes
    EXPECT_NO_THROW({
        DistributedException(ErrorCode::DISTRIBUTED_INIT_FAILED, "Test");
        DistributedException(ErrorCode::DISTRIBUTED_COMM_FAILED, "Test");
        DistributedException(ErrorCode::RANK_MISMATCH, "Test");
        DistributedException(ErrorCode::WORLD_SIZE_MISMATCH, "Test");
    });
}

// ============================================================================
// Error Code String Conversion Tests
// ============================================================================

TEST_F(WheelLibExceptionTest, ErrorCodeString_AllCodes) {
    // Test that all error codes convert to strings
    std::vector<ErrorCode> codes = {
        ErrorCode::SUCCESS,
        ErrorCode::INVALID_CONFIG,
        ErrorCode::MODEL_LOAD_FAILED,
        ErrorCode::DATA_LOAD_FAILED,
        ErrorCode::GPU_OUT_OF_MEMORY,
        ErrorCode::TRAINING_FAILED,
        ErrorCode::DISTRIBUTED_INIT_FAILED,
        ErrorCode::UNKNOWN_ERROR
    };

    for (auto code : codes) {
        WheelLibException ex(code, "Test");
        std::string codeStr = ex.getErrorCodeString();
        EXPECT_FALSE(codeStr.empty());
    }
}

TEST_F(WheelLibExceptionTest, ErrorCodeString_UniqueStrings) {
    // Different error codes should produce different strings
    WheelLibException ex1(ErrorCode::INVALID_CONFIG, "Test");
    WheelLibException ex2(ErrorCode::MODEL_LOAD_FAILED, "Test");

    EXPECT_NE(ex1.getErrorCodeString(), ex2.getErrorCodeString());
}

// ============================================================================
// Exception Throwing and Catching Tests
// ============================================================================

TEST_F(WheelLibExceptionTest, ThrowCatch_ConfigurationException) {
    // Act & Assert
    EXPECT_THROW({
        throw ConfigurationException(ErrorCode::INVALID_CONFIG, "Test");
    }, ConfigurationException);
}

TEST_F(WheelLibExceptionTest, ThrowCatch_AsBaseClass) {
    // Act & Assert
    EXPECT_THROW({
        throw ModelException(ErrorCode::MODEL_LOAD_FAILED, "Test");
    }, WheelLibException);
}

TEST_F(WheelLibExceptionTest, ThrowCatch_AsStdException) {
    // Act & Assert
    EXPECT_THROW({
        throw DataException(ErrorCode::DATA_LOAD_FAILED, "Test");
    }, std::exception);
}

TEST_F(WheelLibExceptionTest, ThrowCatch_PreserveTypeInfo) {
    // Arrange & Act
    try {
        throw GPUException(ErrorCode::GPU_OUT_OF_MEMORY, "GPU error");
    } catch (const WheelLibException& ex) {
        // Assert - Should be able to dynamic_cast back to derived type
        const GPUException* gpuEx = dynamic_cast<const GPUException*>(&ex);
        EXPECT_NE(gpuEx, nullptr);
        return;
    }
    FAIL() << "Exception not caught";
}

// ============================================================================
// Practical Usage Scenarios
// ============================================================================

TEST_F(WheelLibExceptionTest, Scenario_ConfigFileNotFound) {
    // Simulate: config file not found scenario
    try {
        throw ConfigurationException(
            ErrorCode::INVALID_CONFIG,
            "Configuration file not found: config.yaml"
        );
    } catch (const ConfigurationException& ex) {
        EXPECT_EQ(ex.getErrorCode(), ErrorCode::INVALID_CONFIG);
        EXPECT_NE(std::string(ex.what()).find("config.yaml"), std::string::npos);
        SUCCEED();
        return;
    }
    FAIL();
}

TEST_F(WheelLibExceptionTest, Scenario_ModelLoadWithInnerException) {
    // Simulate: model loading failed due to file I/O error
    std::exception_ptr innerEx;
    try {
        throw std::runtime_error("File not found: model.pth");
    } catch (...) {
        innerEx = std::current_exception();
    }

    try {
        throw ModelException(
            ErrorCode::MODEL_LOAD_FAILED,
            "Failed to load pretrained model",
            innerEx
        );
    } catch (const ModelException& ex) {
        EXPECT_EQ(ex.getErrorCode(), ErrorCode::MODEL_LOAD_FAILED);
        EXPECT_NE(ex.getInnerException(), nullptr);

        // Verify we can access inner exception
        try {
            std::rethrow_exception(ex.getInnerException());
        } catch (const std::runtime_error& inner) {
            EXPECT_NE(std::string(inner.what()).find("model.pth"), std::string::npos);
            SUCCEED();
            return;
        }
    }
    FAIL();
}

TEST_F(WheelLibExceptionTest, Scenario_GPUOutOfMemory) {
    // Simulate: GPU out of memory during training
    try {
        throw GPUException(
            ErrorCode::GPU_OUT_OF_MEMORY,
            "CUDA out of memory. Tried to allocate 2.50 GiB"
        );
    } catch (const GPUException& ex) {
        EXPECT_EQ(ex.getErrorCode(), ErrorCode::GPU_OUT_OF_MEMORY);
        std::string msg(ex.what());
        EXPECT_NE(msg.find("CUDA"), std::string::npos);
        EXPECT_NE(msg.find("GiB"), std::string::npos);
        SUCCEED();
        return;
    }
    FAIL();
}

TEST_F(WheelLibExceptionTest, Scenario_TrainingNaNLoss) {
    // Simulate: training diverged with NaN loss
    try {
        throw TrainingException(
            ErrorCode::LOSS_IS_NAN,
            "Training loss became NaN at epoch 5, step 120. Consider reducing learning rate."
        );
    } catch (const TrainingException& ex) {
        EXPECT_EQ(ex.getErrorCode(), ErrorCode::LOSS_IS_NAN);
        std::string msg(ex.what());
        EXPECT_NE(msg.find("NaN"), std::string::npos);
        EXPECT_NE(msg.find("epoch 5"), std::string::npos);
        SUCCEED();
        return;
    }
    FAIL();
}

TEST_F(WheelLibExceptionTest, Scenario_DistributedInitFailure) {
    // Simulate: distributed training initialization failed
    try {
        throw DistributedException(
            ErrorCode::DISTRIBUTED_INIT_FAILED,
            "Failed to initialize process group: connection timeout at rank 2"
        );
    } catch (const DistributedException& ex) {
        EXPECT_EQ(ex.getErrorCode(), ErrorCode::DISTRIBUTED_INIT_FAILED);
        EXPECT_NE(std::string(ex.what()).find("rank 2"), std::string::npos);
        SUCCEED();
        return;
    }
    FAIL();
}

// ============================================================================
// Error Code Ranges
// ============================================================================

TEST_F(WheelLibExceptionTest, ErrorCodeRanges_Configuration) {
    // Configuration errors should be in 1000-1999 range
    EXPECT_GE(static_cast<int>(ErrorCode::INVALID_CONFIG), 1000);
    EXPECT_LT(static_cast<int>(ErrorCode::INVALID_CONFIG), 2000);
}

TEST_F(WheelLibExceptionTest, ErrorCodeRanges_Model) {
    // Model errors should be in 2000-2999 range
    EXPECT_GE(static_cast<int>(ErrorCode::MODEL_LOAD_FAILED), 2000);
    EXPECT_LT(static_cast<int>(ErrorCode::MODEL_LOAD_FAILED), 3000);
}

TEST_F(WheelLibExceptionTest, ErrorCodeRanges_Data) {
    // Data errors should be in 3000-3999 range
    EXPECT_GE(static_cast<int>(ErrorCode::DATA_LOAD_FAILED), 3000);
    EXPECT_LT(static_cast<int>(ErrorCode::DATA_LOAD_FAILED), 4000);
}

TEST_F(WheelLibExceptionTest, ErrorCodeRanges_GPU) {
    // GPU errors should be in 4000-4999 range
    EXPECT_GE(static_cast<int>(ErrorCode::GPU_OUT_OF_MEMORY), 4000);
    EXPECT_LT(static_cast<int>(ErrorCode::GPU_OUT_OF_MEMORY), 5000);
}

TEST_F(WheelLibExceptionTest, ErrorCodeRanges_Training) {
    // Training errors should be in 5000-5999 range
    EXPECT_GE(static_cast<int>(ErrorCode::TRAINING_FAILED), 5000);
    EXPECT_LT(static_cast<int>(ErrorCode::TRAINING_FAILED), 6000);
}

TEST_F(WheelLibExceptionTest, ErrorCodeRanges_Distributed) {
    // Distributed errors should be in 6000-6999 range
    EXPECT_GE(static_cast<int>(ErrorCode::DISTRIBUTED_INIT_FAILED), 6000);
    EXPECT_LT(static_cast<int>(ErrorCode::DISTRIBUTED_INIT_FAILED), 7000);
}

// ============================================================================
// Copy and Move Semantics
// ============================================================================

TEST_F(WheelLibExceptionTest, CopyConstructor) {
    // Arrange
    WheelLibException original(ErrorCode::UNKNOWN_ERROR, "Original message");

    // Act
    WheelLibException copy(original);

    // Assert
    EXPECT_EQ(copy.getErrorCode(), original.getErrorCode());
    EXPECT_STREQ(copy.what(), original.what());
}

TEST_F(WheelLibExceptionTest, CopyAssignment) {
    // Arrange
    WheelLibException original(ErrorCode::INVALID_CONFIG, "Config error");
    WheelLibException copy(ErrorCode::SUCCESS, "");

    // Act
    copy = original;

    // Assert
    EXPECT_EQ(copy.getErrorCode(), original.getErrorCode());
    EXPECT_STREQ(copy.what(), original.what());
}
