#include "pch.h"
#include "Utils/Error/ErrorCodes.h"
#include <gtest/gtest.h>

using namespace WheelDL::Utils;

// =============================================================================
// errorCodeToString Tests (ERR-001 ~ ERR-007)
// =============================================================================

// ERR-001: errorCodeToString returns correct string for SUCCESS
TEST(ErrorCodesTest, ErrorCodeToString_Success) {
    EXPECT_STREQ("SUCCESS", errorCodeToString(ErrorCode::SUCCESS));
}

// ERR-002: errorCodeToString returns correct string for Configuration errors
TEST(ErrorCodesTest, ErrorCodeToString_ConfigurationErrors) {
    EXPECT_STREQ("INVALID_CONFIG", errorCodeToString(ErrorCode::INVALID_CONFIG));
    EXPECT_STREQ("CONFIG_PARSE_FAILED", errorCodeToString(ErrorCode::CONFIG_PARSE_FAILED));
    EXPECT_STREQ("CONFIG_FILE_NOT_FOUND", errorCodeToString(ErrorCode::CONFIG_FILE_NOT_FOUND));
    EXPECT_STREQ("CONFIG_VALIDATION_FAILED", errorCodeToString(ErrorCode::CONFIG_VALIDATION_FAILED));
    EXPECT_STREQ("MISSING_CONFIG_KEY", errorCodeToString(ErrorCode::MISSING_CONFIG_KEY));
    EXPECT_STREQ("INVALID_CONFIG_VALUE", errorCodeToString(ErrorCode::INVALID_CONFIG_VALUE));
}

// ERR-003: errorCodeToString returns correct string for Model errors
TEST(ErrorCodesTest, ErrorCodeToString_ModelErrors) {
    EXPECT_STREQ("MODEL_LOAD_FAILED", errorCodeToString(ErrorCode::MODEL_LOAD_FAILED));
    EXPECT_STREQ("MODEL_SAVE_FAILED", errorCodeToString(ErrorCode::MODEL_SAVE_FAILED));
    EXPECT_STREQ("MODEL_BUILD_FAILED", errorCodeToString(ErrorCode::MODEL_BUILD_FAILED));
    EXPECT_STREQ("MODEL_INVALID_ARCHITECTURE", errorCodeToString(ErrorCode::MODEL_INVALID_ARCHITECTURE));
    EXPECT_STREQ("MODEL_FORWARD_FAILED", errorCodeToString(ErrorCode::MODEL_FORWARD_FAILED));
    EXPECT_STREQ("MODEL_INFERENCE_FAILED", errorCodeToString(ErrorCode::MODEL_INFERENCE_FAILED));
}

// ERR-004: errorCodeToString returns correct string for Data errors
TEST(ErrorCodesTest, ErrorCodeToString_DataErrors) {
    EXPECT_STREQ("DATA_LOAD_FAILED", errorCodeToString(ErrorCode::DATA_LOAD_FAILED));
    EXPECT_STREQ("DATA_INVALID_FORMAT", errorCodeToString(ErrorCode::DATA_INVALID_FORMAT));
    EXPECT_STREQ("DATA_FILE_NOT_FOUND", errorCodeToString(ErrorCode::DATA_FILE_NOT_FOUND));
    EXPECT_STREQ("DATA_ANNOTATION_INVALID", errorCodeToString(ErrorCode::DATA_ANNOTATION_INVALID));
    EXPECT_STREQ("DATA_TRANSFORM_FAILED", errorCodeToString(ErrorCode::DATA_TRANSFORM_FAILED));
    EXPECT_STREQ("DATA_CACHE_FAILED", errorCodeToString(ErrorCode::DATA_CACHE_FAILED));
    EXPECT_STREQ("DATA_PREPROCESSING_FAILED", errorCodeToString(ErrorCode::DATA_PREPROCESSING_FAILED));
    EXPECT_STREQ("DATASET_NOT_FOUND", errorCodeToString(ErrorCode::DATASET_NOT_FOUND));
    EXPECT_STREQ("INVALID_IMAGE_SIZE", errorCodeToString(ErrorCode::INVALID_IMAGE_SIZE));
}

// ERR-005: errorCodeToString returns correct string for GPU errors
TEST(ErrorCodesTest, ErrorCodeToString_GPUErrors) {
    EXPECT_STREQ("GPU_OUT_OF_MEMORY", errorCodeToString(ErrorCode::GPU_OUT_OF_MEMORY));
    EXPECT_STREQ("GPU_NOT_AVAILABLE", errorCodeToString(ErrorCode::GPU_NOT_AVAILABLE));
    EXPECT_STREQ("GPU_INIT_FAILED", errorCodeToString(ErrorCode::GPU_INIT_FAILED));
    EXPECT_STREQ("GPU_ALLOCATION_FAILED", errorCodeToString(ErrorCode::GPU_ALLOCATION_FAILED));
    EXPECT_STREQ("GPU_CUDA_ERROR", errorCodeToString(ErrorCode::GPU_CUDA_ERROR));
    EXPECT_STREQ("CUDNN_ERROR", errorCodeToString(ErrorCode::CUDNN_ERROR));
    EXPECT_STREQ("GPU_DEVICE_MISMATCH", errorCodeToString(ErrorCode::GPU_DEVICE_MISMATCH));
}

// ERR-006: errorCodeToString returns correct string for Training errors
TEST(ErrorCodesTest, ErrorCodeToString_TrainingErrors) {
    EXPECT_STREQ("TRAINING_FAILED", errorCodeToString(ErrorCode::TRAINING_FAILED));
    EXPECT_STREQ("TRAINING_INTERRUPTED", errorCodeToString(ErrorCode::TRAINING_INTERRUPTED));
    EXPECT_STREQ("TRAINING_DIVERGED", errorCodeToString(ErrorCode::TRAINING_DIVERGED));
    EXPECT_STREQ("VALIDATION_FAILED", errorCodeToString(ErrorCode::VALIDATION_FAILED));
    EXPECT_STREQ("CHECKPOINT_SAVE_FAILED", errorCodeToString(ErrorCode::CHECKPOINT_SAVE_FAILED));
    EXPECT_STREQ("CHECKPOINT_LOAD_FAILED", errorCodeToString(ErrorCode::CHECKPOINT_LOAD_FAILED));
    EXPECT_STREQ("LOSS_IS_NAN", errorCodeToString(ErrorCode::LOSS_IS_NAN));
    EXPECT_STREQ("GRADIENT_EXPLOSION", errorCodeToString(ErrorCode::GRADIENT_EXPLOSION));
}

// ERR-007: errorCodeToString returns correct string for Distributed errors
TEST(ErrorCodesTest, ErrorCodeToString_DistributedErrors) {
    EXPECT_STREQ("DISTRIBUTED_INIT_FAILED", errorCodeToString(ErrorCode::DISTRIBUTED_INIT_FAILED));
    EXPECT_STREQ("DISTRIBUTED_COMM_FAILED", errorCodeToString(ErrorCode::DISTRIBUTED_COMM_FAILED));
    EXPECT_STREQ("DISTRIBUTED_SYNC_FAILED", errorCodeToString(ErrorCode::DISTRIBUTED_SYNC_FAILED));
    EXPECT_STREQ("DISTRIBUTED_RANK_ERROR", errorCodeToString(ErrorCode::DISTRIBUTED_RANK_ERROR));
    EXPECT_STREQ("DISTRIBUTED_TIMEOUT", errorCodeToString(ErrorCode::DISTRIBUTED_TIMEOUT));
    EXPECT_STREQ("RANK_MISMATCH", errorCodeToString(ErrorCode::RANK_MISMATCH));
    EXPECT_STREQ("WORLD_SIZE_MISMATCH", errorCodeToString(ErrorCode::WORLD_SIZE_MISMATCH));
}

// ERR-007b: errorCodeToString returns correct string for Task management errors
TEST(ErrorCodesTest, ErrorCodeToString_TaskErrors) {
    EXPECT_STREQ("TASK_NOT_FOUND", errorCodeToString(ErrorCode::TASK_NOT_FOUND));
    EXPECT_STREQ("TASK_ALREADY_RUNNING", errorCodeToString(ErrorCode::TASK_ALREADY_RUNNING));
    EXPECT_STREQ("TASK_ALREADY_COMPLETED", errorCodeToString(ErrorCode::TASK_ALREADY_COMPLETED));
    EXPECT_STREQ("TASK_CANCELLED", errorCodeToString(ErrorCode::TASK_CANCELLED));
    EXPECT_STREQ("TASK_TIMEOUT", errorCodeToString(ErrorCode::TASK_TIMEOUT));
    EXPECT_STREQ("TASK_SUBMIT_FAILED", errorCodeToString(ErrorCode::TASK_SUBMIT_FAILED));
    EXPECT_STREQ("TASK_INVALID_STATE", errorCodeToString(ErrorCode::TASK_INVALID_STATE));
    EXPECT_STREQ("TASK_DEPENDENCY_FAILED", errorCodeToString(ErrorCode::TASK_DEPENDENCY_FAILED));
    EXPECT_STREQ("TASK_RESOURCE_UNAVAILABLE", errorCodeToString(ErrorCode::TASK_RESOURCE_UNAVAILABLE));
    EXPECT_STREQ("TASK_GPU_LIMIT_EXCEEDED", errorCodeToString(ErrorCode::TASK_GPU_LIMIT_EXCEEDED));
}

// ERR-007c: errorCodeToString returns correct string for General errors
TEST(ErrorCodesTest, ErrorCodeToString_GeneralErrors) {
    EXPECT_STREQ("UNKNOWN_ERROR", errorCodeToString(ErrorCode::UNKNOWN_ERROR));
    EXPECT_STREQ("NOT_IMPLEMENTED", errorCodeToString(ErrorCode::NOT_IMPLEMENTED));
    EXPECT_STREQ("INVALID_ARGUMENT", errorCodeToString(ErrorCode::INVALID_ARGUMENT));
    EXPECT_STREQ("OUT_OF_RANGE", errorCodeToString(ErrorCode::OUT_OF_RANGE));
    EXPECT_STREQ("FILE_IO_ERROR", errorCodeToString(ErrorCode::FILE_IO_ERROR));
    EXPECT_STREQ("MEMORY_ALLOCATION_FAILED", errorCodeToString(ErrorCode::MEMORY_ALLOCATION_FAILED));
}

// =============================================================================
// Error Code Value Range Tests (ERR-008 ~ ERR-011)
// =============================================================================

// ERR-008: SUCCESS value is 0
TEST(ErrorCodesTest, ErrorCode_SuccessValue) {
    EXPECT_EQ(0, static_cast<int>(ErrorCode::SUCCESS));
}

// ERR-009: Configuration error values in correct range (1000-1999)
TEST(ErrorCodesTest, ErrorCode_ConfigurationRange) {
    EXPECT_GE(static_cast<int>(ErrorCode::INVALID_CONFIG), 1000);
    EXPECT_LT(static_cast<int>(ErrorCode::INVALID_CONFIG), 2000);

    EXPECT_GE(static_cast<int>(ErrorCode::CONFIG_PARSE_FAILED), 1000);
    EXPECT_LT(static_cast<int>(ErrorCode::CONFIG_PARSE_FAILED), 2000);

    EXPECT_GE(static_cast<int>(ErrorCode::CONFIG_FILE_NOT_FOUND), 1000);
    EXPECT_LT(static_cast<int>(ErrorCode::CONFIG_FILE_NOT_FOUND), 2000);

    EXPECT_GE(static_cast<int>(ErrorCode::CONFIG_VALIDATION_FAILED), 1000);
    EXPECT_LT(static_cast<int>(ErrorCode::CONFIG_VALIDATION_FAILED), 2000);

    EXPECT_GE(static_cast<int>(ErrorCode::MISSING_CONFIG_KEY), 1000);
    EXPECT_LT(static_cast<int>(ErrorCode::MISSING_CONFIG_KEY), 2000);

    EXPECT_GE(static_cast<int>(ErrorCode::INVALID_CONFIG_VALUE), 1000);
    EXPECT_LT(static_cast<int>(ErrorCode::INVALID_CONFIG_VALUE), 2000);
}

// ERR-009b: Model error values in correct range (2000-2999)
TEST(ErrorCodesTest, ErrorCode_ModelRange) {
    EXPECT_GE(static_cast<int>(ErrorCode::MODEL_LOAD_FAILED), 2000);
    EXPECT_LT(static_cast<int>(ErrorCode::MODEL_LOAD_FAILED), 3000);

    EXPECT_GE(static_cast<int>(ErrorCode::MODEL_SAVE_FAILED), 2000);
    EXPECT_LT(static_cast<int>(ErrorCode::MODEL_SAVE_FAILED), 3000);

    EXPECT_GE(static_cast<int>(ErrorCode::MODEL_BUILD_FAILED), 2000);
    EXPECT_LT(static_cast<int>(ErrorCode::MODEL_BUILD_FAILED), 3000);

    EXPECT_GE(static_cast<int>(ErrorCode::MODEL_INVALID_ARCHITECTURE), 2000);
    EXPECT_LT(static_cast<int>(ErrorCode::MODEL_INVALID_ARCHITECTURE), 3000);

    EXPECT_GE(static_cast<int>(ErrorCode::MODEL_FORWARD_FAILED), 2000);
    EXPECT_LT(static_cast<int>(ErrorCode::MODEL_FORWARD_FAILED), 3000);

    EXPECT_GE(static_cast<int>(ErrorCode::MODEL_INFERENCE_FAILED), 2000);
    EXPECT_LT(static_cast<int>(ErrorCode::MODEL_INFERENCE_FAILED), 3000);
}

// ERR-009c: Data error values in correct range (3000-3999)
TEST(ErrorCodesTest, ErrorCode_DataRange) {
    EXPECT_GE(static_cast<int>(ErrorCode::DATA_LOAD_FAILED), 3000);
    EXPECT_LT(static_cast<int>(ErrorCode::DATA_LOAD_FAILED), 4000);

    EXPECT_GE(static_cast<int>(ErrorCode::DATA_INVALID_FORMAT), 3000);
    EXPECT_LT(static_cast<int>(ErrorCode::DATA_INVALID_FORMAT), 4000);

    EXPECT_GE(static_cast<int>(ErrorCode::DATA_FILE_NOT_FOUND), 3000);
    EXPECT_LT(static_cast<int>(ErrorCode::DATA_FILE_NOT_FOUND), 4000);

    EXPECT_GE(static_cast<int>(ErrorCode::DATASET_NOT_FOUND), 3000);
    EXPECT_LT(static_cast<int>(ErrorCode::DATASET_NOT_FOUND), 4000);

    EXPECT_GE(static_cast<int>(ErrorCode::INVALID_IMAGE_SIZE), 3000);
    EXPECT_LT(static_cast<int>(ErrorCode::INVALID_IMAGE_SIZE), 4000);
}

// ERR-009d: GPU error values in correct range (4000-4999)
TEST(ErrorCodesTest, ErrorCode_GPURange) {
    EXPECT_GE(static_cast<int>(ErrorCode::GPU_OUT_OF_MEMORY), 4000);
    EXPECT_LT(static_cast<int>(ErrorCode::GPU_OUT_OF_MEMORY), 5000);

    EXPECT_GE(static_cast<int>(ErrorCode::GPU_NOT_AVAILABLE), 4000);
    EXPECT_LT(static_cast<int>(ErrorCode::GPU_NOT_AVAILABLE), 5000);

    EXPECT_GE(static_cast<int>(ErrorCode::GPU_INIT_FAILED), 4000);
    EXPECT_LT(static_cast<int>(ErrorCode::GPU_INIT_FAILED), 5000);

    EXPECT_GE(static_cast<int>(ErrorCode::GPU_CUDA_ERROR), 4000);
    EXPECT_LT(static_cast<int>(ErrorCode::GPU_CUDA_ERROR), 5000);

    EXPECT_GE(static_cast<int>(ErrorCode::CUDNN_ERROR), 4000);
    EXPECT_LT(static_cast<int>(ErrorCode::CUDNN_ERROR), 5000);
}

// ERR-009e: Training error values in correct range (5000-5999)
TEST(ErrorCodesTest, ErrorCode_TrainingRange) {
    EXPECT_GE(static_cast<int>(ErrorCode::TRAINING_FAILED), 5000);
    EXPECT_LT(static_cast<int>(ErrorCode::TRAINING_FAILED), 6000);

    EXPECT_GE(static_cast<int>(ErrorCode::TRAINING_INTERRUPTED), 5000);
    EXPECT_LT(static_cast<int>(ErrorCode::TRAINING_INTERRUPTED), 6000);

    EXPECT_GE(static_cast<int>(ErrorCode::CHECKPOINT_SAVE_FAILED), 5000);
    EXPECT_LT(static_cast<int>(ErrorCode::CHECKPOINT_SAVE_FAILED), 6000);

    EXPECT_GE(static_cast<int>(ErrorCode::LOSS_IS_NAN), 5000);
    EXPECT_LT(static_cast<int>(ErrorCode::LOSS_IS_NAN), 6000);
}

// ERR-009f: Distributed error values in correct range (6000-6999)
TEST(ErrorCodesTest, ErrorCode_DistributedRange) {
    EXPECT_GE(static_cast<int>(ErrorCode::DISTRIBUTED_INIT_FAILED), 6000);
    EXPECT_LT(static_cast<int>(ErrorCode::DISTRIBUTED_INIT_FAILED), 7000);

    EXPECT_GE(static_cast<int>(ErrorCode::DISTRIBUTED_COMM_FAILED), 6000);
    EXPECT_LT(static_cast<int>(ErrorCode::DISTRIBUTED_COMM_FAILED), 7000);

    EXPECT_GE(static_cast<int>(ErrorCode::RANK_MISMATCH), 6000);
    EXPECT_LT(static_cast<int>(ErrorCode::RANK_MISMATCH), 7000);
}

// ERR-009g: Task error values in correct range (7000-7999)
TEST(ErrorCodesTest, ErrorCode_TaskRange) {
    EXPECT_GE(static_cast<int>(ErrorCode::TASK_NOT_FOUND), 7000);
    EXPECT_LT(static_cast<int>(ErrorCode::TASK_NOT_FOUND), 8000);

    EXPECT_GE(static_cast<int>(ErrorCode::TASK_ALREADY_RUNNING), 7000);
    EXPECT_LT(static_cast<int>(ErrorCode::TASK_ALREADY_RUNNING), 8000);

    EXPECT_GE(static_cast<int>(ErrorCode::TASK_TIMEOUT), 7000);
    EXPECT_LT(static_cast<int>(ErrorCode::TASK_TIMEOUT), 8000);

    EXPECT_GE(static_cast<int>(ErrorCode::TASK_GPU_LIMIT_EXCEEDED), 7000);
    EXPECT_LT(static_cast<int>(ErrorCode::TASK_GPU_LIMIT_EXCEEDED), 8000);
}

// ERR-009h: General error values in correct range (9000-9999)
TEST(ErrorCodesTest, ErrorCode_GeneralRange) {
    EXPECT_GE(static_cast<int>(ErrorCode::UNKNOWN_ERROR), 9000);
    EXPECT_LT(static_cast<int>(ErrorCode::UNKNOWN_ERROR), 10000);

    EXPECT_GE(static_cast<int>(ErrorCode::NOT_IMPLEMENTED), 9000);
    EXPECT_LT(static_cast<int>(ErrorCode::NOT_IMPLEMENTED), 10000);

    EXPECT_GE(static_cast<int>(ErrorCode::INVALID_ARGUMENT), 9000);
    EXPECT_LT(static_cast<int>(ErrorCode::INVALID_ARGUMENT), 10000);

    EXPECT_GE(static_cast<int>(ErrorCode::FILE_IO_ERROR), 9000);
    EXPECT_LT(static_cast<int>(ErrorCode::FILE_IO_ERROR), 10000);

    EXPECT_GE(static_cast<int>(ErrorCode::MEMORY_ALLOCATION_FAILED), 9000);
    EXPECT_LT(static_cast<int>(ErrorCode::MEMORY_ALLOCATION_FAILED), 10000);
}

// ERR-010: errorCodeToString returns "UNKNOWN" for invalid error codes
TEST(ErrorCodesTest, ErrorCodeToString_InvalidCode) {
    ErrorCode invalidCode = static_cast<ErrorCode>(99999);
    EXPECT_STREQ("UNKNOWN", errorCodeToString(invalidCode));
}

// ERR-011: Alias error codes have same value as original
TEST(ErrorCodesTest, ErrorCode_Aliases) {
    // INVALID_MODEL_ARCHITECTURE is alias for MODEL_INVALID_ARCHITECTURE
    EXPECT_EQ(static_cast<int>(ErrorCode::INVALID_MODEL_ARCHITECTURE),
              static_cast<int>(ErrorCode::MODEL_INVALID_ARCHITECTURE));

    // INVALID_DATA_FORMAT is alias for DATA_INVALID_FORMAT
    EXPECT_EQ(static_cast<int>(ErrorCode::INVALID_DATA_FORMAT),
              static_cast<int>(ErrorCode::DATA_INVALID_FORMAT));

    // CUDA_ERROR is alias for GPU_CUDA_ERROR
    EXPECT_EQ(static_cast<int>(ErrorCode::CUDA_ERROR),
              static_cast<int>(ErrorCode::GPU_CUDA_ERROR));
}
