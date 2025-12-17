#pragma once

#include <cstdint>

// ========== Handle Type ==========
typedef void* WheelHandle;

// ========== Result Enum ==========
typedef enum WheelResult {
    WHEEL_RESULT_OK = 0,
    WHEEL_RESULT_ERROR = 1,
    WHEEL_RESULT_TIMEOUT = 2,
    WHEEL_RESULT_STOPPED = 3
} WheelResult;

// ========== Enums (match C++ enums exactly) ==========
typedef enum WheelTaskType {
    WHEEL_TASK_CLASSIFICATION = 0,
    WHEEL_TASK_DETECTION = 1,
    WHEEL_TASK_SEGMENTATION = 2,
    WHEEL_TASK_ANOMALY = 3,
    WHEEL_TASK_OBB = 4,
    WHEEL_TASK_UNKNOWN = 99
} WheelTaskType;

typedef enum WheelOperationType {
    WHEEL_OP_TRAIN = 0,
    WHEEL_OP_VALIDATE = 1,
    WHEEL_OP_PREDICT = 2
} WheelOperationType;

typedef enum WheelTrainingState {
    WHEEL_STATE_IDLE = 0,
    WHEEL_STATE_INITIALIZING = 1,
    WHEEL_STATE_TRAINING = 2,
    WHEEL_STATE_VALIDATING = 3,
    WHEEL_STATE_PAUSED = 4,
    WHEEL_STATE_COMPLETED = 5,
    WHEEL_STATE_FAILED = 6,
    WHEEL_STATE_STOPPED = 7
} WheelTrainingState;

typedef enum WheelProgressStage {
    WHEEL_STAGE_TRAIN_BATCH = 0,
    WHEEL_STAGE_TRAIN_EPOCH = 1,
    WHEEL_STAGE_VAL_BATCH = 2,
    WHEEL_STAGE_VAL_EPOCH = 3,
    WHEEL_STAGE_CHECKPOINT_SAVED = 4
} WheelProgressStage;

// ========== Error Structure ==========
typedef struct WheelError {
    int32_t code;           // ErrorCode value
    char message[512];      // Error message (from ErrorHandler::formatErrorMessage)
    char category[32];      // Exception type: "CONFIG", "MODEL", "DATA", "GPU", "TRAINING", "TASK", "UNKNOWN"
    char taskId[64];        // Only set for TaskException
} WheelError;

// ========== Data Structures ==========
typedef struct WheelMetricsData {
    float loss;
    float accuracy;
    float precision;
    float recall;
    float f1Score;
    float mAP;
    float fitness;
    float threshold;
    float aucROC;
} WheelMetricsData;

typedef struct WheelProgressData {
    WheelProgressStage stage;
    int32_t currentEpoch;
    int32_t totalEpochs;
    int32_t currentBatch;
    int32_t totalBatches;
    float loss;
    float learningRate;
    float gpuMemoryUsage;
    double elapsedTime;
    double eta;
    WheelMetricsData metrics;
    char message[256];
} WheelProgressData;

typedef struct WheelProfilingData {
    double totalTimeMs;
    double gpuTimeMs;
    double dataLoadTimeMs;
    double preprocessTimeMs;
    double inferenceTimeMs;
    double postprocessTimeMs;
    uint64_t peakGpuMemoryMB;
    uint64_t peakCpuMemoryMB;
} WheelProfilingData;

typedef struct WheelTaskResult {
    char taskId[64];
    WheelTrainingState finalState;
    WheelOperationType operationType;
    WheelMetricsData metrics;
    int32_t hasMetrics;         // 0 or 1
    char outputPath[512];
    char checkpointPath[512];
    double totalTimeMs;
    WheelProfilingData profiling;
    char errorMessage[512];
    int32_t errorCode;
} WheelTaskResult;

// ========== Callback Type ==========
#ifdef _WIN32
#define WHEEL_CALLBACK __stdcall
#else
#define WHEEL_CALLBACK
#endif

typedef void (WHEEL_CALLBACK *WheelProgressCallback)(
    const WheelProgressData* progress,
    void* userData
);
