#pragma once

#include "ExportMacros.h"
#include "ExportTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==================== Library Lifecycle ====================

/**
 * @brief Initialize the WheelDL library
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Initialize(WheelError* err);

/**
 * @brief Shutdown the WheelDL library and release resources
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Shutdown(WheelError* err);

/**
 * @brief Get library version string
 * @param buffer Output buffer for version string
 * @param bufferSize Size of output buffer
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_GetVersion(char* buffer, int32_t bufferSize, WheelError* err);

// ==================== Configuration ====================

/**
 * @brief Create configuration from file paths
 * @param modelPath Path to model YAML file (e.g., "yolov8n.yaml")
 * @param hyperParamPath Path to hyperparameters YAML file (e.g., "default.yaml")
 * @param datasetPath Path to dataset JSON file (JSON, not YAML!)
 * @param outHandle Output handle for created configuration
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Config_Create(
    const char* modelPath,
    const char* hyperParamPath,
    const char* datasetPath,
    WheelHandle* outHandle,
    WheelError* err);

/**
 * @brief Destroy configuration and release resources
 * @param handle Configuration handle
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Config_Destroy(WheelHandle handle, WheelError* err);

// ==================== Task Submission ====================

/**
 * @brief Submit training task
 * @param configHandle Configuration handle
 * @param callback Progress callback function (may be null)
 * @param callbackUserData User data to pass to callback
 * @param outTaskId Output buffer for task ID
 * @param taskIdBufferSize Size of task ID buffer
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 * @note 30-second cooldown between submissions (automatic)
 * @note Waits for GPU memory threshold (default 4GB)
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Task_SubmitTraining(
    WheelHandle configHandle,
    WheelProgressCallback callback,
    void* callbackUserData,
    char* outTaskId,
    int32_t taskIdBufferSize,
    WheelError* err);

/**
 * @brief Submit validation task
 * @param configHandle Configuration handle
 * @param checkpointPath Path to checkpoint file
 * @param outTaskId Output buffer for task ID
 * @param taskIdBufferSize Size of task ID buffer
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Task_SubmitValidation(
    WheelHandle configHandle,
    const char* checkpointPath,
    char* outTaskId,
    int32_t taskIdBufferSize,
    WheelError* err);

/**
 * @brief Submit prediction task
 * @param configHandle Configuration handle
 * @param checkpointPath Path to checkpoint file
 * @param outTaskId Output buffer for task ID
 * @param taskIdBufferSize Size of task ID buffer
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Task_SubmitPrediction(
    WheelHandle configHandle,
    const char* checkpointPath,
    char* outTaskId,
    int32_t taskIdBufferSize,
    WheelError* err);

// ==================== Task Query ====================

/**
 * @brief Get task status
 * @param taskId Task ID string
 * @param out Output progress data
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Task_GetStatus(
    const char* taskId, WheelProgressData* out, WheelError* err);

/**
 * @brief Get task result
 * @param taskId Task ID string
 * @param out Output task result
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Task_GetResult(
    const char* taskId, WheelTaskResult* out, WheelError* err);

/**
 * @brief Check if task exists
 * @param taskId Task ID string
 * @param outExists Output: 1 if exists, 0 otherwise
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Task_Exists(
    const char* taskId, int32_t* outExists, WheelError* err);

/**
 * @brief Check if task is running
 * @param taskId Task ID string
 * @param outRunning Output: 1 if running, 0 otherwise
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Task_IsRunning(
    const char* taskId, int32_t* outRunning, WheelError* err);

// ==================== Task Control ====================

/**
 * @brief Stop a running task
 * @param taskId Task ID string
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Task_Stop(const char* taskId, WheelError* err);

/**
 * @brief Wait for task to complete
 * @param taskId Task ID string
 * @param timeoutMs Timeout in milliseconds (0 = infinite)
 * @param out Output task result
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success, WHEEL_RESULT_TIMEOUT on timeout, WHEEL_RESULT_STOPPED if stopped
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Task_Wait(
    const char* taskId,
    uint64_t timeoutMs,
    WheelTaskResult* out,
    WheelError* err);

/**
 * @brief Remove task from registry
 * @param taskId Task ID string
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Task_Remove(const char* taskId, WheelError* err);

/**
 * @brief Clear all completed tasks
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Task_ClearCompleted(WheelError* err);

// ==================== Resource Query ====================

/**
 * @brief Get available GPU memory
 * @param deviceIndex GPU device index
 * @param outMB Output: available memory in MB
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_GetAvailableGpuMemory(int32_t deviceIndex, float* outMB, WheelError* err);

/**
 * @brief Check if library can accept new task
 * @param deviceIndex GPU device index
 * @param outCanAccept Output: 1 if can accept, 0 otherwise
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_CanAcceptNewTask(int32_t deviceIndex, int32_t* outCanAccept, WheelError* err);

/**
 * @brief Get running task count
 * @param outCount Output: number of running tasks
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_GetRunningTaskCount(int32_t* outCount, WheelError* err);

/**
 * @brief Get pending task count
 * @param outCount Output: number of pending tasks
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_GetPendingTaskCount(int32_t* outCount, WheelError* err);

/**
 * @brief Set memory threshold for task acceptance
 * @param thresholdMB Threshold in MB (default: 4096)
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_SetMemoryThreshold(int32_t thresholdMB, WheelError* err);

/**
 * @brief Get current memory threshold
 * @param outThresholdMB Output: threshold in MB
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_GetMemoryThreshold(int32_t* outThresholdMB, WheelError* err);

// ==================== Memory Management ====================

/**
 * @brief Get the number of available CUDA GPU devices
 * @param outCount Output: number of GPU devices (0 if CUDA not available)
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_GetGPUDeviceCount(int32_t* outCount, WheelError* err);

/**
 * @brief Check if CUDA is available
 * @param outAvailable Output: 1 if CUDA available, 0 otherwise
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_IsCudaAvailable(int32_t* outAvailable, WheelError* err);

/**
 * @brief Get GPU memory currently used (allocated by PyTorch) in MB
 * @param deviceIndex GPU device index
 * @param outMB Output: used memory in megabytes
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_GetGPUUsedMB(int32_t deviceIndex, float* outMB, WheelError* err);

/**
 * @brief Get total GPU memory in MB
 * @param deviceIndex GPU device index
 * @param outMB Output: total memory in megabytes
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_GetGPUTotalMB(int32_t deviceIndex, float* outMB, WheelError* err);

/**
 * @brief Get GPU memory usage percentage
 * @param deviceIndex GPU device index
 * @param outPercent Output: usage percentage (0.0 - 100.0)
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_GetGPUUsagePercent(int32_t deviceIndex, float* outPercent, WheelError* err);

/**
 * @brief Check if GPU has enough memory available
 * @param deviceIndex GPU device index
 * @param requiredBytes Required memory in bytes
 * @param outHasEnough Output: 1 if enough memory, 0 otherwise
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_HasEnoughGPU(
    int32_t deviceIndex,
    uint64_t requiredBytes,
    int32_t* outHasEnough,
    WheelError* err);

/**
 * @brief Get CPU memory currently used in bytes
 * @param outBytes Output: used memory in bytes
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_GetCPUUsedBytes(uint64_t* outBytes, WheelError* err);

/**
 * @brief Get available CPU memory in bytes
 * @param outBytes Output: available memory in bytes
 * @param err Error output structure
 * @return WheelResult WHEEL_RESULT_OK on success
 */
WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_GetCPUAvailableBytes(uint64_t* outBytes, WheelError* err);

#ifdef __cplusplus
}
#endif
