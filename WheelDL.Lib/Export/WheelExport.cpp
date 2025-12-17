#include "pch.h"
#include "WheelExport.h"
#include "HandleRegistry.h"
#include "CallbackBridge.h"
#include "../Config/Configuration.h"
#include "../Core/Manager/TaskManager.h"
#include "../Core/Manager/Request.h"
#include "../Core/Manager/Result.h"
#include "../Utils/Memory/MemoryManager.h"
#include <torch/torch.h>
#include <atomic>
#include <cstring>

// Version string
#define WHEELDL_VERSION "1.0.0"

// Namespace aliases for cleaner code
namespace Manager = WheelDL::Core::Manager;

namespace {
    // Initialization state
    std::atomic<bool> g_initialized{false};

    // Type aliases for registries
    using ConfigRegistry = WheelDL::Export::HandleRegistry<WheelDL::Config::Configuration>;

    // Convert C++ TaskType to WheelTaskType
    WheelTaskType convertTaskType(WheelDL::TaskType type) {
        switch (type) {
            case WheelDL::TaskType::CLASSIFICATION: return WHEEL_TASK_CLASSIFICATION;
            case WheelDL::TaskType::DETECTION: return WHEEL_TASK_DETECTION;
            case WheelDL::TaskType::SEGMENTATION: return WHEEL_TASK_SEGMENTATION;
            case WheelDL::TaskType::ANOMALY: return WHEEL_TASK_ANOMALY;
            case WheelDL::TaskType::OBB: return WHEEL_TASK_OBB;
            default: return WHEEL_TASK_UNKNOWN;
        }
    }

    // Convert C++ TrainingState to WheelTrainingState
    WheelTrainingState convertTrainingState(WheelDL::TrainingState state) {
        switch (state) {
            case WheelDL::TrainingState::IDLE: return WHEEL_STATE_IDLE;
            case WheelDL::TrainingState::INITIALIZING: return WHEEL_STATE_INITIALIZING;
            case WheelDL::TrainingState::TRAINING: return WHEEL_STATE_TRAINING;
            case WheelDL::TrainingState::VALIDATING: return WHEEL_STATE_VALIDATING;
            case WheelDL::TrainingState::PAUSED: return WHEEL_STATE_PAUSED;
            case WheelDL::TrainingState::COMPLETED: return WHEEL_STATE_COMPLETED;
            case WheelDL::TrainingState::FAILED: return WHEEL_STATE_FAILED;
            case WheelDL::TrainingState::STOPPED: return WHEEL_STATE_STOPPED;
            default: return WHEEL_STATE_IDLE;
        }
    }

    // Convert C++ OperationType to WheelOperationType
    WheelOperationType convertOperationType(Manager::OperationType op) {
        switch (op) {
            case Manager::OperationType::TRAIN: return WHEEL_OP_TRAIN;
            case Manager::OperationType::VALIDATE: return WHEEL_OP_VALIDATE;
            case Manager::OperationType::PREDICT: return WHEEL_OP_PREDICT;
            default: return WHEEL_OP_TRAIN;
        }
    }

    // Convert TaskResult to WheelTaskResult
    void convertTaskResult(const Manager::TaskResult& src, WheelTaskResult& dst) {
        std::memset(&dst, 0, sizeof(WheelTaskResult));

        WheelDL::Utils::ErrorHandler::safeCopyToBuffer(src.taskId, dst.taskId, sizeof(dst.taskId));
        dst.finalState = convertTrainingState(src.finalState);
        dst.operationType = convertOperationType(src.operationType);

        if (src.metrics.has_value()) {
            dst.hasMetrics = 1;
            WheelDL::Export::CallbackBridge::convertMetricsData(src.metrics.value(), dst.metrics);
        } else {
            dst.hasMetrics = 0;
        }

        WheelDL::Utils::ErrorHandler::safeCopyToBuffer(src.outputPath, dst.outputPath, sizeof(dst.outputPath));
        WheelDL::Utils::ErrorHandler::safeCopyToBuffer(src.checkpointPath, dst.checkpointPath, sizeof(dst.checkpointPath));
        dst.totalTimeMs = src.totalTimeMs;

        // Profiling data
        dst.profiling.totalTimeMs = src.profiling.totalTimeMs;
        dst.profiling.gpuTimeMs = src.profiling.gpuTimeMs;
        dst.profiling.dataLoadTimeMs = src.profiling.dataLoadTimeMs;
        dst.profiling.preprocessTimeMs = src.profiling.preprocessTimeMs;
        dst.profiling.inferenceTimeMs = src.profiling.inferenceTimeMs;
        dst.profiling.postprocessTimeMs = src.profiling.postprocessTimeMs;
        dst.profiling.peakGpuMemoryMB = src.profiling.peakGpuMemoryMB;
        dst.profiling.peakCpuMemoryMB = src.profiling.peakCpuMemoryMB;

        WheelDL::Utils::ErrorHandler::safeCopyToBuffer(src.errorMessage, dst.errorMessage, sizeof(dst.errorMessage));
        dst.errorCode = src.errorCode;
    }
}

// ==================== Library Lifecycle ====================

WHEEL_API WheelResult WHEEL_CALL Wheel_Initialize(WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (g_initialized.exchange(true)) {
        // Already initialized - not an error
        return WHEEL_RESULT_OK;
    }

    // Any initialization logic here
    // TaskManager is a singleton and initializes on first use

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Shutdown(WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!g_initialized.exchange(false)) {
        // Already shutdown - not an error
        return WHEEL_RESULT_OK;
    }

    // Clear handle registries
    ConfigRegistry::getInstance().clear();

    // Clear completed tasks
    Manager::TaskManager::getInstance().clearCompletedTasks();

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_GetVersion(char* buffer, int32_t bufferSize, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!buffer || bufferSize <= 0) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid buffer");
        return WHEEL_RESULT_ERROR;
    }

    WheelDL::Utils::ErrorHandler::safeCopyToBuffer(WHEELDL_VERSION, buffer, static_cast<size_t>(bufferSize));

    WHEEL_SAFE_END(err)
}

// ==================== Configuration ====================

WHEEL_API WheelResult WHEEL_CALL Wheel_Config_Create(
    const char* modelPath,
    const char* hyperParamPath,
    const char* datasetPath,
    WheelHandle* outHandle,
    WheelError* err) {

    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!modelPath || !hyperParamPath || !datasetPath || !outHandle) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto config = std::make_shared<WheelDL::Config::Configuration>();
    config->load(modelPath, hyperParamPath, datasetPath);

    *outHandle = ConfigRegistry::getInstance().registerObject(config);

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Config_Destroy(WheelHandle handle, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    ConfigRegistry::getInstance().unregisterObject(handle);

    WHEEL_SAFE_END(err)
}

// ==================== Task Submission ====================

WHEEL_API WheelResult WHEEL_CALL Wheel_Task_SubmitTraining(
    WheelHandle configHandle,
    WheelProgressCallback callback,
    void* callbackUserData,
    char* outTaskId,
    int32_t taskIdBufferSize,
    WheelError* err) {

    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outTaskId || taskIdBufferSize <= 0) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid task ID buffer");
        return WHEEL_RESULT_ERROR;
    }

    auto config = ConfigRegistry::getInstance().getObject(configHandle);

    Manager::TrainRequest request(config);
    request.progressCallback = WheelDL::Export::CallbackBridge::createBridge(callback, callbackUserData);

    auto& taskManager = Manager::TaskManager::getInstance();
    std::string taskId = taskManager.submitTask(request);

    WheelDL::Utils::ErrorHandler::safeCopyToBuffer(taskId, outTaskId, static_cast<size_t>(taskIdBufferSize));

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Task_SubmitValidation(
    WheelHandle configHandle,
    const char* checkpointPath,
    char* outTaskId,
    int32_t taskIdBufferSize,
    WheelError* err) {

    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!checkpointPath || !outTaskId || taskIdBufferSize <= 0) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto config = ConfigRegistry::getInstance().getObject(configHandle);

    Manager::ValidateRequest request(config);
    request.checkpointPath = checkpointPath;

    auto& taskManager = Manager::TaskManager::getInstance();
    std::string taskId = taskManager.submitTask(request);

    WheelDL::Utils::ErrorHandler::safeCopyToBuffer(taskId, outTaskId, static_cast<size_t>(taskIdBufferSize));

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Task_SubmitPrediction(
    WheelHandle configHandle,
    const char* checkpointPath,
    char* outTaskId,
    int32_t taskIdBufferSize,
    WheelError* err) {

    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!checkpointPath || !outTaskId || taskIdBufferSize <= 0) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto config = ConfigRegistry::getInstance().getObject(configHandle);

    Manager::PredictRequest request(config);
    request.checkpointPath = checkpointPath;

    auto& taskManager = Manager::TaskManager::getInstance();
    std::string taskId = taskManager.submitTask(request);

    WheelDL::Utils::ErrorHandler::safeCopyToBuffer(taskId, outTaskId, static_cast<size_t>(taskIdBufferSize));

    WHEEL_SAFE_END(err)
}

// ==================== Task Query ====================

WHEEL_API WheelResult WHEEL_CALL Wheel_Task_GetStatus(
    const char* taskId, WheelProgressData* out, WheelError* err) {

    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!taskId || !out) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& taskManager = Manager::TaskManager::getInstance();
    WheelDL::ProgressData progress = taskManager.getTaskStatus(taskId);

    WheelDL::Export::CallbackBridge::convertProgressData(progress, *out);

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Task_GetResult(
    const char* taskId, WheelTaskResult* out, WheelError* err) {

    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!taskId || !out) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& taskManager = Manager::TaskManager::getInstance();
    Manager::TaskResult result = taskManager.getTaskResult(taskId);

    convertTaskResult(result, *out);

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Task_Exists(
    const char* taskId, int32_t* outExists, WheelError* err) {

    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!taskId || !outExists) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& taskManager = Manager::TaskManager::getInstance();
    *outExists = taskManager.taskExists(taskId) ? 1 : 0;

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Task_IsRunning(
    const char* taskId, int32_t* outRunning, WheelError* err) {

    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!taskId || !outRunning) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& taskManager = Manager::TaskManager::getInstance();
    *outRunning = taskManager.isTaskRunning(taskId) ? 1 : 0;

    WHEEL_SAFE_END(err)
}

// ==================== Task Control ====================

WHEEL_API WheelResult WHEEL_CALL Wheel_Task_Stop(const char* taskId, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!taskId) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null task ID");
        return WHEEL_RESULT_ERROR;
    }

    auto& taskManager = Manager::TaskManager::getInstance();
    taskManager.stopTask(taskId);

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Task_Wait(
    const char* taskId,
    uint64_t timeoutMs,
    WheelTaskResult* out,
    WheelError* err) {

    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!taskId || !out) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& taskManager = Manager::TaskManager::getInstance();

    try {
        Manager::TaskResult result = taskManager.waitForTask(taskId, timeoutMs);
        convertTaskResult(result, *out);

        if (result.isStopped()) {
            return WHEEL_RESULT_STOPPED;
        }
    }
    catch (const WheelDL::Utils::TaskException& e) {
        if (e.getErrorCode() == WheelDL::Utils::ErrorCode::TASK_TIMEOUT) {
            WheelDL::Export::fillError(err, e.getErrorCode(), "TASK", e.getTaskId(), e.getMessage());
            return WHEEL_RESULT_TIMEOUT;
        }
        throw; // Re-throw other TaskExceptions
    }

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Task_Remove(const char* taskId, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!taskId) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null task ID");
        return WHEEL_RESULT_ERROR;
    }

    auto& taskManager = Manager::TaskManager::getInstance();
    taskManager.removeTask(taskId);

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Task_ClearCompleted(WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    auto& taskManager = Manager::TaskManager::getInstance();
    taskManager.clearCompletedTasks();

    WHEEL_SAFE_END(err)
}

// ==================== Resource Query ====================

WHEEL_API WheelResult WHEEL_CALL Wheel_GetAvailableGpuMemory(int32_t deviceIndex, float* outMB, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outMB) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null output parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& taskManager = Manager::TaskManager::getInstance();
    *outMB = taskManager.getAvailableGpuMemory(deviceIndex);

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_CanAcceptNewTask(int32_t deviceIndex, int32_t* outCanAccept, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outCanAccept) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null output parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& taskManager = Manager::TaskManager::getInstance();
    *outCanAccept = taskManager.canAcceptNewTask(deviceIndex) ? 1 : 0;

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_GetRunningTaskCount(int32_t* outCount, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outCount) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null output parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& taskManager = Manager::TaskManager::getInstance();
    *outCount = static_cast<int32_t>(taskManager.getRunningTaskCount());

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_GetPendingTaskCount(int32_t* outCount, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outCount) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null output parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& taskManager = Manager::TaskManager::getInstance();
    *outCount = static_cast<int32_t>(taskManager.getPendingTaskCount());

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_SetMemoryThreshold(int32_t thresholdMB, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (thresholdMB < 0) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Threshold must be non-negative");
        return WHEEL_RESULT_ERROR;
    }

    auto& taskManager = Manager::TaskManager::getInstance();
    taskManager.setMemoryThreshold(static_cast<size_t>(thresholdMB));

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_GetMemoryThreshold(int32_t* outThresholdMB, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outThresholdMB) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null output parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& taskManager = Manager::TaskManager::getInstance();
    *outThresholdMB = static_cast<int32_t>(taskManager.getMemoryThreshold());

    WHEEL_SAFE_END(err)
}

// ==================== Memory Management ====================

WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_GetGPUDeviceCount(int32_t* outCount, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outCount) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null output parameter");
        return WHEEL_RESULT_ERROR;
    }

#ifdef USE_CUDA
    if (torch::cuda::is_available()) {
        *outCount = static_cast<int32_t>(torch::cuda::device_count());
    } else {
        *outCount = 0;
    }
#else
    *outCount = 0;
#endif

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_IsCudaAvailable(int32_t* outAvailable, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outAvailable) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null output parameter");
        return WHEEL_RESULT_ERROR;
    }

#ifdef USE_CUDA
    *outAvailable = torch::cuda::is_available() ? 1 : 0;
#else
    *outAvailable = 0;
#endif

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_GetGPUUsedMB(int32_t deviceIndex, float* outMB, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outMB) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null output parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& memManager = WheelDL::Utils::MemoryManager::getInstance();
    *outMB = memManager.getGPUMemoryUsed(deviceIndex);

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_GetGPUTotalMB(int32_t deviceIndex, float* outMB, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outMB) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null output parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& memManager = WheelDL::Utils::MemoryManager::getInstance();
    *outMB = memManager.getGPUMemoryTotal(deviceIndex);

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_GetGPUUsagePercent(int32_t deviceIndex, float* outPercent, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outPercent) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null output parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& memManager = WheelDL::Utils::MemoryManager::getInstance();
    *outPercent = memManager.getGPUMemoryUsagePercent(deviceIndex);

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_HasEnoughGPU(
    int32_t deviceIndex,
    uint64_t requiredBytes,
    int32_t* outHasEnough,
    WheelError* err) {

    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outHasEnough) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null output parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& memManager = WheelDL::Utils::MemoryManager::getInstance();
    *outHasEnough = memManager.hasEnoughGPUMemory(static_cast<size_t>(requiredBytes), deviceIndex) ? 1 : 0;

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_GetCPUUsedBytes(uint64_t* outBytes, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outBytes) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null output parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& memManager = WheelDL::Utils::MemoryManager::getInstance();
    *outBytes = static_cast<uint64_t>(memManager.getCPUMemoryUsed());

    WHEEL_SAFE_END(err)
}

WHEEL_API WheelResult WHEEL_CALL Wheel_Memory_GetCPUAvailableBytes(uint64_t* outBytes, WheelError* err) {
    WheelDL::Export::clearError(err);
    WHEEL_SAFE_BEGIN()

    if (!outBytes) {
        WheelDL::Export::fillError(err, WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
            "GENERAL", "", "Invalid null output parameter");
        return WHEEL_RESULT_ERROR;
    }

    auto& memManager = WheelDL::Utils::MemoryManager::getInstance();
    *outBytes = static_cast<uint64_t>(memManager.getCPUMemoryAvailable());

    WHEEL_SAFE_END(err)
}
