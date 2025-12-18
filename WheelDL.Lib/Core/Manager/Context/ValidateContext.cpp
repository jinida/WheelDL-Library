#include "pch.h"
#include "ValidateContext.h"
#include "../../Validator/ClassificationValidator.h"
#include "../../Validator/DetectionValidator.h"
#include "../../Validator/OBBValidator.h"
#include "../../Validator/SegmentationValidator.h"
#include "../../Validator/AnomalyValidator.h"
#include "../../../Utils/Error/WheelLibException.h"
#include "../../../Utils/Export/MetricsExporter.h"

namespace WheelDL {
namespace Core {
namespace Manager {

ValidateContext::ValidateContext(const ValidateRequest& request)
    : BaseContext(request.config, OperationType::VALIDATE)
    , _checkpointPath(request.checkpointPath)
{
    _logger->info("ValidateContext", "Created for task: " + _taskId);
}

TaskResult ValidateContext::run() {
    TaskResult result;
    result.taskId = _taskId;
    result.operationType = OperationType::VALIDATE;
    result.startTime = std::chrono::system_clock::now();

    _state = TrainingState::INITIALIZING;
    _logger->info("ValidateContext", "Starting validation");

    try {
        _state = TrainingState::VALIDATING;

        auto validator = createValidator();
        MetricsData metrics = validator->validate(_checkpointPath);
        result.metrics = metrics;
        std::filesystem::path logsDir = _workspace->getLogsDir();
        std::string jsonPath = (logsDir / "best_metrics.json").string();
        WheelDL::Utils::Export::MetricsExporter::exportToJSON(
            { metrics },
            jsonPath,
            0.0f
        );

        _profiler->exportReport(_workspace->getProfilerDir());
        _config->exportToYAML(_workspace->getResultDir() + "/config.yaml");

        _state = TrainingState::COMPLETED;
        result.finalState = TrainingState::COMPLETED;
        result.outputPath = _workspace->getRoot();

        _logger->info("ValidateContext", "Validation completed");

    } catch (const WheelDL::Utils::StopRequestedException&) {
        // Stop requested by user - not an error
        _state = TrainingState::STOPPED;
        result.finalState = TrainingState::STOPPED;
        _logger->info("ValidateContext", "Validation stopped by user request");

    } catch (const std::exception& e) {
        _state = TrainingState::FAILED;
        result.finalState = TrainingState::FAILED;
        result.errorMessage = e.what();
        result.errorCode = static_cast<int>(WheelDL::Utils::ErrorCode::VALIDATION_FAILED);

        _logger->error("ValidateContext", "Validation failed: " + std::string(e.what()));
    }

    result.endTime = std::chrono::system_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(
        result.endTime - result.startTime).count();

    // Populate profiling data
    result.profiling.totalTimeMs = result.totalTimeMs;

    return result;
}

std::unique_ptr<Validator::BaseValidator> ValidateContext::createValidator() {
    switch (_config->getTaskType()) {
        case TaskType::CLASSIFICATION:
            return std::make_unique<Validator::ClassificationValidator>(
                _config, _logger.get(), _profiler.get(), &_stopRequested);

        case TaskType::DETECTION:
            return std::make_unique<Validator::DetectionValidator>(
                _config, _logger.get(), _profiler.get(), &_stopRequested);

        case TaskType::OBB:
            return std::make_unique<Validator::OBBValidator>(
                _config, _logger.get(), _profiler.get(), &_stopRequested);

        case TaskType::SEGMENTATION:
            return std::make_unique<Validator::SegmentationValidator>(
                _config, _logger.get(), _profiler.get(), &_stopRequested);

        case TaskType::ANOMALY:
            return std::make_unique<Validator::AnomalyValidator>(
                _config, _logger.get(), _profiler.get(), &_stopRequested);

        default:
            throw WheelDL::Utils::TaskException(
                WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                "Unsupported task type",
                _taskId);
    }
}

} // namespace Manager
} // namespace Core
} // namespace WheelDL
