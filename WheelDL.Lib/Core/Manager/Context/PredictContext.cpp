#include "pch.h"
#include "PredictContext.h"
#include "../../Predictor/ClassificationPredictor.h"
#include "../../Predictor/DetectionPredictor.h"
#include "../../Predictor/OBBPredictor.h"
#include "../../Predictor/SegmentationPredictor.h"
#include "../../Predictor/AnomalyPredictor.h"
#include "../../../Utils/Error/WheelLibException.h"

namespace WheelDL {
namespace Core {
namespace Manager {

PredictContext::PredictContext(const PredictRequest& request)
    : BaseContext(request.config, OperationType::PREDICT)
    , _checkpointPath(request.checkpointPath)
{
    _logger->info("PredictContext", "Created for task: " + _taskId);
}

TaskResult PredictContext::run() {
    TaskResult result;
    result.taskId = _taskId;
    result.operationType = OperationType::PREDICT;
    result.startTime = std::chrono::system_clock::now();

    _state = TrainingState::INITIALIZING;
    _logger->info("PredictContext", "Starting prediction");

    try {
        _state = TrainingState::TRAINING;  // Reuse for "processing"

        auto predictor = createPredictor();

        if (_stopRequested.load(std::memory_order_acquire)) {
            _state = TrainingState::STOPPED;
            result.finalState = TrainingState::STOPPED;
            _logger->info("PredictContext", "Cancelled before start");
            result.endTime = std::chrono::system_clock::now();
            result.totalTimeMs = std::chrono::duration<double, std::milli>(
                result.endTime - result.startTime).count();
            result.profiling.totalTimeMs = result.totalTimeMs;
            return result;
        }

        predictor->predictAndExport(_workspace->getResultDir());

        // Check if stopped during prediction (Predictor returns early without exception)
        if (_stopRequested.load(std::memory_order_acquire)) {
            _state = TrainingState::STOPPED;
            result.finalState = TrainingState::STOPPED;
            _logger->info("PredictContext", "Prediction stopped by user request");
        } else {
            _profiler->exportReport(_workspace->getProfilerDir());
            _config->exportToYAML(_workspace->getResultDir() + "/config.yaml");

            _state = TrainingState::COMPLETED;
            result.finalState = TrainingState::COMPLETED;
            result.outputPath = _workspace->getRoot();

            _logger->info("PredictContext", "Prediction completed");
        }

    } catch (const std::exception& e) {
        _state = TrainingState::FAILED;
        result.finalState = TrainingState::FAILED;
        result.errorMessage = e.what();
        result.errorCode = static_cast<int>(WheelDL::Utils::ErrorCode::PREDICTION_FAILED);

        _logger->error("PredictContext", "Prediction failed: " + std::string(e.what()));
    }

    result.endTime = std::chrono::system_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(
        result.endTime - result.startTime).count();

    // Populate profiling data
    result.profiling.totalTimeMs = result.totalTimeMs;

    return result;
}

std::unique_ptr<Predictor::BasePredictor> PredictContext::createPredictor() {
    switch (_config->getTaskType()) {
        case TaskType::CLASSIFICATION:
            return std::make_unique<Predictor::ClassificationPredictor>(
                _config, _checkpointPath, _logger.get(), _profiler.get(), &_stopRequested);

        case TaskType::DETECTION:
            return std::make_unique<Predictor::DetectionPredictor>(
                _config, _checkpointPath, _logger.get(), _profiler.get(), &_stopRequested);

        case TaskType::OBB:
            return std::make_unique<Predictor::OBBPredictor>(
                _config, _checkpointPath, _logger.get(), _profiler.get(), &_stopRequested);

        case TaskType::SEGMENTATION:
            return std::make_unique<Predictor::SegmentationPredictor>(
                _config, _checkpointPath, _logger.get(), _profiler.get(), &_stopRequested);

        case TaskType::ANOMALY:
            return std::make_unique<Predictor::AnomalyPredictor>(
                _config, _checkpointPath, _logger.get(), _profiler.get(), &_stopRequested);

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
