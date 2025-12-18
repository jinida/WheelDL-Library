#include "pch.h"
#include "TrainingContext.h"
#include "../../Trainer/ClassificationTrainer.h"
#include "../../Trainer/DetectionTrainer.h"
#include "../../Trainer/OBBTrainer.h"
#include "../../Trainer/SegmentationTrainer.h"
#include "../../Trainer/AnomalyTrainer.h"
#include "../../../Utils/Error/WheelLibException.h"

namespace WheelDL {
namespace Core {
namespace Manager {

TrainingContext::TrainingContext(const TrainRequest& request)
    : BaseContext(request.config, OperationType::TRAIN)
    , _progressCallback(request.progressCallback)
{
    _progress.totalEpochs = request.config->getEpochs();
    _logger->info("TrainingContext", "Created for task: " + _taskId);
}

TaskResult TrainingContext::run() {
    TaskResult result;
    result.taskId = _taskId;
    result.operationType = OperationType::TRAIN;
    result.startTime = std::chrono::system_clock::now();

    _state = TrainingState::INITIALIZING;
    _logger->info("TrainingContext", "Starting training");

    try {
        _state = TrainingState::TRAINING;

        auto trainer = createTrainer();

        if (_stopRequested.load(std::memory_order_acquire)) {
            _state = TrainingState::STOPPED;
            result.finalState = TrainingState::STOPPED;
            _logger->info("TrainingContext", "Cancelled before start");
            result.endTime = std::chrono::system_clock::now();
            result.totalTimeMs = std::chrono::duration<double, std::milli>(
                result.endTime - result.startTime).count();
            result.profiling.totalTimeMs = result.totalTimeMs;
            return result;
        }

        trainer->train();

        _profiler->exportReport(_workspace->getProfilerDir());
        _config->exportToYAML(_workspace->getResultDir() + "/config.yaml");

        _state = TrainingState::COMPLETED;
        result.finalState = TrainingState::COMPLETED;
        result.outputPath = _workspace->getRoot();
        result.checkpointPath = _workspace->getWeightsDir() + "/best.pt";

        _logger->info("TrainingContext", "Training completed");

    } catch (const WheelDL::Utils::StopRequestedException&) {
        // Stop requested by user - not an error
        _state = TrainingState::STOPPED;
        result.finalState = TrainingState::STOPPED;
        _logger->info("TrainingContext", "Training stopped by user request");

    } catch (const std::exception& e) {
        _state = TrainingState::FAILED;
        result.finalState = TrainingState::FAILED;
        result.errorMessage = e.what();
        result.errorCode = static_cast<int>(WheelDL::Utils::ErrorCode::TRAINING_FAILED);

        _logger->error("TrainingContext", "Training failed: " + std::string(e.what()));
    }

    result.endTime = std::chrono::system_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(
        result.endTime - result.startTime).count();

    // Populate profiling data
    result.profiling.totalTimeMs = result.totalTimeMs;

    return result;
}

std::unique_ptr<Trainer::BaseTrainer> TrainingContext::createTrainer() {
    ProgressCallback wrappedCallback = [this](const ProgressData& progress) {
        _progress = progress;
        if (_progressCallback) {
            _progressCallback(progress);
        }
    };

    switch (_config->getTaskType()) {
        case TaskType::CLASSIFICATION:
            return std::make_unique<Trainer::ClassificationTrainer>(
                _config, _logger.get(), _workspace.get(), _profiler.get(), wrappedCallback, &_stopRequested);

        case TaskType::DETECTION:
            return std::make_unique<Trainer::DetectionTrainer>(
                _config, _logger.get(), _workspace.get(), _profiler.get(), wrappedCallback, &_stopRequested);

        case TaskType::OBB:
            return std::make_unique<Trainer::OBBTrainer>(
                _config, _logger.get(), _workspace.get(), _profiler.get(), wrappedCallback, &_stopRequested);

        case TaskType::SEGMENTATION:
            return std::make_unique<Trainer::SegmentationTrainer>(
                _config, _logger.get(), _workspace.get(), _profiler.get(), wrappedCallback, &_stopRequested);

        case TaskType::ANOMALY:
            return std::make_unique<Trainer::AnomalyTrainer>(
                _config, _logger.get(), _workspace.get(), _profiler.get(), wrappedCallback, &_stopRequested);

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
