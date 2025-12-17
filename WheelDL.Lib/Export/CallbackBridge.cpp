#include "pch.h"
#include "CallbackBridge.h"
#include "../Utils/Error/ErrorHandler.h"
#include <cstring>

namespace WheelDL {
namespace Export {

ProgressCallback CallbackBridge::createBridge(WheelProgressCallback callback, void* userData) {
    if (!callback) {
        // Return empty callback if no C callback provided
        return ProgressCallback();
    }

    // Capture callback and userData in lambda
    return [callback, userData](const ProgressData& progress) {
        try {
            WheelProgressData wheelProgress;
            convertProgressData(progress, wheelProgress);
            callback(&wheelProgress, userData);
        }
        catch (...) {
            // Swallow exceptions - never let them escape to native caller
            // This prevents crashes when C# exceptions occur in callbacks
        }
    };
}

void CallbackBridge::convertProgressData(const ProgressData& src, WheelProgressData& dst) {
    std::memset(&dst, 0, sizeof(WheelProgressData));

    dst.stage = convertProgressStage(src.stage);
    dst.currentEpoch = src.currentEpoch;
    dst.totalEpochs = src.totalEpochs;
    dst.currentBatch = src.currentBatch;
    dst.totalBatches = src.totalBatches;
    dst.loss = src.loss;
    dst.learningRate = src.learningRate;
    dst.gpuMemoryUsage = src.gpuMemoryUsage;
    dst.elapsedTime = src.elapsedTime;
    dst.eta = src.eta;

    convertMetricsData(src.metrics, dst.metrics);

    // Safe string copy for message
    Utils::ErrorHandler::safeCopyToBuffer(src.message, dst.message, sizeof(dst.message));
}

void CallbackBridge::convertMetricsData(const MetricsData& src, WheelMetricsData& dst) {
    dst.loss = src.loss;
    dst.accuracy = src.accuracy;
    dst.precision = src.precision;
    dst.recall = src.recall;
    dst.f1Score = src.f1Score;
    dst.mAP = src.mAP;
    dst.fitness = src.fitness;
    dst.threshold = src.threshold;
    dst.aucROC = src.aucROC;
}

WheelProgressStage CallbackBridge::convertProgressStage(ProgressStage stage) {
    switch (stage) {
        case ProgressStage::TRAIN_BATCH:
            return WHEEL_STAGE_TRAIN_BATCH;
        case ProgressStage::TRAIN_EPOCH:
            return WHEEL_STAGE_TRAIN_EPOCH;
        case ProgressStage::VAL_BATCH:
            return WHEEL_STAGE_VAL_BATCH;
        case ProgressStage::VAL_EPOCH:
            return WHEEL_STAGE_VAL_EPOCH;
        case ProgressStage::CHECKPOINT_SAVED:
            return WHEEL_STAGE_CHECKPOINT_SAVED;
        default:
            return WHEEL_STAGE_TRAIN_BATCH;
    }
}

} // namespace Export
} // namespace WheelDL
