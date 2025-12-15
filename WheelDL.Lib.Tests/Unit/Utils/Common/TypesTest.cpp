#include "pch.h"
#include "Utils/Common/Types.h"
#include <gtest/gtest.h>

using namespace WheelDL;

// =============================================================================
// TaskType Enum Tests (TYP-001 ~ TYP-004)
// =============================================================================

// TYP-001: taskTypeToString returns correct string for all enum values
TEST(TypesTest, TaskTypeToString_AllValues) {
    EXPECT_STREQ("Classification", taskTypeToString(TaskType::CLASSIFICATION));
    EXPECT_STREQ("Detection", taskTypeToString(TaskType::DETECTION));
    EXPECT_STREQ("Segmentation", taskTypeToString(TaskType::SEGMENTATION));
    EXPECT_STREQ("Anomaly", taskTypeToString(TaskType::ANOMALY));
    EXPECT_STREQ("OBB", taskTypeToString(TaskType::OBB));
    EXPECT_STREQ("Unknown", taskTypeToString(TaskType::UNKNOWN));
}

// TYP-002: taskTypeToString returns "Unknown" for invalid enum value
TEST(TypesTest, TaskTypeToString_InvalidValue) {
    TaskType invalidType = static_cast<TaskType>(999);
    EXPECT_STREQ("Unknown", taskTypeToString(invalidType));
}

// TYP-003: TaskType enum values are correctly assigned
TEST(TypesTest, TaskType_EnumValues) {
    EXPECT_EQ(0, static_cast<int>(TaskType::CLASSIFICATION));
    EXPECT_EQ(1, static_cast<int>(TaskType::DETECTION));
    EXPECT_EQ(2, static_cast<int>(TaskType::SEGMENTATION));
    EXPECT_EQ(3, static_cast<int>(TaskType::ANOMALY));
    EXPECT_EQ(4, static_cast<int>(TaskType::OBB));
    EXPECT_EQ(99, static_cast<int>(TaskType::UNKNOWN));
}

// TYP-004: TaskType enum has expected number of values
TEST(TypesTest, TaskType_EnumCount) {
    // All valid task types should be convertible to string
    int validCount = 0;
    for (int i = 0; i <= 99; ++i) {
        TaskType type = static_cast<TaskType>(i);
        const char* str = taskTypeToString(type);
        if (std::string(str) != "Unknown") {
            ++validCount;
        }
    }
    EXPECT_EQ(5, validCount); // CLASSIFICATION, DETECTION, SEGMENTATION, ANOMALY, OBB
}

// =============================================================================
// TrainingState Enum Tests (TYP-005 ~ TYP-008)
// =============================================================================

// TYP-005: trainingStateToString returns correct string for all enum values
TEST(TypesTest, TrainingStateToString_AllValues) {
    EXPECT_STREQ("Idle", trainingStateToString(TrainingState::IDLE));
    EXPECT_STREQ("Initializing", trainingStateToString(TrainingState::INITIALIZING));
    EXPECT_STREQ("Training", trainingStateToString(TrainingState::TRAINING));
    EXPECT_STREQ("Validating", trainingStateToString(TrainingState::VALIDATING));
    EXPECT_STREQ("Paused", trainingStateToString(TrainingState::PAUSED));
    EXPECT_STREQ("Completed", trainingStateToString(TrainingState::COMPLETED));
    EXPECT_STREQ("Failed", trainingStateToString(TrainingState::FAILED));
    EXPECT_STREQ("Stopped", trainingStateToString(TrainingState::STOPPED));
}

// TYP-006: trainingStateToString returns "Unknown" for invalid enum value
TEST(TypesTest, TrainingStateToString_InvalidValue) {
    TrainingState invalidState = static_cast<TrainingState>(999);
    EXPECT_STREQ("Unknown", trainingStateToString(invalidState));
}

// TYP-007: TrainingState enum values are correctly assigned
TEST(TypesTest, TrainingState_EnumValues) {
    EXPECT_EQ(0, static_cast<int>(TrainingState::IDLE));
    EXPECT_EQ(1, static_cast<int>(TrainingState::INITIALIZING));
    EXPECT_EQ(2, static_cast<int>(TrainingState::TRAINING));
    EXPECT_EQ(3, static_cast<int>(TrainingState::VALIDATING));
    EXPECT_EQ(4, static_cast<int>(TrainingState::PAUSED));
    EXPECT_EQ(5, static_cast<int>(TrainingState::COMPLETED));
    EXPECT_EQ(6, static_cast<int>(TrainingState::FAILED));
    EXPECT_EQ(7, static_cast<int>(TrainingState::STOPPED));
}

// TYP-008: TrainingState enum has expected number of values
TEST(TypesTest, TrainingState_EnumCount) {
    int validCount = 0;
    for (int i = 0; i <= 100; ++i) {
        TrainingState state = static_cast<TrainingState>(i);
        const char* str = trainingStateToString(state);
        if (std::string(str) != "Unknown") {
            ++validCount;
        }
    }
    EXPECT_EQ(8, validCount); // IDLE through STOPPED
}

// =============================================================================
// Struct Default Initialization Tests (TYP-009 ~ TYP-013)
// =============================================================================

// TYP-009: MetricsData default initialization
TEST(TypesTest, MetricsData_DefaultInitialization) {
    MetricsData metrics{};

    EXPECT_FLOAT_EQ(0.0f, metrics.loss);
    EXPECT_FLOAT_EQ(0.0f, metrics.accuracy);
    EXPECT_FLOAT_EQ(0.0f, metrics.precision);
    EXPECT_FLOAT_EQ(0.0f, metrics.recall);
    EXPECT_FLOAT_EQ(0.0f, metrics.f1Score);
    EXPECT_FLOAT_EQ(0.0f, metrics.mAP);
    EXPECT_FLOAT_EQ(0.0f, metrics.fitness);
    EXPECT_FLOAT_EQ(0.0f, metrics.threshold);
    EXPECT_FLOAT_EQ(0.0f, metrics.aucROC);
}

// TYP-010: MetricsData custom value assignment
TEST(TypesTest, MetricsData_CustomValues) {
    MetricsData metrics;
    metrics.loss = 0.5f;
    metrics.accuracy = 0.95f;
    metrics.precision = 0.92f;
    metrics.recall = 0.88f;
    metrics.f1Score = 0.90f;
    metrics.mAP = 0.87f;
    metrics.fitness = 0.93f;
    metrics.threshold = 0.5f;
    metrics.aucROC = 0.91f;

    EXPECT_FLOAT_EQ(0.5f, metrics.loss);
    EXPECT_FLOAT_EQ(0.95f, metrics.accuracy);
    EXPECT_FLOAT_EQ(0.92f, metrics.precision);
    EXPECT_FLOAT_EQ(0.88f, metrics.recall);
    EXPECT_FLOAT_EQ(0.90f, metrics.f1Score);
    EXPECT_FLOAT_EQ(0.87f, metrics.mAP);
    EXPECT_FLOAT_EQ(0.93f, metrics.fitness);
    EXPECT_FLOAT_EQ(0.5f, metrics.threshold);
    EXPECT_FLOAT_EQ(0.91f, metrics.aucROC);
}

// TYP-011: ProgressData initialization and value assignment
TEST(TypesTest, ProgressData_Initialization) {
    ProgressData progress{};
    progress.stage = ProgressStage::TRAIN_BATCH;
    progress.currentEpoch = 5;
    progress.totalEpochs = 100;
    progress.currentBatch = 10;
    progress.totalBatches = 50;
    progress.loss = 0.25f;
    progress.learningRate = 0.001f;
    progress.gpuMemoryUsage = 4096.0f;
    progress.elapsedTime = 120.5;
    progress.eta = 3600.0;
    progress.message = "Training in progress";

    EXPECT_EQ(ProgressStage::TRAIN_BATCH, progress.stage);
    EXPECT_EQ(5, progress.currentEpoch);
    EXPECT_EQ(100, progress.totalEpochs);
    EXPECT_EQ(10, progress.currentBatch);
    EXPECT_EQ(50, progress.totalBatches);
    EXPECT_FLOAT_EQ(0.25f, progress.loss);
    EXPECT_FLOAT_EQ(0.001f, progress.learningRate);
    EXPECT_FLOAT_EQ(4096.0f, progress.gpuMemoryUsage);
    EXPECT_DOUBLE_EQ(120.5, progress.elapsedTime);
    EXPECT_DOUBLE_EQ(3600.0, progress.eta);
    EXPECT_EQ("Training in progress", progress.message);
}

// TYP-012: BBox struct initialization
TEST(TypesTest, BBox_Initialization) {
    BBox box{10.0f, 20.0f, 100.0f, 200.0f};

    EXPECT_FLOAT_EQ(10.0f, box.x1);
    EXPECT_FLOAT_EQ(20.0f, box.y1);
    EXPECT_FLOAT_EQ(100.0f, box.x2);
    EXPECT_FLOAT_EQ(200.0f, box.y2);
}

// TYP-013: OBB struct initialization
TEST(TypesTest, OBB_Initialization) {
    OBB obb{50.0f, 100.0f, 80.0f, 40.0f, 0.785f}; // 45 degrees in radians

    EXPECT_FLOAT_EQ(50.0f, obb.cx);
    EXPECT_FLOAT_EQ(100.0f, obb.cy);
    EXPECT_FLOAT_EQ(80.0f, obb.width);
    EXPECT_FLOAT_EQ(40.0f, obb.height);
    EXPECT_FLOAT_EQ(0.785f, obb.angle);
}

// =============================================================================
// Contour Struct Tests (TYP-014)
// =============================================================================

// TYP-014: Contour struct with point data
TEST(TypesTest, Contour_Initialization) {
    Contour contour;
    contour.points = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

    EXPECT_EQ(6u, contour.points.size());
    EXPECT_FLOAT_EQ(10.0f, contour.points[0]);
    EXPECT_FLOAT_EQ(20.0f, contour.points[1]);
    EXPECT_FLOAT_EQ(30.0f, contour.points[2]);
    EXPECT_FLOAT_EQ(40.0f, contour.points[3]);
    EXPECT_FLOAT_EQ(50.0f, contour.points[4]);
    EXPECT_FLOAT_EQ(60.0f, contour.points[5]);
}

// =============================================================================
// ProgressStage Enum Tests (TYP-015)
// =============================================================================

// TYP-015: ProgressStage enum values exist
TEST(TypesTest, ProgressStage_EnumValues) {
    // Verify all ProgressStage values can be used
    ProgressStage stages[] = {
        ProgressStage::TRAIN_BATCH,
        ProgressStage::TRAIN_EPOCH,
        ProgressStage::VAL_BATCH,
        ProgressStage::VAL_EPOCH,
        ProgressStage::CHECKPOINT_SAVED
    };

    EXPECT_EQ(5u, sizeof(stages) / sizeof(stages[0]));
}

// =============================================================================
// Callback Type Tests (TYP-016 ~ TYP-017)
// =============================================================================

// TYP-016: ProgressCallback type can be used
TEST(TypesTest, ProgressCallback_Usage) {
    bool callbackInvoked = false;
    ProgressData receivedData{};

    ProgressCallback callback = [&callbackInvoked, &receivedData](const ProgressData& data) {
        callbackInvoked = true;
        receivedData = data;
    };

    ProgressData testData{};
    testData.currentEpoch = 42;
    testData.message = "test";

    callback(testData);

    EXPECT_TRUE(callbackInvoked);
    EXPECT_EQ(42, receivedData.currentEpoch);
    EXPECT_EQ("test", receivedData.message);
}

// TYP-017: EvaluationCallback type can be used
TEST(TypesTest, EvaluationCallback_Usage) {
    bool callbackInvoked = false;
    MetricsData receivedMetrics{};

    EvaluationCallback callback = [&callbackInvoked, &receivedMetrics](const MetricsData& metrics) {
        callbackInvoked = true;
        receivedMetrics = metrics;
    };

    MetricsData testMetrics{};
    testMetrics.accuracy = 0.95f;
    testMetrics.mAP = 0.87f;

    callback(testMetrics);

    EXPECT_TRUE(callbackInvoked);
    EXPECT_FLOAT_EQ(0.95f, receivedMetrics.accuracy);
    EXPECT_FLOAT_EQ(0.87f, receivedMetrics.mAP);
}
