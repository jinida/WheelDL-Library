#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <opencv2/core/core.hpp>

namespace WheelDL {

	/**
	 * @enum TaskType
	 * @brief Deep learning task types supported by WheelDL.Lib
	 */
	enum class TaskType {
		CLASSIFICATION = 0,  // Image classification
		DETECTION = 1,       // Object detection
		SEGMENTATION = 2,    // Semantic/instance segmentation
		ANOMALY = 3,        // Anomaly detection
		OBB = 4,            // Oriented bounding box detection
		UNKNOWN = 99
	};

	/**
	 * @brief Convert TaskType to string
	 */
	inline const char* taskTypeToString(TaskType type) {
		switch (type) {
		case TaskType::CLASSIFICATION: return "Classification";
		case TaskType::DETECTION: return "Detection";
		case TaskType::OBB: return "OBB";
		case TaskType::SEGMENTATION: return "Segmentation";
		case TaskType::ANOMALY: return "Anomaly";
		default: return "Unknown";
		}
	}

	/**
	 * @enum TrainingState
	 * @brief Training process state
	 */
	enum class TrainingState {
		IDLE = 0,           // Not started
		INITIALIZING = 1,   // Initializing resources
		TRAINING = 2,       // Training in progress
		VALIDATING = 3,     // Validation in progress
		PAUSED = 4,         // Paused by user
		COMPLETED = 5,      // Training completed successfully
		FAILED = 6,         // Training failed with error
		STOPPED = 7         // Stopped by user
	};

	/**
	 * @brief Convert TrainingState to string
	 */
	inline const char* trainingStateToString(TrainingState state)
	{
		switch (state)
		{
		case TrainingState::IDLE: return "Idle";
		case TrainingState::INITIALIZING: return "Initializing";
		case TrainingState::TRAINING: return "Training";
		case TrainingState::VALIDATING: return "Validating";
		case TrainingState::PAUSED: return "Paused";
		case TrainingState::COMPLETED: return "Completed";
		case TrainingState::FAILED: return "Failed";
		case TrainingState::STOPPED: return "Stopped";
		default: return "Unknown";
		}
	}

	/**
	 * @enum ProgressStage
	 * @brief Training progress stage
	 */
	enum class ProgressStage {
		TRAIN_BATCH,
		TRAIN_EPOCH,
		VAL_BATCH,
		VAL_EPOCH,
		CHECKPOINT_SAVED
	};

	/**
	 * @struct MetricsData
	 * @brief Training and validation metrics
	 */
	struct MetricsData 
	{
		float loss = 0.0f;
		float accuracy = 0.0f;
		float precision = 0.0f;
		float recall = 0.0f;
		float f1Score = 0.0f;
		float mAP = 0.0f;
		float fitness = 0.0f;
		float threshold = 0.0f;
		float aucROC = 0.0f;
	};

	/**
	 * @struct ProgressData
	 * @brief Training progress information
	 */
	struct ProgressData {
		ProgressStage stage;
		int currentEpoch;
		int totalEpochs;
		int currentBatch;
		int totalBatches;
		float loss;
		MetricsData metrics;
		float learningRate;
		float gpuMemoryUsage;
		double elapsedTime;
		double eta;  // Estimated Time of Arrival
		std::string message;
	};

	/**
	 * @struct BBox
	 * @brief Bounding box with top-left and bottom-right coordinates
	 */
	struct BBox {
		float x1, y1;  // Top-left corner
		float x2, y2;  // Bottom-right corner
	};

	/**
	 * @struct OBB
	 * @brief Oriented Bounding Box with center, size, and rotation angle
	 */
	struct OBB {
		float cx, cy;   // Center coordinates
		float width;    // Width
		float height;   // Height
		float angle;    // Rotation angle in radians
	};

	/**
	 * @struct Contour
	 * @brief Contour points for segmentation
	 */
	struct Contour {
		std::vector<float> points;  // [x1, y1, x2, y2, ...] flattened coordinates
	};

	/**
	 * @struct PredictionResult
	 * @brief Prediction/inference result (no Torch dependency)
	 */
	struct PredictionResult {
		std::vector<BBox> boxes;                    // Detection bounding boxes
		std::vector<OBB> orientedBoxes;             // Oriented bounding boxes (for OBB task)
		std::vector<float> scores;                  // Confidence scores
		std::vector<unsigned int> classIds;         // Class IDs
		std::vector<Contour> contours;              // Segmentation contours (optional)
		cv::Mat anomalyMap;
		// Original image shape (for coordinate scaling)
		std::pair<int, int> originalShape;          // (height, width)

		// Prediction metadata
		double inferenceTime;                       // Inference time in milliseconds
		int numDetections;                          // Number of detections
	};

	// Forward declaration for callbacks
	struct ProgressData;
	struct MetricsData;

	/**
	 * @brief Callback function for training progress updates
	 */
	using ProgressCallback = std::function<void(const ProgressData&)>;

	/**
	 * @brief Callback function for evaluation progress updates
	 */
	using EvaluationCallback = std::function<void(const MetricsData&)>;

} // namespace WheelDL
