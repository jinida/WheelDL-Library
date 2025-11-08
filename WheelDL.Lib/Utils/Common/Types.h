#pragma once

#include <string>
#include <vector>
#include <cstdint>

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
	inline const char* trainingStateToString(TrainingState state) {
		switch (state) {
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
	 * @struct ProgressData
	 * @brief Training progress information
	 */
	struct ProgressData {
		int currentEpoch;
		int totalEpochs;
		int currentStep;
		int totalSteps;
		float trainLoss;
		float valLoss;
		float learningRate;
		std::string message;
	};

	/**
	 * @struct MetricsData
	 * @brief Training and validation metrics
	 */
	struct MetricsData {
		float loss;
		float accuracy;
		float precision;
		float recall;
		float f1Score;
		float mAP;  // mean Average Precision (for detection)
		float IoU;  // Intersection over Union
	};

	/**
	 * @struct HyperParameters
	 * @brief Training hyperparameters
	 */
	struct HyperParameters {
		int batchSize;
		int epochs;
		float learningRate;
		float weightDecay;
		float momentum;
		int warmupEpochs;
		std::string optimizer;  // "SGD", "Adam", "AdamW"
		std::string scheduler;  // "CosineAnnealing", "Linear", "None"
	};

	/**
	 * @struct DatasetInfo
	 * @brief Dataset information
	 */
	struct DatasetInfo {
		std::string name;
		std::string path;
		int numClasses;
		int numTrainSamples;
		int numValSamples;
		int numTestSamples;
		std::vector<std::string> classNames;
	};

} // namespace WheelDL
