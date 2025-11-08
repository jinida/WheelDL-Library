#pragma once

#include <string>

namespace WheelDL {
	namespace Constants {
		constexpr const char* LIBRARY_NAME = "WheelDL.Lib";
		
		constexpr uint32_t MAJOR_VERSION = 0;
		constexpr uint32_t MINOR_VERSION = 1;
		constexpr uint32_t PATCH_VERSION = 0;

		constexpr uint32_t CACHE_MAJOR_VERSION = 0;
		constexpr uint32_t CACHE_MINOR_VERSION = 1;
		constexpr uint32_t CACHE_PATCH_VERSION = 0;

		// Default hyperparameters
		constexpr int DEFAULT_BATCH_SIZE = 32;
		constexpr int DEFAULT_EPOCHS = 100;
		constexpr float DEFAULT_LEARNING_RATE = 0.001f;
		constexpr float DEFAULT_WEIGHT_DECAY = 0.0001f;
		constexpr float DEFAULT_MOMENTUM = 0.9f;
		constexpr int DEFAULT_WARMUP_EPOCHS = 3;

		// Training constraints
		constexpr int MIN_BATCH_SIZE = 1;
		constexpr int MAX_BATCH_SIZE = 1024;
		constexpr float MIN_LEARNING_RATE = 1e-7f;
		constexpr float MAX_LEARNING_RATE = 1.0f;

		// Image size constraints
		constexpr int MIN_IMAGE_SIZE = 16;
		constexpr int MAX_IMAGE_SIZE = 2048;
		constexpr int DEFAULT_IMAGE_SIZE = 640;

		// Memory management
		constexpr size_t DEFAULT_CACHE_SIZE_MB = 1024;  // 1GB
		constexpr size_t MIN_GPU_MEMORY_MB = 2048;      // 2GB minimum GPU memory
		constexpr float GPU_MEMORY_RESERVE_RATIO = 0.1f;  // Reserve 10% for safety

		// Threading
		constexpr int DEFAULT_NUM_WORKERS = 4;
		constexpr int MAX_NUM_WORKERS = 32;

		// Checkpoint and logging
		constexpr int DEFAULT_SAVE_INTERVAL = 10;  // Save every N epochs
		constexpr int DEFAULT_LOG_INTERVAL = 10;   // Log every N iterations
		constexpr int DEFAULT_VAL_INTERVAL = 1;    // Validate every N epochs

		// Early stopping
		constexpr int DEFAULT_PATIENCE = 10;  // Stop if no improvement for N epochs
		constexpr float DEFAULT_MIN_DELTA = 1e-4f;  // Minimum change to qualify as improvement

		// Model EMA
		constexpr float DEFAULT_EMA_DECAY = 0.9999f;

		// File extensions
		namespace FileExtensions {
			constexpr const char* YAML = ".yaml";
			constexpr const char* YML = ".yml";
			constexpr const char* PT = ".pt";
			constexpr const char* ONNX = ".onnx";
			constexpr const char* JSON = ".json";
			constexpr const char* TXT = ".txt";
		}

		// Configuration keys
		namespace ConfigKeys {
			// Task
			constexpr const char* TASK = "task";
			constexpr const char* NUM_CLASSES = "num_classes";

			// Model
			constexpr const char* MODEL = "model";
			constexpr const char* MODEL_PATH = "model_path";
			constexpr const char* PRETRAINED = "pretrained";

			// Data
			constexpr const char* DATA = "data";
			constexpr const char* TRAIN_PATH = "train_path";
			constexpr const char* VAL_PATH = "val_path";
			constexpr const char* TEST_PATH = "test_path";
			constexpr const char* IMAGE_SIZE = "image_size";

			// Hyperparameters
			constexpr const char* BATCH_SIZE = "batch_size";
			constexpr const char* EPOCHS = "epochs";
			constexpr const char* LEARNING_RATE = "lr";
			constexpr const char* WEIGHT_DECAY = "weight_decay";
			constexpr const char* MOMENTUM = "momentum";
			constexpr const char* OPTIMIZER = "optimizer";
			constexpr const char* SCHEDULER = "scheduler";

			// Augmentation
			constexpr const char* AUGMENTATION = "augmentation";
			constexpr const char* USE_MOSAIC = "mosaic";
			constexpr const char* USE_MIXUP = "mixup";
			constexpr const char* FLIP_LR = "flip_lr";
			constexpr const char* FLIP_UD = "flip_ud";

			// Training
			constexpr const char* DEVICE = "device";
			constexpr const char* NUM_WORKERS = "num_workers";
			constexpr const char* SAVE_DIR = "save_dir";
			constexpr const char* RESUME = "resume";

			// Distributed
			constexpr const char* DISTRIBUTED = "distributed";
			constexpr const char* WORLD_SIZE = "world_size";
			constexpr const char* RANK = "rank";
			constexpr const char* DIST_BACKEND = "dist_backend";
			constexpr const char* DIST_URL = "dist_url";
		}

		// Default paths
		namespace DefaultPaths {
			constexpr const char* RUNS_DIR = "./runs";
			constexpr const char* WEIGHTS_DIR = "./weights";
			constexpr const char* LOGS_DIR = "./logs";
			constexpr const char* CACHE_DIR = "./.cache";
		}

		// Distributed training
		namespace Distributed {
			constexpr const char* NCCL_BACKEND = "nccl";
			constexpr const char* GLOO_BACKEND = "gloo";
			constexpr const char* DEFAULT_MASTER_PORT = "29500";
			constexpr int DEFAULT_TIMEOUT_SECONDS = 1800;  // 30 minutes
		}

	} // namespace Constants
} // namespace WheelDL
