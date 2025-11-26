#pragma once

#include "../Utils/Common/Types.h"
#include "nlohmann/json.hpp"
#include <string>
#include <vector>
#include <map>

// Forward declare yaml-cpp types
namespace YAML {
	class Node;
}

namespace WheelDL {
	namespace Config {

		/**
		 * @class Configuration
		 * @brief Central configuration management class
		 *
		 * Manages all configuration settings loaded from YAML files:
		 * - Model configuration (model.yaml)
		 * - Hyperparameters (default.yaml)
		 * - Dataset configuration (dataset.yaml)
		 */
		class Configuration {
		public:
			/**
			 * @brief Default constructor
			 */
			Configuration();

			/**
			 * @brief Destructor
			 */
			~Configuration();

			/**
			 * @brief Load configuration from YAML files
			 * @param modelPath Path to model YAML file
			 * @param hyperParamPath Path to hyperparameters YAML file
			 * @param datasetPath Path to dataset YAML file (optional)
			 * @throws ConfigurationException if any file fails to load
			 */
			void load(const std::string& modelPath,
				const std::string& hyperParamPath,
				const std::string& datasetPath);

			/**
			 * @brief Get task type (inferred from model config)
			 * @return TaskType Task type
			 */
			TaskType getTaskType() const { return _taskType; }
			
			/**
			 * @brief Get model YAML path
			 * @return std::string Model YAML path
			 */
			std::string getModelPath() const { return _modelPath; }

			/**
			 * @brief Get Dataset Json path
			 * @return std::string Dataset JSON path
			 */
			std::string getDatasetPath() const { return _datasetPath; }

			// ========== Training Hyperparameters ==========

			/**
			 * @brief Get number of training epochs
			 * @return int Epochs
			 */
			int getEpochs() const { return _epochs; }

			/**
			 * @brief Get early stopping patience
			 * @return int Patience (epochs without improvement)
			 */
			int getPatience() const { return _patience; }

			/**
			 * @brief Get batch size
			 * @return int Batch size
			 */
			int getBatchSize() const { return _batchSize; }

			/**
			 * @brief Get image size (square)
			 * @return int Image size (e.g., 640 for 640x640)
			 */
			int getImageSize() const { return _imageSize; }

			/**
			 * @brief Get device string
			 * @return std::string Device (e.g., "cuda", "cpu", "0")
			 */
			std::string getDevice() const { return _device; }

			/**
			 * @brief Get number of dataloader workers
			 * @return int Workers
			 */
			int getWorkers() const { return _workers; }

			/**
			 * @brief Get optimizer name
			 * @return std::string Optimizer (e.g., "SGD", "Adam", "auto")
			 */
			std::string getOptimizer() const { return _optimizer; }

			/**
			 * @brief Get random seed
			 * @return int Seed
			 */
			int getSeed() const { return _seed; }

			/**
			 * @brief Check if deterministic mode is enabled
			 * @return bool True if deterministic
			 */
			bool isDeterministic() const { return _deterministic; }

			/**
			 * @brief Check if cosine LR scheduler is enabled
			 * @return bool True if cosine LR
			 */
			bool useCosineLR() const { return _cosLR; }
			bool useLinearLR() const { return _linearLR; }
			/**
			 * @brief Get close mosaic epochs
			 * @return int Disable mosaic for final N epochs
			 */
			int getCloseMosaic() const { return _closeMosaic; }

			/**
			 * @brief Check if AMP (Automatic Mixed Precision) is enabled
			 * @return bool True if AMP
			 */
			bool useAMP() const { return _amp; }

			/**
			 * @brief Check if BFloat16 is enabled
			 * @return bool True if BFloat16
			 */
			bool useBFloat16() const { return _bfloat16; }

			/**
			 * @brief Get cache type
			 * @return std::string Cache type ("ram", "disk", or empty for False)
			 */
			std::string getCacheType() const { return _cache; }

			/**
			 * @brief Get cache maximum size
			 * @return size_t Maximum number of items in cache (0 = unlimited)
			 */
			size_t getCacheSize() const { return _cacheSize; }

			// ========== Optimizer Settings ==========

			/**
			 * @brief Get initial learning rate
			 * @return float Initial LR (lr0)
			 */
			float getLearningRate() const { return _lr0; }

			/**
			 * @brief Get final LR fraction
			 * @return float Final LR = lr0 * lrf
			 */
			float getLRFinalFraction() const { return _lrf; }

			/**
			 * @brief Get SGD momentum or Adam beta1
			 * @return float Momentum
			 */
			float getMomentum() const { return _momentum; }

			/**
			 * @brief Get weight decay (L2 regularization)
			 * @return float Weight decay
			 */
			float getWeightDecay() const { return _weightDecay; }
			
			/**
			 * @brief Get AMSGrad setting for Adam/AdamW
			 * @return bool AMSGrad setting (true/false)
			 */
			bool getAmsgrad() const { return _amsgrad; }

			/**
			 * @brief Get warmup epochs
			 * @return float Warmup epochs (fractions allowed)
			 */
			float getWarmupEpochs() const { return _warmupEpochs; }

			/**
			 * @brief Get warmup momentum
			 * @return float Initial momentum during warmup
			 */
			float getWarmupMomentum() const { return _warmupMomentum; }

			/**
			 * @brief Get warmup bias learning rate
			 * @return float Bias LR during warmup
			 */
			float getWarmupBiasLR() const { return _warmupBiasLR; }

			// ========== Loss Gains ==========

			/**
			 * @brief Get box loss gain
			 * @return float Box loss gain
			 */
			float getBoxGain() const { return _boxGain; }

			/**
			 * @brief Get classification loss gain
			 * @return float Classification loss gain
			 */
			float getClsGain() const { return _clsGain; }

			/**
			 * @brief Get DFL (Distribution Focal Loss) gain
			 * @return float DFL gain
			 */
			float getDFLGain() const { return _dflGain; }

			// ========== Augmentation Settings ==========

			bool getImageNetNorm() const { return this->isImageNetNormalized; }
			/**
			 * @brief Get HSV hue augmentation fraction
			 * @return float HSV-H
			 */
			float getHSVH() const { return _hsvH; }

			/**
			 * @brief Get HSV saturation augmentation fraction
			 * @return float HSV-S
			 */
			float getHSVS() const { return _hsvS; }

			/**
			 * @brief Get HSV value (brightness) augmentation fraction
			 * @return float HSV-V
			 */
			float getHSVV() const { return _hsvV; }

			/**
			 * @brief Get rotation degrees
			 * @return float Rotation degrees (+/-)
			 */
			float getDegrees() const { return _degrees; }

			/**
			 * @brief Get translation fraction
			 * @return float Translation fraction (+/-)
			 */
			float getTranslate() const { return _translate; }

			/**
			 * @brief Get scale gain
			 * @return float Scale gain (+/-)
			 */
			float getScale() const { return _scale; }

			/**
			 * @brief Get shear degrees
			 * @return float Shear degrees (+/-)
			 */
			float getShear() const { return _shear; }

			/**
			 * @brief Get perspective fraction
			 * @return float Perspective fraction
			 */
			float getPerspective() const { return _perspective; }

			/**
			 * @brief Get vertical flip probability
			 * @return float Flipud probability
			 */
			float getFlipUD() const { return _flipud; }

			/**
			 * @brief Get horizontal flip probability
			 * @return float Fliplr probability
			 */
			float getFlipLR() const { return _fliplr; }

			/**
			 * @brief Get mosaic augmentation probability
			 * @return float Mosaic probability
			 */
			float getMosaic() const { return _mosaic; }

			/**
			 * @brief Get Gaussian blur kernel size
			 * @return int Blur kernel size
			 */
			int getBlurKernelSize() const { return _blurKernelSize; }

			/**
			 * @brief Get Gaussian blur probability
			 * @return float Blur probability
			 */
			float getBlurProbability() const { return _blurProbability; }

			/**
			 * @brief Get border fill value for augmentations
			 * @return int Fill border value
			 */
			int getFillBorder() const { return _fillBorder; }

			// ========== Validation Settings ==========

			/**
			 * @brief Get IoU threshold for NMS
			 * @return float IoU threshold
			 */
			float getIoU() const { return _iou; }

			/**
			 * @brief Get maximum detections per image
			 * @return int Max detections
			 */
			int getMaxDet() const { return _maxDet; }

			// ========== Dataset Configuration ==========
			/**
			 * @brief Get class names map
			 * @return std::map<int, std::string> Class ID to name mapping
			 */
			const std::map<int, std::string>& getClassNames() const { return _classNames; }

			/**
			 * @brief Get number of classes
			 * @return int Number of classes
			 */
			int getNumClasses() const { return _numClasses; }

			
			void setImageSize(int imageSize) { _imageSize = imageSize; }
			/**
			 * @brief Set class names and update number of classes
			 * @param classNames Map of class ID to class name
			 */
			void setClassNames(const std::map<int, std::string>& classNames);

			/**
			 * @brief Set number of classes directly
			 * @param numClasses Number of classes
			 */
			void setNumClasses(int numClasses);

			/**
			 * @brief Get dropout rate (classification only)
			 * @return float Dropout rate
			 */
			float getDropout() const { return _dropout; }

			/**
			 * @brief Check if DDP (Distributed Data Parallel) should be used
			 * @return bool True if device string suggests multi-GPU
			 */
			bool useDDP() const;

			bool IsEfficientAD() const { return _isEfficientAD; }
			bool IsPatchCore() const { return _isPatchCore; }
			void setIsEfficientAD(bool val) { _isEfficientAD = val; }
			void setIsPatchCore(bool val) { _isPatchCore = val; }
			void setEpochs(int epochs) { _epochs = epochs; }
			void setWarmupEpochs(float warmupEpochs) { _warmupEpochs = warmupEpochs; }
			void setBatchSize(int batchSize) { _batchSize = batchSize; }
			void setImageNetNorm(bool val) { isImageNetNormalized = val; }

		private:
			// Task type (inferred from model)
			TaskType _taskType;

			// ========== Model Configuration ==========
			std::string _modelPath;

			// ========== Training Hyperparameters ==========
			int _epochs;
			int _patience;
			int _batchSize;
			int _imageSize;
			std::string _device;
			int _workers;
			std::string _optimizer;
			int _seed;
			bool _deterministic;
			bool _cosLR;
			bool _linearLR;
			int _closeMosaic;
			bool _amp;
			bool _bfloat16;
			std::string _cache;
			size_t _cacheSize;

			// ========== Optimizer Settings ==========
			float _lr0;
			float _lrf;
			float _momentum;
			float _weightDecay;
			float _warmupEpochs;
			float _warmupMomentum;
			float _warmupBiasLR;
			bool _amsgrad;
			
			// ========== Loss Gains ==========
			float _boxGain;
			float _clsGain;
			float _dflGain;

			// ========== Augmentation Settings ==========
			float _hsvH;
			float _hsvS;
			float _hsvV;
			float _degrees;
			float _translate;
			float _scale;
			float _shear;
			float _perspective;
			float _flipud;
			float _fliplr;
			float _mosaic;
			float _mixup;
			float _cutmix;
			int _blurKernelSize;
			float _blurProbability;
			int _fillBorder;
			bool isImageNetNormalized;

			// ========== Validation Settings ==========
			float _iou;
			int _maxDet;

			// ========== Dataset Configuration ==========
			std::string _datasetPath;
			int64_t _numClasses;
			std::map<int, std::string> _classNames;

			// ========== Classification Specific ==========
			float _dropout;

			// ====== Anomaly Specific ======
			bool _isEfficientAD = false;
			bool _isPatchCore = false;

			/**
			 * @brief Load model configuration
			 * @param modelConfig YAML node
			 */

			/**
			 * @brief Load hyperparameters
			 * @param hyperParamConfig YAML node
			 */
			void loadHyperParams(const YAML::Node& hyperParamConfig);

			/**
			 * @brief Load dataset configuration
			 * @param datasetConfig YAML node
			 */
			void loadDatasetConfig(const nlohmann::json& datasetJson);

			/**
			 * @brief Validate configuration values
			 * @throws std::invalid_argument if any value is out of valid range
			 */
			void validateConfiguration();
		};

	} // namespace Config
} // namespace WheelDL
