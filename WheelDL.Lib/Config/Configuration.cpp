#include "pch.h"
#include "Configuration.h"
#include "YamlParser.h"
#include "Utils/Error/WheelLibException.h"
#include <yaml-cpp/yaml.h>

namespace WheelDL {
namespace Config {

Configuration::Configuration()
	: _taskType(TaskType::UNKNOWN)
	, _numClasses(0)
	, _defaultActivation("ReLU")
	, _widthMultiple(1.0f)
	, _depthMultiple(1.0f)
	, _maxChannels(1024)
	, _epochs(100)
	, _patience(100)
	, _batchSize(16)
	, _imageSize(640)
	, _device("")
	, _workers(8)
	, _optimizer("auto")
	, _seed(0)
	, _deterministic(false)
	, _cosLR(false)
	, _closeMosaic(10)
	, _amp(true)
	, _bfloat16(false)
	, _cache("")
	, _lr0(0.01f)
	, _lrf(0.01f)
	, _momentum(0.937f)
	, _weightDecay(0.0005f)
	, _warmupEpochs(3.0f)
	, _warmupMomentum(0.8f)
	, _warmupBiasLR(0.1f)
	, _boxGain(7.5f)
	, _clsGain(0.5f)
	, _dflGain(1.5f)
	, _hsvH(0.015f)
	, _hsvS(0.7f)
	, _hsvV(0.4f)
	, _degrees(0.0f)
	, _translate(0.1f)
	, _scale(0.5f)
	, _shear(0.0f)
	, _perspective(0.0f)
	, _flipud(0.0f)
	, _fliplr(0.5f)
	, _mosaic(1.0f)
	, _mixup(0.0f)
	, _cutmix(0.0f)
	, _blurKernelSize(3)
	, _blurProbability(0.01f)
	, _fillBorder(0)
	, _iou(0.7f)
	, _maxDet(300)
	, _dropout(0.0f)
	, _cacheSize(256)
{
}

Configuration::~Configuration() {
}

void Configuration::loadFromYaml(const std::string& modelPath,
	const std::string& hyperParamPath,
	const std::string& datasetPath) {
	using namespace Utils;

	// Load model configuration
	YAML::Node modelConfig = YamlParser::parseFrom(modelPath);
	loadModelConfig(modelConfig);

	// Infer task type from model
	_taskType = YamlParser::inferTaskTypeFrom(modelConfig);

	// Load hyperparameters
	YAML::Node hyperParamConfig = YamlParser::parseFrom(hyperParamPath);
	loadHyperParams(hyperParamConfig);

	// Load dataset configuration (optional)
	if (!datasetPath.empty()) {
		YAML::Node datasetConfig = YamlParser::parseFrom(datasetPath);
		loadDatasetConfig(datasetConfig);
	}
}

void Configuration::loadModelConfig(const YAML::Node& modelConfig) {
	// Load num_classes
	_numClasses = YamlParser::getInt(modelConfig, "num_classes", 1000);

	// Load default activation
	_defaultActivation = YamlParser::getString(modelConfig, "default_act", "ReLU");

	// Load scale parameters
	if (modelConfig["scale"]) {
		const YAML::Node& scale = modelConfig["scale"];
		_widthMultiple = YamlParser::getFloat(scale, "width_multiple", 1.0f);
		_depthMultiple = YamlParser::getFloat(scale, "depth_multiple", 1.0f);
		_maxChannels = YamlParser::getInt(scale, "max_channels", 1024);
	}
}

void Configuration::loadHyperParams(const YAML::Node& hyperParamConfig) {
	// Training settings
	_epochs = YamlParser::getInt(hyperParamConfig, "epochs", 100);
	_patience = YamlParser::getInt(hyperParamConfig, "patience", 100);
	// Try "batch" first, then "batch_size" for backwards compatibility
	_batchSize = YamlParser::getInt(hyperParamConfig, "batch", YamlParser::getInt(hyperParamConfig, "batch_size", 16));
	// Try "imgsz" first, then "image_size" for backwards compatibility
	_imageSize = YamlParser::getInt(hyperParamConfig, "imgsz", YamlParser::getInt(hyperParamConfig, "image_size", 640));
	_workers = YamlParser::getInt(hyperParamConfig, "workers", 8);

	// Device settings
	if (hyperParamConfig["device"]) {
		try {
			if (hyperParamConfig["device"].IsScalar()) {
				// Try to parse as int first (e.g., device: 0)
				try {
					int deviceInt = hyperParamConfig["device"].as<int>();
					_device = std::to_string(deviceInt);
				}
				catch (...) {
					// If int parsing fails, try as string (e.g., device: "0" or device: "cuda:0")
					_device = hyperParamConfig["device"].as<std::string>("");
				}
			}
			else if (hyperParamConfig["device"].IsSequence()) {
				// Device is list like [0,1,2,3] - convert to string
				std::string deviceStr = "";
				for (size_t i = 0; i < hyperParamConfig["device"].size(); ++i) {
					if (i > 0) deviceStr += ",";
					deviceStr += std::to_string(hyperParamConfig["device"][i].as<int>());
				}
				_device = deviceStr;
			}
			else if (hyperParamConfig["device"].IsNull()) {
				_device = "";
			}
		}
		catch (const YAML::Exception& e) {
			// YAML parsing error - use default empty device (auto-detect)
			_device = "";
		}
		catch (const std::exception& e) {
			// Other errors (bad_cast, etc.) - use default
			_device = "";
		}
	}

	// Optimizer settings
	const auto validOptimizer = { "SGD", "Adam", "AdamW", "auto" };
	_optimizer = YamlParser::getString(hyperParamConfig, "optimizer", "auto");
	if (std::find(validOptimizer.begin(), validOptimizer.end(), _optimizer) == validOptimizer.end())
	{
		_optimizer = "auto"; // Fallback to auto if invalid
	}

	_seed = YamlParser::getInt(hyperParamConfig, "seed", 0);
	_deterministic = YamlParser::getBool(hyperParamConfig, "deterministic", false);
	_cosLR = YamlParser::getBool(hyperParamConfig, "cos_lr", false);
	_closeMosaic = YamlParser::getInt(hyperParamConfig, "close_mosaic", 10);
	_amp = YamlParser::getBool(hyperParamConfig, "amp", true);
	_bfloat16 = YamlParser::getBool(hyperParamConfig, "bfloat16", false);

	// Cache settings
	if (hyperParamConfig["cache"]) {
		try {
			// Try to parse as bool first
			bool cacheEnabled = hyperParamConfig["cache"].as<bool>();
			_cache = cacheEnabled ? "ram" : "";
		}
		catch (...) {
			// If bool parsing fails, parse as string
			_cache = hyperParamConfig["cache"].as<std::string>("");
		}
	}

	// Cache size (max number of cached items, 0 = unlimited)
	_cacheSize = YamlParser::getInt(hyperParamConfig, "cache_size", 0);

	// Learning rate and optimizer parameters
	_lr0 = YamlParser::getFloat(hyperParamConfig, "lr0", 0.01f);
	_lrf = YamlParser::getFloat(hyperParamConfig, "lrf", 0.01f);
	_momentum = YamlParser::getFloat(hyperParamConfig, "momentum", 0.937f);
	_weightDecay = YamlParser::getFloat(hyperParamConfig, "weight_decay", 0.0005f);
	_warmupEpochs = YamlParser::getFloat(hyperParamConfig, "warmup_epochs", 3.0f);
	_warmupMomentum = YamlParser::getFloat(hyperParamConfig, "warmup_momentum", 0.8f);
	_warmupBiasLR = YamlParser::getFloat(hyperParamConfig, "warmup_bias_lr", 0.1f);

	// Loss gains
	_boxGain = YamlParser::getFloat(hyperParamConfig, "box", 7.5f);
	_clsGain = YamlParser::getFloat(hyperParamConfig, "cls", 0.5f);
	_dflGain = YamlParser::getFloat(hyperParamConfig, "dfl", 1.5f);

	// Augmentation settings
	_hsvH = YamlParser::getFloat(hyperParamConfig, "hsv_h", 0.015f);
	_hsvS = YamlParser::getFloat(hyperParamConfig, "hsv_s", 0.7f);
	_hsvV = YamlParser::getFloat(hyperParamConfig, "hsv_v", 0.4f);
	_degrees = (TaskType::DETECTION == _taskType || TaskType::ANOMALY == _taskType) ? 0.0f : YamlParser::getFloat(hyperParamConfig, "degrees", 0.0f);
	_translate = YamlParser::getFloat(hyperParamConfig, "translate", 0.1f);
	_scale = YamlParser::getFloat(hyperParamConfig, "scale", 0.5f);
	_shear = YamlParser::getFloat(hyperParamConfig, "shear", 0.0f);
	// Perspective is not applicable for OBB and ANOMALY tasks
	_perspective = (TaskType::OBB == _taskType || TaskType::ANOMALY == _taskType)
		? 0.0f
		: YamlParser::getFloat(hyperParamConfig, "perspective", 0.0f);
	
	_flipud = YamlParser::getFloat(hyperParamConfig, "flipud", 0.0f);
	_fliplr = YamlParser::getFloat(hyperParamConfig, "fliplr", 0.5f);
	_mosaic = YamlParser::getFloat(hyperParamConfig, "mosaic", 1.0f);
	_mixup = YamlParser::getFloat(hyperParamConfig, "mixup", 0.0f);
	_cutmix = YamlParser::getFloat(hyperParamConfig, "cutmix", 0.0f);
	_blurKernelSize = YamlParser::getInt(hyperParamConfig, "blur_kernel_size", 3);
	_blurProbability = (TaskType::ANOMALY == _taskType) ? 0.0f : YamlParser::getFloat(hyperParamConfig, "blur_probability", 0.01f);
	_fillBorder = YamlParser::getInt(hyperParamConfig, "fill_border", 0);

	// Validation settings
	_iou = YamlParser::getFloat(hyperParamConfig, "iou", 0.7f);
	_maxDet = YamlParser::getInt(hyperParamConfig, "max_det", 300);

	// Classification specific
	_dropout = YamlParser::getFloat(hyperParamConfig, "dropout", 0.0f);

	// Note: validateConfiguration() is not called here to allow lazy validation
	// Users should call validate() explicitly after loading if needed
}

void Configuration::validateConfiguration() {
	// Validate positive integers
	if (_epochs <= 0) {
		throw std::invalid_argument("epochs must be positive, got: " + std::to_string(_epochs));
	}
	if (_batchSize <= 0) {
		throw std::invalid_argument("batch_size must be positive, got: " + std::to_string(_batchSize));
	}
	if (_imageSize <= 0) {
		throw std::invalid_argument("image_size must be positive, got: " + std::to_string(_imageSize));
	}
	if (_workers < 0) {
		throw std::invalid_argument("workers must be non-negative, got: " + std::to_string(_workers));
	}
	if (_maxDet <= 0) {
		throw std::invalid_argument("max_det must be positive, got: " + std::to_string(_maxDet));
	}

	// Validate non-negative integers
	if (_patience < 0) {
		throw std::invalid_argument("patience must be non-negative, got: " + std::to_string(_patience));
	}
	if (_closeMosaic < 0) {
		throw std::invalid_argument("close_mosaic must be non-negative, got: " + std::to_string(_closeMosaic));
	}
	if (_blurKernelSize < 0) {
		throw std::invalid_argument("blur_kernel_size must be non-negative, got: " + std::to_string(_blurKernelSize));
	}

	// Validate probabilities (0.0 - 1.0 range)
	auto validateProbability = [](float value, const std::string& name) {
		if (value < 0.0f || value > 1.0f) {
			throw std::invalid_argument(name + " must be in range [0.0, 1.0], got: " + std::to_string(value));
		}
	};

	validateProbability(_hsvH, "hsv_h");
	validateProbability(_hsvS, "hsv_s");
	validateProbability(_hsvV, "hsv_v");
	validateProbability(_translate, "translate");
	validateProbability(_scale, "scale");
	validateProbability(_flipud, "flipud");
	validateProbability(_fliplr, "fliplr");
	validateProbability(_mosaic, "mosaic");
	validateProbability(_mixup, "mixup");
	validateProbability(_cutmix, "cutmix");
	validateProbability(_blurProbability, "blur_probability");
	validateProbability(_dropout, "dropout");
	validateProbability(_iou, "iou");

	// Validate model architecture parameters
	if (_widthMultiple <= 0.0f) {
		throw std::invalid_argument("width_multiple must be positive, got: " + std::to_string(_widthMultiple));
	}
	if (_widthMultiple > 10.0f) {
		throw std::invalid_argument("width_multiple is too large (> 10.0), got: " + std::to_string(_widthMultiple));
	}
	if (_depthMultiple <= 0.0f) {
		throw std::invalid_argument("depth_multiple must be positive, got: " + std::to_string(_depthMultiple));
	}
	if (_depthMultiple > 10.0f) {
		throw std::invalid_argument("depth_multiple is too large (> 10.0), got: " + std::to_string(_depthMultiple));
	}
	if (_maxChannels <= 0) {
		throw std::invalid_argument("max_channels must be positive, got: " + std::to_string(_maxChannels));
	}
	if (_maxChannels > 4096) {
		throw std::invalid_argument("max_channels is too large (> 4096), got: " + std::to_string(_maxChannels));
	}

	// Validate learning rate parameters
	if (_lr0 < 0.0f) {
		throw std::invalid_argument("lr0 (initial learning rate) must be non-negative, got: " + std::to_string(_lr0));
	}
	if (_lr0 > 1.0f) {
		throw std::invalid_argument("lr0 (initial learning rate) is too large (> 1.0), got: " + std::to_string(_lr0));
	}
	if (_lrf < 0.0f) {
		throw std::invalid_argument("lrf (final learning rate factor) must be non-negative, got: " + std::to_string(_lrf));
	}
	if (_lrf > 1.0f) {
		throw std::invalid_argument("lrf (final learning rate factor) is too large (> 1.0), got: " + std::to_string(_lrf));
	}

	// Validate optimizer parameters
	if (_momentum < 0.0f || _momentum > 1.0f) {
		throw std::invalid_argument("momentum must be in range [0.0, 1.0], got: " + std::to_string(_momentum));
	}
	if (_weightDecay < 0.0f) {
		throw std::invalid_argument("weight_decay must be non-negative, got: " + std::to_string(_weightDecay));
	}
	if (_weightDecay > 1.0f) {
		throw std::invalid_argument("weight_decay is too large (> 1.0), got: " + std::to_string(_weightDecay));
	}

	// Validate training schedule parameters
	if (_warmupEpochs < 0) {
		throw std::invalid_argument("warmup_epochs must be non-negative, got: " + std::to_string(_warmupEpochs));
	}
	if (_warmupEpochs >= _epochs) {
		throw std::invalid_argument("warmup_epochs (" + std::to_string(_warmupEpochs) +
			") must be less than total epochs (" + std::to_string(_epochs) + ")");
	}
}

void Configuration::loadDatasetConfig(const YAML::Node& datasetConfig) {
	// Dataset name
	_datasetName = YamlParser::getString(datasetConfig, "name", "");

	// Paths
	_imagePath = YamlParser::getString(datasetConfig, "image_path", "images/");
	_labelPath = YamlParser::getString(datasetConfig, "label_path", "labels/");

	// Class names
	_classNames = YamlParser::getClassNames(datasetConfig);

	// Update num_classes from dataset class_names only if not already set by model
	if (!_classNames.empty() && _numClasses == 0) {
		_numClasses = static_cast<int>(_classNames.size());
	}
}

bool Configuration::useDDP() const {
	// DDP is used if device contains multiple GPUs
	// Examples: "0,1,2,3" or "[0,1,2,3]"
	if (_device.empty()) {
		return false;
	}

	// Count commas (indicates multiple devices)
	size_t commaCount = 0;
	for (char c : _device) {
		if (c == ',') {
			commaCount++;
		}
	}

	// If there's at least one comma, we have multiple GPUs (e.g., "0,1" has 1 comma = 2 GPUs)
	return commaCount >= 1;
}

void Configuration::setClassNames(const std::map<int, std::string>& classNames) {
	_classNames = classNames;
	if (!_classNames.empty()) {
		_numClasses = static_cast<int>(_classNames.size());
	}
}

void Configuration::setNumClasses(int numClasses) {
	if (numClasses < 0) {
		throw std::invalid_argument("numClasses must be non-negative, got: " + std::to_string(numClasses));
	}
	_numClasses = numClasses;
}

} // namespace Config
} // namespace WheelDL
