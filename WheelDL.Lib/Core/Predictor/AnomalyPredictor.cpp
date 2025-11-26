#include "pch.h"
#include "AnomalyPredictor.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"

namespace WheelDL {
	namespace Core {
		namespace Predictor {

			AnomalyPredictor::AnomalyPredictor(
				std::shared_ptr<Config::Configuration> config,
				const std::string& checkpointPath)
				: BasePredictor(config, checkpointPath)
				, _isModelPrepared(false)
			{
				_logger->info("AnomalyPredictor", "Initializing anomaly detection predictor");

				// Initialize ImageNet normalization tensors (CPU)
				_mean = torch::tensor({ 0.485f, 0.456f, 0.406f }, torch::kFloat32).view({ 1, 3, 1, 1 }).to(_device);
				_std = torch::tensor({ 0.229f, 0.224f, 0.225f }, torch::kFloat32).view({ 1, 3, 1, 1 }).to(_device);
				
				// Get model YAML path from configuration
				_modelYamlPath = config->getModelPath();
				if (_modelYamlPath.empty())
				{
					throw Utils::ConfigurationException(
						Utils::ErrorCode::INVALID_CONFIG,
						"Model YAML path is not specified in configuration"
					);
				}

				// Verify task type
				if (config->getTaskType() != TaskType::ANOMALY)
				{
					throw Utils::ConfigurationException(
						Utils::ErrorCode::INVALID_CONFIG,
						"Configuration task type must be ANOMALY for AnomalyPredictor"
					);
				}

				_logger->info("AnomalyPredictor", "Model YAML: " + _modelYamlPath);

				// Setup model (must be called after derived class construction)
				setupModel();

				// Load checkpoint if provided
				if (!checkpointPath.empty())
				{
					loadCheckpoint(checkpointPath);
				}

				_config->setImageNetNorm(true);
				_logger->info("AnomalyPredictor", "Anomaly threshold: " + std::to_string(_threshold));
				_logger->info("AnomalyPredictor", "Anomaly predictor initialized");
			}

			void AnomalyPredictor::setupModel()
			{
				_profiler.start("setup_model");
				_logger->info("AnomalyPredictor", "Setting up anomaly detection model");

				try {
					// Create AnomalyModel with config and model YAML path
					_model = std::make_unique<Model::AnomalyModel>(_config);
					_logger->info("AnomalyPredictor", "Model setup completed");
				}
				catch (const std::exception& e) {
					throw Utils::ModelException(
						Utils::ErrorCode::MODEL_LOAD_FAILED,
						"Failed to setup anomaly model: " + std::string(e.what())
					);
				}

				_profiler.stop("setup_model");
			}

			torch::Tensor AnomalyPredictor::preprocess(const torch::Tensor& input)
			{
				try {
					torch::Tensor preprocessed = input;

					// Ensure input is [C, H, W] or [1, C, H, W]
					if (preprocessed.dim() == 3) 
					{
						preprocessed = preprocessed.unsqueeze(0);
					}
					else if (preprocessed.dim() != 4) {
						throw Utils::WheelLibException(
							Utils::ErrorCode::PREDICTION_FAILED,
							"Input tensor must be [C, H, W] or [1, C, H, W], got " + std::to_string(preprocessed.dim()) + "D"
						);
					}

					if (preprocessed.dtype() == torch::kFloat32) 
					{
						return preprocessed;
					}

					int targetSize = _config->getImageSize();
					int origH = preprocessed.size(2);
					int origW = preprocessed.size(3);

					// Early return if already correct size
					if (origH == targetSize && origW == targetSize) {
						return preprocessed;
					}

					// Calculate new dimensions (aspect ratio preserving)
					int maxDim = std::max(origH, origW);
					float scale = static_cast<float>(targetSize) / static_cast<float>(maxDim);
					int newH = static_cast<int>(origH * scale + 0.5f);  // Fast rounding
					int newW = static_cast<int>(origW * scale + 0.5f);

					// Resize if needed
					if (newH != origH || newW != origW) {
						preprocessed = torch::nn::functional::interpolate(
							preprocessed,
							torch::nn::functional::InterpolateFuncOptions()
							.size(std::vector<int64_t>{newH, newW})
							.mode(torch::kBilinear)
							.align_corners(false)
						);
					}

					// Pad if needed (calculate padding directly in function call)
					if (newH != targetSize || newW != targetSize) {
						preprocessed = torch::nn::functional::pad(
							preprocessed,
							torch::nn::functional::PadFuncOptions({ 0, targetSize - newW, 0, targetSize - newH })
							.mode(torch::kConstant)
							.value(0.0)
						);
					}

					preprocessed = preprocessed.to(torch::kFloat32).mul(1.0f / 255.0f);
					preprocessed = preprocessed.sub(_mean).div(_std);

					return preprocessed;
				}
				catch (const std::exception& e) {
					throw Utils::WheelLibException(
						Utils::ErrorCode::PREDICTION_FAILED,
						"Preprocessing failed: " + std::string(e.what())
					);
				}
			}

			PredictionResult AnomalyPredictor::postprocess(
				const std::vector<torch::Tensor>& output,
				const std::tuple<int, int>& originalShape)
			{
				PredictionResult result;

				try
				{
					const torch::Tensor& anomalyScore = output[0];
					const torch::Tensor& anomalyMap = output[1];

					float imageLevelScore = anomalyScore.item<float>();
					unsigned int classId = (imageLevelScore > _threshold) ? 1 : 0;

					result.scores.push_back(imageLevelScore);
					result.classIds.push_back(classId);

					int origH = std::get<0>(originalShape);
					int origW = std::get<1>(originalShape);
					result.originalShape = { origH, origW };

					int targetSize = _config->getImageSize();
					int maxDim = std::max(origH, origW);
					float scale = static_cast<float>(targetSize) / static_cast<float>(maxDim);

					int validH = static_cast<int>(origH * scale + 0.5f);
					int validW = static_cast<int>(origW * scale + 0.5f);

					torch::Tensor unpaddedMap = anomalyMap;
					if (unpaddedMap.dim() == 4) {
						unpaddedMap = unpaddedMap.slice(2, 0, validH).slice(3, 0, validW);
					}
					else if (unpaddedMap.dim() == 3) {
						unpaddedMap = unpaddedMap.slice(1, 0, validH).slice(2, 0, validW);
					}
					else {
						unpaddedMap = unpaddedMap.slice(0, 0, validH).slice(1, 0, validW);
					}

					if (unpaddedMap.dim() == 2) {
						unpaddedMap = unpaddedMap.unsqueeze(0).unsqueeze(0);
					}
					else if (unpaddedMap.dim() == 3) {
						unpaddedMap = unpaddedMap.unsqueeze(0);
					}

					torch::Tensor restoredMap = torch::nn::functional::interpolate(
						unpaddedMap,
						torch::nn::functional::InterpolateFuncOptions()
						.size(std::vector<int64_t>{origH, origW})
						.mode(torch::kBilinear)
						.align_corners(false)
					);

					restoredMap = restoredMap.squeeze();

					auto minVal = restoredMap.min();
					auto maxVal = restoredMap.max();
					auto denominator = maxVal - minVal;

					if (denominator.item<float>() > 1e-6) {
						restoredMap = (restoredMap - minVal) / denominator;
					}

					auto anomalyMapCPU = restoredMap.mul(255).clamp(0, 255).to(torch::kUInt8).to(torch::kCPU).contiguous();

					cv::Mat grayMap(anomalyMapCPU.size(0), anomalyMapCPU.size(1), CV_8UC1, anomalyMapCPU.data_ptr<uint8_t>());
					cv::applyColorMap(grayMap, result.anomalyMap, cv::COLORMAP_JET);
				}
				catch (const std::exception& e)
				{
					_logger->error("AnomalyPredictor", "Postprocessing failed: " + std::string(e.what()));
					result.scores.push_back(0.0f);
					result.classIds.push_back(0);
				}

				return result;
			}

		} // namespace Predictor
	} // namespace Core
} // namespace WheelDL
