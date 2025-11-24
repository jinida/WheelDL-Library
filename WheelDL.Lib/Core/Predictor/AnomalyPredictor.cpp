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
                , _threshold(0.5f)
                , _isModelPrepared(false)
            {
                _logger->info("AnomalyPredictor", "Initializing anomaly detection predictor");

                // Get model YAML path from configuration
                _modelYamlPath = config->getModelPath();
                if (_modelYamlPath.empty()) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Model YAML path is not specified in configuration"
                    );
                }

                // Verify task type
                if (config->getTaskType() != TaskType::ANOMALY) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be ANOMALY for AnomalyPredictor"
                    );
                }

                _logger->info("AnomalyPredictor", "Model YAML: " + _modelYamlPath);
                _logger->info("AnomalyPredictor", "Anomaly threshold: " + std::to_string(_threshold));

                // Setup model (must be called after derived class construction)
                setupModel();

                // Load checkpoint if provided
                if (!checkpointPath.empty()) {
                    loadCheckpoint(checkpointPath);
                }

                _logger->info("AnomalyPredictor", "Anomaly predictor initialized");
            }

            void AnomalyPredictor::setupModel()
            {
                _profiler.start("setup_model");
                _logger->info("AnomalyPredictor", "Setting up anomaly detection model");

                try {
                    // Create AnomalyModel with config and model YAML path
                    _model = std::make_unique<Model::AnomalyModel>(_config);

                    // Move model to device (done in BasePredictor::setupDevice)
                    // Model will be loaded from checkpoint in constructor or via loadCheckpoint()

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
                    if (preprocessed.dim() == 3) {
                        // [C, H, W] -> [1, C, H, W]
                        preprocessed = preprocessed.unsqueeze(0);
                    }
                    else if (preprocessed.dim() != 4) {
                        throw Utils::WheelLibException(
                            Utils::ErrorCode::PREDICTION_FAILED,
                            "Input tensor must be [C, H, W] or [1, C, H, W], got " + std::to_string(preprocessed.dim()) + "D"
                        );
                    }

                    // Ensure float type
                    if (preprocessed.scalar_type() != torch::kFloat32) {
                        preprocessed = preprocessed.to(torch::kFloat32);
                    }

                    // Normalize if needed (expected range: [0, 1])
                    // If input is in [0, 255], normalize to [0, 1]
                    if (preprocessed.max().item<float>() > 1.0f) {
                        preprocessed = preprocessed / 255.0f;
                    }

                    // Resize to model input size if needed
                    int targetSize = _config->getImageSize();
                    int currentH = preprocessed.size(2);
                    int currentW = preprocessed.size(3);

                    if (currentH != targetSize || currentW != targetSize) {
                        preprocessed = torch::nn::functional::interpolate(
                            preprocessed,
                            torch::nn::functional::InterpolateFuncOptions()
                                .size(std::vector<int64_t>{targetSize, targetSize})
                                .mode(torch::kBilinear)
                                .align_corners(false)
                        );
                    }

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
                const torch::Tensor& output,
                const torch::Tensor& originalInput)
            {
                PredictionResult result;

                try 
                {
                    // Move to CPU
                    auto& anomalyMap = output;
                    // Compute image-level anomaly score
                    float imageLevelScore = computeImageLevelScore(anomalyMap);
                    // Determine if image is anomalous
                    unsigned int classId = (imageLevelScore > _threshold) ? 1 : 0;
                    // Extract contours for anomalous regions

                    // Fill PredictionResult
                    result.scores.push_back(imageLevelScore);
                    result.classIds.push_back(classId);

                    // Store original shape
                    if (originalInput.dim() >= 2) {
                        int h = originalInput.size(-2);
                        int w = originalInput.size(-1);
                        result.originalShape = {h, w};
                    }
                    else {
                        result.originalShape = {anomalyMap.size(0), anomalyMap.size(1)};
                    }
                }
                catch (const std::exception& e) {
                    _logger->error("AnomalyPredictor", "Postprocessing failed: " + std::string(e.what()));
                    // Return empty result on error
                    result.scores.push_back(0.0f);
                    result.classIds.push_back(0);
                    result.numDetections = 0;
                }

                return result;
            }

            float AnomalyPredictor::computeImageLevelScore(const torch::Tensor& anomalyMap)
            {
                // Compute maximum anomaly score in the map (most anomalous pixel)
                float maxScore = anomalyMap.max().item<float>();

                // Alternatively, could use mean or percentile
                // float meanScore = anomalyMap.mean().item<float>();

                return maxScore;
            }

            std::vector<Contour> AnomalyPredictor::extractAnomalyContours(
                const torch::Tensor& anomalyMap,
                float threshold)
            {
                std::vector<Contour> contours;

                try {
                    // Convert anomaly map to binary mask using threshold
                    torch::Tensor binaryMask = (anomalyMap > threshold).to(torch::kUInt8);

                    // Convert to OpenCV Mat for contour extraction
                    cv::Mat mask(
                        anomalyMap.size(0),
                        anomalyMap.size(1),
                        CV_8UC1,
                        binaryMask.data_ptr<uint8_t>()
                    );

                    // Find contours
                    std::vector<std::vector<cv::Point>> cvContours;
                    cv::findContours(mask, cvContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                    // Convert OpenCV contours to WheelDL Contour format
                    for (const auto& cvContour : cvContours) {
                        if (cvContour.size() < 3) {
                            continue;  // Skip tiny contours
                        }

                        Contour contour;
                        contour.points.reserve(cvContour.size() * 2);

                        for (const auto& pt : cvContour) {
                            contour.points.push_back(static_cast<float>(pt.x));
                            contour.points.push_back(static_cast<float>(pt.y));
                        }

                        contours.push_back(contour);
                    }

                    _logger->info("AnomalyPredictor", "Extracted " + std::to_string(contours.size()) + " contours");
                }
                catch (const std::exception& e) {
                    _logger->error("AnomalyPredictor", "Failed to extract contours: " + std::string(e.what()));
                }

                return contours;
            }


        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
