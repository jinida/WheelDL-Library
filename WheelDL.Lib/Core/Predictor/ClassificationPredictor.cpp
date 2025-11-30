#include "pch.h"
#include "ClassificationPredictor.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            ClassificationPredictor::ClassificationPredictor(
                std::shared_ptr<Config::Configuration> config,
                const std::string& checkpointPath)
                : BasePredictor(config, checkpointPath)
                , _numClasses(config->getNumClasses())
                , _useImageNetNorm(config->getImageNetNorm())
            {
                _logger->info("ClassificationPredictor", "Initializing classification predictor");

                // Initialize normalization tensors
                if (_useImageNetNorm) {
                    _mean = torch::tensor({ 0.485f, 0.456f, 0.406f }, torch::kFloat32).view({ 1, 3, 1, 1 }).to(_device);
                    _std = torch::tensor({ 0.229f, 0.224f, 0.225f }, torch::kFloat32).view({ 1, 3, 1, 1 }).to(_device);
                }
                else
                {
                    _mean = torch::tensor({ 0.0f, 0.0f, 0.0f }, torch::kFloat32).view({ 1, 3, 1, 1 }).to(_device);
                    _std = torch::tensor({ 1.0f, 1.0f, 1.0f }, torch::kFloat32).view({ 1, 3, 1, 1 }).to(_device);
                }

                // Get model YAML path from configuration
                _modelYamlPath = config->getModelPath();
                if (_modelYamlPath.empty()) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Model YAML path is not specified in configuration"
                    );
                }

                // Verify task type
                if (config->getTaskType() != TaskType::CLASSIFICATION) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be CLASSIFICATION for ClassificationPredictor"
                    );
                }

                _logger->info("ClassificationPredictor", "Model YAML: " + _modelYamlPath);
                _logger->info("ClassificationPredictor", "Number of classes: " + std::to_string(_numClasses));

                // Setup model
                setupModel();

                // Load checkpoint if provided
                if (!checkpointPath.empty()) {
                    loadCheckpoint(checkpointPath);
                }

                _logger->info("ClassificationPredictor", "Classification predictor initialized");
            }

            void ClassificationPredictor::setupModel()
            {
                _profiler.start("setup_model");
                _logger->info("ClassificationPredictor", "Setting up classification model");

                try {
                    _model = std::make_unique<Model::ClassificationModel>(_config);
                    _logger->info("ClassificationPredictor", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup classification model: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_model");
            }

            torch::Tensor ClassificationPredictor::preprocess(const torch::Tensor& input)
            {
                try {
                    torch::Tensor preprocessed = input;

                    // Ensure input is [C, H, W] or [1, C, H, W]
                    if (preprocessed.dim() == 3) {
                        preprocessed = preprocessed.unsqueeze(0);
                    }
                    else if (preprocessed.dim() != 4) {
                        throw Utils::WheelLibException(
                            Utils::ErrorCode::PREDICTION_FAILED,
                            "Input tensor must be [C, H, W] or [1, C, H, W], got " +
                            std::to_string(preprocessed.dim()) + "D"
                        );
                    }

                    // If already float and normalized, return as-is
                    if (preprocessed.dtype() == torch::kFloat32 &&
                        preprocessed.max().item<float>() <= 1.0f) {
                        return preprocessed;
                    }

                    int targetSize = _config->getImageSize();
                    int origH = preprocessed.size(2);
                    int origW = preprocessed.size(3);

                    // Resize to target size
                    if (origH != targetSize || origW != targetSize) {
                        preprocessed = torch::nn::functional::interpolate(
                            preprocessed.to(torch::kFloat32),
                            torch::nn::functional::InterpolateFuncOptions()
                                .size(std::vector<int64_t>{targetSize, targetSize})
                                .mode(torch::kBilinear)
                                .align_corners(false)
                        );
                    }
                    else {
                        preprocessed = preprocessed.to(torch::kFloat32);
                    }

                    // Normalize to [0, 1] if input is [0, 255]
                    if (preprocessed.max().item<float>() > 1.0f) {
                        preprocessed = preprocessed.mul(1.0f / 255.0f);
                    }

                    // Apply normalization
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

            PredictionResult ClassificationPredictor::postprocess(
                const std::vector<torch::Tensor>& output,
                const std::tuple<int, int>& originalShape)
            {
                PredictionResult result;

                try 
                {
                    // output[0]: logits [1, num_classes]
                    const torch::Tensor& logits = output[0];

                    // Apply softmax to get probabilities
                    auto probabilities = torch::softmax(logits, 1);  // [1, num_classes]

                    // Get top-1 prediction (argmax)
                    auto maxResult = probabilities.max(1);
                    auto maxScore = std::get<0>(maxResult).squeeze();  // scalar
                    auto maxIndex = std::get<1>(maxResult).squeeze();  // scalar

                    // Move to CPU and extract values
                    float confidence = maxScore.cpu().item<float>();
                    int64_t classId = maxIndex.cpu().item<int64_t>();

                    result.scores.push_back(confidence);
                    result.classIds.push_back(static_cast<unsigned int>(classId));

                    // Store original shape
                    result.originalShape = {
                        std::get<0>(originalShape),
                        std::get<1>(originalShape)
                    };
                }
                catch (const std::exception& e) {
                    _logger->error("ClassificationPredictor", "Postprocessing failed: " + std::string(e.what()));
                    result.scores.push_back(0.0f);
                    result.classIds.push_back(0);
                }

                return result;
            }

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
