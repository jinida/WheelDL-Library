#include "pch.h"
#include "DetectionPredictor.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            DetectionPredictor::DetectionPredictor(
                std::shared_ptr<Config::Configuration> config,
                const std::string& checkpointPath)
                : BasePredictor(config, checkpointPath)
                , _numClasses(config->getNumClasses())
                , _confThresh(0.25f)
                , _iouThresh(config->getIoU())
                , _maxDet(config->getMaxDet())
            {
                _logger->info("DetectionPredictor", "Initializing detection predictor");

                _mean = torch::tensor({ 0.0f, 0.0f, 0.0f }, torch::kFloat32).view({ 1, 3, 1, 1 }).to(_device);
                _std = torch::tensor({ 1.0f, 1.0f, 1.0f }, torch::kFloat32).view({ 1, 3, 1, 1 }).to(_device);

                // Get model YAML path from configuration
                _modelYamlPath = config->getModelPath();
                if (_modelYamlPath.empty()) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Model YAML path is not specified in configuration"
                    );
                }

                // Verify task type
                if (config->getTaskType() != TaskType::DETECTION) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be DETECTION for DetectionPredictor"
                    );
                }

                _logger->info("DetectionPredictor", "Model YAML: " + _modelYamlPath);
                _logger->info("DetectionPredictor", "Number of classes: " + std::to_string(_numClasses));
                _logger->info("DetectionPredictor", "Confidence threshold: " + std::to_string(_confThresh));
                _logger->info("DetectionPredictor", "IoU threshold: " + std::to_string(_iouThresh));

                // Setup model
                setupModel();

                // Load checkpoint if provided
                if (!checkpointPath.empty()) {
                    loadCheckpoint(checkpointPath);
                }

                _logger->info("DetectionPredictor", "Detection predictor initialized");
            }

            void DetectionPredictor::setupModel()
            {
                _profiler.start("setup_model");
                _logger->info("DetectionPredictor", "Setting up detection model");

                try {
                    _model = std::make_unique<Model::DetectionModel>(_config);
                    
                    _logger->info("DetectionPredictor", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup detection model: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_model");
            }

            torch::Tensor DetectionPredictor::preprocess(const torch::Tensor& input)
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
                    int origH = static_cast<int>(preprocessed.size(2));
                    int origW = static_cast<int>(preprocessed.size(3));

                    // LetterBox resize (maintain aspect ratio)
                    float scale = std::min(
                        static_cast<float>(targetSize) / origH,
                        static_cast<float>(targetSize) / origW
                    );

                    int newH = static_cast<int>(origH * scale);
                    int newW = static_cast<int>(origW * scale);

                    // Resize
                    preprocessed = torch::nn::functional::interpolate(
                        preprocessed.to(torch::kFloat32),
                        torch::nn::functional::InterpolateFuncOptions()
                            .size(std::vector<int64_t>{newH, newW})
                            .mode(torch::kBilinear)
                            .align_corners(false)
                    );

                    // Pad to target size (center padding)
                    int padH = targetSize - newH;
                    int padW = targetSize - newW;
                    int padTop = padH / 2;
                    int padBottom = padH - padTop;
                    int padLeft = padW / 2;
                    int padRight = padW - padLeft;

                    if (padH > 0 || padW > 0) {
                        preprocessed = torch::nn::functional::pad(
                            preprocessed,
                            torch::nn::functional::PadFuncOptions({padLeft, padRight, padTop, padBottom})
                                .mode(torch::kConstant)
                                .value(0.0f)  // Gray padding
                        );
                    }

                    // Normalize to [0, 1] if input is [0, 255]
                    if (preprocessed.max().item<float>() > 1.0f) {
                        preprocessed = preprocessed.mul(1.0f / 255.0f);
                    }

                    // Apply normalization (for YOLO, typically just [0,1])
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

            PredictionResult DetectionPredictor::postprocess(
                const std::vector<torch::Tensor>& output,
                const std::tuple<int, int>& originalShape)
            {
                PredictionResult result;
                result.originalShape = {std::get<0>(originalShape), std::get<1>(originalShape)};

                try
                {
                    if (output.empty())
                    {
                        result.numDetections = 0;
                        return result;
                    }

                    // output[0]: [batch, num_anchors, 4 + num_classes]
                    torch::Tensor pred = output[0];

                    // Use shared NMS from Model::Utils
                    auto detections = Model::Utils::nonMaxSuppression(pred, _confThresh, _iouThresh, _maxDet);
                    
                    if (detections.empty()) 
                    {
                        result.numDetections = 0;
                        return result;
					}

                    if (detections[0].size(0) == 0)
                    {
                        result.numDetections = 0;
                        return result;
                    }

                    // Scale boxes to original image size
                    auto boxes = detections[0].slice(1, 0, 4);
                    boxes = scaleBoxes(boxes, _config->getImageSize(), originalShape);

                    // Extract results
                    auto conf = detections[0].slice(1, 4, 5).squeeze(1);
                    auto cls = detections[0].slice(1, 5, 6).squeeze(1);

                    // Move to CPU and convert
                    auto boxesCPU = boxes.cpu().contiguous();
                    auto confCPU = conf.cpu().contiguous();
                    auto clsCPU = cls.cpu().to(torch::kLong).contiguous();

                    auto boxAccessor = boxesCPU.accessor<float, 2>();
                    auto confAccessor = confCPU.accessor<float, 1>();
                    auto clsAccessor = clsCPU.accessor<int64_t, 1>();

                    int numDet = static_cast<int>(boxesCPU.size(0));
                    result.boxes.reserve(numDet);
                    result.scores.reserve(numDet);
                    result.classIds.reserve(numDet);

                    for (int i = 0; i < numDet; ++i) {
                        BBox box;
                        box.x1 = boxAccessor[i][0];
                        box.y1 = boxAccessor[i][1];
                        box.x2 = boxAccessor[i][2];
                        box.y2 = boxAccessor[i][3];

                        result.boxes.push_back(box);
                        result.scores.push_back(confAccessor[i]);
                        result.classIds.push_back(static_cast<unsigned int>(clsAccessor[i]));
                    }

                    result.numDetections = numDet;
                }
                catch (const std::exception& e) {
                    _logger->error("DetectionPredictor", "Postprocessing failed: " + std::string(e.what()));
                    result.numDetections = 0;
                }

                return result;
            }

            torch::Tensor DetectionPredictor::scaleBoxes(
                const torch::Tensor& boxes,
                int inputSize,
                const std::tuple<int, int>& originalShape)
            {
                int origH = std::get<0>(originalShape);
                int origW = std::get<1>(originalShape);

                // Calculate scale and padding used during preprocessing
                float scale = std::min(
                    static_cast<float>(inputSize) / origH,
                    static_cast<float>(inputSize) / origW
                );

                int newH = static_cast<int>(origH * scale);
                int newW = static_cast<int>(origW * scale);

                int padH = inputSize - newH;
                int padW = inputSize - newW;
                float padTop = padH / 2.0f;
                float padLeft = padW / 2.0f;

                // Remove padding offset and scale back
                auto scaledBoxes = boxes.clone();

                // x1, x2 adjustment
                scaledBoxes.select(1, 0) = (boxes.select(1, 0) - padLeft) / scale;
                scaledBoxes.select(1, 2) = (boxes.select(1, 2) - padLeft) / scale;

                // y1, y2 adjustment
                scaledBoxes.select(1, 1) = (boxes.select(1, 1) - padTop) / scale;
                scaledBoxes.select(1, 3) = (boxes.select(1, 3) - padTop) / scale;

                // Clip to image bounds
                scaledBoxes.select(1, 0).clamp_(0, origW);
                scaledBoxes.select(1, 1).clamp_(0, origH);
                scaledBoxes.select(1, 2).clamp_(0, origW);
                scaledBoxes.select(1, 3).clamp_(0, origH);

                return scaledBoxes;
            }

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
