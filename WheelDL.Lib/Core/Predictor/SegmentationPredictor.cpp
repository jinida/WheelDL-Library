#include "pch.h"
#include "SegmentationPredictor.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            SegmentationPredictor::SegmentationPredictor(
                std::shared_ptr<Config::Configuration> config,
                const std::string& checkpointPath)
                : BasePredictor(config, checkpointPath)
                , _numClasses(config->getNumClasses())
                , _useImageNetNorm(config->getImageNetNorm())
            {
                _logger->info("SegmentationPredictor", "Initializing segmentation predictor");

                // Initialize normalization tensors
                if (_useImageNetNorm) 
                {
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
                if (config->getTaskType() != TaskType::SEGMENTATION) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be SEGMENTATION for SegmentationPredictor"
                    );
                }

                _logger->info("SegmentationPredictor", "Model YAML: " + _modelYamlPath);
                _logger->info("SegmentationPredictor", "Number of classes: " + std::to_string(_numClasses));

                // Setup model
                setupModel();

                // Load checkpoint if provided
                if (!checkpointPath.empty()) {
                    loadCheckpoint(checkpointPath);
                }

                _minContourArea = 5;

                _logger->info("SegmentationPredictor", "Confidence threshold: " + std::to_string(_threshold));
                _logger->info("SegmentationPredictor", "Segmentation predictor initialized");
            }

            void SegmentationPredictor::setupModel()
            {
                _profiler.start("setup_model");
                _logger->info("SegmentationPredictor", "Setting up segmentation model");

                try {
                    _model = std::make_unique<Model::SegmentationModel>(_config);
                    _logger->info("SegmentationPredictor", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup segmentation model: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_model");
            }

            torch::Tensor SegmentationPredictor::preprocess(const torch::Tensor& input)
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
                            "Input tensor must be [C, H, W] or [1, C, H, W], got " +
                            std::to_string(preprocessed.dim()) + "D"
                        );
                    }

                    // Convert to float32 if needed
                    if (preprocessed.dtype() != torch::kFloat32) {
                        preprocessed = preprocessed.to(torch::kFloat32);
                    }

                    int targetSize = _config->getImageSize();
                    int origH = static_cast<int>(preprocessed.size(2));
                    int origW = static_cast<int>(preprocessed.size(3));

                    // Check if resize is needed
                    bool needsResize = (origH != targetSize || origW != targetSize);

                    if (needsResize) {
                        // LetterBox resize (maintain aspect ratio)
                        float scale = std::min(
                            static_cast<float>(targetSize) / origH,
                            static_cast<float>(targetSize) / origW
                        );

                        int newH = static_cast<int>(origH * scale);
                        int newW = static_cast<int>(origW * scale);

                        // Resize
                        preprocessed = torch::nn::functional::interpolate(
                            preprocessed,
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
                                    .value(0.0f)
                            );
                        }
                    }

                    // Normalize to [0, 1] if input is [0, 255]
                    if (preprocessed.max().item<float>() > 1.0f) {
                        preprocessed = preprocessed.div(255.0f);
                    }

                    // Apply ImageNet normalization if enabled
                    if (_useImageNetNorm) {
                        preprocessed = preprocessed.sub(_mean).div(_std);
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

            PredictionResult SegmentationPredictor::postprocess(
                const std::vector<torch::Tensor>& output,
                const std::tuple<int, int>& originalShape)
            {
                PredictionResult result;
                result.originalShape = { std::get<0>(originalShape), std::get<1>(originalShape) };

                if (output.empty()) {
                    result.numDetections = 0;
                    return result;
                }

                try
                {
                    int origH = std::get<0>(originalShape);
                    int origW = std::get<1>(originalShape);
                    int inputSize = _config->getImageSize();

                    float epsilon = 1e-6f;
                    float clampedThresh = std::max(epsilon, std::min(_threshold, 1.0f - epsilon));
                    float logitThreshold = std::log(clampedThresh / (1.0f - clampedThresh));

                    auto rawOutput = output[0].squeeze(0).cpu().contiguous();
                    int maskH = rawOutput.size(1);
                    int maskW = rawOutput.size(2);
                    int64_t maskArea = static_cast<int64_t>(maskH) * maskW;

                    float scale = std::min(
                        static_cast<float>(inputSize) / origH,
                        static_cast<float>(inputSize) / origW
                    );

                    float padX = (inputSize - origW * scale) / 2.0f;
                    float padY = (inputSize - origH * scale) / 2.0f;

                    float maskScaleX = maskW / static_cast<float>(inputSize);
                    float maskScaleY = maskH / static_cast<float>(inputSize);

                    const float* logitsPtr = rawOutput.data_ptr<float>();

                    cv::Mat maskMat;
                    std::vector<std::vector<cv::Point>> cvContours;

                    for (int c = 0; c < _numClasses; ++c)
                    {
                        const float* classLogits = logitsPtr + c * maskArea;

                        cv::Mat logitMat(maskH, maskW, CV_32FC1, const_cast<float*>(classLogits));

                        cv::threshold(logitMat, maskMat, logitThreshold, 255, cv::THRESH_BINARY);
                        maskMat.convertTo(maskMat, CV_8UC1);

                        cv::findContours(maskMat, cvContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                        for (const auto& originalContour : cvContours)
                        {
                            if (originalContour.size() < 3) continue;

                            double area = cv::contourArea(originalContour);

                            double scaledArea = area / (scale * maskScaleX * scale * maskScaleY);
                            if (scaledArea < _minContourArea) continue;

                            std::vector<cv::Point> cvContour = originalContour;
                            const int MAX_POINTS = 500;

                            if (cvContour.size() > MAX_POINTS) {
                                double perimeter = cv::arcLength(originalContour, true);
                                double epsilon = perimeter * 0.005;

                                cv::approxPolyDP(originalContour, cvContour, epsilon, true);

                                if (cvContour.size() > MAX_POINTS) {
                                    int step = cvContour.size() / MAX_POINTS + 1;

                                    std::vector<cv::Point> subsampledContour;
                                    for (size_t i = 0; i < cvContour.size(); i += step) {
                                        subsampledContour.push_back(cvContour[i]);
                                    }
                                    cvContour = subsampledContour;
                                }
                            }

                            cv::Rect bbox = cv::boundingRect(cvContour);
                            bbox &= cv::Rect(0, 0, maskW, maskH);

                            Contour contour;
                            contour.points.reserve(cvContour.size() * 2);

                            for (const auto& pt : cvContour) {
                                float x = (pt.x / maskScaleX - padX) / scale;
                                float y = (pt.y / maskScaleY - padY) / scale;

                                x = std::max(0.0f, std::min(x, static_cast<float>(origW)));
                                y = std::max(0.0f, std::min(y, static_cast<float>(origH)));

                                contour.points.push_back(x);
                                contour.points.push_back(y);
                            }

                            float sumProb = 0.0f;
                            int count = 0;

                            for (int y = bbox.y; y < bbox.y + bbox.height; ++y) {
                                const float* row = classLogits + y * maskW;
                                for (int x = bbox.x; x < bbox.x + bbox.width; ++x) {
                                    float lg = row[x];
                                    if (lg > logitThreshold) {
                                        float prob = 1.0f / (1.0f + std::exp(-lg));
                                        sumProb += prob;
                                        ++count;
                                    }
                                }
                            }

                            result.contours.push_back(std::move(contour));
                            result.classIds.push_back(static_cast<unsigned int>(c));
                            result.scores.push_back(count > 0 ? sumProb / count : 0.0f);
                        }
                    }

                    result.numDetections = static_cast<int>(result.contours.size());
                }
                catch (const std::exception& e) {
                    _logger->error("SegmentationPredictor", "Postprocessing failed: " + std::string(e.what()));
                    result.numDetections = 0;
                }

                return result;
            }
        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
