#include "pch.h"
#include "OBBPredictor.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            OBBPredictor::OBBPredictor(
                std::shared_ptr<Config::Configuration> config,
                const std::string& checkpointPath)
                : BasePredictor(config, checkpointPath)
                , _numClasses(config->getNumClasses())
                , _confThresh(0.25f)
                , _iouThresh(config->getIoU())
                , _maxDet(config->getMaxDet())
            {
                _logger->info("OBBPredictor", "Initializing OBB predictor");

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
                if (config->getTaskType() != TaskType::OBB) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be OBB for OBBPredictor"
                    );
                }

                _logger->info("OBBPredictor", "Model YAML: " + _modelYamlPath);
                _logger->info("OBBPredictor", "Number of classes: " + std::to_string(_numClasses));
                _logger->info("OBBPredictor", "Confidence threshold: " + std::to_string(_confThresh));
                _logger->info("OBBPredictor", "IoU threshold: " + std::to_string(_iouThresh));

                // Setup model
                setupModel();

                // Load checkpoint if provided
                if (!checkpointPath.empty()) {
                    loadCheckpoint(checkpointPath);
                }

                _logger->info("OBBPredictor", "OBB predictor initialized");
            }

            void OBBPredictor::setupModel()
            {
                _profiler.start("setup_model");
                _logger->info("OBBPredictor", "Setting up OBB detection model");

                try {
                    _model = std::make_unique<Model::OBBModel>(_config);

                    _logger->info("OBBPredictor", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup OBB detection model: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_model");
            }

            torch::Tensor OBBPredictor::preprocess(const torch::Tensor& input)
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
                    if (preprocessed.dtype() == torch::kFloat32 && preprocessed.max().item<float>() <= 1.0f) 
                    {
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

            PredictionResult OBBPredictor::postprocess(
                const std::vector<torch::Tensor>& output,
                const std::tuple<int, int>& originalShape)
            {
                PredictionResult result;
                result.originalShape = {std::get<0>(originalShape), std::get<1>(originalShape)};
                result.anomalyMap = cv::Mat();
                try
                {
                    if (output.empty())
                    {
                        result.numDetections = 0;
                        return result;
                    }

                    torch::Tensor pred = output[0];

                    auto detections = Model::Utils::nonMaxSuppressionOBB(pred, _confThresh, _iouThresh, _maxDet);

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

                    // Scale OBBs to original image size
                    auto obbs = detections[0].slice(1, 0, 5);  // [cx, cy, w, h, angle]
                    obbs = scaleOBBs(obbs, _config->getImageSize(), originalShape);

                    // Extract results
                    auto conf = detections[0].slice(1, 5, 6).squeeze(1);
                    auto cls = detections[0].slice(1, 6, 7).squeeze(1);

                    // Move to CPU and convert
                    auto obbsCPU = obbs.cpu().contiguous();
                    auto confCPU = conf.cpu().contiguous();
                    auto clsCPU = cls.cpu().to(torch::kLong).contiguous();

                    auto obbAccessor = obbsCPU.accessor<float, 2>();
                    auto confAccessor = confCPU.accessor<float, 1>();
                    auto clsAccessor = clsCPU.accessor<int64_t, 1>();

                    int numDet = static_cast<int>(obbsCPU.size(0));
                    result.orientedBoxes.reserve(numDet);
                    result.scores.reserve(numDet);
                    result.classIds.reserve(numDet);

                    for (int i = 0; i < numDet; ++i) {
                        OBB obb;
                        obb.cx = obbAccessor[i][0];
                        obb.cy = obbAccessor[i][1];
                        obb.width = obbAccessor[i][2];
                        obb.height = obbAccessor[i][3];
                        obb.angle = obbAccessor[i][4];

                        result.orientedBoxes.push_back(obb);
                        result.scores.push_back(confAccessor[i]);
                        result.classIds.push_back(static_cast<unsigned int>(clsAccessor[i]));
                    }

                    result.numDetections = numDet;
                }
                catch (const std::exception& e) {
                    _logger->error("OBBPredictor", "Postprocessing failed: " + std::string(e.what()));
                    result.numDetections = 0;
                }

                return result;
            }

            torch::Tensor OBBPredictor::scaleOBBs(
                const torch::Tensor& obbs,
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

                // Convert OBB (cx, cy, w, h, angle) to 4 corner points
                // corners: [N, 4, 2] where 4 corners with (x, y) each
                auto corners = Model::Utils::xywhr2xyxyxyxy(obbs);  // [N, 4, 2]

                // Scale corner points: remove padding and scale back
                // x coordinates
                corners.select(2, 0) = (corners.select(2, 0) - padLeft) / scale;
                // y coordinates
                corners.select(2, 1) = (corners.select(2, 1) - padTop) / scale;

                // Convert back to OBB format (cx, cy, w, h, angle)
                // Calculate center from corners
                auto cx = corners.select(2, 0).mean(1);  // [N]
                auto cy = corners.select(2, 1).mean(1);  // [N]

                // Calculate width and height from corners
                // Width: distance between corner 0 and corner 1
                auto dx01 = corners.select(1, 1).select(1, 0) - corners.select(1, 0).select(1, 0);
                auto dy01 = corners.select(1, 1).select(1, 1) - corners.select(1, 0).select(1, 1);
                auto w = torch::sqrt(dx01 * dx01 + dy01 * dy01);

                // Height: distance between corner 1 and corner 2
                auto dx12 = corners.select(1, 2).select(1, 0) - corners.select(1, 1).select(1, 0);
                auto dy12 = corners.select(1, 2).select(1, 1) - corners.select(1, 1).select(1, 1);
                auto h = torch::sqrt(dx12 * dx12 + dy12 * dy12);

                // Angle stays the same (rotation is preserved after uniform scaling)
                auto angle = obbs.select(1, 4);

                // Stack back to [N, 5]
                auto scaledOBBs = torch::stack({cx, cy, w, h, angle}, 1);

                return scaledOBBs;
            }

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
