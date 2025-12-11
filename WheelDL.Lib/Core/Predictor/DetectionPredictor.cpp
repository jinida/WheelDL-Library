#include "pch.h"
#include "DetectionPredictor.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            DetectionPredictor::DetectionPredictor(
                std::shared_ptr<Config::Configuration> config,
                const std::string& checkpointPath,
                WheelDL::Utils::Logger* logger,
                WheelDL::Utils::PerformanceProfiler* profiler,
                std::atomic<bool>* stopFlag)
                : BasePredictor(config, checkpointPath, logger, profiler, stopFlag)
                , _numClasses(config->getNumClasses())
                , _confThresh(0.25f)
                , _iouThresh(config->getIoU())
                , _maxDet(config->getMaxDet())
            {
                _logger->info("DetectionPredictor", "Initializing detection predictor");

                // Verify task type
                if (config->getTaskType() != TaskType::DETECTION) {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be DETECTION for DetectionPredictor"
                    );
                }

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
                _profiler->start("setup_model");
                _logger->info("DetectionPredictor", "Setting up detection model");

                try {
                    _model = std::make_unique<Model::DetectionModel>(_config);

                    _logger->info("DetectionPredictor", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw WheelDL::Utils::ModelException(
                        WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup detection model: " + std::string(e.what())
                    );
                }

                _profiler->stop("setup_model");
            }

            void DetectionPredictor::postprocess(
                const std::vector<torch::Tensor>& output,
                const std::tuple<int, int>& originalShape,
                const std::string& imagePath)
            {
                try
                {
                    if (output.empty())
                    {
                        return;
                    }

                    // output[0]: [batch, num_anchors, 4 + num_classes]
                    torch::Tensor pred = output[0];

                    // Use shared NMS from Model::Utils
                    auto detections = Model::Utils::nonMaxSuppression(pred, _confThresh, _iouThresh, _maxDet);

                    if (detections.empty() || detections[0].size(0) == 0)
                    {
                        return;
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

                    // Store in internal container
                    DetectionResult result;
                    result.imagePath = imagePath;
                    result.originalShape = {std::get<0>(originalShape), std::get<1>(originalShape)};
                    result.inferenceTimeMs = static_cast<float>(_profiler->getDuration("inference"));

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
                        result.classIds.push_back(static_cast<int>(clsAccessor[i]));
                    }

                    _results.push_back(std::move(result));
                }
                catch (const std::exception& e) {
                    _logger->error("DetectionPredictor", "Postprocessing failed: " + std::string(e.what()));
                }
            }

            void DetectionPredictor::exportResults(const std::string& outputDir)
            {
                _profiler->start("export_results");

                nlohmann::json predictionsJson = nlohmann::json::array();
                predictionsJson.get<nlohmann::json::array_t>().reserve(_results.size());
                auto classNames = _config->getClassNames();

                for (const auto& result : _results) {
                    nlohmann::json predJson;
                    predJson["image_path"] = result.imagePath;
                    predJson["inference_time_ms"] = result.inferenceTimeMs;
                    predJson["num_detections"] = result.boxes.size();

                    nlohmann::json detectionsJson = nlohmann::json::array();
                    for (size_t i = 0; i < result.boxes.size(); ++i) {
                        nlohmann::json detJson;

                        // Bounding box (nested object format)
                        const auto& box = result.boxes[i];
                        detJson["bbox"] = {
                            {"x1", box.x1},
                            {"y1", box.y1},
                            {"x2", box.x2},
                            {"y2", box.y2}
                        };

                        // Class info
                        int classId = result.classIds[i];
                        detJson["class_id"] = classId;

                        auto it = classNames.find(classId);
                        if (it != classNames.end()) {
                            detJson["class_name"] = it->second;
                        }

                        // Confidence
                        detJson["confidence"] = result.scores[i];

                        detectionsJson.push_back(detJson);
                    }

                    predJson["detections"] = detectionsJson;
                    predictionsJson.push_back(predJson);
                }

                // Write to file
                std::filesystem::path outPath = std::filesystem::path(outputDir) / "predictions.json";
                std::filesystem::create_directories(outputDir);

                nlohmann::json rootJson;
                rootJson["predictions"] = predictionsJson;
                rootJson["total_images"] = _results.size();

                // Calculate total detections
                size_t totalDetections = 0;
                for (const auto& result : _results) {
                    totalDetections += result.boxes.size();
                }
                rootJson["total_detections"] = totalDetections;

                std::ofstream file(outPath);
                if (!file.is_open()) {
                    _logger->error("DetectionPredictor", "Failed to create predictions.json file");
                    _profiler->stop("export_results");
                    return;
                }
                file << std::setw(4) << rootJson << std::endl;
                file.close();

                if (file.fail()) {
                    _logger->error("DetectionPredictor", "Failed to write predictions.json file");
                }

                _logger->info("DetectionPredictor", "Exported " +
                    std::to_string(_results.size()) + " images with " +
                    std::to_string(totalDetections) + " total detections to " + outPath.string());

                _profiler->stop("export_results");
            }

            void DetectionPredictor::clearResults()
            {
                _results.clear();
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
