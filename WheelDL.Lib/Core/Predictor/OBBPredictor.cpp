#include "pch.h"
#include "OBBPredictor.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            OBBPredictor::OBBPredictor(
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
                _logger->info("OBBPredictor", "Initializing OBB predictor");

                // Verify task type
                if (config->getTaskType() != TaskType::OBB) {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be OBB for OBBPredictor"
                    );
                }

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
                _profiler->start("setup_model");
                _logger->info("OBBPredictor", "Setting up OBB detection model");

                try {
                    _model = std::make_unique<Model::OBBModel>(_config);

                    _logger->info("OBBPredictor", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw WheelDL::Utils::ModelException(
                        WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup OBB detection model: " + std::string(e.what())
                    );
                }

                _profiler->stop("setup_model");
            }

            void OBBPredictor::postprocess(
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

                    torch::Tensor pred = output[0];

                    auto detections = Model::Utils::nonMaxSuppressionOBB(pred, _confThresh, _iouThresh, _maxDet);

                    if (detections.empty() || detections[0].size(0) == 0)
                    {
                        return;
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

                    // Store in internal container
                    OBBResult result;
                    result.imagePath = imagePath;
                    result.originalShape = {std::get<0>(originalShape), std::get<1>(originalShape)};
                    result.inferenceTimeMs = static_cast<float>(_profiler->getDuration("inference"));

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
                        result.classIds.push_back(static_cast<int>(clsAccessor[i]));
                    }

                    _results.push_back(std::move(result));
                }
                catch (const std::exception& e) {
                    _logger->error("OBBPredictor", "Postprocessing failed: " + std::string(e.what()));
                }
            }

            void OBBPredictor::exportResults(const std::string& outputDir)
            {
                _profiler->start("export_results");

                nlohmann::json predictionsJson = nlohmann::json::array();
                predictionsJson.get<nlohmann::json::array_t>().reserve(_results.size());
                auto classNames = _config->getClassNames();

                for (const auto& result : _results) {
                    nlohmann::json predJson;
                    predJson["image_path"] = result.imagePath;
                    predJson["inference_time_ms"] = result.inferenceTimeMs;
                    predJson["num_detections"] = result.orientedBoxes.size();

                    nlohmann::json detectionsJson = nlohmann::json::array();
                    for (size_t i = 0; i < result.orientedBoxes.size(); ++i) {
                        nlohmann::json detJson;

                        // Oriented Bounding Box
                        const auto& obb = result.orientedBoxes[i];
                        detJson["obb"] = {
                            {"cx", obb.cx},
                            {"cy", obb.cy},
                            {"width", obb.width},
                            {"height", obb.height},
                            {"angle", obb.angle}
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
                    totalDetections += result.orientedBoxes.size();
                }
                rootJson["total_detections"] = totalDetections;

                std::ofstream file(outPath);
                if (!file.is_open()) {
                    _logger->error("OBBPredictor", "Failed to create predictions.json file");
                    _profiler->stop("export_results");
                    return;
                }
                file << std::setw(4) << rootJson << std::endl;
                file.close();

                if (file.fail()) {
                    _logger->error("OBBPredictor", "Failed to write predictions.json file");
                }

                _logger->info("OBBPredictor", "Exported " +
                    std::to_string(_results.size()) + " images with " +
                    std::to_string(totalDetections) + " total OBB detections to " + outPath.string());

                _profiler->stop("export_results");
            }

            void OBBPredictor::clearResults()
            {
                _results.clear();
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
