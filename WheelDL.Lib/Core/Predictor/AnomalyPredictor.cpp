#include "pch.h"
#include "AnomalyPredictor.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            AnomalyPredictor::AnomalyPredictor(
                std::shared_ptr<Config::Configuration> config,
                const std::string& checkpointPath,
                WheelDL::Utils::Logger* logger,
                WheelDL::Utils::PerformanceProfiler* profiler,
                std::atomic<bool>* stopFlag)
                : BasePredictor(config, checkpointPath, logger, profiler, stopFlag)
            {
                _logger->info("AnomalyPredictor", "Initializing anomaly detection predictor");

                // Verify task type
                if (config->getTaskType() != TaskType::ANOMALY)
                {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be ANOMALY for AnomalyPredictor"
                    );
                }

                // Setup model
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
                _profiler->start("setup_model");
                _logger->info("AnomalyPredictor", "Setting up anomaly detection model");

                try {
                    _model = std::make_unique<Model::AnomalyModel>(_config);
                    _logger->info("AnomalyPredictor", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw WheelDL::Utils::ModelException(
                        WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup anomaly model: " + std::string(e.what())
                    );
                }

                _profiler->stop("setup_model");
            }

            void AnomalyPredictor::postprocess(
                const std::vector<torch::Tensor>& output,
                const std::tuple<int, int>& originalShape,
                const std::string& imagePath)
            {
                try
                {
                    if (output.size() < 2) {
                        _logger->error("AnomalyPredictor", "Expected 2 output tensors (score, map)");
                        return;
                    }

                    const torch::Tensor& anomalyScore = output[0];
                    const torch::Tensor& anomalyMap = output[1];

                    float imageLevelScore = anomalyScore.item<float>();
                    int classId = (imageLevelScore > _threshold) ? 1 : 0;

                    int origH = std::get<0>(originalShape);
                    int origW = std::get<1>(originalShape);

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
                    cv::Mat colorMap;
                    cv::applyColorMap(grayMap, colorMap, cv::COLORMAP_JET);

                    // Store in internal container
                    AnomalyResult result;
                    result.imagePath = imagePath;
                    result.anomalyScore = imageLevelScore;
                    result.classId = classId;
                    result.anomalyMap = colorMap.clone();
                    result.originalShape = {origH, origW};
                    result.inferenceTimeMs = static_cast<float>(_profiler->getDuration("inference"));

                    _results.push_back(std::move(result));
                }
                catch (const std::exception& e)
                {
                    _logger->error("AnomalyPredictor", "Postprocessing failed: " + std::string(e.what()));
                }
            }

            void AnomalyPredictor::exportResults(const std::string& outputDir)
            {
                namespace fs = std::filesystem;

                _profiler->start("export_results");

                fs::path resultsPath = fs::path(outputDir);
                fs::path anomalyMapsDir = resultsPath / "anomaly_maps";
                fs::create_directories(resultsPath);
                fs::create_directories(anomalyMapsDir);

                nlohmann::json predictionsJson = nlohmann::json::array();
                predictionsJson.get<nlohmann::json::array_t>().reserve(_results.size());

                for (const auto& result : _results) {
                    nlohmann::json predJson;
                    predJson["image_path"] = result.imagePath;
                    predJson["inference_time_ms"] = result.inferenceTimeMs;
                    predJson["anomaly_score"] = result.anomalyScore;
                    predJson["class_id"] = result.classId;
                    predJson["label"] = (result.classId == 0) ? "normal" : "anomaly";

                    if (!result.anomalyMap.empty()) {
                        std::hash<std::string> hasher;
                        size_t hashValue = hasher(result.imagePath);
                        std::stringstream ss;
                        ss << std::hex << hashValue;

                        std::string filename = ss.str() + ".jpg";
                        fs::path mapPath = anomalyMapsDir / filename;

                        std::vector<uchar> buffer;
                        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 80 };

                        bool encoded = cv::imencode(".jpg", result.anomalyMap, buffer, params);

                        if (encoded) {
                            std::ofstream outFile(mapPath, std::ios::binary);
                            if (outFile.is_open()) {
                                outFile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
                                outFile.close();

                                predJson["anomaly_map"] = "./anomaly_maps/" + filename;
                            }
                            else {
                                _logger->warn("AnomalyPredictor", "Failed to save anomaly map to disk for: " + result.imagePath);
                            }
                        }
                        else {
                            _logger->warn("AnomalyPredictor", "Failed to encode anomaly map for: " + result.imagePath);
                        }
                    }

                    predictionsJson.push_back(predJson);
                }

                // Write JSON to file
                fs::path jsonPath = resultsPath / "predictions.json";
                std::ofstream jsonFile(jsonPath);
                if (!jsonFile.is_open()) {
                    _logger->error("AnomalyPredictor", "Failed to create predictions.json file");
                    _profiler->stop("export_results");
                    return;
                }

                nlohmann::json rootJson;
                rootJson["predictions"] = predictionsJson;
                rootJson["total_predictions"] = _results.size();
                jsonFile << std::setw(4) << rootJson << std::endl;
                jsonFile.close();

                _logger->info("AnomalyPredictor", "Exported " +
                    std::to_string(_results.size()) + " prediction results to " + jsonPath.string());

                _profiler->stop("export_results");
            }

            void AnomalyPredictor::clearResults()
            {
                _results.clear();
            }

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
