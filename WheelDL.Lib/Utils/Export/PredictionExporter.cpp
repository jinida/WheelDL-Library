#include "pch.h"
#include "PredictionExporter.h"
#include "../../Config/JsonParser.h"
#include "../Logger/Logger.h"
#include <fstream>
#include <iomanip>
#include <filesystem>

namespace WheelDL {
    namespace Utils {
        namespace Export {

            bool PredictionExporter::exportToJSON(
                const std::vector<WheelDL::PredictionResult>& predictions,
                const std::vector<std::string>& imagePaths,
                const std::string& filepath)
            {
                if (predictions.empty())
                {
                    auto logger = Logger::getInstance();
                    logger->warn("PredictionExporter", "No predictions to export");
                    return false;
                }

                if (predictions.size() != imagePaths.size())
                {
                    auto logger = Logger::getInstance();
                    logger->error("PredictionExporter", "Predictions and image paths size mismatch");
                    return false;
                }

                try
                {
                    // Ensure directory exists
                    if (!ensureDirectoryExists(filepath))
                    {
                        return false;
                    }

                    // Build JSON array
                    nlohmann::json predictionsJson = nlohmann::json::array();

                    for (size_t i = 0; i < predictions.size(); ++i)
                    {
                        const auto& pred = predictions[i];
                        const auto& imgPath = imagePaths[i];

                        nlohmann::json predJson;
                        predJson["image_path"] = imgPath;
                        predJson["num_detections"] = pred.numDetections;
                        predJson["inference_time_ms"] = pred.inferenceTime;

                        // Original shape
                        predJson["original_shape"] = {
                            {"height", pred.originalShape.first},
                            {"width", pred.originalShape.second}
                        };

                        // Bounding boxes
                        if (!pred.boxes.empty())
                        {
                            nlohmann::json boxesJson = nlohmann::json::array();
                            for (const auto& box : pred.boxes)
                            {
                                boxesJson.push_back({
                                    {"x1", box.x1},
                                    {"y1", box.y1},
                                    {"x2", box.x2},
                                    {"y2", box.y2}
                                });
                            }
                            predJson["boxes"] = boxesJson;
                        }

                        // Scores
                        if (!pred.scores.empty())
                        {
                            predJson["scores"] = pred.scores;
                        }

                        // Class IDs
                        if (!pred.classIds.empty())
                        {
                            predJson["class_ids"] = pred.classIds;
                        }

                        // Contours (for segmentation)
                        if (!pred.contours.empty())
                        {
                            nlohmann::json contoursJson = nlohmann::json::array();
                            for (const auto& contour : pred.contours)
                            {
                                contoursJson.push_back(contour.points);
                            }
                            predJson["contours"] = contoursJson;
                        }

                        predictionsJson.push_back(predJson);
                    }

                    // Wrap in root object
                    nlohmann::json rootJson;
                    rootJson["predictions"] = predictionsJson;
                    rootJson["total_predictions"] = predictions.size();

                    // Save to file
                    std::ofstream jsonFile(filepath);
                    if (!jsonFile.is_open())
                    {
                        auto logger = Logger::getInstance();
                        logger->error("PredictionExporter", "Failed to open JSON file: " + filepath);
                        return false;
                    }

                    jsonFile << std::setw(4) << rootJson << std::endl;
                    jsonFile.close();

                    auto logger = Logger::getInstance();
                    logger->info("PredictionExporter", "Predictions exported to JSON: " + filepath);
                    return true;
                }
                catch (const std::exception& e)
                {
                    auto logger = Logger::getInstance();
                    logger->error("PredictionExporter", "Failed to export predictions: " + std::string(e.what()));
                    return false;
                }
            }
        } // namespace Export
    } // namespace Utils
} // namespace WheelDL
