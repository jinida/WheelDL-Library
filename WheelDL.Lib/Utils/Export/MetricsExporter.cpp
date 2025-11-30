#include "pch.h"
#include "MetricsExporter.h"
#include "../../Config/JsonParser.h"
#include "../Logger/Logger.h"
#include <fstream>
#include <iomanip>
#include <filesystem>

namespace WheelDL {
    namespace Utils {
        namespace Export {

            bool MetricsExporter::exportToJSON(
                const std::vector<WheelDL::MetricsData>& metrics,
                const std::string& filepath,
                float bestFitness)
            {
                if (metrics.empty())
                {
                    auto logger = Logger::getInstance();
                    logger->warn("MetricsExporter", "No metrics to export to JSON");
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
                    nlohmann::json metricsJson = nlohmann::json::array();

                    for (size_t epoch = 0; epoch < metrics.size(); ++epoch)
                    {
                        const auto& metric = metrics[epoch];

                        nlohmann::json epochJson;
                        epochJson["epoch"] = epoch;
                        epochJson["loss"] = metric.loss;
                        epochJson["accuracy"] = metric.accuracy;
                        epochJson["precision"] = metric.precision;
                        epochJson["recall"] = metric.recall;
                        epochJson["f1_score"] = metric.f1Score;
                        epochJson["mAP"] = metric.mAP;
                        epochJson["fitness"] = metric.fitness;
                        epochJson["threshold"] = metric.threshold;
						epochJson["aucROC"] = metric.aucROC;

                        metricsJson.push_back(epochJson);
                    }

                    // Wrap in root object
                    nlohmann::json rootJson;
                    rootJson["training_metrics"] = metricsJson;
                    rootJson["total_epochs"] = metrics.size();
                    if (bestFitness >= 0.0f)
                    {
                        rootJson["best_fitness"] = bestFitness;
                    }

                    // Save to file
                    std::ofstream jsonFile(filepath);
                    if (!jsonFile.is_open())
                    {
                        auto logger = Logger::getInstance();
                        logger->error("MetricsExporter", "Failed to open JSON file: " + filepath);
                        return false;
                    }

                    jsonFile << std::setw(4) << rootJson << std::endl;
                    jsonFile.close();

                    auto logger = Logger::getInstance();
                    logger->info("MetricsExporter", "Metrics exported to JSON: " + filepath);
                    return true;
                }
                catch (const std::exception& e)
                {
                    auto logger = Logger::getInstance();
                    logger->error("MetricsExporter", "Failed to export JSON: " + std::string(e.what()));
                    return false;
                }
            }

            bool MetricsExporter::exportToCSV(
                const std::vector<WheelDL::MetricsData>& metrics,
                const std::string& filepath)
            {
                if (metrics.empty())
                {
                    auto logger = Logger::getInstance();
                    logger->warn("MetricsExporter", "No metrics to export to CSV");
                    return false;
                }

                try
                {
                    // Ensure directory exists
                    if (!ensureDirectoryExists(filepath))
                    {
                        return false;
                    }

                    // Open file
                    std::ofstream csvFile(filepath);
                    if (!csvFile.is_open())
                    {
                        auto logger = Logger::getInstance();
                        logger->error("MetricsExporter", "Failed to open CSV file: " + filepath);
                        return false;
                    }

                    // Write header
                    csvFile << "epoch,loss,accuracy,precision,recall,f1_score,mAP,fitness,threshold\n";

                    // Write data rows
                    for (size_t epoch = 0; epoch < metrics.size(); ++epoch)
                    {
                        const auto& metric = metrics[epoch];
                        csvFile << (epoch + 1) << ","
                               << metric.loss << ","
                               << metric.accuracy << ","
                               << metric.precision << ","
                               << metric.recall << ","
                               << metric.f1Score << ","
                               << metric.mAP << ","
                               << metric.fitness << ","
							    << metric.aucROC << ","
                               << metric.threshold << "\n";

                    }

                    csvFile.close();

                    auto logger = Logger::getInstance();
                    logger->info("MetricsExporter", "Metrics exported to CSV: " + filepath);
                    return true;
                }
                catch (const std::exception& e)
                {
                    auto logger = Logger::getInstance();
                    logger->error("MetricsExporter", "Failed to export CSV: " + std::string(e.what()));
                    return false;
                }
            }

            bool MetricsExporter::exportAll(
                const std::vector<WheelDL::MetricsData>& metrics,
                const std::string& directory,
                const std::string& jsonFilename,
                const std::string& csvFilename,
                float bestFitness)
            {
                if (metrics.empty())
                {
                    auto logger = Logger::getInstance();
                    logger->warn("MetricsExporter", "No metrics to export");
                    return false;
                }

                // Ensure directory exists
                if (!std::filesystem::exists(directory))
                {
                    try
                    {
                        std::filesystem::create_directories(directory);
                    }
                    catch (const std::exception& e)
                    {
                        auto logger = Logger::getInstance();
                        logger->error("MetricsExporter", "Failed to create directory: " + std::string(e.what()));
                        return false;
                    }
                }

                // Build full paths
                std::filesystem::path dirPath(directory);
                std::string jsonPath = (dirPath / jsonFilename).string();
                std::string csvPath = (dirPath / csvFilename).string();

                // Export both formats
                bool jsonSuccess = exportToJSON(metrics, jsonPath, bestFitness);
                bool csvSuccess = exportToCSV(metrics, csvPath);

                return jsonSuccess && csvSuccess;
            }

            bool MetricsExporter::ensureDirectoryExists(const std::string& filepath)
            {
                try
                {
                    std::filesystem::path path(filepath);
                    std::filesystem::path directory = path.parent_path();

                    if (!directory.empty() && !std::filesystem::exists(directory))
                    {
                        std::filesystem::create_directories(directory);
                    }

                    return true;
                }
                catch (const std::exception& e)
                {
                    auto logger = Logger::getInstance();
                    logger->error("MetricsExporter", "Failed to create directory: " + std::string(e.what()));
                    return false;
                }
            }

        } // namespace Export
    } // namespace Utils
} // namespace WheelDL
