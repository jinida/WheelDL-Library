#pragma once

#include <string>
#include <vector>
#include "../Common/Types.h"

namespace WheelDL {
    namespace Utils {
        namespace Export {

            /**
             * @class MetricsExporter
             * @brief Static utility class for exporting training metrics to various formats
             *
             * Provides functionality to save training metrics history to:
             * - JSON format (for programmatic access)
             * - CSV format (for data analysis tools)
             *
             * All methods are static - no need to instantiate.
             *
             * Usage:
             * @code
             * std::vector<MetricsData> epochMetrics;
             * // ... populate metrics during training ...
             *
             * MetricsExporter::exportToJSON(epochMetrics, "./results/metrics.json", bestFitness);
             * MetricsExporter::exportToCSV(epochMetrics, "./results/metrics.csv");
             * // Or export both at once:
             * MetricsExporter::exportAll(epochMetrics, "./results", bestFitness);
             * @endcode
             */
            class MetricsExporter {
            public:
                // Delete constructors to prevent instantiation
                MetricsExporter() = delete;
                ~MetricsExporter() = delete;
                MetricsExporter(const MetricsExporter&) = delete;
                MetricsExporter& operator=(const MetricsExporter&) = delete;

                /**
                 * @brief Export metrics to JSON file
                 * @param metrics Vector of epoch metrics
                 * @param filepath Path to output JSON file
                 * @param bestFitness Best fitness achieved during training (optional)
                 * @return true if successful, false otherwise
                 */
                static bool exportToJSON(
                    const std::vector<WheelDL::MetricsData>& metrics,
                    const std::string& filepath,
                    float bestFitness = -1.0f);

                /**
                 * @brief Export metrics to CSV file
                 * @param metrics Vector of epoch metrics
                 * @param filepath Path to output CSV file
                 * @return true if successful, false otherwise
                 */
                static bool exportToCSV(
                    const std::vector<WheelDL::MetricsData>& metrics,
                    const std::string& filepath);

                /**
                 * @brief Export metrics to both JSON and CSV
                 * @param metrics Vector of epoch metrics
                 * @param directory Output directory path
                 * @param jsonFilename JSON filename (default: "training_metrics.json")
                 * @param csvFilename CSV filename (default: "training_metrics.csv")
                 * @param bestFitness Best fitness achieved during training (optional)
                 * @return true if both exports successful, false otherwise
                 */
                static bool exportAll(
                    const std::vector<WheelDL::MetricsData>& metrics,
                    const std::string& directory,
                    const std::string& jsonFilename = "training_metrics.json",
                    const std::string& csvFilename = "training_metrics.csv",
                    float bestFitness = -1.0f);

            private:
                /**
                 * @brief Ensure directory exists, create if needed
                 * @param filepath File path (directory will be extracted)
                 * @return true if directory exists or was created
                 */
                static bool ensureDirectoryExists(const std::string& filepath);
            };

        } // namespace Export
    } // namespace Utils
} // namespace WheelDL
