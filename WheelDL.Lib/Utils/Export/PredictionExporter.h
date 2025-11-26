#pragma once

#include <string>
#include <vector>
#include "../Common/Types.h"

namespace WheelDL {
    namespace Utils {
        namespace Export {

            /**
             * @class PredictionExporter
             * @brief Static utility class for exporting prediction results to JSON
             *
             * Provides functionality to save prediction/inference results to JSON format.
             * Handles multiple prediction results.
             *
             * Note: Anomaly maps should be saved separately using AnomalyMapExporter.
             *
             * All methods are static - no need to instantiate.
             *
             * Usage:
             * @code
             * std::vector<PredictionResult> predictions;
             * std::vector<std::string> imagePaths;
             * // ... run predictions ...
             *
             * PredictionExporter::exportToJSON(predictions, imagePaths, "./results/predictions.json");
             * @endcode
             */
            class PredictionExporter {
            public:
                // Delete constructors to prevent instantiation
                PredictionExporter() = delete;
                ~PredictionExporter() = delete;
                PredictionExporter(const PredictionExporter&) = delete;
                PredictionExporter& operator=(const PredictionExporter&) = delete;

                /**
                 * @brief Export prediction results to JSON file
                 * @param predictions Vector of prediction results
                 * @param imagePaths Vector of image paths (must match predictions size)
                 * @param filepath Path to output JSON file
                 * @return true if successful, false otherwise
                 */
                static bool exportToJSON(
                    const std::vector<WheelDL::PredictionResult>& predictions,
                    const std::vector<std::string>& imagePaths,
                    const std::string& filepath);

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
