#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace WheelDL {
    namespace Utils {
        namespace Export {

            /**
             * @class ImageExporter
             * @brief Static utility class for exporting OpenCV images to files
             *
             * Provides functionality to save OpenCV Mat images to image files.
             * Commonly used for saving anomaly maps, heatmaps, or any visualization.
             * Supports single or batch export.
             *
             * All methods are static - no need to instantiate.
             *
             * Usage:
             * @code
             * cv::Mat image = ...;
             * ImageExporter::save(image, "./results/output.png");
             *
             * // Batch save
             * std::vector<cv::Mat> images;
             * std::vector<std::string> imagePaths;
             * ImageExporter::saveBatch(images, imagePaths, "./results/outputs");
             * @endcode
             */
            class ImageExporter {
            public:
                // Delete constructors to prevent instantiation
                ImageExporter() = delete;
                ~ImageExporter() = delete;
                ImageExporter(const ImageExporter&) = delete;
                ImageExporter& operator=(const ImageExporter&) = delete;

                /**
                 * @brief Save single image to file
                 * @param image OpenCV Mat containing image
                 * @param filepath Path to save image (e.g., "./output/image.png")
                 * @return true if successful, false otherwise
                 */
                static bool save(
                    const cv::Mat& image,
                    const std::string& filepath);

                /**
                 * @brief Save multiple images to files
                 * @param images Vector of images
                 * @param imagePaths Vector of original image paths (used for naming)
                 * @param outputDir Output directory for images
                 * @param suffix Suffix to add to filename (default: "_output")
                 * @return Number of successfully saved images
                 */
                static size_t saveBatch(
                    const std::vector<cv::Mat>& images,
                    const std::vector<std::string>& imagePaths,
                    const std::string& outputDir,
                    const std::string& suffix = "_output");

            private:
                /**
                 * @brief Ensure directory exists, create if needed
                 * @param filepath File path (directory will be extracted)
                 * @return true if directory exists or was created
                 */
                static bool ensureDirectoryExists(const std::string& filepath);

                /**
                 * @brief Extract filename stem from path
                 * @param filepath Full file path
                 * @return Filename without extension
                 */
                static std::string getFilenameStem(const std::string& filepath);
            };

        } // namespace Export
    } // namespace Utils
} // namespace WheelDL
