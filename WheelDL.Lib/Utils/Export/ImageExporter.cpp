#include "pch.h"
#include "ImageExporter.h"
#include "../Logger/Logger.h"
#include <filesystem>

namespace WheelDL {
    namespace Utils {
        namespace Export {

            bool ImageExporter::save(
                const cv::Mat& image,
                const std::string& filepath)
            {
                if (image.empty())
                {
                    return false;
                }

                try
                {
                    // Ensure directory exists
                    if (!ensureDirectoryExists(filepath))
                    {
                        return false;
                    }

                    // Save image
                    bool success = cv::imwrite(filepath, image);
                    return success;
                }
                catch (const std::exception& e)
                {
                    return false;
                }
            }

            size_t ImageExporter::saveBatch(
                const std::vector<cv::Mat>& images,
                const std::vector<std::string>& imagePaths,
                const std::string& outputDir,
                const std::string& suffix)
            {
                if (images.empty())
                {
                    return 0;
                }

                if (images.size() != imagePaths.size())
                {
                    return 0;
                }

                try
                {
                    // Create output directory
                    if (!std::filesystem::exists(outputDir))
                    {
                        std::filesystem::create_directories(outputDir);
                    }

                    size_t successCount = 0;

                    for (size_t i = 0; i < images.size(); ++i)
                    {
                        const auto& image = images[i];
                        const auto& imagePath = imagePaths[i];

                        if (image.empty())
                        {
                            continue; // Skip empty images
                        }

                        // Build output filename with parent folder info to avoid overwrites
                        std::filesystem::path p(imagePath);
                        std::string stem = p.stem().string();

                        // Get parent folder name to make filename unique
                        std::string parentFolder = "";
                        if (p.has_parent_path() && p.parent_path().has_filename()) {
                            parentFolder = p.parent_path().filename().string();
                        }

                        // Create unique filename: parentFolder_filename_suffix.png
                        std::string outputFilename;
                        if (!parentFolder.empty()) {
                            outputFilename = parentFolder + "_" + stem + suffix + ".png";
                        } else {
                            outputFilename = stem + suffix + ".png";
                        }

                        std::string outputPath = outputDir + "/" + outputFilename;

                        if (save(image, outputPath))
                        {
                            successCount++;
                        }
                    }

                    return successCount;
                }
                catch (const std::exception& e)
                {
                    return 0;
                }
            }

            bool ImageExporter::ensureDirectoryExists(const std::string& filepath)
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
                    return false;
                }
            }

            std::string ImageExporter::getFilenameStem(const std::string& filepath)
            {
                std::filesystem::path path(filepath);
                return path.stem().string();
            }

        } // namespace Export
    } // namespace Utils
} // namespace WheelDL
