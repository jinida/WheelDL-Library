#include "pch.h"
#include "ImageIO.h"
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <vector>

namespace WheelDL
{
    namespace Data
    {
        namespace Utils
        {
            cv::Mat ImageIO::loadImage(const std::string& path, int imageSize)
            {
                // Convert to filesystem::path for proper Unicode handling
                std::filesystem::path filePath(path);

                // Check if file exists
                if (!std::filesystem::exists(filePath))
                {
                    throw std::runtime_error("Image file does not exist: " + path);
                }

                // Use imdecode to support Unicode paths (Korean, Japanese, Chinese, etc.)
                // On Windows, use wide string path for proper Unicode support
                std::ifstream file(filePath, std::ios::binary);
                if (!file.is_open())
                {
                    throw std::runtime_error("Failed to open image file: " + path);
                }

                // Read file into buffer
                file.seekg(0, std::ios::end);
                std::streampos fileSizePos = file.tellg();

                // Check if tellg() failed
                if (fileSizePos < 0 || !file.good())
                {
                    throw std::runtime_error("Failed to get file size: " + path);
                }

                size_t fileSize = static_cast<size_t>(fileSizePos);
                file.seekg(0, std::ios::beg);

                std::vector<uint8_t> buffer(fileSize);
                file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
                file.close();

                // Decode image from buffer
                cv::Mat image;
                try
                {
                    image = cv::imdecode(buffer, cv::IMREAD_COLOR);
                }
                catch (const cv::Exception& e)
                {
                    throw std::runtime_error("Failed to decode image (corrupted or invalid format): " + path + " - " + e.what());
                }

                if (image.empty())
                {
                    throw std::runtime_error("Failed to decode image: " + path);
                }

				resizeImage(image, imageSize);
                return image;
            }

            void ImageIO::saveImage(const std::string& path, const cv::Mat& image)
            {
                if (image.empty())
                {
                    throw std::runtime_error("Cannot save empty image");
                }

                // Create parent directory if it doesn't exist
                std::filesystem::path filePath(path);
                std::filesystem::path parentPath = filePath.parent_path();

                if (!parentPath.empty() && !std::filesystem::exists(parentPath))
                {
                    std::error_code ec;
                    std::filesystem::create_directories(parentPath, ec);
                    if (ec)
                    {
                        throw std::runtime_error("Failed to create directory: " + path + " - " + ec.message());
                    }
                }

                // Get file extension
                std::string extension = filePath.extension().string();
                if (extension.empty())
                {
                    extension = ".jpg";  // Default to JPEG
                }

                // Encode image to buffer (supports Unicode paths)
                std::vector<uint8_t> buffer;
                bool success = false;

                try
                {
                    success = cv::imencode(extension, image, buffer);
                }
                catch (const cv::Exception& e)
                {
                    throw std::runtime_error("Failed to encode image (invalid format or codec not available): " + path + " - " + e.what());
                }

                if (!success || buffer.empty())
                {
                    throw std::runtime_error("Failed to encode image: " + path);
                }

                // Write buffer to file
                // Use filesystem::path for proper Unicode handling on Windows
                std::ofstream file(filePath, std::ios::binary);
                if (!file.is_open())
                {
                    throw std::runtime_error("Failed to open file for writing: " + path);
                }

                file.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
                file.close();
            }

            void ImageIO::resizeImage(cv::Mat& image, int imageSize)
            {
                if (imageSize <= 0)
                {
                    return; // No resizing needed
                }
                int originalWidth = image.cols;
                int originalHeight = image.rows;
                float scale = static_cast<float>(imageSize) / std::max(originalWidth, originalHeight);
                int newWidth = static_cast<int>(originalWidth * scale);
                int newHeight = static_cast<int>(originalHeight * scale);
				auto resizeFlag = (scale < 1.0f) ? cv::INTER_AREA : cv::INTER_LINEAR;
				cv::resize(image, image, cv::Size(newWidth, newHeight), 0, 0, resizeFlag);
            }
        } // namespace Utils
    } // namespace Data
} // namespace WheelDL
