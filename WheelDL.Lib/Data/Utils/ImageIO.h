#pragma once

#include <string>
#include <opencv2/opencv.hpp>

namespace WheelDL
{
    namespace Data
    {
        namespace Utils
        {
            /**
             * @class ImageIO
             * @brief Static utility class for image I/O operations
             *
             * Provides functions for loading, saving, and color space conversion
             */
            class ImageIO
            {
            public:
                // Delete constructor (static class)
                ImageIO() = delete;
                ~ImageIO() = delete;
                ImageIO(const ImageIO&) = delete;
                ImageIO& operator=(const ImageIO&) = delete;

                /**
                 * @brief Load image from file path
                 * @param path Path to image file
                 * @return cv::Mat Loaded image (BGR format)
                 * @throws std::runtime_error if image cannot be loaded
                 */
                static cv::Mat loadImage(const std::string& path, int imageSize);
                static cv::Mat loadImage(const std::string& path) {
                    return loadImage(path, -1);
				}
                /**
                 * @brief Save image to file path
                 * @param path Path to save the image
                 * @param image Image to save
                 * @throws std::runtime_error if image cannot be saved
                 */
                static void saveImage(const std::string& path, const cv::Mat& image);

                /**
                 * @brief Get image dimensions without decoding the entire image
                 * @param path Path to image file
                 * @return std::pair<int, int> Width and height of the image
                 * @throws std::runtime_error if image format is unsupported or file cannot be read
                 *
                 * Supported formats: PNG, JPEG, BMP, WEBP, TIFF
                 * This function only reads the file header, making it much faster than loading the entire image.
                 */
                static std::pair<int, int> getImageDimensions(const std::string& path);

            private:
				// Private helper functions can be added here if needed
				static void resizeImage(cv::Mat& image, int imageSize);
            };
        } // namespace Utils
    } // namespace Data
} // namespace WheelDL
