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

                if (imageSize == -1)
					return image;

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

            std::pair<int, int> ImageIO::getImageDimensions(const std::string& path)
            {
                // Convert to filesystem::path for proper Unicode handling
                std::filesystem::path filePath(path);

                // Check if file exists
                if (!std::filesystem::exists(filePath))
                {
                    throw std::runtime_error("Image file does not exist: " + path);
                }

                // Open file in binary mode
                std::ifstream file(filePath, std::ios::binary);
                if (!file.is_open())
                {
                    throw std::runtime_error("Failed to open image file: " + path);
                }

                // Read first 32 bytes (enough for most format headers)
                std::vector<uint8_t> header(32);
                file.read(reinterpret_cast<char*>(header.data()), 32);
                size_t bytesRead = static_cast<size_t>(file.gcount());

                if (bytesRead < 8)
                {
                    throw std::runtime_error("File too small to be a valid image: " + path);
                }

                // Helper lambda to read big-endian uint32
                auto readBE32 = [](const uint8_t* data) -> uint32_t {
                    return (static_cast<uint32_t>(data[0]) << 24) |
                           (static_cast<uint32_t>(data[1]) << 16) |
                           (static_cast<uint32_t>(data[2]) << 8) |
                           static_cast<uint32_t>(data[3]);
                };

                // Helper lambda to read little-endian uint32
                auto readLE32 = [](const uint8_t* data) -> uint32_t {
                    return static_cast<uint32_t>(data[0]) |
                           (static_cast<uint32_t>(data[1]) << 8) |
                           (static_cast<uint32_t>(data[2]) << 16) |
                           (static_cast<uint32_t>(data[3]) << 24);
                };

                // Helper lambda to read big-endian uint16
                auto readBE16 = [](const uint8_t* data) -> uint16_t {
                    return (static_cast<uint16_t>(data[0]) << 8) |
                           static_cast<uint16_t>(data[1]);
                };

                // Detect PNG: 89 50 4E 47 0D 0A 1A 0A
                if (header[0] == 0x89 && header[1] == 0x50 && header[2] == 0x4E && header[3] == 0x47)
                {
                    // PNG format: IHDR chunk starts at byte 8
                    // Skip 8 bytes (signature) + 4 bytes (IHDR length) + 4 bytes (IHDR tag) = 16 bytes
                    if (bytesRead >= 24)
                    {
                        int width = static_cast<int>(readBE32(&header[16]));
                        int height = static_cast<int>(readBE32(&header[20]));
                        return { width, height };
                    }
                }

                // Detect JPEG: FF D8
                else if (header[0] == 0xFF && header[1] == 0xD8)
                {
                    // JPEG format: need to find SOF marker (Start of Frame)
                    file.seekg(0, std::ios::beg);
                    std::vector<uint8_t> buffer(1024); // Read larger chunk for JPEG

                    while (file.read(reinterpret_cast<char*>(buffer.data()), buffer.size()) || file.gcount() > 0)
                    {
                        size_t size = static_cast<size_t>(file.gcount());
                        for (size_t i = 0; i + 9 < size; ++i)
                        {
                            // Look for SOF markers: FF C0 to FF CF (except C4, C8, CC)
                            if (buffer[i] == 0xFF && (buffer[i + 1] >= 0xC0 && buffer[i + 1] <= 0xCF) &&
                                buffer[i + 1] != 0xC4 && buffer[i + 1] != 0xC8 && buffer[i + 1] != 0xCC)
                            {
                                // SOF format: FF Cx [length:2] [precision:1] [height:2] [width:2]
                                int height = static_cast<int>(readBE16(&buffer[i + 5]));
                                int width = static_cast<int>(readBE16(&buffer[i + 7]));
                                return { width, height };
                            }
                        }
                    }
                    throw std::runtime_error("Failed to find JPEG SOF marker: " + path);
                }

                // Detect BMP: 42 4D (BM)
                else if (header[0] == 0x42 && header[1] == 0x4D)
                {
                    // BMP format: width at offset 18, height at offset 22
                    if (bytesRead >= 26)
                    {
                        int width = static_cast<int>(readLE32(&header[18]));
                        int height = static_cast<int>(readLE32(&header[22]));
                        return { width, std::abs(height) }; // Height can be negative
                    }
                }

                // Detect WEBP: RIFF ... WEBP
                else if (header[0] == 0x52 && header[1] == 0x49 && header[2] == 0x46 && header[3] == 0x46 &&
                         bytesRead >= 30 && header[8] == 0x57 && header[9] == 0x45 && header[10] == 0x42 && header[11] == 0x50)
                {
                    // WEBP VP8 format
                    if (header[12] == 0x56 && header[13] == 0x50 && header[14] == 0x38)
                    {
                        if (header[15] == 0x20) // VP8
                        {
                            // VP8 lossy format - need to read more
                            std::vector<uint8_t> vp8Header(30);
                            file.seekg(0, std::ios::beg);
                            file.read(reinterpret_cast<char*>(vp8Header.data()), 30);

                            if (file.gcount() >= 30)
                            {
                                int width = (vp8Header[26] | (vp8Header[27] << 8)) & 0x3fff;
                                int height = (vp8Header[28] | (vp8Header[29] << 8)) & 0x3fff;
                                return { width, height };
                            }
                        }
                        else if (header[15] == 0x4C) // VP8L
                        {
                            // VP8 lossless format
                            if (bytesRead >= 25)
                            {
                                uint32_t bits = readLE32(&header[21]);
                                int width = static_cast<int>((bits & 0x3FFF) + 1);
                                int height = static_cast<int>(((bits >> 14) & 0x3FFF) + 1);
                                return { width, height };
                            }
                        }
                    }
                }

                // Detect TIFF: II (little-endian) or MM (big-endian)
                else if ((header[0] == 0x49 && header[1] == 0x49 && header[2] == 0x2A && header[3] == 0x00) ||  // "II*\0" (little-endian)
                         (header[0] == 0x4D && header[1] == 0x4D && header[2] == 0x00 && header[3] == 0x2A))    // "MM\0*" (big-endian)
                {
                    bool isLittleEndian = (header[0] == 0x49);

                    // Helper to read based on endianness
                    auto readTIFF32 = [&](const uint8_t* data) -> uint32_t {
                        return isLittleEndian ? readLE32(data) : readBE32(data);
                    };

                    auto readTIFF16 = [&](const uint8_t* data) -> uint16_t {
                        return isLittleEndian
                            ? (static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8))
                            : ((static_cast<uint16_t>(data[0]) << 8) | static_cast<uint16_t>(data[1]));
                    };

                    // Read IFD offset (bytes 4-7)
                    uint32_t ifdOffset = readTIFF32(&header[4]);

                    // Read IFD
                    std::vector<uint8_t> ifdBuffer(256);
                    file.seekg(ifdOffset, std::ios::beg);
                    file.read(reinterpret_cast<char*>(ifdBuffer.data()), 256);

                    if (file.gcount() < 2)
                    {
                        throw std::runtime_error("Invalid TIFF IFD: " + path);
                    }

                    // Number of directory entries
                    uint16_t numEntries = readTIFF16(&ifdBuffer[0]);

                    int width = 0;
                    int height = 0;

                    // Read each IFD entry (12 bytes each)
                    for (uint16_t i = 0; i < numEntries && i < 20; ++i) // Limit to first 20 entries
                    {
                        size_t entryOffset = 2 + i * 12;
                        if (entryOffset + 12 > ifdBuffer.size())
                            break;

                        uint16_t tag = readTIFF16(&ifdBuffer[entryOffset]);
                        uint16_t type = readTIFF16(&ifdBuffer[entryOffset + 2]);
                        uint32_t value = readTIFF32(&ifdBuffer[entryOffset + 8]);

                        // Tag 256 (0x0100): ImageWidth
                        if (tag == 0x0100)
                        {
                            width = (type == 3) ? readTIFF16(&ifdBuffer[entryOffset + 8]) : static_cast<int>(value);
                        }
                        // Tag 257 (0x0101): ImageLength (Height)
                        else if (tag == 0x0101)
                        {
                            height = (type == 3) ? readTIFF16(&ifdBuffer[entryOffset + 8]) : static_cast<int>(value);
                        }

                        if (width > 0 && height > 0)
                        {
                            return { width, height };
                        }
                    }

                    if (width > 0 && height > 0)
                    {
                        return { width, height };
                    }
                    throw std::runtime_error("Failed to read TIFF dimensions: " + path);
                }

                // If we reach here, format is not supported
                throw std::runtime_error("Unsupported image format or corrupted file: " + path);
            }
        } // namespace Utils
    } // namespace Data
} // namespace WheelDL
