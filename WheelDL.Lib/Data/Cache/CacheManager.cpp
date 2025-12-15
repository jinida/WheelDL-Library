#include "pch.h"
#include "Data/Cache/CacheManager.h"
#include "Utils/Common/Constants.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <filesystem>

namespace WheelDL
{
    namespace Data
    {
        namespace Cache
        {
            // Format: MAJOR * 1000000 + MINOR * 1000 + PATCH
            constexpr uint32_t CACHE_VERSION =
                Constants::CACHE_MAJOR_VERSION * 1000000 +
                Constants::CACHE_MINOR_VERSION * 1000 +
                Constants::CACHE_PATCH_VERSION;

            // Helper to write binary data
            template<typename T>
            static void writeBinary(std::ostream& stream, const T& value)
            {
                stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
            }

            // Helper to read binary data
            template<typename T>
            static void readBinary(std::istream& stream, T& value)
            {
                stream.read(reinterpret_cast<char*>(&value), sizeof(T));
                if (!stream) {
                    throw std::runtime_error("Failed to read binary data from stream");
                }
            }

            // Helper to write string
            static void writeString(std::ostream& stream, const std::string& str)
            {
                if (str.size() > std::numeric_limits<uint32_t>::max())
                {
                    throw std::runtime_error("String size exceeds uint32_t maximum");
                }
                uint32_t length = static_cast<uint32_t>(str.size());
                writeBinary(stream, length);
                stream.write(str.data(), length);
            }

            // Helper to read string
            static std::string readString(std::istream& stream)
            {
                uint32_t length;
                readBinary(stream, length);

                // Validate string length (max 100 MB to prevent memory exhaustion)
                constexpr uint32_t MAX_STRING_SIZE = 100 * 1024 * 1024;
                if (length > MAX_STRING_SIZE) {
                    throw std::runtime_error("String size exceeds maximum allowed size");
                }

                std::string result(length, '\0');
                stream.read(&result[0], length);
                if (!stream) {
                    throw std::runtime_error("Failed to read string data from stream");
                }
                return result;
            }

            // Helper to write vector
            template<typename T>
            static void writeVector(std::ostream& stream, const std::vector<T>& vec)
            {
                if (vec.size() > std::numeric_limits<uint32_t>::max())
                {
                    throw std::runtime_error("Vector size exceeds uint32_t maximum");
                }
                uint32_t size = static_cast<uint32_t>(vec.size());
                writeBinary(stream, size);
                if (size > 0) {
                    stream.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
                }
            }

            // Helper to read vector
            template<typename T>
            static std::vector<T> readVector(std::istream& stream)
            {
                uint32_t size;
                readBinary(stream, size);

                // Validate vector size (max 10 million elements to prevent memory exhaustion)
                constexpr uint32_t MAX_VECTOR_SIZE = 10 * 1000 * 1000;
                if (size > MAX_VECTOR_SIZE) {
                    throw std::runtime_error("Vector size exceeds maximum allowed size");
                }

                std::vector<T> result(size);
                if (size > 0) {
                    stream.read(reinterpret_cast<char*>(result.data()), size * sizeof(T));
                    if (!stream) {
                        throw std::runtime_error("Failed to read vector data from stream");
                    }
                }
                return result;
            }

            std::string CacheManager::serializeAnnotation(const Annotation& annotations)
            {
                std::ostringstream oss(std::ios::binary);

                // Serialize label type
                writeBinary(oss, static_cast<uint32_t>(annotations.labelType_));

                // Serialize classes
                writeVector(oss, annotations.classes_);

                // Serialize points (vector<vector<float>>)
                if (annotations.points_.size() > std::numeric_limits<uint32_t>::max())
                {
                    throw std::runtime_error("Points vector size exceeds uint32_t maximum");
                }
                uint32_t numPoints = static_cast<uint32_t>(annotations.points_.size());
                writeBinary(oss, numPoints);
                for (const auto& point : annotations.points_) {
                    writeVector(oss, point);
                }

                return oss.str();
            }

            Annotation CacheManager::deserializeAnnotation(const std::string& data)
            {
                std::istringstream iss(data, std::ios::binary);

                // Deserialize label type
                uint32_t labelTypeValue;
                readBinary(iss, labelTypeValue);
                LabelType labelType = static_cast<LabelType>(labelTypeValue);

                Annotation annotations(labelType);

                // Deserialize classes
                annotations.classes_ = readVector<int>(iss);

                // Deserialize points (vector<vector<float>>)
                uint32_t numPoints;
                readBinary(iss, numPoints);

                // Validate points count (max 10 million to prevent memory exhaustion)
                constexpr uint32_t MAX_POINTS_COUNT = 10 * 1000 * 1000;
                if (numPoints > MAX_POINTS_COUNT) {
                    throw std::runtime_error("Points count exceeds maximum allowed size");
                }

                annotations.points_.resize(numPoints);
                for (uint32_t i = 0; i < numPoints; ++i) {
                    annotations.points_[i] = readVector<float>(iss);
                }

                return annotations;
            }

            std::unordered_map<std::string, Annotation> CacheManager::loadLabelCacheFrom(const std::string& path)
            {
                std::ifstream file(path, std::ios::binary);
                if (!file.is_open()) {
                    throw std::runtime_error("Failed to open cache file: " + path);
                }

                // Read version
                uint32_t version;
                readBinary(file, version);
                if (version != CACHE_VERSION) {
                    throw std::runtime_error("Cache version mismatch. Expected " +
                        std::to_string(CACHE_VERSION) + ", got " + std::to_string(version));
                }

                // Read dataset hash
                std::string datasetHash = readString(file);

                // Read number of entries
                uint32_t numEntries;
                readBinary(file, numEntries);

                // Read all entries
                std::unordered_map<std::string, Annotation> cache;
                for (uint32_t i = 0; i < numEntries; ++i) {
                    std::string imagePath = readString(file);
                    std::string annotationData = readString(file);
                    cache[imagePath] = deserializeAnnotation(annotationData);
                }

                return cache;
            }

            void CacheManager::saveLabelCacheTo(
                const std::string& path,
                const std::unordered_map<std::string, Annotation>& cache,
                const std::string& datasetHash)
            {
                // Use temporary file for atomic write
                std::string tempPath = path + ".tmp";

                // RAII guard to ensure temporary file cleanup
                bool commitSucceeded = false;
                auto cleanupGuard = [&tempPath, &commitSucceeded](void*) {
                    if (!commitSucceeded) {
                        // Use error_code version to avoid exception during cleanup
                        std::error_code ec;
                        std::filesystem::remove(tempPath, ec);
                        // Ignore error - best effort cleanup
                    }
                };
                auto guardWrapper = std::unique_ptr<void, decltype(cleanupGuard)>(
                    reinterpret_cast<void*>(1), cleanupGuard
                );

                try
                {
                    std::ofstream file(tempPath, std::ios::binary);
                    if (!file.is_open()) {
                        throw std::runtime_error("Failed to create cache file: " + tempPath);
                    }

                    // Write version
                    writeBinary(file, CACHE_VERSION);

                    // Write dataset hash for validation
                    writeString(file, datasetHash);

                    // Write number of entries
                    if (cache.size() > std::numeric_limits<uint32_t>::max())
                    {
                        throw std::runtime_error("Cache size exceeds uint32_t maximum");
                    }
                    uint32_t numEntries = static_cast<uint32_t>(cache.size());
                    writeBinary(file, numEntries);

                    // Write all entries
                    for (const auto& [imagePath, annotations] : cache) {
                        writeString(file, imagePath);
                        std::string annotationData = serializeAnnotation(annotations);
                        writeString(file, annotationData);
                    }

                    file.close();

                    // Atomically replace old file with new file
                    std::filesystem::rename(tempPath, path);
                    commitSucceeded = true;  // Mark success to prevent cleanup
                }
                catch (...)
                {
                    // Cleanup will be handled by RAII guard
                    throw;
                }
            }

            std::string CacheManager::computeFileHash(const std::string& filePath)
            {
                namespace fs = std::filesystem;

                // Use filesystem API for file size (no file I/O needed)
                std::error_code ec;
                size_t fileSize = static_cast<size_t>(fs::file_size(filePath, ec));
                if (ec) {
                    return "";
                }

                // Combine file size with hash of content
                std::hash<std::string> hasher;
                size_t hash = hasher(std::to_string(fileSize));

                // Hash entire file content in chunks for better cache validation
                const size_t CHUNK_SIZE = 8192;  // 8KB chunks for streaming
                if (fileSize > 0) {
                    std::ifstream file(filePath, std::ios::binary);
                    if (file.is_open()) {
                        std::vector<char> buffer(CHUNK_SIZE);
                        // Add safety checks to prevent infinite loop
                        size_t maxIterations = (fileSize / CHUNK_SIZE) + 2;  // +2 for safety margin
                        size_t iteration = 0;
                        while ((file.read(buffer.data(), buffer.size()) || file.gcount() > 0) &&
                               file.good() && iteration < maxIterations) {
                            std::string chunk(buffer.begin(), buffer.begin() + file.gcount());
                            hash ^= hasher(chunk) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
                            iteration++;
                        }
                    }
                }

                // Convert to hex string
                std::ostringstream oss;
                oss << std::hex << hash;
                return oss.str();
            }

            std::string CacheManager::computeHash(const std::string& dataPath)
            {
                namespace fs = std::filesystem;

                if (!fs::exists(dataPath)) {
                    throw std::runtime_error("Data path does not exist: " + dataPath);
                }

                std::vector<std::string> fileHashes;
                std::hash<std::string> hasher;

                // Recursively collect all files with error handling
                try {
                    for (const auto& entry : fs::recursive_directory_iterator(
                        dataPath,
                        fs::directory_options::skip_permission_denied)) {

                        try {
                            if (entry.is_regular_file()) {
                                std::string filePath = entry.path().string();
                                std::string fileName = entry.path().filename().string();

                                // Hash combination of file name and file hash
                                std::string fileHash = computeFileHash(filePath);
                                if (!fileHash.empty()) {
                                    fileHashes.push_back(fileName + ":" + fileHash);
                                }
                            }
                        } catch (const std::exception&) {
                            // Skip files that cause errors (e.g., permission denied)
                            continue;
                        }
                    }
                } catch (const std::exception& e) {
                    throw std::runtime_error("Error traversing directory: " + std::string(e.what()));
                }

                // Sort for consistent hashing
                std::sort(fileHashes.begin(), fileHashes.end());

                // Combine all file hashes
                std::string combined;
                for (const auto& fh : fileHashes) {
                    combined += fh;
                }

                size_t finalHash = hasher(combined);

                // Convert to hex string
                std::ostringstream oss;
                oss << std::hex << finalHash;
                return oss.str();
            }

            bool CacheManager::verifyCacheFrom(const std::string& cachePath, const std::string& datasetHash)
            {
                std::ifstream file(cachePath, std::ios::binary);
                if (!file.is_open()) {
                    return false;
                }

                // Read version
                uint32_t version;
                readBinary(file, version);
                if (version != CACHE_VERSION) {
                    return false;
                }

                // Read stored dataset hash
                std::string storedHash = readString(file);

                // Compare with provided hash
                return storedHash == datasetHash;
            }

        } // namespace Cache
    } // namespace Data
} // namespace WheelDL
