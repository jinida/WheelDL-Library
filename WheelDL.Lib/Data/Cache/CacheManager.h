#pragma once

#include "Data/Common/Annotation.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

namespace WheelDL
{
    namespace Data
    {
        namespace Cache
        {
            /**
             * @class CacheManager
             * @brief Utility class for loading and saving label caches
             *
             * Caches are stored as binary files for fast loading.
             * Hash-based validation ensures cache integrity.
             */
            class CacheManager
            {
            public:
                /**
                 * @brief Load label cache from file
                 * @param path Path to cache file
                 * @return Map of image paths to annotations
                 * @throws std::runtime_error if cache file cannot be loaded
                 */
                static std::unordered_map<std::string, Annotation> loadLabelCacheFrom(const std::string& path);

                /**
                 * @brief Save label cache to file
                 * @param path Path to cache file
                 * @param cache Map of image paths to annotations
                 * @param datasetHash Dataset hash for validation
                 * @throws std::runtime_error if cache cannot be saved
                 */
                static void saveLabelCacheTo(
                    const std::string& path,
                    const std::unordered_map<std::string, Annotation>& cache,
                    const std::string& datasetHash
                );

                /**
                 * @brief Compute hash of dataset directory
                 * @param dataPath Path to dataset directory
                 * @return Hash string (hex format)
                 */
                static std::string computeHash(const std::string& dataPath);

                /**
                 * @brief Verify cache validity against dataset hash
                 * @param cachePath Path to cache file
                 * @param datasetHash Expected dataset hash
                 * @return True if cache is valid, false otherwise
                 */
                static bool verifyCacheFrom(const std::string& cachePath, const std::string& datasetHash);

                /**
                 * @brief Serialize annotations to binary buffer
                 * @param annotations Annotation to serialize
                 * @return Binary data as string
                 */
                static std::string serializeAnnotation(const Annotation& annotations);

                /**
                 * @brief Deserialize annotations from binary buffer
                 * @param data Binary data as string
                 * @return Deserialized annotations
                 */
                static Annotation deserializeAnnotation(const std::string& data);

            private:
                // Static utility class - no instantiation
                CacheManager() = delete;
                ~CacheManager() = delete;
                CacheManager(const CacheManager&) = delete;
                CacheManager& operator=(const CacheManager&) = delete;

                /**
                 * @brief Compute hash of a single file
                 * @param filePath Path to file
                 * @return Hash string
                 */
                static std::string computeFileHash(const std::string& filePath);
            };

        } // namespace Cache
    } // namespace Data
} // namespace WheelDL
