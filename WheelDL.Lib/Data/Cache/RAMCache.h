#pragma once

#include <string>
#include <unordered_map>
#include <list>
#include <mutex>
#include <optional>
#include <opencv2/opencv.hpp>

namespace WheelDL
{
    namespace Data
    {
        namespace Cache
        {
            /**
             * @class RAMCache
             * @brief In-memory cache for images
             *
             * Thread-safe image cache stored in RAM using cv::Mat
             */
            class RAMCache
            {
            public:
                /**
                 * @brief Construct RAMCache with optional max size and memory limit
                 * @param maxSize Maximum number of cached items (0 = unlimited)
                 * @param maxMemoryBytes Maximum memory usage in bytes (0 = unlimited, default: 1GB)
                 */
                explicit RAMCache(size_t maxSize = 0, size_t maxMemoryBytes = 1024ULL * 1024 * 1024);
                ~RAMCache() = default;

                // Non-copyable
                RAMCache(const RAMCache&) = delete;
                RAMCache& operator=(const RAMCache&) = delete;

                /**
                 * @brief Store image in cache
                 * @param key Unique identifier for the image
                 * @param image Image to cache
                 */
                void put(const std::string& key, const cv::Mat& image);

                /**
                 * @brief Retrieve image from cache
                 * @param key Unique identifier for the image
                 * @return std::optional<cv::Mat> Image if found, std::nullopt otherwise
                 * @note Updates LRU order, so this is not a const operation
                 */
                std::optional<cv::Mat> get(const std::string& key);

                /**
                 * @brief Check if key exists in cache
                 * @param key Unique identifier to check
                 * @return bool True if key exists
                 */
                bool has(const std::string& key) const;

                /**
                 * @brief Clear all cached images
                 */
                void clear();

                /**
                 * @brief Get number of cached images
                 * @return size_t Number of images in cache
                 */
                size_t size() const;

                /**
                 * @brief Get current memory usage in bytes
                 * @return size_t Current memory usage
                 */
                size_t getMemoryUsage() const;

            private:
                /**
                 * @brief Evict least recently used item if cache is full
                 */
                void evictLRU();

                /**
                 * @brief Calculate memory size of an image
                 * @param image Image to calculate size for
                 * @return size_t Memory size in bytes
                 */
                static size_t calculateImageSize(const cv::Mat& image);

                // Use recursive_mutex to safely handle potential recursive calls
                mutable std::recursive_mutex mutex_;
                size_t maxSize_;  // Maximum cache size (0 = unlimited)
                size_t maxMemoryBytes_;  // Maximum memory usage in bytes (0 = unlimited)
                size_t currentMemoryBytes_;  // Current memory usage in bytes

                // LRU implementation: list for order, map for fast access
                std::list<std::string> lruList_;  // Most recent at front
                // Store image, LRU iterator, and image size
                std::unordered_map<std::string, std::tuple<cv::Mat, std::list<std::string>::iterator, size_t>> cache_;
            };

        } // namespace Cache
    } // namespace Data
} // namespace WheelDL
