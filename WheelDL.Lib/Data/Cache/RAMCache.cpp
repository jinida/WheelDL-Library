#include "pch.h"
#include "RAMCache.h"

namespace WheelDL
{
    namespace Data
    {
        namespace Cache
        {
            RAMCache::RAMCache(size_t maxSize, size_t maxMemoryBytes)
                : maxSize_(maxSize)
                , maxMemoryBytes_(maxMemoryBytes)
                , currentMemoryBytes_(0)
            {
            }

            size_t RAMCache::calculateImageSize(const cv::Mat& image)
            {
                if (image.empty()) {
                    return 0;
                }
                // Calculate memory size: number of elements * size of each element
                return image.total() * image.elemSize();
            }

            void RAMCache::put(const std::string& key, const cv::Mat& image)
            {
                std::lock_guard<std::recursive_mutex> lock(mutex_);

                // Skip empty images
                if (image.empty()) {
                    return;
                }

                cv::Mat clonedImage = image.clone();
                size_t imageSize = calculateImageSize(clonedImage);

                auto it = cache_.find(key);

                if (it != cache_.end())
                {
                    // Key exists - update image and move to front of LRU list
                    size_t oldSize = std::get<2>(it->second);
                    currentMemoryBytes_ -= oldSize;

                    lruList_.erase(std::get<1>(it->second));
                    lruList_.push_front(key);

                    std::get<0>(it->second) = clonedImage;
                    std::get<1>(it->second) = lruList_.begin();
                    std::get<2>(it->second) = imageSize;

                    currentMemoryBytes_ += imageSize;
                }
                else
                {
                    // New key - check if we need to evict based on count or memory
                    // Calculate how many items to evict in one go for better performance
                    size_t itemsToEvict = 0;

                    if (maxSize_ > 0 && cache_.size() >= maxSize_) {
                        itemsToEvict = cache_.size() - maxSize_ + 1;
                    }

                    if (maxMemoryBytes_ > 0 && currentMemoryBytes_ + imageSize > maxMemoryBytes_) {
                        // Calculate additional evictions needed for memory constraint
                        size_t memoryToFree = (currentMemoryBytes_ + imageSize) - maxMemoryBytes_;
                        size_t memoryFreed = 0;
                        size_t memoryEvictCount = 0;

                        // Estimate evictions needed based on average item size
                        auto it = lruList_.rbegin();
                        while (it != lruList_.rend() && memoryFreed < memoryToFree) {
                            auto cacheIt = cache_.find(*it);
                            if (cacheIt != cache_.end()) {
                                memoryFreed += std::get<2>(cacheIt->second);
                                memoryEvictCount++;
                            }
                            ++it;
                        }

                        itemsToEvict = std::max(itemsToEvict, memoryEvictCount);
                    }

                    // Evict calculated number of items
                    for (size_t i = 0; i < itemsToEvict && !cache_.empty(); ++i) {
                        evictLRU();
                    }

                    // Check if we have enough space now
                    if (maxMemoryBytes_ > 0 && currentMemoryBytes_ + imageSize > maxMemoryBytes_) {
                        // Image is too large for cache
                        return;
                    }

                    // Add new item
                    lruList_.push_front(key);
                    cache_[key] = std::make_tuple(clonedImage, lruList_.begin(), imageSize);
                    currentMemoryBytes_ += imageSize;
                }
            }

            std::optional<cv::Mat> RAMCache::get(const std::string& key)
            {
                std::lock_guard<std::recursive_mutex> lock(mutex_);

                auto it = cache_.find(key);
                if (it != cache_.end())
                {
                    // Move to front of LRU list (mark as recently used)
                    lruList_.erase(std::get<1>(it->second));
                    lruList_.push_front(key);
                    std::get<1>(it->second) = lruList_.begin();

                    // Return shallow copy (cv::Mat uses reference counting internally)
                    // This is thread-safe for read-only operations
                    // If you need to modify the image, call clone() explicitly
                    return std::get<0>(it->second);
                }

                return std::nullopt;
            }

            bool RAMCache::has(const std::string& key) const
            {
                std::lock_guard<std::recursive_mutex> lock(mutex_);
                return cache_.find(key) != cache_.end();
            }

            void RAMCache::clear()
            {
                std::lock_guard<std::recursive_mutex> lock(mutex_);
                cache_.clear();
                lruList_.clear();
                currentMemoryBytes_ = 0;
            }

            size_t RAMCache::size() const
            {
                std::lock_guard<std::recursive_mutex> lock(mutex_);
                return cache_.size();
            }

            size_t RAMCache::getMemoryUsage() const
            {
                std::lock_guard<std::recursive_mutex> lock(mutex_);
                return currentMemoryBytes_;
            }

            void RAMCache::evictLRU()
            {
                // Remove least recently used item (at the back of the list)
                if (!lruList_.empty())
                {
                    std::string lruKey = lruList_.back();
                    lruList_.pop_back();

                    auto it = cache_.find(lruKey);
                    if (it != cache_.end()) {
                        currentMemoryBytes_ -= std::get<2>(it->second);
                        cache_.erase(it);
                    }
                }
            }

        } // namespace Cache
    } // namespace Data
} // namespace WheelDL
