#include "pch.h"
#include "Data/Cache/RAMCache.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

using namespace WheelDL::Data::Cache;

// =============================================================================
// Test Fixture
// =============================================================================

class RAMCacheTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper: Create test image with specified size
    cv::Mat createTestImage(int width = 100, int height = 100, int type = CV_8UC3) {
        cv::Mat image(height, width, type);
        image.setTo(cv::Scalar(128, 64, 32));
        return image;
    }

    // Helper: Calculate expected image size in bytes
    size_t calculateImageSize(const cv::Mat& image) {
        if (image.empty()) return 0;
        return image.total() * image.elemSize();
    }

    // Helper: Create image with specific memory size (approximately)
    cv::Mat createImageWithSize(size_t targetBytes) {
        // CV_8UC3: 3 bytes per pixel
        size_t pixelsNeeded = targetBytes / 3;
        int side = static_cast<int>(std::sqrt(pixelsNeeded));
        if (side < 1) side = 1;
        return createTestImage(side, side);
    }
};

// =============================================================================
// 5.1 Construction Tests (RC-001 ~ RC-005)
// =============================================================================

// RC-001: Default constructor
TEST_F(RAMCacheTest, Ctor_Default) {
    RAMCache cache;
    EXPECT_EQ(0, cache.size());
    EXPECT_EQ(0, cache.getMemoryUsage());
}

// RC-002: Constructor with maxSize
TEST_F(RAMCacheTest, Ctor_WithMaxSize) {
    RAMCache cache(10);  // Max 10 items
    EXPECT_EQ(0, cache.size());

    // Add 10 items
    for (int i = 0; i < 10; ++i) {
        cache.put("key" + std::to_string(i), createTestImage(10, 10));
    }
    EXPECT_EQ(10, cache.size());

    // Add 11th item - should evict one
    cache.put("key10", createTestImage(10, 10));
    EXPECT_EQ(10, cache.size());
}

// RC-003: Constructor with maxMemory
TEST_F(RAMCacheTest, Ctor_WithMaxMemory) {
    // 1KB max memory
    RAMCache cache(0, 1024);
    EXPECT_EQ(0, cache.size());
    EXPECT_EQ(0, cache.getMemoryUsage());
}

// RC-004: Constructor with both limits
TEST_F(RAMCacheTest, Ctor_WithBothLimits) {
    RAMCache cache(5, 1024 * 1024);  // 5 items, 1MB
    EXPECT_EQ(0, cache.size());
}

// RC-005: Constructor maxSize=0 (unlimited)
TEST_F(RAMCacheTest, Ctor_UnlimitedSize) {
    RAMCache cache(0, 10 * 1024 * 1024);  // Unlimited count, 10MB memory

    // Add many items
    for (int i = 0; i < 100; ++i) {
        cache.put("key" + std::to_string(i), createTestImage(10, 10));
    }
    EXPECT_EQ(100, cache.size());
}

// =============================================================================
// 5.2 put/get Operations Tests (RC-006 ~ RC-023)
// =============================================================================

// RC-006: put single image
TEST_F(RAMCacheTest, Put_SingleImage) {
    RAMCache cache;
    cv::Mat image = createTestImage();

    cache.put("test_key", image);

    EXPECT_EQ(1, cache.size());
    EXPECT_TRUE(cache.has("test_key"));
}

// RC-007: put empty image (should skip)
TEST_F(RAMCacheTest, Put_EmptyImage) {
    RAMCache cache;
    cv::Mat emptyImage;

    cache.put("empty_key", emptyImage);

    EXPECT_EQ(0, cache.size());
    EXPECT_FALSE(cache.has("empty_key"));
}

// RC-008: put updates existing key
TEST_F(RAMCacheTest, Put_UpdatesExistingKey) {
    RAMCache cache;
    cv::Mat image1 = createTestImage(50, 50);
    cv::Mat image2 = createTestImage(100, 100);

    cache.put("key", image1);
    size_t size1 = cache.getMemoryUsage();

    cache.put("key", image2);
    size_t size2 = cache.getMemoryUsage();

    EXPECT_EQ(1, cache.size());
    EXPECT_NE(size1, size2);  // Memory should change
}

// RC-009: put triggers size eviction
TEST_F(RAMCacheTest, Put_TriggersSizeEviction) {
    RAMCache cache(3);  // Max 3 items

    cache.put("key1", createTestImage(10, 10));
    cache.put("key2", createTestImage(10, 10));
    cache.put("key3", createTestImage(10, 10));
    EXPECT_EQ(3, cache.size());

    cache.put("key4", createTestImage(10, 10));
    EXPECT_EQ(3, cache.size());
    EXPECT_FALSE(cache.has("key1"));  // LRU evicted
    EXPECT_TRUE(cache.has("key4"));
}

// RC-010: put triggers memory eviction
TEST_F(RAMCacheTest, Put_TriggersMemoryEviction) {
    // Small memory limit
    size_t imageSize = calculateImageSize(createTestImage(50, 50));
    RAMCache cache(0, imageSize * 2 + 100);  // Room for ~2 images

    cache.put("key1", createTestImage(50, 50));
    cache.put("key2", createTestImage(50, 50));

    // This should trigger eviction
    cache.put("key3", createTestImage(50, 50));

    EXPECT_LE(cache.size(), 2);
}

// RC-011: put image too large for cache
TEST_F(RAMCacheTest, Put_ImageTooLarge) {
    RAMCache cache(0, 1000);  // 1000 bytes max

    cv::Mat largeImage = createTestImage(100, 100);  // Much larger than 1000 bytes
    cache.put("large_key", largeImage);

    // Image should not be stored (too large)
    EXPECT_EQ(0, cache.size());
}

// RC-012: get existing key
TEST_F(RAMCacheTest, Get_ExistingKey) {
    RAMCache cache;
    cv::Mat original = createTestImage();
    original.at<cv::Vec3b>(0, 0) = cv::Vec3b(255, 0, 0);

    cache.put("key", original);
    auto result = cache.get("key");

    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(original.size(), result->size());
}

// RC-013: get non-existing key
TEST_F(RAMCacheTest, Get_NonExistingKey) {
    RAMCache cache;
    auto result = cache.get("non_existent");

    EXPECT_FALSE(result.has_value());
}

// RC-014: get updates LRU order
TEST_F(RAMCacheTest, Get_UpdatesLRUOrder) {
    RAMCache cache(3);

    cache.put("key1", createTestImage(10, 10));
    cache.put("key2", createTestImage(10, 10));
    cache.put("key3", createTestImage(10, 10));

    // Access key1 (makes it most recently used)
    cache.get("key1");

    // Add key4 - should evict key2 (now the LRU)
    cache.put("key4", createTestImage(10, 10));

    EXPECT_TRUE(cache.has("key1"));   // Recently accessed
    EXPECT_FALSE(cache.has("key2"));  // LRU evicted
    EXPECT_TRUE(cache.has("key3"));
    EXPECT_TRUE(cache.has("key4"));
}

// RC-015: get returns shallow copy from cache (reference counting)
TEST_F(RAMCacheTest, Get_ReturnsShallowCopy) {
    RAMCache cache;
    cv::Mat original = createTestImage();

    cache.put("key", original);

    // Get same key twice - should return shallow copies sharing same data
    auto result1 = cache.get("key");
    auto result2 = cache.get("key");

    EXPECT_TRUE(result1.has_value());
    EXPECT_TRUE(result2.has_value());
    // Multiple gets return shallow copies sharing the cached data
    EXPECT_EQ(result1->data, result2->data);
}

// RC-016: put/get multiple images
TEST_F(RAMCacheTest, PutGet_MultipleImages) {
    RAMCache cache;

    for (int i = 0; i < 10; ++i) {
        cache.put("key" + std::to_string(i), createTestImage(20, 20));
    }

    EXPECT_EQ(10, cache.size());

    for (int i = 0; i < 10; ++i) {
        auto result = cache.get("key" + std::to_string(i));
        EXPECT_TRUE(result.has_value());
    }
}

// RC-017: put clones image (independent copy)
TEST_F(RAMCacheTest, Put_ClonesImage) {
    RAMCache cache;
    cv::Mat original = createTestImage();
    cv::Vec3b originalPixel = original.at<cv::Vec3b>(0, 0);

    cache.put("key", original);

    // Modify original
    original.at<cv::Vec3b>(0, 0) = cv::Vec3b(255, 255, 255);

    // Cached image should be unchanged
    auto cached = cache.get("key");
    EXPECT_TRUE(cached.has_value());
    cv::Vec3b cachedPixel = cached->at<cv::Vec3b>(0, 0);
    EXPECT_EQ(originalPixel, cachedPixel);
}

// RC-018: LRU eviction order
TEST_F(RAMCacheTest, LRU_EvictionOrder) {
    RAMCache cache(3);

    cache.put("first", createTestImage(10, 10));
    cache.put("second", createTestImage(10, 10));
    cache.put("third", createTestImage(10, 10));

    // Add fourth - should evict "first"
    cache.put("fourth", createTestImage(10, 10));

    EXPECT_FALSE(cache.has("first"));
    EXPECT_TRUE(cache.has("second"));
    EXPECT_TRUE(cache.has("third"));
    EXPECT_TRUE(cache.has("fourth"));
}

// RC-019: memory calculation
TEST_F(RAMCacheTest, MemoryCalculation) {
    RAMCache cache;
    cv::Mat image = createTestImage(100, 100);  // 100*100*3 = 30000 bytes

    cache.put("key", image);

    size_t expectedSize = 100 * 100 * 3;  // CV_8UC3
    EXPECT_EQ(expectedSize, cache.getMemoryUsage());
}

// RC-020: calculateImageSize empty returns 0
TEST_F(RAMCacheTest, CalculateImageSize_Empty) {
    RAMCache cache;
    cv::Mat emptyImage;

    cache.put("key", emptyImage);

    EXPECT_EQ(0, cache.getMemoryUsage());
    EXPECT_EQ(0, cache.size());
}

// RC-021: put with memory eviction loop
TEST_F(RAMCacheTest, Put_MemoryEvictionLoop) {
    size_t smallImageSize = calculateImageSize(createTestImage(20, 20));
    RAMCache cache(0, smallImageSize * 3);  // Room for 3 small images

    // Add 5 small images
    for (int i = 0; i < 5; ++i) {
        cache.put("key" + std::to_string(i), createTestImage(20, 20));
    }

    // Should have evicted older ones to stay within memory limit
    EXPECT_LE(cache.getMemoryUsage(), smallImageSize * 3);
}

// RC-022: put batch eviction count
TEST_F(RAMCacheTest, Put_BatchEvictionCount) {
    RAMCache cache(5);  // Max 5 items

    // Fill cache
    for (int i = 0; i < 5; ++i) {
        cache.put("key" + std::to_string(i), createTestImage(10, 10));
    }
    EXPECT_EQ(5, cache.size());

    // Add one more
    cache.put("key5", createTestImage(10, 10));
    EXPECT_EQ(5, cache.size());
}

// RC-023: memory tracking accuracy
TEST_F(RAMCacheTest, MemoryTrackingAccuracy) {
    RAMCache cache;

    cv::Mat img1 = createTestImage(50, 50);   // 50*50*3 = 7500
    cv::Mat img2 = createTestImage(100, 50);  // 100*50*3 = 15000

    cache.put("key1", img1);
    size_t mem1 = cache.getMemoryUsage();
    EXPECT_EQ(7500, mem1);

    cache.put("key2", img2);
    size_t mem2 = cache.getMemoryUsage();
    EXPECT_EQ(7500 + 15000, mem2);
}

// =============================================================================
// 5.3 has/clear/size Operations Tests (RC-024 ~ RC-031)
// =============================================================================

// RC-024: has existing key
TEST_F(RAMCacheTest, Has_ExistingKey) {
    RAMCache cache;
    cache.put("key", createTestImage());

    EXPECT_TRUE(cache.has("key"));
}

// RC-025: has non-existing key
TEST_F(RAMCacheTest, Has_NonExistingKey) {
    RAMCache cache;

    EXPECT_FALSE(cache.has("non_existent"));
}

// RC-026: clear empties cache
TEST_F(RAMCacheTest, Clear_EmptiesCache) {
    RAMCache cache;
    cache.put("key1", createTestImage());
    cache.put("key2", createTestImage());
    cache.put("key3", createTestImage());

    EXPECT_EQ(3, cache.size());

    cache.clear();

    EXPECT_EQ(0, cache.size());
    EXPECT_FALSE(cache.has("key1"));
}

// RC-027: clear resets memory
TEST_F(RAMCacheTest, Clear_ResetsMemory) {
    RAMCache cache;
    cache.put("key1", createTestImage(100, 100));

    EXPECT_GT(cache.getMemoryUsage(), 0);

    cache.clear();

    EXPECT_EQ(0, cache.getMemoryUsage());
}

// RC-028: size empty cache
TEST_F(RAMCacheTest, Size_EmptyCache) {
    RAMCache cache;
    EXPECT_EQ(0, cache.size());
}

// RC-029: size with items
TEST_F(RAMCacheTest, Size_WithItems) {
    RAMCache cache;
    cache.put("key1", createTestImage());
    cache.put("key2", createTestImage());

    EXPECT_EQ(2, cache.size());
}

// RC-030: getMemoryUsage
TEST_F(RAMCacheTest, GetMemoryUsage) {
    RAMCache cache;
    EXPECT_EQ(0, cache.getMemoryUsage());

    cache.put("key", createTestImage(10, 10));

    EXPECT_EQ(10 * 10 * 3, cache.getMemoryUsage());
}

// RC-031: clear after put
TEST_F(RAMCacheTest, Clear_AfterPut) {
    RAMCache cache;

    for (int i = 0; i < 10; ++i) {
        cache.put("key" + std::to_string(i), createTestImage());
    }

    cache.clear();

    EXPECT_EQ(0, cache.size());
    EXPECT_EQ(0, cache.getMemoryUsage());

    // Can add items again after clear
    cache.put("new_key", createTestImage());
    EXPECT_EQ(1, cache.size());
}

// =============================================================================
// 5.4 Thread Safety Tests (RC-032 ~ RC-039)
// =============================================================================

// RC-032: concurrent put
TEST_F(RAMCacheTest, ThreadSafe_ConcurrentPut) {
    RAMCache cache;
    std::atomic<int> counter{0};
    const int numThreads = 4;
    const int itemsPerThread = 25;

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < itemsPerThread; ++i) {
                std::string key = "thread" + std::to_string(t) + "_key" + std::to_string(i);
                cache.put(key, createTestImage(10, 10));
                counter++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(numThreads * itemsPerThread, counter.load());
    EXPECT_EQ(numThreads * itemsPerThread, cache.size());
}

// RC-033: concurrent get
TEST_F(RAMCacheTest, ThreadSafe_ConcurrentGet) {
    RAMCache cache;

    // Pre-populate cache
    for (int i = 0; i < 10; ++i) {
        cache.put("key" + std::to_string(i), createTestImage(10, 10));
    }

    std::atomic<int> successCount{0};
    const int numThreads = 4;
    const int readsPerThread = 25;

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < readsPerThread; ++i) {
                auto result = cache.get("key" + std::to_string(i % 10));
                if (result.has_value()) {
                    successCount++;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(numThreads * readsPerThread, successCount.load());
}

// RC-034: concurrent put/get
TEST_F(RAMCacheTest, ThreadSafe_ConcurrentPutGet) {
    RAMCache cache;
    std::atomic<bool> running{true};
    std::atomic<int> putCount{0};
    std::atomic<int> getCount{0};

    // Writer thread
    std::thread writer([&]() {
        for (int i = 0; i < 100 && running; ++i) {
            cache.put("key" + std::to_string(i % 10), createTestImage(10, 10));
            putCount++;
        }
    });

    // Reader threads
    std::vector<std::thread> readers;
    for (int t = 0; t < 3; ++t) {
        readers.emplace_back([&]() {
            for (int i = 0; i < 50 && running; ++i) {
                cache.get("key" + std::to_string(i % 10));
                getCount++;
            }
        });
    }

    writer.join();
    for (auto& reader : readers) {
        reader.join();
    }

    EXPECT_EQ(100, putCount.load());
}

// RC-035: concurrent has
TEST_F(RAMCacheTest, ThreadSafe_ConcurrentHas) {
    RAMCache cache;
    cache.put("key", createTestImage());

    std::atomic<int> trueCount{0};
    const int numThreads = 4;
    const int checksPerThread = 100;

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < checksPerThread; ++i) {
                if (cache.has("key")) {
                    trueCount++;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(numThreads * checksPerThread, trueCount.load());
}

// RC-036: concurrent clear
TEST_F(RAMCacheTest, ThreadSafe_ConcurrentClear) {
    RAMCache cache;

    std::atomic<int> clearCount{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 10; ++i) {
                cache.put("key" + std::to_string(i), createTestImage(10, 10));
            }
            cache.clear();
            clearCount++;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(4, clearCount.load());
}

// RC-037: concurrent size
TEST_F(RAMCacheTest, ThreadSafe_ConcurrentSize) {
    RAMCache cache;
    cache.put("key", createTestImage());

    std::atomic<int> sizeCheckCount{0};
    const int numThreads = 4;

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 100; ++i) {
                size_t s = cache.size();
                (void)s;  // Suppress unused warning
                sizeCheckCount++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(numThreads * 100, sizeCheckCount.load());
}

// RC-038: concurrent getMemoryUsage
TEST_F(RAMCacheTest, ThreadSafe_ConcurrentGetMemoryUsage) {
    RAMCache cache;
    cache.put("key", createTestImage(50, 50));

    std::atomic<int> checkCount{0};
    const int numThreads = 4;

    std::vector<std::thread> threads;
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 100; ++i) {
                size_t mem = cache.getMemoryUsage();
                (void)mem;
                checkCount++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(numThreads * 100, checkCount.load());
}

// RC-039: high contention stress test
TEST_F(RAMCacheTest, ThreadSafe_HighContention) {
    RAMCache cache(100);  // Limit to 100 items

    std::atomic<bool> done{false};
    std::atomic<int> operations{0};

    auto worker = [&](int id) {
        while (!done) {
            std::string key = "key" + std::to_string(id) + "_" + std::to_string(operations % 50);

            // Random operation
            int op = operations % 4;
            if (op == 0) {
                cache.put(key, createTestImage(10, 10));
            } else if (op == 1) {
                cache.get(key);
            } else if (op == 2) {
                cache.has(key);
            } else {
                cache.size();
            }
            operations++;
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back(worker, t);
    }

    // Run for 100ms
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    done = true;

    for (auto& thread : threads) {
        thread.join();
    }

    // Just verify no crashes occurred
    EXPECT_GT(operations.load(), 0);
}

// =============================================================================
// 5.5 evictLRU Tests (RC-040 ~ RC-045)
// =============================================================================

// RC-040: evictLRU removes oldest
TEST_F(RAMCacheTest, EvictLRU_RemovesOldest) {
    RAMCache cache(2);

    cache.put("old", createTestImage(10, 10));
    cache.put("newer", createTestImage(10, 10));

    // Add third item - should evict "old"
    cache.put("newest", createTestImage(10, 10));

    EXPECT_FALSE(cache.has("old"));
    EXPECT_TRUE(cache.has("newer"));
    EXPECT_TRUE(cache.has("newest"));
}

// RC-041: evictLRU empty list (early return)
TEST_F(RAMCacheTest, EvictLRU_EmptyList) {
    RAMCache cache(1);

    // Empty cache, no eviction needed
    cache.put("key", createTestImage(10, 10));

    EXPECT_EQ(1, cache.size());
}

// RC-042: evictLRU updates memory
TEST_F(RAMCacheTest, EvictLRU_UpdatesMemory) {
    size_t imageSize = calculateImageSize(createTestImage(10, 10));
    RAMCache cache(2);

    cache.put("key1", createTestImage(10, 10));
    cache.put("key2", createTestImage(10, 10));

    size_t memBefore = cache.getMemoryUsage();
    EXPECT_EQ(imageSize * 2, memBefore);

    // Trigger eviction
    cache.put("key3", createTestImage(10, 10));

    size_t memAfter = cache.getMemoryUsage();
    EXPECT_EQ(imageSize * 2, memAfter);  // Still 2 images
}

// RC-043: evictLRU cache map sync
TEST_F(RAMCacheTest, EvictLRU_CacheMapSync) {
    RAMCache cache(3);

    cache.put("a", createTestImage(10, 10));
    cache.put("b", createTestImage(10, 10));
    cache.put("c", createTestImage(10, 10));

    // Evict "a"
    cache.put("d", createTestImage(10, 10));

    // Verify both map and list are in sync
    EXPECT_EQ(3, cache.size());
    EXPECT_FALSE(cache.has("a"));
    EXPECT_TRUE(cache.has("b"));
    EXPECT_TRUE(cache.has("c"));
    EXPECT_TRUE(cache.has("d"));
}

// RC-044: evictLRU multiple calls
TEST_F(RAMCacheTest, EvictLRU_MultipleCalls) {
    RAMCache cache(2);

    cache.put("1", createTestImage(10, 10));
    cache.put("2", createTestImage(10, 10));
    cache.put("3", createTestImage(10, 10));  // Evicts "1"
    cache.put("4", createTestImage(10, 10));  // Evicts "2"
    cache.put("5", createTestImage(10, 10));  // Evicts "3"

    EXPECT_EQ(2, cache.size());
    EXPECT_TRUE(cache.has("4"));
    EXPECT_TRUE(cache.has("5"));
}

// RC-045: evictLRU with missing map entry (edge case)
TEST_F(RAMCacheTest, EvictLRU_Consistency) {
    RAMCache cache(5);

    // Fill cache
    for (int i = 0; i < 5; ++i) {
        cache.put("key" + std::to_string(i), createTestImage(10, 10));
    }

    // Multiple evictions
    for (int i = 5; i < 15; ++i) {
        cache.put("key" + std::to_string(i), createTestImage(10, 10));
    }

    // Verify consistency
    EXPECT_EQ(5, cache.size());

    // Only the last 5 should remain
    for (int i = 10; i < 15; ++i) {
        EXPECT_TRUE(cache.has("key" + std::to_string(i)));
    }
}
