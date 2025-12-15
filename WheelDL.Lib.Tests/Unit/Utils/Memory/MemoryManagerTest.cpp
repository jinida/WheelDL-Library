#include "pch.h"
#include <gtest/gtest.h>
#include "Utils/Memory/MemoryManager.h"
#include <thread>
#include <vector>

using namespace WheelDL::Utils;

// =============================================================================
// Test Fixture
// =============================================================================

class MemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Get the singleton instance
    }

    void TearDown() override {
        // Nothing to clean up for singleton
    }
};

// =============================================================================
// Singleton Tests (MM-001 ~ MM-002)
// =============================================================================

// MM-001: GetInstance returns singleton
TEST_F(MemoryManagerTest, GetInstance_Singleton) {
    MemoryManager& instance1 = MemoryManager::getInstance();
    MemoryManager& instance2 = MemoryManager::getInstance();

    EXPECT_EQ(&instance1, &instance2);
}

// MM-002: GetInstance is thread-safe
TEST_F(MemoryManagerTest, GetInstance_ThreadSafe) {
    std::vector<MemoryManager*> instances(10, nullptr);
    std::vector<std::thread> threads;

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&instances, i]() {
            instances[i] = &MemoryManager::getInstance();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All instances should be the same
    for (int i = 1; i < 10; ++i) {
        EXPECT_EQ(instances[0], instances[i]);
    }
}

// =============================================================================
// CPU Memory Tests (MM-003 ~ MM-004)
// =============================================================================

// MM-003: GetCPUMemoryUsed returns positive value
TEST_F(MemoryManagerTest, GetCPUMemoryUsed_Positive) {
    MemoryManager& mm = MemoryManager::getInstance();
    size_t used = mm.getCPUMemoryUsed();

    // On Windows, this should return > 0
#ifdef _WIN32
    EXPECT_GT(used, 0u);
#else
    // On other platforms, it returns 0 as placeholder
    EXPECT_GE(used, 0u);
#endif
}

// MM-004: GetCPUMemoryAvailable returns positive value
TEST_F(MemoryManagerTest, GetCPUMemoryAvailable_Positive) {
    MemoryManager& mm = MemoryManager::getInstance();
    size_t available = mm.getCPUMemoryAvailable();

    // On Windows, this should return > 0
#ifdef _WIN32
    EXPECT_GT(available, 0u);
#else
    // On other platforms, it returns 0 as placeholder
    EXPECT_GE(available, 0u);
#endif
}

// MM-016: GetCPUMemoryUsed exception handled
TEST_F(MemoryManagerTest, GetCPUMemoryUsed_ExceptionHandled) {
    MemoryManager& mm = MemoryManager::getInstance();

    // Should not throw, returns 0 on failure
    EXPECT_NO_THROW({
        size_t used = mm.getCPUMemoryUsed();
        (void)used;
    });
}

// MM-017: GetCPUMemoryAvailable exception handled
TEST_F(MemoryManagerTest, GetCPUMemoryAvailable_ExceptionHandled) {
    MemoryManager& mm = MemoryManager::getInstance();

    // Should not throw, returns 0 on failure
    EXPECT_NO_THROW({
        size_t available = mm.getCPUMemoryAvailable();
        (void)available;
    });
}

// =============================================================================
// GPU Memory Tests - No CUDA (MM-005 ~ MM-008)
// =============================================================================

#ifndef USE_CUDA
// MM-005: GetGPUMemoryUsed returns 0 without CUDA
TEST_F(MemoryManagerTest, GetGPUMemoryUsed_NoCUDA) {
    MemoryManager& mm = MemoryManager::getInstance();
    float used = mm.getGPUMemoryUsed();

    EXPECT_EQ(0.0f, used);
}

// MM-006: GetGPUMemoryTotal returns 0 without CUDA
TEST_F(MemoryManagerTest, GetGPUMemoryTotal_NoCUDA) {
    MemoryManager& mm = MemoryManager::getInstance();
    float total = mm.getGPUMemoryTotal();

    EXPECT_EQ(0.0f, total);
}

// MM-007: GetGPUMemoryUsagePercent returns 0 without CUDA
TEST_F(MemoryManagerTest, GetGPUMemoryUsagePercent_NoCUDA) {
    MemoryManager& mm = MemoryManager::getInstance();
    float percent = mm.getGPUMemoryUsagePercent();

    EXPECT_EQ(0.0f, percent);
}

// MM-008: HasEnoughGPUMemory returns false without CUDA
TEST_F(MemoryManagerTest, HasEnoughGPUMemory_NoCUDA) {
    MemoryManager& mm = MemoryManager::getInstance();
    bool hasEnough = mm.hasEnoughGPUMemory(1024);

    EXPECT_FALSE(hasEnough);
}
#endif

// =============================================================================
// GPU Memory Tests - With CUDA (MM-009 ~ MM-015)
// =============================================================================

#ifdef USE_CUDA
// MM-009: GetGPUMemoryUsed returns valid value with CUDA
TEST_F(MemoryManagerTest, CUDA_GetGPUMemoryUsed_Valid) {
    MemoryManager& mm = MemoryManager::getInstance();
    float used = mm.getGPUMemoryUsed();

    // With CUDA available, should return >= 0
    EXPECT_GE(used, 0.0f);
}

// MM-010: GetGPUMemoryTotal returns valid value with CUDA
TEST_F(MemoryManagerTest, CUDA_GetGPUMemoryTotal_Valid) {
    MemoryManager& mm = MemoryManager::getInstance();
    float total = mm.getGPUMemoryTotal();

    // With CUDA available, should return > 0
    EXPECT_GT(total, 0.0f);
}

// MM-011: GetGPUMemoryUsagePercent returns value in range 0-100
TEST_F(MemoryManagerTest, CUDA_GetGPUMemoryUsagePercent_Range) {
    MemoryManager& mm = MemoryManager::getInstance();
    float percent = mm.getGPUMemoryUsagePercent();

    EXPECT_GE(percent, 0.0f);
    EXPECT_LE(percent, 100.0f);
}

// MM-012: HasEnoughGPUMemory returns true for small request
TEST_F(MemoryManagerTest, CUDA_HasEnoughGPUMemory_True) {
    MemoryManager& mm = MemoryManager::getInstance();

    // Debug: Print actual values to understand the issue
    float totalMem = mm.getGPUMemoryTotal();
    float usedMem = mm.getGPUMemoryUsed();
    float percent = mm.getGPUMemoryUsagePercent();

    std::cout << "[DEBUG] GPU Total: " << totalMem << " MB" << std::endl;
    std::cout << "[DEBUG] GPU Used: " << usedMem << " MB" << std::endl;
    std::cout << "[DEBUG] GPU Percent: " << percent << "%" << std::endl;

    // Request 1KB - should be available
    bool hasEnough = mm.hasEnoughGPUMemory(1024);
    std::cout << "[DEBUG] hasEnoughGPUMemory(1024): " << (hasEnough ? "true" : "false") << std::endl;

    // If we have valid total memory, 1KB should definitely be available
    if (totalMem > 0.0f) {
        EXPECT_TRUE(hasEnough) << "Total GPU memory is " << totalMem
                               << " MB but hasEnoughGPUMemory(1024) returned false";
    }
}

// MM-013: HasEnoughGPUMemory returns false for excessive request
TEST_F(MemoryManagerTest, CUDA_HasEnoughGPUMemory_False) {
    MemoryManager& mm = MemoryManager::getInstance();

    // Request 1 petabyte - should not be available
    size_t excessiveRequest = 1ULL << 50;  // 1 PB
    bool hasEnough = mm.hasEnoughGPUMemory(excessiveRequest);
    EXPECT_FALSE(hasEnough);
}

// MM-014: Invalid device index returns 0
TEST_F(MemoryManagerTest, CUDA_InvalidDeviceIndex_Negative) {
    MemoryManager& mm = MemoryManager::getInstance();

    float used = mm.getGPUMemoryUsed(-1);
    float total = mm.getGPUMemoryTotal(-1);
    float percent = mm.getGPUMemoryUsagePercent(-1);
    bool hasEnough = mm.hasEnoughGPUMemory(1024, -1);

    EXPECT_EQ(0.0f, used);
    EXPECT_EQ(0.0f, total);
    EXPECT_EQ(0.0f, percent);
    EXPECT_FALSE(hasEnough);
}

// MM-015: Device index out of range returns 0
TEST_F(MemoryManagerTest, CUDA_DeviceIndexOutOfRange) {
    MemoryManager& mm = MemoryManager::getInstance();

    // Use very high device index
    int outOfRangeDevice = 999;

    float used = mm.getGPUMemoryUsed(outOfRangeDevice);
    float total = mm.getGPUMemoryTotal(outOfRangeDevice);
    float percent = mm.getGPUMemoryUsagePercent(outOfRangeDevice);
    bool hasEnough = mm.hasEnoughGPUMemory(1024, outOfRangeDevice);

    EXPECT_EQ(0.0f, used);
    EXPECT_EQ(0.0f, total);
    EXPECT_EQ(0.0f, percent);
    EXPECT_FALSE(hasEnough);
}
#endif

// =============================================================================
// Memory Consistency Tests
// =============================================================================

TEST_F(MemoryManagerTest, CPUMemory_UsedPlusAvailable_Reasonable) {
    MemoryManager& mm = MemoryManager::getInstance();

#ifdef _WIN32
    size_t used = mm.getCPUMemoryUsed();
    size_t available = mm.getCPUMemoryAvailable();

    // Both should be positive
    EXPECT_GT(used, 0u);
    EXPECT_GT(available, 0u);

    // Total should be reasonable (at least 1GB, probably)
    size_t total = used + available;
    EXPECT_GT(total, 1024ULL * 1024 * 1024);  // > 1GB
#else
    SUCCEED();  // Skip on non-Windows
#endif
}

TEST_F(MemoryManagerTest, MultipleReads_Consistent) {
    MemoryManager& mm = MemoryManager::getInstance();

    // Read multiple times - values should be relatively stable
    size_t used1 = mm.getCPUMemoryUsed();
    size_t used2 = mm.getCPUMemoryUsed();
    size_t used3 = mm.getCPUMemoryUsed();

    // Values should be within 10% of each other (memory can fluctuate)
#ifdef _WIN32
    if (used1 > 0) {
        double ratio12 = static_cast<double>(used2) / used1;
        double ratio13 = static_cast<double>(used3) / used1;

        EXPECT_GT(ratio12, 0.9);
        EXPECT_LT(ratio12, 1.1);
        EXPECT_GT(ratio13, 0.9);
        EXPECT_LT(ratio13, 1.1);
    }
#else
    SUCCEED();
#endif
}
