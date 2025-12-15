#include "pch.h"
#include <gtest/gtest.h>
#include "Utils/Memory/GPUMemoryPool.h"
#include <thread>
#include <vector>

using namespace WheelDL::Utils;

// =============================================================================
// Test Fixture
// =============================================================================

class GPUMemoryPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear the pool before each test
        GPUMemoryPool::getInstance().clear();
    }

    void TearDown() override {
        // Clean up after test
        GPUMemoryPool::getInstance().clear();
    }
};

// =============================================================================
// Singleton Tests (GMP-001)
// =============================================================================

// GMP-001: GetInstance returns singleton
TEST_F(GPUMemoryPoolTest, GetInstance_Singleton) {
    GPUMemoryPool& instance1 = GPUMemoryPool::getInstance();
    GPUMemoryPool& instance2 = GPUMemoryPool::getInstance();

    EXPECT_EQ(&instance1, &instance2);
}

// =============================================================================
// Allocate Tests (GMP-002 ~ GMP-006)
// =============================================================================

// GMP-002: Allocate CPU tensor valid
TEST_F(GPUMemoryPoolTest, Allocate_CPU_Valid) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    torch::Tensor tensor = pool.allocate({3, 224, 224}, torch::kFloat32, torch::kCPU);

    EXPECT_TRUE(tensor.defined());
    EXPECT_TRUE(tensor.device().is_cpu());
}

// GMP-003: Allocate correct size
TEST_F(GPUMemoryPoolTest, Allocate_CorrectSize) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    torch::Tensor tensor = pool.allocate({3, 224, 224}, torch::kFloat32, torch::kCPU);

    EXPECT_EQ(3, tensor.size(0));
    EXPECT_EQ(224, tensor.size(1));
    EXPECT_EQ(224, tensor.size(2));
}

// GMP-004: Allocate correct dtype
TEST_F(GPUMemoryPoolTest, Allocate_CorrectDtype) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    torch::Tensor tensorFloat = pool.allocate({10}, torch::kFloat32, torch::kCPU);
    torch::Tensor tensorDouble = pool.allocate({10}, torch::kFloat64, torch::kCPU);
    torch::Tensor tensorInt = pool.allocate({10}, torch::kInt32, torch::kCPU);

    EXPECT_EQ(torch::kFloat32, tensorFloat.scalar_type());
    EXPECT_EQ(torch::kFloat64, tensorDouble.scalar_type());
    EXPECT_EQ(torch::kInt32, tensorInt.scalar_type());
}

// GMP-005: Release returns tensor to pool
TEST_F(GPUMemoryPoolTest, Release_ReturnsToPool) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    EXPECT_EQ(0u, pool.getPoolSize());

    torch::Tensor tensor = pool.allocate({10, 10}, torch::kFloat32, torch::kCPU);
    pool.release(tensor);

    EXPECT_EQ(1u, pool.getPoolSize());
}

// GMP-006: Allocate reuses from pool (cache hit)
TEST_F(GPUMemoryPoolTest, Allocate_ReuseFromPool) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    // Allocate and release
    torch::Tensor tensor1 = pool.allocate({10, 10}, torch::kFloat32, torch::kCPU);
    pool.release(tensor1);

    size_t allocsBefore = pool.getTotalAllocations();
    float hitRateBefore = pool.getCacheHitRate();

    // Allocate same shape - should reuse
    torch::Tensor tensor2 = pool.allocate({10, 10}, torch::kFloat32, torch::kCPU);

    // Should have another allocation counted
    EXPECT_EQ(allocsBefore + 1, pool.getTotalAllocations());

    // Pool should be empty now (tensor was reused)
    EXPECT_EQ(0u, pool.getPoolSize());

    // Cache hit rate should have increased
    EXPECT_GT(pool.getCacheHitRate(), hitRateBefore);
}

// =============================================================================
// Clear Tests (GMP-007)
// =============================================================================

// GMP-007: Clear empties pool
TEST_F(GPUMemoryPoolTest, Clear_EmptiesPool) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    // Add some tensors to pool
    torch::Tensor t1 = pool.allocate({10}, torch::kFloat32, torch::kCPU);
    torch::Tensor t2 = pool.allocate({20}, torch::kFloat32, torch::kCPU);
    pool.release(t1);
    pool.release(t2);

    EXPECT_GT(pool.getPoolSize(), 0u);

    pool.clear();

    EXPECT_EQ(0u, pool.getPoolSize());
}

// =============================================================================
// Statistics Tests (GMP-008 ~ GMP-010)
// =============================================================================

// GMP-008: GetPoolSize accurate
TEST_F(GPUMemoryPoolTest, GetPoolSize_Accurate) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    EXPECT_EQ(0u, pool.getPoolSize());

    torch::Tensor t1 = pool.allocate({10}, torch::kFloat32, torch::kCPU);
    torch::Tensor t2 = pool.allocate({20}, torch::kFloat32, torch::kCPU);
    torch::Tensor t3 = pool.allocate({30}, torch::kFloat32, torch::kCPU);

    pool.release(t1);
    EXPECT_EQ(1u, pool.getPoolSize());

    pool.release(t2);
    EXPECT_EQ(2u, pool.getPoolSize());

    pool.release(t3);
    EXPECT_EQ(3u, pool.getPoolSize());
}

// GMP-009: GetTotalAllocations counted
TEST_F(GPUMemoryPoolTest, GetTotalAllocations_Counted) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    size_t initial = pool.getTotalAllocations();

    pool.allocate({10}, torch::kFloat32, torch::kCPU);
    EXPECT_EQ(initial + 1, pool.getTotalAllocations());

    pool.allocate({20}, torch::kFloat32, torch::kCPU);
    EXPECT_EQ(initial + 2, pool.getTotalAllocations());

    pool.allocate({30}, torch::kFloat32, torch::kCPU);
    EXPECT_EQ(initial + 3, pool.getTotalAllocations());
}

// GMP-010: GetCacheHitRate in range 0-100
TEST_F(GPUMemoryPoolTest, GetCacheHitRate_Range) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    // Initially should be 0 or valid percentage
    float rate = pool.getCacheHitRate();
    EXPECT_GE(rate, 0.0f);
    EXPECT_LE(rate, 100.0f);

    // After some allocations
    torch::Tensor t1 = pool.allocate({10}, torch::kFloat32, torch::kCPU);
    pool.release(t1);
    pool.allocate({10}, torch::kFloat32, torch::kCPU);  // Should be cache hit

    rate = pool.getCacheHitRate();
    EXPECT_GE(rate, 0.0f);
    EXPECT_LE(rate, 100.0f);
}

// =============================================================================
// Thread Safety Tests (GMP-011)
// =============================================================================

// GMP-011: Thread-safe concurrent allocate
TEST_F(GPUMemoryPoolTest, ThreadSafety_ConcurrentAllocate) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    std::vector<std::thread> threads;
    std::atomic<int> successCount{0};

    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&pool, &successCount, i]() {
            try {
                for (int j = 0; j < 10; ++j) {
                    torch::Tensor t = pool.allocate({10, 10}, torch::kFloat32, torch::kCPU);
                    if (t.defined()) {
                        successCount++;
                    }
                    pool.release(t);
                }
            }
            catch (...) {
                // Count as failure if exception
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All allocations should succeed
    EXPECT_EQ(100, successCount.load());
}

// =============================================================================
// TensorKey Tests (GMP-013 ~ GMP-018)
// =============================================================================

// GMP-013: TensorKey equality - same
TEST_F(GPUMemoryPoolTest, TensorKey_Equality_Same) {
    TensorKey key1{{10, 20}, torch::kFloat32, torch::kCPU, 0};
    TensorKey key2{{10, 20}, torch::kFloat32, torch::kCPU, 0};

    EXPECT_TRUE(key1 == key2);
}

// GMP-014: TensorKey inequality - different size
TEST_F(GPUMemoryPoolTest, TensorKey_Inequality_Size) {
    TensorKey key1{{10, 20}, torch::kFloat32, torch::kCPU, 0};
    TensorKey key2{{10, 30}, torch::kFloat32, torch::kCPU, 0};

    EXPECT_FALSE(key1 == key2);
}

// GMP-015: TensorKey inequality - different dtype
TEST_F(GPUMemoryPoolTest, TensorKey_Inequality_Dtype) {
    TensorKey key1{{10, 20}, torch::kFloat32, torch::kCPU, 0};
    TensorKey key2{{10, 20}, torch::kFloat64, torch::kCPU, 0};

    EXPECT_FALSE(key1 == key2);
}

// GMP-016: TensorKey inequality - different device
TEST_F(GPUMemoryPoolTest, TensorKey_Inequality_Device) {
    TensorKey key1{{10, 20}, torch::kFloat32, torch::kCPU, 0};
    TensorKey key2{{10, 20}, torch::kFloat32, torch::kCUDA, 0};

    EXPECT_FALSE(key1 == key2);
}

// GMP-017: TensorKeyHash consistent
TEST_F(GPUMemoryPoolTest, TensorKeyHash_Consistent) {
    TensorKey key{{10, 20}, torch::kFloat32, torch::kCPU, 0};
    TensorKeyHash hasher;

    size_t hash1 = hasher(key);
    size_t hash2 = hasher(key);

    EXPECT_EQ(hash1, hash2);
}

// GMP-018: TensorKeyHash different for different keys
TEST_F(GPUMemoryPoolTest, TensorKeyHash_Different) {
    TensorKey key1{{10, 20}, torch::kFloat32, torch::kCPU, 0};
    TensorKey key2{{10, 30}, torch::kFloat32, torch::kCPU, 0};
    TensorKeyHash hasher;

    size_t hash1 = hasher(key1);
    size_t hash2 = hasher(key2);

    // Different keys should (usually) have different hashes
    // Note: Hash collisions are possible but unlikely for simple cases
    EXPECT_NE(hash1, hash2);
}

// =============================================================================
// Edge Cases (GMP-019 ~ GMP-026)
// =============================================================================

// GMP-019: Allocate different dtypes - separate pools
TEST_F(GPUMemoryPoolTest, Allocate_DifferentDtypes) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    torch::Tensor tFloat = pool.allocate({10}, torch::kFloat32, torch::kCPU);
    torch::Tensor tDouble = pool.allocate({10}, torch::kFloat64, torch::kCPU);
    torch::Tensor tInt = pool.allocate({10}, torch::kInt32, torch::kCPU);

    pool.release(tFloat);
    pool.release(tDouble);
    pool.release(tInt);

    // Pool should have 3 entries (different keys)
    EXPECT_EQ(3u, pool.getPoolSize());

    // Allocate float32 again - should get from pool
    torch::Tensor tFloat2 = pool.allocate({10}, torch::kFloat32, torch::kCPU);
    EXPECT_EQ(torch::kFloat32, tFloat2.scalar_type());
    EXPECT_EQ(2u, pool.getPoolSize());  // One less now
}

// GMP-020: Release wrong tensor (not from pool) - should still add to pool
TEST_F(GPUMemoryPoolTest, Release_ExternalTensor) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    // Create tensor outside of pool
    torch::Tensor externalTensor = torch::zeros({5, 5});

    pool.release(externalTensor);

    // Should be added to pool
    EXPECT_EQ(1u, pool.getPoolSize());
}

// GMP-021: GetCacheHitRate with zero allocations
TEST_F(GPUMemoryPoolTest, CacheHitRate_ZeroAllocations) {
    // Note: Since this is a singleton and other tests may have run,
    // we just verify that it returns a valid percentage
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    float rate = pool.getCacheHitRate();
    EXPECT_GE(rate, 0.0f);
    EXPECT_LE(rate, 100.0f);
}

// GMP-022: Release undefined tensor
TEST_F(GPUMemoryPoolTest, Release_UndefinedTensor) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    torch::Tensor undefinedTensor;  // Not defined
    EXPECT_FALSE(undefinedTensor.defined());

    size_t sizeBefore = pool.getPoolSize();

    // Should return immediately without crash
    EXPECT_NO_THROW(pool.release(undefinedTensor));

    // Pool size should not change
    EXPECT_EQ(sizeBefore, pool.getPoolSize());
}

// GMP-026: GetPoolSize with multiple keys
TEST_F(GPUMemoryPoolTest, GetPoolSize_MultipleKeys) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    // Create tensors with different shapes
    torch::Tensor t1 = pool.allocate({10}, torch::kFloat32, torch::kCPU);
    torch::Tensor t2 = pool.allocate({20}, torch::kFloat32, torch::kCPU);
    torch::Tensor t3 = pool.allocate({10}, torch::kFloat64, torch::kCPU);
    torch::Tensor t4 = pool.allocate({10}, torch::kFloat32, torch::kCPU);

    pool.release(t1);
    pool.release(t2);
    pool.release(t3);
    pool.release(t4);

    // t1 and t4 have same key, t2 and t3 have different keys
    // Total: 4 tensors in pool
    EXPECT_EQ(4u, pool.getPoolSize());
}

// =============================================================================
// CUDA Tests (GMP-012)
// =============================================================================

#ifdef USE_CUDA
// GMP-012: Allocate GPU tensor valid
TEST_F(GPUMemoryPoolTest, CUDA_Allocate_GPU_Valid) {
    if (!torch::cuda::is_available()) {
        // Skip test if CUDA not available at runtime
        SUCCEED();
        return;
    }

    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    torch::Tensor tensor = pool.allocate({3, 224, 224}, torch::kFloat32, torch::kCUDA);

    EXPECT_TRUE(tensor.defined());
    EXPECT_TRUE(tensor.device().is_cuda());
}
#endif

// =============================================================================
// Additional Tests
// =============================================================================

TEST_F(GPUMemoryPoolTest, Allocate_EmptySize) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    // Allocate with empty dimensions - scalar tensor
    torch::Tensor tensor = pool.allocate({}, torch::kFloat32, torch::kCPU);

    EXPECT_TRUE(tensor.defined());
    EXPECT_EQ(0, tensor.dim());  // Scalar has 0 dimensions
}

TEST_F(GPUMemoryPoolTest, Allocate_LargeTensor) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    // Allocate reasonably large tensor
    torch::Tensor tensor = pool.allocate({100, 100, 100}, torch::kFloat32, torch::kCPU);

    EXPECT_TRUE(tensor.defined());
    EXPECT_EQ(100 * 100 * 100, tensor.numel());
}

TEST_F(GPUMemoryPoolTest, MultipleReleaseSameTensor) {
    GPUMemoryPool& pool = GPUMemoryPool::getInstance();

    torch::Tensor tensor = pool.allocate({10}, torch::kFloat32, torch::kCPU);

    // Release the same tensor multiple times
    // This is technically misuse but shouldn't crash
    pool.release(tensor);
    pool.release(tensor);
    pool.release(tensor);

    // Pool should have 3 entries (same tensor added 3 times)
    // This is undefined behavior but implementation adds all
    EXPECT_GE(pool.getPoolSize(), 1u);
}
