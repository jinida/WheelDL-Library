#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Utils/Memory/GPUMemoryPool.h"
#include <torch/torch.h>

using namespace WheelDL::Utils;

/**
 * @class GPUMemoryPoolTest
 * @brief Test suite for GPUMemoryPool object pooling
 */
class GPUMemoryPoolTest : public ::testing::Test {
protected:
	void SetUp() override {
		pool_ = &GPUMemoryPool::getInstance();
		pool_->clear(); // Start with clean pool

		// Check if CUDA is available
		cuda_available_ = torch::cuda::is_available();
	}

	void TearDown() override {
		if (cuda_available_) {
			pool_->clear(); // Clean up after each test
			if (torch::cuda::is_available()) {
				torch::cuda::synchronize();
			}
		}
	}

	GPUMemoryPool* pool_;
	bool cuda_available_ = false;
};

// ========== Singleton Tests ==========

TEST_F(GPUMemoryPoolTest, SingletonInstance) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Test that getInstance returns the same instance
	GPUMemoryPool& instance1 = GPUMemoryPool::getInstance();
	GPUMemoryPool& instance2 = GPUMemoryPool::getInstance();

	EXPECT_EQ(&instance1, &instance2);
}

// ========== Allocation Tests ==========

TEST_F(GPUMemoryPoolTest, BasicAllocation) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Allocate a simple tensor on GPU
	std::vector<int64_t> sizes = {2, 3};
	torch::Tensor tensor = pool_->allocate(sizes, torch::kFloat32, torch::kCUDA);

	EXPECT_TRUE(tensor.defined());
	EXPECT_EQ(tensor.sizes(), torch::IntArrayRef(sizes));
	EXPECT_EQ(tensor.scalar_type(), torch::kFloat32);
	EXPECT_TRUE(tensor.is_cuda());
}

TEST_F(GPUMemoryPoolTest, MultipleAllocations) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Allocate multiple tensors with different sizes on GPU
	auto tensor1 = pool_->allocate({2, 3}, torch::kFloat32, torch::kCUDA);
	auto tensor2 = pool_->allocate({4, 5}, torch::kFloat32, torch::kCUDA);
	auto tensor3 = pool_->allocate({1, 1}, torch::kInt32, torch::kCUDA);

	EXPECT_TRUE(tensor1.defined());
	EXPECT_TRUE(tensor2.defined());
	EXPECT_TRUE(tensor3.defined());
	EXPECT_TRUE(tensor1.is_cuda());
	EXPECT_TRUE(tensor2.is_cuda());
	EXPECT_TRUE(tensor3.is_cuda());
}

// ========== Release and Reuse Tests ==========

TEST_F(GPUMemoryPoolTest, ReleaseAndReuse) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Allocate, release, and allocate again on GPU
	std::vector<int64_t> sizes = {3, 4};

	auto tensor1 = pool_->allocate(sizes, torch::kFloat32, torch::kCUDA);
	EXPECT_TRUE(tensor1.is_cuda());
	size_t first_alloc = pool_->getTotalAllocations();

	pool_->release(tensor1);
	EXPECT_GT(pool_->getPoolSize(), 0ULL);

	// Second allocation should reuse from pool
	auto tensor2 = pool_->allocate(sizes, torch::kFloat32, torch::kCUDA);
	EXPECT_TRUE(tensor2.is_cuda());
	size_t second_alloc = pool_->getTotalAllocations();

	EXPECT_EQ(second_alloc, first_alloc + 1);
	EXPECT_GT(pool_->getCacheHitRate(), 0.0f);
}

TEST_F(GPUMemoryPoolTest, DifferentSizesNoReuse) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Different sizes should not reuse tensors on GPU
	auto tensor1 = pool_->allocate({2, 3}, torch::kFloat32, torch::kCUDA);
	EXPECT_TRUE(tensor1.is_cuda());
	pool_->release(tensor1);

	// Different size, should allocate new
	auto tensor2 = pool_->allocate({4, 5}, torch::kFloat32, torch::kCUDA);
	EXPECT_TRUE(tensor2.is_cuda());

	// Pool should have both sizes now
	EXPECT_GE(pool_->getPoolSize(), 1ULL);
}

// ========== Pool Statistics Tests ==========

TEST_F(GPUMemoryPoolTest, GetPoolSize) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	EXPECT_EQ(pool_->getPoolSize(), 0ULL);

	auto tensor = pool_->allocate({2, 2}, torch::kFloat32, torch::kCUDA);
	EXPECT_TRUE(tensor.is_cuda());
	pool_->release(tensor);

	EXPECT_EQ(pool_->getPoolSize(), 1ULL);
}

TEST_F(GPUMemoryPoolTest, GetTotalAllocations) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	size_t initial = pool_->getTotalAllocations();

	auto t1 = pool_->allocate({2, 2}, torch::kFloat32, torch::kCUDA);
	EXPECT_TRUE(t1.is_cuda());
	EXPECT_EQ(pool_->getTotalAllocations(), initial + 1);

	auto t2 = pool_->allocate({3, 3}, torch::kFloat32, torch::kCUDA);
	EXPECT_TRUE(t2.is_cuda());
	EXPECT_EQ(pool_->getTotalAllocations(), initial + 2);
}

TEST_F(GPUMemoryPoolTest, GetCacheHitRate) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Clear pool to reset counters from previous tests
	pool_->clear();

	// Get a fresh pool instance
	GPUMemoryPool& fresh_pool = GPUMemoryPool::getInstance();

	std::vector<int64_t> sizes = {2, 2};

	// First allocation - miss
	auto tensor1 = fresh_pool.allocate(sizes, torch::kFloat32, torch::kCUDA);
	EXPECT_TRUE(tensor1.is_cuda());

	// Release and allocate again - hit
	fresh_pool.release(tensor1);
	auto tensor2 = fresh_pool.allocate(sizes, torch::kFloat32, torch::kCUDA);
	EXPECT_TRUE(tensor2.is_cuda());

	// Should have at least some hit rate (1/2 = 50%)
	EXPECT_GT(fresh_pool.getCacheHitRate(), 0.0f);
	EXPECT_LE(fresh_pool.getCacheHitRate(), 100.0f);
}

// ========== Clear Tests ==========

TEST_F(GPUMemoryPoolTest, Clear) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Add some tensors to pool on GPU
	auto tensor1 = pool_->allocate({2, 2}, torch::kFloat32, torch::kCUDA);
	auto tensor2 = pool_->allocate({3, 3}, torch::kFloat32, torch::kCUDA);
	EXPECT_TRUE(tensor1.is_cuda());
	EXPECT_TRUE(tensor2.is_cuda());
	pool_->release(tensor1);
	pool_->release(tensor2);

	EXPECT_GT(pool_->getPoolSize(), 0ULL);

	// Clear pool
	pool_->clear();
	EXPECT_EQ(pool_->getPoolSize(), 0ULL);
}
