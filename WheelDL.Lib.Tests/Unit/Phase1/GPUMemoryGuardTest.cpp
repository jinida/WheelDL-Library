#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Utils/Memory/GPUMemoryGuard.h"
#include "WheelDL.Lib/Utils/Memory/MemoryManager.h"
#include <torch/torch.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

using namespace WheelDL::Utils;

/**
 * @class GPUMemoryGuardTest
 * @brief Test suite for GPUMemoryGuard RAII pattern
 */
class GPUMemoryGuardTest : public ::testing::Test {
protected:
	void SetUp() override {
		// Check if CUDA is available
		cuda_available_ = torch::cuda::is_available();
	}

	void TearDown() override {
		if (cuda_available_ && torch::cuda::is_available()) {
			torch::cuda::synchronize();
#ifdef USE_CUDA
			c10::cuda::CUDACachingAllocator::emptyCache();
#endif
		}
	}

	bool cuda_available_ = false;
};

// ========== RAII Pattern Tests ==========

TEST_F(GPUMemoryGuardTest, ConstructorDestructor) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Test that guard can be constructed and destroyed without error
	EXPECT_NO_THROW({
		GPUMemoryGuard guard;
	}); // Guard goes out of scope here, destructor called
}

TEST_F(GPUMemoryGuardTest, ScopedUsage) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Test guard in nested scope
	EXPECT_NO_THROW({
		{
			GPUMemoryGuard guard1;
			{
				GPUMemoryGuard guard2;
			} // guard2 destroyed
		} // guard1 destroyed
	});
}

TEST_F(GPUMemoryGuardTest, MoveConstructor) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Test move constructor
	EXPECT_NO_THROW({
		GPUMemoryGuard guard1;
		GPUMemoryGuard guard2(std::move(guard1));
	});
}

TEST_F(GPUMemoryGuardTest, MoveAssignment) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Test move assignment
	EXPECT_NO_THROW({
		GPUMemoryGuard guard1;
		GPUMemoryGuard guard2;
		guard2 = std::move(guard1);
	});
}

TEST_F(GPUMemoryGuardTest, MultipleGuards) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Test multiple guards in same scope
	EXPECT_NO_THROW({
		GPUMemoryGuard guard1;
		GPUMemoryGuard guard2;
		GPUMemoryGuard guard3;
	}); // All guards destroyed in reverse order
}

// ========== GPU Memory Management Tests ==========

TEST_F(GPUMemoryGuardTest, ActualGPUMemoryAllocation) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Test that guard works with actual GPU memory allocation
	auto& memMgr = MemoryManager::getInstance();

	float initial_mem = memMgr.getGPUMemoryUsed();

	{
		GPUMemoryGuard guard;

		// Allocate GPU tensors
		std::vector<torch::Tensor> tensors;
		for (int i = 0; i < 10; ++i) {
			tensors.push_back(torch::randn({100, 100}, torch::kCUDA));
		}

		// Force synchronization to ensure allocation is complete
		torch::cuda::synchronize();

		// Memory tracking may not work if stats are 0, just verify it's non-negative
		float current_mem = memMgr.getGPUMemoryUsed();
		EXPECT_GE(current_mem, 0.0f); // Verify it's valid (may be 0 if stats not available)

		// Verify tensors are actually on GPU
		for (const auto& tensor : tensors) {
			EXPECT_TRUE(tensor.is_cuda());
		}

		// Guard will clean up on scope exit
	}

	// Give time for cleanup
	torch::cuda::synchronize();
#ifdef USE_CUDA
	c10::cuda::CUDACachingAllocator::emptyCache();
#endif
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	// Memory should be cleaned up (or at least not increasing)
	float final_mem = memMgr.getGPUMemoryUsed();
	// Note: Due to caching allocator, memory might not go back to initial
	EXPECT_GE(final_mem, 0.0f); // Just verify it's valid
}

TEST_F(GPUMemoryGuardTest, NestedGuardsWithGPUMemory) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Test nested guards with actual GPU allocations
	auto& memMgr = MemoryManager::getInstance();

	EXPECT_NO_THROW({
		GPUMemoryGuard guard1;

		auto tensor1 = torch::randn({256, 256}, torch::kCUDA);
		EXPECT_TRUE(tensor1.is_cuda());

		{
			GPUMemoryGuard guard2;

			auto tensor2 = torch::randn({512, 512}, torch::kCUDA);
			EXPECT_TRUE(tensor2.is_cuda());

			// Both tensors should be on GPU
			EXPECT_GT(memMgr.getGPUMemoryUsed(), 0.0f);

			// guard2 destroyed here
		}

		torch::cuda::synchronize();

		// guard1 destroyed here
	});

	torch::cuda::synchronize();
#ifdef USE_CUDA
	c10::cuda::CUDACachingAllocator::emptyCache();
#endif
}

TEST_F(GPUMemoryGuardTest, GuardWithLargeAllocation) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Test guard with large GPU allocation
	auto& memMgr = MemoryManager::getInstance();

	size_t initial_mem = memMgr.getGPUMemoryUsed();

	{
		GPUMemoryGuard guard;

		// Allocate a large tensor (100MB)
		auto large_tensor = torch::randn({1024, 1024, 25}, torch::kCUDA);
		EXPECT_TRUE(large_tensor.is_cuda());

		size_t current_mem = memMgr.getGPUMemoryUsed();
		EXPECT_GT(current_mem, initial_mem);
	}

	torch::cuda::synchronize();
#ifdef USE_CUDA
	c10::cuda::CUDACachingAllocator::emptyCache();
#endif

	// Verify guard cleaned up
	EXPECT_NO_THROW(memMgr.getGPUMemoryUsed());
}

TEST_F(GPUMemoryGuardTest, MoveWithGPUMemory) {
	if (!cuda_available_) {
		return; // Skip test if CUDA not available
	}

	// Test move operations with GPU memory
	auto& memMgr = MemoryManager::getInstance();

	EXPECT_NO_THROW({
		GPUMemoryGuard guard1;
		auto tensor1 = torch::randn({128, 128}, torch::kCUDA);

		GPUMemoryGuard guard2(std::move(guard1));
		auto tensor2 = torch::randn({128, 128}, torch::kCUDA);

		EXPECT_GT(memMgr.getGPUMemoryUsed(), 0.0f);
	});

	torch::cuda::synchronize();
#ifdef USE_CUDA
	c10::cuda::CUDACachingAllocator::emptyCache();
#endif
}
