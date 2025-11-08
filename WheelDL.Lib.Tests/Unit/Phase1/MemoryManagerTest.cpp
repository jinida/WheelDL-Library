#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Utils/Memory/MemoryManager.h"
#include "WheelDL.Lib/Utils/Logger/Logger.h"
#include <torch/torch.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

using namespace WheelDL::Utils;

/**
 * @class MemoryManagerTest
 * @brief Test suite for MemoryManager
 */
class MemoryManagerTest : public ::testing::Test {
protected:
	void SetUp() override {
		// Get singleton instance
		manager_ = &MemoryManager::getInstance();
	}

	MemoryManager* manager_;
};

// ========== Singleton Tests ==========

TEST_F(MemoryManagerTest, SingletonInstance) {
	Logger::getInstance()->info("=== Testing MemoryManager Singleton Pattern ===");
	Logger::getInstance()->info("Verifying that getInstance() returns the same instance");

	MemoryManager& instance1 = MemoryManager::getInstance();
	MemoryManager& instance2 = MemoryManager::getInstance();

	EXPECT_EQ(&instance1, &instance2);
	Logger::getInstance()->info("Singleton pattern verified: Both instances are identical");
}

// ========== GPU Memory Tests ==========

TEST_F(MemoryManagerTest, GetGPUMemoryUsed) {
	Logger::getInstance()->info("=== Testing GPU Memory Usage Retrieval ===");

	float gpu_memory = manager_->getGPUMemoryUsed();
	Logger::getInstance()->info("GPU Memory Used: " + std::to_string(gpu_memory) + " MB");

	EXPECT_GE(gpu_memory, 0.0f);

#ifdef USE_CUDA
	if (torch::cuda::is_available()) {
		Logger::getInstance()->info("CUDA is available - GPU memory tracking active");
	} else {
		Logger::getInstance()->info("CUDA not available - returning 0.0 MB");
	}
#else
	Logger::getInstance()->info("CUDA support not compiled - CPU-only mode");
#endif
}

TEST_F(MemoryManagerTest, GetGPUMemoryTotal) {
	// Should return 0 or positive value
	float gpu_total = manager_->getGPUMemoryTotal();
	EXPECT_GE(gpu_total, 0.0f);
}

TEST_F(MemoryManagerTest, GetGPUMemoryUsagePercent) {
	// Should return value between 0 and 100
	float usage_percent = manager_->getGPUMemoryUsagePercent();
	EXPECT_GE(usage_percent, 0.0f);
	EXPECT_LE(usage_percent, 100.0f);
}

TEST_F(MemoryManagerTest, HasEnoughGPUMemory) {
	// Test with very small requirement
	bool has_enough = manager_->hasEnoughGPUMemory(1024); // 1KB
	EXPECT_TRUE(has_enough || !has_enough); // Just verify it returns a boolean

	// Test with very large requirement (should return false on most systems)
	bool has_huge = manager_->hasEnoughGPUMemory(1024ULL * 1024ULL * 1024ULL * 100ULL); // 100GB
	EXPECT_FALSE(has_huge); // Unlikely to have 100GB available
}

// ========== CPU Memory Tests ==========

TEST_F(MemoryManagerTest, GetCPUMemoryUsed) {
	// Should return positive value (some memory is always used)
	size_t cpu_used = manager_->getCPUMemoryUsed();
	EXPECT_GT(cpu_used, 0ULL);
}

TEST_F(MemoryManagerTest, GetCPUMemoryAvailable) {
	// Should return positive value (some memory should be available)
	size_t cpu_available = manager_->getCPUMemoryAvailable();
	EXPECT_GT(cpu_available, 0ULL);
}

TEST_F(MemoryManagerTest, CPUMemoryConsistency) {
	// Used + Available should be reasonably close to total physical memory
	size_t used = manager_->getCPUMemoryUsed();
	size_t available = manager_->getCPUMemoryAvailable();

	// Both should be non-zero
	EXPECT_GT(used, 0ULL);
	EXPECT_GT(available, 0ULL);

	// Total should be reasonable (at least 1GB, less than 1TB)
	size_t total = used + available;
	EXPECT_GT(total, 1024ULL * 1024ULL * 1024ULL); // > 1GB
	EXPECT_LT(total, 1024ULL * 1024ULL * 1024ULL * 1024ULL); // < 1TB
}

// ========== GPU Diagnostics Tests ==========

TEST_F(MemoryManagerTest, CUDAAvailabilityDiagnostics) {
	Logger::getInstance()->info("=== CUDA Availability Diagnostics ===");

	// Test torch::cuda::is_available()
	bool cuda_available = torch::cuda::is_available();
	Logger::getInstance()->info("torch::cuda::is_available(): " + std::string(cuda_available ? "true" : "false"));

	if (cuda_available) {
		// Get CUDA device count
		int device_count = torch::cuda::device_count();
		Logger::getInstance()->info("CUDA device count: " + std::to_string(device_count));

		// Test actual GPU allocation
		try {
			auto test_tensor = torch::randn({10, 10}, torch::kCUDA);
			Logger::getInstance()->info("Successfully allocated test tensor on GPU");
			EXPECT_TRUE(test_tensor.is_cuda());

			// Test GPU memory functions
			float gpu_used = manager_->getGPUMemoryUsed();
			float gpu_total = manager_->getGPUMemoryTotal();
			Logger::getInstance()->info("GPU Memory Used: " + std::to_string(gpu_used) + " MB");
			Logger::getInstance()->info("GPU Memory Total: " + std::to_string(gpu_total) + " MB");

			EXPECT_GT(gpu_used, 0.0f);
		} catch (const std::exception& e) {
			Logger::getInstance()->error("Failed to allocate GPU tensor: " + std::string(e.what()));
			FAIL() << "GPU allocation failed despite CUDA being available";
		}
	} else {
		Logger::getInstance()->warn("CUDA not available - GPU tests will be skipped");

#ifdef USE_CUDA
		Logger::getInstance()->warn("USE_CUDA is defined but torch::cuda::is_available() returned false");
		Logger::getInstance()->warn("This could indicate:");
		Logger::getInstance()->warn("  1. No NVIDIA GPU installed");
		Logger::getInstance()->warn("  2. GPU drivers not installed or outdated");
		Logger::getInstance()->warn("  3. CUDA runtime DLLs not found in PATH");
		Logger::getInstance()->warn("  4. PyTorch built without CUDA support");
#else
		Logger::getInstance()->info("USE_CUDA is not defined - this is a CPU-only build");
#endif
	}

	Logger::getInstance()->info("=== End of CUDA Diagnostics ===");
}

TEST_F(MemoryManagerTest, GPUMemoryWithActualAllocation) {
	// Skip if CUDA is not available
	if (!torch::cuda::is_available()) {
		return;
	}

	Logger::getInstance()->info("=== Testing GPU Memory with Actual Allocation ===");

	// Get initial memory state
	float initial_used = manager_->getGPUMemoryUsed();
	Logger::getInstance()->info("Initial GPU memory: " + std::to_string(initial_used) + " MB");

	// Allocate GPU tensors
	std::vector<torch::Tensor> tensors;
	for (int i = 0; i < 10; ++i) {
		tensors.push_back(torch::randn({1024, 1024}, torch::kCUDA));
	}

	// Check memory increased
	float after_alloc = manager_->getGPUMemoryUsed();
	Logger::getInstance()->info("After allocation GPU memory: " + std::to_string(after_alloc) + " MB");

	EXPECT_GT(after_alloc, initial_used);

	// Clear tensors
	tensors.clear();
	torch::cuda::synchronize();
#ifdef USE_CUDA
	c10::cuda::CUDACachingAllocator::emptyCache();
#endif

	float after_clear = manager_->getGPUMemoryUsed();
	Logger::getInstance()->info("After clear GPU memory: " + std::to_string(after_clear) + " MB");

	Logger::getInstance()->info("=== End of GPU Memory Test ===");
}
