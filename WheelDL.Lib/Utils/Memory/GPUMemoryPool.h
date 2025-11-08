#pragma once

#include <torch/torch.h>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <functional>

namespace WheelDL {
namespace Utils {

/**
 * @struct TensorKey
 * @brief Key for tensor pooling based on shape and dtype
 */
struct TensorKey {
	std::vector<int64_t> sizes;
	torch::ScalarType dtype;
	torch::DeviceType deviceType;
	int8_t deviceIndex;

	bool operator==(const TensorKey& other) const {
		return sizes == other.sizes &&
		       dtype == other.dtype &&
		       deviceType == other.deviceType &&
		       deviceIndex == other.deviceIndex;
	}
};

/**
 * @struct TensorKeyHash
 * @brief Hash function for TensorKey to use in unordered_map
 */
struct TensorKeyHash {
	std::size_t operator()(const TensorKey& key) const {
		std::size_t hash = 0;
		for (auto size : key.sizes) {
			hash ^= std::hash<int64_t>{}(size) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
		}
		hash ^= std::hash<int>{}(static_cast<int>(key.dtype));
		hash ^= std::hash<int>{}(static_cast<int>(key.deviceType));
		hash ^= std::hash<int>{}(static_cast<int>(key.deviceIndex));
		return hash;
	}
};

/**
 * @class GPUMemoryPool
 * @brief Singleton Object Pool for tensor reuse
 *
 * Manages a pool of tensors for reuse to avoid frequent allocation/deallocation.
 * Uses Singleton pattern for global access and Object Pool pattern for tensor management.
 */
class GPUMemoryPool {
public:
	/**
	 * @brief Get singleton instance
	 * @return GPUMemoryPool& Singleton instance
	 */
	static GPUMemoryPool& getInstance();

	/**
	 * @brief Allocate tensor from pool or create new one
	 * @param sizes Tensor dimensions
	 * @param dtype Tensor data type
	 * @param device Torch device (CPU or CUDA)
	 * @return torch::Tensor Allocated tensor
	 */
	torch::Tensor allocate(const std::vector<int64_t>& sizes,
	                       torch::ScalarType dtype = torch::kFloat32,
	                       const torch::Device& device = torch::kCPU);

	/**
	 * @brief Return tensor to pool for reuse
	 * @param tensor Tensor to release back to pool
	 */
	void release(torch::Tensor tensor);

	/**
	 * @brief Clear all pooled tensors
	 */
	void clear();

	/**
	 * @brief Get current pool size
	 * @return size_t Number of tensors in pool
	 */
	size_t getPoolSize() const;

	/**
	 * @brief Get total number of allocations
	 * @return size_t Total allocations made
	 */
	size_t getTotalAllocations() const;

	/**
	 * @brief Get cache hit rate
	 * @return float Hit rate percentage (0.0 - 100.0)
	 */
	float getCacheHitRate() const;

private:
	GPUMemoryPool();
	~GPUMemoryPool();

	// Delete copy constructor and assignment operator
	GPUMemoryPool(const GPUMemoryPool&) = delete;
	GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;

	// Thread-safe pool storage
	mutable std::mutex pool_mutex_;
	std::unordered_map<TensorKey, std::queue<torch::Tensor>, TensorKeyHash> tensor_pool_;

	// Statistics (atomic for lock-free updates)
	std::atomic<size_t> total_allocations_;
	std::atomic<size_t> cache_hits_;
};

} // namespace Utils
} // namespace WheelDL
