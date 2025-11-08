#include "pch.h"
#include "Utils/Memory/GPUMemoryPool.h"

namespace WheelDL {
	namespace Utils {

		GPUMemoryPool::GPUMemoryPool()
			: total_allocations_(0)
			, cache_hits_(0) {
		}

		GPUMemoryPool::~GPUMemoryPool() {
			clear();
		}

		GPUMemoryPool& GPUMemoryPool::getInstance() {
			static GPUMemoryPool instance;
			return instance;
		}

		torch::Tensor GPUMemoryPool::allocate(const std::vector<int64_t>& sizes,
			torch::ScalarType dtype,
			const torch::Device& device) {
			// Use CPU device type for key since we store tensors on CPU
			TensorKey key{ sizes, dtype, torch::kCPU, 0 };
			total_allocations_.fetch_add(1, std::memory_order_relaxed);

			// Try to get tensor from pool
			torch::Tensor pooledTensor;
			bool foundInPool = false;
			{
				std::lock_guard<std::mutex> lock(pool_mutex_);

				// Check if we have a pooled tensor available
				auto it = tensor_pool_.find(key);
				if (it != tensor_pool_.end() && !it->second.empty()) {
					// Reuse tensor from pool
					pooledTensor = it->second.front();
					it->second.pop();
					foundInPool = true;
					cache_hits_.fetch_add(1, std::memory_order_relaxed);
				}
			} // Lock released here

			if (foundInPool) {
				// Perform device transfer outside of lock for better performance
				try {
					torch::Tensor result = pooledTensor.to(device);
					return result;
				}
				catch (...) {
					// If device transfer fails, discard the failed tensor
					// Fall through to allocate new tensor
				}
			}

			// Allocate new tensor
			torch::TensorOptions options = torch::TensorOptions().dtype(dtype).device(device);
			return torch::empty(sizes, options);
		}

		void GPUMemoryPool::release(torch::Tensor tensor) {
			if (!tensor.defined()) {
				return;
			}

			std::lock_guard<std::mutex> lock(pool_mutex_);

			// Create key from tensor properties
			// Store on CPU to save GPU memory, so use CPU device type
			TensorKey key{ tensor.sizes().vec(), tensor.scalar_type(), torch::kCPU, 0 };

			// Add tensor to pool (move to CPU first)
			try {
				tensor_pool_[key].push(tensor.cpu());
			}
			catch (const std::exception& e) {
				// If CPU transfer fails, log and discard tensor
				// Better to lose one tensor than to crash
				// TODO: Add proper logging when logger is available
			}
		}

		void GPUMemoryPool::clear() {
			std::lock_guard<std::mutex> lock(pool_mutex_);
			tensor_pool_.clear();
		}

		size_t GPUMemoryPool::getPoolSize() const {
			std::lock_guard<std::mutex> lock(pool_mutex_);

			size_t total_size = 0;
			for (const auto& pair : tensor_pool_) {
				total_size += pair.second.size();
			}
			return total_size;
		}

		size_t GPUMemoryPool::getTotalAllocations() const {
			// No mutex needed - atomic variable access is thread-safe
			return total_allocations_.load(std::memory_order_relaxed);
		}

		float GPUMemoryPool::getCacheHitRate() const {
			// No mutex needed - atomic variables access is thread-safe
			size_t total = total_allocations_.load(std::memory_order_relaxed);
			if (total == 0) {
				return 0.0f;
			}
			size_t hits = cache_hits_.load(std::memory_order_relaxed);
			return (static_cast<float>(hits) / static_cast<float>(total)) * 100.0f;
		}

	} // namespace Utils
} // namespace WheelDL
