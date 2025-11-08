#pragma once

#include <cstddef>

namespace WheelDL {
namespace Utils {

/**
 * @class MemoryManager
 * @brief Singleton class for monitoring GPU and CPU memory usage
 *
 * Provides methods to query GPU and CPU memory statistics.
 * Uses LibTorch CUDA APIs for GPU memory monitoring.
 */
class MemoryManager {
public:
	/**
	 * @brief Get singleton instance
	 * @return MemoryManager& Singleton instance
	 */
	static MemoryManager& getInstance();

	// ========== GPU Memory ==========

	/**
	 * @brief Get current GPU memory used in MB
	 * @param deviceIndex GPU device index (default: 0)
	 * @return float GPU memory used in megabytes
	 */
	float getGPUMemoryUsed(int deviceIndex = 0) const;

	/**
	 * @brief Get total GPU memory in MB
	 * @param deviceIndex GPU device index (default: 0)
	 * @return float Total GPU memory in megabytes
	 */
	float getGPUMemoryTotal(int deviceIndex = 0) const;

	/**
	 * @brief Get GPU memory usage percentage
	 * @param deviceIndex GPU device index (default: 0)
	 * @return float Usage percentage (0.0 - 100.0)
	 */
	float getGPUMemoryUsagePercent(int deviceIndex = 0) const;

	/**
	 * @brief Check if enough GPU memory is available
	 * @param requiredBytes Required memory in bytes
	 * @param deviceIndex GPU device index (default: 0)
	 * @return bool True if enough memory is available
	 */
	bool hasEnoughGPUMemory(size_t requiredBytes, int deviceIndex = 0) const;

	// ========== CPU Memory ==========

	/**
	 * @brief Get current CPU memory used in bytes
	 * @return size_t CPU memory used
	 */
	size_t getCPUMemoryUsed() const;

	/**
	 * @brief Get available CPU memory in bytes
	 * @return size_t Available CPU memory
	 */
	size_t getCPUMemoryAvailable() const;

private:
	MemoryManager();
	~MemoryManager();

	// Delete copy constructor and assignment operator
	MemoryManager(const MemoryManager&) = delete;
	MemoryManager& operator=(const MemoryManager&) = delete;

#ifdef USE_CUDA
	// Helper struct to hold GPU memory stats
	struct GPUMemoryStats {
		size_t allocated;
		size_t total;
		bool valid;
	};

	// Helper function to get GPU memory stats
	GPUMemoryStats getGPUMemoryStats(int deviceIndex) const;
#endif
};

} // namespace Utils
} // namespace WheelDL
