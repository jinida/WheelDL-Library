#pragma once

namespace WheelDL {
namespace Utils {

/**
 * @class GPUMemoryGuard
 * @brief RAII guard for GPU memory management
 *
 * Automatically clears GPU memory cache when the object goes out of scope.
 * Uses RAII pattern to ensure proper GPU memory cleanup.
 */
class GPUMemoryGuard {
public:
	/**
	 * @brief Constructor (no-op)
	 */
	GPUMemoryGuard();

	/**
	 * @brief Destructor - calls empty_cache()
	 *
	 * Clears the GPU caching allocator to free up unused memory.
	 * This is called automatically when the guard goes out of scope.
	 */
	~GPUMemoryGuard();

	// Delete copy constructor and assignment operator
	GPUMemoryGuard(const GPUMemoryGuard&) = delete;
	GPUMemoryGuard& operator=(const GPUMemoryGuard&) = delete;

	// Allow move operations (explicitly defined)
	GPUMemoryGuard(GPUMemoryGuard&&) noexcept;
	GPUMemoryGuard& operator=(GPUMemoryGuard&&) noexcept;
};

} // namespace Utils
} // namespace WheelDL
