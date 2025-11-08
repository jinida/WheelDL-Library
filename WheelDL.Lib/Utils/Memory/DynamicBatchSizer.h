#pragma once

#include <cstddef>
#include <mutex>

namespace WheelDL {
namespace Utils {

/**
 * @class DynamicBatchSizer
 * @brief Thread-safe dynamic batch size adjustment based on available memory
 *
 * Monitors GPU/CPU memory usage and recommends optimal batch sizes
 * to prevent out-of-memory errors while maximizing throughput.
 *
 * @note All methods are thread-safe
 */
class DynamicBatchSizer {
public:
	/**
	 * @brief Constructor
	 * @param min_batch Minimum batch size (must be > 0)
	 * @param max_batch Maximum batch size (must be >= min_batch)
	 * @param target_memory_usage Target memory usage percentage (default: 85.0)
	 * @throws std::invalid_argument if min_batch == 0, max_batch == 0, or min_batch > max_batch
	 */
	DynamicBatchSizer(size_t min_batch = 1,
	                  size_t max_batch = 128,
	                  float target_memory_usage = 85.0f);

	~DynamicBatchSizer();

	/**
	 * @brief Compute optimal batch size based on current memory usage
	 * @param current_usage Current memory usage percentage
	 * @param bytes_per_sample Estimated memory per sample in bytes
	 * @return size_t Recommended batch size
	 */
	size_t computeOptimalBatchSize(float current_usage, size_t bytes_per_sample);

	/**
	 * @brief Check if batch size should be adjusted
	 * @param current_usage Current memory usage percentage
	 * @return bool True if adjustment is recommended
	 */
	bool shouldAdjustBatchSize(float current_usage) const;

	/**
	 * @brief Set minimum batch size
	 * @param min_batch New minimum batch size (must be > 0 and <= current max_batch)
	 */
	void setMinBatch(size_t min_batch);

	/**
	 * @brief Set maximum batch size
	 * @param max_batch New maximum batch size (must be > 0 and >= current min_batch)
	 */
	void setMaxBatch(size_t max_batch);

	/**
	 * @brief Set target memory usage
	 * @param target Target memory usage percentage (0.0 - 100.0)
	 */
	void setTargetMemoryUsage(float target);

	/**
	 * @brief Get current minimum batch size
	 * @return size_t Minimum batch size
	 */
	size_t getMinBatch() const;

	/**
	 * @brief Get current maximum batch size
	 * @return size_t Maximum batch size
	 */
	size_t getMaxBatch() const;

	/**
	 * @brief Get current batch size
	 * @return size_t Current batch size
	 */
	size_t getCurrentBatch() const;

	/**
	 * @brief Get target memory usage
	 * @return float Target memory usage percentage
	 */
	float getTargetMemoryUsage() const;

private:
	size_t min_batch_;
	size_t max_batch_;
	float target_memory_usage_;
	size_t current_batch_;
	mutable std::mutex mutex_;  ///< Mutex for thread-safe access
};

} // namespace Utils
} // namespace WheelDL
