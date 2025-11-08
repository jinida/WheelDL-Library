#include "pch.h"
#include "Utils/Memory/DynamicBatchSizer.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace WheelDL {
namespace Utils {

DynamicBatchSizer::DynamicBatchSizer(size_t min_batch,
                                     size_t max_batch,
                                     float target_memory_usage)
	: min_batch_(min_batch)
	, max_batch_(max_batch)
	, target_memory_usage_(target_memory_usage)
	, current_batch_(min_batch) {

	// Validate parameters
	if (min_batch == 0) {
		throw std::invalid_argument("DynamicBatchSizer: min_batch must be greater than 0");
	}
	if (max_batch == 0) {
		throw std::invalid_argument("DynamicBatchSizer: max_batch must be greater than 0");
	}
	if (min_batch > max_batch) {
		throw std::invalid_argument("DynamicBatchSizer: min_batch must be <= max_batch");
	}
}

DynamicBatchSizer::~DynamicBatchSizer() {
}

size_t DynamicBatchSizer::computeOptimalBatchSize(float current_usage, size_t bytes_per_sample) {
	std::lock_guard<std::mutex> lock(mutex_);

	// If current usage is too high, reduce batch size
	if (current_usage > target_memory_usage_) {
		// Calculate reduction factor based on how much we're over target
		float excess = current_usage - target_memory_usage_;
		float reduction_factor = 1.0f - (excess / 100.0f);
		reduction_factor = std::max(0.5f, reduction_factor); // Reduce by at most 50% at once

		size_t new_batch = static_cast<size_t>(current_batch_ * reduction_factor);
		// Ensure we don't go below min_batch
		current_batch_ = std::max(min_batch_, new_batch);
	}
	// If current usage is below target, we can increase batch size
	else if (current_usage < target_memory_usage_ - 10.0f) {
		// Calculate increase factor based on available headroom
		float headroom = target_memory_usage_ - current_usage;
		float increase_factor = 1.0f + (headroom / target_memory_usage_);
		increase_factor = std::min(1.5f, increase_factor); // Increase by at most 50% at once

		size_t new_batch = static_cast<size_t>(current_batch_ * increase_factor);

		// If bytes_per_sample is provided, use it to constrain the batch size
		// to avoid exceeding available memory
		if (bytes_per_sample > 0) {
			// Properly calculate max allowed batch size based on available memory
			// headroom represents percentage points of available memory relative to target
			// Assuming we know current memory usage in bytes would allow precise calculation
			// For now, use a conservative heuristic based on the percentage headroom
			float memory_based_factor = 1.0f + (headroom / target_memory_usage_) * 0.5f;
			size_t memory_constrained_batch = static_cast<size_t>(current_batch_ * memory_based_factor);
			new_batch = std::min(new_batch, memory_constrained_batch);
		}

		// Ensure we don't exceed max_batch
		current_batch_ = std::min(max_batch_, new_batch);
	}

	// Final safety clamp to min/max bounds
	current_batch_ = std::max(min_batch_, std::min(max_batch_, current_batch_));

	return current_batch_;
}

bool DynamicBatchSizer::shouldAdjustBatchSize(float current_usage) const {
	std::lock_guard<std::mutex> lock(mutex_);
	// Adjust if we're too far from target (>10% difference)
	float difference = std::abs(current_usage - target_memory_usage_);
	return difference > 10.0f;
}

void DynamicBatchSizer::setMinBatch(size_t min_batch) {
	std::lock_guard<std::mutex> lock(mutex_);

	if (min_batch == 0) {
		throw std::invalid_argument("DynamicBatchSizer::setMinBatch: min_batch must be greater than 0");
	}
	if (min_batch > max_batch_) {
		throw std::invalid_argument("DynamicBatchSizer::setMinBatch: min_batch must be <= max_batch");
	}

	min_batch_ = min_batch;
	// Ensure current batch is within bounds
	if (current_batch_ < min_batch_) {
		current_batch_ = min_batch_;
	}
}

void DynamicBatchSizer::setMaxBatch(size_t max_batch) {
	std::lock_guard<std::mutex> lock(mutex_);

	if (max_batch == 0) {
		throw std::invalid_argument("DynamicBatchSizer::setMaxBatch: max_batch must be greater than 0");
	}
	if (max_batch < min_batch_) {
		throw std::invalid_argument("DynamicBatchSizer::setMaxBatch: max_batch must be >= min_batch");
	}

	max_batch_ = max_batch;
	// Ensure current batch is within bounds
	if (current_batch_ > max_batch_) {
		current_batch_ = max_batch_;
	}
}

void DynamicBatchSizer::setTargetMemoryUsage(float target) {
	std::lock_guard<std::mutex> lock(mutex_);
	// Clamp target to reasonable range (50-95%)
	target_memory_usage_ = std::max(50.0f, std::min(95.0f, target));
}

size_t DynamicBatchSizer::getMinBatch() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return min_batch_;
}

size_t DynamicBatchSizer::getMaxBatch() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return max_batch_;
}

size_t DynamicBatchSizer::getCurrentBatch() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return current_batch_;
}

float DynamicBatchSizer::getTargetMemoryUsage() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return target_memory_usage_;
}

} // namespace Utils
} // namespace WheelDL
