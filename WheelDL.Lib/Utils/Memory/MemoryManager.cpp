#include "pch.h"
#include "Utils/Memory/MemoryManager.h"
#include <torch/torch.h>
#include <stdexcept>

// Only include CUDA headers if CUDA is available
#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

// Include Windows headers for MEMORYSTATUSEX
#ifdef _WIN32
#include <windows.h>
#endif

namespace WheelDL
{
	namespace Utils
	{
		MemoryManager::MemoryManager() {}

		MemoryManager::~MemoryManager() {}

		MemoryManager& MemoryManager::getInstance()
		{
			static MemoryManager instance;
			return instance;
		}

		// ========== GPU Memory ==========

#ifdef USE_CUDA
		// Helper function to get GPU memory stats (reduces code duplication)
		MemoryManager::GPUMemoryStats MemoryManager::getGPUMemoryStats(int deviceIndex) const
		{
			GPUMemoryStats stats = {0, 0, false};

			if (!torch::cuda::is_available()) {
				return stats;
			}

			// Validate device index
			if (deviceIndex < 0 || deviceIndex >= torch::cuda::device_count()) {
				return stats;
			}

			// First, get total GPU memory from CUDA (this is reliable)
			try {
				cudaDeviceProp deviceProp;
				cudaError_t err = cudaGetDeviceProperties(&deviceProp, deviceIndex);
				if (err != cudaSuccess) {
					return stats;
				}
				stats.total = deviceProp.totalGlobalMem;
			} catch (...) {
				return stats;
			}

			// Then, try to get allocated memory from PyTorch CachingAllocator
			// This may fail if CUDA context is not initialized yet
			try {
				auto device_stats = c10::cuda::CUDACachingAllocator::getDeviceStats(deviceIndex);
#if TORCH_VERSION_MAJOR < 2
				stats.allocated = device_stats.allocated_bytes[static_cast<size_t>(c10::cuda::CUDACachingAllocator::StatType::AGGREGATE)].current;
#else
				stats.allocated = device_stats.allocated_bytes[static_cast<size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE)].current;
#endif
			} catch (...) {
				// CachingAllocator failed - assume 0 allocated (conservative estimate)
				// This can happen if no PyTorch CUDA operations have been performed yet
				stats.allocated = 0;
			}

			stats.valid = true;
			return stats;
		}
#endif

		float MemoryManager::getGPUMemoryUsed(int deviceIndex) const
		{
#ifdef USE_CUDA
			if (!torch::cuda::is_available()) {
				return 0.0f;
			}

			// Validate device index
			if (deviceIndex < 0 || deviceIndex >= torch::cuda::device_count()) {
				return 0.0f;
			}

			try {
				auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(deviceIndex);
#if TORCH_VERSION_MAJOR < 2
				size_t allocated = stats.allocated_bytes[static_cast<size_t>(c10::cuda::CUDACachingAllocator::StatType::AGGREGATE)].current;
#else
				size_t allocated = stats.allocated_bytes[static_cast<size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE)].current;
#endif

				// Convert bytes to megabytes
				return static_cast<float>(allocated) / (1024.0f * 1024.0f);
			} catch (...) {
				return 0.0f;
			}
#else
			// CPU-only build, no GPU memory available
			return 0.0f;
#endif
		}

		float MemoryManager::getGPUMemoryTotal(int deviceIndex) const
		{
#ifdef USE_CUDA
			if (!torch::cuda::is_available()) {
				return 0.0f;
			}

			// Validate device index
			if (deviceIndex < 0 || deviceIndex >= torch::cuda::device_count()) {
				return 0.0f;
			}

			try {
				cudaDeviceProp deviceProp;
				cudaError_t err = cudaGetDeviceProperties(&deviceProp, deviceIndex);
				if (err != cudaSuccess) {
					return 0.0f;
				}
				size_t totalMemory = deviceProp.totalGlobalMem;

				// Convert bytes to megabytes
				return static_cast<float>(totalMemory) / (1024.0f * 1024.0f);
			} catch (...) {
				return 0.0f;
			}
#else
			// CPU-only build, no GPU memory available
			return 0.0f;
#endif
		}

		float MemoryManager::getGPUMemoryUsagePercent(int deviceIndex) const
		{
#ifdef USE_CUDA
			auto stats = getGPUMemoryStats(deviceIndex);
			if (!stats.valid || stats.total == 0) {
				return 0.0f;
			}

			// Calculate percentage: allocated / total GPU memory
			return (static_cast<float>(stats.allocated) / static_cast<float>(stats.total)) * 100.0f;
#else
			// CPU-only build, no GPU memory available
			return 0.0f;
#endif
		}

		bool MemoryManager::hasEnoughGPUMemory(size_t requiredBytes, int deviceIndex) const
		{
#ifdef USE_CUDA
			auto stats = getGPUMemoryStats(deviceIndex);
			if (!stats.valid) {
				return false;
			}

			// Calculate available memory: total - allocated
			size_t available = (stats.total > stats.allocated) ? (stats.total - stats.allocated) : 0;

			return available >= requiredBytes;
#else
			// CPU-only build, no GPU memory available
			return false;
#endif
		}

		// ========== CPU Memory ==========

		size_t MemoryManager::getCPUMemoryUsed() const
		{
#ifdef _WIN32
			try {
				MEMORYSTATUSEX memInfo;
				memInfo.dwLength = sizeof(MEMORYSTATUSEX);

				if (!GlobalMemoryStatusEx(&memInfo)) {
					return 0;  // Failed to get memory status, return 0
				}

				// Total physical memory minus available physical memory
				return static_cast<size_t>(memInfo.ullTotalPhys - memInfo.ullAvailPhys);
			} catch (...) {
				return 0;  // Exception occurred, return 0
			}
#else
			// Non-Windows platforms - use sysconf or /proc/meminfo
			// For now, return 0 as placeholder
			return 0;
#endif
		}

		size_t MemoryManager::getCPUMemoryAvailable() const
		{
#ifdef _WIN32
			try {
				MEMORYSTATUSEX memInfo;
				memInfo.dwLength = sizeof(MEMORYSTATUSEX);

				if (!GlobalMemoryStatusEx(&memInfo)) {
					return 0;  // Failed to get memory status, return 0
				}

				return static_cast<size_t>(memInfo.ullAvailPhys);
			} catch (...) {
				return 0;  // Exception occurred, return 0
			}
#else
			// Non-Windows platforms - use sysconf or /proc/meminfo
			// For now, return 0 as placeholder
			return 0;
#endif
		}

	} // namespace Utils
} // namespace WheelDL
