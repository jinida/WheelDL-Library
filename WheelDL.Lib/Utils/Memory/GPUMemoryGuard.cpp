#include "pch.h"
#include "Utils/Memory/GPUMemoryGuard.h"
#include <torch/torch.h>

// Only include CUDA headers if CUDA is available
#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace WheelDL {
namespace Utils {

GPUMemoryGuard::GPUMemoryGuard() {
	// Constructor is a no-op as specified in the architecture
}

GPUMemoryGuard::~GPUMemoryGuard() {
#ifdef USE_CUDA
	// Clear GPU cache if CUDA is available
	if (torch::cuda::is_available()) {
		c10::cuda::CUDACachingAllocator::emptyCache();
	}
#endif
	// For CPU-only builds, destructor is a no-op
}

// Move constructor (explicitly defined)
GPUMemoryGuard::GPUMemoryGuard(GPUMemoryGuard&&) noexcept {
	// No state to move for this RAII guard
	// The destructor will be called for the moved-to object
}

// Move assignment operator (explicitly defined)
GPUMemoryGuard& GPUMemoryGuard::operator=(GPUMemoryGuard&&) noexcept {
	// No state to move for this RAII guard
	// Both objects will have their destructors called
	return *this;
}

} // namespace Utils
} // namespace WheelDL
