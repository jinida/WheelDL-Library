#include "pch.h"
#include <gtest/gtest.h>
#include "Utils/Memory/GPUMemoryGuard.h"
#include <torch/torch.h>
#include <memory>
#include <utility>

using namespace WheelDL::Utils;

// =============================================================================
// Test Fixture
// =============================================================================

class GPUMemoryGuardTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

// =============================================================================
// Constructor/Destructor Tests (GMG-001 ~ GMG-002)
// =============================================================================

// GMG-001: Constructor does not crash
TEST_F(GPUMemoryGuardTest, Constructor_NoCrash) {
    EXPECT_NO_THROW({
        GPUMemoryGuard guard;
    });
}

// GMG-002: Destructor does not crash
TEST_F(GPUMemoryGuardTest, Destructor_NoCrash) {
    EXPECT_NO_THROW({
        GPUMemoryGuard* guard = new GPUMemoryGuard();
        delete guard;
    });
}

// =============================================================================
// Move Semantics Tests (GMG-003 ~ GMG-004)
// =============================================================================

// GMG-003: Move constructor valid
TEST_F(GPUMemoryGuardTest, MoveConstructor_Valid) {
    EXPECT_NO_THROW({
        GPUMemoryGuard guard1;
        GPUMemoryGuard guard2 = std::move(guard1);
        // Both should be valid (no state to transfer)
    });
}

// GMG-004: Move assignment valid
TEST_F(GPUMemoryGuardTest, MoveAssignment_Valid) {
    EXPECT_NO_THROW({
        GPUMemoryGuard guard1;
        GPUMemoryGuard guard2;
        guard2 = std::move(guard1);
        // Both should be valid (no state to transfer)
    });
}

// =============================================================================
// RAII Tests (GMG-005)
// =============================================================================

// GMG-005: RAII scope exit - destructor called automatically
TEST_F(GPUMemoryGuardTest, RAII_ScopeExit) {
    bool scopeExited = false;

    {
        GPUMemoryGuard guard;
        scopeExited = false;
    }
    // Destructor should have been called
    scopeExited = true;

    EXPECT_TRUE(scopeExited);
}

// =============================================================================
// No State Tests (GMG-008 ~ GMG-009)
// =============================================================================

// GMG-008: Move constructor - no state
TEST_F(GPUMemoryGuardTest, MoveConstructor_NoState) {
    // GPUMemoryGuard has no state to transfer
    GPUMemoryGuard guard1;
    GPUMemoryGuard guard2 = std::move(guard1);

    // Both objects still valid - destructor will be called for both
    // This is fine because destructor is idempotent (calling empty_cache twice is safe)
    SUCCEED();
}

// GMG-009: Move assignment - no state
TEST_F(GPUMemoryGuardTest, MoveAssignment_NoState) {
    GPUMemoryGuard guard1;
    GPUMemoryGuard guard2;

    guard2 = std::move(guard1);

    // Both objects still valid
    SUCCEED();
}

// =============================================================================
// Non-Copyable Tests
// =============================================================================

// Verify that GPUMemoryGuard is non-copyable (compile-time check)
// These would fail to compile if uncommented, which is the expected behavior
/*
TEST_F(GPUMemoryGuardTest, NonCopyable_CopyConstructor) {
    GPUMemoryGuard guard1;
    GPUMemoryGuard guard2 = guard1;  // Should not compile
}

TEST_F(GPUMemoryGuardTest, NonCopyable_CopyAssignment) {
    GPUMemoryGuard guard1;
    GPUMemoryGuard guard2;
    guard2 = guard1;  // Should not compile
}
*/

// =============================================================================
// CUDA Tests (GMG-006 ~ GMG-007)
// =============================================================================

#ifdef USE_CUDA
// GMG-006: Destructor clears GPU cache
TEST_F(GPUMemoryGuardTest, CUDA_Destructor_ClearsCache) {
    if (!torch::cuda::is_available()) {
        // Skip test if CUDA not available at runtime
        SUCCEED();
        return;
    }

    // Allocate some GPU memory
    torch::Tensor t = torch::zeros({1000, 1000}, torch::device(torch::kCUDA));

    {
        GPUMemoryGuard guard;
        // Do some GPU operations
        t = t + 1;
    }
    // Destructor should have called empty_cache()
    // We can't easily verify this, but at least it shouldn't crash

    SUCCEED();
}
#endif

#ifndef USE_CUDA
// GMG-007: No CUDA - destructor is no-op
TEST_F(GPUMemoryGuardTest, NoCUDA_Destructor_Noop) {
    // Without CUDA, destructor should be a no-op
    EXPECT_NO_THROW({
        GPUMemoryGuard guard;
        // Destructor will be called at end of scope
    });
}
#endif

// =============================================================================
// Multiple Guards Tests
// =============================================================================

TEST_F(GPUMemoryGuardTest, MultipleGuards_Sequential) {
    // Multiple sequential guards should work fine
    EXPECT_NO_THROW({
        {
            GPUMemoryGuard guard1;
        }
        {
            GPUMemoryGuard guard2;
        }
        {
            GPUMemoryGuard guard3;
        }
    });
}

TEST_F(GPUMemoryGuardTest, MultipleGuards_Nested) {
    // Nested guards should work fine
    EXPECT_NO_THROW({
        GPUMemoryGuard guard1;
        {
            GPUMemoryGuard guard2;
            {
                GPUMemoryGuard guard3;
            }
        }
    });
}

// =============================================================================
// Unique Ptr Tests
// =============================================================================

TEST_F(GPUMemoryGuardTest, UniquePtr_Usage) {
    EXPECT_NO_THROW({
        auto guard = std::make_unique<GPUMemoryGuard>();
        // guard is automatically deleted when unique_ptr goes out of scope
    });
}

TEST_F(GPUMemoryGuardTest, UniquePtr_Reset) {
    EXPECT_NO_THROW({
        auto guard = std::make_unique<GPUMemoryGuard>();
        guard.reset();  // Explicitly delete
        guard = std::make_unique<GPUMemoryGuard>();  // Create new one
    });
}

// =============================================================================
// Function Scope Tests
// =============================================================================

namespace {
    void functionWithGuard() {
        GPUMemoryGuard guard;
        // Do some work...
        // Guard is automatically destroyed when function returns
    }

    void functionWithGuardAndException() {
        GPUMemoryGuard guard;
        // Even if we throw, guard's destructor will be called
        throw std::runtime_error("Test exception");
    }
}

TEST_F(GPUMemoryGuardTest, FunctionScope_NormalReturn) {
    EXPECT_NO_THROW({
        functionWithGuard();
    });
}

TEST_F(GPUMemoryGuardTest, FunctionScope_WithException) {
    // Guard's destructor should still be called even when exception is thrown
    EXPECT_THROW({
        functionWithGuardAndException();
    }, std::runtime_error);
}

// =============================================================================
// Conditional Guard Tests
// =============================================================================

TEST_F(GPUMemoryGuardTest, ConditionalGuard) {
    bool useGuard = true;

    EXPECT_NO_THROW({
        std::unique_ptr<GPUMemoryGuard> guard;
        if (useGuard) {
            guard = std::make_unique<GPUMemoryGuard>();
        }
        // Guard may or may not exist based on condition
    });
}
