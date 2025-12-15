#include "pch.h"
#include "Data/Dataset/BaseDataset.h"
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace WheelDL::Data::Dataset;

// =============================================================================
// Test Fixture
// =============================================================================

class DataExampleTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper: Create test tensor
    torch::Tensor createTestTensor(std::vector<int64_t> shape) {
        return torch::randn(shape);
    }

    // Helper: Check if tensor is on expected device
    bool isOnDevice(const torch::Tensor& tensor, torch::DeviceType deviceType) {
        return tensor.device().type() == deviceType;
    }
};

// =============================================================================
// 3.1 CacheType Enum Tests (CT-001 ~ CT-005)
// =============================================================================

// CT-001: CacheType NONE value
TEST_F(DataExampleTest, CacheType_NONE_Value) {
    EXPECT_EQ(static_cast<int>(CacheType::NONE), 0);
}

// CT-002: CacheType RAM value
TEST_F(DataExampleTest, CacheType_RAM_Value) {
    EXPECT_EQ(static_cast<int>(CacheType::RAM), 1);
}

// CT-003: CacheType comparison equal
TEST_F(DataExampleTest, CacheType_Comparison_Equal) {
    CacheType type1 = CacheType::RAM;
    CacheType type2 = CacheType::RAM;
    EXPECT_EQ(type1, type2);
}

// CT-004: CacheType comparison not equal
TEST_F(DataExampleTest, CacheType_Comparison_NotEqual) {
    CacheType type1 = CacheType::NONE;
    CacheType type2 = CacheType::RAM;
    EXPECT_NE(type1, type2);
}

// CT-005: CacheType switch case
TEST_F(DataExampleTest, CacheType_SwitchCase) {
    auto getCacheName = [](CacheType type) -> std::string {
        switch (type) {
            case CacheType::NONE: return "NONE";
            case CacheType::RAM: return "RAM";
            default: return "UNKNOWN";
        }
    };

    EXPECT_EQ(getCacheName(CacheType::NONE), "NONE");
    EXPECT_EQ(getCacheName(CacheType::RAM), "RAM");
}

// =============================================================================
// 3.2 DataExample Struct - Fields Tests (DE-001 ~ DE-009)
// =============================================================================

// DE-001: DataExample default construction
TEST_F(DataExampleTest, DataExample_DefaultConstruction) {
    DataExample example;
    // Should compile and not crash
    SUCCEED();
}

// DE-002: DataExample data initial value undefined
TEST_F(DataExampleTest, DataExample_Data_Undefined) {
    DataExample example;
    EXPECT_FALSE(example.data.defined());
}

// DE-003: DataExample classes initial value undefined
TEST_F(DataExampleTest, DataExample_Classes_Undefined) {
    DataExample example;
    EXPECT_FALSE(example.classes.defined());
}

// DE-004: DataExample targets initial value undefined
TEST_F(DataExampleTest, DataExample_Targets_Undefined) {
    DataExample example;
    EXPECT_FALSE(example.targets.defined());
}

// DE-005: DataExample batchIndices initial value undefined
TEST_F(DataExampleTest, DataExample_BatchIndices_Undefined) {
    DataExample example;
    EXPECT_FALSE(example.batchIndices.defined());
}

// DE-006: DataExample data defined after assignment
TEST_F(DataExampleTest, DataExample_Data_Defined) {
    DataExample example;
    example.data = torch::randn({3, 224, 224});

    EXPECT_TRUE(example.data.defined());
    EXPECT_EQ(example.data.dim(), 3);
    EXPECT_EQ(example.data.size(0), 3);
    EXPECT_EQ(example.data.size(1), 224);
    EXPECT_EQ(example.data.size(2), 224);
}

// DE-007: DataExample classes defined after assignment
TEST_F(DataExampleTest, DataExample_Classes_Defined) {
    DataExample example;
    example.classes = torch::tensor({0, 1, 2}, torch::kLong);

    EXPECT_TRUE(example.classes.defined());
    EXPECT_EQ(example.classes.dim(), 1);
    EXPECT_EQ(example.classes.size(0), 3);
    EXPECT_EQ(example.classes.dtype(), torch::kLong);
}

// DE-008: DataExample targets defined after assignment
TEST_F(DataExampleTest, DataExample_Targets_Defined) {
    DataExample example;
    example.targets = torch::randn({5, 4});  // 5 boxes with 4 coordinates

    EXPECT_TRUE(example.targets.defined());
    EXPECT_EQ(example.targets.dim(), 2);
    EXPECT_EQ(example.targets.size(0), 5);
    EXPECT_EQ(example.targets.size(1), 4);
}

// DE-009: DataExample batchIndices defined after assignment
TEST_F(DataExampleTest, DataExample_BatchIndices_Defined) {
    DataExample example;
    example.batchIndices = torch::tensor({0, 0, 1, 1, 2}, torch::kLong);

    EXPECT_TRUE(example.batchIndices.defined());
    EXPECT_EQ(example.batchIndices.dim(), 1);
    EXPECT_EQ(example.batchIndices.size(0), 5);
}

// =============================================================================
// 3.3 DataExample::toDevice() Tests (DE-010 ~ DE-020)
// =============================================================================

// DE-010: toDevice with all tensors defined -> CPU
TEST_F(DataExampleTest, ToDevice_AllDefined_CPU) {
    DataExample example;
    example.data = torch::randn({3, 64, 64});
    example.classes = torch::tensor({1, 2, 3}, torch::kLong);
    example.targets = torch::randn({3, 4});
    example.batchIndices = torch::tensor({0, 0, 0}, torch::kLong);

    example.toDevice(torch::kCPU);

    EXPECT_TRUE(isOnDevice(example.data, torch::kCPU));
    EXPECT_TRUE(isOnDevice(example.classes, torch::kCPU));
    EXPECT_TRUE(isOnDevice(example.targets, torch::kCPU));
    EXPECT_TRUE(isOnDevice(example.batchIndices, torch::kCPU));
}

// DE-011: toDevice with all tensors defined -> CUDA (skip if no CUDA)
TEST_F(DataExampleTest, ToDevice_AllDefined_CUDA) {
    if (!torch::cuda::is_available()) {
        SUCCEED() << "CUDA not available, skipping test";
        return;
    }

    DataExample example;
    example.data = torch::randn({3, 64, 64});
    example.classes = torch::tensor({1, 2, 3}, torch::kLong);
    example.targets = torch::randn({3, 4});
    example.batchIndices = torch::tensor({0, 0, 0}, torch::kLong);

    example.toDevice(torch::kCUDA);

    EXPECT_TRUE(isOnDevice(example.data, torch::kCUDA));
    EXPECT_TRUE(isOnDevice(example.classes, torch::kCUDA));
    EXPECT_TRUE(isOnDevice(example.targets, torch::kCUDA));
    EXPECT_TRUE(isOnDevice(example.batchIndices, torch::kCUDA));
}

// DE-012: toDevice with only data undefined
TEST_F(DataExampleTest, ToDevice_DataUndefined) {
    DataExample example;
    // data is not defined
    example.classes = torch::tensor({1, 2, 3}, torch::kLong);
    example.targets = torch::randn({3, 4});
    example.batchIndices = torch::tensor({0, 0, 0}, torch::kLong);

    // Should not throw
    EXPECT_NO_THROW(example.toDevice(torch::kCPU));

    EXPECT_FALSE(example.data.defined());
    EXPECT_TRUE(example.classes.defined());
    EXPECT_TRUE(example.targets.defined());
    EXPECT_TRUE(example.batchIndices.defined());
}

// DE-013: toDevice with only classes undefined
TEST_F(DataExampleTest, ToDevice_ClassesUndefined) {
    DataExample example;
    example.data = torch::randn({3, 64, 64});
    // classes is not defined
    example.targets = torch::randn({3, 4});
    example.batchIndices = torch::tensor({0, 0, 0}, torch::kLong);

    EXPECT_NO_THROW(example.toDevice(torch::kCPU));

    EXPECT_TRUE(example.data.defined());
    EXPECT_FALSE(example.classes.defined());
    EXPECT_TRUE(example.targets.defined());
    EXPECT_TRUE(example.batchIndices.defined());
}

// DE-014: toDevice with only targets undefined
TEST_F(DataExampleTest, ToDevice_TargetsUndefined) {
    DataExample example;
    example.data = torch::randn({3, 64, 64});
    example.classes = torch::tensor({1, 2, 3}, torch::kLong);
    // targets is not defined
    example.batchIndices = torch::tensor({0, 0, 0}, torch::kLong);

    EXPECT_NO_THROW(example.toDevice(torch::kCPU));

    EXPECT_TRUE(example.data.defined());
    EXPECT_TRUE(example.classes.defined());
    EXPECT_FALSE(example.targets.defined());
    EXPECT_TRUE(example.batchIndices.defined());
}

// DE-015: toDevice with only batchIndices undefined
TEST_F(DataExampleTest, ToDevice_BatchIndicesUndefined) {
    DataExample example;
    example.data = torch::randn({3, 64, 64});
    example.classes = torch::tensor({1, 2, 3}, torch::kLong);
    example.targets = torch::randn({3, 4});
    // batchIndices is not defined

    EXPECT_NO_THROW(example.toDevice(torch::kCPU));

    EXPECT_TRUE(example.data.defined());
    EXPECT_TRUE(example.classes.defined());
    EXPECT_TRUE(example.targets.defined());
    EXPECT_FALSE(example.batchIndices.defined());
}

// DE-016: toDevice with all tensors undefined
TEST_F(DataExampleTest, ToDevice_AllUndefined) {
    DataExample example;
    // All fields undefined

    EXPECT_NO_THROW(example.toDevice(torch::kCPU));

    EXPECT_FALSE(example.data.defined());
    EXPECT_FALSE(example.classes.defined());
    EXPECT_FALSE(example.targets.defined());
    EXPECT_FALSE(example.batchIndices.defined());
}

// DE-017: toDevice with partial defined (data and classes only)
TEST_F(DataExampleTest, ToDevice_PartialDefined_1) {
    DataExample example;
    example.data = torch::randn({3, 64, 64});
    example.classes = torch::tensor({1, 2, 3}, torch::kLong);
    // targets and batchIndices not defined

    example.toDevice(torch::kCPU);

    EXPECT_TRUE(example.data.defined());
    EXPECT_TRUE(example.classes.defined());
    EXPECT_FALSE(example.targets.defined());
    EXPECT_FALSE(example.batchIndices.defined());
}

// DE-018: toDevice with partial defined (targets and batchIndices only)
TEST_F(DataExampleTest, ToDevice_PartialDefined_2) {
    DataExample example;
    // data and classes not defined
    example.targets = torch::randn({5, 4});
    example.batchIndices = torch::tensor({0, 1, 2, 3, 4}, torch::kLong);

    example.toDevice(torch::kCPU);

    EXPECT_FALSE(example.data.defined());
    EXPECT_FALSE(example.classes.defined());
    EXPECT_TRUE(example.targets.defined());
    EXPECT_TRUE(example.batchIndices.defined());
}

// DE-019: toDevice returns self reference
TEST_F(DataExampleTest, ToDevice_ReturnsSelf) {
    DataExample example;
    example.data = torch::randn({3, 64, 64});

    DataExample& returned = example.toDevice(torch::kCPU);

    EXPECT_EQ(&returned, &example);
}

// DE-020: toDevice device verification after move
TEST_F(DataExampleTest, ToDevice_DeviceVerification) {
    if (!torch::cuda::is_available()) {
        SUCCEED() << "CUDA not available, skipping test";
        return;
    }

    DataExample example;
    example.data = torch::randn({3, 64, 64});
    example.classes = torch::tensor({1, 2}, torch::kLong);

    // Start on CPU
    EXPECT_TRUE(isOnDevice(example.data, torch::kCPU));
    EXPECT_TRUE(isOnDevice(example.classes, torch::kCPU));

    // Move to CUDA
    example.toDevice(torch::kCUDA);
    EXPECT_TRUE(isOnDevice(example.data, torch::kCUDA));
    EXPECT_TRUE(isOnDevice(example.classes, torch::kCUDA));

    // Move back to CPU
    example.toDevice(torch::kCPU);
    EXPECT_TRUE(isOnDevice(example.data, torch::kCPU));
    EXPECT_TRUE(isOnDevice(example.classes, torch::kCPU));
}
