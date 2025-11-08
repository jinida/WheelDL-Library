#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Utils/Memory/DynamicBatchSizer.h"

using namespace WheelDL::Utils;

/**
 * @class DynamicBatchSizerTest
 * @brief Test suite for DynamicBatchSizer
 */
class DynamicBatchSizerTest : public ::testing::Test {
protected:
	void SetUp() override {
		sizer_ = new DynamicBatchSizer(1, 128, 85.0f);
	}

	void TearDown() override {
		delete sizer_;
	}

	DynamicBatchSizer* sizer_;
};

// ========== Constructor Tests ==========

TEST_F(DynamicBatchSizerTest, DefaultConstruction) {
	DynamicBatchSizer sizer(1, 64, 80.0f);

	EXPECT_EQ(sizer.getMinBatch(), 1ULL);
	EXPECT_EQ(sizer.getMaxBatch(), 64ULL);
	EXPECT_FLOAT_EQ(sizer.getTargetMemoryUsage(), 80.0f);
}

// ========== Batch Size Computation Tests ==========

TEST_F(DynamicBatchSizerTest, ComputeOptimalBatchSizeHighUsage) {
	// High memory usage should reduce batch size
	size_t batch = sizer_->computeOptimalBatchSize(95.0f, 1024);

	// Batch should be within bounds
	EXPECT_GE(batch, sizer_->getMinBatch());
	EXPECT_LE(batch, sizer_->getMaxBatch());
}

TEST_F(DynamicBatchSizerTest, ComputeOptimalBatchSizeLowUsage) {
	// Low memory usage should increase batch size
	size_t initial_batch = sizer_->computeOptimalBatchSize(50.0f, 1024);

	// Call again with low usage
	size_t increased_batch = sizer_->computeOptimalBatchSize(50.0f, 1024);

	// Batch should still be within bounds
	EXPECT_GE(increased_batch, sizer_->getMinBatch());
	EXPECT_LE(increased_batch, sizer_->getMaxBatch());
}

TEST_F(DynamicBatchSizerTest, ComputeOptimalBatchSizeTargetUsage) {
	// Usage near target should not change batch much
	size_t batch1 = sizer_->computeOptimalBatchSize(85.0f, 1024);
	size_t batch2 = sizer_->computeOptimalBatchSize(85.0f, 1024);

	// Should be stable around target
	EXPECT_GE(batch1, sizer_->getMinBatch());
	EXPECT_LE(batch1, sizer_->getMaxBatch());
}

TEST_F(DynamicBatchSizerTest, BatchSizeRespectsBounds) {
	// Even with extreme memory usage, should respect min/max bounds
	size_t batch_very_high = sizer_->computeOptimalBatchSize(99.0f, 1024);
	EXPECT_GE(batch_very_high, sizer_->getMinBatch());

	size_t batch_very_low = sizer_->computeOptimalBatchSize(10.0f, 1024);
	EXPECT_LE(batch_very_low, sizer_->getMaxBatch());
}

// ========== Should Adjust Tests ==========

TEST_F(DynamicBatchSizerTest, ShouldAdjustBatchSizeTrue) {
	// Large difference from target should require adjustment
	EXPECT_TRUE(sizer_->shouldAdjustBatchSize(50.0f)); // 35% difference
	EXPECT_TRUE(sizer_->shouldAdjustBatchSize(99.0f)); // 14% difference
}

TEST_F(DynamicBatchSizerTest, ShouldAdjustBatchSizeFalse) {
	// Small difference from target should not require adjustment
	EXPECT_FALSE(sizer_->shouldAdjustBatchSize(85.0f)); // 0% difference
	EXPECT_FALSE(sizer_->shouldAdjustBatchSize(88.0f)); // 3% difference
	EXPECT_FALSE(sizer_->shouldAdjustBatchSize(82.0f)); // 3% difference
}

// ========== Setter/Getter Tests ==========

TEST_F(DynamicBatchSizerTest, SetGetMinBatch) {
	sizer_->setMinBatch(10);
	EXPECT_EQ(sizer_->getMinBatch(), 10ULL);
}

TEST_F(DynamicBatchSizerTest, SetGetMaxBatch) {
	sizer_->setMaxBatch(256);
	EXPECT_EQ(sizer_->getMaxBatch(), 256ULL);
}

TEST_F(DynamicBatchSizerTest, SetGetTargetMemoryUsage) {
	sizer_->setTargetMemoryUsage(75.0f);
	EXPECT_FLOAT_EQ(sizer_->getTargetMemoryUsage(), 75.0f);
}

TEST_F(DynamicBatchSizerTest, SetTargetMemoryUsageClamped) {
	// Test clamping to valid range (50-95%)
	sizer_->setTargetMemoryUsage(30.0f); // Too low
	EXPECT_GE(sizer_->getTargetMemoryUsage(), 50.0f);

	sizer_->setTargetMemoryUsage(98.0f); // Too high
	EXPECT_LE(sizer_->getTargetMemoryUsage(), 95.0f);
}

TEST_F(DynamicBatchSizerTest, SetMinBatchAdjustsCurrent) {
	// Compute a batch size
	sizer_->computeOptimalBatchSize(85.0f, 1024);

	// Set min batch higher
	sizer_->setMinBatch(50);

	// Next computation should respect new min
	size_t batch = sizer_->computeOptimalBatchSize(85.0f, 1024);
	EXPECT_GE(batch, 50ULL);
}

TEST_F(DynamicBatchSizerTest, SetMaxBatchAdjustsCurrent) {
	// Compute a batch size with low memory (should increase)
	sizer_->computeOptimalBatchSize(20.0f, 1024);

	// Set max batch lower
	sizer_->setMaxBatch(32);

	// Next computation should respect new max
	size_t batch = sizer_->computeOptimalBatchSize(20.0f, 1024);
	EXPECT_LE(batch, 32ULL);
}
