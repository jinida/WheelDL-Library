#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Loss/SegmentationLoss.h"
#include "Data/Dataset/BaseDataset.h"

using namespace WheelDL::Model::Loss;
using namespace WheelDL::Data::Dataset;

class SegmentationLossTest : public ::testing::Test {
protected:
    torch::Device device_ = torch::kCPU;

    void SetUp() override {
        torch::manual_seed(42);
#ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            device_ = torch::kCUDA;
        }
#endif
    }

    // Create prediction tensor: [N, C, H, W] logits
    torch::Tensor createPrediction(int64_t batchSize, int64_t numClasses, int64_t height, int64_t width) {
        return torch::randn({ batchSize, numClasses, height, width }, device_);
    }

    // Create target tensor: [N, C, H, W] binary masks
    torch::Tensor createTarget(int64_t batchSize, int64_t numClasses, int64_t height, int64_t width) {
        return torch::randint(0, 2, { batchSize, numClasses, height, width },
            torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    }

    // Create DataExample with mask targets
    DataExample createDataExample(int64_t batchSize, int64_t numClasses, int64_t height, int64_t width) {
        DataExample example;
        example.targets = createTarget(batchSize, numClasses, height, width);
        return example;
    }

    // Create perfect prediction (matches target exactly when sigmoid applied)
    torch::Tensor createPerfectPrediction(const torch::Tensor& target) {
        // Large positive for 1s, large negative for 0s
        return (target * 2.0f - 1.0f) * 10.0f;
    }

    // Create inverse prediction (opposite of target)
    torch::Tensor createInversePrediction(const torch::Tensor& target) {
        return ((1.0f - target) * 2.0f - 1.0f) * 10.0f;
    }
};

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_F(SegmentationLossTest, Constructor_DefaultParameters) {
    SegmentationLoss loss(10);

    EXPECT_EQ(loss.name(), "SegmentationLoss");

    auto [bceWeight, diceWeight] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(bceWeight, 1.0f);
    EXPECT_FLOAT_EQ(diceWeight, 1.0f);
}

TEST_F(SegmentationLossTest, Constructor_CustomParameters) {
    SegmentationLoss loss(10, 2.0f, 0.5f, 1e-5f);

    auto [bceWeight, diceWeight] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(bceWeight, 2.0f);
    EXPECT_FLOAT_EQ(diceWeight, 0.5f);
}

TEST_F(SegmentationLossTest, Constructor_SingleClass) {
    SegmentationLoss loss(1);
    EXPECT_EQ(loss.name(), "SegmentationLoss");
}

TEST_F(SegmentationLossTest, Constructor_ManyClasses) {
    SegmentationLoss loss(100);
    EXPECT_EQ(loss.name(), "SegmentationLoss");
}

// ============================================================================
// API Tests
// ============================================================================

TEST_F(SegmentationLossTest, Name_ReturnsCorrectName) {
    SegmentationLoss loss(10);
    EXPECT_EQ(loss.name(), "SegmentationLoss");
}

TEST_F(SegmentationLossTest, SetLossWeights_Valid) {
    SegmentationLoss loss(10);

    loss.setLossWeights(2.0f, 0.5f);

    auto [bceWeight, diceWeight] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(bceWeight, 2.0f);
    EXPECT_FLOAT_EQ(diceWeight, 0.5f);
}

TEST_F(SegmentationLossTest, SetLossWeights_Negative_ClampedToZero) {
    SegmentationLoss loss(10);

    loss.setLossWeights(-1.0f, -0.5f);

    auto [bceWeight, diceWeight] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(bceWeight, 0.0f);
    EXPECT_FLOAT_EQ(diceWeight, 0.0f);
}

TEST_F(SegmentationLossTest, SetLossWeights_Zero) {
    SegmentationLoss loss(10);

    loss.setLossWeights(0.0f, 0.0f);

    auto [bceWeight, diceWeight] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(bceWeight, 0.0f);
    EXPECT_FLOAT_EQ(diceWeight, 0.0f);
}

TEST_F(SegmentationLossTest, GetLossWeights_Default) {
    SegmentationLoss loss(10);

    auto [bceWeight, diceWeight] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(bceWeight, 1.0f);
    EXPECT_FLOAT_EQ(diceWeight, 1.0f);
}

TEST_F(SegmentationLossTest, SetSmooth_Valid) {
    SegmentationLoss loss(10);
    EXPECT_NO_THROW(loss.setSmooth(1e-5f));
}

TEST_F(SegmentationLossTest, SetSmooth_Zero_Throws) {
    SegmentationLoss loss(10);
    EXPECT_THROW(loss.setSmooth(0.0f), std::invalid_argument);
}

TEST_F(SegmentationLossTest, SetSmooth_Negative_Throws) {
    SegmentationLoss loss(10);
    EXPECT_THROW(loss.setSmooth(-1e-6f), std::invalid_argument);
}

// ============================================================================
// Compute Tests - Basic
// ============================================================================

TEST_F(SegmentationLossTest, Compute_Basic) {
    SegmentationLoss loss(10);

    auto pred = createPrediction(2, 10, 64, 64);
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("bce") != resultMap.end());
    EXPECT_TRUE(resultMap.find("dice") != resultMap.end());
    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
}

TEST_F(SegmentationLossTest, Compute_WithDataExample) {
    SegmentationLoss loss(10);

    auto pred = createPrediction(2, 10, 64, 64);
    auto example = createDataExample(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, example);

    EXPECT_TRUE(resultMap.find("bce") != resultMap.end());
    EXPECT_TRUE(resultMap.find("dice") != resultMap.end());
    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
}

TEST_F(SegmentationLossTest, Compute_BatchSize1) {
    SegmentationLoss loss(10);

    auto pred = createPrediction(1, 10, 64, 64);
    auto target = createTarget(1, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(SegmentationLossTest, Compute_BatchSize8) {
    SegmentationLoss loss(10);

    auto pred = createPrediction(8, 10, 64, 64);
    auto target = createTarget(8, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(SegmentationLossTest, Compute_LossComponentsNonNegative) {
    SegmentationLoss loss(10);

    auto pred = createPrediction(2, 10, 64, 64);
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    EXPECT_GE(resultMap["bce"].item<float>(), 0.0f);
    EXPECT_GE(resultMap["dice"].item<float>(), 0.0f);
    EXPECT_GE(resultMap["total"].item<float>(), 0.0f);
}

// ============================================================================
// Mask BCE Loss Tests
// ============================================================================

TEST_F(SegmentationLossTest, BCE_PerfectMask) {
    SegmentationLoss loss(1, 1.0f, 0.0f);  // Only BCE

    auto target = torch::ones({ 1, 1, 32, 32 }, device_);
    auto pred = createPerfectPrediction(target);

    auto resultMap = loss.compute(pred, target);

    // Perfect prediction should give very low loss
    EXPECT_LT(resultMap["bce"].item<float>(), 0.01f);
}

TEST_F(SegmentationLossTest, BCE_InverseMask) {
    SegmentationLoss loss(1, 1.0f, 0.0f);  // Only BCE

    auto target = torch::ones({ 1, 1, 32, 32 }, device_);
    auto pred = createInversePrediction(target);

    auto resultMap = loss.compute(pred, target);

    // Inverse prediction should give high loss
    EXPECT_GT(resultMap["bce"].item<float>(), 5.0f);
}

TEST_F(SegmentationLossTest, BCE_EmptyMask) {
    SegmentationLoss loss(1, 1.0f, 0.0f);

    auto target = torch::zeros({ 1, 1, 32, 32 }, device_);
    auto pred = createPrediction(1, 1, 32, 32);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["bce"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["bce"].isinf().any().item<bool>());
}

TEST_F(SegmentationLossTest, BCE_FullMask) {
    SegmentationLoss loss(1, 1.0f, 0.0f);

    auto target = torch::ones({ 1, 1, 32, 32 }, device_);
    auto pred = createPrediction(1, 1, 32, 32);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["bce"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["bce"].isinf().any().item<bool>());
}

TEST_F(SegmentationLossTest, BCE_HighResolution) {
    SegmentationLoss loss(1, 1.0f, 0.0f);

    auto pred = createPrediction(1, 1, 640, 640);
    auto target = createTarget(1, 1, 640, 640);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["bce"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["bce"].isinf().any().item<bool>());
}

// ============================================================================
// Dice Loss Tests
// ============================================================================

TEST_F(SegmentationLossTest, Dice_PerfectOverlap) {
    SegmentationLoss loss(1, 0.0f, 1.0f);  // Only Dice

    auto target = torch::ones({ 1, 1, 32, 32 }, device_);
    auto pred = createPerfectPrediction(target);

    auto resultMap = loss.compute(pred, target);

    // Perfect overlap should give very low Dice loss
    EXPECT_LT(resultMap["dice"].item<float>(), 0.01f);
}

TEST_F(SegmentationLossTest, Dice_NoOverlap) {
    SegmentationLoss loss(1, 0.0f, 1.0f);

    auto target = torch::ones({ 1, 1, 32, 32 }, device_);
    auto pred = createInversePrediction(target);

    auto resultMap = loss.compute(pred, target);

    // No overlap should give high Dice loss (close to 1.0)
    EXPECT_GT(resultMap["dice"].item<float>(), 0.9f);
}

TEST_F(SegmentationLossTest, Dice_PartialOverlap) {
    SegmentationLoss loss(1, 0.0f, 1.0f);

    // Create 50% overlap scenario
    auto target = torch::zeros({ 1, 1, 32, 32 }, device_);
    target.index_put_({ 0, 0, torch::indexing::Slice(0, 16), torch::indexing::Slice() }, 1.0f);

    auto pred = torch::zeros({ 1, 1, 32, 32 }, device_);
    pred.index_put_({ 0, 0, torch::indexing::Slice(8, 24), torch::indexing::Slice() }, 10.0f);  // Logits

    auto resultMap = loss.compute(pred, target);

    // Should be somewhere between 0 and 1
    EXPECT_GT(resultMap["dice"].item<float>(), 0.0f);
    EXPECT_LT(resultMap["dice"].item<float>(), 1.0f);
}

TEST_F(SegmentationLossTest, Dice_SmoothTerm_EmptyPrediction) {
    SegmentationLoss loss(1, 0.0f, 1.0f, 1.0f);  // Large smooth term

    auto target = torch::ones({ 1, 1, 32, 32 }, device_);
    auto pred = torch::ones({ 1, 1, 32, 32 }, device_) * (-10.0f);  // All zeros after sigmoid

    auto resultMap = loss.compute(pred, target);

    // With smooth term, should not be inf
    EXPECT_FALSE(resultMap["dice"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["dice"].isinf().any().item<bool>());
}

// ============================================================================
// Combined Loss Tests
// ============================================================================

TEST_F(SegmentationLossTest, Combined_BCEWeight) {
    SegmentationLoss loss1(10, 1.0f, 1.0f);
    SegmentationLoss loss2(10, 2.0f, 1.0f);

    auto pred = createPrediction(2, 10, 64, 64);
    auto target = createTarget(2, 10, 64, 64);

    auto result1 = loss1.compute(pred, target);
    auto result2 = loss2.compute(pred, target);

    // Total should increase when BCE weight increases
    EXPECT_GT(result2["total"].item<float>(), result1["total"].item<float>());
}

TEST_F(SegmentationLossTest, Combined_DiceWeight) {
    SegmentationLoss loss1(10, 1.0f, 1.0f);
    SegmentationLoss loss2(10, 1.0f, 2.0f);

    auto pred = createPrediction(2, 10, 64, 64);
    auto target = createTarget(2, 10, 64, 64);

    auto result1 = loss1.compute(pred, target);
    auto result2 = loss2.compute(pred, target);

    // Total should increase when Dice weight increases
    EXPECT_GT(result2["total"].item<float>(), result1["total"].item<float>());
}

TEST_F(SegmentationLossTest, Combined_MultiInstance) {
    SegmentationLoss loss(5);

    // Multiple instances (batch)
    auto pred = createPrediction(4, 5, 64, 64);
    auto target = createTarget(4, 5, 64, 64);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(SegmentationLossTest, Combined_VaryingSizes) {
    SegmentationLoss loss(10);

    // Test different spatial sizes
    std::vector<int64_t> sizes = { 32, 64, 128, 256 };

    for (auto size : sizes) {
        auto pred = createPrediction(1, 10, size, size);
        auto target = createTarget(1, 10, size, size);

        auto resultMap = loss.compute(pred, target);

        EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
        EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
    }
}

// ============================================================================
// Branch Coverage Tests
// ============================================================================

TEST_F(SegmentationLossTest, Branch_DiceWeightZero) {
    SegmentationLoss loss(10, 1.0f, 0.0f);  // Dice weight = 0

    auto pred = createPrediction(2, 10, 64, 64);
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    // Dice should be 0 when weight is 0
    EXPECT_FLOAT_EQ(resultMap["dice"].item<float>(), 0.0f);
}

TEST_F(SegmentationLossTest, Branch_DiceWeightPositive) {
    SegmentationLoss loss(10, 0.0f, 1.0f);  // BCE weight = 0

    auto pred = createPrediction(2, 10, 64, 64);
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    // Dice should be computed
    EXPECT_GT(resultMap["dice"].item<float>(), 0.0f);
}

TEST_F(SegmentationLossTest, Branch_BCEWeightZero) {
    SegmentationLoss loss(10, 0.0f, 1.0f);  // BCE weight = 0

    auto pred = createPrediction(2, 10, 64, 64);
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    // Total should equal dice component only when BCE weight is 0
    // BCE component may still be computed but not contribute to total
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(SegmentationLossTest, Branch_BothWeightsZero) {
    SegmentationLoss loss(10, 0.0f, 0.0f);

    auto pred = createPrediction(2, 10, 64, 64);
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    // Total should be 0 when both weights are 0
    EXPECT_FLOAT_EQ(resultMap["total"].item<float>(), 0.0f);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_F(SegmentationLossTest, NumericalStability_LargePredictions) {
    SegmentationLoss loss(10);

    auto pred = torch::randn({ 2, 10, 64, 64 }, device_) * 100.0f;
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(SegmentationLossTest, NumericalStability_SmallPredictions) {
    SegmentationLoss loss(10);

    auto pred = torch::randn({ 2, 10, 64, 64 }, device_) * 0.001f;
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(SegmentationLossTest, NumericalStability_ZeroPredictions) {
    SegmentationLoss loss(10);

    auto pred = torch::zeros({ 2, 10, 64, 64 }, device_);
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

// ============================================================================
// Different Class Counts Tests
// ============================================================================

TEST_F(SegmentationLossTest, Compute_SingleClass) {
    SegmentationLoss loss(1);

    auto pred = createPrediction(2, 1, 64, 64);
    auto target = createTarget(2, 1, 64, 64);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(SegmentationLossTest, Compute_FewClasses) {
    SegmentationLoss loss(3);

    auto pred = createPrediction(2, 3, 64, 64);
    auto target = createTarget(2, 3, 64, 64);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(SegmentationLossTest, Compute_ManyClasses) {
    SegmentationLoss loss(50);

    auto pred = createPrediction(2, 50, 64, 64);
    auto target = createTarget(2, 50, 64, 64);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

// ============================================================================
// Gradient Tests
// ============================================================================

TEST_F(SegmentationLossTest, Gradient_BackwardPass) {
    SegmentationLoss loss(10);

    auto pred = torch::randn({ 2, 10, 64, 64 }, torch::TensorOptions().device(device_).requires_grad(true));
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);
    resultMap["total"].backward();

    EXPECT_TRUE(pred.grad().defined());
    EXPECT_FALSE(pred.grad().isnan().any().item<bool>());
    EXPECT_FALSE(pred.grad().isinf().any().item<bool>());
}

TEST_F(SegmentationLossTest, Gradient_BCEOnly) {
    SegmentationLoss loss(10, 1.0f, 0.0f);

    auto pred = torch::randn({ 2, 10, 64, 64 }, torch::TensorOptions().device(device_).requires_grad(true));
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);
    resultMap["total"].backward();

    EXPECT_TRUE(pred.grad().defined());
    EXPECT_FALSE(pred.grad().isnan().any().item<bool>());
}

TEST_F(SegmentationLossTest, Gradient_DiceOnly) {
    SegmentationLoss loss(10, 0.0f, 1.0f);

    auto pred = torch::randn({ 2, 10, 64, 64 }, torch::TensorOptions().device(device_).requires_grad(true));
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);
    resultMap["total"].backward();

    EXPECT_TRUE(pred.grad().defined());
    EXPECT_FALSE(pred.grad().isnan().any().item<bool>());
}

// ============================================================================
// Return Map Structure Tests
// ============================================================================

TEST_F(SegmentationLossTest, ReturnMap_ContainsAllKeys) {
    SegmentationLoss loss(10);

    auto pred = createPrediction(2, 10, 64, 64);
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    EXPECT_EQ(resultMap.size(), 3);
    EXPECT_TRUE(resultMap.count("bce") == 1);
    EXPECT_TRUE(resultMap.count("dice") == 1);
    EXPECT_TRUE(resultMap.count("total") == 1);
}

TEST_F(SegmentationLossTest, ReturnMap_TensorShapes) {
    SegmentationLoss loss(10);

    auto pred = createPrediction(2, 10, 64, 64);
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    // All values should be scalar tensors
    EXPECT_EQ(resultMap["bce"].dim(), 0);
    EXPECT_EQ(resultMap["dice"].dim(), 0);
    EXPECT_EQ(resultMap["total"].dim(), 0);
}

TEST_F(SegmentationLossTest, ReturnMap_TotalEqualsSum) {
    SegmentationLoss loss(10, 1.0f, 1.0f);

    auto pred = createPrediction(2, 10, 64, 64);
    auto target = createTarget(2, 10, 64, 64);

    auto resultMap = loss.compute(pred, target);

    auto expectedTotal = resultMap["bce"].item<float>() + resultMap["dice"].item<float>();
    EXPECT_NEAR(resultMap["total"].item<float>(), expectedTotal, 1e-5f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SegmentationLossTest, EdgeCase_SmallSpatialSize) {
    SegmentationLoss loss(10);

    auto pred = createPrediction(2, 10, 4, 4);
    auto target = createTarget(2, 10, 4, 4);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(SegmentationLossTest, EdgeCase_LargeSpatialSize) {
    SegmentationLoss loss(10);

    auto pred = createPrediction(1, 10, 512, 512);
    auto target = createTarget(1, 10, 512, 512);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(SegmentationLossTest, EdgeCase_NonSquareSpatial) {
    SegmentationLoss loss(10);

    auto pred = createPrediction(2, 10, 64, 128);
    auto target = createTarget(2, 10, 64, 128);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(SegmentationLossTest, EdgeCase_AllZeroTarget) {
    SegmentationLoss loss(10);

    auto pred = createPrediction(2, 10, 64, 64);
    auto target = torch::zeros({ 2, 10, 64, 64 }, device_);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(SegmentationLossTest, EdgeCase_AllOneTarget) {
    SegmentationLoss loss(10);

    auto pred = createPrediction(2, 10, 64, 64);
    auto target = torch::ones({ 2, 10, 64, 64 }, device_);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}
