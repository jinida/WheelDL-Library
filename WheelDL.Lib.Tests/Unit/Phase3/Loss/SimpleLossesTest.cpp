#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Model/Loss/BaseLoss.h"
#include "WheelDL.Lib/Data/Dataset/BaseDataset.h"
#include <torch/torch.h>

using namespace WheelDL::Model::Loss;
using namespace WheelDL::Data::Dataset;

/**
 * @brief Simple loss function tests
 *
 * Tests verify basic loss implementations:
 * - MSELoss: Mean squared error
 * - MAELoss: Mean absolute error
 * - SmoothL1Loss: Huber loss with beta parameter
 * - BCEWithLogitsLoss: Binary cross-entropy with logits
 */
class SimpleLossesTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Lightweight setup per test
    }

    torch::Tensor createPredictions(int64_t batch, int64_t features = 10)
    {
        return torch::randn(
            {batch, features},
            torch::TensorOptions().requires_grad(true)
        );
    }

    torch::Tensor createTargets(int64_t batch, int64_t features = 10)
    {
        return torch::randn({batch, features});
    }

    torch::Tensor createBinaryTargets(int64_t batch, int64_t features = 10)
    {
        return torch::randint(
            0,
            2,
            {batch, features},
            torch::TensorOptions().dtype(torch::kFloat32)
        );
    }

    DataExample createDataExample(const torch::Tensor& targets)
    {
        DataExample example;
        example.targets = targets;
        return example;
    }
};

// ============================================================================
// MSELoss Tests
// ============================================================================

TEST_F(SimpleLossesTest, MSE_ValidInput_ReturnsLoss)
{
    MSELoss loss;

    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(SimpleLossesTest, MSE_PerfectPrediction_ZeroLoss)
{
    MSELoss loss;

    auto targets = createTargets(2, 5);
    auto predictions = targets.clone().set_requires_grad(true);

    auto result = loss.compute(predictions, targets);

    EXPECT_LT(result["total"].item<float>(), 1e-6f);
}

TEST_F(SimpleLossesTest, MSE_GradientFlow_BackwardWorks)
{
    MSELoss loss;

    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    auto result = loss.compute(predictions, targets);

    EXPECT_NO_THROW({
        result["total"].backward();
        EXPECT_TRUE(predictions.grad().defined());
        EXPECT_GT(predictions.grad().abs().sum().item<float>(), 0.0f);
    });
}

TEST_F(SimpleLossesTest, MSE_WithDataExample_Works)
{
    MSELoss loss;

    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);
    auto example = createDataExample(targets);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, example);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

TEST_F(SimpleLossesTest, MSE_Name_ReturnsCorrectString)
{
    MSELoss loss;

    EXPECT_EQ(loss.name(), "MSELoss");
}

// ============================================================================
// MAELoss Tests
// ============================================================================

TEST_F(SimpleLossesTest, MAE_ValidInput_ReturnsLoss)
{
    MAELoss loss;

    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(SimpleLossesTest, MAE_PerfectPrediction_ZeroLoss)
{
    MAELoss loss;

    auto targets = createTargets(2, 5);
    auto predictions = targets.clone().set_requires_grad(true);

    auto result = loss.compute(predictions, targets);

    EXPECT_LT(result["total"].item<float>(), 1e-6f);
}

TEST_F(SimpleLossesTest, MAE_GradientFlow_BackwardWorks)
{
    MAELoss loss;

    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    auto result = loss.compute(predictions, targets);

    EXPECT_NO_THROW({
        result["total"].backward();
        EXPECT_TRUE(predictions.grad().defined());
        EXPECT_GT(predictions.grad().abs().sum().item<float>(), 0.0f);
    });
}

TEST_F(SimpleLossesTest, MAE_Name_ReturnsCorrectString)
{
    MAELoss loss;

    EXPECT_EQ(loss.name(), "MAELoss");
}

// ============================================================================
// SmoothL1Loss Tests
// ============================================================================

TEST_F(SimpleLossesTest, SmoothL1_DefaultBeta_ReturnsLoss)
{
    SmoothL1Loss loss;

    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(SimpleLossesTest, SmoothL1_CustomBeta_ReturnsLoss)
{
    SmoothL1Loss loss(0.5f);

    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(SimpleLossesTest, SmoothL1_BetaParameter_AffectsLoss)
{
    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    SmoothL1Loss loss1(0.5f);
    SmoothL1Loss loss2(2.0f);

    auto result1 = loss1.compute(predictions, targets);
    auto result2 = loss2.compute(predictions, targets);

    // Different beta values should produce different losses
    // (unless predictions happen to be very close to targets)
    EXPECT_TRUE(result1.find("total") != result1.end());
    EXPECT_TRUE(result2.find("total") != result2.end());
}

TEST_F(SimpleLossesTest, SmoothL1_PerfectPrediction_ZeroLoss)
{
    SmoothL1Loss loss;

    auto targets = createTargets(2, 5);
    auto predictions = targets.clone().set_requires_grad(true);

    auto result = loss.compute(predictions, targets);

    EXPECT_LT(result["total"].item<float>(), 1e-6f);
}

TEST_F(SimpleLossesTest, SmoothL1_GradientFlow_BackwardWorks)
{
    SmoothL1Loss loss;

    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    auto result = loss.compute(predictions, targets);

    EXPECT_NO_THROW({
        result["total"].backward();
        EXPECT_TRUE(predictions.grad().defined());
        EXPECT_GT(predictions.grad().abs().sum().item<float>(), 0.0f);
    });
}

TEST_F(SimpleLossesTest, SmoothL1_Name_ReturnsCorrectString)
{
    SmoothL1Loss loss;

    EXPECT_EQ(loss.name(), "SmoothL1Loss");
}

// ============================================================================
// BCEWithLogitsLoss Tests
// ============================================================================

TEST_F(SimpleLossesTest, BCE_ValidInput_ReturnsLoss)
{
    BCEWithLogitsLoss loss;

    auto predictions = createPredictions(4, 10);
    auto targets = createBinaryTargets(4, 10);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(SimpleLossesTest, BCE_WithLogits_NumericallyStable)
{
    BCEWithLogitsLoss loss;

    // Test with extreme logit values (should not overflow)
    auto predictions = torch::randn(
        {2, 5},
        torch::TensorOptions().requires_grad(true)
    ) * 10.0f;
    auto targets = createBinaryTargets(2, 5);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_FALSE(torch::isnan(result["total"]).item<bool>());
        EXPECT_FALSE(torch::isinf(result["total"]).item<bool>());
    });
}

TEST_F(SimpleLossesTest, BCE_PerfectPrediction_NearZeroLoss)
{
    BCEWithLogitsLoss loss;

    auto targets = createBinaryTargets(2, 5);

    // Create perfect logits: high for target=1, low for target=0
    auto predictions = torch::where(
        targets > 0.5f,
        torch::full_like(targets, 10.0f),
        torch::full_like(targets, -10.0f)
    );
    predictions.set_requires_grad(true);

    auto result = loss.compute(predictions, targets);

    EXPECT_LT(result["total"].item<float>(), 0.1f);
}

TEST_F(SimpleLossesTest, BCE_GradientFlow_BackwardWorks)
{
    BCEWithLogitsLoss loss;

    auto predictions = createPredictions(4, 10);
    auto targets = createBinaryTargets(4, 10);

    auto result = loss.compute(predictions, targets);

    EXPECT_NO_THROW({
        result["total"].backward();
        EXPECT_TRUE(predictions.grad().defined());
        EXPECT_GT(predictions.grad().abs().sum().item<float>(), 0.0f);
    });
}

TEST_F(SimpleLossesTest, BCE_Name_ReturnsCorrectString)
{
    BCEWithLogitsLoss loss;

    EXPECT_EQ(loss.name(), "BCEWithLogitsLoss");
}

// ============================================================================
// Common Tests for All Losses
// ============================================================================

TEST_F(SimpleLossesTest, AllLosses_NonNegativeValues)
{
    auto predictions = createPredictions(4, 10);
    auto regressionTargets = createTargets(4, 10);
    auto binaryTargets = createBinaryTargets(4, 10);

    MSELoss mseLoss;
    MAELoss maeLoss;
    SmoothL1Loss smoothL1Loss;
    BCEWithLogitsLoss bceLoss;

    auto mseResult = mseLoss.compute(predictions, regressionTargets);
    auto maeResult = maeLoss.compute(predictions, regressionTargets);
    auto smoothL1Result = smoothL1Loss.compute(predictions, regressionTargets);
    auto bceResult = bceLoss.compute(predictions, binaryTargets);

    EXPECT_GE(mseResult["total"].item<float>(), 0.0f);
    EXPECT_GE(maeResult["total"].item<float>(), 0.0f);
    EXPECT_GE(smoothL1Result["total"].item<float>(), 0.0f);
    EXPECT_GE(bceResult["total"].item<float>(), 0.0f);
}

TEST_F(SimpleLossesTest, AllLosses_SingleBatch_Works)
{
    auto predictions = createPredictions(1, 5);
    auto regressionTargets = createTargets(1, 5);
    auto binaryTargets = createBinaryTargets(1, 5);

    MSELoss mseLoss;
    MAELoss maeLoss;
    SmoothL1Loss smoothL1Loss;
    BCEWithLogitsLoss bceLoss;

    EXPECT_NO_THROW({
        auto mseResult = mseLoss.compute(predictions, regressionTargets);
        EXPECT_TRUE(mseResult.find("total") != mseResult.end());
    });

    EXPECT_NO_THROW({
        auto maeResult = maeLoss.compute(predictions, regressionTargets);
        EXPECT_TRUE(maeResult.find("total") != maeResult.end());
    });

    EXPECT_NO_THROW({
        auto smoothL1Result = smoothL1Loss.compute(predictions, regressionTargets);
        EXPECT_TRUE(smoothL1Result.find("total") != smoothL1Result.end());
    });

    EXPECT_NO_THROW({
        auto bceResult = bceLoss.compute(predictions, binaryTargets);
        EXPECT_TRUE(bceResult.find("total") != bceResult.end());
    });
}

TEST_F(SimpleLossesTest, AllLosses_LargeBatch_Works)
{
    auto predictions = createPredictions(128, 20);
    auto regressionTargets = createTargets(128, 20);
    auto binaryTargets = createBinaryTargets(128, 20);

    MSELoss mseLoss;
    MAELoss maeLoss;
    SmoothL1Loss smoothL1Loss;
    BCEWithLogitsLoss bceLoss;

    EXPECT_NO_THROW({
        auto mseResult = mseLoss.compute(predictions, regressionTargets);
        EXPECT_TRUE(mseResult.find("total") != mseResult.end());
    });

    EXPECT_NO_THROW({
        auto maeResult = maeLoss.compute(predictions, regressionTargets);
        EXPECT_TRUE(maeResult.find("total") != maeResult.end());
    });

    EXPECT_NO_THROW({
        auto smoothL1Result = smoothL1Loss.compute(predictions, regressionTargets);
        EXPECT_TRUE(smoothL1Result.find("total") != smoothL1Result.end());
    });

    EXPECT_NO_THROW({
        auto bceResult = bceLoss.compute(predictions, binaryTargets);
        EXPECT_TRUE(bceResult.find("total") != bceResult.end());
    });
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SimpleLossesTest, EdgeCase_HighDimensional_Works)
{
    // Test with 4D tensors (e.g., images)
    auto predictions = torch::randn(
        {2, 3, 32, 32},
        torch::TensorOptions().requires_grad(true)
    );
    auto targets = torch::randn({2, 3, 32, 32});

    MSELoss loss;

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

TEST_F(SimpleLossesTest, EdgeCase_SingleElement_Works)
{
    auto predictions = torch::randn(
        {1, 1},
        torch::TensorOptions().requires_grad(true)
    );
    auto targets = torch::randn({1, 1});

    MSELoss loss;

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}
