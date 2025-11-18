#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Model/Loss/SegmentationLoss.h"
#include "WheelDL.Lib/Data/Dataset/BaseDataset.h"
#include <torch/torch.h>

using namespace WheelDL::Model::Loss;
using namespace WheelDL::Data::Dataset;

/**
 * @brief Segmentation loss function tests
 *
 * Tests verify segmentation loss including:
 * - BCE Loss: Binary cross-entropy with logits
 * - Dice Loss: Better boundary handling
 * - Combined Loss: BCE + Dice with weights
 */
class SegmentationLossTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Lightweight setup per test
        numClasses = 21;  // e.g., PASCAL VOC
        height = 64;
        width = 64;
    }

    torch::Tensor createPredictions(int64_t batch)
    {
        // Logits before sigmoid [N, num_classes, H, W]
        return torch::randn(
            {batch, numClasses, height, width},
            torch::TensorOptions().requires_grad(true)
        );
    }

    torch::Tensor createTargets(int64_t batch)
    {
        // Binary masks [N, num_classes, H, W]
        return torch::randint(
            0,
            2,
            {batch, numClasses, height, width},
            torch::TensorOptions().dtype(torch::kFloat32)
        );
    }

    torch::Tensor createPerfectTargets(int64_t batch)
    {
        // Create targets where one class is active per pixel
        auto targets = torch::zeros({batch, numClasses, height, width});

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t h = 0; h < height; ++h) {
                for (int64_t w = 0; w < width; ++w) {
                    int64_t classIdx = (h * width + w) % numClasses;
                    targets[b][classIdx][h][w] = 1.0f;
                }
            }
        }

        return targets;
    }

    torch::Tensor createImbalancedTargets(int64_t batch)
    {
        // 99% background (class 0), 1% foreground
        auto targets = torch::zeros({batch, numClasses, height, width});

        // Set a few pixels to foreground classes
        int64_t fgPixels = static_cast<int64_t>(height * width * 0.01f);
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t i = 0; i < fgPixels; ++i) {
                int64_t h = torch::randint(0, height, {1}).item<int64_t>();
                int64_t w = torch::randint(0, width, {1}).item<int64_t>();
                int64_t c = torch::randint(1, numClasses, {1}).item<int64_t>();
                targets[b][c][h][w] = 1.0f;
            }
        }

        return targets;
    }

    DataExample createDataExample(const torch::Tensor& targets)
    {
        DataExample example;
        example.targets = targets;
        return example;
    }

    int64_t numClasses;
    int64_t height;
    int64_t width;
};

// ============================================================================
// BCE Loss Tests
// ============================================================================

TEST_F(SegmentationLossTest, BCE_ValidInput_ReturnsLoss)
{
    SegmentationLoss loss(
        numClasses,
        1.0f,   // BCE weight
        0.0f    // Dice weight (only BCE)
    );

    auto predictions = createPredictions(2);
    auto targets = createTargets(2);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        EXPECT_TRUE(result.find("bce") != result.end());
        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["bce"].item<float>(), 0.0f);
    });
}

TEST_F(SegmentationLossTest, BCE_PixelWise_ComputesCorrectly)
{
    SegmentationLoss loss(numClasses, 1.0f, 0.0f);

    auto predictions = createPredictions(1);
    auto targets = createTargets(1);

    auto result = loss.compute(predictions, targets);

    EXPECT_TRUE(result.find("bce") != result.end());
    EXPECT_TRUE(result["bce"].defined());
    EXPECT_EQ(result["bce"].dim(), 0);  // Scalar
}

TEST_F(SegmentationLossTest, BCE_PerfectPrediction_NearZeroLoss)
{
    SegmentationLoss loss(numClasses, 1.0f, 0.0f);

    auto targets = createTargets(1);

    // Create perfect predictions (high logits for target=1, low for target=0)
    auto predictions = torch::where(
        targets > 0.5f,
        torch::full_like(targets, 10.0f),
        torch::full_like(targets, -10.0f)
    );
    predictions.set_requires_grad(true);

    auto result = loss.compute(predictions, targets);

    EXPECT_LT(result["bce"].item<float>(), 0.1f);
}

TEST_F(SegmentationLossTest, BCE_AllWrong_HighLoss)
{
    SegmentationLoss loss(numClasses, 1.0f, 0.0f);

    auto targets = createTargets(1);

    // Create completely wrong predictions
    auto predictions = torch::where(
        targets > 0.5f,
        torch::full_like(targets, -10.0f),  // Low logit for target=1
        torch::full_like(targets, 10.0f)    // High logit for target=0
    );
    predictions.set_requires_grad(true);

    auto result = loss.compute(predictions, targets);

    // Should have high loss
    EXPECT_GT(result["bce"].item<float>(), 5.0f);
}

// ============================================================================
// Dice Loss Tests
// ============================================================================

TEST_F(SegmentationLossTest, Dice_ValidInput_ReturnsLoss)
{
    SegmentationLoss loss(
        numClasses,
        0.0f,   // BCE weight (only Dice)
        1.0f    // Dice weight
    );

    auto predictions = createPredictions(2);
    auto targets = createTargets(2);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        EXPECT_TRUE(result.find("dice") != result.end());
        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["dice"].item<float>(), 0.0f);
    });
}

TEST_F(SegmentationLossTest, Dice_PerfectOverlap_NearZeroLoss)
{
    SegmentationLoss loss(numClasses, 0.0f, 1.0f);

    auto targets = createTargets(1);

    // Create perfect predictions
    auto predictions = torch::where(
        targets > 0.5f,
        torch::full_like(targets, 10.0f),
        torch::full_like(targets, -10.0f)
    );
    predictions.set_requires_grad(true);

    auto result = loss.compute(predictions, targets);

    EXPECT_LT(result["dice"].item<float>(), 0.1f);
}

TEST_F(SegmentationLossTest, Dice_NoOverlap_HighLoss)
{
    SegmentationLoss loss(numClasses, 0.0f, 1.0f);

    auto targets = createTargets(1);

    // Create predictions with no overlap
    auto predictions = torch::where(
        targets > 0.5f,
        torch::full_like(targets, -10.0f),
        torch::full_like(targets, 10.0f)
    );
    predictions.set_requires_grad(true);

    auto result = loss.compute(predictions, targets);

    // Dice loss for no overlap should be close to 1.0
    EXPECT_GT(result["dice"].item<float>(), 0.5f);
}

TEST_F(SegmentationLossTest, Dice_EmptyMask_HandlesGracefully)
{
    SegmentationLoss loss(numClasses, 0.0f, 1.0f);

    auto predictions = createPredictions(1);
    auto targets = torch::zeros({1, numClasses, height, width});

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("dice") != result.end());
    });
}

// ============================================================================
// Combined Loss Tests
// ============================================================================

TEST_F(SegmentationLossTest, Combined_BCEAndDice_ReturnsAllComponents)
{
    SegmentationLoss loss(
        numClasses,
        1.0f,   // BCE weight
        1.0f    // Dice weight
    );

    auto predictions = createPredictions(2);
    auto targets = createTargets(2);

    auto result = loss.compute(predictions, targets);

    EXPECT_TRUE(result.find("bce") != result.end());
    EXPECT_TRUE(result.find("dice") != result.end());
    EXPECT_TRUE(result.find("total") != result.end());
}

TEST_F(SegmentationLossTest, Combined_WeightedSum_ComputesCorrectly)
{
    float bceWeight = 0.7f;
    float diceWeight = 0.3f;

    SegmentationLoss loss(numClasses, bceWeight, diceWeight);

    auto predictions = createPredictions(2);
    auto targets = createTargets(2);

    auto result = loss.compute(predictions, targets);

    float expectedTotal =
        result["bce"].item<float>() * bceWeight +
        result["dice"].item<float>() * diceWeight;

    EXPECT_NEAR(
        result["total"].item<float>(),
        expectedTotal,
        1e-5f
    );
}

TEST_F(SegmentationLossTest, Combined_DifferentWeights_AffectTotal)
{
    auto predictions = createPredictions(2);
    auto targets = createTargets(2);

    SegmentationLoss loss1(numClasses, 1.0f, 0.5f);
    SegmentationLoss loss2(numClasses, 0.5f, 1.0f);

    auto result1 = loss1.compute(predictions, targets);
    auto result2 = loss2.compute(predictions, targets);

    // Different weights should produce different total loss
    EXPECT_NE(
        result1["total"].item<float>(),
        result2["total"].item<float>()
    );
}

TEST_F(SegmentationLossTest, Combined_SetWeights_UpdatesCorrectly)
{
    SegmentationLoss loss(numClasses);

    loss.setLossWeights(0.6f, 0.4f);

    auto [bceWeight, diceWeight] = loss.getLossWeights();

    EXPECT_FLOAT_EQ(bceWeight, 0.6f);
    EXPECT_FLOAT_EQ(diceWeight, 0.4f);
}

TEST_F(SegmentationLossTest, Combined_WithDataExample_Works)
{
    SegmentationLoss loss(numClasses);

    auto predictions = createPredictions(2);
    auto targets = createTargets(2);
    auto example = createDataExample(targets);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, example);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

// ============================================================================
// Loss Value Tests
// ============================================================================

TEST_F(SegmentationLossTest, LossValues_AllNonNegative)
{
    SegmentationLoss loss(numClasses);

    auto predictions = createPredictions(2);
    auto targets = createTargets(2);

    auto result = loss.compute(predictions, targets);

    EXPECT_GE(result["bce"].item<float>(), 0.0f);
    EXPECT_GE(result["dice"].item<float>(), 0.0f);
    EXPECT_GE(result["total"].item<float>(), 0.0f);
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

TEST_F(SegmentationLossTest, GradientFlow_BCE_BackwardWorks)
{
    SegmentationLoss loss(numClasses, 1.0f, 0.0f);

    auto predictions = createPredictions(2);
    auto targets = createTargets(2);

    auto result = loss.compute(predictions, targets);

    EXPECT_NO_THROW({
        result["total"].backward();
        EXPECT_TRUE(predictions.grad().defined());
        EXPECT_GT(predictions.grad().abs().sum().item<float>(), 0.0f);
    });
}

TEST_F(SegmentationLossTest, GradientFlow_Dice_BackwardWorks)
{
    SegmentationLoss loss(numClasses, 0.0f, 1.0f);

    auto predictions = createPredictions(2);
    auto targets = createTargets(2);

    auto result = loss.compute(predictions, targets);

    EXPECT_NO_THROW({
        result["total"].backward();
        EXPECT_TRUE(predictions.grad().defined());
        EXPECT_GT(predictions.grad().abs().sum().item<float>(), 0.0f);
    });
}

TEST_F(SegmentationLossTest, GradientFlow_Combined_BackwardWorks)
{
    SegmentationLoss loss(numClasses);

    auto predictions = createPredictions(2);
    auto targets = createTargets(2);

    auto result = loss.compute(predictions, targets);

    EXPECT_NO_THROW({
        result["total"].backward();
        EXPECT_TRUE(predictions.grad().defined());
    });
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SegmentationLossTest, EdgeCase_ExtremeImbalance_HandlesCorrectly)
{
    SegmentationLoss loss(numClasses);

    auto predictions = createPredictions(2);
    auto targets = createImbalancedTargets(2);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(SegmentationLossTest, EdgeCase_SinglePixelMask_Works)
{
    SegmentationLoss loss(numClasses);

    auto predictions = createPredictions(1);
    auto targets = torch::zeros({1, numClasses, height, width});

    // Set single pixel to 1
    targets[0][5][10][20] = 1.0f;

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

TEST_F(SegmentationLossTest, EdgeCase_SingleBatch_Works)
{
    SegmentationLoss loss(numClasses);

    auto predictions = createPredictions(1);
    auto targets = createTargets(1);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

TEST_F(SegmentationLossTest, EdgeCase_LargeBatch_Works)
{
    SegmentationLoss loss(numClasses);

    auto predictions = createPredictions(16);
    auto targets = createTargets(16);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

TEST_F(SegmentationLossTest, EdgeCase_SingleClass_Works)
{
    SegmentationLoss loss(1);

    auto predictions = torch::randn(
        {2, 1, height, width},
        torch::TensorOptions().requires_grad(true)
    );
    auto targets = torch::randint(
        0,
        2,
        {2, 1, height, width},
        torch::TensorOptions().dtype(torch::kFloat32)
    );

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

// ============================================================================
// Smoothing Factor Tests
// ============================================================================

TEST_F(SegmentationLossTest, SmoothFactor_SetValid_UpdatesCorrectly)
{
    SegmentationLoss loss(numClasses);

    EXPECT_NO_THROW({
        loss.setSmooth(1e-5f);
    });
}

TEST_F(SegmentationLossTest, SmoothFactor_SetInvalid_Throws)
{
    SegmentationLoss loss(numClasses);

    EXPECT_THROW(
        loss.setSmooth(0.0f),
        std::invalid_argument
    );

    EXPECT_THROW(
        loss.setSmooth(-1e-6f),
        std::invalid_argument
    );
}

// ============================================================================
// Name Test
// ============================================================================

TEST_F(SegmentationLossTest, Name_ReturnsCorrectString)
{
    SegmentationLoss loss(numClasses);

    EXPECT_EQ(loss.name(), "SegmentationLoss");
}
