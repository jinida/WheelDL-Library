#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Model/Loss/ClassificationLoss.h"
#include "WheelDL.Lib/Data/Dataset/BaseDataset.h"
#include <torch/torch.h>

using namespace WheelDL::Model::Loss;
using namespace WheelDL::Data::Dataset;

/**
 * @brief Classification loss function tests
 *
 * Tests verify all loss function implementations including:
 * - CrossEntropy: Standard classification loss
 * - Focal Loss: Hard example weighting with alpha/gamma
 * - Label Smoothing: Soft targets with epsilon
 */
class ClassificationLossTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Lightweight setup per test - loss functions are stateless
    }

    torch::Tensor createPredictions(int64_t batch, int64_t numClasses)
    {
        return torch::randn(
            {batch, numClasses},
            torch::TensorOptions().requires_grad(true)
        );
    }

    torch::Tensor createTargets(int64_t batch, int64_t numClasses)
    {
        return torch::randint(
            0,
            numClasses,
            {batch}
        );
    }

    torch::Tensor createSoftTargets(int64_t batch, int64_t numClasses)
    {
        auto targets = torch::zeros({batch, numClasses});
        for (int64_t i = 0; i < batch; ++i) {
            int64_t trueClass = torch::randint(0, numClasses, {1}).item<int64_t>();
            targets[i][trueClass] = 1.0f;
        }
        return targets;
    }

    DataExample createDataExample(const torch::Tensor& targets)
    {
        DataExample example;
        example.classes = targets;
        return example;
    }
};

// ============================================================================
// CrossEntropy Loss Tests
// ============================================================================

TEST_F(ClassificationLossTest, CrossEntropy_ValidInput_ReturnsLoss)
{
    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    ClassificationLoss loss(
        ClassificationLoss::LossType::CROSS_ENTROPY,
        10
    );

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(ClassificationLossTest, CrossEntropy_MultiClass_ComputesCorrectly)
{
    int64_t numClasses = 100;
    auto predictions = createPredictions(8, numClasses);
    auto targets = createTargets(8, numClasses);

    ClassificationLoss loss(
        ClassificationLoss::LossType::CROSS_ENTROPY,
        numClasses
    );

    auto result = loss.compute(predictions, targets);

    EXPECT_TRUE(result.find("total") != result.end());
    EXPECT_TRUE(result["total"].defined());
    EXPECT_EQ(result["total"].dim(), 0);
}

TEST_F(ClassificationLossTest, CrossEntropy_BinaryClassification_Works)
{
    auto predictions = createPredictions(16, 2);
    auto targets = createTargets(16, 2);

    ClassificationLoss loss(
        ClassificationLoss::LossType::CROSS_ENTROPY,
        2
    );

    auto result = loss.compute(predictions, targets);

    EXPECT_TRUE(result.find("total") != result.end());
    EXPECT_GE(result["total"].item<float>(), 0.0f);
}

TEST_F(ClassificationLossTest, CrossEntropy_PerfectPrediction_NearZeroLoss)
{
    int64_t batch = 4;
    int64_t numClasses = 10;
    auto targets = createTargets(batch, numClasses);

    // Create perfect predictions (very high logits for correct class)
    auto predictions = torch::full(
        {batch, numClasses},
        -10.0f
    );
    for (int64_t i = 0; i < batch; ++i) {
        int64_t targetClass = targets[i].item<int64_t>();
        predictions.index_put_({i, targetClass}, 10.0f);
    }
    predictions.requires_grad_(true);

    ClassificationLoss loss(
        ClassificationLoss::LossType::CROSS_ENTROPY,
        numClasses
    );

    auto result = loss.compute(predictions, targets);

    EXPECT_LT(result["total"].item<float>(), 0.1f);
}

TEST_F(ClassificationLossTest, CrossEntropy_WithDataExample_Works)
{
    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);
    auto example = createDataExample(targets);

    ClassificationLoss loss(
        ClassificationLoss::LossType::CROSS_ENTROPY,
        10
    );

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, example);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

// ============================================================================
// Focal Loss Tests
// ============================================================================

TEST_F(ClassificationLossTest, FocalLoss_ValidInput_ReturnsLoss)
{
    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    ClassificationLoss loss(
        ClassificationLoss::LossType::FOCAL,
        10,
        0.0f,     // label smoothing
        0.25f,    // focal alpha
        2.0f      // focal gamma
    );

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(ClassificationLossTest, FocalLoss_GammaParameter_AffectsLoss)
{
    auto predictions = createPredictions(8, 10);
    auto targets = createTargets(8, 10);

    ClassificationLoss loss1(
        ClassificationLoss::LossType::FOCAL,
        10,
        0.0f,
        0.25f,
        1.0f
    );

    ClassificationLoss loss2(
        ClassificationLoss::LossType::FOCAL,
        10,
        0.0f,
        0.25f,
        3.0f
    );

    auto result1 = loss1.compute(predictions, targets);
    auto result2 = loss2.compute(predictions, targets);

    // Different gamma should produce different loss values
    EXPECT_NE(
        result1["total"].item<float>(),
        result2["total"].item<float>()
    );
}

TEST_F(ClassificationLossTest, FocalLoss_HardExamples_HigherWeight)
{
    int64_t batch = 4;
    int64_t numClasses = 10;
    auto targets = torch::zeros({batch}, torch::kLong);

    // Easy example: high confidence correct prediction
    auto easyPred = torch::full(
        {1, numClasses},
        -5.0f
    );
    easyPred.index_put_({0, 0}, 5.0f);
    easyPred.requires_grad_(true);

    // Hard example: low confidence correct prediction
    auto hardPred = torch::full(
        {1, numClasses},
        0.0f
    );
    hardPred.index_put_({0, 0}, 0.5f);
    hardPred.requires_grad_(true);

    ClassificationLoss loss(
        ClassificationLoss::LossType::FOCAL,
        numClasses,
        0.0f,
        0.25f,
        2.0f
    );

    auto easyResult = loss.compute(easyPred, targets[0].unsqueeze(0));
    auto hardResult = loss.compute(hardPred, targets[0].unsqueeze(0));

    // Focal loss should weight hard examples more
    EXPECT_GT(
        hardResult["total"].item<float>(),
        easyResult["total"].item<float>()
    );
}

TEST_F(ClassificationLossTest, FocalLoss_SetParameters_UpdatesCorrectly)
{
    ClassificationLoss loss(
        ClassificationLoss::LossType::FOCAL,
        10
    );

    EXPECT_NO_THROW({
        loss.setFocalParams(0.5f, 3.0f);
    });
}

TEST_F(ClassificationLossTest, FocalLoss_InvalidGamma_Throws)
{
    ClassificationLoss loss(
        ClassificationLoss::LossType::FOCAL,
        10
    );

    EXPECT_THROW(
        loss.setFocalParams(0.25f, 6.0f),
        std::invalid_argument
    );
}

// ============================================================================
// Label Smoothing Tests
// ============================================================================

TEST_F(ClassificationLossTest, LabelSmoothing_ValidInput_ReturnsLoss)
{
    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    ClassificationLoss loss(
        ClassificationLoss::LossType::LABEL_SMOOTHING,
        10,
        0.1f
    );

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(ClassificationLossTest, LabelSmoothing_EpsilonParameter_AffectsLoss)
{
    auto predictions = createPredictions(8, 10);
    auto targets = createTargets(8, 10);

    ClassificationLoss loss1(
        ClassificationLoss::LossType::LABEL_SMOOTHING,
        10,
        0.0f
    );

    ClassificationLoss loss2(
        ClassificationLoss::LossType::LABEL_SMOOTHING,
        10,
        0.2f
    );

    auto result1 = loss1.compute(predictions, targets);
    auto result2 = loss2.compute(predictions, targets);

    // Different smoothing should produce different loss
    EXPECT_NE(
        result1["total"].item<float>(),
        result2["total"].item<float>()
    );
}

TEST_F(ClassificationLossTest, LabelSmoothing_SoftTargets_Works)
{
    auto predictions = createPredictions(4, 10);
    auto softTargets = createSoftTargets(4, 10);

    ClassificationLoss loss(
        ClassificationLoss::LossType::LABEL_SMOOTHING,
        10,
        0.1f
    );

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, softTargets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

TEST_F(ClassificationLossTest, LabelSmoothing_SetSmoothing_UpdatesCorrectly)
{
    ClassificationLoss loss(
        ClassificationLoss::LossType::LABEL_SMOOTHING,
        10
    );

    EXPECT_NO_THROW({
        loss.setLabelSmoothing(0.15f);
    });
}

// ============================================================================
// Loss Value Tests
// ============================================================================

TEST_F(ClassificationLossTest, AllLossTypes_ValuesNonNegative)
{
    auto predictions = createPredictions(8, 10);
    auto targets = createTargets(8, 10);

    std::vector<ClassificationLoss::LossType> lossTypes = {
        ClassificationLoss::LossType::CROSS_ENTROPY,
        ClassificationLoss::LossType::FOCAL,
        ClassificationLoss::LossType::LABEL_SMOOTHING
    };

    for (auto lossType : lossTypes) {
        ClassificationLoss loss(lossType, 10);
        auto result = loss.compute(predictions, targets);

        EXPECT_GE(result["total"].item<float>(), 0.0f);
    }
}

TEST_F(ClassificationLossTest, LossValues_ReasonableRange)
{
    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    ClassificationLoss loss(
        ClassificationLoss::LossType::CROSS_ENTROPY,
        10
    );

    auto result = loss.compute(predictions, targets);

    // For random predictions on 10 classes, loss should be around -log(0.1) ≈ 2.3
    // Allow reasonable range
    EXPECT_LT(result["total"].item<float>(), 10.0f);
    EXPECT_GT(result["total"].item<float>(), 0.0f);
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

TEST_F(ClassificationLossTest, GradientFlow_BackwardPass_Works)
{
    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    ClassificationLoss loss(
        ClassificationLoss::LossType::CROSS_ENTROPY,
        10
    );

    auto result = loss.compute(predictions, targets);

    EXPECT_NO_THROW({
        result["total"].backward();
        EXPECT_TRUE(predictions.grad().defined());
        EXPECT_GT(predictions.grad().abs().sum().item<float>(), 0.0f);
    });
}

TEST_F(ClassificationLossTest, GradientFlow_FocalLoss_Works)
{
    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    ClassificationLoss loss(
        ClassificationLoss::LossType::FOCAL,
        10
    );

    auto result = loss.compute(predictions, targets);

    EXPECT_NO_THROW({
        result["total"].backward();
        EXPECT_TRUE(predictions.grad().defined());
    });
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ClassificationLossTest, EdgeCase_SingleSample_Works)
{
    auto predictions = createPredictions(1, 10);
    auto targets = createTargets(1, 10);

    ClassificationLoss loss(
        ClassificationLoss::LossType::CROSS_ENTROPY,
        10
    );

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

TEST_F(ClassificationLossTest, EdgeCase_LargeBatch_Works)
{
    auto predictions = createPredictions(256, 10);
    auto targets = createTargets(256, 10);

    ClassificationLoss loss(
        ClassificationLoss::LossType::CROSS_ENTROPY,
        10
    );

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

TEST_F(ClassificationLossTest, EdgeCase_ManyClasses_Works)
{
    int64_t numClasses = 1000;
    auto predictions = createPredictions(4, numClasses);
    auto targets = createTargets(4, numClasses);

    ClassificationLoss loss(
        ClassificationLoss::LossType::CROSS_ENTROPY,
        numClasses
    );

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

// ============================================================================
// Output Format Tests
// ============================================================================

TEST_F(ClassificationLossTest, OutputFormat_ContainsTotalKey)
{
    auto predictions = createPredictions(4, 10);
    auto targets = createTargets(4, 10);

    ClassificationLoss loss(
        ClassificationLoss::LossType::CROSS_ENTROPY,
        10
    );

    auto result = loss.compute(predictions, targets);

    EXPECT_TRUE(result.find("total") != result.end());
    EXPECT_EQ(result.size(), 1);
}

TEST_F(ClassificationLossTest, LossName_ReturnsCorrectString)
{
    ClassificationLoss ceLoss(
        ClassificationLoss::LossType::CROSS_ENTROPY,
        10
    );
    ClassificationLoss focalLoss(
        ClassificationLoss::LossType::FOCAL,
        10
    );
    ClassificationLoss lsLoss(
        ClassificationLoss::LossType::LABEL_SMOOTHING,
        10
    );

    EXPECT_EQ(ceLoss.name(), "ClassificationLoss(CrossEntropy)");
    EXPECT_EQ(focalLoss.name(), "ClassificationLoss(Focal)");
    EXPECT_EQ(lsLoss.name(), "ClassificationLoss(LabelSmoothing)");
}

TEST_F(ClassificationLossTest, GetLossType_ReturnsCorrectType)
{
    ClassificationLoss loss(
        ClassificationLoss::LossType::FOCAL,
        10
    );

    EXPECT_EQ(
        loss.getLossType(),
        ClassificationLoss::LossType::FOCAL
    );
}
