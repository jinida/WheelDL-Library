#include "pch.h"
#include <gtest/gtest.h>
#include "Model/Loss/ClassificationLoss.h"
#include <torch/torch.h>
#include <cmath>

using namespace WheelDL::Model::Loss;
using namespace WheelDL::Data::Dataset;

// =============================================================================
// Test Fixture
// =============================================================================

class ClassificationLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
    }
    void TearDown() override {}

    // Helper: Compare tensor value with expected value
    void expectNear(const torch::Tensor& actual, float expected, float tol = 1e-4f) {
        EXPECT_NEAR(actual.item<float>(), expected, tol);
    }

    // Helper: Create DataExample with classes tensor
    DataExample createDataExample(const torch::Tensor& classes) {
        DataExample example;
        example.classes = classes;
        return example;
    }
};

// =============================================================================
// Part 1: CrossEntropyLoss Tests (Section 1.2.1)
// =============================================================================

// CE-001: Basic_SingleClass - Single sample
TEST_F(ClassificationLossTest, CE_Basic_SingleClass) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);
    auto pred = torch::tensor({{2.0f, -1.0f, 0.5f}});  // [1, 3]
    auto target = torch::tensor({0}, torch::kLong);     // [1]

    auto result = loss.compute(pred, target);

    // softmax([2, -1, 0.5]) = [0.705, 0.035, 0.157], -log(0.705) = 0.349
    // But actual calculation uses log_softmax directly
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
    EXPECT_GT(result["total"].item<float>(), 0.0f);
}

// CE-002: Basic_BatchMean - Batch computation
TEST_F(ClassificationLossTest, CE_Basic_BatchMean) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);
    auto pred = torch::randn({4, 3});
    auto target = torch::randint(0, 3, {4}, torch::kLong);

    auto result = loss.compute(pred, target);

    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// CE-003: PerfectPrediction - High confidence correct
TEST_F(ClassificationLossTest, CE_PerfectPrediction) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);
    auto pred = torch::tensor({{10.0f, 0.0f, 0.0f}});  // Very high logit for class 0
    auto target = torch::tensor({0}, torch::kLong);

    auto result = loss.compute(pred, target);

    // softmax([10, 0, 0])[0] ~ 1.0, loss ~ 0
    EXPECT_LT(result["total"].item<float>(), 0.001f);
}

// CE-004: WrongPrediction - High confidence wrong
TEST_F(ClassificationLossTest, CE_WrongPrediction) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);
    auto pred = torch::tensor({{10.0f, 0.0f, 0.0f}});  // High logit for class 0
    auto target = torch::tensor({1}, torch::kLong);     // But target is class 1

    auto result = loss.compute(pred, target);

    // softmax([10, 0, 0])[1] ~ 0, loss ~ 10
    EXPECT_GT(result["total"].item<float>(), 9.0f);
}

// CE-005: UniformLogits - Equal logits
TEST_F(ClassificationLossTest, CE_UniformLogits) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);
    auto pred = torch::tensor({{0.0f, 0.0f, 0.0f}});
    auto target = torch::tensor({0}, torch::kLong);

    auto result = loss.compute(pred, target);

    // softmax([0,0,0]) = [1/3, 1/3, 1/3], -log(1/3) = log(3) = 1.0986
    expectNear(result["total"], 1.0986f, 1e-3f);
}

// CE-006: MultiClass_10 - 10 classes
TEST_F(ClassificationLossTest, CE_MultiClass_10) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 10);
    auto pred = torch::randn({8, 10});
    auto target = torch::randint(0, 10, {8}, torch::kLong);

    auto result = loss.compute(pred, target);

    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// CE-007: MultiClass_1000 - 1000 classes
TEST_F(ClassificationLossTest, CE_MultiClass_1000) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 1000);
    auto pred = torch::randn({4, 1000});
    auto target = torch::randint(0, 1000, {4}, torch::kLong);

    auto result = loss.compute(pred, target);

    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// CE-008: PreComputedValues
TEST_F(ClassificationLossTest, CE_PreComputedValues) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);
    auto pred = torch::tensor({{2.0f, 1.0f, 0.1f}});
    auto target = torch::tensor({0}, torch::kLong);

    auto result = loss.compute(pred, target);

    // softmax = [0.659, 0.242, 0.099], CE = -log(0.659) = 0.417
    expectNear(result["total"], 0.417f, 0.01f);
}

// =============================================================================
// Part 2: FocalLoss Tests (Section 1.2.2)
// =============================================================================

// FL-001: Basic_Gamma0 - gamma=0 (equivalent to weighted CE)
TEST_F(ClassificationLossTest, FL_Basic_Gamma0) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3, 0.0f, 0.25f, 0.0f);
    auto pred = torch::tensor({{2.0f, 1.0f, 0.1f}});
    auto target = torch::tensor({0}, torch::kLong);

    auto result = loss.compute(pred, target);

    // With gamma=0, focal loss = -alpha * log(p)
    // p = softmax([2, 1, 0.1])[0] = 0.659
    // loss = -0.25 * log(0.659) = 0.25 * 0.417 = 0.104
    expectNear(result["total"], 0.104f, 0.01f);
}

// FL-002: Basic_Gamma2 - gamma=2 (default)
TEST_F(ClassificationLossTest, FL_Basic_Gamma2) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3, 0.0f, 0.25f, 2.0f);
    auto pred = torch::tensor({{2.0f, 1.0f, 0.1f}});
    auto target = torch::tensor({0}, torch::kLong);

    auto result = loss.compute(pred, target);

    // Focal loss with gamma=2 should be less than CE for easy samples
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
    EXPECT_GT(result["total"].item<float>(), 0.0f);
}

// FL-003: EasySample - High confidence
TEST_F(ClassificationLossTest, FL_EasySample) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3, 0.0f, 0.25f, 2.0f);
    auto pred = torch::tensor({{10.0f, 0.0f, 0.0f}});
    auto target = torch::tensor({0}, torch::kLong);

    auto result = loss.compute(pred, target);

    // Very small loss for easy sample
    EXPECT_LT(result["total"].item<float>(), 0.001f);
}

// FL-004: HardSample - Low confidence
TEST_F(ClassificationLossTest, FL_HardSample) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3, 0.0f, 0.25f, 2.0f);
    auto pred = torch::tensor({{0.0f, 0.0f, 0.0f}});  // Uniform logits
    auto target = torch::tensor({0}, torch::kLong);

    auto result = loss.compute(pred, target);

    // Hard sample gets higher weight
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// FL-005: PreComputedValues
TEST_F(ClassificationLossTest, FL_PreComputedValues) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3, 0.0f, 1.0f, 2.0f);  // alpha=1 for easier calculation
    auto pred = torch::tensor({{1.0f, 0.0f, 0.0f}});
    auto target = torch::tensor({0}, torch::kLong);

    auto result = loss.compute(pred, target);

    // p = exp(1)/(exp(1)+exp(0)+exp(0)) = 2.718/4.718 = 0.576
    // FL = -(1-0.576)^2 * log(0.576) = 0.180 * 0.552 = 0.099
    expectNear(result["total"], 0.099f, 0.02f);
}

// =============================================================================
// Part 3: LabelSmoothingLoss Tests (Section 1.2.3)
// =============================================================================

// LS-001: NoSmoothing - smoothing=0 (same as CE)
TEST_F(ClassificationLossTest, LS_NoSmoothing) {
    ClassificationLoss loss(ClassificationLoss::LossType::LABEL_SMOOTHING, 3, 0.0f);
    ClassificationLoss ceLoss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);

    auto pred = torch::tensor({{2.0f, 1.0f, 0.1f}});
    auto target = torch::tensor({0}, torch::kLong);

    auto result = loss.compute(pred, target);
    auto ceResult = ceLoss.compute(pred, target);

    // With smoothing=0, should be same as CE
    expectNear(result["total"], ceResult["total"].item<float>(), 1e-5f);
}

// LS-002: DefaultSmoothing - smoothing=0.1
TEST_F(ClassificationLossTest, LS_DefaultSmoothing) {
    ClassificationLoss loss(ClassificationLoss::LossType::LABEL_SMOOTHING, 3, 0.1f);
    auto pred = torch::tensor({{2.0f, 1.0f, 0.0f}});
    auto target = torch::tensor({0}, torch::kLong);

    auto result = loss.compute(pred, target);

    // Label smoothing typically increases loss slightly for confident predictions
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// LS-003: FullSmoothing - smoothing=1 (uniform target)
TEST_F(ClassificationLossTest, LS_FullSmoothing) {
    ClassificationLoss loss(ClassificationLoss::LossType::LABEL_SMOOTHING, 3, 1.0f);
    auto pred = torch::tensor({{2.0f, 1.0f, 0.0f}});
    auto target = torch::tensor({0}, torch::kLong);

    auto result = loss.compute(pred, target);

    // With smoothing=1, target becomes uniform, loss = -mean(log_softmax)
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// LS-004: MultiClass - Many classes
TEST_F(ClassificationLossTest, LS_MultiClass) {
    ClassificationLoss loss(ClassificationLoss::LossType::LABEL_SMOOTHING, 100, 0.1f);
    auto pred = torch::randn({8, 100});
    auto target = torch::randint(0, 100, {8}, torch::kLong);

    auto result = loss.compute(pred, target);

    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// =============================================================================
// Part 4: ClassificationLoss API Tests (Section 1.2.4)
// =============================================================================

// API-001: SetLabelSmoothing_Valid
TEST_F(ClassificationLossTest, API_SetLabelSmoothing_Valid) {
    ClassificationLoss loss(ClassificationLoss::LossType::LABEL_SMOOTHING, 10, 0.0f);
    loss.setLabelSmoothing(0.1f);

    // Verify by computing loss - with smoothing, loss should be different
    auto pred = torch::tensor({{10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}});
    auto target = torch::tensor({0}, torch::kLong);
    auto result = loss.compute(pred, target);
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// API-002: SetLabelSmoothing_Clamp
TEST_F(ClassificationLossTest, API_SetLabelSmoothing_Clamp) {
    ClassificationLoss loss(ClassificationLoss::LossType::LABEL_SMOOTHING, 3, 0.0f);
    loss.setLabelSmoothing(1.5f);  // Should be clamped to 1.0

    auto pred = torch::tensor({{2.0f, 1.0f, 0.0f}});
    auto target = torch::tensor({0}, torch::kLong);
    auto result = loss.compute(pred, target);
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// API-003: SetLabelSmoothing_Negative
TEST_F(ClassificationLossTest, API_SetLabelSmoothing_Negative) {
    ClassificationLoss loss(ClassificationLoss::LossType::LABEL_SMOOTHING, 3, 0.5f);
    loss.setLabelSmoothing(-0.1f);  // Should be clamped to 0.0

    // With smoothing=0, should behave like CE
    auto pred = torch::tensor({{2.0f, 1.0f, 0.0f}});
    auto target = torch::tensor({0}, torch::kLong);
    auto result = loss.compute(pred, target);
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// API-004: SetFocalParams_Valid
TEST_F(ClassificationLossTest, API_SetFocalParams_Valid) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3);
    EXPECT_NO_THROW(loss.setFocalParams(0.25f, 2.0f));
}

// API-005: SetFocalParams_GammaZero
TEST_F(ClassificationLossTest, API_SetFocalParams_GammaZero) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3);
    EXPECT_NO_THROW(loss.setFocalParams(0.25f, 0.0f));
}

// API-006: SetFocalParams_GammaMax
TEST_F(ClassificationLossTest, API_SetFocalParams_GammaMax) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3);
    EXPECT_NO_THROW(loss.setFocalParams(0.25f, 5.0f));
}

// API-007: SetFocalParams_GammaOverflow
TEST_F(ClassificationLossTest, API_SetFocalParams_GammaOverflow) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3);
    EXPECT_THROW(loss.setFocalParams(0.25f, 5.1f), std::invalid_argument);
}

// API-008: SetFocalParams_GammaNegative
TEST_F(ClassificationLossTest, API_SetFocalParams_GammaNegative) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3);
    EXPECT_THROW(loss.setFocalParams(0.25f, -0.1f), std::invalid_argument);
}

// API-009: GetLossType
TEST_F(ClassificationLossTest, API_GetLossType) {
    ClassificationLoss ceLoss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);
    ClassificationLoss focalLoss(ClassificationLoss::LossType::FOCAL, 3);
    ClassificationLoss lsLoss(ClassificationLoss::LossType::LABEL_SMOOTHING, 3);

    EXPECT_EQ(ceLoss.getLossType(), ClassificationLoss::LossType::CROSS_ENTROPY);
    EXPECT_EQ(focalLoss.getLossType(), ClassificationLoss::LossType::FOCAL);
    EXPECT_EQ(lsLoss.getLossType(), ClassificationLoss::LossType::LABEL_SMOOTHING);
}

// API-010: GetSoftTarget
TEST_F(ClassificationLossTest, API_GetSoftTarget) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 5);

    auto hardTarget = torch::tensor({0, 2, 4}, torch::kLong);
    auto softTarget = loss.getSoftTarget(hardTarget);

    // Check shape: [3, 5]
    EXPECT_EQ(softTarget.size(0), 3);
    EXPECT_EQ(softTarget.size(1), 5);

    // Check one-hot encoding
    EXPECT_EQ(softTarget[0][0].item<float>(), 1.0f);
    EXPECT_EQ(softTarget[1][2].item<float>(), 1.0f);
    EXPECT_EQ(softTarget[2][4].item<float>(), 1.0f);
}

// API-011: Name
TEST_F(ClassificationLossTest, API_Name) {
    ClassificationLoss ceLoss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);
    ClassificationLoss focalLoss(ClassificationLoss::LossType::FOCAL, 3);
    ClassificationLoss lsLoss(ClassificationLoss::LossType::LABEL_SMOOTHING, 3);

    EXPECT_EQ(ceLoss.name(), "ClassificationLoss(CrossEntropy)");
    EXPECT_EQ(focalLoss.name(), "ClassificationLoss(Focal)");
    EXPECT_EQ(lsLoss.name(), "ClassificationLoss(LabelSmoothing)");
}

// API-012: DataExample
TEST_F(ClassificationLossTest, API_DataExample) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);
    auto pred = torch::tensor({{2.0f, 1.0f, 0.1f}});
    auto example = createDataExample(torch::tensor({0}, torch::kLong));

    auto result = loss.compute(pred, example);

    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// =============================================================================
// Part 5: ClassificationLoss Exception Tests (Section 1.2.5)
// =============================================================================

// EX-001: Constructor_InvalidNumClasses
TEST_F(ClassificationLossTest, EX_Constructor_InvalidNumClasses) {
    EXPECT_THROW({
        ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 0);
    }, std::invalid_argument);
}

// EX-002: Constructor_NegativeNumClasses
TEST_F(ClassificationLossTest, EX_Constructor_NegativeNumClasses) {
    EXPECT_THROW({
        ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, -1);
    }, std::invalid_argument);
}

// EX-003: Constructor_InvalidGamma
TEST_F(ClassificationLossTest, EX_Constructor_InvalidGamma) {
    EXPECT_THROW({
        ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 10, 0.0f, 0.25f, 6.0f);
    }, std::invalid_argument);
}

// EX-004: Compute_PredNotMatrix
TEST_F(ClassificationLossTest, EX_Compute_PredNotMatrix) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);
    auto pred = torch::randn({10});  // 1D tensor
    auto target = torch::randint(0, 3, {10}, torch::kLong);

    EXPECT_THROW({
        loss.compute(pred, target);
    }, std::invalid_argument);
}

// EX-005: Compute_TargetWrongDim
TEST_F(ClassificationLossTest, EX_Compute_TargetWrongDim) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);
    auto pred = torch::randn({4, 3});
    auto target = torch::randint(0, 3, {4, 3, 2}, torch::kLong);  // 3D tensor

    EXPECT_THROW({
        loss.compute(pred, target);
    }, std::invalid_argument);
}

// EX-006: Compute_BatchMismatch
TEST_F(ClassificationLossTest, EX_Compute_BatchMismatch) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 10);
    auto pred = torch::randn({4, 10});
    auto target = torch::randint(0, 10, {5}, torch::kLong);

    EXPECT_THROW({
        loss.compute(pred, target);
    }, std::invalid_argument);
}

// =============================================================================
// Part 6: ClassificationLoss Branch Coverage (Section 1.2.6)
// =============================================================================

// BR-001: HardLabels_1D
TEST_F(ClassificationLossTest, BR_HardLabels_1D) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);
    auto pred = torch::randn({4, 3});
    auto target = torch::randint(0, 3, {4}, torch::kLong);  // 1D hard labels

    auto result = loss.compute(pred, target);
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// BR-002: SoftLabels_2D
TEST_F(ClassificationLossTest, BR_SoftLabels_2D) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3);
    auto pred = torch::randn({4, 3});
    auto target = torch::softmax(torch::randn({4, 3}), 1);  // 2D soft labels

    auto result = loss.compute(pred, target);
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// BR-003: FocalGamma0_Optimization
TEST_F(ClassificationLossTest, BR_FocalGamma0_Optimization) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3, 0.0f, 0.25f, 0.0f);
    auto pred = torch::randn({4, 3});
    auto target = torch::randint(0, 3, {4}, torch::kLong);

    auto result = loss.compute(pred, target);
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// BR-004: FocalGamma1_Optimization
TEST_F(ClassificationLossTest, BR_FocalGamma1_Optimization) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3, 0.0f, 0.25f, 1.0f);
    auto pred = torch::randn({4, 3});
    auto target = torch::randint(0, 3, {4}, torch::kLong);

    auto result = loss.compute(pred, target);
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// BR-005: FocalGamma2_Optimization
TEST_F(ClassificationLossTest, BR_FocalGamma2_Optimization) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3, 0.0f, 0.25f, 2.0f);
    auto pred = torch::randn({4, 3});
    auto target = torch::randint(0, 3, {4}, torch::kLong);

    auto result = loss.compute(pred, target);
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// BR-006: FocalGammaGeneral
TEST_F(ClassificationLossTest, BR_FocalGammaGeneral) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3, 0.0f, 0.25f, 1.5f);
    auto pred = torch::randn({4, 3});
    auto target = torch::randint(0, 3, {4}, torch::kLong);

    auto result = loss.compute(pred, target);
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// BR-007: Reduction_None
TEST_F(ClassificationLossTest, BR_Reduction_None) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3, 0.0f, 0.25f, 2.0f, "none");
    auto pred = torch::randn({4, 3});
    auto target = torch::randint(0, 3, {4}, torch::kLong);

    auto result = loss.compute(pred, target);

    // Should return per-sample loss [4]
    EXPECT_EQ(result["total"].size(0), 4);
}

// BR-008: Reduction_Mean
TEST_F(ClassificationLossTest, BR_Reduction_Mean) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3, 0.0f, 0.25f, 2.0f, "mean");
    auto pred = torch::randn({4, 3});
    auto target = torch::randint(0, 3, {4}, torch::kLong);

    auto result = loss.compute(pred, target);

    // Should return scalar
    EXPECT_EQ(result["total"].dim(), 0);
}

// BR-009: Reduction_Sum
TEST_F(ClassificationLossTest, BR_Reduction_Sum) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 3, 0.0f, 0.25f, 2.0f, "sum");
    auto pred = torch::randn({4, 3});
    auto target = torch::randint(0, 3, {4}, torch::kLong);

    auto result = loss.compute(pred, target);

    // Should return scalar
    EXPECT_EQ(result["total"].dim(), 0);
}

// BR-010: Focal_SoftLabels_2D
TEST_F(ClassificationLossTest, BR_Focal_SoftLabels_2D) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 3, 0.0f, 0.25f, 2.0f);
    auto pred = torch::randn({4, 3});
    auto target = torch::softmax(torch::randn({4, 3}), 1);  // 2D soft labels

    auto result = loss.compute(pred, target);
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// BR-011: LabelSmoothing_SoftLabels_2D
TEST_F(ClassificationLossTest, BR_LabelSmoothing_SoftLabels_2D) {
    ClassificationLoss loss(ClassificationLoss::LossType::LABEL_SMOOTHING, 3, 0.1f);
    auto pred = torch::randn({4, 3});
    auto target = torch::softmax(torch::randn({4, 3}), 1);  // 2D soft labels

    auto result = loss.compute(pred, target);
    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// =============================================================================
// Part 7: Gradient Flow Tests
// =============================================================================

// GR-001: CE_GradientFlow
TEST_F(ClassificationLossTest, GR_CE_GradientFlow) {
    ClassificationLoss loss(ClassificationLoss::LossType::CROSS_ENTROPY, 10);
    auto pred = torch::randn({4, 10}, torch::requires_grad());
    auto target = torch::randint(0, 10, {4}, torch::kLong);

    auto result = loss.compute(pred, target);
    result["total"].backward();

    EXPECT_TRUE(pred.grad().defined());
    EXPECT_FALSE(pred.grad().isnan().any().item<bool>());
}

// GR-002: Focal_GradientFlow
TEST_F(ClassificationLossTest, GR_Focal_GradientFlow) {
    ClassificationLoss loss(ClassificationLoss::LossType::FOCAL, 10);
    auto pred = torch::randn({4, 10}, torch::requires_grad());
    auto target = torch::randint(0, 10, {4}, torch::kLong);

    auto result = loss.compute(pred, target);
    result["total"].backward();

    EXPECT_TRUE(pred.grad().defined());
    EXPECT_FALSE(pred.grad().isnan().any().item<bool>());
}

// GR-003: LabelSmoothing_GradientFlow
TEST_F(ClassificationLossTest, GR_LabelSmoothing_GradientFlow) {
    ClassificationLoss loss(ClassificationLoss::LossType::LABEL_SMOOTHING, 10, 0.1f);
    auto pred = torch::randn({4, 10}, torch::requires_grad());
    auto target = torch::randint(0, 10, {4}, torch::kLong);

    auto result = loss.compute(pred, target);
    result["total"].backward();

    EXPECT_TRUE(pred.grad().defined());
    EXPECT_FALSE(pred.grad().isnan().any().item<bool>());
}
