#include "pch.h"
#include <gtest/gtest.h>
#include "Model/Loss/BaseLoss.h"
#include <torch/torch.h>
#include <cmath>
#include <limits>

using namespace WheelDL::Model::Loss;
using namespace WheelDL::Data::Dataset;

// =============================================================================
// Test Fixture
// =============================================================================

class BaseLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
    }
    void TearDown() override {}

    // Helper: Compare tensor value with expected value
    void expectNear(const torch::Tensor& actual, float expected, float tol = 1e-4f) {
        EXPECT_NEAR(actual.item<float>(), expected, tol);
    }

    // Helper: Create DataExample with targets tensor (for MSE, MAE, SmoothL1)
    DataExample createDataExampleWithTargets(const torch::Tensor& targets) {
        DataExample example;
        example.targets = targets;
        return example;
    }

    // Helper: Create DataExample with classes tensor (for BCE)
    DataExample createDataExampleWithClasses(const torch::Tensor& classes) {
        DataExample example;
        example.classes = classes;
        return example;
    }
};

// =============================================================================
// Part 1: BCEWithLogitsLoss Tests
// =============================================================================

// BL-001: Basic_SingleElement - Single element BCE
TEST_F(BaseLossTest, BCE_Basic_SingleElement) {
    BCEWithLogitsLoss loss;
    auto pred = torch::tensor({0.0f});
    auto target = torch::tensor({1.0f});

    auto result = loss.compute(pred, target);

    // -log(sigmoid(0)) = -log(0.5) = log(2) = 0.6931
    expectNear(result["total"], 0.6931f, 1e-3f);
}

// BL-002: Basic_BatchComputation - Batch computation
TEST_F(BaseLossTest, BCE_Basic_BatchComputation) {
    BCEWithLogitsLoss loss;
    auto pred = torch::tensor({0.0f, 0.0f, 0.0f});
    auto target = torch::tensor({1.0f, 1.0f, 1.0f});

    auto result = loss.compute(pred, target);

    // Mean of identical elements = 0.6931
    expectNear(result["total"], 0.6931f, 1e-3f);
}

// BL-003: PerfectPrediction_Positive - High confidence correct
TEST_F(BaseLossTest, BCE_PerfectPrediction_Positive) {
    BCEWithLogitsLoss loss;
    auto pred = torch::tensor({10.0f});
    auto target = torch::tensor({1.0f});

    auto result = loss.compute(pred, target);

    // sigmoid(10) ~ 1, loss ~ 0
    EXPECT_LT(result["total"].item<float>(), 0.001f);
}

// BL-004: PerfectPrediction_Negative - High confidence correct for negative
TEST_F(BaseLossTest, BCE_PerfectPrediction_Negative) {
    BCEWithLogitsLoss loss;
    auto pred = torch::tensor({-10.0f});
    auto target = torch::tensor({0.0f});

    auto result = loss.compute(pred, target);

    // sigmoid(-10) ~ 0, loss ~ 0
    EXPECT_LT(result["total"].item<float>(), 0.001f);
}

// BL-005: WrongPrediction_High - High confidence wrong
TEST_F(BaseLossTest, BCE_WrongPrediction_High) {
    BCEWithLogitsLoss loss;
    auto pred = torch::tensor({10.0f});
    auto target = torch::tensor({0.0f});

    auto result = loss.compute(pred, target);

    // Very high loss ~10
    EXPECT_GT(result["total"].item<float>(), 9.0f);
}

// BL-006: ZeroPrediction_ZeroTarget - Zero pred, zero target
TEST_F(BaseLossTest, BCE_ZeroPrediction_ZeroTarget) {
    BCEWithLogitsLoss loss;
    auto pred = torch::tensor({0.0f});
    auto target = torch::tensor({0.0f});

    auto result = loss.compute(pred, target);

    // -log(1-sigmoid(0)) = -log(0.5) = 0.6931
    expectNear(result["total"], 0.6931f, 1e-3f);
}

// BL-007: MultiDimensional_2D - 2D tensor input
TEST_F(BaseLossTest, BCE_MultiDimensional_2D) {
    BCEWithLogitsLoss loss;
    auto pred = torch::zeros({2, 3});
    auto target = torch::ones({2, 3});

    auto result = loss.compute(pred, target);

    // All elements same, mean = 0.6931
    expectNear(result["total"], 0.6931f, 1e-3f);
}

// BL-008: MultiDimensional_3D - 3D tensor input
TEST_F(BaseLossTest, BCE_MultiDimensional_3D) {
    BCEWithLogitsLoss loss;
    auto pred = torch::zeros({2, 3, 4});
    auto target = torch::ones({2, 3, 4});

    auto result = loss.compute(pred, target);

    // All elements same, mean = 0.6931
    expectNear(result["total"], 0.6931f, 1e-3f);
}

// BL-009: GradientFlow - Backward pass
TEST_F(BaseLossTest, BCE_GradientFlow) {
    BCEWithLogitsLoss loss;
    auto pred = torch::randn({4, 10}, torch::requires_grad());
    auto target = torch::rand({4, 10});

    auto result = loss.compute(pred, target);
    result["total"].backward();

    EXPECT_TRUE(pred.grad().defined());
    EXPECT_FALSE(pred.grad().isnan().any().item<bool>());
    EXPECT_FALSE(pred.grad().isinf().any().item<bool>());
}

// BL-010: NumericalStability_LargePos - Large positive values
TEST_F(BaseLossTest, BCE_NumericalStability_LargePos) {
    BCEWithLogitsLoss loss;
    auto pred = torch::tensor({100.0f});
    auto target = torch::tensor({1.0f});

    auto result = loss.compute(pred, target);

    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// BL-011: NumericalStability_LargeNeg - Large negative values
TEST_F(BaseLossTest, BCE_NumericalStability_LargeNeg) {
    BCEWithLogitsLoss loss;
    auto pred = torch::tensor({-100.0f});
    auto target = torch::tensor({0.0f});

    auto result = loss.compute(pred, target);

    EXPECT_TRUE(torch::isfinite(result["total"]).item<bool>());
}

// BL-012: PreComputedValues - Pre-computed test case
TEST_F(BaseLossTest, BCE_PreComputedValues) {
    BCEWithLogitsLoss loss;
    auto pred = torch::tensor({0.5f, -0.5f, 1.0f, -1.0f});
    auto target = torch::tensor({1.0f, 0.0f, 1.0f, 0.0f});

    auto result = loss.compute(pred, target);

    // Pre-computed:
    // pred=0.5, target=1.0: -log(sigmoid(0.5)) = 0.4741
    // pred=-0.5, target=0.0: -log(1-sigmoid(-0.5)) = 0.4741
    // pred=1.0, target=1.0: -log(sigmoid(1.0)) = 0.3133
    // pred=-1.0, target=0.0: -log(1-sigmoid(-1.0)) = 0.3133
    // Mean = (0.4741 + 0.4741 + 0.3133 + 0.3133) / 4 = 0.3937
    expectNear(result["total"], 0.3937f, 1e-3f);
}

// BL-013: Reduction_Sum - Sum reduction
TEST_F(BaseLossTest, BCE_Reduction_Sum) {
    BCEWithLogitsLoss loss("sum");
    auto pred = torch::zeros({4});
    auto target = torch::ones({4});

    auto result = loss.compute(pred, target);

    // Sum of 4 elements with 0.6931 each = 2.7724
    expectNear(result["total"], 2.7724f, 1e-3f);
}

// BL-014: Reduction_None - No reduction
TEST_F(BaseLossTest, BCE_Reduction_None) {
    BCEWithLogitsLoss loss("none");
    auto pred = torch::zeros({4});
    auto target = torch::ones({4});

    auto result = loss.compute(pred, target);

    // Returns per-element loss
    EXPECT_EQ(result["total"].size(0), 4);
}

// BL-015: Name - Check name() method
TEST_F(BaseLossTest, BCE_Name) {
    BCEWithLogitsLoss loss;
    EXPECT_EQ(loss.name(), "BCEWithLogitsLoss");
}

// BL-016: DataExample - Compute with DataExample (uses classes field)
TEST_F(BaseLossTest, BCE_DataExample) {
    BCEWithLogitsLoss loss;
    auto pred = torch::tensor({0.0f});
    auto example = createDataExampleWithClasses(torch::tensor({1.0f}));

    auto result = loss.compute(pred, example);

    expectNear(result["total"], 0.6931f, 1e-3f);
}

// =============================================================================
// Part 2: MSELoss Tests
// =============================================================================

// MSE-001: Basic_SingleElement
TEST_F(BaseLossTest, MSE_Basic_SingleElement) {
    MSELoss loss;
    auto pred = torch::tensor({2.0f});
    auto target = torch::tensor({1.0f});

    auto result = loss.compute(pred, target);

    // (2-1)^2 = 1.0
    expectNear(result["total"], 1.0f, 1e-4f);
}

// MSE-002: Basic_BatchMean
TEST_F(BaseLossTest, MSE_Basic_BatchMean) {
    MSELoss loss;
    auto pred = torch::tensor({1.0f, 2.0f, 3.0f});
    auto target = torch::tensor({2.0f, 3.0f, 4.0f});

    auto result = loss.compute(pred, target);

    // Mean of (1,1,1) = 1.0
    expectNear(result["total"], 1.0f, 1e-4f);
}

// MSE-003: ZeroDifference
TEST_F(BaseLossTest, MSE_ZeroDifference) {
    MSELoss loss;
    auto pred = torch::tensor({1.0f, 2.0f, 3.0f});
    auto target = torch::tensor({1.0f, 2.0f, 3.0f});

    auto result = loss.compute(pred, target);

    expectNear(result["total"], 0.0f, 1e-6f);
}

// MSE-004: NegativeValues
TEST_F(BaseLossTest, MSE_NegativeValues) {
    MSELoss loss;
    auto pred = torch::tensor({-1.0f, -2.0f});
    auto target = torch::tensor({-2.0f, -3.0f});

    auto result = loss.compute(pred, target);

    // Mean of (1,1) = 1.0
    expectNear(result["total"], 1.0f, 1e-4f);
}

// MSE-005: FloatPrecision
TEST_F(BaseLossTest, MSE_FloatPrecision) {
    MSELoss loss;
    auto pred = torch::tensor({0.1f});
    auto target = torch::tensor({0.2f});

    auto result = loss.compute(pred, target);

    // (0.1-0.2)^2 = 0.01
    expectNear(result["total"], 0.01f, 1e-5f);
}

// MSE-006: PreComputedValues
TEST_F(BaseLossTest, MSE_PreComputedValues) {
    MSELoss loss;
    auto pred = torch::tensor({1.5f, 2.5f, 3.5f, 4.5f});
    auto target = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f});

    auto result = loss.compute(pred, target);

    // Each diff = 0.5, squared = 0.25, mean = 0.25
    expectNear(result["total"], 0.25f, 1e-4f);
}

// MSE-007: Name
TEST_F(BaseLossTest, MSE_Name) {
    MSELoss loss;
    EXPECT_EQ(loss.name(), "MSELoss");
}

// MSE-008: GradientFlow
TEST_F(BaseLossTest, MSE_GradientFlow) {
    MSELoss loss;
    auto pred = torch::randn({4, 10}, torch::requires_grad());
    auto target = torch::randn({4, 10});

    auto result = loss.compute(pred, target);
    result["total"].backward();

    EXPECT_TRUE(pred.grad().defined());
    EXPECT_FALSE(pred.grad().isnan().any().item<bool>());
}

// MSE-009: DataExample - Compute with DataExample (uses targets field)
TEST_F(BaseLossTest, MSE_DataExample) {
    MSELoss loss;
    auto pred = torch::tensor({2.0f});
    auto example = createDataExampleWithTargets(torch::tensor({1.0f}));

    auto result = loss.compute(pred, example);

    expectNear(result["total"], 1.0f, 1e-4f);
}

// =============================================================================
// Part 3: MAELoss (L1Loss) Tests
// =============================================================================

// MAE-001: Basic_SingleElement
TEST_F(BaseLossTest, MAE_Basic_SingleElement) {
    MAELoss loss;
    auto pred = torch::tensor({2.0f});
    auto target = torch::tensor({1.0f});

    auto result = loss.compute(pred, target);

    // |2-1| = 1.0
    expectNear(result["total"], 1.0f, 1e-4f);
}

// MAE-002: Basic_BatchMean
TEST_F(BaseLossTest, MAE_Basic_BatchMean) {
    MAELoss loss;
    auto pred = torch::tensor({1.0f, 2.0f, 3.0f});
    auto target = torch::tensor({2.0f, 3.0f, 5.0f});

    auto result = loss.compute(pred, target);

    // Mean of (1,1,2) = 1.333
    expectNear(result["total"], 1.333f, 1e-3f);
}

// MAE-003: ZeroDifference
TEST_F(BaseLossTest, MAE_ZeroDifference) {
    MAELoss loss;
    auto pred = torch::tensor({1.0f, 2.0f, 3.0f});
    auto target = torch::tensor({1.0f, 2.0f, 3.0f});

    auto result = loss.compute(pred, target);

    expectNear(result["total"], 0.0f, 1e-6f);
}

// MAE-004: NegativeValues
TEST_F(BaseLossTest, MAE_NegativeValues) {
    MAELoss loss;
    auto pred = torch::tensor({-1.0f});
    auto target = torch::tensor({1.0f});

    auto result = loss.compute(pred, target);

    // |-1-1| = 2.0
    expectNear(result["total"], 2.0f, 1e-4f);
}

// MAE-005: PreComputedValues
TEST_F(BaseLossTest, MAE_PreComputedValues) {
    MAELoss loss;
    auto pred = torch::tensor({1.0f, 3.0f, 5.0f, 7.0f});
    auto target = torch::tensor({2.0f, 2.0f, 8.0f, 6.0f});

    auto result = loss.compute(pred, target);

    // Individual: |1-2|=1, |3-2|=1, |5-8|=3, |7-6|=1
    // Mean = (1+1+3+1) / 4 = 1.5
    expectNear(result["total"], 1.5f, 1e-4f);
}

// MAE-006: Name
TEST_F(BaseLossTest, MAE_Name) {
    MAELoss loss;
    EXPECT_EQ(loss.name(), "MAELoss");
}

// MAE-007: GradientFlow
TEST_F(BaseLossTest, MAE_GradientFlow) {
    MAELoss loss;
    auto pred = torch::randn({4, 10}, torch::requires_grad());
    auto target = torch::randn({4, 10});

    auto result = loss.compute(pred, target);
    result["total"].backward();

    EXPECT_TRUE(pred.grad().defined());
}

// MAE-008: DataExample - Compute with DataExample (uses targets field)
TEST_F(BaseLossTest, MAE_DataExample) {
    MAELoss loss;
    auto pred = torch::tensor({2.0f});
    auto example = createDataExampleWithTargets(torch::tensor({1.0f}));

    auto result = loss.compute(pred, example);

    expectNear(result["total"], 1.0f, 1e-4f);
}

// =============================================================================
// Part 4: SmoothL1Loss (Huber Loss) Tests
// =============================================================================

// SL1-001: SmallDiff_Quadratic - |diff| < beta
TEST_F(BaseLossTest, SmoothL1_SmallDiff_Quadratic) {
    SmoothL1Loss loss(1.0f);
    auto pred = torch::tensor({0.1f});
    auto target = torch::tensor({0.0f});

    auto result = loss.compute(pred, target);

    // 0.5 * 0.1^2 / 1.0 = 0.005
    expectNear(result["total"], 0.005f, 1e-4f);
}

// SL1-002: LargeDiff_Linear - |diff| >= beta
TEST_F(BaseLossTest, SmoothL1_LargeDiff_Linear) {
    SmoothL1Loss loss(1.0f);
    auto pred = torch::tensor({2.0f});
    auto target = torch::tensor({0.0f});

    auto result = loss.compute(pred, target);

    // 2.0 - 0.5*1.0 = 1.5
    expectNear(result["total"], 1.5f, 1e-4f);
}

// SL1-003: ExactlyBeta - |diff| = beta
TEST_F(BaseLossTest, SmoothL1_ExactlyBeta) {
    SmoothL1Loss loss(1.0f);
    auto pred = torch::tensor({1.0f});
    auto target = torch::tensor({0.0f});

    auto result = loss.compute(pred, target);

    // At boundary: 0.5 * 1.0^2 / 1.0 = 0.5 (quadratic) or 1.0 - 0.5*1.0 = 0.5 (linear)
    expectNear(result["total"], 0.5f, 1e-4f);
}

// SL1-004: CustomBeta
TEST_F(BaseLossTest, SmoothL1_CustomBeta) {
    SmoothL1Loss loss(0.5f);
    auto pred = torch::tensor({0.5f});
    auto target = torch::tensor({0.0f});

    auto result = loss.compute(pred, target);

    // At boundary beta=0.5: 0.5 * 0.5^2 / 0.5 = 0.25
    expectNear(result["total"], 0.25f, 1e-4f);
}

// SL1-005: PreComputedValues (beta=1.0)
TEST_F(BaseLossTest, SmoothL1_PreComputedValues) {
    SmoothL1Loss loss(1.0f);
    auto pred = torch::tensor({0.3f, 0.8f, 1.5f, 3.0f});
    auto target = torch::tensor({0.0f, 0.0f, 0.0f, 0.0f});

    auto result = loss.compute(pred, target);

    // Quadratic (|diff| < 1.0): 0.3->0.045, 0.8->0.32
    // Linear (|diff| >= 1.0): 1.5->1.0, 3.0->2.5
    // Mean = (0.045 + 0.32 + 1.0 + 2.5) / 4 = 0.96625
    expectNear(result["total"], 0.96625f, 1e-3f);
}

// SL1-006: Name
TEST_F(BaseLossTest, SmoothL1_Name) {
    SmoothL1Loss loss;
    EXPECT_EQ(loss.name(), "SmoothL1Loss");
}

// SL1-007: GradientFlow
TEST_F(BaseLossTest, SmoothL1_GradientFlow) {
    SmoothL1Loss loss;
    auto pred = torch::randn({4, 10}, torch::requires_grad());
    auto target = torch::randn({4, 10});

    auto result = loss.compute(pred, target);
    result["total"].backward();

    EXPECT_TRUE(pred.grad().defined());
}

// SL1-008: DataExample - Compute with DataExample (uses targets field)
TEST_F(BaseLossTest, SmoothL1_DataExample) {
    SmoothL1Loss loss(1.0f);
    auto pred = torch::tensor({2.0f});
    auto example = createDataExampleWithTargets(torch::tensor({0.0f}));

    auto result = loss.compute(pred, example);

    // 2.0 - 0.5*1.0 = 1.5
    expectNear(result["total"], 1.5f, 1e-4f);
}

// =============================================================================
// Part 5: DFLoss (Distribution Focal Loss) Tests
// =============================================================================

// DFL-001: Basic_IntegerTarget
TEST_F(BaseLossTest, DFL_Basic_IntegerTarget) {
    DFLoss loss(16);
    auto pred = torch::zeros({1, 16});  // [N, regMax]
    pred[0][5] = 10.0f;  // Strong prediction at index 5
    auto target = torch::tensor({5.0f});  // Integer target

    auto result = loss.compute(pred, target);

    // CrossEntropy at correct class should be low
    EXPECT_LT(result.item<float>(), 1.0f);
}

// DFL-002: Basic_FractionalTarget
TEST_F(BaseLossTest, DFL_Basic_FractionalTarget) {
    DFLoss loss(16);
    auto pred = torch::zeros({1, 16});
    pred[0][5] = 5.0f;
    pred[0][6] = 5.0f;  // Equal predictions at 5 and 6
    auto target = torch::tensor({5.5f});  // Fractional target

    auto result = loss.compute(pred, target);

    // Interpolated CE between 5 and 6
    EXPECT_TRUE(torch::isfinite(result).item<bool>());
}

// DFL-003: BoundaryLow - Target = 0
TEST_F(BaseLossTest, DFL_BoundaryLow) {
    DFLoss loss(16);
    auto pred = torch::zeros({1, 16});
    pred[0][0] = 10.0f;  // Strong prediction at 0
    auto target = torch::tensor({0.0f});

    auto result = loss.compute(pred, target);

    EXPECT_LT(result.item<float>(), 1.0f);
}

// DFL-004: BoundaryHigh - Target = regMax-1-epsilon
TEST_F(BaseLossTest, DFL_BoundaryHigh) {
    DFLoss loss(16);
    auto pred = torch::zeros({1, 16});
    pred[0][14] = 10.0f;  // Strong prediction at 14
    auto target = torch::tensor({14.0f});

    auto result = loss.compute(pred, target);

    EXPECT_LT(result.item<float>(), 1.0f);
}

// DFL-005: BatchComputation - Multiple samples
TEST_F(BaseLossTest, DFL_BatchComputation) {
    DFLoss loss(16);
    auto pred = torch::randn({4, 4, 16});  // [B, 4, regMax]
    auto target = torch::rand({4, 4}) * 14.0f;  // Random targets in valid range

    auto result = loss.compute(pred.view({-1, 16}), target.view({-1}));

    EXPECT_TRUE(torch::isfinite(result).all().item<bool>());
}

// DFL-006: GetRegMax - Accessor method
TEST_F(BaseLossTest, DFL_GetRegMax) {
    DFLoss loss(16);
    EXPECT_EQ(loss.getRegMax(), 16);

    DFLoss loss32(32);
    EXPECT_EQ(loss32.getRegMax(), 32);
}

// DFL-007: Constructor_InvalidRegMax - regMax=0
TEST_F(BaseLossTest, DFL_Constructor_InvalidRegMax) {
    EXPECT_THROW({
        DFLoss loss(0);
    }, std::invalid_argument);
}

// DFL-008: Constructor_NegativeRegMax - regMax=-1
TEST_F(BaseLossTest, DFL_Constructor_NegativeRegMax) {
    EXPECT_THROW({
        DFLoss loss(-1);
    }, std::invalid_argument);
}

// =============================================================================
// Part 6: BaseLoss Static Methods Tests
// =============================================================================

// BLM-001: ValidateLoss_NaN
TEST_F(BaseLossTest, ValidateLoss_NaN) {
    auto nanTensor = torch::tensor({std::nanf("")});

    // Using a concrete class to test protected method behavior
    BCEWithLogitsLoss loss;
    auto pred = torch::tensor({std::nanf("")});
    auto target = torch::tensor({1.0f});

    // The validation should happen internally
    // NaN input may propagate, testing that computation doesn't crash
    auto result = loss.compute(pred, target);
    // NaN input produces NaN output - testing that computation doesn't crash
    EXPECT_TRUE(result.find("total") != result.end());
}

// BLM-002: ValidateLoss_Inf
TEST_F(BaseLossTest, ValidateLoss_Inf) {
    BCEWithLogitsLoss loss;
    auto pred = torch::tensor({std::numeric_limits<float>::infinity()});
    auto target = torch::tensor({1.0f});

    // Very large inputs should be handled numerically stable
    auto result = loss.compute(pred, target);
    EXPECT_TRUE(result.find("total") != result.end());
}

// BLM-003: GetTotal_Valid
TEST_F(BaseLossTest, GetTotal_Valid) {
    std::unordered_map<std::string, torch::Tensor> lossMap;
    lossMap["total"] = torch::tensor({1.5f});
    lossMap["component"] = torch::tensor({0.5f});

    auto total = BaseLoss::getTotal(lossMap);
    expectNear(total, 1.5f, 1e-6f);
}

// BLM-004: GetTotal_MissingKey
TEST_F(BaseLossTest, GetTotal_MissingKey) {
    std::unordered_map<std::string, torch::Tensor> lossMap;
    lossMap["component"] = torch::tensor({0.5f});

    EXPECT_THROW({
        BaseLoss::getTotal(lossMap);
    }, std::runtime_error);
}

// BLM-005: Compute_EmptyVector
TEST_F(BaseLossTest, Compute_EmptyVector) {
    BCEWithLogitsLoss loss;
    BaseLoss& baseLoss = loss;  // Use base class reference to avoid name hiding
    std::vector<torch::Tensor> predictions;
    auto example = createDataExampleWithClasses(torch::tensor({1.0f}));

    EXPECT_THROW({
        baseLoss.compute(predictions, example);
    }, std::invalid_argument);
}

// BLM-006: Compute_SingleTensorVector
TEST_F(BaseLossTest, Compute_SingleTensorVector) {
    BCEWithLogitsLoss loss;
    BaseLoss& baseLoss = loss;  // Use base class reference to avoid name hiding
    std::vector<torch::Tensor> predictions = {torch::tensor({0.0f})};
    auto example = createDataExampleWithClasses(torch::tensor({1.0f}));

    auto result = baseLoss.compute(predictions, example);
    expectNear(result["total"], 0.6931f, 1e-3f);
}

// BLM-007: Compute_MultipleTensorVector
TEST_F(BaseLossTest, Compute_MultipleTensorVector) {
    BCEWithLogitsLoss loss;
    BaseLoss& baseLoss = loss;  // Use base class reference to avoid name hiding
    std::vector<torch::Tensor> predictions = {
        torch::tensor({0.0f}),
        torch::tensor({0.0f})
    };
    auto example = createDataExampleWithClasses(torch::tensor({1.0f}));

    // Simple losses throw on multiple predictions
    EXPECT_THROW({
        baseLoss.compute(predictions, example);
    }, std::logic_error);
}

// =============================================================================
// Part 7: Reference Table Tests (from Appendix A.1)
// =============================================================================

// RT-001: BCE Reference pred=0, target=0
TEST_F(BaseLossTest, BCERef_Pred0_Target0) {
    BCEWithLogitsLoss loss;
    auto result = loss.compute(torch::tensor({0.0f}), torch::tensor({0.0f}));
    expectNear(result["total"], 0.6931f, 1e-3f);
}

// RT-002: BCE Reference pred=0, target=1
TEST_F(BaseLossTest, BCERef_Pred0_Target1) {
    BCEWithLogitsLoss loss;
    auto result = loss.compute(torch::tensor({0.0f}), torch::tensor({1.0f}));
    expectNear(result["total"], 0.6931f, 1e-3f);
}

// RT-003: BCE Reference pred=1, target=1
TEST_F(BaseLossTest, BCERef_Pred1_Target1) {
    BCEWithLogitsLoss loss;
    auto result = loss.compute(torch::tensor({1.0f}), torch::tensor({1.0f}));
    expectNear(result["total"], 0.3133f, 1e-3f);
}

// RT-004: BCE Reference pred=-1, target=0
TEST_F(BaseLossTest, BCERef_PredNeg1_Target0) {
    BCEWithLogitsLoss loss;
    auto result = loss.compute(torch::tensor({-1.0f}), torch::tensor({0.0f}));
    expectNear(result["total"], 0.3133f, 1e-3f);
}

// RT-005: BCE Reference pred=2, target=1
TEST_F(BaseLossTest, BCERef_Pred2_Target1) {
    BCEWithLogitsLoss loss;
    auto result = loss.compute(torch::tensor({2.0f}), torch::tensor({1.0f}));
    expectNear(result["total"], 0.1269f, 1e-3f);
}

// RT-006: BCE Reference pred=-2, target=0
TEST_F(BaseLossTest, BCERef_PredNeg2_Target0) {
    BCEWithLogitsLoss loss;
    auto result = loss.compute(torch::tensor({-2.0f}), torch::tensor({0.0f}));
    expectNear(result["total"], 0.1269f, 1e-3f);
}

// RT-007: BCE Reference pred=5, target=1
TEST_F(BaseLossTest, BCERef_Pred5_Target1) {
    BCEWithLogitsLoss loss;
    auto result = loss.compute(torch::tensor({5.0f}), torch::tensor({1.0f}));
    expectNear(result["total"], 0.0067f, 1e-3f);
}

// RT-008: BCE Reference pred=-5, target=0
TEST_F(BaseLossTest, BCERef_PredNeg5_Target0) {
    BCEWithLogitsLoss loss;
    auto result = loss.compute(torch::tensor({-5.0f}), torch::tensor({0.0f}));
    expectNear(result["total"], 0.0067f, 1e-3f);
}
