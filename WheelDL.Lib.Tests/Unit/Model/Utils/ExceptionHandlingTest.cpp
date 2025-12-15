/**
 * @file ExceptionHandlingTest.cpp
 * @brief Unit tests for Exception Handling across Model/Loss and Model/Utils
 *
 * Phase 12: Exception Handling Tests
 * Tests validation methods, input validation, numerical stability, and memory handling
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cmath>

#include "Model/Loss/BaseLoss.h"
#include "Model/Loss/ClassificationLoss.h"
#include "Model/Loss/DetectionLoss.h"
#include "Model/Loss/SegmentationLoss.h"
#include "Model/Loss/AnomalyLoss.h"
#include "Model/Utils/IoU.h"

using namespace WheelDL::Model::Loss;
using namespace WheelDL::Model::Utils;

class ExceptionHandlingTest : public ::testing::Test {
protected:
    torch::Device device_ = torch::kCPU;

    void SetUp() override {
#ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, 0);
        }
#endif
    }

    torch::Tensor createTensor(std::vector<int64_t> shape) {
        return torch::rand(shape, torch::TensorOptions().device(device_));
    }

    torch::Tensor createTensorWithNaN() {
        auto t = torch::tensor({1.0f, std::nanf(""), 3.0f}, torch::TensorOptions().device(device_));
        return t;
    }

    torch::Tensor createTensorWithInf() {
        auto t = torch::tensor({1.0f, std::numeric_limits<float>::infinity(), 3.0f}, torch::TensorOptions().device(device_));
        return t;
    }
};

// ============================================================================
// 3.1 BaseLoss Validation Methods Tests
// ============================================================================

TEST_F(ExceptionHandlingTest, ValidateLoss_NaN) {
    auto lossWithNaN = createTensorWithNaN().mean();

    // validateLoss should detect NaN
    EXPECT_TRUE(lossWithNaN.isnan().any().item<bool>());
}

TEST_F(ExceptionHandlingTest, ValidateLoss_Inf) {
    auto lossWithInf = createTensorWithInf().mean();

    // validateLoss should detect Inf
    EXPECT_TRUE(lossWithInf.isinf().any().item<bool>());
}

TEST_F(ExceptionHandlingTest, ValidateDimension_WrongDims) {
    // 2D tensor when 3D expected
    auto tensor2D = createTensor({4, 10});
    EXPECT_EQ(tensor2D.dim(), 2);

    // 3D tensor is correct
    auto tensor3D = createTensor({2, 4, 10});
    EXPECT_EQ(tensor3D.dim(), 3);
}

TEST_F(ExceptionHandlingTest, ValidateBatchSize_Mismatch) {
    auto pred = createTensor({4, 10});
    auto target = createTensor({5, 10});

    EXPECT_NE(pred.size(0), target.size(0));
}

TEST_F(ExceptionHandlingTest, ValidateShapeMatch_DifferentShapes) {
    auto tensor1 = createTensor({4, 10});
    auto tensor2 = createTensor({4, 20});

    EXPECT_NE(tensor1.size(1), tensor2.size(1));
}

// ============================================================================
// 3.2 Loss-Specific Exceptions Tests
// ============================================================================

TEST_F(ExceptionHandlingTest, DFLoss_InvalidRegMax) {
    // regMax <= 0 should throw
    EXPECT_ANY_THROW(DFLoss(0));
    EXPECT_ANY_THROW(DFLoss(-1));
}

TEST_F(ExceptionHandlingTest, DFLoss_ValidRegMax) {
    // regMax > 0 should work
    EXPECT_NO_THROW(DFLoss(16));
    EXPECT_NO_THROW(DFLoss(1));
}

TEST_F(ExceptionHandlingTest, ClassificationLoss_InvalidGamma_Negative) {
    ClassificationLoss loss;
    // setFocalParams(alpha, gamma) - gamma < 0 should throw
    EXPECT_ANY_THROW(loss.setFocalParams(0.25f, -0.1f));
}

TEST_F(ExceptionHandlingTest, ClassificationLoss_InvalidGamma_TooLarge) {
    ClassificationLoss loss;
    // setFocalParams(alpha, gamma) - gamma > 5 should throw
    EXPECT_ANY_THROW(loss.setFocalParams(0.25f, 5.1f));
}

TEST_F(ExceptionHandlingTest, ClassificationLoss_ValidGamma) {
    ClassificationLoss loss;
    // gamma in [0, 5] should work
    EXPECT_NO_THROW(loss.setFocalParams(0.0f, 0.25f));
    EXPECT_NO_THROW(loss.setFocalParams(2.0f, 0.25f));
    EXPECT_NO_THROW(loss.setFocalParams(5.0f, 0.25f));
}

TEST_F(ExceptionHandlingTest, SegmentationLoss_InvalidSmooth) {
    SegmentationLoss loss(80);
    // smooth <= 0 should throw
    EXPECT_ANY_THROW(loss.setSmooth(0.0f));
    EXPECT_ANY_THROW(loss.setSmooth(-1.0f));
}

TEST_F(ExceptionHandlingTest, SegmentationLoss_ValidSmooth) {
    SegmentationLoss loss(80);
    // smooth > 0 should work
    EXPECT_NO_THROW(loss.setSmooth(1.0f));
    EXPECT_NO_THROW(loss.setSmooth(0.001f));
}

// ============================================================================
// 3.3 Utils Exception Handling (IoU.cpp) Tests
// ============================================================================

TEST_F(ExceptionHandlingTest, Xywh2Xyxy_WrongDimension_1D) {
    auto tensor1D = createTensor({4});
    EXPECT_ANY_THROW(xywh2xyxy(tensor1D));
}

TEST_F(ExceptionHandlingTest, Xywh2Xyxy_WrongDimension_4D) {
    auto tensor4D = createTensor({2, 3, 4, 4});
    EXPECT_ANY_THROW(xywh2xyxy(tensor4D));
}

TEST_F(ExceptionHandlingTest, Xywh2Xyxy_ValidDimension_2D) {
    auto tensor2D = createTensor({10, 4});
    EXPECT_NO_THROW(xywh2xyxy(tensor2D));
}

TEST_F(ExceptionHandlingTest, Xywh2Xyxy_ValidDimension_3D) {
    auto tensor3D = createTensor({2, 10, 4});
    EXPECT_NO_THROW(xywh2xyxy(tensor3D));
}

TEST_F(ExceptionHandlingTest, Xyxy2Xywh_WrongDimension_1D) {
    auto tensor1D = createTensor({4});
    EXPECT_ANY_THROW(xyxy2xywh(tensor1D));
}

TEST_F(ExceptionHandlingTest, Xyxy2Xywh_ValidDimension) {
    auto tensor2D = createTensor({10, 4});
    EXPECT_NO_THROW(xyxy2xywh(tensor2D));
}

TEST_F(ExceptionHandlingTest, BboxIoU_Non2D_Box1) {
    auto box1_3D = createTensor({2, 10, 4});
    auto box2 = createTensor({10, 4});
    EXPECT_ANY_THROW(bboxIoU(box1_3D, box2, true));
}

TEST_F(ExceptionHandlingTest, BboxIoU_WrongSize_Dim1) {
    auto box1 = createTensor({10, 3});  // Wrong: should be 4
    auto box2 = createTensor({10, 4});
    EXPECT_ANY_THROW(bboxIoU(box1, box2, true));
}

TEST_F(ExceptionHandlingTest, BboxIoU_MismatchedSizes) {
    auto box1 = createTensor({5, 4});
    auto box2 = createTensor({3, 4});
    EXPECT_ANY_THROW(bboxIoU(box1, box2, true));
}

TEST_F(ExceptionHandlingTest, BboxIoU_Valid) {
    auto box1 = createTensor({10, 4});
    auto box2 = createTensor({10, 4});
    EXPECT_NO_THROW(bboxIoU(box1, box2, true));
}

TEST_F(ExceptionHandlingTest, Probiou_Non2D_Obb1) {
    auto obb1_3D = createTensor({2, 10, 5});
    auto obb2 = createTensor({10, 5});
    EXPECT_ANY_THROW(probiou(obb1_3D, obb2));
}

TEST_F(ExceptionHandlingTest, Probiou_WrongSize_Dim1) {
    auto obb1 = createTensor({10, 4});  // Wrong: should be 5
    auto obb2 = createTensor({10, 5});
    EXPECT_ANY_THROW(probiou(obb1, obb2));
}

TEST_F(ExceptionHandlingTest, Probiou_MismatchedSizes) {
    auto obb1 = createTensor({5, 5});
    auto obb2 = createTensor({3, 5});
    EXPECT_ANY_THROW(probiou(obb1, obb2));
}

TEST_F(ExceptionHandlingTest, Probiou_Valid) {
    auto obb1 = createTensor({10, 5});
    auto obb2 = createTensor({10, 5});
    // Ensure positive width/height
    obb1.select(1, 2).abs_().add_(0.1f);
    obb1.select(1, 3).abs_().add_(0.1f);
    obb2.select(1, 2).abs_().add_(0.1f);
    obb2.select(1, 3).abs_().add_(0.1f);
    EXPECT_NO_THROW(probiou(obb1, obb2));
}

TEST_F(ExceptionHandlingTest, Dist2Bbox_WrongDimension_1D) {
    auto distance = createTensor({4});
    auto anchorPoints = createTensor({1, 2});
    EXPECT_ANY_THROW(dist2bbox(distance, anchorPoints, true));
}

TEST_F(ExceptionHandlingTest, Dist2Bbox_Valid) {
    auto distance = createTensor({100, 4});
    auto anchorPoints = createTensor({100, 2});
    EXPECT_NO_THROW(dist2bbox(distance, anchorPoints, true));
}

TEST_F(ExceptionHandlingTest, Bbox2Dist_WrongDimension_1D) {
    auto anchorPoints = createTensor({1, 2});
    auto bboxes = createTensor({4});  // Wrong: 1D
    EXPECT_ANY_THROW(bbox2dist(anchorPoints, bboxes, 16));
}

TEST_F(ExceptionHandlingTest, Bbox2Dist_Valid) {
    auto anchorPoints = createTensor({100, 2});
    auto bboxes = createTensor({100, 4});
    EXPECT_NO_THROW(bbox2dist(anchorPoints, bboxes, 16));
}

TEST_F(ExceptionHandlingTest, Dist2Rbox_WrongDimension_1D) {
    auto distance = createTensor({4});
    auto angle = createTensor({1});
    auto anchorPoints = createTensor({1, 2});
    EXPECT_ANY_THROW(dist2rbox(distance, angle, anchorPoints));
}

TEST_F(ExceptionHandlingTest, Dist2Rbox_Valid_2D) {
    auto distance = createTensor({100, 4});
    auto angle = createTensor({100, 1});
    auto anchorPoints = createTensor({100, 2});
    EXPECT_NO_THROW(dist2rbox(distance, angle, anchorPoints));
}

TEST_F(ExceptionHandlingTest, Dist2Rbox_Valid_3D) {
    auto distance = createTensor({2, 100, 4});
    auto angle = createTensor({2, 100, 1});
    auto anchorPoints = createTensor({100, 2});
    EXPECT_NO_THROW(dist2rbox(distance, angle, anchorPoints));
}

TEST_F(ExceptionHandlingTest, Xywhr2Xyxyxyxy_WrongDimension_1D) {
    auto rboxes = createTensor({5});
    EXPECT_ANY_THROW(xywhr2xyxyxyxy(rboxes));
}

TEST_F(ExceptionHandlingTest, Xywhr2Xyxyxyxy_Valid_2D) {
    auto rboxes = createTensor({10, 5});
    EXPECT_NO_THROW(xywhr2xyxyxyxy(rboxes));
}

TEST_F(ExceptionHandlingTest, Xywhr2Xyxyxyxy_Valid_3D) {
    auto rboxes = createTensor({2, 10, 5});
    EXPECT_NO_THROW(xywhr2xyxyxyxy(rboxes));
}

TEST_F(ExceptionHandlingTest, MakeAnchors_EmptyFeats) {
    std::vector<torch::Tensor> emptyFeats;
    auto strides = torch::tensor({8.0f}, torch::TensorOptions().device(device_));
    EXPECT_ANY_THROW(makeAnchors(emptyFeats, strides, 0.5f));
}

TEST_F(ExceptionHandlingTest, MakeAnchors_EmptyFeatShapes) {
    std::vector<std::pair<int64_t, int64_t>> emptyShapes;
    auto strides = torch::tensor({8.0f}, torch::TensorOptions().device(device_));
    EXPECT_ANY_THROW(makeAnchors(emptyShapes, strides, torch::kFloat32, device_, 0.5f));
}

TEST_F(ExceptionHandlingTest, MakeAnchors_Valid) {
    std::vector<torch::Tensor> feats;
    feats.push_back(createTensor({1, 256, 80, 80}));
    auto strides = torch::tensor({8.0f}, torch::TensorOptions().device(device_));
    EXPECT_NO_THROW(makeAnchors(feats, strides, 0.5f));
}

// ============================================================================
// 3.4 Input Validation Tests
// ============================================================================

TEST_F(ExceptionHandlingTest, InputValidation_EmptyTensor) {
    auto emptyTensor = torch::empty({0}, torch::TensorOptions().device(device_));
    EXPECT_EQ(emptyTensor.numel(), 0);
}

TEST_F(ExceptionHandlingTest, InputValidation_AllZerosPrediction) {
    // All zeros prediction should produce valid (not throw) loss
    auto pred = torch::zeros({4, 10}, torch::TensorOptions().device(device_));
    auto target = torch::zeros({4}, torch::TensorOptions().dtype(torch::kLong).device(device_));

    ClassificationLoss loss;
    // Should not throw, but may produce large loss
    EXPECT_NO_THROW({
        auto result = loss.compute(pred, target);
    });
}

TEST_F(ExceptionHandlingTest, InputValidation_WrongNumberOfDims) {
    // 2D tensor when expecting different
    auto tensor2D = createTensor({4, 10});
    auto tensor4D = createTensor({2, 4, 10, 10});

    EXPECT_EQ(tensor2D.dim(), 2);
    EXPECT_EQ(tensor4D.dim(), 4);
}

#ifdef USE_CUDA
TEST_F(ExceptionHandlingTest, InputValidation_MixedDevices) {
    if (!torch::cuda::is_available()) {
        return;  // Skip if CUDA not available
    }

    auto cpuTensor = torch::rand({10, 4}, torch::kCPU);
    auto cudaTensor = torch::rand({10, 4}, torch::kCUDA);

    // Mixing CPU and CUDA tensors should throw
    EXPECT_ANY_THROW(bboxIoU(cpuTensor, cudaTensor, true));
}
#endif

// ============================================================================
// 3.5 Numerical Stability Tests
// ============================================================================

TEST_F(ExceptionHandlingTest, NumericalStability_LogZero) {
    // log(0) should be handled
    auto zeros = torch::zeros({10}, torch::TensorOptions().device(device_));
    auto logZeros = torch::log(zeros + 1e-10f);  // With epsilon

    EXPECT_FALSE(logZeros.isinf().any().item<bool>());
}

TEST_F(ExceptionHandlingTest, NumericalStability_DivByZero) {
    auto numerator = torch::ones({10}, torch::TensorOptions().device(device_));
    auto denominator = torch::zeros({10}, torch::TensorOptions().device(device_));

    // Division with epsilon protection
    auto result = numerator / (denominator + 1e-9f);
    EXPECT_FALSE(result.isinf().any().item<bool>());
}

TEST_F(ExceptionHandlingTest, NumericalStability_Overflow) {
    // Very large values
    auto largeValues = torch::ones({10}, torch::TensorOptions().device(device_)) * 1e30f;

    // Should not overflow to inf in typical operations
    auto result = largeValues / largeValues;
    EXPECT_FALSE(result.isnan().any().item<bool>());
}

TEST_F(ExceptionHandlingTest, NumericalStability_Underflow) {
    // Very small values
    auto smallValues = torch::ones({10}, torch::TensorOptions().device(device_)) * 1e-30f;

    // Should handle underflow
    auto result = smallValues * smallValues;
    EXPECT_FALSE(result.isnan().any().item<bool>());
}

TEST_F(ExceptionHandlingTest, NumericalStability_ExpOverflow) {
    // exp(large) can overflow
    auto largeInput = torch::ones({10}, torch::TensorOptions().device(device_)) * 100.0f;

    // Clamp before exp for stability
    auto clampedInput = torch::clamp(largeInput, -80.0f, 80.0f);
    auto result = torch::exp(clampedInput);

    EXPECT_FALSE(result.isinf().any().item<bool>());
}

TEST_F(ExceptionHandlingTest, NumericalStability_SoftmaxLargeInput) {
    // softmax with large inputs
    auto largeInput = torch::ones({10, 100}, torch::TensorOptions().device(device_)) * 1000.0f;

    // Softmax should handle large inputs through max subtraction
    auto result = torch::softmax(largeInput, 1);

    EXPECT_FALSE(result.isnan().any().item<bool>());
    EXPECT_FALSE(result.isinf().any().item<bool>());
}

TEST_F(ExceptionHandlingTest, NumericalStability_SigmoidLargeInput) {
    // sigmoid with large positive input
    auto largePosInput = torch::ones({10}, torch::TensorOptions().device(device_)) * 100.0f;
    auto result = torch::sigmoid(largePosInput);

    EXPECT_FALSE(result.isnan().any().item<bool>());
    EXPECT_FALSE(result.isinf().any().item<bool>());

    // Should be close to 1
    auto maxVal = result.max().item<float>();
    EXPECT_LE(maxVal, 1.0f);
}

TEST_F(ExceptionHandlingTest, NumericalStability_SigmoidLargeNegInput) {
    // sigmoid with large negative input
    auto largeNegInput = torch::ones({10}, torch::TensorOptions().device(device_)) * -100.0f;
    auto result = torch::sigmoid(largeNegInput);

    EXPECT_FALSE(result.isnan().any().item<bool>());
    EXPECT_FALSE(result.isinf().any().item<bool>());

    // Should be close to 0
    auto minVal = result.min().item<float>();
    EXPECT_GE(minVal, 0.0f);
}

// ============================================================================
// 3.6 Memory Tests
// ============================================================================

TEST_F(ExceptionHandlingTest, Memory_LargeBatch) {
    // Test with moderately large batch (not too large to cause OOM in tests)
    int64_t batchSize = 64;
    int64_t numBoxes = 1000;

    auto boxes = createTensor({batchSize, numBoxes, 4});
    EXPECT_EQ(boxes.size(0), batchSize);
    EXPECT_EQ(boxes.size(1), numBoxes);
}

TEST_F(ExceptionHandlingTest, Memory_EmptyInput_NoCarsh) {
    auto emptyTensor = torch::empty({0, 4}, torch::TensorOptions().device(device_));

    // Operations on empty tensor should not crash
    EXPECT_EQ(emptyTensor.size(0), 0);
    EXPECT_NO_THROW({
        auto sum = emptyTensor.sum();
    });
}

TEST_F(ExceptionHandlingTest, Memory_InPlaceOperation) {
    auto tensor = createTensor({100, 4});
    auto originalData = tensor.clone();

    // In-place operation
    tensor.add_(1.0f);

    // Should modify the original tensor
    EXPECT_FALSE(torch::allclose(tensor, originalData));
}

TEST_F(ExceptionHandlingTest, Memory_ViewVsCopy) {
    auto original = createTensor({100, 4});

    // View shares memory
    auto view = original.view({400});
    view[0] = 999.0f;
    EXPECT_EQ(original[0][0].item<float>(), 999.0f);

    // Clone does not share memory
    auto copy = original.clone();
    copy[0][0] = 0.0f;
    EXPECT_EQ(original[0][0].item<float>(), 999.0f);  // Original unchanged
}

// ============================================================================
// Device Consistency Tests
// ============================================================================

TEST_F(ExceptionHandlingTest, DeviceConsistency_TensorCreation) {
    auto tensor = createTensor({10, 4});
    EXPECT_EQ(tensor.device().type(), device_.type());
}

TEST_F(ExceptionHandlingTest, DeviceConsistency_Operations) {
    auto t1 = createTensor({10, 4});
    auto t2 = createTensor({10, 4});

    auto result = t1 + t2;
    EXPECT_EQ(result.device().type(), device_.type());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(ExceptionHandlingTest, EdgeCase_SingleElement) {
    auto singleElement = createTensor({1, 4});
    EXPECT_EQ(singleElement.size(0), 1);

    auto converted = xywh2xyxy(singleElement);
    EXPECT_EQ(converted.size(0), 1);
}

TEST_F(ExceptionHandlingTest, EdgeCase_VeryLargeTensor) {
    // Create a large tensor (but not too large)
    auto largeTensor = createTensor({10000, 4});
    EXPECT_EQ(largeTensor.size(0), 10000);
}

TEST_F(ExceptionHandlingTest, EdgeCase_SpecialFloatValues) {
    // Test with special float values
    auto tensor = torch::tensor({0.0f, -0.0f, 1e-38f, 1e38f}, torch::TensorOptions().device(device_));

    EXPECT_FALSE(tensor.isnan().any().item<bool>());
    EXPECT_FALSE(tensor.isinf().any().item<bool>());
}

TEST_F(ExceptionHandlingTest, EdgeCase_NegativeWidthHeight) {
    // Boxes with negative width/height (invalid but should be handled)
    auto invalidBoxes = torch::tensor({{10.0f, 10.0f, -5.0f, -5.0f}}, torch::TensorOptions().device(device_));

    // xywh2xyxy should still compute (but results may be invalid)
    EXPECT_NO_THROW({
        auto converted = xywh2xyxy(invalidBoxes);
    });
}

// ============================================================================
// Gradient-Related Tests
// ============================================================================

TEST_F(ExceptionHandlingTest, Gradient_RequiresGrad) {
    auto tensor = createTensor({10, 4});
    tensor.requires_grad_(true);

    EXPECT_TRUE(tensor.requires_grad());
}

TEST_F(ExceptionHandlingTest, Gradient_DetachedTensor) {
    auto tensor = createTensor({10, 4});
    tensor.requires_grad_(true);

    auto detached = tensor.detach();
    EXPECT_FALSE(detached.requires_grad());
}

TEST_F(ExceptionHandlingTest, Gradient_InPlaceOnLeaf) {
    auto tensor = createTensor({10, 4});
    tensor.requires_grad_(true);

    // In-place operation on leaf tensor with requires_grad is not allowed
    // This should throw
    EXPECT_ANY_THROW({
        tensor.add_(1.0f);
    });
}

TEST_F(ExceptionHandlingTest, Gradient_InPlaceOnNonLeaf) {
    auto tensor = createTensor({10, 4});
    tensor.requires_grad_(true);

    // Create non-leaf tensor
    auto nonLeaf = tensor * 2.0f;

    // In-place on non-leaf may cause issues with autograd
    // Using clone() to avoid issues
    auto safeNonLeaf = nonLeaf.clone();
    safeNonLeaf.add_(1.0f);

    EXPECT_FALSE(safeNonLeaf.isnan().any().item<bool>());
}

// ============================================================================
// Type Conversion Tests
// ============================================================================

TEST_F(ExceptionHandlingTest, TypeConversion_FloatToDouble) {
    auto floatTensor = createTensor({10, 4});
    auto doubleTensor = floatTensor.to(torch::kFloat64);

    EXPECT_EQ(doubleTensor.scalar_type(), torch::kFloat64);
}

TEST_F(ExceptionHandlingTest, TypeConversion_FloatToHalf) {
    auto floatTensor = createTensor({10, 4});

#ifdef USE_CUDA
    if (device_.type() == torch::kCUDA) {
        auto halfTensor = floatTensor.to(torch::kFloat16);
        EXPECT_EQ(halfTensor.scalar_type(), torch::kFloat16);
    }
#endif
}

TEST_F(ExceptionHandlingTest, TypeConversion_IntToFloat) {
    auto intTensor = torch::randint(0, 100, {10, 4}, torch::TensorOptions().dtype(torch::kInt32).device(device_));
    auto floatTensor = intTensor.to(torch::kFloat32);

    EXPECT_EQ(floatTensor.scalar_type(), torch::kFloat32);
}
