#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Loss/DetectionLoss.h"
#include "Data/Dataset/BaseDataset.h"

using namespace WheelDL::Model::Loss;
using namespace WheelDL::Data::Dataset;

class DetectionLossTest : public ::testing::Test {
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

    torch::Tensor createStride() {
        return torch::tensor({ 8.0f, 16.0f, 32.0f }).to(device_);
    }

    // Create prediction tensor for detection: [B, C, H, W] format
    // C = regMax*4 + numClasses = 16*4 + 80 = 144 for default
    torch::Tensor createPrediction(int64_t batchSize, int64_t numClasses = 80, int64_t regMax = 16, int64_t height = 80, int64_t width = 80) {
        int64_t channels = regMax * 4 + numClasses;
        return torch::randn({ batchSize, channels, height, width }, device_);
    }

    // Create concatenated prediction: [B, C, TotalAnchors]
    torch::Tensor createConcatenatedPrediction(int64_t batchSize, int64_t numClasses = 80, int64_t regMax = 16, int64_t imageSize = 640) {
        int64_t channels = regMax * 4 + numClasses;
        // For strides 8, 16, 32 and imageSize 640: 80*80 + 40*40 + 20*20 = 8400 anchors
        int64_t totalAnchors = (imageSize / 8) * (imageSize / 8) +
            (imageSize / 16) * (imageSize / 16) +
            (imageSize / 32) * (imageSize / 32);
        return torch::randn({ batchSize, channels, totalAnchors }, device_);
    }

    // Create multi-scale predictions: vector of [B, C, H, W]
    std::vector<torch::Tensor> createMultiScalePredictions(int64_t batchSize, int64_t numClasses = 80, int64_t regMax = 16, int64_t imageSize = 640) {
        int64_t channels = regMax * 4 + numClasses;
        std::vector<torch::Tensor> predictions;

        // Stride 8: 80x80
        predictions.push_back(torch::randn({ batchSize, channels, imageSize / 8, imageSize / 8 }, device_));
        // Stride 16: 40x40
        predictions.push_back(torch::randn({ batchSize, channels, imageSize / 16, imageSize / 16 }, device_));
        // Stride 32: 20x20
        predictions.push_back(torch::randn({ batchSize, channels, imageSize / 32, imageSize / 32 }, device_));

        return predictions;
    }

    // Create targets: [N, 6] format where columns are [batch_idx, class_id, cx, cy, w, h] (normalized)
    DataExample createDetectionTarget(int64_t numTargets, int64_t batchSize, int64_t numClasses = 80) {
        DataExample example;

        if (numTargets > 0) {
            // Create targets with batch indices
            auto batchIndices = torch::randint(0, batchSize, { numTargets, 1 }, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
            auto classIds = torch::randint(0, numClasses, { numTargets, 1 }, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
            // cx, cy in [0.2, 0.8], w, h in [0.1, 0.3] (normalized)
            auto cx = torch::rand({ numTargets, 1 }, device_) * 0.6f + 0.2f;
            auto cy = torch::rand({ numTargets, 1 }, device_) * 0.6f + 0.2f;
            auto w = torch::rand({ numTargets, 1 }, device_) * 0.2f + 0.1f;
            auto h = torch::rand({ numTargets, 1 }, device_) * 0.2f + 0.1f;

            example.targets = torch::cat({ batchIndices, classIds, cx, cy, w, h }, 1);
        }
        else {
            example.targets = torch::zeros({ 0, 6 }, device_);
        }

        return example;
    }

    // Create empty target
    DataExample createEmptyTarget() {
        DataExample example;
        example.targets = torch::zeros({ 0, 6 }, device_);
        return example;
    }
};

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_F(DetectionLossTest, Constructor_DefaultParameters) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);

    EXPECT_EQ(loss.name(), "DetectionLoss");

    auto [boxGain, clsGain, dflGain] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(boxGain, 7.5f);
    EXPECT_FLOAT_EQ(clsGain, 0.5f);
    EXPECT_FLOAT_EQ(dflGain, 1.5f);
}

TEST_F(DetectionLossTest, Constructor_CustomParameters) {
    auto stride = createStride();
    DetectionLoss loss(10, 320, stride, 5.0f, 1.0f, 2.0f, 5);

    auto [boxGain, clsGain, dflGain] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(boxGain, 5.0f);
    EXPECT_FLOAT_EQ(clsGain, 1.0f);
    EXPECT_FLOAT_EQ(dflGain, 2.0f);
}

TEST_F(DetectionLossTest, Constructor_SingleClass) {
    auto stride = createStride();
    DetectionLoss loss(1, 640, stride);
    EXPECT_EQ(loss.name(), "DetectionLoss");
}

TEST_F(DetectionLossTest, Constructor_ManyClasses) {
    auto stride = createStride();
    DetectionLoss loss(1000, 640, stride);
    EXPECT_EQ(loss.name(), "DetectionLoss");
}

TEST_F(DetectionLossTest, Constructor_SmallImageSize) {
    auto stride = torch::tensor({ 8.0f, 16.0f }).to(device_);
    DetectionLoss loss(80, 128, stride);
    EXPECT_EQ(loss.name(), "DetectionLoss");
}

TEST_F(DetectionLossTest, Constructor_LargeImageSize) {
    auto stride = createStride();
    DetectionLoss loss(80, 1280, stride);
    EXPECT_EQ(loss.name(), "DetectionLoss");
}

// ============================================================================
// API Tests
// ============================================================================

TEST_F(DetectionLossTest, Name_ReturnsCorrectName) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    EXPECT_EQ(loss.name(), "DetectionLoss");
}

TEST_F(DetectionLossTest, SetLossWeights_Valid) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);

    loss.setLossWeights(10.0f, 2.0f, 3.0f);

    auto [boxGain, clsGain, dflGain] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(boxGain, 10.0f);
    EXPECT_FLOAT_EQ(clsGain, 2.0f);
    EXPECT_FLOAT_EQ(dflGain, 3.0f);
}

TEST_F(DetectionLossTest, SetLossWeights_Zero) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);

    loss.setLossWeights(0.0f, 0.0f, 0.0f);

    auto [boxGain, clsGain, dflGain] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(boxGain, 0.0f);
    EXPECT_FLOAT_EQ(clsGain, 0.0f);
    EXPECT_FLOAT_EQ(dflGain, 0.0f);
}

TEST_F(DetectionLossTest, GetLossWeights_Default) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);

    auto [boxGain, clsGain, dflGain] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(boxGain, 7.5f);
    EXPECT_FLOAT_EQ(clsGain, 0.5f);
    EXPECT_FLOAT_EQ(dflGain, 1.5f);
}

TEST_F(DetectionLossTest, ToDevice_CPU) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    EXPECT_NO_THROW(loss.to(torch::kCPU));
}

#ifdef USE_CUDA
TEST_F(DetectionLossTest, ToDevice_CUDA) {
    if (torch::cuda::is_available()) {
        auto stride = createStride().cuda();
        DetectionLoss loss(80, 640, stride);
        EXPECT_NO_THROW(loss.to(torch::kCUDA));
    }
}
#endif

// ============================================================================
// Compute Tests - Single Tensor
// ============================================================================

TEST_F(DetectionLossTest, Compute_SingleTensor_WithTargets) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createDetectionTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("box") != resultMap.end());
    EXPECT_TRUE(resultMap.find("cls") != resultMap.end());
    EXPECT_TRUE(resultMap.find("dfl") != resultMap.end());
    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
}

TEST_F(DetectionLossTest, Compute_SingleTensor_NoTargets) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createEmptyTarget();

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("box") != resultMap.end());
    EXPECT_TRUE(resultMap.find("cls") != resultMap.end());
    EXPECT_TRUE(resultMap.find("dfl") != resultMap.end());
    EXPECT_TRUE(resultMap.find("total") != resultMap.end());

    // With no targets, box and dfl losses should be 0
    EXPECT_FLOAT_EQ(resultMap["box"].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(resultMap["dfl"].item<float>(), 0.0f);
}

TEST_F(DetectionLossTest, Compute_SingleTensor_BatchSize1) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(1, 80, 16, 640);
    auto target = createDetectionTarget(3, 1, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(DetectionLossTest, Compute_SingleTensor_BatchSize8) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(8, 80, 16, 640);
    auto target = createDetectionTarget(20, 8, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(DetectionLossTest, Compute_SingleTensor_LossComponentsValid) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createDetectionTarget(10, 2, 80);

    auto resultMap = loss.compute(pred, target);

    // All loss components should be non-negative
    EXPECT_GE(resultMap["box"].item<float>(), 0.0f);
    EXPECT_GE(resultMap["cls"].item<float>(), 0.0f);
    EXPECT_GE(resultMap["dfl"].item<float>(), 0.0f);
    EXPECT_GE(resultMap["total"].item<float>(), 0.0f);
}

// ============================================================================
// Compute Tests - Multi-Scale (Vector of Tensors)
// ============================================================================

TEST_F(DetectionLossTest, Compute_MultiScale_WithTargets) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto predictions = createMultiScalePredictions(2, 80, 16, 640);
    auto target = createDetectionTarget(5, 2, 80);

    auto resultMap = loss.compute(predictions, target);

    EXPECT_TRUE(resultMap.find("box") != resultMap.end());
    EXPECT_TRUE(resultMap.find("cls") != resultMap.end());
    EXPECT_TRUE(resultMap.find("dfl") != resultMap.end());
    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
}

TEST_F(DetectionLossTest, Compute_MultiScale_NoTargets) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto predictions = createMultiScalePredictions(2, 80, 16, 640);
    auto target = createEmptyTarget();

    auto resultMap = loss.compute(predictions, target);

    EXPECT_FLOAT_EQ(resultMap["box"].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(resultMap["dfl"].item<float>(), 0.0f);
}

TEST_F(DetectionLossTest, Compute_MultiScale_SingleScale) {
    auto stride = torch::tensor({ 8.0f }).to(device_);
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 80;
    std::vector<torch::Tensor> predictions;
    predictions.push_back(torch::randn({ 2, channels, 80, 80 }, device_));

    auto target = createDetectionTarget(3, 2, 80);

    auto resultMap = loss.compute(predictions, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
}

TEST_F(DetectionLossTest, Compute_MultiScale_FourScales) {
    auto stride = torch::tensor({ 4.0f, 8.0f, 16.0f, 32.0f }).to(device_);
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 80;
    std::vector<torch::Tensor> predictions;
    predictions.push_back(torch::randn({ 2, channels, 160, 160 }, device_));  // stride 4
    predictions.push_back(torch::randn({ 2, channels, 80, 80 }, device_));    // stride 8
    predictions.push_back(torch::randn({ 2, channels, 40, 40 }, device_));    // stride 16
    predictions.push_back(torch::randn({ 2, channels, 20, 20 }, device_));    // stride 32

    auto target = createDetectionTarget(8, 2, 80);

    auto resultMap = loss.compute(predictions, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

// ============================================================================
// Exception Tests
// ============================================================================

TEST_F(DetectionLossTest, Compute_EmptyPredictions_Throws) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    std::vector<torch::Tensor> emptyPredictions;
    auto target = createDetectionTarget(5, 2, 80);

    EXPECT_THROW(loss.compute(emptyPredictions, target), std::invalid_argument);
}

// ============================================================================
// Loss Weights Effect Tests
// ============================================================================

TEST_F(DetectionLossTest, LossWeights_BoxGainEffect) {
    auto stride = createStride();

    // First with default weights
    DetectionLoss loss1(80, 640, stride);
    loss1.to(device_);
    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createDetectionTarget(5, 2, 80);
    auto result1 = loss1.compute(pred, target);

    // Then with doubled box gain
    DetectionLoss loss2(80, 640, stride, 15.0f, 0.5f, 1.5f);
    loss2.to(device_);
    auto result2 = loss2.compute(pred, target);

    // Box component should be doubled (approximately)
    // Note: The total won't be exactly doubled due to other components
    EXPECT_NE(result1["box"].item<float>(), result2["box"].item<float>());
}

TEST_F(DetectionLossTest, LossWeights_ClsGainEffect) {
    auto stride = createStride();

    DetectionLoss loss1(80, 640, stride);
    loss1.to(device_);
    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createDetectionTarget(5, 2, 80);
    auto result1 = loss1.compute(pred, target);

    DetectionLoss loss2(80, 640, stride, 7.5f, 1.0f, 1.5f);
    loss2.to(device_);
    auto result2 = loss2.compute(pred, target);

    // Classification loss should differ
    EXPECT_NE(result1["cls"].item<float>(), result2["cls"].item<float>());
}

TEST_F(DetectionLossTest, LossWeights_DflGainEffect) {
    auto stride = createStride();

    DetectionLoss loss1(80, 640, stride);
    loss1.to(device_);
    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createDetectionTarget(5, 2, 80);
    auto result1 = loss1.compute(pred, target);

    DetectionLoss loss2(80, 640, stride, 7.5f, 0.5f, 3.0f);
    loss2.to(device_);
    auto result2 = loss2.compute(pred, target);

    EXPECT_NE(result1["dfl"].item<float>(), result2["dfl"].item<float>());
}

TEST_F(DetectionLossTest, LossWeights_ZeroWeights) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride, 0.0f, 0.0f, 0.0f);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createDetectionTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    // With zero weights, individual components should be 0
    EXPECT_FLOAT_EQ(resultMap["box"].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(resultMap["cls"].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(resultMap["dfl"].item<float>(), 0.0f);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_F(DetectionLossTest, NumericalStability_LargePredictions) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 80;
    auto pred = torch::randn({ 2, channels, 8400 }, device_) * 100.0f;  // Large values
    auto target = createDetectionTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(DetectionLossTest, NumericalStability_SmallPredictions) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 80;
    auto pred = torch::randn({ 2, channels, 8400 }, device_) * 0.001f;  // Small values
    auto target = createDetectionTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(DetectionLossTest, NumericalStability_ZeroPredictions) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 80;
    auto pred = torch::zeros({ 2, channels, 8400 }, device_);
    auto target = createDetectionTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

// ============================================================================
// Different Class Counts Tests
// ============================================================================

TEST_F(DetectionLossTest, Compute_SingleClass) {
    auto stride = createStride();
    DetectionLoss loss(1, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 1;
    int64_t totalAnchors = 80 * 80 + 40 * 40 + 20 * 20;
    auto pred = torch::randn({ 2, channels, totalAnchors }, device_);
    auto target = createDetectionTarget(3, 2, 1);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(DetectionLossTest, Compute_FewClasses) {
    auto stride = createStride();
    DetectionLoss loss(10, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 10;
    int64_t totalAnchors = 80 * 80 + 40 * 40 + 20 * 20;
    auto pred = torch::randn({ 2, channels, totalAnchors }, device_);
    auto target = createDetectionTarget(5, 2, 10);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(DetectionLossTest, Compute_ManyClasses) {
    auto stride = createStride();
    DetectionLoss loss(200, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 200;
    int64_t totalAnchors = 80 * 80 + 40 * 40 + 20 * 20;
    auto pred = torch::randn({ 2, channels, totalAnchors }, device_);
    auto target = createDetectionTarget(5, 2, 200);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

// ============================================================================
// Different Image Sizes Tests
// ============================================================================

TEST_F(DetectionLossTest, Compute_SmallImage) {
    auto stride = torch::tensor({ 8.0f, 16.0f }).to(device_);
    DetectionLoss loss(80, 128, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 80;
    // 128/8=16, 128/16=8 -> 16*16 + 8*8 = 320 anchors
    int64_t totalAnchors = 16 * 16 + 8 * 8;
    auto pred = torch::randn({ 2, channels, totalAnchors }, device_);
    auto target = createDetectionTarget(3, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
}

TEST_F(DetectionLossTest, Compute_LargeImage) {
    auto stride = createStride();
    DetectionLoss loss(80, 1280, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 80;
    // 1280/8=160, 1280/16=80, 1280/32=40
    int64_t totalAnchors = 160 * 160 + 80 * 80 + 40 * 40;
    auto pred = torch::randn({ 2, channels, totalAnchors }, device_);
    auto target = createDetectionTarget(10, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
}

// ============================================================================
// Target Variation Tests
// ============================================================================

TEST_F(DetectionLossTest, Compute_SingleTarget) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createDetectionTarget(1, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(DetectionLossTest, Compute_ManyTargets) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(4, 80, 16, 640);
    auto target = createDetectionTarget(100, 4, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(DetectionLossTest, Compute_AllTargetsOneBatch) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);

    // All targets in batch 0
    DataExample example;
    auto numTargets = 5;
    auto batchIndices = torch::zeros({ numTargets, 1 }, device_);  // All in batch 0
    auto classIds = torch::randint(0, 80, { numTargets, 1 }, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto cx = torch::rand({ numTargets, 1 }, device_) * 0.6f + 0.2f;
    auto cy = torch::rand({ numTargets, 1 }, device_) * 0.6f + 0.2f;
    auto w = torch::rand({ numTargets, 1 }, device_) * 0.2f + 0.1f;
    auto h = torch::rand({ numTargets, 1 }, device_) * 0.2f + 0.1f;
    example.targets = torch::cat({ batchIndices, classIds, cx, cy, w, h }, 1);

    auto resultMap = loss.compute(pred, example);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

// ============================================================================
// Gradient Tests
// ============================================================================

TEST_F(DetectionLossTest, Gradient_BackwardPass) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 80;
    auto pred = torch::randn({ 2, channels, 8400 }, torch::TensorOptions().device(device_).requires_grad(true));
    auto target = createDetectionTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);
    resultMap["total"].backward();

    EXPECT_TRUE(pred.grad().defined());
    EXPECT_FALSE(pred.grad().isnan().any().item<bool>());
    EXPECT_FALSE(pred.grad().isinf().any().item<bool>());
}

TEST_F(DetectionLossTest, Gradient_MultiScale_BackwardPass) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 80;
    std::vector<torch::Tensor> predictions;
    predictions.push_back(torch::randn({ 2, channels, 80, 80 }, torch::TensorOptions().device(device_).requires_grad(true)));
    predictions.push_back(torch::randn({ 2, channels, 40, 40 }, torch::TensorOptions().device(device_).requires_grad(true)));
    predictions.push_back(torch::randn({ 2, channels, 20, 20 }, torch::TensorOptions().device(device_).requires_grad(true)));

    auto target = createDetectionTarget(5, 2, 80);

    auto resultMap = loss.compute(predictions, target);
    resultMap["total"].backward();

    for (const auto& pred : predictions) {
        EXPECT_TRUE(pred.grad().defined());
        EXPECT_FALSE(pred.grad().isnan().any().item<bool>());
    }
}

// ============================================================================
// Return Map Structure Tests
// ============================================================================

TEST_F(DetectionLossTest, ReturnMap_ContainsAllKeys) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createDetectionTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_EQ(resultMap.size(), 4);
    EXPECT_TRUE(resultMap.count("box") == 1);
    EXPECT_TRUE(resultMap.count("cls") == 1);
    EXPECT_TRUE(resultMap.count("dfl") == 1);
    EXPECT_TRUE(resultMap.count("total") == 1);
}

TEST_F(DetectionLossTest, ReturnMap_TensorShapes) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createDetectionTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    // All values should be scalar tensors
    EXPECT_EQ(resultMap["box"].dim(), 0);
    EXPECT_EQ(resultMap["cls"].dim(), 0);
    EXPECT_EQ(resultMap["dfl"].dim(), 0);
    EXPECT_EQ(resultMap["total"].dim(), 0);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(DetectionLossTest, EdgeCase_VerySmallTargetBox) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(1, 80, 16, 640);

    // Create target with very small box
    DataExample example;
    example.targets = torch::tensor({
        {0.0f, 0.0f, 0.5f, 0.5f, 0.01f, 0.01f}  // Very small w, h
        }).to(device_);

    auto resultMap = loss.compute(pred, example);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(DetectionLossTest, EdgeCase_TargetAtImageEdge) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(1, 80, 16, 640);

    // Create target at image edge
    DataExample example;
    example.targets = torch::tensor({
        {0.0f, 0.0f, 0.95f, 0.95f, 0.1f, 0.1f}  // Near edge
        }).to(device_);

    auto resultMap = loss.compute(pred, example);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(DetectionLossTest, EdgeCase_TargetAtImageCenter) {
    auto stride = createStride();
    DetectionLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(1, 80, 16, 640);

    // Create target at image center
    DataExample example;
    example.targets = torch::tensor({
        {0.0f, 0.0f, 0.5f, 0.5f, 0.2f, 0.2f}  // Center
        }).to(device_);

    auto resultMap = loss.compute(pred, example);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}
