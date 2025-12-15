#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Loss/OBBLoss.h"
#include "Data/Dataset/BaseDataset.h"

using namespace WheelDL::Model::Loss;
using namespace WheelDL::Data::Dataset;

class OBBLossTest : public ::testing::Test {
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

    // Create prediction tensor for OBB: [B, C, TotalAnchors]
    // C = regMax*4 + 1(angle) + numClasses = 16*4 + 1 + 80 = 145 for default
    torch::Tensor createConcatenatedPrediction(int64_t batchSize, int64_t numClasses = 80, int64_t regMax = 16, int64_t imageSize = 640) {
        int64_t channels = regMax * 4 + 1 + numClasses;  // +1 for angle
        int64_t totalAnchors = (imageSize / 8) * (imageSize / 8) +
            (imageSize / 16) * (imageSize / 16) +
            (imageSize / 32) * (imageSize / 32);
        return torch::randn({ batchSize, channels, totalAnchors }, device_);
    }

    // Create multi-scale predictions: vector of [B, C, H, W]
    std::vector<torch::Tensor> createMultiScalePredictions(int64_t batchSize, int64_t numClasses = 80, int64_t regMax = 16, int64_t imageSize = 640) {
        int64_t channels = regMax * 4 + 1 + numClasses;  // +1 for angle
        std::vector<torch::Tensor> predictions;

        // Stride 8: 80x80
        predictions.push_back(torch::randn({ batchSize, channels, imageSize / 8, imageSize / 8 }, device_));
        // Stride 16: 40x40
        predictions.push_back(torch::randn({ batchSize, channels, imageSize / 16, imageSize / 16 }, device_));
        // Stride 32: 20x20
        predictions.push_back(torch::randn({ batchSize, channels, imageSize / 32, imageSize / 32 }, device_));

        return predictions;
    }

    // Create OBB targets: [N, 7] format where columns are [batch_idx, class_id, cx, cy, w, h, angle] (normalized)
    DataExample createOBBTarget(int64_t numTargets, int64_t batchSize, int64_t numClasses = 80) {
        DataExample example;

        if (numTargets > 0) {
            auto batchIndices = torch::randint(0, batchSize, { numTargets, 1 }, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
            auto classIds = torch::randint(0, numClasses, { numTargets, 1 }, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
            // cx, cy in [0.2, 0.8], w, h in [0.1, 0.3] (normalized)
            auto cx = torch::rand({ numTargets, 1 }, device_) * 0.6f + 0.2f;
            auto cy = torch::rand({ numTargets, 1 }, device_) * 0.6f + 0.2f;
            auto w = torch::rand({ numTargets, 1 }, device_) * 0.2f + 0.1f;
            auto h = torch::rand({ numTargets, 1 }, device_) * 0.2f + 0.1f;
            // angle in [-pi/2, pi/2]
            auto angle = (torch::rand({ numTargets, 1 }, device_) - 0.5f) * 3.14159f;

            example.targets = torch::cat({ batchIndices, classIds, cx, cy, w, h, angle }, 1);
        }
        else {
            example.targets = torch::zeros({ 0, 7 }, device_);
        }

        return example;
    }

    // Create empty target
    DataExample createEmptyTarget() {
        DataExample example;
        example.targets = torch::zeros({ 0, 7 }, device_);
        return example;
    }

    // Create target with specific angle
    DataExample createOBBTargetWithAngle(int64_t batchSize, float angle) {
        DataExample example;
        auto batchIndices = torch::zeros({ 1, 1 }, device_);
        auto classIds = torch::zeros({ 1, 1 }, device_);
        auto cx = torch::tensor({ {0.5f} }, device_);
        auto cy = torch::tensor({ {0.5f} }, device_);
        auto w = torch::tensor({ {0.2f} }, device_);
        auto h = torch::tensor({ {0.1f} }, device_);
        auto angleT = torch::tensor({ {angle} }, device_);

        example.targets = torch::cat({ batchIndices, classIds, cx, cy, w, h, angleT }, 1);
        return example;
    }
};

// ============================================================================
// Constructor Tests
// ============================================================================

TEST_F(OBBLossTest, Constructor_DefaultParameters) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);

    EXPECT_EQ(loss.name(), "OBBLoss");

    auto [boxGain, clsGain, dflGain] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(boxGain, 7.5f);
    EXPECT_FLOAT_EQ(clsGain, 0.5f);
    EXPECT_FLOAT_EQ(dflGain, 1.5f);
}

TEST_F(OBBLossTest, Constructor_CustomParameters) {
    auto stride = createStride();
    OBBLoss loss(10, 320, stride, 5.0f, 1.0f, 2.0f, 5);

    auto [boxGain, clsGain, dflGain] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(boxGain, 5.0f);
    EXPECT_FLOAT_EQ(clsGain, 1.0f);
    EXPECT_FLOAT_EQ(dflGain, 2.0f);
}

TEST_F(OBBLossTest, Constructor_SingleClass) {
    auto stride = createStride();
    OBBLoss loss(1, 640, stride);
    EXPECT_EQ(loss.name(), "OBBLoss");
}

TEST_F(OBBLossTest, Constructor_ManyClasses) {
    auto stride = createStride();
    OBBLoss loss(1000, 640, stride);
    EXPECT_EQ(loss.name(), "OBBLoss");
}

TEST_F(OBBLossTest, Constructor_SmallImageSize) {
    auto stride = torch::tensor({ 8.0f, 16.0f }).to(device_);
    OBBLoss loss(80, 128, stride);
    EXPECT_EQ(loss.name(), "OBBLoss");
}

TEST_F(OBBLossTest, Constructor_LargeImageSize) {
    auto stride = createStride();
    OBBLoss loss(80, 1280, stride);
    EXPECT_EQ(loss.name(), "OBBLoss");
}

// ============================================================================
// API Tests
// ============================================================================

TEST_F(OBBLossTest, Name_ReturnsCorrectName) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    EXPECT_EQ(loss.name(), "OBBLoss");
}

TEST_F(OBBLossTest, SetLossWeights_Valid) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);

    loss.setLossWeights(10.0f, 2.0f, 3.0f);

    auto [boxGain, clsGain, dflGain] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(boxGain, 10.0f);
    EXPECT_FLOAT_EQ(clsGain, 2.0f);
    EXPECT_FLOAT_EQ(dflGain, 3.0f);
}

TEST_F(OBBLossTest, SetLossWeights_Zero) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);

    loss.setLossWeights(0.0f, 0.0f, 0.0f);

    auto [boxGain, clsGain, dflGain] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(boxGain, 0.0f);
    EXPECT_FLOAT_EQ(clsGain, 0.0f);
    EXPECT_FLOAT_EQ(dflGain, 0.0f);
}

TEST_F(OBBLossTest, GetLossWeights_Default) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);

    auto [boxGain, clsGain, dflGain] = loss.getLossWeights();
    EXPECT_FLOAT_EQ(boxGain, 7.5f);
    EXPECT_FLOAT_EQ(clsGain, 0.5f);
    EXPECT_FLOAT_EQ(dflGain, 1.5f);
}

TEST_F(OBBLossTest, ToDevice_CPU) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    EXPECT_NO_THROW(loss.to(torch::kCPU));
}

#ifdef USE_CUDA
TEST_F(OBBLossTest, ToDevice_CUDA) {
    if (torch::cuda::is_available()) {
        auto stride = createStride();
        OBBLoss loss(80, 640, stride);
        EXPECT_NO_THROW(loss.to(torch::kCUDA));
    }
}
#endif

// ============================================================================
// Compute Tests - Single Tensor
// ============================================================================

TEST_F(OBBLossTest, Compute_SingleTensor_WithTargets) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createOBBTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("box") != resultMap.end());
    EXPECT_TRUE(resultMap.find("cls") != resultMap.end());
    EXPECT_TRUE(resultMap.find("dfl") != resultMap.end());
    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
}

TEST_F(OBBLossTest, Compute_SingleTensor_NoTargets) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
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

TEST_F(OBBLossTest, Compute_SingleTensor_BatchSize1) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(1, 80, 16, 640);
    auto target = createOBBTarget(3, 1, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(OBBLossTest, Compute_SingleTensor_BatchSize8) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(8, 80, 16, 640);
    auto target = createOBBTarget(20, 8, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(OBBLossTest, Compute_SingleTensor_LossComponentsValid) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createOBBTarget(10, 2, 80);

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

TEST_F(OBBLossTest, Compute_MultiScale_WithTargets) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto predictions = createMultiScalePredictions(2, 80, 16, 640);
    auto target = createOBBTarget(5, 2, 80);

    auto resultMap = loss.compute(predictions, target);

    EXPECT_TRUE(resultMap.find("box") != resultMap.end());
    EXPECT_TRUE(resultMap.find("cls") != resultMap.end());
    EXPECT_TRUE(resultMap.find("dfl") != resultMap.end());
    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
}

TEST_F(OBBLossTest, Compute_MultiScale_NoTargets) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto predictions = createMultiScalePredictions(2, 80, 16, 640);
    auto target = createEmptyTarget();

    auto resultMap = loss.compute(predictions, target);

    EXPECT_FLOAT_EQ(resultMap["box"].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(resultMap["dfl"].item<float>(), 0.0f);
}

TEST_F(OBBLossTest, Compute_MultiScale_SingleScale) {
    auto stride = torch::tensor({ 8.0f }).to(device_);
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 1 + 80;  // regMax*4 + angle + numClasses
    std::vector<torch::Tensor> predictions;
    predictions.push_back(torch::randn({ 2, channels, 80, 80 }, device_));

    auto target = createOBBTarget(3, 2, 80);

    auto resultMap = loss.compute(predictions, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
}

TEST_F(OBBLossTest, Compute_MultiScale_FourScales) {
    auto stride = torch::tensor({ 4.0f, 8.0f, 16.0f, 32.0f }).to(device_);
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 1 + 80;
    std::vector<torch::Tensor> predictions;
    predictions.push_back(torch::randn({ 2, channels, 160, 160 }, device_));  // stride 4
    predictions.push_back(torch::randn({ 2, channels, 80, 80 }, device_));    // stride 8
    predictions.push_back(torch::randn({ 2, channels, 40, 40 }, device_));    // stride 16
    predictions.push_back(torch::randn({ 2, channels, 20, 20 }, device_));    // stride 32

    auto target = createOBBTarget(8, 2, 80);

    auto resultMap = loss.compute(predictions, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

// ============================================================================
// Exception Tests
// ============================================================================

TEST_F(OBBLossTest, Compute_EmptyPredictions_Throws) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    std::vector<torch::Tensor> emptyPredictions;
    auto target = createOBBTarget(5, 2, 80);

    EXPECT_THROW(loss.compute(emptyPredictions, target), std::invalid_argument);
}

// ============================================================================
// Angle Variation Tests (OBB-specific)
// ============================================================================

TEST_F(OBBLossTest, Compute_ZeroAngle) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(1, 80, 16, 640);
    auto target = createOBBTargetWithAngle(1, 0.0f);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(OBBLossTest, Compute_Angle45Degrees) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(1, 80, 16, 640);
    auto target = createOBBTargetWithAngle(1, 0.785f);  // pi/4

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(OBBLossTest, Compute_Angle90Degrees) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(1, 80, 16, 640);
    auto target = createOBBTargetWithAngle(1, 1.5708f);  // pi/2

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(OBBLossTest, Compute_NegativeAngle) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(1, 80, 16, 640);
    auto target = createOBBTargetWithAngle(1, -0.785f);  // -pi/4

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(OBBLossTest, Compute_VariedAngles) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(4, 80, 16, 640);

    // Create targets with various angles
    DataExample example;
    auto angles = torch::tensor({ {0.0f}, {0.785f}, {1.5708f}, {-0.785f} }, device_);
    auto batchIndices = torch::tensor({ {0.0f}, {1.0f}, {2.0f}, {3.0f} }, device_);
    auto classIds = torch::zeros({ 4, 1 }, device_);
    auto cx = torch::ones({ 4, 1 }, device_) * 0.5f;
    auto cy = torch::ones({ 4, 1 }, device_) * 0.5f;
    auto w = torch::ones({ 4, 1 }, device_) * 0.2f;
    auto h = torch::ones({ 4, 1 }, device_) * 0.1f;

    example.targets = torch::cat({ batchIndices, classIds, cx, cy, w, h, angles }, 1);

    auto resultMap = loss.compute(pred, example);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

// ============================================================================
// Loss Weights Effect Tests
// ============================================================================

TEST_F(OBBLossTest, LossWeights_BoxGainEffect) {
    auto stride = createStride();

    OBBLoss loss1(80, 640, stride);
    loss1.to(device_);
    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createOBBTarget(5, 2, 80);
    auto result1 = loss1.compute(pred, target);

    OBBLoss loss2(80, 640, stride, 15.0f, 0.5f, 1.5f);
    loss2.to(device_);
    auto result2 = loss2.compute(pred, target);

    EXPECT_NE(result1["box"].item<float>(), result2["box"].item<float>());
}

TEST_F(OBBLossTest, LossWeights_ClsGainEffect) {
    auto stride = createStride();

    OBBLoss loss1(80, 640, stride);
    loss1.to(device_);
    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createOBBTarget(5, 2, 80);
    auto result1 = loss1.compute(pred, target);

    OBBLoss loss2(80, 640, stride, 7.5f, 1.0f, 1.5f);
    loss2.to(device_);
    auto result2 = loss2.compute(pred, target);

    EXPECT_NE(result1["cls"].item<float>(), result2["cls"].item<float>());
}

TEST_F(OBBLossTest, LossWeights_DflGainEffect) {
    auto stride = createStride();

    OBBLoss loss1(80, 640, stride);
    loss1.to(device_);
    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createOBBTarget(5, 2, 80);
    auto result1 = loss1.compute(pred, target);

    OBBLoss loss2(80, 640, stride, 7.5f, 0.5f, 3.0f);
    loss2.to(device_);
    auto result2 = loss2.compute(pred, target);

    EXPECT_NE(result1["dfl"].item<float>(), result2["dfl"].item<float>());
}

TEST_F(OBBLossTest, LossWeights_ZeroWeights) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride, 0.0f, 0.0f, 0.0f);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createOBBTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FLOAT_EQ(resultMap["box"].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(resultMap["cls"].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(resultMap["dfl"].item<float>(), 0.0f);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_F(OBBLossTest, NumericalStability_LargePredictions) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 1 + 80;
    int64_t totalAnchors = 80 * 80 + 40 * 40 + 20 * 20;
    auto pred = torch::randn({ 2, channels, totalAnchors }, device_) * 100.0f;
    auto target = createOBBTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(OBBLossTest, NumericalStability_SmallPredictions) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 1 + 80;
    int64_t totalAnchors = 80 * 80 + 40 * 40 + 20 * 20;
    auto pred = torch::randn({ 2, channels, totalAnchors }, device_) * 0.001f;
    auto target = createOBBTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(OBBLossTest, NumericalStability_ZeroPredictions) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 1 + 80;
    int64_t totalAnchors = 80 * 80 + 40 * 40 + 20 * 20;
    auto pred = torch::zeros({ 2, channels, totalAnchors }, device_);
    auto target = createOBBTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

// ============================================================================
// Different Class Counts Tests
// ============================================================================

TEST_F(OBBLossTest, Compute_SingleClass) {
    auto stride = createStride();
    OBBLoss loss(1, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 1 + 1;  // regMax*4 + angle + numClasses
    int64_t totalAnchors = 80 * 80 + 40 * 40 + 20 * 20;
    auto pred = torch::randn({ 2, channels, totalAnchors }, device_);
    auto target = createOBBTarget(3, 2, 1);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(OBBLossTest, Compute_FewClasses) {
    auto stride = createStride();
    OBBLoss loss(10, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 1 + 10;
    int64_t totalAnchors = 80 * 80 + 40 * 40 + 20 * 20;
    auto pred = torch::randn({ 2, channels, totalAnchors }, device_);
    auto target = createOBBTarget(5, 2, 10);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(OBBLossTest, Compute_ManyClasses) {
    auto stride = createStride();
    OBBLoss loss(200, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 1 + 200;
    int64_t totalAnchors = 80 * 80 + 40 * 40 + 20 * 20;
    auto pred = torch::randn({ 2, channels, totalAnchors }, device_);
    auto target = createOBBTarget(5, 2, 200);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

// ============================================================================
// Different Image Sizes Tests
// ============================================================================

TEST_F(OBBLossTest, Compute_SmallImage) {
    auto stride = torch::tensor({ 8.0f, 16.0f }).to(device_);
    OBBLoss loss(80, 128, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 1 + 80;
    int64_t totalAnchors = 16 * 16 + 8 * 8;
    auto pred = torch::randn({ 2, channels, totalAnchors }, device_);
    auto target = createOBBTarget(3, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
}

TEST_F(OBBLossTest, Compute_LargeImage) {
    auto stride = createStride();
    OBBLoss loss(80, 1280, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 1 + 80;
    int64_t totalAnchors = 160 * 160 + 80 * 80 + 40 * 40;
    auto pred = torch::randn({ 2, channels, totalAnchors }, device_);
    auto target = createOBBTarget(10, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
}

// ============================================================================
// OBB-Specific Edge Cases
// ============================================================================

TEST_F(OBBLossTest, EdgeCase_ThinObject) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(1, 80, 16, 640);

    // Create target with thin object (high aspect ratio)
    DataExample example;
    example.targets = torch::tensor({
        {0.0f, 0.0f, 0.5f, 0.5f, 0.3f, 0.02f, 0.0f}  // w >> h
        }).to(device_);

    auto resultMap = loss.compute(pred, example);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(OBBLossTest, EdgeCase_SquareObject) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(1, 80, 16, 640);

    // Create target with square object
    DataExample example;
    example.targets = torch::tensor({
        {0.0f, 0.0f, 0.5f, 0.5f, 0.2f, 0.2f, 0.785f}  // w == h, 45 degree rotation
        }).to(device_);

    auto resultMap = loss.compute(pred, example);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(OBBLossTest, EdgeCase_VerySmallBox) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(1, 80, 16, 640);

    DataExample example;
    example.targets = torch::tensor({
        {0.0f, 0.0f, 0.5f, 0.5f, 0.01f, 0.01f, 0.0f}
        }).to(device_);

    auto resultMap = loss.compute(pred, example);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(OBBLossTest, EdgeCase_BoxAtEdge) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(1, 80, 16, 640);

    DataExample example;
    example.targets = torch::tensor({
        {0.0f, 0.0f, 0.95f, 0.95f, 0.1f, 0.1f, 0.0f}
        }).to(device_);

    auto resultMap = loss.compute(pred, example);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

// ============================================================================
// Target Variation Tests
// ============================================================================

TEST_F(OBBLossTest, Compute_SingleTarget) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createOBBTarget(1, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(OBBLossTest, Compute_ManyTargets) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(4, 80, 16, 640);
    auto target = createOBBTarget(100, 4, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

// ============================================================================
// Gradient Tests
// ============================================================================

TEST_F(OBBLossTest, Gradient_BackwardPass) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 1 + 80;
    int64_t totalAnchors = 80 * 80 + 40 * 40 + 20 * 20;
    auto pred = torch::randn({ 2, channels, totalAnchors }, torch::TensorOptions().device(device_).requires_grad(true));
    auto target = createOBBTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);
    resultMap["total"].backward();

    EXPECT_TRUE(pred.grad().defined());
    EXPECT_FALSE(pred.grad().isnan().any().item<bool>());
    EXPECT_FALSE(pred.grad().isinf().any().item<bool>());
}

TEST_F(OBBLossTest, Gradient_MultiScale_BackwardPass) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    int64_t channels = 16 * 4 + 1 + 80;
    std::vector<torch::Tensor> predictions;
    predictions.push_back(torch::randn({ 2, channels, 80, 80 }, torch::TensorOptions().device(device_).requires_grad(true)));
    predictions.push_back(torch::randn({ 2, channels, 40, 40 }, torch::TensorOptions().device(device_).requires_grad(true)));
    predictions.push_back(torch::randn({ 2, channels, 20, 20 }, torch::TensorOptions().device(device_).requires_grad(true)));

    auto target = createOBBTarget(5, 2, 80);

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

TEST_F(OBBLossTest, ReturnMap_ContainsAllKeys) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createOBBTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    EXPECT_EQ(resultMap.size(), 4);
    EXPECT_TRUE(resultMap.count("box") == 1);
    EXPECT_TRUE(resultMap.count("cls") == 1);
    EXPECT_TRUE(resultMap.count("dfl") == 1);
    EXPECT_TRUE(resultMap.count("total") == 1);
}

TEST_F(OBBLossTest, ReturnMap_TensorShapes) {
    auto stride = createStride();
    OBBLoss loss(80, 640, stride);
    loss.to(device_);

    auto pred = createConcatenatedPrediction(2, 80, 16, 640);
    auto target = createOBBTarget(5, 2, 80);

    auto resultMap = loss.compute(pred, target);

    // All values should be scalar tensors
    EXPECT_EQ(resultMap["box"].dim(), 0);
    EXPECT_EQ(resultMap["cls"].dim(), 0);
    EXPECT_EQ(resultMap["dfl"].dim(), 0);
    EXPECT_EQ(resultMap["total"].dim(), 0);
}
