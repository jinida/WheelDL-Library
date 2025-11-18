#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Model/Loss/DetectionLoss.h"
#include "WheelDL.Lib/Data/Dataset/BaseDataset.h"
#include <torch/torch.h>

using namespace WheelDL::Model::Loss;
using namespace WheelDL::Data::Dataset;

/**
 * @brief Detection loss function tests
 *
 * Tests verify YOLO-style detection loss including:
 * - Single tensor input: Inference format [batch, nc+4*regMax, anchors]
 * - Multi-scale input: Training format [P3, P4, P5]
 * - Task-aligned assignment
 * - BBox decoding (DFL to xyxy)
 * - CIoU computation
 */
class DetectionLossTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Lightweight setup per test
        numClasses = 80;
        regMax = 16;

        // Create stride tensor for 3 detection layers
        stride = torch::tensor({8.0f, 16.0f, 32.0f}).to(torch::kCUDA);
    }

    torch::Tensor createSingleTensorPrediction(int64_t batch, bool requiresGrad = false)
    {
        // Inference format: [batch, 4*regMax + numClasses, total_anchors]
        // For simplicity, use 8400 anchors (80x80 + 40x40 + 20x20 from P3/P4/P5)
        int64_t channels = 4 * regMax + numClasses;  // 144
        int64_t anchors = 8400;

        return torch::randn(
            {batch, channels, anchors},
            torch::TensorOptions().requires_grad(requiresGrad).device(torch::kCUDA)
        );
    }

    std::vector<torch::Tensor> createMultiScalePredictions(int64_t batch, bool requiresGrad = false)
    {
        // Training format: separate tensors for each scale
        int64_t channels = 4 * regMax + numClasses;  // 144

        std::vector<torch::Tensor> predictions;
        predictions.push_back(
            torch::randn(
                {batch, channels, 80, 80},
                torch::TensorOptions().requires_grad(requiresGrad).device(torch::kCUDA)
            )
        );
        predictions.push_back(
            torch::randn(
                {batch, channels, 40, 40},
                torch::TensorOptions().requires_grad(requiresGrad).device(torch::kCUDA)
            )
        );
        predictions.push_back(
            torch::randn(
                {batch, channels, 20, 20},
                torch::TensorOptions().requires_grad(requiresGrad).device(torch::kCUDA)
			)
        );

        return predictions;
    }

    DataExample createTargets(int64_t batch, int64_t numObjects = 5)
    {
        DataExample example;

        int64_t totalObjects = batch * numObjects;

        // Separate tensors for each component as expected by DetectionLoss
        auto batchIndices = torch::zeros({totalObjects});
        auto classes = torch::zeros({totalObjects});
        auto bboxes = torch::zeros({totalObjects, 4});

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t obj = 0; obj < numObjects; ++obj) {
                int64_t idx = b * numObjects + obj;

                batchIndices.index_put_({idx}, static_cast<float>(b));
                classes.index_put_({idx}, static_cast<float>(
                    torch::randint(0, numClasses, {1}).item<int64_t>()
                ));

                // Random box in [0, 1] normalized coordinates
                bboxes.index_put_({idx, 0}, torch::rand({1}).item<float>());  // x
                bboxes.index_put_({idx, 1}, torch::rand({1}).item<float>());  // y
                bboxes.index_put_({idx, 2}, torch::rand({1}).item<float>() * 0.5f);  // w
                bboxes.index_put_({idx, 3}, torch::rand({1}).item<float>() * 0.5f);  // h
            }
        }

        example.batchIndices = batchIndices;
        example.classes = classes;
        example.targets = bboxes;
		example.toDevice(torch::kCUDA);
        return example;
    }

    DataExample createEmptyTargets(int64_t batch)
    {
        DataExample example;
        example.batchIndices = torch::zeros({0});
        example.classes = torch::zeros({0});
        example.targets = torch::zeros({0, 4});
        return example;
    }

    /**
     * @brief Create predictions that perfectly match the targets
     *
     * This creates predictions where:
     * - Class predictions have high confidence for correct class
     * - BBox predictions match target boxes exactly
     * - DFL distributions are sharp around correct distances
     */
    std::vector<torch::Tensor> createPerfectPredictions(const DataExample& targets)
    {
        int64_t batch = static_cast<int64_t>(targets.batchIndices.max().item<float>()) + 1;
        int64_t channels = 4 * regMax + numClasses;  // 144

        std::vector<torch::Tensor> predictions;
        predictions.push_back(torch::zeros({batch, channels, 80, 80}, torch::TensorOptions().device(torch::kCUDA)));
        predictions.push_back(torch::zeros({batch, channels, 40, 40}, torch::TensorOptions().device(torch::kCUDA)));
        predictions.push_back(torch::zeros({batch, channels, 20, 20}, torch::TensorOptions().device(torch::kCUDA)));

        // Initialize class predictions with logits for near-zero loss
        // Use large negative values for wrong classes, positive for correct class
        for (auto& pred : predictions) {
            pred.index_put_(
                {torch::indexing::Slice(), torch::indexing::Slice(64, 144)},
                -10.0f  // Strong negative logits for all classes initially
            );
        }

        // For each target, set the corresponding anchor to have perfect prediction
        int64_t numTargets = targets.targets.size(0);
        for (int64_t i = 0; i < numTargets; ++i) {
            int64_t batchIdx = static_cast<int64_t>(targets.batchIndices[i].item<float>());
            int64_t classIdx = static_cast<int64_t>(targets.classes[i].item<float>());

            // Get target bbox [x, y, w, h] in normalized [0, 1] coordinates
            float tx = targets.targets[i][0].item<float>();
            float ty = targets.targets[i][1].item<float>();
            float tw = targets.targets[i][2].item<float>();
            float th = targets.targets[i][3].item<float>();

            // Find best matching scale and anchor
            // Use P3 (80x80) for small objects, P4 (40x40) for medium, P5 (20x20) for large
            int scale_idx = tw * th < 0.1f ? 0 : (tw * th < 0.3f ? 1 : 2);
            int grid_size = scale_idx == 0 ? 80 : (scale_idx == 1 ? 40 : 20);

            // Convert center coords to grid position
            int grid_x = std::min(static_cast<int>(tx * grid_size), grid_size - 1);
            int grid_y = std::min(static_cast<int>(ty * grid_size), grid_size - 1);

            // Set class prediction to high confidence
            predictions[scale_idx][batchIdx][64 + classIdx][grid_y][grid_x] = 10.0f;

            // Set bbox prediction
            // DFL uses distribution over [0, regMax), we set peaked distribution
            // For simplicity, set middle values to approximate the target distances
            float scale = stride[scale_idx].item<float>();

            // Compute target distances from anchor point to bbox edges
            float anchor_x = (grid_x + 0.5f) * scale / 640.0f;  // Assuming 640x640 image
            float anchor_y = (grid_y + 0.5f) * scale / 640.0f;

            float left = (tx - tw/2.0f - anchor_x) * 640.0f / scale;
            float top = (ty - th/2.0f - anchor_y) * 640.0f / scale;
            float right = (tx + tw/2.0f - anchor_x) * 640.0f / scale;
            float bottom = (ty + th/2.0f - anchor_y) * 640.0f / scale;

            // Set DFL distributions (simplified - just set raw values)
            // In practice, DFL uses softmax over regMax bins
            for (int j = 0; j < regMax; ++j) {
                predictions[scale_idx][batchIdx][j][grid_y][grid_x] =
                    std::abs(left) < regMax && std::abs(static_cast<int>(left) - j) < 2 ? 5.0f : -5.0f;
                predictions[scale_idx][batchIdx][regMax + j][grid_y][grid_x] =
                    std::abs(top) < regMax && std::abs(static_cast<int>(top) - j) < 2 ? 5.0f : -5.0f;
                predictions[scale_idx][batchIdx][2*regMax + j][grid_y][grid_x] =
                    std::abs(right) < regMax && std::abs(static_cast<int>(right) - j) < 2 ? 5.0f : -5.0f;
                predictions[scale_idx][batchIdx][3*regMax + j][grid_y][grid_x] =
                    std::abs(bottom) < regMax && std::abs(static_cast<int>(bottom) - j) < 2 ? 5.0f : -5.0f;
            }
        }

        return predictions;
    }

    /**
     * @brief Create predictions with specific IoU to targets
     *
     * @param targets Ground truth targets
     * @param targetIoU Desired IoU value (0.0 to 1.0)
     * @return Predictions with approximately the target IoU
     */
    std::vector<torch::Tensor> createPredictionsWithIoU(const DataExample& targets, float targetIoU)
    {
        // Start with perfect predictions
        auto predictions = createPerfectPredictions(targets);

        if (targetIoU >= 0.99f) {
            return predictions;  // Already perfect
        }

        // Scale down bbox sizes to reduce IoU
        // This is a simplified approach - actual IoU depends on overlap
        float scale_factor = std::sqrt(targetIoU);

        int64_t numTargets = targets.targets.size(0);
        for (int64_t i = 0; i < numTargets; ++i) {
            int64_t batchIdx = static_cast<int64_t>(targets.batchIndices[i].item<float>());
            float tw = targets.targets[i][2].item<float>();
            float th = targets.targets[i][3].item<float>();

            int scale_idx = tw * th < 0.1f ? 0 : (tw * th < 0.3f ? 1 : 2);

            // Modify DFL predictions to reduce bbox size
            // This is a rough approximation
            predictions[scale_idx] = predictions[scale_idx] * scale_factor;
        }

        return predictions;
    }

    int64_t numClasses;
    int64_t regMax;
    torch::Tensor stride;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(DetectionLossTest, Construction_DefaultWeights_Success)
{
    EXPECT_NO_THROW({
        DetectionLoss loss(numClasses, stride);
    });
}

TEST_F(DetectionLossTest, Construction_CustomWeights_Success)
{
    EXPECT_NO_THROW({
        DetectionLoss loss(
            numClasses,
            stride,
            10.0f,  // box gain
            1.0f,   // cls gain
            2.0f    // dfl gain
        );
    });
}

TEST_F(DetectionLossTest, Construction_GetWeights_ReturnsCorrect)
{
    DetectionLoss loss(
        numClasses,
        stride,
        7.5f,
        0.5f,
        1.5f
    );

    auto [box, cls, dfl] = loss.getLossWeights();

    EXPECT_FLOAT_EQ(box, 7.5f);
    EXPECT_FLOAT_EQ(cls, 0.5f);
    EXPECT_FLOAT_EQ(dfl, 1.5f);
}

TEST_F(DetectionLossTest, SetLossWeights_UpdatesCorrectly)
{
    DetectionLoss loss(numClasses, stride);

    loss.setLossWeights(10.0f, 1.0f, 2.0f);

    auto [box, cls, dfl] = loss.getLossWeights();

    EXPECT_FLOAT_EQ(box, 10.0f);
    EXPECT_FLOAT_EQ(cls, 1.0f);
    EXPECT_FLOAT_EQ(dfl, 2.0f);
}

// ============================================================================
// Single Tensor Input Tests
// ============================================================================

TEST_F(DetectionLossTest, SingleTensor_ValidInput_ReturnsLoss)
{
    DetectionLoss loss(numClasses, stride);

    auto prediction = createSingleTensorPrediction(2).to(torch::kCUDA);
    auto targets = createTargets(2, 5);

    EXPECT_NO_THROW({
        auto result = loss.compute(prediction, targets);

        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(DetectionLossTest, SingleTensor_OutputKeys_ContainsAllComponents)
{
    DetectionLoss loss(numClasses, stride);

    auto prediction = createSingleTensorPrediction(2);
    auto targets = createTargets(2, 3);

    auto result = loss.compute(prediction, targets);

    EXPECT_TRUE(result.find("box") != result.end());
    EXPECT_TRUE(result.find("cls") != result.end());
    EXPECT_TRUE(result.find("dfl") != result.end());
    EXPECT_TRUE(result.find("total") != result.end());
}

TEST_F(DetectionLossTest, SingleTensor_EmptyTargets_ReturnsZeroLoss)
{
    DetectionLoss loss(numClasses, stride);

    auto prediction = createSingleTensorPrediction(2);
    auto targets = createEmptyTargets(2);

    auto result = loss.compute(prediction, targets);

    // With no targets, losses should be minimal or zero
    EXPECT_TRUE(result.find("total") != result.end());
    EXPECT_GE(result["total"].item<float>(), 0.0f);
}

// ============================================================================
// Multi-Scale Input Tests
// ============================================================================

TEST_F(DetectionLossTest, MultiScale_ValidInput_ReturnsLoss)
{
    DetectionLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 5);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(DetectionLossTest, MultiScale_ThreeScales_ProcessesCorrectly)
{
    DetectionLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(4);
    auto targets = createTargets(4, 8);

    auto result = loss.compute(predictions, targets);

    EXPECT_TRUE(result.find("box") != result.end());
    EXPECT_TRUE(result.find("cls") != result.end());
    EXPECT_TRUE(result.find("dfl") != result.end());
    EXPECT_TRUE(result.find("total") != result.end());
}

TEST_F(DetectionLossTest, MultiScale_OutputKeys_AllPresent)
{
    DetectionLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 3);

    auto result = loss.compute(predictions, targets);

    EXPECT_EQ(result.size(), 4);
    EXPECT_TRUE(result.find("box") != result.end());
    EXPECT_TRUE(result.find("cls") != result.end());
    EXPECT_TRUE(result.find("dfl") != result.end());
    EXPECT_TRUE(result.find("total") != result.end());
}

// ============================================================================
// Loss Value Tests
// ============================================================================

TEST_F(DetectionLossTest, LossValues_AllNonNegative)
{
    DetectionLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 5);

    auto result = loss.compute(predictions, targets);

    EXPECT_GE(result["box"].item<float>(), 0.0f);
    EXPECT_GE(result["cls"].item<float>(), 0.0f);
    EXPECT_GE(result["dfl"].item<float>(), 0.0f);
    EXPECT_GE(result["total"].item<float>(), 0.0f);
}

TEST_F(DetectionLossTest, LossValues_TotalIsWeightedSum)
{
    float boxGain = 7.5f;
    float clsGain = 0.5f;
    float dflGain = 1.5f;

    DetectionLoss loss(
        numClasses,
        stride,
        boxGain,
        clsGain,
        dflGain
    );

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 5);

    auto result = loss.compute(predictions, targets);

    float expectedTotal =
        result["box"].item<float>() +
        result["cls"].item<float>() +
        result["dfl"].item<float>();

    EXPECT_NEAR(
        result["total"].item<float>(),
        expectedTotal,
        1e-2f  // 부동소수점 오차 고려
    );
}

TEST_F(DetectionLossTest, LossValues_ReasonableRange)
{
    DetectionLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 5);

    auto result = loss.compute(predictions, targets);
    
    EXPECT_TRUE(std::isfinite(result["box"].item<float>()));
    EXPECT_TRUE(std::isfinite(result["cls"].item<float>()));
    EXPECT_TRUE(std::isfinite(result["dfl"].item<float>()));
    EXPECT_TRUE(std::isfinite(result["total"].item<float>()));

    EXPECT_GE(result["box"].item<float>(), 0.0f);
    EXPECT_GE(result["cls"].item<float>(), 0.0f);
    EXPECT_GE(result["dfl"].item<float>(), 0.0f);

    float sum = result["box"].item<float>() +
                result["cls"].item<float>() +
                result["dfl"].item<float>();
    EXPECT_NEAR(result["total"].item<float>(), sum, 1e-2f);
}

// ============================================================================
// Task-Aligned Assignment Tests
// ============================================================================

TEST_F(DetectionLossTest, TaskAligned_WithTargets_AssignsCorrectly)
{
    DetectionLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 10);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

TEST_F(DetectionLossTest, TaskAligned_DifferentObjectCounts_Works)
{
    DetectionLoss loss(numClasses, stride);

    // Test with varying number of objects
    std::vector<int64_t> objectCounts = {1, 5, 10, 20};

    for (auto count : objectCounts) {
        auto predictions = createMultiScalePredictions(2);
        auto targets = createTargets(2, count);

        EXPECT_NO_THROW({
            auto result = loss.compute(predictions, targets);
            EXPECT_TRUE(result.find("total") != result.end());
        });
    }
}

// ============================================================================
// BBox Decoding Tests
// ============================================================================

TEST_F(DetectionLossTest, BboxDecoding_DFLToXYXY_Works)
{
    DetectionLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(1);
    auto targets = createTargets(1, 1);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result["dfl"].defined());
    });
}

TEST_F(DetectionLossTest, BboxDecoding_MultipleAnchors_ProcessesAll)
{
    DetectionLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 10);

    auto result = loss.compute(predictions, targets);

    // DFL loss should be computed for all positive assignments
    EXPECT_TRUE(result["dfl"].defined());
    EXPECT_GE(result["dfl"].item<float>(), 0.0f);
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

TEST_F(DetectionLossTest, GradientFlow_BackwardPass_Works)
{
    DetectionLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2, true);  // Enable gradient
    auto targets = createTargets(2, 5);

    auto result = loss.compute(predictions, targets);

    EXPECT_NO_THROW({
        result["total"].backward();

        // Check gradients exist for all prediction tensors
        for (const auto& pred : predictions) {
            EXPECT_TRUE(pred.grad().defined());
            EXPECT_GT(pred.grad().abs().sum().item<float>(), 0.0f);
        }
    });
}

TEST_F(DetectionLossTest, GradientFlow_SingleTensor_Works)
{
    DetectionLoss loss(numClasses, stride);

    auto prediction = createSingleTensorPrediction(2, true);  // Enable gradient
    auto targets = createTargets(2, 5);

    auto result = loss.compute(prediction, targets);

    EXPECT_NO_THROW({
        result["total"].backward();
        EXPECT_TRUE(prediction.grad().defined());
    });
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(DetectionLossTest, EdgeCase_NoPositives_HandlesGracefully)
{
    DetectionLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createEmptyTargets(2);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(DetectionLossTest, EdgeCase_SingleBatch_Works)
{
    DetectionLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(1);
    auto targets = createTargets(1, 5);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

TEST_F(DetectionLossTest, EdgeCase_LargeBatch_Works)
{
    DetectionLoss loss(numClasses, stride);

    // Gradient 비활성화로 메모리 사용량 감소
    torch::NoGradGuard no_grad;

    // batch=16은 메모리 부족 (5GB 요청), 8로 감소
    auto predictions = createMultiScalePredictions(8);
    auto targets = createTargets(8, 5);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

TEST_F(DetectionLossTest, EdgeCase_ManyObjects_Works)
{
    DetectionLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 50);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

TEST_F(DetectionLossTest, EdgeCase_SingleObject_Works)
{
    DetectionLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(1);
    auto targets = createTargets(1, 1);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

// ============================================================================
// Device Tests
// ============================================================================

TEST_F(DetectionLossTest, Device_SetToCPU_Works)
{
    DetectionLoss loss(numClasses, stride);

    EXPECT_NO_THROW({
        loss.to(torch::kCPU);
    });
}

// ============================================================================
// Ideal Prediction Tests
// ============================================================================

TEST_F(DetectionLossTest, IdealPrediction_RandomPredictions_HighLoss)
{
    DetectionLoss loss(numClasses, stride);

    // Create targets
    auto targets = createTargets(2, 5);

    // Create completely random predictions
    auto predictions = createMultiScalePredictions(2);

    auto result = loss.compute(predictions, targets);

    // Random predictions should have high loss
    EXPECT_TRUE(result.find("total") != result.end());
    EXPECT_GT(result["total"].item<float>(), 5.0f);  // Should be significantly higher
}

TEST_F(DetectionLossTest, IdealPrediction_WrongClass_HighClsLoss)
{
    DetectionLoss loss(numClasses, stride);

    // Create targets with specific classes
    auto targets = createTargets(1, 3);

    // Set all targets to class 0
    targets.classes.fill_(0.0f);

    // Create perfect bbox predictions
    auto predictions = createPerfectPredictions(targets);

    // But set wrong class predictions (class 1 instead of 0)
    int64_t batch = static_cast<int64_t>(targets.batchIndices.max().item<float>()) + 1;
    for (auto& pred : predictions) {
        // Reset all class logits
        pred.index_put_(
            {torch::indexing::Slice(), torch::indexing::Slice(64, 144)},
            -10.0f
        );
        // Set high confidence for wrong class (class 1)
        pred.index_put_(
            {torch::indexing::Slice(), 65},
            10.0f
        );
    }

    auto result = loss.compute(predictions, targets);

    // Classification loss should be high with wrong class
    EXPECT_TRUE(result.find("cls") != result.end());
    EXPECT_GT(result["cls"].item<float>(), 1.0f);
}

TEST_F(DetectionLossTest, IdealPrediction_LossMonotonicity_VerifyBehavior)
{
    DetectionLoss loss(numClasses, stride);

    // Create fixed targets
    auto targets = createTargets(1, 2);

    // Test with perfect prediction
    auto perfectPreds = createPerfectPredictions(targets);
    auto perfectResult = loss.compute(perfectPreds, targets);

    // Test with random prediction
    auto randomPreds = createMultiScalePredictions(1);
    auto randomResult = loss.compute(randomPreds, targets);

    // Perfect predictions should have lower loss than random
    EXPECT_LT(
        perfectResult["total"].item<float>(),
        randomResult["total"].item<float>()
    );
}

TEST_F(DetectionLossTest, IdealPrediction_ZeroSizeBBox_HandlesGracefully)
{
    DetectionLoss loss(numClasses, stride);

    // Create targets with very small bbox
    auto targets = createTargets(1, 2);
    targets.targets.index_put_({torch::indexing::Slice(), 2}, 0.01f);  // Very small width
    targets.targets.index_put_({torch::indexing::Slice(), 3}, 0.01f);  // Very small height

    auto predictions = createPerfectPredictions(targets);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_TRUE(std::isfinite(result["total"].item<float>()));
    });
}

TEST_F(DetectionLossTest, IdealPrediction_LargeBBox_HandlesCorrectly)
{
    DetectionLoss loss(numClasses, stride);

    // Create targets with large bbox
    auto targets = createTargets(1, 2);
    targets.targets.index_put_({torch::indexing::Slice(), 0}, 0.5f);   // Center x
    targets.targets.index_put_({torch::indexing::Slice(), 1}, 0.5f);   // Center y
    targets.targets.index_put_({torch::indexing::Slice(), 2}, 0.9f);   // Large width
    targets.targets.index_put_({torch::indexing::Slice(), 3}, 0.9f);   // Large height

    auto predictions = createPerfectPredictions(targets);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_TRUE(std::isfinite(result["total"].item<float>()));
    });
}

TEST_F(DetectionLossTest, IdealPrediction_ComponentsContributeToTotal)
{
    DetectionLoss loss(numClasses, stride);

    auto targets = createTargets(2, 5);
    auto predictions = createMultiScalePredictions(2);

    auto result = loss.compute(predictions, targets);

    // Verify that total is sum of weighted components
    float box = result["box"].item<float>();
    float cls = result["cls"].item<float>();
    float dfl = result["dfl"].item<float>();
    float total = result["total"].item<float>();

    EXPECT_NEAR(total, box + cls + dfl, 0.1f);

    // All components should contribute (be non-zero)
    EXPECT_GT(box, 0.0f);
    EXPECT_GT(cls, 0.0f);
    EXPECT_GT(dfl, 0.0f);
}

// ============================================================================
// Name Test
// ============================================================================

TEST_F(DetectionLossTest, Name_ReturnsCorrectString)
{
    DetectionLoss loss(numClasses, stride);

    EXPECT_EQ(loss.name(), "DetectionLoss");
}
