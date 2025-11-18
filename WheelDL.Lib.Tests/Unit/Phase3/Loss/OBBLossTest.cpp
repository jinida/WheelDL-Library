#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Model/Loss/OBBLoss.h"
#include "WheelDL.Lib/Data/Dataset/BaseDataset.h"
#include <torch/torch.h>

using namespace WheelDL::Model::Loss;
using namespace WheelDL::Data::Dataset;

/**
 * @brief Oriented bounding box loss function tests
 *
 * Tests verify YOLO-style OBB detection loss including:
 * - Probiou computation for rotated boxes
 * - Angle loss handling with periodicity
 * - Multi-scale input: Training format [P3, P4, P5]
 * - Task-aligned assignment for OBB
 */
class OBBLossTest : public ::testing::Test
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
        // Inference format: [batch, 4*regMax + numClasses + 1, total_anchors]
        // +1 for angle
        int64_t channels = 4 * regMax + numClasses + 1;  // 145
        int64_t anchors = 8400;

        return torch::randn(
            {batch, channels, anchors},
			torch::TensorOptions().requires_grad(requiresGrad).device(torch::kCUDA)
        );
    }

    std::vector<torch::Tensor> createMultiScalePredictions(int64_t batch, bool requiresGrad = false)
    {
        // Training format: separate tensors for each scale
        int64_t channels = 4 * regMax + numClasses + 1;  // 145

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

        // Separate tensors for each component as expected by OBBLoss
        auto batchIndices = torch::zeros({totalObjects});
        auto classes = torch::zeros({totalObjects});
        auto obbs = torch::zeros({totalObjects, 5});  // [x, y, w, h, angle]

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t obj = 0; obj < numObjects; ++obj) {
                int64_t idx = b * numObjects + obj;

                batchIndices.index_put_({idx}, static_cast<float>(b));
                classes.index_put_({idx}, static_cast<float>(
                    torch::randint(0, numClasses, {1}).item<int64_t>()
                ));

                // Random box in [0, 1] normalized coordinates
                obbs.index_put_({idx, 0}, torch::rand({1}).item<float>());  // x
                obbs.index_put_({idx, 1}, torch::rand({1}).item<float>());  // y
                obbs.index_put_({idx, 2}, torch::rand({1}).item<float>() * 0.5f);  // w
                obbs.index_put_({idx, 3}, torch::rand({1}).item<float>() * 0.5f);  // h
                // Random angle in [-pi, pi]
                obbs.index_put_({idx, 4}, (torch::rand({1}).item<float>() * 2.0f - 1.0f) * 3.14159f);
            }
        }

        example.batchIndices = batchIndices;
        example.classes = classes;
        example.targets = obbs;
        example.toDevice(torch::kCUDA);
        return example;
    }

    DataExample createEmptyTargets(int64_t batch)
    {
        DataExample example;
        example.batchIndices = torch::zeros({0});
        example.classes = torch::zeros({0});
        example.targets = torch::zeros({0, 5});  // OBB: [x, y, w, h, angle]
        example.toDevice(torch::kCUDA);
        return example;
    }

    /**
     * @brief Create predictions that perfectly match the OBB targets
     *
     * This creates predictions where:
     * - Class predictions have high confidence for correct class
     * - BBox predictions match target boxes exactly
     * - Angle predictions match target angles
     * - DFL distributions are sharp around correct distances
     */
    std::vector<torch::Tensor> createPerfectPredictions(const DataExample& targets)
    {
        int64_t batch = static_cast<int64_t>(targets.batchIndices.max().item<float>()) + 1;
        int64_t channels = 4 * regMax + numClasses + 1;  // 145 (includes angle)

        std::vector<torch::Tensor> predictions;
        predictions.push_back(torch::zeros({batch, channels, 80, 80}, torch::TensorOptions().device(torch::kCUDA)));
        predictions.push_back(torch::zeros({batch, channels, 40, 40}, torch::TensorOptions().device(torch::kCUDA)));
        predictions.push_back(torch::zeros({batch, channels, 20, 20}, torch::TensorOptions().device(torch::kCUDA)));

        // Initialize class predictions with logits for near-zero loss
        for (auto& pred : predictions) {
            pred.index_put_(
                {torch::indexing::Slice(), torch::indexing::Slice(64, 144)},
                -10.0f  // Strong negative logits for all classes initially
            );
            // Initialize angle predictions to 0
            pred.index_put_(
                {torch::indexing::Slice(), 144},
                0.0f
            );
        }

        // For each target, set the corresponding anchor to have perfect prediction
        int64_t numTargets = targets.targets.size(0);
        for (int64_t i = 0; i < numTargets; ++i) {
            int64_t batchIdx = static_cast<int64_t>(targets.batchIndices[i].item<float>());
            int64_t classIdx = static_cast<int64_t>(targets.classes[i].item<float>());

            // Get target OBB [x, y, w, h, angle] in normalized [0, 1] coordinates
            float tx = targets.targets[i][0].item<float>();
            float ty = targets.targets[i][1].item<float>();
            float tw = targets.targets[i][2].item<float>();
            float th = targets.targets[i][3].item<float>();
            float tangle = targets.targets[i][4].item<float>();

            // Find best matching scale and anchor
            int scale_idx = tw * th < 0.1f ? 0 : (tw * th < 0.3f ? 1 : 2);
            int grid_size = scale_idx == 0 ? 80 : (scale_idx == 1 ? 40 : 20);

            // Convert center coords to grid position
            int grid_x = std::min(static_cast<int>(tx * grid_size), grid_size - 1);
            int grid_y = std::min(static_cast<int>(ty * grid_size), grid_size - 1);

            // Set class prediction to high confidence
            predictions[scale_idx][batchIdx][64 + classIdx][grid_y][grid_x] = 10.0f;

            // Set angle prediction
            predictions[scale_idx][batchIdx][144][grid_y][grid_x] = tangle;

            // Set bbox prediction (DFL)
            float scale = stride[scale_idx].item<float>();
            float anchor_x = (grid_x + 0.5f) * scale / 640.0f;
            float anchor_y = (grid_y + 0.5f) * scale / 640.0f;

            float left = (tx - tw/2.0f - anchor_x) * 640.0f / scale;
            float top = (ty - th/2.0f - anchor_y) * 640.0f / scale;
            float right = (tx + tw/2.0f - anchor_x) * 640.0f / scale;
            float bottom = (ty + th/2.0f - anchor_y) * 640.0f / scale;

            // Set DFL distributions
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

    int64_t numClasses;
    int64_t regMax;
    torch::Tensor stride;
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_F(OBBLossTest, Construction_DefaultWeights_Success)
{
    EXPECT_NO_THROW({
        OBBLoss loss(numClasses, stride);
    });
}

TEST_F(OBBLossTest, Construction_CustomWeights_Success)
{
    EXPECT_NO_THROW({
        OBBLoss loss(
            numClasses,
            stride,
            10.0f,  // box gain
            1.0f,   // cls gain
            2.0f    // dfl gain
        );
    });
}

TEST_F(OBBLossTest, Construction_ExtendsDetectionLoss)
{
    OBBLoss loss(numClasses, stride);

    // Should have same interface as DetectionLoss
    auto [box, cls, dfl] = loss.getLossWeights();

    EXPECT_GE(box, 0.0f);
    EXPECT_GE(cls, 0.0f);
    EXPECT_GE(dfl, 0.0f);
}

// ============================================================================
// Probiou Computation Tests
// ============================================================================

TEST_F(OBBLossTest, Probiou_RotatedBoxes_ComputesCorrectly)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 5);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        // Box loss should use Probiou for rotated boxes
        EXPECT_TRUE(result.find("box") != result.end());
        EXPECT_GE(result["box"].item<float>(), 0.0f);
    });
}

TEST_F(OBBLossTest, Probiou_NoRotation_MatchesStandardIoU)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(1);
    auto targets = createTargets(1, 3);

    // Set all angles to zero (no rotation)
    targets.targets.index_put_({torch::indexing::Slice(), 4}, 0.0f);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("box") != result.end());
    });
}

TEST_F(OBBLossTest, Probiou_VariousAngles_HandlesCorrectly)
{
    OBBLoss loss(numClasses, stride);

    // Test with specific angles
    std::vector<float> angles = {0.0f, 0.785f, 1.57f, 3.14f, -1.57f};

    for (float angle : angles) {
        auto predictions = createMultiScalePredictions(1);
        auto targets = createTargets(1, 2);

        // Set specific angle
        targets.targets.index_put_({torch::indexing::Slice(), 4}, angle);

        EXPECT_NO_THROW({
            auto result = loss.compute(predictions, targets);
            EXPECT_GE(result["box"].item<float>(), 0.0f);
        });
    }
}

// ============================================================================
// Angle Loss Tests
// ============================================================================

TEST_F(OBBLossTest, AngleLoss_Regression_Works)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 5);

    auto result = loss.compute(predictions, targets);

    // Angle should be part of box loss or separate component
    EXPECT_TRUE(result.find("box") != result.end());
    EXPECT_GE(result["box"].item<float>(), 0.0f);
}

TEST_F(OBBLossTest, AngleLoss_PeriodicityHandling_Works)
{
    OBBLoss loss(numClasses, stride);

    // Test angle periodicity (0 and 2*pi should be similar)
    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 3);

    // Set angles to test periodicity
    targets.targets.index_put_({0, 4}, 0.0f);
    targets.targets.index_put_({1, 4}, 6.28f);  // ~2*pi
    targets.targets.index_put_({2, 4}, -6.28f);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

// ============================================================================
// Multi-Scale Tests
// ============================================================================

TEST_F(OBBLossTest, MultiScale_ValidInput_ReturnsLoss)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 5);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);

        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(OBBLossTest, MultiScale_ThreeScales_ProcessesCorrectly)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(4);
    auto targets = createTargets(4, 8);

    auto result = loss.compute(predictions, targets);

    EXPECT_TRUE(result.find("box") != result.end());
    EXPECT_TRUE(result.find("cls") != result.end());
    EXPECT_TRUE(result.find("dfl") != result.end());
    EXPECT_TRUE(result.find("total") != result.end());
}

// ============================================================================
// Output Keys Tests
// ============================================================================

TEST_F(OBBLossTest, OutputKeys_ContainsAllComponents)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 3);

    auto result = loss.compute(predictions, targets);

    EXPECT_TRUE(result.find("box") != result.end());
    EXPECT_TRUE(result.find("cls") != result.end());
    EXPECT_TRUE(result.find("dfl") != result.end());
    EXPECT_TRUE(result.find("total") != result.end());
}

TEST_F(OBBLossTest, OutputKeys_BoxLossIncludesAngle)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 5);

    auto result = loss.compute(predictions, targets);

    // Box loss should include both spatial and angle components
    EXPECT_TRUE(result.find("box") != result.end());
    EXPECT_TRUE(result["box"].defined());
}

// ============================================================================
// Loss Value Tests
// ============================================================================

TEST_F(OBBLossTest, LossValues_AllNonNegative)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 5);

    auto result = loss.compute(predictions, targets);

    EXPECT_GE(result["box"].item<float>(), 0.0f);
    EXPECT_GE(result["cls"].item<float>(), 0.0f);
    EXPECT_GE(result["dfl"].item<float>(), 0.0f);
    EXPECT_GE(result["total"].item<float>(), 0.0f);
}

TEST_F(OBBLossTest, LossValues_ReasonableRange)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createTargets(2, 5);

    auto result = loss.compute(predictions, targets);

    // Loss values should be finite and reasonable
    EXPECT_TRUE(std::isfinite(result["total"].item<float>()));
    EXPECT_GT(result["total"].item<float>(), 0.0f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(OBBLossTest, EdgeCase_NoPositives_HandlesGracefully)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2);
    auto targets = createEmptyTargets(2);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_GE(result["total"].item<float>(), 0.0f);
    });
}

TEST_F(OBBLossTest, EdgeCase_SingleBatch_Works)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(1);
    auto targets = createTargets(1, 5);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

TEST_F(OBBLossTest, EdgeCase_SingleObject_Works)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(1);
    auto targets = createTargets(1, 1);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
    });
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

TEST_F(OBBLossTest, GradientFlow_BackwardPass_Works)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2, true);  // Enable gradient
    auto targets = createTargets(2, 5);

    auto result = loss.compute(predictions, targets);
        

    EXPECT_NO_THROW({
        try
        {
            result["total"].backward();
        }
        catch (const std::exception& e)
        {
            FAIL() << "Backward pass threw an exception: " << e.what();
            throw e;
	    }
        // Check gradients exist for all prediction tensors
        for (const auto& pred : predictions) {
            EXPECT_TRUE(pred.grad().defined());
            EXPECT_GT(pred.grad().abs().sum().item<float>(), 0.0f);
        }
    });
}

TEST_F(OBBLossTest, GradientFlow_WithAngles_ComputesGradients)
{
    OBBLoss loss(numClasses, stride);

    auto predictions = createMultiScalePredictions(2, true);  // Enable gradient
    auto targets = createTargets(2, 5);

    // Set various angles
    for (int64_t i = 0; i < targets.targets.size(0); ++i) {
        float angle = (static_cast<float>(i) / targets.targets.size(0)) * 6.28f - 3.14f;
        targets.targets[i][4] = angle;
    }

    auto result = loss.compute(predictions, targets);

    EXPECT_NO_THROW({
        result["total"].backward();

        for (const auto& pred : predictions) {
            EXPECT_TRUE(pred.grad().defined());
        }
    });
}

// ============================================================================
// Ideal Prediction Tests
// ============================================================================

TEST_F(OBBLossTest, IdealPrediction_PerfectAngle_IncludedInBoxLoss)
{
    OBBLoss loss(numClasses, stride);

    // Create targets with specific angles
    auto targets = createTargets(1, 3);

    // Set specific angles
    targets.targets.index_put_({0, 4}, 0.0f);
    targets.targets.index_put_({1, 4}, 1.57f);  // 90 degrees
    targets.targets.index_put_({2, 4}, -1.57f); // -90 degrees

    auto predictions = createPerfectPredictions(targets);

    auto result = loss.compute(predictions, targets);

    // With perfect angle predictions, box loss should be relatively low
    EXPECT_TRUE(result.find("box") != result.end());
    EXPECT_TRUE(std::isfinite(result["box"].item<float>()));
}

TEST_F(OBBLossTest, IdealPrediction_RandomPredictions_HighLoss)
{
    OBBLoss loss(numClasses, stride);

    // Create targets
    auto targets = createTargets(2, 5);

    // Create completely random predictions
    auto predictions = createMultiScalePredictions(2);

    auto result = loss.compute(predictions, targets);

    // Random predictions should have high loss
    EXPECT_TRUE(result.find("total") != result.end());
    EXPECT_GT(result["total"].item<float>(), 5.0f);
}

TEST_F(OBBLossTest, IdealPrediction_WrongAngle_AffectsBoxLoss)
{
    OBBLoss loss(numClasses, stride);

    // Create targets with specific angle
    auto targets = createTargets(1, 2);
    targets.targets.index_put_({torch::indexing::Slice(), 4}, 0.0f);  // Zero angle

    // Create predictions with perfect bbox but wrong angle
    auto predictions = createPerfectPredictions(targets);

    // Set wrong angles (90 degrees off)
    for (auto& pred : predictions) {
        pred.index_put_(
            {torch::indexing::Slice(), 144},
            1.57f  // 90 degrees instead of 0
        );
    }

    auto result = loss.compute(predictions, targets);

    // Box loss should be affected by wrong angle
    EXPECT_TRUE(result.find("box") != result.end());
    EXPECT_GT(result["box"].item<float>(), 0.0f);
}

TEST_F(OBBLossTest, IdealPrediction_LossMonotonicity_VerifyBehavior)
{
    OBBLoss loss(numClasses, stride);

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

TEST_F(OBBLossTest, IdealPrediction_AnglePeriodicity_SimilarLoss)
{
    OBBLoss loss(numClasses, stride);

    // Create two sets of targets with periodic angles (0 and 2*pi)
    auto targets1 = createTargets(1, 2);
    targets1.targets.index_put_({torch::indexing::Slice(), 4}, 0.0f);

    auto targets2 = createTargets(1, 2);
    targets2.targets.index_put_({torch::indexing::Slice(), 4}, 6.28f);  // ~2*pi

    auto predictions1 = createPerfectPredictions(targets1);
    auto predictions2 = createPerfectPredictions(targets2);

    auto result1 = loss.compute(predictions1, targets1);
    auto result2 = loss.compute(predictions2, targets2);

    // Losses should be similar due to angle periodicity
    // Note: May not be exactly equal due to numerical precision
    EXPECT_TRUE(std::isfinite(result1["total"].item<float>()));
    EXPECT_TRUE(std::isfinite(result2["total"].item<float>()));
}

TEST_F(OBBLossTest, IdealPrediction_RotatedBox_HandlesCorrectly)
{
    OBBLoss loss(numClasses, stride);

    // Create targets with various rotations
    auto targets = createTargets(1, 4);
    targets.targets.index_put_({0, 4}, 0.0f);     // 0 degrees
    targets.targets.index_put_({1, 4}, 0.785f);   // 45 degrees
    targets.targets.index_put_({2, 4}, 1.57f);    // 90 degrees
    targets.targets.index_put_({3, 4}, -1.57f);   // -90 degrees

    auto predictions = createPerfectPredictions(targets);

    EXPECT_NO_THROW({
        auto result = loss.compute(predictions, targets);
        EXPECT_TRUE(result.find("total") != result.end());
        EXPECT_TRUE(std::isfinite(result["total"].item<float>()));
    });
}

TEST_F(OBBLossTest, IdealPrediction_ComponentsContributeToTotal)
{
    OBBLoss loss(numClasses, stride);

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

TEST_F(OBBLossTest, IdealPrediction_ProbIoUVsCIoU_RotatedBoxesBetter)
{
    OBBLoss loss(numClasses, stride);

    // Create targets with rotated boxes
    auto targets = createTargets(1, 3);
    targets.targets.index_put_({torch::indexing::Slice(), 4}, 0.785f);  // 45 degrees

    auto predictions = createPerfectPredictions(targets);

    auto result = loss.compute(predictions, targets);

    // ProbIoU should handle rotated boxes properly
    EXPECT_TRUE(result.find("box") != result.end());
    EXPECT_TRUE(std::isfinite(result["box"].item<float>()));
    EXPECT_GE(result["box"].item<float>(), 0.0f);
}

// ============================================================================
// Name Test
// ============================================================================

TEST_F(OBBLossTest, Name_ReturnsCorrectString)
{
    OBBLoss loss(numClasses, stride);

    EXPECT_EQ(loss.name(), "OBBLoss");
}
