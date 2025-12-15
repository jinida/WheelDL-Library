/**
 * @file TaskAlignedAssignerTest.cpp
 * @brief Unit tests for TaskAlignedAssigner and OBBTaskAlignedAssigner
 *
 * Phase 11: TaskAlignedAssigner Tests
 * Tests task-aligned assignment for object detection and OBB
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Utils/TaskAlignedAssigner.h"

using namespace WheelDL::Model::Utils;

class TaskAlignedAssignerTest : public ::testing::Test {
protected:
    torch::Device device_ = torch::kCPU;
    const float eps_ = 1e-4f;

    // Default parameters matching the class defaults
    const int64_t defaultTopk_ = 13;
    const int64_t defaultNumClasses_ = 80;
    const float defaultAlpha_ = 1.0f;
    const float defaultBeta_ = 6.0f;
    const float defaultEps_ = 1e-9f;

    void SetUp() override {
#ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, 0);
        }
#endif
    }

    /**
     * @brief Create predicted scores tensor
     * @param batchSize Batch size
     * @param numAnchors Number of anchors
     * @param numClasses Number of classes
     * @return Tensor [batch, numAnchors, numClasses]
     */
    torch::Tensor createPdScores(int64_t batchSize, int64_t numAnchors, int64_t numClasses) {
        return torch::rand({batchSize, numAnchors, numClasses}, torch::TensorOptions().device(device_));
    }

    /**
     * @brief Create predicted bboxes tensor
     * @param batchSize Batch size
     * @param numAnchors Number of anchors
     * @return Tensor [batch, numAnchors, 4] in xyxy format
     */
    torch::Tensor createPdBboxes(int64_t batchSize, int64_t numAnchors) {
        auto bboxes = torch::zeros({batchSize, numAnchors, 4}, torch::TensorOptions().device(device_));
        for (int64_t b = 0; b < batchSize; ++b) {
            for (int64_t i = 0; i < numAnchors; ++i) {
                float x1 = static_cast<float>(i % 10) * 64.0f;
                float y1 = static_cast<float>(i / 10) * 64.0f;
                bboxes[b][i][0] = x1;
                bboxes[b][i][1] = y1;
                bboxes[b][i][2] = x1 + 64.0f;
                bboxes[b][i][3] = y1 + 64.0f;
            }
        }
        return bboxes;
    }

    /**
     * @brief Create anchor points tensor
     * @param numAnchors Number of anchors
     * @return Tensor [numAnchors, 2]
     */
    torch::Tensor createAncPoints(int64_t numAnchors) {
        auto points = torch::zeros({numAnchors, 2}, torch::TensorOptions().device(device_));
        for (int64_t i = 0; i < numAnchors; ++i) {
            points[i][0] = static_cast<float>(i % 10) * 64.0f + 32.0f;  // center x
            points[i][1] = static_cast<float>(i / 10) * 64.0f + 32.0f;  // center y
        }
        return points;
    }

    /**
     * @brief Create GT labels tensor
     * @param batchSize Batch size
     * @param maxNumGt Maximum number of GT boxes
     * @param numClasses Number of classes
     * @return Tensor [batch, maxNumGt, 1]
     */
    torch::Tensor createGtLabels(int64_t batchSize, int64_t maxNumGt, int64_t numClasses) {
        return torch::randint(0, numClasses, {batchSize, maxNumGt, 1},
            torch::TensorOptions().dtype(torch::kLong).device(device_));
    }

    /**
     * @brief Create GT bboxes tensor
     * @param batchSize Batch size
     * @param maxNumGt Maximum number of GT boxes
     * @return Tensor [batch, maxNumGt, 4] in xyxy format
     */
    torch::Tensor createGtBboxes(int64_t batchSize, int64_t maxNumGt) {
        auto bboxes = torch::zeros({batchSize, maxNumGt, 4}, torch::TensorOptions().device(device_));
        for (int64_t b = 0; b < batchSize; ++b) {
            for (int64_t i = 0; i < maxNumGt; ++i) {
                float x1 = static_cast<float>(i) * 100.0f;
                float y1 = static_cast<float>(i) * 100.0f;
                bboxes[b][i][0] = x1;
                bboxes[b][i][1] = y1;
                bboxes[b][i][2] = x1 + 80.0f;
                bboxes[b][i][3] = y1 + 80.0f;
            }
        }
        return bboxes;
    }

    /**
     * @brief Create GT mask tensor
     * @param batchSize Batch size
     * @param maxNumGt Maximum number of GT boxes
     * @param numValid Number of valid GTs (rest are masked out)
     * @return Tensor [batch, maxNumGt, 1]
     */
    torch::Tensor createMaskGt(int64_t batchSize, int64_t maxNumGt, int64_t numValid = -1) {
        if (numValid < 0) numValid = maxNumGt;
        auto mask = torch::zeros({batchSize, maxNumGt, 1},
            torch::TensorOptions().dtype(torch::kBool).device(device_));
        for (int64_t b = 0; b < batchSize; ++b) {
            for (int64_t i = 0; i < numValid && i < maxNumGt; ++i) {
                mask[b][i][0] = true;
            }
        }
        return mask;
    }

    /**
     * @brief Create OBB GT bboxes tensor
     * @param batchSize Batch size
     * @param maxNumGt Maximum number of GT boxes
     * @return Tensor [batch, maxNumGt, 5] in [cx, cy, w, h, angle] format
     */
    torch::Tensor createOBBGtBboxes(int64_t batchSize, int64_t maxNumGt) {
        auto bboxes = torch::zeros({batchSize, maxNumGt, 5}, torch::TensorOptions().device(device_));
        for (int64_t b = 0; b < batchSize; ++b) {
            for (int64_t i = 0; i < maxNumGt; ++i) {
                bboxes[b][i][0] = 100.0f + static_cast<float>(i) * 150.0f;  // cx
                bboxes[b][i][1] = 100.0f + static_cast<float>(i) * 150.0f;  // cy
                bboxes[b][i][2] = 80.0f;   // w
                bboxes[b][i][3] = 40.0f;   // h
                bboxes[b][i][4] = static_cast<float>(i) * 0.3f;  // angle
            }
        }
        return bboxes;
    }

    /**
     * @brief Create OBB predicted bboxes tensor
     * @param batchSize Batch size
     * @param numAnchors Number of anchors
     * @return Tensor [batch, numAnchors, 5] in [cx, cy, w, h, angle] format
     */
    torch::Tensor createOBBPdBboxes(int64_t batchSize, int64_t numAnchors) {
        auto bboxes = torch::zeros({batchSize, numAnchors, 5}, torch::TensorOptions().device(device_));
        for (int64_t b = 0; b < batchSize; ++b) {
            for (int64_t i = 0; i < numAnchors; ++i) {
                bboxes[b][i][0] = static_cast<float>(i % 10) * 64.0f + 32.0f;  // cx
                bboxes[b][i][1] = static_cast<float>(i / 10) * 64.0f + 32.0f;  // cy
                bboxes[b][i][2] = 60.0f;   // w
                bboxes[b][i][3] = 30.0f;   // h
                bboxes[b][i][4] = 0.1f;    // angle
            }
        }
        return bboxes;
    }
};

// ============================================================================
// 2.5.1 Basic Assignment Tests
// ============================================================================

TEST_F(TaskAlignedAssignerTest, Constructor_DefaultParameters) {
    TaskAlignedAssigner assigner;
    // Should construct without error with default parameters
    SUCCEED();
}

TEST_F(TaskAlignedAssignerTest, Constructor_CustomParameters) {
    TaskAlignedAssigner assigner(10, 20, 0.5f, 3.0f, 1e-6f);
    SUCCEED();
}

TEST_F(TaskAlignedAssignerTest, Basic_SingleGT_MultiAnchors) {
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = createGtBboxes(batchSize, maxNumGt);
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // Check output shapes
    EXPECT_EQ(targetLabels.size(0), batchSize);
    EXPECT_EQ(targetLabels.size(1), numAnchors);
    EXPECT_EQ(targetBboxes.size(0), batchSize);
    EXPECT_EQ(targetBboxes.size(1), numAnchors);
    EXPECT_EQ(targetBboxes.size(2), 4);
    EXPECT_EQ(fgMask.size(0), batchSize);
    EXPECT_EQ(fgMask.size(1), numAnchors);
}

TEST_F(TaskAlignedAssignerTest, Basic_MultiGT_MultiAnchors) {
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 2;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 5;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = createGtBboxes(batchSize, maxNumGt);
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    EXPECT_EQ(targetLabels.size(0), batchSize);
    EXPECT_EQ(targetLabels.size(1), numAnchors);

    // With multiple GTs, some anchors should be assigned
    auto fgCount = fgMask.sum().item<int64_t>();
    EXPECT_GE(fgCount, 0);
}

TEST_F(TaskAlignedAssignerTest, Basic_NoGT) {
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 0;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = torch::zeros({batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto gtBboxes = torch::zeros({batchSize, maxNumGt, 4}, torch::TensorOptions().device(device_));
    auto maskGt = torch::zeros({batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kBool).device(device_));

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // No GT means no foreground
    auto fgCount = fgMask.sum().item<int64_t>();
    EXPECT_EQ(fgCount, 0);
}

TEST_F(TaskAlignedAssignerTest, Basic_AllAnchorsMatch) {
    TaskAlignedAssigner assigner(100, 10, defaultAlpha_, defaultBeta_, defaultEps_);  // Large topk

    int64_t batchSize = 1;
    int64_t numAnchors = 16;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    // Create a large GT box that covers all anchors
    auto pdScores = torch::ones({batchSize, numAnchors, numClasses}, torch::TensorOptions().device(device_)) * 0.9f;
    auto pdBboxes = torch::zeros({batchSize, numAnchors, 4}, torch::TensorOptions().device(device_));
    auto ancPoints = torch::zeros({numAnchors, 2}, torch::TensorOptions().device(device_));

    // All anchors and predictions in a small region
    for (int64_t i = 0; i < numAnchors; ++i) {
        pdBboxes[0][i][0] = 0.0f;
        pdBboxes[0][i][1] = 0.0f;
        pdBboxes[0][i][2] = 100.0f;
        pdBboxes[0][i][3] = 100.0f;
        ancPoints[i][0] = 50.0f;
        ancPoints[i][1] = 50.0f;
    }

    auto gtLabels = torch::zeros({batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 100.0f, 100.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = torch::ones({batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kBool).device(device_));

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // All anchors should potentially match (depending on topk selection)
    EXPECT_GE(fgMask.sum().item<int64_t>(), 1);
}

TEST_F(TaskAlignedAssignerTest, Basic_NoAnchorsMatch) {
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);

    // Put predictions far from GT
    auto pdBboxes = torch::zeros({batchSize, numAnchors, 4}, torch::TensorOptions().device(device_));
    for (int64_t i = 0; i < numAnchors; ++i) {
        pdBboxes[0][i][0] = 1000.0f;
        pdBboxes[0][i][1] = 1000.0f;
        pdBboxes[0][i][2] = 1064.0f;
        pdBboxes[0][i][3] = 1064.0f;
    }

    auto ancPoints = torch::zeros({numAnchors, 2}, torch::TensorOptions().device(device_));
    for (int64_t i = 0; i < numAnchors; ++i) {
        ancPoints[i][0] = 1032.0f;
        ancPoints[i][1] = 1032.0f;
    }

    auto gtLabels = torch::zeros({batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 64.0f, 64.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = torch::ones({batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kBool).device(device_));

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // No anchors inside GT box
    EXPECT_EQ(fgMask.sum().item<int64_t>(), 0);
}

TEST_F(TaskAlignedAssignerTest, Basic_TopKSelection_Default) {
    // Default topk = 13
    TaskAlignedAssigner assigner(13, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);

    // Large GT to cover multiple anchors
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 640.0f, 640.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // Should have at most topk foreground anchors per GT
    auto fgCount = fgMask.sum().item<int64_t>();
    EXPECT_LE(fgCount, 13 * maxNumGt);
}

TEST_F(TaskAlignedAssignerTest, Basic_CustomTopK) {
    // Custom topk = 5
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 640.0f, 640.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // Should have at most topk=5 foreground anchors per GT
    auto fgCount = fgMask.sum().item<int64_t>();
    EXPECT_LE(fgCount, 5 * maxNumGt);
}

// ============================================================================
// 2.5.2 Alignment Score Tests
// ============================================================================

TEST_F(TaskAlignedAssignerTest, Score_AlphaEffect_Default) {
    // Default alpha = 1.0
    TaskAlignedAssigner assigner(5, 10, 1.0f, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 50;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 320.0f, 320.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // Should produce valid results
    EXPECT_FALSE(targetScores.isnan().any().item<bool>());
}

TEST_F(TaskAlignedAssignerTest, Score_BetaEffect_Default) {
    // Default beta = 6.0 (strong IoU emphasis)
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, 6.0f, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 50;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 320.0f, 320.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    EXPECT_FALSE(targetScores.isnan().any().item<bool>());
}

TEST_F(TaskAlignedAssignerTest, Score_CustomAlpha) {
    // Custom alpha = 0.5
    TaskAlignedAssigner assigner(5, 10, 0.5f, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 50;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 320.0f, 320.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    EXPECT_FALSE(targetScores.isnan().any().item<bool>());
}

TEST_F(TaskAlignedAssignerTest, Score_CustomBeta) {
    // Custom beta = 3.0
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, 3.0f, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 50;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 320.0f, 320.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    EXPECT_FALSE(targetScores.isnan().any().item<bool>());
}

// ============================================================================
// 2.5.3 Dynamic K Selection Tests
// ============================================================================

TEST_F(TaskAlignedAssignerTest, DynamicK_SmallGT) {
    TaskAlignedAssigner assigner(13, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);

    // Small GT box - few positive samples
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 50.0f, 50.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // Should still produce valid results even with few candidates
    EXPECT_FALSE(fgMask.isnan().any().item<bool>());
}

TEST_F(TaskAlignedAssignerTest, DynamicK_LargeGT) {
    TaskAlignedAssigner assigner(13, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);

    // Large GT box - many positive samples
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 640.0f, 640.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    auto fgCount = fgMask.sum().item<int64_t>();
    EXPECT_GE(fgCount, 1);  // Should have some foreground
}

TEST_F(TaskAlignedAssignerTest, DynamicK_KClamping) {
    // topk larger than number of anchors
    TaskAlignedAssigner assigner(100, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 10;  // Few anchors
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 640.0f, 640.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    // Should not crash when topk > numAnchors
    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // k should be clamped to available anchors
    auto fgCount = fgMask.sum().item<int64_t>();
    EXPECT_LE(fgCount, numAnchors);
}

// ============================================================================
// 2.5.4 OBBTaskAlignedAssigner Tests
// ============================================================================

TEST_F(TaskAlignedAssignerTest, OBB_Constructor) {
    OBBTaskAlignedAssigner assigner;
    SUCCEED();
}

TEST_F(TaskAlignedAssignerTest, OBB_RotatedGT) {
    OBBTaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createOBBPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);

    // Rotated GT box
    auto gtBboxes = torch::tensor({{{320.0f, 320.0f, 200.0f, 100.0f, 0.5f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    EXPECT_EQ(targetBboxes.size(2), 5);  // OBB format: cx, cy, w, h, angle
}

TEST_F(TaskAlignedAssignerTest, OBB_MixedAngles) {
    OBBTaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 3;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createOBBPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);

    // GTs with different angles
    auto gtBboxes = torch::tensor({{{200.0f, 200.0f, 100.0f, 50.0f, 0.0f},
                                    {400.0f, 400.0f, 100.0f, 50.0f, 0.785f},  // 45 degrees
                                    {300.0f, 300.0f, 100.0f, 50.0f, 1.57f}}}  // 90 degrees
                                   , torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    EXPECT_FALSE(targetScores.isnan().any().item<bool>());
}

TEST_F(TaskAlignedAssignerTest, OBB_ThinRotatedBoxes) {
    OBBTaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createOBBPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);

    // High aspect ratio rotated box
    auto gtBboxes = torch::tensor({{{320.0f, 320.0f, 200.0f, 20.0f, 0.785f}}},  // 10:1 aspect ratio
                                   torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // Should handle thin boxes correctly
    EXPECT_FALSE(targetScores.isnan().any().item<bool>());
}

// ============================================================================
// 2.5.5 Edge Cases Tests
// ============================================================================

TEST_F(TaskAlignedAssignerTest, EdgeCase_NoGTBoxes) {
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 0;  // nMaxBoxes == 0

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = torch::zeros({batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto gtBboxes = torch::zeros({batchSize, maxNumGt, 4}, torch::TensorOptions().device(device_));
    auto maskGt = torch::zeros({batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kBool).device(device_));

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // Returns default tensors with background index
    EXPECT_EQ(targetLabels.size(1), numAnchors);
    EXPECT_EQ(fgMask.sum().item<int64_t>(), 0);
}

TEST_F(TaskAlignedAssignerTest, EdgeCase_NoValidGTsInBatch) {
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 2;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 3;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = createGtBboxes(batchSize, maxNumGt);

    // All GTs masked out
    auto maskGt = torch::zeros({batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kBool).device(device_));

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // Skips batch, zeros
    EXPECT_EQ(fgMask.sum().item<int64_t>(), 0);
}

TEST_F(TaskAlignedAssignerTest, EdgeCase_TopKLargerThanAnchors) {
    TaskAlignedAssigner assigner(1000, 10, defaultAlpha_, defaultBeta_, defaultEps_);  // Very large topk

    int64_t batchSize = 1;
    int64_t numAnchors = 10;  // Few anchors
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 640.0f, 640.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    // k = min(topk, size) should be applied
    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    EXPECT_LE(fgMask.sum().item<int64_t>(), numAnchors);
}

TEST_F(TaskAlignedAssignerTest, EdgeCase_MultiGTSameAnchor) {
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 16;
    int64_t numClasses = 10;
    int64_t maxNumGt = 2;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);

    // Two overlapping GTs that might match same anchors
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 200.0f, 200.0f},
                                    {50.0f, 50.0f, 250.0f, 250.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // selectHighestOverlaps should resolve conflicts
    EXPECT_FALSE(fgMask.isnan().any().item<bool>());
}

TEST_F(TaskAlignedAssignerTest, EdgeCase_FgMaskGreaterThan1) {
    TaskAlignedAssigner assigner(10, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 25;
    int64_t numClasses = 10;
    int64_t maxNumGt = 3;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);

    // Multiple overlapping GTs
    auto gtBboxes = torch::tensor({{{100.0f, 100.0f, 300.0f, 300.0f},
                                    {150.0f, 150.0f, 350.0f, 350.0f},
                                    {200.0f, 200.0f, 400.0f, 400.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // Should select highest overlap
    EXPECT_FALSE(fgMask.isnan().any().item<bool>());
}

// ============================================================================
// Device Consistency Tests
// ============================================================================

TEST_F(TaskAlignedAssignerTest, DeviceConsistency_Outputs) {
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 50;
    int64_t numClasses = 10;
    int64_t maxNumGt = 2;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = createGtBboxes(batchSize, maxNumGt);
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // All outputs should be on the same device as inputs
    EXPECT_EQ(targetLabels.device().type(), device_.type());
    EXPECT_EQ(targetBboxes.device().type(), device_.type());
    EXPECT_EQ(targetScores.device().type(), device_.type());
    EXPECT_EQ(fgMask.device().type(), device_.type());
    EXPECT_EQ(targetGtIdx.device().type(), device_.type());
}

TEST_F(TaskAlignedAssignerTest, OBB_DeviceConsistency) {
    OBBTaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 50;
    int64_t numClasses = 10;
    int64_t maxNumGt = 2;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createOBBPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = createOBBGtBboxes(batchSize, maxNumGt);
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    EXPECT_EQ(targetLabels.device().type(), device_.type());
    EXPECT_EQ(targetBboxes.device().type(), device_.type());
}

// ============================================================================
// Batch Processing Tests
// ============================================================================

TEST_F(TaskAlignedAssignerTest, BatchProcessing_MultiBatch) {
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 4;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 5;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = createGtBboxes(batchSize, maxNumGt);
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    EXPECT_EQ(targetLabels.size(0), batchSize);
    EXPECT_EQ(targetBboxes.size(0), batchSize);
    EXPECT_EQ(targetScores.size(0), batchSize);
    EXPECT_EQ(fgMask.size(0), batchSize);
}

TEST_F(TaskAlignedAssignerTest, BatchProcessing_VariedGTCounts) {
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 3;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;
    int64_t maxNumGt = 5;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = createGtBboxes(batchSize, maxNumGt);

    // Varied GT counts per batch
    auto maskGt = torch::zeros({batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kBool).device(device_));
    maskGt[0][0][0] = true;  // Batch 0: 1 GT
    maskGt[1][0][0] = true;  // Batch 1: 3 GTs
    maskGt[1][1][0] = true;
    maskGt[1][2][0] = true;
    maskGt[2][0][0] = true;  // Batch 2: 5 GTs
    maskGt[2][1][0] = true;
    maskGt[2][2][0] = true;
    maskGt[2][3][0] = true;
    maskGt[2][4][0] = true;

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // Should handle varied counts per batch
    EXPECT_EQ(targetLabels.size(0), batchSize);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_F(TaskAlignedAssignerTest, NumericalStability_SmallScores) {
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 50;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    // Very small scores
    auto pdScores = torch::ones({batchSize, numAnchors, numClasses}, torch::TensorOptions().device(device_)) * 1e-10f;
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 320.0f, 320.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    EXPECT_FALSE(targetScores.isnan().any().item<bool>());
    EXPECT_FALSE(targetScores.isinf().any().item<bool>());
}

TEST_F(TaskAlignedAssignerTest, NumericalStability_LargeScores) {
    TaskAlignedAssigner assigner(5, 10, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 1;
    int64_t numAnchors = 50;
    int64_t numClasses = 10;
    int64_t maxNumGt = 1;

    // Very large scores (before sigmoid)
    auto pdScores = torch::ones({batchSize, numAnchors, numClasses}, torch::TensorOptions().device(device_)) * 100.0f;
    auto pdBboxes = createPdBboxes(batchSize, numAnchors);
    auto ancPoints = createAncPoints(numAnchors);
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = torch::tensor({{{0.0f, 0.0f, 320.0f, 320.0f}}}, torch::TensorOptions().device(device_));
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    EXPECT_FALSE(targetScores.isnan().any().item<bool>());
    EXPECT_FALSE(targetScores.isinf().any().item<bool>());
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(TaskAlignedAssignerTest, Performance_ManyAnchors) {
    TaskAlignedAssigner assigner(13, 80, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 2;
    int64_t numAnchors = 8400;  // Typical for YOLO
    int64_t numClasses = 80;
    int64_t maxNumGt = 20;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = torch::rand({batchSize, numAnchors, 4}, torch::TensorOptions().device(device_)) * 640.0f;
    auto ancPoints = torch::rand({numAnchors, 2}, torch::TensorOptions().device(device_)) * 640.0f;
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = torch::rand({batchSize, maxNumGt, 4}, torch::TensorOptions().device(device_)) * 640.0f;

    // Ensure valid xyxy format
    gtBboxes.select(2, 2) = gtBboxes.select(2, 0) + torch::abs(gtBboxes.select(2, 2));
    gtBboxes.select(2, 3) = gtBboxes.select(2, 1) + torch::abs(gtBboxes.select(2, 3));

    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto start = std::chrono::high_resolution_clock::now();
    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should complete in reasonable time
    EXPECT_LT(duration.count(), 10000);  // < 10 seconds
}

TEST_F(TaskAlignedAssignerTest, OBB_Performance) {
    OBBTaskAlignedAssigner assigner(13, 80, defaultAlpha_, defaultBeta_, defaultEps_);

    int64_t batchSize = 2;
    int64_t numAnchors = 4200;
    int64_t numClasses = 80;
    int64_t maxNumGt = 10;

    auto pdScores = createPdScores(batchSize, numAnchors, numClasses);
    auto pdBboxes = torch::rand({batchSize, numAnchors, 5}, torch::TensorOptions().device(device_));
    pdBboxes.select(2, 0) *= 640.0f;  // cx
    pdBboxes.select(2, 1) *= 640.0f;  // cy
    pdBboxes.select(2, 2) = pdBboxes.select(2, 2).abs() * 100.0f + 10.0f;  // w
    pdBboxes.select(2, 3) = pdBboxes.select(2, 3).abs() * 50.0f + 10.0f;   // h
    pdBboxes.select(2, 4) *= 3.14159f;  // angle

    auto ancPoints = torch::rand({numAnchors, 2}, torch::TensorOptions().device(device_)) * 640.0f;
    auto gtLabels = createGtLabels(batchSize, maxNumGt, numClasses);
    auto gtBboxes = createOBBGtBboxes(batchSize, maxNumGt);
    auto maskGt = createMaskGt(batchSize, maxNumGt, maxNumGt);

    auto start = std::chrono::high_resolution_clock::now();
    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    EXPECT_LT(duration.count(), 15000);  // < 15 seconds
}
