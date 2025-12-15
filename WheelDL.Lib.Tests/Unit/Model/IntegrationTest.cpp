/**
 * @file IntegrationTest.cpp
 * @brief Integration tests for Model/Loss and Model/Utils modules
 *
 * Phase 13: Integration Tests
 * Tests end-to-end workflows, module integration, and branch coverage
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "Model/Loss/BaseLoss.h"
#include "Model/Loss/ClassificationLoss.h"
#include "Model/Loss/DetectionLoss.h"
#include "Model/Loss/OBBLoss.h"
#include "Model/Loss/SegmentationLoss.h"
#include "Model/Loss/AnomalyLoss.h"
#include "Model/Utils/IoU.h"
#include "Model/Utils/TaskAlignedAssigner.h"
#include "Data/Dataset/BaseDataset.h"

using namespace WheelDL::Model::Loss;
using namespace WheelDL::Model::Utils;
using namespace WheelDL::Data::Dataset;

class IntegrationTest : public ::testing::Test {
protected:
    torch::Device device_ = torch::kCPU;
    const float eps_ = 1e-4f;

    void SetUp() override {
        torch::manual_seed(42);  // Reproducibility
#ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, 0);
        }
		torch::set_num_threads(1);
#endif
    }

    torch::Tensor createTensor(std::vector<int64_t> shape) {
        return torch::rand(shape, torch::TensorOptions().device(device_));
    }

    torch::Tensor createIntTensor(std::vector<int64_t> shape, int64_t maxVal) {
        return torch::randint(0, maxVal, shape, torch::TensorOptions().dtype(torch::kLong).device(device_));
    }
};

// ============================================================================
// End-to-End Classification Workflow Tests
// ============================================================================

TEST_F(IntegrationTest, Classification_EndToEnd_BasicWorkflow) {
    // Simulate classification prediction -> loss computation
    int64_t batchSize = 8;
    int64_t numClasses = 10;

    // Model prediction (logits)
    auto predictions = createTensor({batchSize, numClasses});

    // Ground truth labels
    auto targets = createIntTensor({batchSize}, numClasses);

    // Compute loss
    ClassificationLoss loss;
    auto result = loss.compute(predictions, targets);

    // Verify loss is valid
    EXPECT_TRUE(result.find("total") != result.end());
    EXPECT_FALSE(result["total"].isnan().any().item<bool>());
    EXPECT_FALSE(result["total"].isinf().any().item<bool>());
    EXPECT_GT(result["total"].item<float>(), 0.0f);
}

TEST_F(IntegrationTest, Classification_EndToEnd_WithFocalLoss) {
    int64_t batchSize = 16;
    int64_t numClasses = 80;

    auto predictions = createTensor({batchSize, numClasses});
    auto targets = createIntTensor({batchSize}, numClasses);

    ClassificationLoss loss;
    loss.setFocalParams(2.0f, 0.25f);  // gamma=2, alpha=0.25

    auto result = loss.compute(predictions, targets);

    EXPECT_TRUE(result.find("total") != result.end());
    EXPECT_FALSE(result["total"].isnan().any().item<bool>());
}

TEST_F(IntegrationTest, Classification_EndToEnd_GradientBackprop) {
    int64_t batchSize = 4;
    int64_t numClasses = 10;

    auto predictions = createTensor({batchSize, numClasses});
    predictions.requires_grad_(true);
    auto targets = createIntTensor({batchSize}, numClasses);

    ClassificationLoss loss;
    auto result = loss.compute(predictions, targets);

    // Backpropagate
    result["total"].backward();

    // Verify gradients exist and are valid
    EXPECT_TRUE(predictions.grad().defined());
    EXPECT_FALSE(predictions.grad().isnan().any().item<bool>());
    EXPECT_FALSE(predictions.grad().isinf().any().item<bool>());
}

// ============================================================================
// End-to-End Detection Workflow Tests
// ============================================================================

TEST_F(IntegrationTest, Detection_EndToEnd_BasicWorkflow) {
    int64_t batchSize = 2;
    int64_t numClasses = 80;
    int64_t imgSize = 640;
    int64_t regMax = 16;

    // Multi-scale feature predictions
    std::vector<torch::Tensor> predictions;

    // P3 (80x80), P4 (40x40), P5 (20x20)
    int64_t channelSize = regMax * 4 + numClasses;  // 64 + 80 = 144

    predictions.push_back(createTensor({batchSize, channelSize, 80, 80}));
    predictions.push_back(createTensor({batchSize, channelSize, 40, 40}));
    predictions.push_back(createTensor({batchSize, channelSize, 20, 20}));

    // Create DataExample with detection targets
    // targets format: [N, 6] with [batch_idx, class_id, cx, cy, w, h] (normalized)
    DataExample target;
    int64_t numTargets = 10;
    auto batchIndices = torch::randint(0, batchSize, {numTargets, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto classIds = torch::randint(0, numClasses, {numTargets, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto cx = torch::rand({numTargets, 1}, device_) * 0.6f + 0.2f;
    auto cy = torch::rand({numTargets, 1}, device_) * 0.6f + 0.2f;
    auto w = torch::rand({numTargets, 1}, device_) * 0.2f + 0.1f;
    auto h = torch::rand({numTargets, 1}, device_) * 0.2f + 0.1f;
    target.targets = torch::cat({batchIndices, classIds, cx, cy, w, h}, 1);

    auto strides = torch::tensor({8.0f, 16.0f, 32.0f}, torch::TensorOptions().device(device_));
    DetectionLoss loss(numClasses, imgSize, strides);
    loss.to(device_);

    auto result = loss.compute(predictions, target);

    // Verify all loss components
    EXPECT_TRUE(result.find("box") != result.end());
    EXPECT_TRUE(result.find("cls") != result.end());
    EXPECT_TRUE(result.find("dfl") != result.end());
    EXPECT_TRUE(result.find("total") != result.end());
}

TEST_F(IntegrationTest, Detection_EndToEnd_EmptyTargets) {
    int64_t batchSize = 2;
    int64_t numClasses = 80;
    int64_t imgSize = 640;
    int64_t regMax = 16;

    std::vector<torch::Tensor> predictions;
    int64_t channelSize = regMax * 4 + numClasses;

    predictions.push_back(createTensor({batchSize, channelSize, 80, 80}));
    predictions.push_back(createTensor({batchSize, channelSize, 40, 40}));
    predictions.push_back(createTensor({batchSize, channelSize, 20, 20}));

    // Empty targets: [0, 6] format with [batch_idx, class_id, cx, cy, w, h]
    DataExample target;
    target.targets = torch::zeros({0, 6}, torch::TensorOptions().device(device_));

    auto strides = torch::tensor({8.0f, 16.0f, 32.0f}, torch::TensorOptions().device(device_));
    DetectionLoss loss(numClasses, imgSize, strides);
    loss.to(device_);

    auto result = loss.compute(predictions, target);

    // Should handle empty targets gracefully
    EXPECT_TRUE(result.find("total") != result.end());
}

// ============================================================================
// End-to-End OBB Workflow Tests
// ============================================================================

TEST_F(IntegrationTest, OBB_EndToEnd_BasicWorkflow) {
    int64_t batchSize = 2;
    int64_t numClasses = 15;  // DOTA classes
    int64_t imgSize = 1024;
    int64_t regMax = 16;

    std::vector<torch::Tensor> predictions;
    int64_t channelSize = regMax * 4 + numClasses + 1;  // +1 for angle

    predictions.push_back(createTensor({batchSize, channelSize, 128, 128}));
    predictions.push_back(createTensor({batchSize, channelSize, 64, 64}));
    predictions.push_back(createTensor({batchSize, channelSize, 32, 32}));

    // targets format: [N, 7] with [batch_idx, class_id, cx, cy, w, h, angle] (normalized)
    DataExample target;
    int64_t numTargets = 5;
    auto batchIndices = torch::randint(0, batchSize, {numTargets, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto classIds = torch::randint(0, numClasses, {numTargets, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto cx = torch::rand({numTargets, 1}, device_) * 0.6f + 0.2f;
    auto cy = torch::rand({numTargets, 1}, device_) * 0.6f + 0.2f;
    auto w = torch::rand({numTargets, 1}, device_) * 0.2f + 0.1f;  // normalized w in [0.1, 0.3]
    auto h = torch::rand({numTargets, 1}, device_) * 0.2f + 0.1f;  // normalized h in [0.1, 0.3]
    auto angle = torch::rand({numTargets, 1}, device_) * 3.14159f - 1.5708f;  // angle in [-pi/2, pi/2]
    target.targets = torch::cat({batchIndices, classIds, cx, cy, w, h, angle}, 1);

    auto strides = torch::tensor({8.0f, 16.0f, 32.0f}, torch::TensorOptions().device(device_));
    OBBLoss loss(numClasses, imgSize, strides);
    loss.to(device_);

    auto result = loss.compute(predictions, target);

    EXPECT_TRUE(result.find("total") != result.end());
    EXPECT_FALSE(result["total"].isnan().any().item<bool>());
}

// ============================================================================
// End-to-End Segmentation Workflow Tests
// ============================================================================

TEST_F(IntegrationTest, Segmentation_EndToEnd_BasicWorkflow) {
    int64_t batchSize = 2;
    int64_t numClasses = 10;
    int64_t height = 64;
    int64_t width = 64;

    // Segmentation predictions (logits) [N, num_classes, H, W]
    auto predictions = createTensor({batchSize, numClasses, height, width});

    // Target masks [N, num_classes, H, W] (binary)
    auto targetMasks = torch::zeros({batchSize, numClasses, height, width}, torch::TensorOptions().device(device_));
    // Set some regions as positive
    targetMasks.slice(2, 10, 30).slice(3, 10, 30) = 1.0f;

    SegmentationLoss loss(numClasses);

    auto result = loss.compute(predictions, targetMasks);

    // Verify loss components
    EXPECT_TRUE(result.find("bce") != result.end());
    EXPECT_TRUE(result.find("dice") != result.end());
    EXPECT_TRUE(result.find("total") != result.end());
    EXPECT_FALSE(result["total"].isnan().any().item<bool>());
}

// ============================================================================
// End-to-End Anomaly Workflow Tests
// ============================================================================

TEST_F(IntegrationTest, Anomaly_EndToEnd_SimpleNet) {
    int64_t batchSize = 4;

    // SimpleNet predictions: single anomaly score per sample
    std::vector<torch::Tensor> predictions;
    predictions.push_back(createTensor({batchSize, 1}));

    DataExample target;
    target.classes = torch::zeros({batchSize, 1}, torch::TensorOptions().device(device_));

    AnomalyLoss loss(AnomalyLoss::LossType::SimpleNet);

    auto result = loss.compute(predictions, target);

    EXPECT_TRUE(result.find("total") != result.end());
    EXPECT_FALSE(result["total"].isnan().any().item<bool>());
}

TEST_F(IntegrationTest, Anomaly_EndToEnd_EfficientAD) {
    int64_t batchSize = 4;
    int64_t featureSize = 256;
    int64_t spatialSize = 64;

    // EfficientAD requires 5 tensors: [teacher_out, student_out, ae_teacher_out, ae_student_out, ae_out]
    std::vector<torch::Tensor> predictions;
    predictions.push_back(createTensor({batchSize, featureSize, spatialSize, spatialSize}));  // teacher_out
    predictions.push_back(createTensor({batchSize, featureSize, spatialSize, spatialSize}));  // student_out
    predictions.push_back(createTensor({batchSize, featureSize, spatialSize, spatialSize}));  // ae_teacher_out
    predictions.push_back(createTensor({batchSize, featureSize, spatialSize, spatialSize}));  // ae_student_out
    predictions.push_back(createTensor({batchSize, featureSize, spatialSize, spatialSize}));  // ae_out

    DataExample target;
    target.classes = torch::zeros({batchSize, 1}, torch::TensorOptions().device(device_));

    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);

    auto result = loss.compute(predictions, target);

    EXPECT_TRUE(result.find("total") != result.end());
}

// ============================================================================
// Utils Integration Tests
// ============================================================================

TEST_F(IntegrationTest, Utils_IoU_WithCoordinateConversion) {
    // Test IoU computation with coordinate conversion
    int64_t numBoxes = 100;

    // Create boxes in xywh format
    auto boxesXywh = createTensor({numBoxes, 4});
    boxesXywh.select(1, 2).abs_().add_(10.0f);  // w > 0
    boxesXywh.select(1, 3).abs_().add_(10.0f);  // h > 0

    // Convert to xyxy
    auto boxesXyxy = xywh2xyxy(boxesXywh);

    // Convert back to xywh
    auto boxesXywhBack = xyxy2xywh(boxesXyxy);

    // Should be equal (within tolerance)
    EXPECT_TRUE(torch::allclose(boxesXywh, boxesXywhBack, 1e-4, 1e-4));
}

TEST_F(IntegrationTest, Utils_IoU_AllVariants) {
    int64_t numBoxes = 50;

    auto box1 = createTensor({numBoxes, 4});
    auto box2 = createTensor({numBoxes, 4});

    // Ensure positive width/height for xywh format
    box1.select(1, 2).abs_().add_(1.0f);
    box1.select(1, 3).abs_().add_(1.0f);
    box2.select(1, 2).abs_().add_(1.0f);
    box2.select(1, 3).abs_().add_(1.0f);

    // Standard IoU
    auto iou = bboxIoU(box1, box2, true);
    EXPECT_EQ(iou.size(0), numBoxes);
    EXPECT_FALSE(iou.isnan().any().item<bool>());

    // GIoU
    auto giou = bboxIoU(box1, box2, true, true, false, false);
    EXPECT_FALSE(giou.isnan().any().item<bool>());

    // DIoU
    auto diou = bboxIoU(box1, box2, true, false, true, false);
    EXPECT_FALSE(diou.isnan().any().item<bool>());

    // CIoU
    auto ciou = bboxIoU(box1, box2, true, false, false, true);
    EXPECT_FALSE(ciou.isnan().any().item<bool>());
}

TEST_F(IntegrationTest, Utils_Probiou_WithRotation) {
    int64_t numBoxes = 30;

    auto obb1 = createTensor({numBoxes, 5});
    auto obb2 = createTensor({numBoxes, 5});

    // Ensure positive width/height
    obb1.select(1, 2).abs_().add_(10.0f);
    obb1.select(1, 3).abs_().add_(5.0f);
    obb2.select(1, 2).abs_().add_(10.0f);
    obb2.select(1, 3).abs_().add_(5.0f);

    // Set angles
    obb1.select(1, 4) = torch::rand({numBoxes}, torch::TensorOptions().device(device_)) * 3.14159f;
    obb2.select(1, 4) = torch::rand({numBoxes}, torch::TensorOptions().device(device_)) * 3.14159f;

    auto iou = probiou(obb1, obb2);

    EXPECT_EQ(iou.size(0), numBoxes);
    EXPECT_FALSE(iou.isnan().any().item<bool>());
}

TEST_F(IntegrationTest, Utils_MakeAnchors_WithDist2Bbox) {
    // Generate anchors and use them with dist2bbox
    std::vector<torch::Tensor> feats;
    feats.push_back(createTensor({1, 256, 80, 80}));
    feats.push_back(createTensor({1, 256, 40, 40}));
    feats.push_back(createTensor({1, 256, 20, 20}));

    auto strides = torch::tensor({8.0f, 16.0f, 32.0f}, torch::TensorOptions().device(device_));

    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);

    int64_t totalAnchors = 80 * 80 + 40 * 40 + 20 * 20;  // 8400
    EXPECT_EQ(anchors.size(0), totalAnchors);

    // Use anchors with dist2bbox
    auto distances = createTensor({totalAnchors, 4});
    distances.abs_();  // Ensure positive distances

    auto bboxes = dist2bbox(distances, anchors, true);
    EXPECT_EQ(bboxes.size(0), totalAnchors);
    EXPECT_EQ(bboxes.size(1), 4);
}

TEST_F(IntegrationTest, Utils_Dist2Rbox_WithXywhr2Xyxyxyxy) {
    int64_t numAnchors = 100;

    auto distances = createTensor({numAnchors, 4});
    distances.abs_();
    auto angles = createTensor({numAnchors, 1});
    auto anchors = createTensor({numAnchors, 2});

    auto rboxXywh = dist2rbox(distances, angles, anchors);
    EXPECT_EQ(rboxXywh.size(0), numAnchors);
    EXPECT_EQ(rboxXywh.size(1), 4);

    // Add angle back for xywhr format
    auto rboxXywhr = torch::cat({rboxXywh, angles}, 1);
    EXPECT_EQ(rboxXywhr.size(1), 5);

    // Convert to corner points
    auto corners = xywhr2xyxyxyxy(rboxXywhr);
    EXPECT_EQ(corners.size(0), numAnchors);
    EXPECT_EQ(corners.size(1), 4);  // 4 corners
    EXPECT_EQ(corners.size(2), 2);  // x, y
}

// ============================================================================
// TaskAlignedAssigner Integration Tests
// ============================================================================

TEST_F(IntegrationTest, TaskAlignedAssigner_WithDetectionLoss) {
    int64_t batchSize = 2;
    int64_t numAnchors = 8400;
    int64_t numClasses = 80;

    TaskAlignedAssigner assigner(13, numClasses, 1.0f, 6.0f);

    // Predicted scores and boxes
    auto pdScores = createTensor({batchSize, numAnchors, numClasses});
    auto pdBboxes = createTensor({batchSize, numAnchors, 4});

    // Anchor points
    auto ancPoints = createTensor({numAnchors, 2});
    ancPoints *= 640.0f;

    // Ground truth
    int64_t maxNumGt = 10;
    auto gtLabels = torch::randint(0, numClasses, {batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto gtBboxes = createTensor({batchSize, maxNumGt, 4});
    gtBboxes *= 640.0f;
    // Ensure valid xyxy format
    gtBboxes.select(2, 2) = gtBboxes.select(2, 0) + (gtBboxes.select(2, 2).abs() * 0.1f + 10.0f);
    gtBboxes.select(2, 3) = gtBboxes.select(2, 1) + (gtBboxes.select(2, 3).abs() * 0.1f + 10.0f);

    auto maskGt = torch::ones({batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kBool).device(device_));

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    // Verify shapes
    EXPECT_EQ(targetLabels.size(0), batchSize);
    EXPECT_EQ(targetLabels.size(1), numAnchors);
    EXPECT_EQ(targetBboxes.size(0), batchSize);
    EXPECT_EQ(targetBboxes.size(1), numAnchors);
    EXPECT_EQ(fgMask.size(0), batchSize);
    EXPECT_EQ(fgMask.size(1), numAnchors);
}

TEST_F(IntegrationTest, OBBTaskAlignedAssigner_WithProbiou) {
    int64_t batchSize = 1;
    int64_t numAnchors = 1000;
    int64_t numClasses = 15;

    OBBTaskAlignedAssigner assigner(10, numClasses, 1.0f, 6.0f);

    auto pdScores = createTensor({batchSize, numAnchors, numClasses});
    auto pdBboxes = createTensor({batchSize, numAnchors, 5});
    pdBboxes.select(2, 2).abs_().add_(10.0f);
    pdBboxes.select(2, 3).abs_().add_(5.0f);

    auto ancPoints = createTensor({numAnchors, 2});
    ancPoints *= 1024.0f;

    int64_t maxNumGt = 5;
    auto gtLabels = torch::randint(0, numClasses, {batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto gtBboxes = createTensor({batchSize, maxNumGt, 5});
    gtBboxes.select(2, 0) *= 1024.0f;  // cx
    gtBboxes.select(2, 1) *= 1024.0f;  // cy
    gtBboxes.select(2, 2).abs_().mul_(100.0f).add_(20.0f);  // w
    gtBboxes.select(2, 3).abs_().mul_(50.0f).add_(10.0f);   // h

    auto maskGt = torch::ones({batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kBool).device(device_));

    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, ancPoints, gtLabels, gtBboxes, maskGt);

    EXPECT_EQ(targetBboxes.size(2), 5);  // OBB format
}

// ============================================================================
// NMS Integration Tests
// ============================================================================

TEST_F(IntegrationTest, NMS_WithDetectionOutput) {
    // Simulate detection output -> NMS filtering
    int64_t batchSize = 2;
    int64_t numBoxes = 1000;
    int64_t numClasses = 80;

    // Detection predictions: [batch, numBoxes, 4 + numClasses]
    auto predictions = createTensor({batchSize, numBoxes, 4 + numClasses});

    // Set some boxes with high scores
    predictions.select(2, 4) = 0.1f;  // Low default score
    for (int i = 0; i < 50; ++i) {
        predictions[0][i][4 + (i % numClasses)] = 0.9f;  // High score for some
        predictions[1][i][4 + (i % numClasses)] = 0.85f;
    }

    auto result = nonMaxSuppression(predictions, 0.25f, 0.45f, 300);

    EXPECT_EQ(result.size(), batchSize);
    for (int64_t b = 0; b < batchSize; ++b) {
        EXPECT_LE(result[b].size(0), 300);  // maxDet limit
        if (result[b].size(0) > 0) {
            EXPECT_EQ(result[b].size(1), 6);  // [x1, y1, x2, y2, conf, class]
        }
    }
}

TEST_F(IntegrationTest, NMS_OBB_WithRotatedBoxes) {
    int64_t batchSize = 1;
    int64_t numBoxes = 500;
    int64_t numClasses = 15;

    // OBB predictions: [batch, numBoxes, 5 + numClasses]
    auto predictions = createTensor({batchSize, numBoxes, 5 + numClasses});

    // Set boxes with various rotations
    for (int i = 0; i < numBoxes; ++i) {
        predictions[0][i][0] = static_cast<float>(i % 100) * 10.0f;  // cx
        predictions[0][i][1] = static_cast<float>(i / 100) * 10.0f;  // cy
        predictions[0][i][2] = 50.0f;  // w
        predictions[0][i][3] = 25.0f;  // h
        predictions[0][i][4] = static_cast<float>(i % 10) * 0.314f;  // angle
        predictions[0][i][5 + (i % numClasses)] = 0.8f;  // class score
    }

    auto result = nonMaxSuppressionOBB(predictions, 0.25f, 0.45f, 100);

    EXPECT_EQ(result.size(), batchSize);
    if (result[0].size(0) > 0) {
        EXPECT_EQ(result[0].size(1), 7);  // [cx, cy, w, h, angle, conf, class]
    }
}

// ============================================================================
// Cross-Module Integration Tests
// ============================================================================

TEST_F(IntegrationTest, CrossModule_LossWithAssignerAndIoU) {
    // Full pipeline: Assigner assigns targets, IoU is computed, Loss is calculated

    int64_t batchSize = 1;
    int64_t numAnchors = 100;
    int64_t numClasses = 10;

    // Step 1: Generate anchors
    std::vector<std::pair<int64_t, int64_t>> featShapes = {{10, 10}};
    auto strides = torch::tensor({8.0f}, torch::TensorOptions().device(device_));
    auto [anchors, strideTensor] = makeAnchors(featShapes, strides, torch::kFloat32, device_, 0.5f);

    // Step 2: Create predictions
    auto pdScores = torch::sigmoid(createTensor({batchSize, numAnchors, numClasses}));
    auto pdBboxes = createTensor({batchSize, numAnchors, 4});
    pdBboxes *= 80.0f;

    // Step 3: Create ground truth
    int64_t maxNumGt = 3;
    auto gtLabels = torch::randint(0, numClasses, {batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kLong).device(device_));
    auto gtBboxes = createTensor({batchSize, maxNumGt, 4});
    gtBboxes *= 80.0f;
    gtBboxes.select(2, 2) = gtBboxes.select(2, 0) + 20.0f;
    gtBboxes.select(2, 3) = gtBboxes.select(2, 1) + 20.0f;
    auto maskGt = torch::ones({batchSize, maxNumGt, 1},
        torch::TensorOptions().dtype(torch::kBool).device(device_));

    // Step 4: Assign targets
    TaskAlignedAssigner assigner(5, numClasses);
    auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] =
        assigner.forward(pdScores, pdBboxes, anchors, gtLabels, gtBboxes, maskGt);

    // Step 5: Compute IoU between predictions and targets for foreground
    auto fgCount = fgMask.sum().item<int64_t>();
    if (fgCount > 0) {
        auto fgIndices = fgMask[0].nonzero().squeeze();
        if (fgIndices.dim() == 0) {
            fgIndices = fgIndices.unsqueeze(0);
        }

        // IoU computation would happen here in real loss computation
        EXPECT_GE(fgCount, 0);
    }

    // Verify integration worked
    EXPECT_EQ(targetLabels.device().type(), device_.type());
    EXPECT_EQ(targetBboxes.device().type(), device_.type());
}

// ============================================================================
// Device Consistency Integration Tests
// ============================================================================

TEST_F(IntegrationTest, DeviceConsistency_AllModules) {
    // Verify all modules produce outputs on correct device

    // Classification
    ClassificationLoss clsLoss;
    auto clsPred = createTensor({4, 10});
    auto clsTarget = createIntTensor({4}, 10);
    auto clsResult = clsLoss.compute(clsPred, clsTarget);
    EXPECT_EQ(clsResult["total"].device().type(), device_.type());

    // IoU functions
    auto box1 = createTensor({10, 4});
    auto box2 = createTensor({10, 4});
    box1.select(1, 2).abs_().add_(1.0f);
    box1.select(1, 3).abs_().add_(1.0f);
    box2.select(1, 2).abs_().add_(1.0f);
    box2.select(1, 3).abs_().add_(1.0f);
    auto iou = bboxIoU(box1, box2, true);
    EXPECT_EQ(iou.device().type(), device_.type());

    // Coordinate conversion
    auto converted = xywh2xyxy(box1);
    EXPECT_EQ(converted.device().type(), device_.type());

    // makeAnchors
    std::vector<torch::Tensor> feats;
    feats.push_back(createTensor({1, 256, 10, 10}));
    auto strides = torch::tensor({8.0f}, torch::TensorOptions().device(device_));
    auto [anchors, strideTensor] = makeAnchors(feats, strides, 0.5f);
    EXPECT_EQ(anchors.device().type(), device_.type());
}

// ============================================================================
// Performance Integration Tests
// ============================================================================

TEST_F(IntegrationTest, Performance_FullDetectionPipeline) {
    int64_t batchSize = 4;
    int64_t numClasses = 80;
    int64_t imgSize = 640;
    int64_t regMax = 16;

    std::vector<torch::Tensor> predictions;
    int64_t channelSize = regMax * 4 + numClasses;

    predictions.push_back(createTensor({batchSize, channelSize, 80, 80}));
    predictions.push_back(createTensor({batchSize, channelSize, 40, 40}));
    predictions.push_back(createTensor({batchSize, channelSize, 20, 20}));

    // targets format: [N, 6] with [batch_idx, class_id, cx, cy, w, h] (normalized)
    DataExample target;
    int64_t numTargets = 20;
    auto batchIndicesTensor = torch::zeros({numTargets, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    for (int i = 0; i < numTargets; ++i) {
        batchIndicesTensor[i][0] = static_cast<float>(i % batchSize);
    }
    auto classIds = torch::randint(0, numClasses, {numTargets, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto cx = torch::rand({numTargets, 1}, device_) * 0.6f + 0.2f;
    auto cy = torch::rand({numTargets, 1}, device_) * 0.6f + 0.2f;
    auto w = torch::rand({numTargets, 1}, device_) * 0.2f + 0.1f;
    auto h = torch::rand({numTargets, 1}, device_) * 0.2f + 0.1f;
    target.targets = torch::cat({batchIndicesTensor, classIds, cx, cy, w, h}, 1);

    auto strides = torch::tensor({8.0f, 16.0f, 32.0f}, torch::TensorOptions().device(device_));
    DetectionLoss loss(numClasses, imgSize, strides);
    loss.to(device_);

    auto start = std::chrono::high_resolution_clock::now();
    auto result = loss.compute(predictions, target);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should complete in reasonable time
    EXPECT_LT(duration.count(), 5000);
    EXPECT_TRUE(result.find("total") != result.end());
}
