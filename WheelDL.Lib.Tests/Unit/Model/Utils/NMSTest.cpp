/**
 * @file NMSTest.cpp
 * @brief Unit tests for Non-Maximum Suppression functions
 *
 * Phase 10: NMS Tests
 * Tests nonMaxSuppression and nonMaxSuppressionOBB functions
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Utils/IoU.h"

using namespace WheelDL::Model::Utils;

class NMSTest : public ::testing::Test {
protected:
    torch::Device device_ = torch::kCPU;
    const float eps_ = 1e-4f;

    void SetUp() override {
#ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, 0);
        }
#endif
    }

    /**
     * @brief Create prediction tensor for standard NMS
     *
     * Format: [batch, num_boxes, 4 + num_classes]
     * Box: [cx, cy, w, h, class_scores...]
     */
    torch::Tensor createPrediction(
        const std::vector<std::vector<float>>& boxes,
        const std::vector<std::vector<float>>& classScores,
        int batchSize = 1
    ) {
        int numBoxes = static_cast<int>(boxes.size());
        int numClasses = classScores.empty() ? 0 : static_cast<int>(classScores[0].size());
        int dim2 = 4 + numClasses;

        auto pred = torch::zeros({batchSize, numBoxes, dim2}, torch::TensorOptions().device(device_));

        for (int i = 0; i < numBoxes; ++i) {
            // Box coordinates [cx, cy, w, h]
            for (int j = 0; j < 4; ++j) {
                pred[0][i][j] = boxes[i][j];
            }
            // Class scores
            for (int j = 0; j < numClasses; ++j) {
                pred[0][i][4 + j] = classScores[i][j];
            }
        }

        return pred;
    }

    /**
     * @brief Create OBB prediction tensor
     *
     * Format: [batch, num_boxes, 5 + num_classes]
     * Box: [cx, cy, w, h, angle, class_scores...]
     */
    torch::Tensor createOBBPrediction(
        const std::vector<std::vector<float>>& obbs,
        const std::vector<std::vector<float>>& classScores,
        int batchSize = 1
    ) {
        int numBoxes = static_cast<int>(obbs.size());
        int numClasses = classScores.empty() ? 0 : static_cast<int>(classScores[0].size());
        int dim2 = 5 + numClasses;

        auto pred = torch::zeros({batchSize, numBoxes, dim2}, torch::TensorOptions().device(device_));

        for (int i = 0; i < numBoxes; ++i) {
            // OBB coordinates [cx, cy, w, h, angle]
            for (int j = 0; j < 5; ++j) {
                pred[0][i][j] = obbs[i][j];
            }
            // Class scores
            for (int j = 0; j < numClasses; ++j) {
                pred[0][i][5 + j] = classScores[i][j];
            }
        }

        return pred;
    }
};

// ============================================================================
// 2.4.1 Standard NMS Tests
// ============================================================================

TEST_F(NMSTest, Standard_NoOverlap) {
    // Three disjoint boxes - all should be kept
    std::vector<std::vector<float>> boxes = {
        {5.0f, 5.0f, 10.0f, 10.0f},    // Box 1: centered at (5,5), size 10x10
        {50.0f, 50.0f, 10.0f, 10.0f},  // Box 2: centered at (50,50), size 10x10
        {100.0f, 100.0f, 10.0f, 10.0f} // Box 3: centered at (100,100), size 10x10
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}, {0.8f}, {0.7f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);  // One batch
    EXPECT_EQ(result[0].size(0), 3);  // All 3 boxes kept
}

TEST_F(NMSTest, Standard_AllOverlap) {
    // Three identical boxes - only highest score should be kept
    std::vector<std::vector<float>> boxes = {
        {50.0f, 50.0f, 20.0f, 20.0f},
        {50.0f, 50.0f, 20.0f, 20.0f},
        {50.0f, 50.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.7f}, {0.9f}, {0.8f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(0), 1);  // Only 1 box kept (highest score)

    // Verify it's the highest score box
    if (result[0].size(0) > 0) {
        auto resultCpu = result[0].cpu();
        EXPECT_NEAR(resultCpu[0][4].item<float>(), 0.9f, eps_);  // Confidence
    }
}

TEST_F(NMSTest, Standard_PartialOverlap) {
    // Pre-computed example from test plan
    // boxes = [[0,0,10,10], [1,1,11,11], [20,20,30,30]] -> convert to xywh
    // Box 0 and 1 overlap (IoU ~ 0.68), Box 2 is separate
    std::vector<std::vector<float>> boxes = {
        {5.0f, 5.0f, 10.0f, 10.0f},    // [0,0,10,10] in xyxy -> cx=5, cy=5, w=10, h=10
        {6.0f, 6.0f, 10.0f, 10.0f},    // [1,1,11,11] in xyxy -> cx=6, cy=6, w=10, h=10
        {25.0f, 25.0f, 10.0f, 10.0f}   // [20,20,30,30] in xyxy -> cx=25, cy=25, w=10, h=10
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}, {0.75f}, {0.8f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    // Box 0 (score 0.9) kept, Box 2 (score 0.8) kept, Box 1 (score 0.75) suppressed
    EXPECT_EQ(result[0].size(0), 2);
}

TEST_F(NMSTest, Standard_IoUThreshold_05) {
    // Two overlapping boxes with IoU > 0.5
    // Box 1: cx=10, cy=10, w=20, h=20 -> xyxy: (0,0,20,20)
    // Box 2: cx=12, cy=12, w=20, h=20 -> xyxy: (2,2,22,22)
    // Intersection: (2,2,20,20) = 18*18 = 324
    // Union: 400 + 400 - 324 = 476
    // IoU = 324/476 = 0.68 > 0.5
    std::vector<std::vector<float>> boxes = {
        {10.0f, 10.0f, 20.0f, 20.0f},
        {12.0f, 12.0f, 20.0f, 20.0f}  // Shifted by 2, IoU ~0.68
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}, {0.8f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);  // iouThresh = 0.5

    ASSERT_EQ(result.size(), 1);
    // With IoU > 0.5, lower score box should be suppressed
    EXPECT_EQ(result[0].size(0), 1);
}

TEST_F(NMSTest, Standard_IoUThreshold_03) {
    // More aggressive threshold
    std::vector<std::vector<float>> boxes = {
        {10.0f, 10.0f, 20.0f, 20.0f},
        {15.0f, 15.0f, 20.0f, 20.0f}  // Shifted by 5, IoU should be moderate
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}, {0.8f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.3f, 300);  // iouThresh = 0.3 (aggressive)

    ASSERT_EQ(result.size(), 1);
    // With lower threshold, more boxes get suppressed
    EXPECT_LE(result[0].size(0), 2);
}

TEST_F(NMSTest, Standard_ScoreOrdering) {
    // Verify boxes are processed in score order
    std::vector<std::vector<float>> boxes = {
        {10.0f, 10.0f, 20.0f, 20.0f},
        {50.0f, 50.0f, 20.0f, 20.0f},
        {90.0f, 90.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.3f}, {0.9f}, {0.6f}  // Middle box has highest score
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(0), 3);  // All boxes kept (no overlap)

    // First result should have highest confidence
    if (result[0].size(0) > 0) {
        auto resultCpu = result[0].cpu();
        EXPECT_NEAR(resultCpu[0][4].item<float>(), 0.9f, eps_);
    }
}

TEST_F(NMSTest, Standard_MaxDetections) {
    // Create many boxes, limit by maxDet
    std::vector<std::vector<float>> boxes;
    std::vector<std::vector<float>> scores;

    for (int i = 0; i < 50; ++i) {
        boxes.push_back({static_cast<float>(i * 30), static_cast<float>(i * 30), 10.0f, 10.0f});
        scores.push_back({0.9f - i * 0.01f});
    }

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.1f, 0.5f, 10);  // maxDet = 10

    ASSERT_EQ(result.size(), 1);
    EXPECT_LE(result[0].size(0), 10);  // At most 10 detections
}

TEST_F(NMSTest, Standard_EmptyInput) {
    // Empty prediction
    auto pred = torch::zeros({1, 0, 5}, torch::TensorOptions().device(device_));
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(0), 0);  // Empty output
}

TEST_F(NMSTest, Standard_SingleBox) {
    std::vector<std::vector<float>> boxes = {
        {50.0f, 50.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(0), 1);  // Single box kept
}

// ============================================================================
// 2.4.2 NMS Parameter Tests
// ============================================================================

TEST_F(NMSTest, Parameter_ConfThresh_High) {
    std::vector<std::vector<float>> boxes = {
        {10.0f, 10.0f, 20.0f, 20.0f},
        {50.0f, 50.0f, 20.0f, 20.0f},
        {90.0f, 90.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.95f}, {0.85f}, {0.5f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.9f, 0.5f, 300);  // confThresh = 0.9

    ASSERT_EQ(result.size(), 1);
    // Only boxes with score >= 0.9 should pass
    EXPECT_LE(result[0].size(0), 1);  // Only 0.95 passes
}

TEST_F(NMSTest, Parameter_ConfThresh_Low) {
    std::vector<std::vector<float>> boxes = {
        {10.0f, 10.0f, 20.0f, 20.0f},
        {50.0f, 50.0f, 20.0f, 20.0f},
        {90.0f, 90.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.15f}, {0.12f}, {0.11f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.1f, 0.5f, 300);  // confThresh = 0.1

    ASSERT_EQ(result.size(), 1);
    // All boxes with score >= 0.1 should pass
    EXPECT_EQ(result[0].size(0), 3);
}

TEST_F(NMSTest, Parameter_ConfThresh_Zero) {
    std::vector<std::vector<float>> boxes = {
        {10.0f, 10.0f, 20.0f, 20.0f},
        {50.0f, 50.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.01f}, {0.001f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.0f, 0.5f, 300);  // confThresh = 0.0

    ASSERT_EQ(result.size(), 1);
    // All boxes pass confidence threshold
    EXPECT_EQ(result[0].size(0), 2);
}

TEST_F(NMSTest, Parameter_IoUThresh_High) {
    // Nearly identical boxes - high threshold should keep more
    std::vector<std::vector<float>> boxes = {
        {50.0f, 50.0f, 20.0f, 20.0f},
        {51.0f, 51.0f, 20.0f, 20.0f}  // Very close, high IoU
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}, {0.85f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.95f, 300);  // iouThresh = 0.95 (rarely suppress)

    ASSERT_EQ(result.size(), 1);
    // High threshold means IoU must be > 0.95 to suppress
    // These boxes have high IoU but likely < 0.95, so both kept
    EXPECT_GE(result[0].size(0), 1);
}

TEST_F(NMSTest, Parameter_IoUThresh_Low) {
    // Even moderately overlapping boxes get suppressed
    std::vector<std::vector<float>> boxes = {
        {50.0f, 50.0f, 20.0f, 20.0f},
        {55.0f, 55.0f, 20.0f, 20.0f}  // Shifted by 5
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}, {0.85f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.1f, 300);  // iouThresh = 0.1 (aggressive)

    ASSERT_EQ(result.size(), 1);
    // Low threshold aggressively suppresses
    EXPECT_EQ(result[0].size(0), 1);
}

TEST_F(NMSTest, Parameter_MaxDet_Small) {
    std::vector<std::vector<float>> boxes;
    std::vector<std::vector<float>> scores;

    for (int i = 0; i < 100; ++i) {
        boxes.push_back({static_cast<float>(i * 20), static_cast<float>(i * 20), 10.0f, 10.0f});
        scores.push_back({0.9f});
    }

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 10);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(0), 10);  // Exactly 10 detections
}

TEST_F(NMSTest, Parameter_MaxDet_Large) {
    std::vector<std::vector<float>> boxes;
    std::vector<std::vector<float>> scores;

    for (int i = 0; i < 50; ++i) {
        boxes.push_back({static_cast<float>(i * 30), static_cast<float>(i * 30), 10.0f, 10.0f});
        scores.push_back({0.9f});
    }

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 1000);  // Large maxDet

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(0), 50);  // All boxes kept (maxDet not limiting)
}

TEST_F(NMSTest, Parameter_ReturnFormat) {
    std::vector<std::vector<float>> boxes = {
        {50.0f, 50.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0].size(0), 1);

    // Output format: [x1, y1, x2, y2, confidence, class_id]
    EXPECT_EQ(result[0].size(1), 6);
}

// ============================================================================
// 2.4.3 Class-Aware NMS Tests
// ============================================================================

TEST_F(NMSTest, ClassAware_MultiClass) {
    // Same location, different classes - both should be kept
    std::vector<std::vector<float>> boxes = {
        {50.0f, 50.0f, 20.0f, 20.0f},
        {50.0f, 50.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.9f, 0.1f},   // Class 0 is dominant
        {0.1f, 0.9f}    // Class 1 is dominant
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    // NMS is class-aware: identical boxes with different classes kept
    EXPECT_EQ(result[0].size(0), 2);
}

TEST_F(NMSTest, ClassAware_MixedClasses) {
    // Overlapping boxes but different classes
    std::vector<std::vector<float>> boxes = {
        {50.0f, 50.0f, 20.0f, 20.0f},
        {52.0f, 52.0f, 20.0f, 20.0f}  // Slight overlap
    };
    std::vector<std::vector<float>> scores = {
        {0.9f, 0.0f, 0.0f},   // Class 0
        {0.0f, 0.0f, 0.85f}   // Class 2
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    // Different classes - both kept despite overlap
    EXPECT_EQ(result[0].size(0), 2);
}

TEST_F(NMSTest, ClassAware_SameClass) {
    // Overlapping boxes, same class - standard NMS applies
    std::vector<std::vector<float>> boxes = {
        {50.0f, 50.0f, 20.0f, 20.0f},
        {51.0f, 51.0f, 20.0f, 20.0f}  // Very close
    };
    std::vector<std::vector<float>> scores = {
        {0.9f, 0.0f},   // Class 0
        {0.85f, 0.0f}   // Also class 0
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    // Same class, high overlap - one suppressed
    EXPECT_EQ(result[0].size(0), 1);
}

// ============================================================================
// 2.4.4 nonMaxSuppressionOBB Tests (Oriented NMS)
// ============================================================================

TEST_F(NMSTest, OBB_AlignedBoxes) {
    // No rotation (angle = 0), same as standard NMS
    std::vector<std::vector<float>> obbs = {
        {50.0f, 50.0f, 20.0f, 20.0f, 0.0f},
        {50.0f, 50.0f, 20.0f, 20.0f, 0.0f}  // Identical
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}, {0.8f}
    };

    auto pred = createOBBPrediction(obbs, scores);
    auto result = nonMaxSuppressionOBB(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(0), 1);  // One suppressed due to identical position
}

TEST_F(NMSTest, OBB_RotatedOverlap) {
    // Same center, one rotated - lower overlap
    std::vector<std::vector<float>> obbs = {
        {50.0f, 50.0f, 30.0f, 10.0f, 0.0f},         // Horizontal
        {50.0f, 50.0f, 30.0f, 10.0f, 1.5708f}       // Rotated 90 degrees (pi/2)
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}, {0.85f}
    };

    auto pred = createOBBPrediction(obbs, scores);
    auto result = nonMaxSuppressionOBB(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    // Perpendicular boxes have lower IoU, both might be kept
    EXPECT_GE(result[0].size(0), 1);
}

TEST_F(NMSTest, OBB_DiagonalBoxes) {
    // 45 degree rotated boxes
    const float angle45 = 0.7854f;  // pi/4
    std::vector<std::vector<float>> obbs = {
        {50.0f, 50.0f, 20.0f, 10.0f, angle45},
        {100.0f, 100.0f, 20.0f, 10.0f, angle45}  // Separate
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}, {0.85f}
    };

    auto pred = createOBBPrediction(obbs, scores);
    auto result = nonMaxSuppressionOBB(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(0), 2);  // Both kept (separate locations)
}

TEST_F(NMSTest, OBB_ReturnFormat) {
    std::vector<std::vector<float>> obbs = {
        {50.0f, 50.0f, 20.0f, 20.0f, 0.5f}
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}
    };

    auto pred = createOBBPrediction(obbs, scores);
    auto result = nonMaxSuppressionOBB(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0].size(0), 1);

    // Output format: [cx, cy, w, h, angle, confidence, class_id]
    EXPECT_EQ(result[0].size(1), 7);
}

TEST_F(NMSTest, OBB_EmptyInput) {
    auto pred = torch::zeros({1, 0, 6}, torch::TensorOptions().device(device_));
    auto result = nonMaxSuppressionOBB(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(0), 0);
}

TEST_F(NMSTest, OBB_SingleBox) {
    std::vector<std::vector<float>> obbs = {
        {50.0f, 50.0f, 20.0f, 20.0f, 0.3f}
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}
    };

    auto pred = createOBBPrediction(obbs, scores);
    auto result = nonMaxSuppressionOBB(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(0), 1);
}

// ============================================================================
// Device Consistency Tests
// ============================================================================

TEST_F(NMSTest, Standard_DeviceConsistency) {
    std::vector<std::vector<float>> boxes = {
        {50.0f, 50.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    if (result[0].size(0) > 0) {
        EXPECT_EQ(result[0].device().type(), device_.type());
    }
}

TEST_F(NMSTest, OBB_DeviceConsistency) {
    std::vector<std::vector<float>> obbs = {
        {50.0f, 50.0f, 20.0f, 20.0f, 0.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}
    };

    auto pred = createOBBPrediction(obbs, scores);
    auto result = nonMaxSuppressionOBB(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    if (result[0].size(0) > 0) {
        EXPECT_EQ(result[0].device().type(), device_.type());
    }
}

// ============================================================================
// Batch Processing Tests
// ============================================================================

TEST_F(NMSTest, Standard_MultiBatch) {
    // Create prediction for 2 batches
    int batchSize = 2;
    int numBoxes = 5;
    int numClasses = 2;

    auto pred = torch::zeros({batchSize, numBoxes, 4 + numClasses}, torch::TensorOptions().device(device_));

    // Batch 0: boxes at different positions
    for (int i = 0; i < numBoxes; ++i) {
        pred[0][i][0] = static_cast<float>(i * 50);  // cx
        pred[0][i][1] = static_cast<float>(i * 50);  // cy
        pred[0][i][2] = 20.0f;  // w
        pred[0][i][3] = 20.0f;  // h
        pred[0][i][4] = 0.9f - i * 0.1f;  // class 0 score
    }

    // Batch 1: boxes at different positions
    for (int i = 0; i < numBoxes; ++i) {
        pred[1][i][0] = static_cast<float>(i * 30 + 10);
        pred[1][i][1] = static_cast<float>(i * 30 + 10);
        pred[1][i][2] = 15.0f;
        pred[1][i][3] = 15.0f;
        pred[1][i][4] = 0.85f - i * 0.1f;
    }

    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    EXPECT_EQ(result.size(), 2);  // Two batches
}

TEST_F(NMSTest, OBB_MultiBatch) {
    int batchSize = 2;
    int numBoxes = 3;
    int numClasses = 1;

    auto pred = torch::zeros({batchSize, numBoxes, 5 + numClasses}, torch::TensorOptions().device(device_));

    // Batch 0
    for (int i = 0; i < numBoxes; ++i) {
        pred[0][i][0] = static_cast<float>(i * 50);
        pred[0][i][1] = static_cast<float>(i * 50);
        pred[0][i][2] = 20.0f;
        pred[0][i][3] = 10.0f;
        pred[0][i][4] = static_cast<float>(i) * 0.3f;  // angle
        pred[0][i][5] = 0.9f - i * 0.1f;
    }

    // Batch 1
    for (int i = 0; i < numBoxes; ++i) {
        pred[1][i][0] = static_cast<float>(i * 40);
        pred[1][i][1] = static_cast<float>(i * 40);
        pred[1][i][2] = 15.0f;
        pred[1][i][3] = 8.0f;
        pred[1][i][4] = static_cast<float>(i) * 0.2f;
        pred[1][i][5] = 0.8f - i * 0.15f;
    }

    auto result = nonMaxSuppressionOBB(pred, 0.25f, 0.5f, 300);

    EXPECT_EQ(result.size(), 2);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(NMSTest, EdgeCase_AllBelowConfThresh) {
    std::vector<std::vector<float>> boxes = {
        {50.0f, 50.0f, 20.0f, 20.0f},
        {100.0f, 100.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.1f}, {0.15f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.5f, 0.5f, 300);  // All below 0.5

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(0), 0);  // No detections pass threshold
}

TEST_F(NMSTest, EdgeCase_ZeroSizeBox) {
    std::vector<std::vector<float>> boxes = {
        {50.0f, 50.0f, 0.0f, 0.0f},  // Zero size
        {100.0f, 100.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}, {0.8f}
    };

    auto pred = createPrediction(boxes, scores);
    // Should not crash
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
}

TEST_F(NMSTest, EdgeCase_NegativeCoordinates) {
    std::vector<std::vector<float>> boxes = {
        {-10.0f, -10.0f, 20.0f, 20.0f},
        {10.0f, 10.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}, {0.8f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    // Should handle negative coordinates
    EXPECT_GE(result[0].size(0), 1);
}

TEST_F(NMSTest, EdgeCase_LargeCoordinates) {
    std::vector<std::vector<float>> boxes = {
        {10000.0f, 10000.0f, 200.0f, 200.0f},
        {10500.0f, 10500.0f, 200.0f, 200.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}, {0.8f}
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0].size(0), 2);  // Both should be kept (no overlap)
}

TEST_F(NMSTest, EdgeCase_ManyClasses) {
    // Test with many classes
    std::vector<std::vector<float>> boxes = {
        {50.0f, 50.0f, 20.0f, 20.0f},
        {52.0f, 52.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores(2, std::vector<float>(80, 0.0f));
    scores[0][0] = 0.9f;   // Class 0
    scores[1][79] = 0.85f; // Class 79

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    // Different classes - both kept
    EXPECT_EQ(result[0].size(0), 2);
}

TEST_F(NMSTest, EdgeCase_TiedScores) {
    // Multiple boxes with exactly the same score
    std::vector<std::vector<float>> boxes = {
        {50.0f, 50.0f, 20.0f, 20.0f},
        {100.0f, 100.0f, 20.0f, 20.0f},
        {150.0f, 150.0f, 20.0f, 20.0f}
    };
    std::vector<std::vector<float>> scores = {
        {0.9f}, {0.9f}, {0.9f}  // All same score
    };

    auto pred = createPrediction(boxes, scores);
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);

    ASSERT_EQ(result.size(), 1);
    // All boxes separate, all should be kept
    EXPECT_EQ(result[0].size(0), 3);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST_F(NMSTest, Performance_ManyBoxes) {
    // Test with many boxes
    int numBoxes = 1000;
    auto pred = torch::zeros({1, numBoxes, 5}, torch::TensorOptions().device(device_));

    for (int i = 0; i < numBoxes; ++i) {
        pred[0][i][0] = static_cast<float>(i % 100) * 10.0f;  // cx
        pred[0][i][1] = static_cast<float>(i / 100) * 10.0f;  // cy
        pred[0][i][2] = 8.0f;  // w
        pred[0][i][3] = 8.0f;  // h
        pred[0][i][4] = 0.5f + static_cast<float>(rand() % 100) / 200.0f;  // score
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto result = nonMaxSuppression(pred, 0.25f, 0.5f, 300);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    ASSERT_EQ(result.size(), 1);
    EXPECT_LE(result[0].size(0), 300);  // maxDet limit

    // Should complete in reasonable time (< 5 seconds)
    EXPECT_LT(duration.count(), 5000);
}

TEST_F(NMSTest, OBB_Performance_ManyBoxes) {
    int numBoxes = 500;
    auto pred = torch::zeros({1, numBoxes, 6}, torch::TensorOptions().device(device_));

    for (int i = 0; i < numBoxes; ++i) {
        pred[0][i][0] = static_cast<float>(i % 50) * 20.0f;
        pred[0][i][1] = static_cast<float>(i / 50) * 20.0f;
        pred[0][i][2] = 15.0f;
        pred[0][i][3] = 10.0f;
        pred[0][i][4] = static_cast<float>(i % 10) * 0.1f;  // angle
        pred[0][i][5] = 0.5f + static_cast<float>(rand() % 100) / 200.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto result = nonMaxSuppressionOBB(pred, 0.25f, 0.5f, 300);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    ASSERT_EQ(result.size(), 1);
    EXPECT_LT(duration.count(), 10000);  // Should complete in < 10 seconds
}
