#pragma once

/**
 * @file MockDataExample.h
 * @brief Mock DataExample helpers for Model/Task tests
 *
 * Provides helper functions to create mock DataExample objects
 * for testing Task model classes.
 *
 * Phase 1 of Model_Task_Test_Plan.md
 */

#include "Data/Dataset/BaseDataset.h"
#include <torch/torch.h>
#include <vector>
#include <cstdint>

namespace WheelDL {
namespace Test {

using Data::Dataset::DataExample;

/**
 * @brief Create a basic mock DataExample with random image data
 *
 * @param shape Image shape [B, C, H, W] or [C, H, W]
 * @param numTargets Number of targets (boxes/labels) to generate
 * @param numClasses Number of classes for random class generation
 * @return DataExample with random data
 */
inline DataExample createMockDataExample(
    const std::vector<int64_t>& shape,
    int numTargets = 5,
    int numClasses = 80)
{
    DataExample example;

    // Create image tensor - normalized random values [0, 1]
    example.data = torch::rand(shape);

    // Create classes tensor - random class indices
    if (numTargets > 0) {
        example.classes = torch::randint(0, numClasses, {numTargets});
    }

    // Create empty targets and batchIndices by default
    // (subclasses will fill these based on task type)
    example.targets = torch::empty({0});
    example.batchIndices = torch::empty({0});

    return example;
}

/**
 * @brief Create a mock DataExample for classification testing
 *
 * @param batchSize Batch size
 * @param channels Number of channels (default 3)
 * @param height Image height (default 224)
 * @param width Image width (default 224)
 * @param numClasses Number of classes (default 10)
 * @return DataExample with classification data
 */
inline DataExample createClassificationDataExample(
    int batchSize = 1,
    int channels = 3,
    int height = 224,
    int width = 224,
    int numClasses = 10)
{
    DataExample example;

    // Image data: [B, C, H, W]
    example.data = torch::rand({batchSize, channels, height, width});

    // Classes: [B] - single class per image for classification
    example.classes = torch::randint(0, numClasses, {batchSize});

    // Targets: same as classes for classification
    example.targets = example.classes.clone();

    // Batch indices: sequential
    example.batchIndices = torch::arange(batchSize);

    return example;
}

/**
 * @brief Create a mock DataExample for detection testing
 *
 * Format follows YOLO format:
 * - targets: [N, 6] where 6 = [batch_idx, class_id, x, y, w, h] (normalized)
 *
 * @param batchSize Batch size
 * @param imageSize Image size (square)
 * @param numTargets Total number of targets across all images
 * @param numClasses Number of classes
 * @return DataExample with detection data
 */
inline DataExample createDetectionDataExample(
    int batchSize = 1,
    int imageSize = 640,
    int numTargets = 5,
    int numClasses = 80)
{
    DataExample example;

    // Image data: [B, C, H, W]
    example.data = torch::rand({batchSize, 3, imageSize, imageSize});

    // Targets: [N, 6] - [batch_idx, class_id, x_center, y_center, width, height]
    // All coordinates are normalized [0, 1]
    if (numTargets > 0) {
        auto targets = torch::zeros({numTargets, 6});

        // Batch indices (distribute targets across batches)
        targets.index({torch::indexing::Slice(), 0}) =
            torch::randint(0, batchSize, {numTargets}).to(torch::kFloat32);

        // Class IDs
        targets.index({torch::indexing::Slice(), 1}) =
            torch::randint(0, numClasses, {numTargets}).to(torch::kFloat32);

        // Box coordinates (x_center, y_center in [0.1, 0.9] to avoid edge cases)
        targets.index({torch::indexing::Slice(), torch::indexing::Slice(2, 4)}) =
            torch::rand({numTargets, 2}) * 0.8f + 0.1f;

        // Box dimensions (width, height in [0.05, 0.3])
        targets.index({torch::indexing::Slice(), torch::indexing::Slice(4, 6)}) =
            torch::rand({numTargets, 2}) * 0.25f + 0.05f;

        example.targets = targets;

        // Classes tensor
        example.classes = targets.index({torch::indexing::Slice(), 1}).to(torch::kInt64);

        // Batch indices
        example.batchIndices = targets.index({torch::indexing::Slice(), 0}).to(torch::kInt64);
    }
    else {
        example.targets = torch::zeros({0, 6});
        example.classes = torch::zeros({0}, torch::kInt64);
        example.batchIndices = torch::zeros({0}, torch::kInt64);
    }

    return example;
}

/**
 * @brief Create a mock DataExample for OBB (Oriented Bounding Box) testing
 *
 * Format:
 * - targets: [N, 7] where 7 = [batch_idx, class_id, x, y, w, h, angle]
 *
 * @param batchSize Batch size
 * @param imageSize Image size (square)
 * @param numTargets Total number of targets
 * @param numClasses Number of classes
 * @return DataExample with OBB data
 */
inline DataExample createOBBDataExample(
    int batchSize = 1,
    int imageSize = 640,
    int numTargets = 5,
    int numClasses = 15)
{
    DataExample example;

    // Image data: [B, C, H, W]
    example.data = torch::rand({batchSize, 3, imageSize, imageSize});

    // Targets: [N, 7] - [batch_idx, class_id, x, y, w, h, angle]
    if (numTargets > 0) {
        auto targets = torch::zeros({numTargets, 7});

        // Batch indices
        targets.index({torch::indexing::Slice(), 0}) =
            torch::randint(0, batchSize, {numTargets}).to(torch::kFloat32);

        // Class IDs
        targets.index({torch::indexing::Slice(), 1}) =
            torch::randint(0, numClasses, {numTargets}).to(torch::kFloat32);

        // Box coordinates (x_center, y_center)
        targets.index({torch::indexing::Slice(), torch::indexing::Slice(2, 4)}) =
            torch::rand({numTargets, 2}) * 0.8f + 0.1f;

        // Box dimensions (width, height)
        targets.index({torch::indexing::Slice(), torch::indexing::Slice(4, 6)}) =
            torch::rand({numTargets, 2}) * 0.25f + 0.05f;

        // Angle (in radians, [-pi/2, pi/2])
        targets.index({torch::indexing::Slice(), 6}) =
            torch::rand({numTargets}) * M_PI - M_PI / 2;

        example.targets = targets;
        example.classes = targets.index({torch::indexing::Slice(), 1}).to(torch::kInt64);
        example.batchIndices = targets.index({torch::indexing::Slice(), 0}).to(torch::kInt64);
    }
    else {
        example.targets = torch::zeros({0, 7});
        example.classes = torch::zeros({0}, torch::kInt64);
        example.batchIndices = torch::zeros({0}, torch::kInt64);
    }

    return example;
}

/**
 * @brief Create a mock DataExample for segmentation testing
 *
 * Format:
 * - targets: [N, 6+P*2] where 6 = [batch_idx, class_id, x, y, w, h] and P*2 = polygon points
 *
 * @param batchSize Batch size
 * @param imageSize Image size (square)
 * @param numTargets Total number of targets
 * @param numClasses Number of classes
 * @param numMaskPoints Number of polygon mask points per target
 * @return DataExample with segmentation data
 */
inline DataExample createSegmentationDataExample(
    int batchSize = 1,
    int imageSize = 640,
    int numTargets = 5,
    int numClasses = 80,
    int numMaskPoints = 32)
{
    DataExample example;

    // Image data: [B, C, H, W]
    example.data = torch::rand({batchSize, 3, imageSize, imageSize});

    // Targets: [N, 6 + numMaskPoints*2]
    int targetWidth = 6 + numMaskPoints * 2;
    if (numTargets > 0) {
        auto targets = torch::zeros({numTargets, targetWidth});

        // Batch indices
        targets.index({torch::indexing::Slice(), 0}) =
            torch::randint(0, batchSize, {numTargets}).to(torch::kFloat32);

        // Class IDs
        targets.index({torch::indexing::Slice(), 1}) =
            torch::randint(0, numClasses, {numTargets}).to(torch::kFloat32);

        // Box coordinates
        targets.index({torch::indexing::Slice(), torch::indexing::Slice(2, 4)}) =
            torch::rand({numTargets, 2}) * 0.8f + 0.1f;

        // Box dimensions
        targets.index({torch::indexing::Slice(), torch::indexing::Slice(4, 6)}) =
            torch::rand({numTargets, 2}) * 0.25f + 0.05f;

        // Mask polygon points (normalized coordinates)
        targets.index({torch::indexing::Slice(), torch::indexing::Slice(6, targetWidth)}) =
            torch::rand({numTargets, numMaskPoints * 2});

        example.targets = targets;
        example.classes = targets.index({torch::indexing::Slice(), 1}).to(torch::kInt64);
        example.batchIndices = targets.index({torch::indexing::Slice(), 0}).to(torch::kInt64);
    }
    else {
        example.targets = torch::zeros({0, targetWidth});
        example.classes = torch::zeros({0}, torch::kInt64);
        example.batchIndices = torch::zeros({0}, torch::kInt64);
    }

    return example;
}

/**
 * @brief Create a mock DataExample for anomaly detection testing
 *
 * Anomaly detection typically only has normal images during training.
 *
 * @param batchSize Batch size
 * @param imageSize Image size (square, typically 256)
 * @param isAnomalous Whether to mark as anomalous (for validation)
 * @return DataExample with anomaly data
 */
inline DataExample createAnomalyDataExample(
    int batchSize = 1,
    int imageSize = 256,
    bool isAnomalous = false)
{
    DataExample example;

    // Image data: [B, C, H, W]
    example.data = torch::rand({batchSize, 3, imageSize, imageSize});

    // Classes: 0 = normal, 1 = anomalous
    example.classes = torch::full({batchSize}, isAnomalous ? 1 : 0, torch::kInt64);

    // Targets: same as classes for anomaly detection
    example.targets = example.classes.clone();

    // Batch indices
    example.batchIndices = torch::arange(batchSize);

    return example;
}

/**
 * @brief Create a mock DataExample with specific tensor values (for edge case testing)
 *
 * @param data Pre-created data tensor
 * @param classes Pre-created classes tensor
 * @param targets Pre-created targets tensor
 * @param batchIndices Pre-created batch indices tensor
 * @return DataExample with specified tensors
 */
inline DataExample createCustomDataExample(
    const torch::Tensor& data,
    const torch::Tensor& classes = torch::Tensor(),
    const torch::Tensor& targets = torch::Tensor(),
    const torch::Tensor& batchIndices = torch::Tensor())
{
    DataExample example;
    example.data = data;
    example.classes = classes;
    example.targets = targets;
    example.batchIndices = batchIndices;
    return example;
}

/**
 * @brief Create an empty DataExample (for error testing)
 *
 * @return Empty DataExample with undefined tensors
 */
inline DataExample createEmptyDataExample() {
    return DataExample();
}

/**
 * @brief Create a DataExample with zero-sized tensors
 *
 * @param imageShape Shape for the image tensor
 * @return DataExample with zero-sized target tensors
 */
inline DataExample createZeroTargetDataExample(const std::vector<int64_t>& imageShape) {
    DataExample example;
    example.data = torch::rand(imageShape);
    example.classes = torch::zeros({0}, torch::kInt64);
    example.targets = torch::zeros({0, 6});
    example.batchIndices = torch::zeros({0}, torch::kInt64);
    return example;
}

} // namespace Test
} // namespace WheelDL
