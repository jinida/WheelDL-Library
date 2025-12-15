#include "pch.h"
#include "Data/Transforms/Collation.h"
#include "Data/Dataset/BaseDataset.h"
#include "Data/Dataset/PredDataset.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <tuple>

using namespace WheelDL::Data;
using namespace WheelDL::Data::Dataset;

// =============================================================================
// Test Fixture
// =============================================================================

class CollationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper: Create DataExample with specified dimensions
    DataExample createDataExample(int c = 3, int h = 64, int w = 64, int numObjects = 2, int targetDim = 2) {
        DataExample example;
        example.data = torch::randn({ c, h, w });

        // Create classes tensor with numObjects elements
        std::vector<int64_t> classIds(numObjects);
        for (int i = 0; i < numObjects; ++i) {
            classIds[i] = i % 10;  // Class IDs 0-9
        }
        example.classes = torch::tensor(classIds, torch::kLong);

        if (targetDim == 2) {
            // 2D targets: [numObjects, 4] for detection
            example.targets = torch::randn({ numObjects, 4 });
        }
        else if (targetDim == 3) {
            // 3D targets: [numObjects, H, W] for segmentation masks
            example.targets = torch::randn({ numObjects, 32, 32 });
        }
        else {
            example.targets = torch::Tensor();
        }

        return example;
    }

    // Helper: Create DataExample with empty targets
    DataExample createEmptyTargetExample(int c = 3, int h = 64, int w = 64) {
        DataExample example;
        example.data = torch::randn({ c, h, w });
        example.classes = torch::tensor({ 0 }, torch::kLong);
        example.targets = torch::Tensor();  // Undefined tensor
        return example;
    }

    // Helper: Create PredDataExample
    PredDataExample createPredDataExample(int c = 3, int h = 64, int w = 64,
                                          const std::string& path = "test.jpg",
                                          int origH = 480, int origW = 640) {
        PredDataExample example;
        example.data = torch::randn({ c, h, w });
        example.imagePath = { path };
        example.originalShape = { std::make_tuple(origH, origW) };
        return example;
    }
};

// =============================================================================
// 11.1 DataExampleCollation Tests (COL-001 ~ COL-010)
// =============================================================================

// COL-001: apply_batch single example
TEST_F(CollationTest, DataExampleCollation_SingleExample) {
    DataExampleCollation collation;
    std::vector<DataExample> examples;
    examples.push_back(createDataExample(3, 64, 64, 2, 2));

    DataExample result = collation.apply_batch(std::move(examples));

    EXPECT_TRUE(result.data.defined());
    EXPECT_EQ(4, result.data.dim());  // [batch, C, H, W]
    EXPECT_EQ(1, result.data.size(0));  // batch size = 1
    EXPECT_EQ(3, result.data.size(1));  // channels
    EXPECT_EQ(64, result.data.size(2)); // height
    EXPECT_EQ(64, result.data.size(3)); // width
}

// COL-002: apply_batch multiple examples
TEST_F(CollationTest, DataExampleCollation_MultipleExamples) {
    DataExampleCollation collation;
    std::vector<DataExample> examples;
    examples.push_back(createDataExample(3, 64, 64, 2, 2));
    examples.push_back(createDataExample(3, 64, 64, 3, 2));
    examples.push_back(createDataExample(3, 64, 64, 1, 2));

    DataExample result = collation.apply_batch(std::move(examples));

    EXPECT_TRUE(result.data.defined());
    EXPECT_EQ(3, result.data.size(0));  // batch size = 3

    // Classes should be concatenated: 2 + 3 + 1 = 6
    EXPECT_TRUE(result.classes.defined());
    EXPECT_EQ(6, result.classes.size(0));

    // Targets should be concatenated for 2D: 2 + 3 + 1 = 6
    EXPECT_TRUE(result.targets.defined());
    EXPECT_EQ(6, result.targets.size(0));
}

// COL-003: apply_batch empty targets (0 objects per image)
TEST_F(CollationTest, DataExampleCollation_EmptyTargets) {
    DataExampleCollation collation;
    std::vector<DataExample> examples;

    // Create examples with empty (0 size) targets - valid tensors with no objects
    DataExample ex1;
    ex1.data = torch::randn({ 3, 64, 64 });
    ex1.classes = torch::empty({ 0 }, torch::kLong);  // No classes
    ex1.targets = torch::empty({ 0, 4 });  // Empty 2D targets (0 objects, 4 coords)

    DataExample ex2;
    ex2.data = torch::randn({ 3, 64, 64 });
    ex2.classes = torch::empty({ 0 }, torch::kLong);
    ex2.targets = torch::empty({ 0, 4 });

    examples.push_back(std::move(ex1));
    examples.push_back(std::move(ex2));

    DataExample result = collation.apply_batch(std::move(examples));

    EXPECT_TRUE(result.data.defined());
    EXPECT_EQ(2, result.data.size(0));  // Batch size = 2
    EXPECT_TRUE(result.targets.defined());  // Should be defined but empty
    EXPECT_EQ(0, result.targets.size(0));  // No objects concatenated
}

// COL-004: apply_batch 3D targets
TEST_F(CollationTest, DataExampleCollation_3DTargets) {
    DataExampleCollation collation;
    std::vector<DataExample> examples;
    examples.push_back(createDataExample(3, 64, 64, 2, 3));  // 3D targets
    examples.push_back(createDataExample(3, 64, 64, 2, 3));

    DataExample result = collation.apply_batch(std::move(examples));

    EXPECT_TRUE(result.targets.defined());
    EXPECT_EQ(4, result.targets.dim());  // [batch, numObjects, H, W]
    EXPECT_EQ(2, result.targets.size(0));  // batch size
    EXPECT_FALSE(result.batchIndices.defined());  // No batch indices for 3D
}

// COL-005: apply_batch 2D targets
TEST_F(CollationTest, DataExampleCollation_2DTargets) {
    DataExampleCollation collation;
    std::vector<DataExample> examples;
    examples.push_back(createDataExample(3, 64, 64, 2, 2));  // 2D targets
    examples.push_back(createDataExample(3, 64, 64, 3, 2));

    DataExample result = collation.apply_batch(std::move(examples));

    EXPECT_TRUE(result.targets.defined());
    EXPECT_EQ(2, result.targets.dim());  // [total_objects, 4]
    EXPECT_EQ(5, result.targets.size(0));  // 2 + 3 = 5 objects
    EXPECT_TRUE(result.batchIndices.defined());  // Batch indices created
}

// COL-006: apply_batch batchIndices creation
TEST_F(CollationTest, DataExampleCollation_BatchIndicesCreation) {
    DataExampleCollation collation;
    std::vector<DataExample> examples;

    // First example: 2 objects
    examples.push_back(createDataExample(3, 64, 64, 2, 2));
    // Second example: 3 objects
    examples.push_back(createDataExample(3, 64, 64, 3, 2));

    DataExample result = collation.apply_batch(std::move(examples));

    EXPECT_TRUE(result.batchIndices.defined());
    EXPECT_EQ(5, result.batchIndices.size(0));  // 2 + 3 = 5 indices

    // First 2 should be 0, next 3 should be 1
    auto indices = result.batchIndices.accessor<int64_t, 1>();
    EXPECT_EQ(0, indices[0]);
    EXPECT_EQ(0, indices[1]);
    EXPECT_EQ(1, indices[2]);
    EXPECT_EQ(1, indices[3]);
    EXPECT_EQ(1, indices[4]);
}

// COL-007: apply_batch data stacking
TEST_F(CollationTest, DataExampleCollation_DataStacking) {
    DataExampleCollation collation;
    std::vector<DataExample> examples;

    // Create examples with known values
    DataExample ex1;
    ex1.data = torch::ones({ 3, 2, 2 });
    ex1.classes = torch::tensor({ 0 }, torch::kLong);
    ex1.targets = torch::randn({ 1, 4 });

    DataExample ex2;
    ex2.data = torch::zeros({ 3, 2, 2 });
    ex2.classes = torch::tensor({ 1 }, torch::kLong);
    ex2.targets = torch::randn({ 1, 4 });

    examples.push_back(std::move(ex1));
    examples.push_back(std::move(ex2));

    DataExample result = collation.apply_batch(std::move(examples));

    // First batch should be all ones
    EXPECT_FLOAT_EQ(1.0f, result.data[0][0][0][0].item<float>());
    // Second batch should be all zeros
    EXPECT_FLOAT_EQ(0.0f, result.data[1][0][0][0].item<float>());
}

// COL-008: apply_batch classes concatenation
TEST_F(CollationTest, DataExampleCollation_ClassesConcatenation) {
    DataExampleCollation collation;
    std::vector<DataExample> examples;

    DataExample ex1;
    ex1.data = torch::randn({ 3, 64, 64 });
    ex1.classes = torch::tensor({ 0, 1 }, torch::kLong);
    ex1.targets = torch::randn({ 2, 4 });

    DataExample ex2;
    ex2.data = torch::randn({ 3, 64, 64 });
    ex2.classes = torch::tensor({ 2, 3, 4 }, torch::kLong);
    ex2.targets = torch::randn({ 3, 4 });

    examples.push_back(std::move(ex1));
    examples.push_back(std::move(ex2));

    DataExample result = collation.apply_batch(std::move(examples));

    EXPECT_EQ(5, result.classes.size(0));  // 2 + 3 = 5

    auto classes = result.classes.accessor<int64_t, 1>();
    EXPECT_EQ(0, classes[0]);
    EXPECT_EQ(1, classes[1]);
    EXPECT_EQ(2, classes[2]);
    EXPECT_EQ(3, classes[3]);
    EXPECT_EQ(4, classes[4]);
}

// COL-009: apply_batch move semantics
TEST_F(CollationTest, DataExampleCollation_MoveSemantics) {
    DataExampleCollation collation;
    std::vector<DataExample> examples;

    DataExample ex = createDataExample(3, 64, 64, 2, 2);
    torch::Tensor originalData = ex.data.clone();

    examples.push_back(std::move(ex));

    DataExample result = collation.apply_batch(std::move(examples));

    // Result should have valid data
    EXPECT_TRUE(result.data.defined());
    EXPECT_EQ(1, result.data.size(0));
}

// COL-010: apply_batch with mixed object counts
TEST_F(CollationTest, DataExampleCollation_MixedObjectCounts) {
    DataExampleCollation collation;
    std::vector<DataExample> examples;

    // Example 1: 2 objects
    DataExample ex1;
    ex1.data = torch::randn({ 3, 64, 64 });
    ex1.classes = torch::tensor({ 0, 1 }, torch::kLong);
    ex1.targets = torch::randn({ 2, 4 });

    // Example 2: 0 objects (empty but valid)
    DataExample ex2;
    ex2.data = torch::randn({ 3, 64, 64 });
    ex2.classes = torch::empty({ 0 }, torch::kLong);
    ex2.targets = torch::empty({ 0, 4 });

    // Example 3: 1 object
    DataExample ex3;
    ex3.data = torch::randn({ 3, 64, 64 });
    ex3.classes = torch::tensor({ 2 }, torch::kLong);
    ex3.targets = torch::randn({ 1, 4 });

    examples.push_back(std::move(ex1));
    examples.push_back(std::move(ex2));
    examples.push_back(std::move(ex3));

    DataExample result = collation.apply_batch(std::move(examples));

    EXPECT_TRUE(result.data.defined());
    EXPECT_EQ(3, result.data.size(0));  // Batch size = 3
    EXPECT_TRUE(result.targets.defined());
    EXPECT_EQ(3, result.targets.size(0));  // 2 + 0 + 1 = 3 objects

    // Verify batch indices: [0, 0, 2] (no index for empty batch 1)
    EXPECT_TRUE(result.batchIndices.defined());
    EXPECT_EQ(3, result.batchIndices.size(0));
}

// =============================================================================
// 11.2 PredDataExampleCollation Tests (COL-011 ~ COL-016)
// =============================================================================

// COL-011: apply_batch single example
TEST_F(CollationTest, PredDataExampleCollation_SingleExample) {
    PredDataExampleCollation collation;
    std::vector<PredDataExample> examples;
    examples.push_back(createPredDataExample(3, 64, 64, "test1.jpg", 480, 640));

    PredDataExample result = collation.apply_batch(std::move(examples));

    EXPECT_TRUE(result.data.defined());
    EXPECT_EQ(4, result.data.dim());  // [batch, C, H, W]
    EXPECT_EQ(1, result.data.size(0));  // batch size = 1
    EXPECT_EQ(1u, result.imagePath.size());
    EXPECT_EQ("test1.jpg", result.imagePath[0]);
    EXPECT_EQ(1u, result.originalShape.size());
}

// COL-012: apply_batch multiple examples
TEST_F(CollationTest, PredDataExampleCollation_MultipleExamples) {
    PredDataExampleCollation collation;
    std::vector<PredDataExample> examples;
    examples.push_back(createPredDataExample(3, 64, 64, "test1.jpg", 480, 640));
    examples.push_back(createPredDataExample(3, 64, 64, "test2.jpg", 720, 1280));
    examples.push_back(createPredDataExample(3, 64, 64, "test3.jpg", 1080, 1920));

    PredDataExample result = collation.apply_batch(std::move(examples));

    EXPECT_TRUE(result.data.defined());
    EXPECT_EQ(3, result.data.size(0));  // batch size = 3
    EXPECT_EQ(3u, result.imagePath.size());
    EXPECT_EQ(3u, result.originalShape.size());
}

// COL-013: apply_batch imagePaths collection
TEST_F(CollationTest, PredDataExampleCollation_ImagePathsCollection) {
    PredDataExampleCollation collation;
    std::vector<PredDataExample> examples;
    examples.push_back(createPredDataExample(3, 64, 64, "path/to/image1.jpg", 480, 640));
    examples.push_back(createPredDataExample(3, 64, 64, "path/to/image2.png", 720, 1280));

    PredDataExample result = collation.apply_batch(std::move(examples));

    EXPECT_EQ(2u, result.imagePath.size());
    EXPECT_EQ("path/to/image1.jpg", result.imagePath[0]);
    EXPECT_EQ("path/to/image2.png", result.imagePath[1]);
}

// COL-014: apply_batch originalShapes collection
TEST_F(CollationTest, PredDataExampleCollation_OriginalShapesCollection) {
    PredDataExampleCollation collation;
    std::vector<PredDataExample> examples;
    examples.push_back(createPredDataExample(3, 64, 64, "test1.jpg", 480, 640));
    examples.push_back(createPredDataExample(3, 64, 64, "test2.jpg", 720, 1280));

    PredDataExample result = collation.apply_batch(std::move(examples));

    EXPECT_EQ(2u, result.originalShape.size());

    auto [h1, w1] = result.originalShape[0];
    EXPECT_EQ(480, h1);
    EXPECT_EQ(640, w1);

    auto [h2, w2] = result.originalShape[1];
    EXPECT_EQ(720, h2);
    EXPECT_EQ(1280, w2);
}

// COL-015: apply_batch data stacking
TEST_F(CollationTest, PredDataExampleCollation_DataStacking) {
    PredDataExampleCollation collation;
    std::vector<PredDataExample> examples;

    PredDataExample ex1;
    ex1.data = torch::ones({ 3, 2, 2 });
    ex1.imagePath = { "img1.jpg" };
    ex1.originalShape = { std::make_tuple(100, 100) };

    PredDataExample ex2;
    ex2.data = torch::zeros({ 3, 2, 2 });
    ex2.imagePath = { "img2.jpg" };
    ex2.originalShape = { std::make_tuple(200, 200) };

    examples.push_back(std::move(ex1));
    examples.push_back(std::move(ex2));

    PredDataExample result = collation.apply_batch(std::move(examples));

    EXPECT_EQ(4, result.data.dim());  // [batch, C, H, W]
    EXPECT_EQ(2, result.data.size(0));  // batch size

    // First batch should be all ones
    EXPECT_FLOAT_EQ(1.0f, result.data[0][0][0][0].item<float>());
    // Second batch should be all zeros
    EXPECT_FLOAT_EQ(0.0f, result.data[1][0][0][0].item<float>());
}

// COL-016: apply_batch move semantics
TEST_F(CollationTest, PredDataExampleCollation_MoveSemantics) {
    PredDataExampleCollation collation;
    std::vector<PredDataExample> examples;

    PredDataExample ex = createPredDataExample(3, 64, 64, "original.jpg", 480, 640);

    examples.push_back(std::move(ex));

    PredDataExample result = collation.apply_batch(std::move(examples));

    // Result should have valid data
    EXPECT_TRUE(result.data.defined());
    EXPECT_EQ(1, result.data.size(0));
    EXPECT_EQ(1u, result.imagePath.size());
    EXPECT_EQ("original.jpg", result.imagePath[0]);
}
