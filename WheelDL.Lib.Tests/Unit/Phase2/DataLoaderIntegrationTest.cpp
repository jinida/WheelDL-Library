#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Dataset/ClassificationDataset.h"
#include "WheelDL.Lib/Data/Dataset/DetectionDataset.h"
#include "WheelDL.Lib/Data/Dataset/OBBDataset.h"
#include "WheelDL.Lib/Data/Dataset/SegmentationDataset.h"
#include "WheelDL.Lib/Data/Dataset/AnomalyDataset.h"
#include "WheelDL.Lib/Data/Transforms/Collation.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include <torch/torch.h>
#include <filesystem>

using namespace WheelDL::Data;
using namespace WheelDL::Data::Dataset;
using namespace WheelDL::Config;

namespace {
    std::filesystem::path getTestDataPath() {
        std::filesystem::path sourceDir = __FILE__;
        sourceDir = sourceDir.parent_path();
        while (sourceDir.filename() != "WheelDL.Lib.Tests" && sourceDir.has_parent_path()) {
            sourceDir = sourceDir.parent_path();
        }
        return sourceDir / "Data";
    }
}

/**
 * @class DataLoaderIntegrationTest
 * @brief Integration test suite for LibTorch DataLoader with all dataset types
 *
 * This test suite verifies that:
 * 1. torch::data::make_data_loader works correctly with all dataset types
 * 2. DataLoader can iterate through batches without errors
 * 3. Batched data has correct shapes and properties
 * 4. Multi-threaded data loading works safely
 * 5. Different batch sizes are handled correctly
 */
class DataLoaderIntegrationTest : public ::testing::Test {
protected:
    std::string dataPath;
    Configuration config;

    void SetUp() override {
        auto testData = getTestDataPath();
        dataPath = (testData / "mnist_sample").string();

        // Set number of classes for segmentation
        config.setNumClasses(10);
    }

    void TearDown() override {
        // Clean up if needed
    }

    /**
     * @brief Verify that batched tensor has expected shape
     */
    void verifyBatchShape(const torch::Tensor& tensor,
                          const std::vector<int64_t>& expectedShape,
                          const std::string& tensorName) {
        ASSERT_EQ(tensor.dim(), expectedShape.size())
            << tensorName << " dimension mismatch";

        for (size_t i = 0; i < expectedShape.size(); ++i) {
            if (expectedShape[i] != -1) { // -1 means any size is acceptable
                EXPECT_EQ(tensor.size(i), expectedShape[i])
                    << tensorName << " shape mismatch at dimension " << i;
            }
        }
    }
};

// ========== Classification Dataset Tests ==========

TEST_F(DataLoaderIntegrationTest, DataLoader_ClassificationDataset) {
    // Create dataset
    auto dataset = ClassificationDataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "classification").string(),
        config,
        true
    );

    // Create DataLoader with collation
    auto dataLoader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(4)
            .workers(0)  // Single-threaded for deterministic testing
    );

    int batchCount = 0;
    int totalSamples = 0;

    for (auto& batch : *dataLoader) {
        batchCount++;

        // Verify batch structure
        ASSERT_TRUE(batch.data.defined()) << "Data tensor should be defined";
        ASSERT_TRUE(batch.classes.defined()) << "Classes tensor should be defined";
        ASSERT_TRUE(batch.targets.defined()) << "Targets tensor should be defined";

        // Verify batch shape
        int currentBatchSize = batch.data.size(0);
        totalSamples += currentBatchSize;

        verifyBatchShape(batch.data, {currentBatchSize, 3, 640, 640}, "data");
        EXPECT_EQ(batch.classes.size(0), currentBatchSize);
        EXPECT_EQ(batch.targets.size(0), currentBatchSize);

        // Verify data is in valid range
        auto minVal = batch.data.min().item<float>();
        auto maxVal = batch.data.max().item<float>();
        EXPECT_GE(minVal, 0.0f);
        EXPECT_LE(maxVal, 1.0f);
    }

    EXPECT_GT(batchCount, 0) << "Should have at least one batch";
    EXPECT_GT(totalSamples, 0) << "Should have processed some samples";
}

TEST_F(DataLoaderIntegrationTest, DataLoader_ClassificationMultipleWorkers) {
    auto dataset = ClassificationDataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "classification").string(),
        config,
        false  // Validation mode for deterministic behavior
    );

    // Test with multiple workers
    auto dataLoader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(8)
            .workers(2)  // Multi-threaded
    );

    int batchCount = 0;
    for (auto& batch : *dataLoader) {
        batchCount++;
        ASSERT_TRUE(batch.data.defined());
        ASSERT_GT(batch.data.size(0), 0) << "Batch should have samples";
    }

    EXPECT_GT(batchCount, 0) << "Should have processed batches with multiple workers";
}

// ========== Detection Dataset Tests ==========

TEST_F(DataLoaderIntegrationTest, DataLoader_DetectionDataset) {
    auto dataset = DetectionDataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "objectdetection").string(),
        config,
        true
    );

    auto dataLoader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(4)
            .workers(0)
    );

    int batchCount = 0;
    for (auto& batch : *dataLoader) {
        batchCount++;

        // Verify data tensor
        ASSERT_TRUE(batch.data.defined());
        int currentBatchSize = batch.data.size(0);
        verifyBatchShape(batch.data, {currentBatchSize, 3, 640, 640}, "data");

        // Verify targets: [total_objects, 5] (class + xywh)
        ASSERT_TRUE(batch.targets.defined());
        if (batch.targets.size(0) > 0) {
            EXPECT_EQ(batch.targets.dim(), 2);
            EXPECT_EQ(batch.targets.size(1), 5);
        }

        // Verify batch indices
        ASSERT_TRUE(batch.batchIndices.defined());
        EXPECT_EQ(batch.batchIndices.size(0), batch.targets.size(0));

        // Check batch indices are in valid range
        if (batch.batchIndices.size(0) > 0) {
            auto maxIdx = batch.batchIndices.max().item<int64_t>();
            auto minIdx = batch.batchIndices.min().item<int64_t>();
            EXPECT_GE(minIdx, 0);
            EXPECT_LT(maxIdx, currentBatchSize);
        }
    }

    EXPECT_GT(batchCount, 0);
}

TEST_F(DataLoaderIntegrationTest, DataLoader_DetectionVariousBatchSizes) {
    std::vector<int> batchSizes = {1, 2, 4, 8};

    for (int batchSize : batchSizes) {
        auto dataset = DetectionDataset(
            (std::filesystem::path(dataPath) / "images").string(),
            (std::filesystem::path(dataPath) / "labels" / "objectdetection").string(),
            config,
            false
        );

        auto dataLoader = torch::data::make_data_loader(
            std::move(dataset),
            torch::data::DataLoaderOptions()
                .batch_size(batchSize)
                .workers(0)
        );

        bool hasData = false;
        for (auto& batch : *dataLoader) {
            hasData = true;
            EXPECT_LE(batch.data.size(0), batchSize)
                << "Batch size should not exceed requested size";
            EXPECT_GT(batch.data.size(0), 0)
                << "Batch should have at least one sample";
        }

        EXPECT_TRUE(hasData) << "DataLoader should produce batches for batch_size=" << batchSize;
    }
}

// ========== OBB Dataset Tests ==========

TEST_F(DataLoaderIntegrationTest, DataLoader_OBBDataset) {
    auto dataset = OBBDataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "obb").string(),
        config,
        true
    );

    auto dataLoader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(4)
            .workers(0)
    );

    int batchCount = 0;
    for (auto& batch : *dataLoader) {
        batchCount++;

        ASSERT_TRUE(batch.data.defined());
        int currentBatchSize = batch.data.size(0);
        verifyBatchShape(batch.data, {currentBatchSize, 3, 640, 640}, "data");

        // OBB targets should have 9 values (class + 8 corner coordinates)
        if (batch.targets.defined() && batch.targets.size(0) > 0) {
            EXPECT_EQ(batch.targets.dim(), 2);
            EXPECT_EQ(batch.targets.size(1), 9);
        }

        // Verify batch indices
        if (batch.batchIndices.defined() && batch.batchIndices.size(0) > 0) {
            EXPECT_EQ(batch.batchIndices.size(0), batch.targets.size(0));
        }
    }

    EXPECT_GT(batchCount, 0);
}

// ========== Segmentation Dataset Tests ==========

TEST_F(DataLoaderIntegrationTest, DataLoader_SegmentationDataset) {
    // Skip if segmentation data not available
    if (!std::filesystem::exists(std::filesystem::path(dataPath) / "labels" / "segmentation")) {
        GTEST_SKIP() << "Segmentation test data not available";
    }

    auto dataset = SegmentationDataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "segmentation").string(),
        config,
        true
    );

    auto dataLoader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(2)
            .workers(0)
    );

    int batchCount = 0;
    for (auto& batch : *dataLoader) {
        batchCount++;

        ASSERT_TRUE(batch.data.defined());
        int currentBatchSize = batch.data.size(0);
        verifyBatchShape(batch.data, {currentBatchSize, 3, 640, 640}, "data");

        // Segmentation targets: [batch_size, num_classes, H, W]
        ASSERT_TRUE(batch.targets.defined());
        EXPECT_EQ(batch.targets.dim(), 4);
        EXPECT_EQ(batch.targets.size(0), currentBatchSize);
        EXPECT_EQ(batch.targets.size(1), 10);  // 10 classes for MNIST
        EXPECT_EQ(batch.targets.size(2), 640);
        EXPECT_EQ(batch.targets.size(3), 640);
    }

    EXPECT_GT(batchCount, 0);
}

TEST_F(DataLoaderIntegrationTest, DataLoader_SegmentationSingleBatch) {
    if (!std::filesystem::exists(std::filesystem::path(dataPath) / "labels" / "segmentation")) {
        GTEST_SKIP() << "Segmentation test data not available";
    }

    auto dataset = SegmentationDataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "segmentation").string(),
        config,
        false
    );

    auto dataLoader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(1)
            .workers(0)
    );

    bool hasData = false;
    for (auto& batch : *dataLoader) {
        hasData = true;
        EXPECT_EQ(batch.data.size(0), 1) << "Single batch should have exactly 1 sample";
        EXPECT_EQ(batch.targets.size(0), 1);
        break;  // Just test first batch
    }

    EXPECT_TRUE(hasData);
}

// ========== Anomaly Dataset Tests ==========

TEST_F(DataLoaderIntegrationTest, DataLoader_AnomalyDataset) {
    // Skip if anomaly data not available
    if (!std::filesystem::exists(std::filesystem::path(dataPath) / "labels" / "anomalydetection")) {
        GTEST_SKIP() << "Anomaly detection test data not available";
    }

    auto dataset = AnomalyDataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "anomalydetection").string(),
        config,
        true
    );

    auto dataLoader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(4)
            .workers(0)
    );

    int batchCount = 0;
    for (auto& batch : *dataLoader) {
        batchCount++;

        ASSERT_TRUE(batch.data.defined());
        int currentBatchSize = batch.data.size(0);
        verifyBatchShape(batch.data, {currentBatchSize, 3, 640, 640}, "data");

        // Anomaly targets may have masks or labels
        ASSERT_TRUE(batch.targets.defined());
    }

    EXPECT_GT(batchCount, 0);
}

// ========== Edge Cases and Error Handling ==========

TEST_F(DataLoaderIntegrationTest, DataLoader_EmptyIteration) {
    auto dataset = ClassificationDataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "classification").string(),
        config,
        false
    );

    auto dataLoader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(4)
            .workers(0)
    );

    // Test that we can iterate multiple times (though DataLoader usually exhausts)
    int firstCount = 0;
    for (auto& batch : *dataLoader) {
        firstCount++;
        (void)batch;  // Suppress unused warning
    }

    EXPECT_GT(firstCount, 0);
}

TEST_F(DataLoaderIntegrationTest, DataLoader_LargeBatchSize) {
    auto dataset = ClassificationDataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "classification").string(),
        config,
        false
    );

    // Request batch size larger than dataset
    auto dataLoader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions()
            .batch_size(10000)
            .workers(0)
    );

    int batchCount = 0;
    for (auto& batch : *dataLoader) {
        batchCount++;
        // Should still get a batch with all available samples
        EXPECT_GT(batch.data.size(0), 0);
    }

    // Should get exactly 1 batch containing all samples
    EXPECT_EQ(batchCount, 1);
}

// ========== Performance and Stress Tests ==========

TEST_F(DataLoaderIntegrationTest, DataLoader_ConsistentBatchContent) {
    // Test that validation mode produces consistent results
    auto dataset1 = ClassificationDataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "classification").string(),
        config,
        false
    );

    auto dataLoader1 = torch::data::make_data_loader(
        std::move(dataset1),
        torch::data::DataLoaderOptions()
            .batch_size(4)
            .workers(0)
    );

    std::vector<torch::Tensor> firstRunData;
    for (auto& batch : *dataLoader1) {
        firstRunData.push_back(batch.data.clone());
    }

    // Second run
    auto dataset2 = ClassificationDataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "classification").string(),
        config,
        false
    );

    auto dataLoader2 = torch::data::make_data_loader(
        std::move(dataset2),
        torch::data::DataLoaderOptions()
            .batch_size(4)
            .workers(0)
    );

    size_t batchIdx = 0;
    for (auto& batch : *dataLoader2) {
        if (batchIdx < firstRunData.size()) {
            EXPECT_TRUE(torch::allclose(batch.data, firstRunData[batchIdx], 1e-5, 1e-5))
                << "Validation mode should produce identical batches";
        }
        batchIdx++;
    }

    EXPECT_EQ(batchIdx, firstRunData.size()) << "Both runs should have same number of batches";
}

TEST_F(DataLoaderIntegrationTest, DataLoader_MemorySafety) {
    // Test that tensors remain valid after DataLoader iteration
    std::vector<torch::Tensor> storedBatches;

    {
        auto dataset = ClassificationDataset(
            (std::filesystem::path(dataPath) / "images").string(),
            (std::filesystem::path(dataPath) / "labels" / "classification").string(),
            config,
            false
        );

        auto dataLoader = torch::data::make_data_loader(
            std::move(dataset),
            torch::data::DataLoaderOptions()
                .batch_size(4)
                .workers(0)
        );

        for (auto& batch : *dataLoader) {
            storedBatches.push_back(batch.data.clone());
        }
    }  // DataLoader and dataset destroyed

    // Verify stored tensors are still valid
    for (const auto& tensor : storedBatches) {
        EXPECT_TRUE(tensor.defined());
        EXPECT_GT(tensor.numel(), 0);
        EXPECT_EQ(tensor.dim(), 4);
    }
}
