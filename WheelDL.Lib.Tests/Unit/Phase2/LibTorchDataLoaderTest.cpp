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

	Configuration getTestConfig() {
		Configuration config;
		config.setClassNames(std::map<int, std::string>{
			{0, "zero"},
			{1, "one"},
			{2, "two"},
			{3, "three"},
			{4, "four"},
			{5, "five"},
			{6, "six"},
			{7, "seven"},
			{8, "eight"},
			{9, "nine"}
		});
		return config;
	}
}

class LibTorchDataLoaderTest : public ::testing::Test 
{
protected:
	static std::string dataPath;
	static std::string annotationPath;
	
	static void SetUpTestSuite()
	{
		std::cout << "=== SetUpTestSuite called ===" << std::endl;
		try
		{
			auto testDataPath = getTestDataPath();
			std::cout << "testDataPath: " << testDataPath.string() << std::endl;

			dataPath = (testDataPath / "mnist_sample" / "images").string();
			annotationPath = (testDataPath / "mnist_sample" / "labels").string();

			std::cout << "dataPath set to: " << dataPath << std::endl;
			std::cout << "annotationPath set to: " << annotationPath << std::endl;
		}
		catch (const std::exception& e)
		{
			std::cerr << "SetUpTestSuite failed: " << e.what() << std::endl;
		}
	}
	static void TearDownTestSuite() {
		// Clean up if needed
	}
};

// Static member definition - initialize with actual paths
std::string LibTorchDataLoaderTest::dataPath = []() {
	auto testDataPath = getTestDataPath();
	std::cout << "Initializing dataPath from: " << testDataPath.string() << std::endl;
	return (testDataPath / "mnist_sample" / "images").string();
}();

std::string LibTorchDataLoaderTest::annotationPath = []() {
	auto testDataPath = getTestDataPath();
	std::cout << "Initializing annotationPath from: " << testDataPath.string() << std::endl;
	return (testDataPath / "mnist_sample" / "labels").string();
}();

TEST_F(LibTorchDataLoaderTest, ClassificationDataset_DataLoaderCompatibility)
{
	auto config = getTestConfig();
	std::cout << "dataPath: " << dataPath << std::endl;
	std::cout << "annotationPath: " << annotationPath << std::endl;
	std::string fullAnnotationPath = (std::filesystem::path(annotationPath) / "classification").string();
	std::cout << "fullAnnotationPath: " << fullAnnotationPath << std::endl;
	auto dataLoader = torch::data::make_data_loader(
		ClassificationDataset(
			dataPath,
			fullAnnotationPath,
			config,
			false
		).map(DataExampleCollation()),
		torch::data::DataLoaderOptions().batch_size(4).workers(0)
	);

	// Verify iteration works and produces valid batches
	bool hasData = false;
	int batchCount = 0;
	for (auto& batch : *dataLoader) {
		hasData = true;
		batchCount++;

		// Validate batch structure
		ASSERT_TRUE(batch.data.defined()) << "Batch data not defined";
		ASSERT_TRUE(batch.targets.defined()) << "Batch targets not defined";
		ASSERT_TRUE(batch.classes.defined()) << "Batch classes not defined";

		// Validate dimensions
		EXPECT_EQ(batch.data.dim(), 4) << "Expected 4D data tensor [B, C, H, W]";
		EXPECT_GT(batch.data.size(0), 0) << "Batch size should be > 0";
		EXPECT_EQ(batch.data.size(1), 3) << "Expected 3 channels (RGB)";

		// Validate value range
		auto minVal = batch.data.min().item<float>();
		auto maxVal = batch.data.max().item<float>();
		EXPECT_GE(minVal, 0.0f) << "Min value should be >= 0";
		EXPECT_LE(maxVal, 1.0f) << "Max value should be <= 1";

		std::cout << "ClassificationDataset batch " << batchCount << " validated successfully" << std::endl;
		break; // Only check first batch
	}

	EXPECT_TRUE(hasData) << "DataLoader produced no batches";
}

TEST_F(LibTorchDataLoaderTest, DetectionDataset_DataLoaderCompatibility)
{
	auto config = getTestConfig();
	auto dataLoader = torch::data::make_data_loader(
		DetectionDataset(
			dataPath,
			(std::filesystem::path(annotationPath) / "objectdetection").string(),
			config,
			false
		).map(DataExampleCollation()),
		torch::data::DataLoaderOptions().batch_size(4).workers(0)
	);

	bool hasData = false;
	int batchCount = 0;
	for (auto& batch : *dataLoader) {
		hasData = true;
		batchCount++;

		// Validate batch structure
		ASSERT_TRUE(batch.data.defined()) << "Batch data not defined";
		ASSERT_TRUE(batch.targets.defined()) << "Batch targets not defined";
		ASSERT_TRUE(batch.batchIndices.defined()) << "Batch indices not defined";

		// Validate dimensions
		EXPECT_EQ(batch.data.dim(), 4) << "Expected 4D data tensor [B, C, H, W]";

		// Validate targets format
		if (batch.targets.size(0) > 0) {
			EXPECT_EQ(batch.targets.dim(), 2) << "Expected 2D targets tensor";
			EXPECT_EQ(batch.targets.size(1), 5) << "Expected 5 values per target [class, x, y, w, h]";

			// Validate batch indices
			EXPECT_EQ(batch.batchIndices.size(0), batch.targets.size(0)) << "Batch indices size mismatch";

			// Validate bbox coordinates in [0, 1]
			auto bboxCoords = batch.targets.index({ torch::indexing::Slice(), torch::indexing::Slice(1, 5) });
			auto bboxMin = bboxCoords.min().item<float>();
			auto bboxMax = bboxCoords.max().item<float>();
			EXPECT_GE(bboxMin, 0.0f) << "Bbox coordinates should be >= 0";
			EXPECT_LE(bboxMax, 1.0f) << "Bbox coordinates should be <= 1";
		}

		std::cout << "DetectionDataset batch " << batchCount << " validated successfully" << std::endl;
		break; // Only check first batch
	}

	EXPECT_TRUE(hasData) << "DataLoader produced no batches";
}

TEST_F(LibTorchDataLoaderTest, OBBDataset_DataLoaderCompatibility)
{
	auto config = getTestConfig();
	auto dataLoader = torch::data::make_data_loader(
		OBBDataset(
			dataPath,
			(std::filesystem::path(annotationPath) / "obb").string(),
			config,
			false
		).map(DataExampleCollation()),
		torch::data::DataLoaderOptions().batch_size(2).workers(0)
	);

	bool hasData = false;
	int batchCount = 0;
	for (auto& batch : *dataLoader) {
		hasData = true;
		batchCount++;

		// Validate batch structure
		ASSERT_TRUE(batch.data.defined()) << "Batch data not defined";
		EXPECT_EQ(batch.data.dim(), 4) << "Expected 4D data tensor [B, C, H, W]";

		// Validate OBB targets format
		if (batch.targets.defined() && batch.targets.size(0) > 0) {
			EXPECT_EQ(batch.targets.dim(), 2) << "Expected 2D OBB targets tensor";
			EXPECT_EQ(batch.targets.size(1), 9) << "Expected 9 values per OBB target";
		}

		std::cout << "OBBDataset batch " << batchCount << " validated successfully" << std::endl;
		break; // Only check first batch
	}

	EXPECT_TRUE(hasData) << "DataLoader produced no batches";
}

TEST_F(LibTorchDataLoaderTest, SegmentationDataset_DataLoaderCompatibility)
{
	auto config = getTestConfig();
	auto dataLoader = torch::data::make_data_loader(
		SegmentationDataset(
			dataPath,
			(std::filesystem::path(annotationPath) / "segmentation").string(),
			config,
			false
		).map(DataExampleCollation()),
		torch::data::DataLoaderOptions().batch_size(2).workers(0)
	);

	bool hasData = false;
	int batchCount = 0;
	for (auto& batch : *dataLoader) {
		hasData = true;
		batchCount++;

		// Validate batch structure
		ASSERT_TRUE(batch.data.defined()) << "Batch data not defined";
		ASSERT_TRUE(batch.targets.defined()) << "Batch targets not defined";

		// Validate dimensions
		EXPECT_EQ(batch.data.dim(), 4) << "Expected 4D data tensor [B, C, H, W]";
		EXPECT_EQ(batch.targets.dim(), 4) << "Expected 4D targets tensor [B, num_classes, H, W]";
		EXPECT_EQ(batch.targets.size(1), config.getNumClasses()) << "Targets should have num_classes channels";

		std::cout << "SegmentationDataset batch " << batchCount << " validated successfully" << std::endl;
		break; // Only check first batch
	}

	EXPECT_TRUE(hasData) << "DataLoader produced no batches";
}

TEST_F(LibTorchDataLoaderTest, AnomalyDataset_DataLoaderCompatibility)
{
	auto config = getTestConfig();
	auto dataLoader = torch::data::make_data_loader(
		AnomalyDataset(
			dataPath,
			(std::filesystem::path(annotationPath) / "anomalydetection").string(),
			config,
			false
		).map(DataExampleCollation()),
		torch::data::DataLoaderOptions().batch_size(2).workers(0)
	);

	bool hasData = false;
	int batchCount = 0;
	for (auto& batch : *dataLoader)
	{
		hasData = true;
		batchCount++;

		// Validate batch structure
		ASSERT_TRUE(batch.data.defined()) << "Batch data not defined";
		ASSERT_TRUE(batch.classes.defined()) << "Batch classes not defined";
		ASSERT_TRUE(batch.targets.defined()) << "Batch targets not defined";
		ASSERT_TRUE(batch.batchIndices.defined()) << "Batch indices not defined";

		// Validate dimensions
		EXPECT_EQ(batch.data.dim(), 4) << "Expected 4D data tensor [B, C, H, W]";
		EXPECT_GT(batch.data.size(0), 0) << "Batch size should be > 0";

		// Validate targets format (anomaly labels)
		// Each sample has a single label (0=normal, 1=anomaly)
		// After collation, targets are concatenated: [label1, label2, ...]
		EXPECT_EQ(batch.targets.dim(), 1) << "Expected 1D targets tensor after concatenation";
		EXPECT_EQ(batch.targets.size(0), batch.data.size(0)) << "Number of labels should match batch size";

		// Validate batch indices match targets
		EXPECT_EQ(batch.batchIndices.dim(), 1) << "Expected 1D batch indices tensor";
		EXPECT_EQ(batch.batchIndices.size(0), batch.targets.size(0)) << "Batch indices size should match targets";

		auto minLabel = batch.targets.min().item<int>();
		auto maxLabel = batch.targets.max().item<int>();
		EXPECT_GE(minLabel, 0) << "Minimum label should be >= 0";
		EXPECT_LE(maxLabel, 1) << "Maximum label should be <= 1";
		break; // Only check first batch
	}

	EXPECT_TRUE(hasData) << "DataLoader produced no batches";
}

TEST_F(LibTorchDataLoaderTest, SequentialSampler_Compatibility)
{
	auto config = getTestConfig();
	size_t datasetSize = 0;
	{
		ClassificationDataset tempDataset(
			dataPath,
			(std::filesystem::path(annotationPath) / "classification").string(),
			config,
			false
		);
		datasetSize = tempDataset.size().value();
	}

	auto dataLoader = torch::data::make_data_loader(
		ClassificationDataset(
			dataPath,
			(std::filesystem::path(annotationPath) / "classification").string(),
			config,
			false
		).map(DataExampleCollation()),
		torch::data::samplers::SequentialSampler(datasetSize),
		torch::data::DataLoaderOptions().batch_size(4).workers(0)
	);

	bool hasData = false;
	for (auto& batch : *dataLoader) {
		hasData = true;
		ASSERT_TRUE(batch.data.defined()) << "Batch data not defined";
		std::cout << "SequentialSampler batch validated successfully" << std::endl;
		break;
	}

	EXPECT_TRUE(hasData) << "SequentialSampler produced no batches";
}

TEST_F(LibTorchDataLoaderTest, RandomSampler_Compatibility)
{
	auto config = getTestConfig();
	size_t datasetSize = 0;
	{
		DetectionDataset tempDataset(
			dataPath,
			(std::filesystem::path(annotationPath) / "objectdetection").string(),
			config,
			false
		);
		datasetSize = tempDataset.size().value();
	}

	auto dataLoader = torch::data::make_data_loader(
		DetectionDataset(
			dataPath,
			(std::filesystem::path(annotationPath) / "objectdetection").string(),
			config,
			false
		).map(DataExampleCollation()),
		torch::data::samplers::RandomSampler(datasetSize),
		torch::data::DataLoaderOptions().batch_size(4).workers(0)
	);

	bool hasData = false;
	for (auto& batch : *dataLoader) {
		hasData = true;
		ASSERT_TRUE(batch.data.defined()) << "Batch data not defined";
		std::cout << "RandomSampler batch validated successfully" << std::endl;
		break;
	}

	EXPECT_TRUE(hasData) << "RandomSampler produced no batches";
}

TEST_F(LibTorchDataLoaderTest, MultipleWorkers_Compatibility)
{
	auto config = getTestConfig();
	auto dataLoader = torch::data::make_data_loader(
		ClassificationDataset(
			dataPath,
			(std::filesystem::path(annotationPath) / "classification").string(),
			config,
			false
		).map(DataExampleCollation()),
		torch::data::DataLoaderOptions().batch_size(4).workers(2)
	);

	bool hasData = false;
	for (auto& batch : *dataLoader) {
		hasData = true;
		ASSERT_TRUE(batch.data.defined()) << "Batch data not defined";
		std::cout << "MultipleWorkers batch validated successfully" << std::endl;
		break;
	}

	EXPECT_TRUE(hasData) << "Multi-worker DataLoader produced no batches";
}

TEST_F(LibTorchDataLoaderTest, DropLast_Compatibility)
{
	auto config = getTestConfig();
	auto dataLoader = torch::data::make_data_loader(
		DetectionDataset(
			dataPath,
			(std::filesystem::path(annotationPath) / "objectdetection").string(),
			config,
			false
		).map(DataExampleCollation()),
		torch::data::DataLoaderOptions().batch_size(3).workers(0).drop_last(true)
	);

	bool hasData = false;
	for (auto& batch : *dataLoader) {
		hasData = true;
		ASSERT_TRUE(batch.data.defined());
		// With drop_last=true, batch size should be exactly 3
		EXPECT_EQ(batch.data.size(0), 3);
		break;
	}

	EXPECT_TRUE(hasData) << "drop_last DataLoader produced no batches";
}
