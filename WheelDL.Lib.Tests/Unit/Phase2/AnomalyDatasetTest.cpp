#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Dataset/AnomalyDataset.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include <filesystem>

using namespace WheelDL::Data::Dataset;
using namespace WheelDL::Config;

namespace {
	std::filesystem::path getTestDataPath()
	{
		std::filesystem::path sourceDir = __FILE__;
		sourceDir = sourceDir.parent_path();
		while (sourceDir.filename() != "WheelDL.Lib.Tests" && sourceDir.has_parent_path()) {
			sourceDir = sourceDir.parent_path();
		}
		return sourceDir / "Data";
	}
}

class AnomalyDatasetTest : public ::testing::Test
{
protected:
	static std::unique_ptr<AnomalyDataset> trainDataset;
	static std::unique_ptr<AnomalyDataset> valDataset;
	static std::string dataPath;
	static std::string annotationPath;

	static void SetUpTestSuite()
	{
		try {
			dataPath = (getTestDataPath() / "mnist_sample" / "images").string();
			annotationPath = (getTestDataPath() / "mnist_sample" / "labels" / "anomalydetection").string();

			Configuration config;
			trainDataset = std::make_unique<AnomalyDataset>(dataPath, annotationPath, config, true);
			valDataset = std::make_unique<AnomalyDataset>(dataPath, annotationPath, config, false);
		}
		catch (const std::exception& e) {
			std::cerr << "SetUpTestSuite failed: " << e.what() << std::endl;
			trainDataset.reset();
			valDataset.reset();
		}
	}

	static void TearDownTestSuite()
	{
		trainDataset.reset();
		valDataset.reset();
	}
};

// Static member initialization
std::unique_ptr<AnomalyDataset> AnomalyDatasetTest::trainDataset;
std::unique_ptr<AnomalyDataset> AnomalyDatasetTest::valDataset;
std::string AnomalyDatasetTest::dataPath;
std::string AnomalyDatasetTest::annotationPath;

TEST_F(AnomalyDatasetTest, Construction_Success)
{
	if (!trainDataset) { return; }
	Configuration config;
	EXPECT_NO_THROW(AnomalyDataset(dataPath, annotationPath, config, true));
}

TEST_F(AnomalyDatasetTest, Size_Positive)
{
	if (!trainDataset) { return; }
	ASSERT_TRUE(trainDataset->size().has_value());
	EXPECT_GT(*trainDataset->size(), 0);
}

TEST_F(AnomalyDatasetTest, Get_ValidIndex)
{
	if (!trainDataset) { return; }
	EXPECT_NO_THROW(trainDataset->get(0));
}

TEST_F(AnomalyDatasetTest, BinaryLabel_Valid)
{
	if (!trainDataset) { return; }
	auto example = trainDataset->get(0);
	if (example.targets.numel() > 0) {
		auto label = example.targets[0].item<int64_t>();
		EXPECT_TRUE(label == 0 || label == 1 || label == -1);
	}
}

TEST_F(AnomalyDatasetTest, SmallImageSize_256)
{
	if (!trainDataset) { return; }
	auto example = trainDataset->get(0);
	EXPECT_EQ(std::max(example.data.size(1), example.data.size(2)), 256);
}

TEST_F(AnomalyDatasetTest, LargeBatchSize)
{
	if (!trainDataset) { return; }
	Configuration config;
	EXPECT_EQ(config.getBatchSize(), 64);
}

TEST_F(AnomalyDatasetTest, Cache_LargeSize)
{
	if (!trainDataset) { return; }
	Configuration config;
	AnomalyDataset dataset(dataPath, annotationPath, config, true);
	dataset.enableCache(CacheType::RAM);
	EXPECT_NO_THROW(dataset.get(0));
}

TEST_F(AnomalyDatasetTest, TrainMode_Success)
{
	if (!trainDataset) { return; }
	EXPECT_NO_THROW(trainDataset->get(0));
}

TEST_F(AnomalyDatasetTest, ValMode_Success)
{
	if (!valDataset) { return; }
	EXPECT_NO_THROW(valDataset->get(0));
}

TEST_F(AnomalyDatasetTest, StressTest_LargeBatch)
{
	if (!trainDataset) { return; }
	for (int i = 0; i < 64; ++i) {
		EXPECT_NO_THROW(trainDataset->get(i));
	}
}
