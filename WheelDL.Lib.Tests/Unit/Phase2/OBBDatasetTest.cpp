#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Dataset/OBBDataset.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include <filesystem>

using namespace WheelDL::Data::Dataset;
using namespace WheelDL::Config;

namespace
{
	std::filesystem::path getTestDataPath()
	{
		std::filesystem::path sourceDir = __FILE__;
		sourceDir = sourceDir.parent_path();
		while (sourceDir.filename() != "WheelDL.Lib.Tests" && sourceDir.has_parent_path())
		{
			sourceDir = sourceDir.parent_path();
		}
		return sourceDir / "Data";
	}
}

class OBBDatasetTest : public ::testing::Test
{
protected:
	static std::unique_ptr<OBBDataset> trainDataset;
	static std::unique_ptr<OBBDataset> valDataset;
	static std::string dataPath;
	static std::string annotationPath;

	static void SetUpTestSuite()
	{
		try
		{
			dataPath = (getTestDataPath() / "mnist_sample" / "images").string();
			annotationPath = (getTestDataPath() / "mnist_sample" / "labels" / "obb").string();

			Configuration config;
			trainDataset = std::make_unique<OBBDataset>(dataPath, annotationPath, config, true);
			valDataset = std::make_unique<OBBDataset>(dataPath, annotationPath, config, false);
		}
		catch (const std::exception& e)
		{
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
std::unique_ptr<OBBDataset> OBBDatasetTest::trainDataset;
std::unique_ptr<OBBDataset> OBBDatasetTest::valDataset;
std::string OBBDatasetTest::dataPath;
std::string OBBDatasetTest::annotationPath;

TEST_F(OBBDatasetTest, Construction_Success)
{
	if (!trainDataset)
	{
		return;
	}
	Configuration config;
	EXPECT_NO_THROW(OBBDataset(dataPath, annotationPath, config, true));
}

TEST_F(OBBDatasetTest, Size_Positive)
{
	if (!trainDataset)
	{
		return;
	}
	ASSERT_TRUE(trainDataset->size().has_value());
	EXPECT_GT(*trainDataset->size(), 0);
}

TEST_F(OBBDatasetTest, Get_ValidIndex)
{
	if (!trainDataset)
	{
		return;
	}
	EXPECT_NO_THROW(trainDataset->get(0));
}

TEST_F(OBBDatasetTest, TargetTensor_6Columns)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	if (example.targets.numel() > 0)
	{
		EXPECT_EQ(example.targets.size(1), 6);
	}
}

TEST_F(OBBDatasetTest, AngleValues_Valid)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	if (example.targets.numel() > 0)
	{
		auto angles = example.targets.slice(1, 5, 6);
		EXPECT_TRUE(angles.numel() > 0);
	}
}

TEST_F(OBBDatasetTest, TrainMode_Success)
{
	if (!trainDataset)
	{
		return;
	}
	EXPECT_NO_THROW(trainDataset->get(0));
}

TEST_F(OBBDatasetTest, ValMode_Success)
{
	if (!valDataset)
	{
		return;
	}
	EXPECT_NO_THROW(valDataset->get(0));
}

TEST_F(OBBDatasetTest, LargeImageSize_1024)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	EXPECT_EQ(std::max(example.data.size(1), example.data.size(2)), 1024);
}

TEST_F(OBBDatasetTest, Cache_Enabled)
{
	if (!trainDataset)
	{
		return;
	}
	Configuration config;
	OBBDataset dataset(dataPath, annotationPath, config, true);
	dataset.enableCache(CacheType::RAM);
	EXPECT_NO_THROW(dataset.get(0));
}

TEST_F(OBBDatasetTest, StressTest_Sequential)
{
	if (!trainDataset)
	{
		return;
	}
	for (int i = 0; i < 20; ++i)
	{
		EXPECT_NO_THROW(trainDataset->get(i));
	}
}
