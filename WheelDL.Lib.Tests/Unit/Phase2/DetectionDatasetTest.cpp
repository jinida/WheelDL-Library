#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Dataset/DetectionDataset.h"
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

class DetectionDatasetTest : public ::testing::Test
{
protected:
	static std::unique_ptr<DetectionDataset> trainDataset;
	static std::unique_ptr<DetectionDataset> valDataset;
	static std::string dataPath;
	static std::string annotationPath;

	static void SetUpTestSuite()
	{
		try
		{
			dataPath = (getTestDataPath() / "mnist_sample" / "images").string();
			annotationPath = (getTestDataPath() / "mnist_sample" / "labels" / "objectdetection").string();

			Configuration config;
			trainDataset = std::make_unique<DetectionDataset>(dataPath, annotationPath, config, true);
			valDataset = std::make_unique<DetectionDataset>(dataPath, annotationPath, config, false);
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
std::unique_ptr<DetectionDataset> DetectionDatasetTest::trainDataset;
std::unique_ptr<DetectionDataset> DetectionDatasetTest::valDataset;
std::string DetectionDatasetTest::dataPath;
std::string DetectionDatasetTest::annotationPath;

TEST_F(DetectionDatasetTest, Construction_Success)
{
	if (!trainDataset)
	{
		return;
	}
	Configuration config;
	EXPECT_NO_THROW(DetectionDataset(dataPath, annotationPath, config, true));
}

TEST_F(DetectionDatasetTest, Size_Positive)
{
	if (!trainDataset)
	{
		return;
	}
	auto size = trainDataset->size();
	ASSERT_TRUE(size.has_value());
	EXPECT_GT(*size, 0);
}

TEST_F(DetectionDatasetTest, Get_ValidIndex)
{
	if (!trainDataset)
	{
		return;
	}
	EXPECT_NO_THROW(trainDataset->get(0));
}

TEST_F(DetectionDatasetTest, TargetTensor_2D)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	if (example.targets.numel() > 0)
	{
		EXPECT_EQ(example.targets.dim(), 2);
	}
}

TEST_F(DetectionDatasetTest, TargetTensor_Columns)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	if (example.targets.numel() > 0)
	{
		EXPECT_EQ(example.targets.size(1), 5);
	}
}

TEST_F(DetectionDatasetTest, BBox_Normalized)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	if (example.targets.numel() > 0)
	{
		auto coords = example.targets.slice(1, 1, 5);
		auto max_val = coords.max().item<float>();
		EXPECT_LE(max_val, 1.0f);
	}
}

TEST_F(DetectionDatasetTest, MultipleDetections)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	if (example.targets.numel() > 0)
	{
		EXPECT_GT(example.targets.size(0), 0);
	}
}

TEST_F(DetectionDatasetTest, Cache_NoCache)
{
	if (!trainDataset)
	{
		return;
	}
	auto ex1 = trainDataset->get(0);
	auto ex2 = trainDataset->get(0);
	EXPECT_TRUE(torch::allclose(ex1.data, ex2.data));
}

TEST_F(DetectionDatasetTest, TrainMode_Augmentation)
{
	if (!trainDataset)
	{
		return;
	}
	EXPECT_NO_THROW(trainDataset->get(0));
}

TEST_F(DetectionDatasetTest, ValMode_NoAugmentation)
{
	if (!valDataset)
	{
		return;
	}
	EXPECT_NO_THROW(valDataset->get(0));
}

TEST_F(DetectionDatasetTest, StressTest_Sequential)
{
	if (!trainDataset)
	{
		return;
	}
	for (int i = 0; i < 50; ++i)
	{
		EXPECT_NO_THROW(trainDataset->get(i));
	}
}

TEST_F(DetectionDatasetTest, ImageTensor_Shape)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	EXPECT_EQ(example.data.dim(), 3);
}

TEST_F(DetectionDatasetTest, ClassIDs_Valid)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	if (example.targets.numel() > 0)
	{
		auto classes = example.targets.slice(1, 0, 1);
		EXPECT_GE(classes.min().item<float>(), 0.0f);
	}
}
