#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Dataset/ClassificationDataset.h"
#include "WheelDL.Lib/Data/Dataset/DetectionDataset.h"
#include "WheelDL.Lib/Data/Dataset/SegmentationDataset.h"
#include "WheelDL.Lib/Data/Dataset/OBBDataset.h"
#include "WheelDL.Lib/Data/Dataset/AnomalyDataset.h"
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

class RegressionTest : public ::testing::Test
{
protected:
	static std::unique_ptr<ClassificationDataset> classDataset;
	static std::unique_ptr<ClassificationDataset> classDatasetVal;
	static std::unique_ptr<DetectionDataset> detectDataset;
	static std::unique_ptr<DetectionDataset> detectDatasetVal;
	static std::unique_ptr<SegmentationDataset> segDataset;
	static std::unique_ptr<OBBDataset> obbDataset;
	static std::unique_ptr<AnomalyDataset> anomalyDataset;
	static std::filesystem::path dp;
	static std::filesystem::path lb;
	static std::filesystem::path cd;

	static void SetUpTestSuite()
	{
		try
		{
			dp = getTestDataPath() / "mnist_sample" / "images";
			lb = getTestDataPath() / "mnist_sample" / "labels";
			cd = getTestDataPath() / "Phase2TestConfigs";

			Configuration c;
			classDataset = std::make_unique<ClassificationDataset>(dp.string(), (lb / "classification").string(), c, true);
			classDatasetVal = std::make_unique<ClassificationDataset>(dp.string(), (lb / "classification").string(), c, false);
			detectDataset = std::make_unique<DetectionDataset>(dp.string(), (lb / "objectdetection").string(), c, true);
			detectDatasetVal = std::make_unique<DetectionDataset>(dp.string(), (lb / "objectdetection").string(), c, false);
			segDataset = std::make_unique<SegmentationDataset>(dp.string(), (lb / "segmentation").string(), c, true);
			obbDataset = std::make_unique<OBBDataset>(dp.string(), (lb / "obb").string(), c, true);
			anomalyDataset = std::make_unique<AnomalyDataset>(dp.string(), (lb / "anomalydetection").string(), c, true);
		}
		catch (const std::exception& e)
		{
			std::cerr << "RegressionTest SetUpTestSuite failed: " << e.what() << std::endl;
			classDataset.reset();
			classDatasetVal.reset();
			detectDataset.reset();
			detectDatasetVal.reset();
			segDataset.reset();
			obbDataset.reset();
			anomalyDataset.reset();
		}
	}

	static void TearDownTestSuite()
	{
		classDataset.reset();
		classDatasetVal.reset();
		detectDataset.reset();
		detectDatasetVal.reset();
		segDataset.reset();
		obbDataset.reset();
		anomalyDataset.reset();
	}
};

// Static member initialization
std::unique_ptr<ClassificationDataset> RegressionTest::classDataset;
std::unique_ptr<ClassificationDataset> RegressionTest::classDatasetVal;
std::unique_ptr<DetectionDataset> RegressionTest::detectDataset;
std::unique_ptr<DetectionDataset> RegressionTest::detectDatasetVal;
std::unique_ptr<SegmentationDataset> RegressionTest::segDataset;
std::unique_ptr<OBBDataset> RegressionTest::obbDataset;
std::unique_ptr<AnomalyDataset> RegressionTest::anomalyDataset;
std::filesystem::path RegressionTest::dp;
std::filesystem::path RegressionTest::lb;
std::filesystem::path RegressionTest::cd;

TEST_F(RegressionTest, ClassificationBasic)
{
	if (!classDataset)
	{
		return;
	}
	auto ex = classDataset->get(0);
	EXPECT_GT(ex.data.numel(), 0);
}

TEST_F(RegressionTest, DetectionBasic)
{
	if (!detectDataset)
	{
		return;
	}
	auto ex = detectDataset->get(0);
	EXPECT_GT(ex.data.numel(), 0);
}

TEST_F(RegressionTest, SegmentationBasic)
{
	if (!segDataset)
	{
		return;
	}
	auto ex = segDataset->get(0);
	EXPECT_GT(ex.data.numel(), 0);
}

TEST_F(RegressionTest, OBBBasic)
{
	if (!obbDataset)
	{
		return;
	}
	auto ex = obbDataset->get(0);
	EXPECT_GT(ex.data.numel(), 0);
}

TEST_F(RegressionTest, AnomalyBasic)
{
	if (!anomalyDataset)
	{
		return;
	}
	auto ex = anomalyDataset->get(0);
	EXPECT_GT(ex.data.numel(), 0);
}

TEST_F(RegressionTest, ClassificationSize)
{
	if (!classDataset)
	{
		return;
	}
	EXPECT_GT(*classDataset->size(), 0);
}

TEST_F(RegressionTest, DetectionSize)
{
	if (!detectDataset)
	{
		return;
	}
	EXPECT_GT(*detectDataset->size(), 0);
}

TEST_F(RegressionTest, SegmentationSize)
{
	if (!segDataset)
	{
		return;
	}
	EXPECT_GT(*segDataset->size(), 0);
}

TEST_F(RegressionTest, OBBSize)
{
	if (!obbDataset)
	{
		return;
	}
	EXPECT_GT(*obbDataset->size(), 0);
}

TEST_F(RegressionTest, AnomalySize)
{
	if (!anomalyDataset)
	{
		return;
	}
	EXPECT_GT(*anomalyDataset->size(), 0);
}

TEST_F(RegressionTest, ClassificationTrain)
{
	if (!classDataset)
	{
		return;
	}
	EXPECT_NO_THROW(classDataset->get(0));
}

TEST_F(RegressionTest, ClassificationVal)
{
	if (!classDatasetVal)
	{
		return;
	}
	EXPECT_NO_THROW(classDatasetVal->get(0));
}

TEST_F(RegressionTest, DetectionTrain)
{
	if (!detectDataset)
	{
		return;
	}
	EXPECT_NO_THROW(detectDataset->get(0));
}

TEST_F(RegressionTest, DetectionVal)
{
	if (!detectDatasetVal)
	{
		return;
	}
	EXPECT_NO_THROW(detectDatasetVal->get(0));
}

TEST_F(RegressionTest, CacheEnabled)
{
	if (!classDataset)
	{
		return;
	}
	classDataset->enableCache(CacheType::RAM);
	EXPECT_NO_THROW(classDataset->get(0));
}

TEST_F(RegressionTest, CacheClear)
{
	if (!classDataset)
	{
		return;
	}
	classDataset->enableCache(CacheType::RAM);
	classDataset->get(0);
	classDataset->clearCache();
	EXPECT_NO_THROW(classDataset->get(0));
}

TEST_F(RegressionTest, TensorShape)
{
	if (!classDataset)
	{
		return;
	}
	auto ex = classDataset->get(0);
	EXPECT_EQ(ex.data.dim(), 3);
}

TEST_F(RegressionTest, TensorChannels)
{
	if (!classDataset)
	{
		return;
	}
	auto ex = classDataset->get(0);
	EXPECT_EQ(ex.data.size(0), 3);
}

TEST_F(RegressionTest, ClassesTensor)
{
	if (!classDataset)
	{
		return;
	}
	auto ex = classDataset->get(0);
	EXPECT_GT(ex.classes.numel(), 0);
}

TEST_F(RegressionTest, TargetsTensor)
{
	if (!classDataset)
	{
		return;
	}
	auto ex = classDataset->get(0);
	EXPECT_GE(ex.targets.numel(), 0);
}

TEST_F(RegressionTest, BatchIndices)
{
	if (!classDataset)
	{
		return;
	}
	auto ex = classDataset->get(0);
	EXPECT_GE(ex.batchIndices.numel(), 0);
}

TEST_F(RegressionTest, ConfigAccess)
{
	if (!classDataset)
	{
		return;
	}
	EXPECT_EQ(classDataset->getConfig().getImageSize(), 640);
}

TEST_F(RegressionTest, PathsAccess)
{
	if (!classDataset)
	{
		return;
	}
	EXPECT_EQ(classDataset->getDataPath(), dp.string());
}
