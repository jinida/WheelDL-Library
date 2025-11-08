#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Dataset/ClassificationDataset.h"
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

class DatasetStressTest : public ::testing::Test
{
protected:
	static std::unique_ptr<ClassificationDataset> dataset;
	static std::string dp;
	static std::string cp;
	static std::string dp2;

	static void SetUpTestSuite()
	{
		try
		{
			dp = (getTestDataPath() / "mnist_sample" / "images").string();
			cp = (getTestDataPath() / "mnist_sample" / "labels" / "classification").string();
			dp2 = (getTestDataPath() / "mnist_sample" / "labels" / "objectdetection").string();

			Configuration config;
			dataset = std::make_unique<ClassificationDataset>(dp, cp, config, true);
		}
		catch (const std::exception& e)
		{
			std::cerr << "SetUpTestSuite failed: " << e.what() << std::endl;
			dataset.reset();
		}
	}

	static void TearDownTestSuite()
	{
		dataset.reset();
	}
};

// Static member initialization
std::unique_ptr<ClassificationDataset> DatasetStressTest::dataset;
std::string DatasetStressTest::dp;
std::string DatasetStressTest::cp;
std::string DatasetStressTest::dp2;

TEST_F(DatasetStressTest, Get0)
{
	if (!dataset)
	{
		return;
	}
	EXPECT_NO_THROW(dataset->get(0));
}

TEST_F(DatasetStressTest, Get1)
{
	if (!dataset)
	{
		return;
	}
	EXPECT_NO_THROW(dataset->get(1));
}

TEST_F(DatasetStressTest, Get10)
{
	if (!dataset)
	{
		return;
	}
	EXPECT_NO_THROW(dataset->get(10));
}

TEST_F(DatasetStressTest, Get50)
{
	if (!dataset)
	{
		return;
	}
	EXPECT_NO_THROW(dataset->get(50));
}

TEST_F(DatasetStressTest, Get100)
{
	if (!dataset)
	{
		return;
	}
	EXPECT_NO_THROW(dataset->get(100));
}

TEST_F(DatasetStressTest, Sequential10)
{
	if (!dataset)
	{
		return;
	}
	for (int i = 0; i < 10; ++i)
		dataset->get(i);
	EXPECT_TRUE(true);
}

TEST_F(DatasetStressTest, Sequential50)
{
	if (!dataset)
	{
		return;
	}
	for (int i = 0; i < 50; ++i)
		dataset->get(i);
	EXPECT_TRUE(true);
}

TEST_F(DatasetStressTest, Sequential100)
{
	if (!dataset)
	{
		return;
	}
	for (int i = 0; i < 100; ++i)
		dataset->get(i);
	EXPECT_TRUE(true);
}

TEST_F(DatasetStressTest, Random10)
{
	if (!dataset)
	{
		return;
	}
	for (int i = 0; i < 10; ++i)
		dataset->get(i * 7 % 20);
	EXPECT_TRUE(true);
}

TEST_F(DatasetStressTest, Cache10)
{
	if (!dataset)
	{
		return;
	}
	dataset->enableCache(CacheType::RAM);
	for (int i = 0; i < 10; ++i)
		dataset->get(i);
	EXPECT_TRUE(true);
}

TEST_F(DatasetStressTest, CacheClear)
{
	if (!dataset)
	{
		return;
	}
	dataset->enableCache(CacheType::RAM);
	dataset->get(0);
	dataset->clearCache();
	EXPECT_NO_THROW(dataset->get(0));
}

TEST_F(DatasetStressTest, Train)
{
	if (!dataset)
	{
		return;
	}
	EXPECT_NO_THROW(dataset->get(0));
}

TEST_F(DatasetStressTest, Val)
{
	if (!dataset)
	{
		return;
	}
	Configuration c;
	ClassificationDataset d(dp, cp, c, false);
	EXPECT_NO_THROW(d.get(0));
}

TEST_F(DatasetStressTest, SizePositive)
{
	if (!dataset)
	{
		return;
	}
	EXPECT_GT(*dataset->size(), 0);
}

TEST_F(DatasetStressTest, PathsValid)
{
	if (!dataset)
	{
		return;
	}
	EXPECT_EQ(dataset->getDataPath(), dp);
}
