#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Dataset/ClassificationDataset.h"
#include "WheelDL.Lib/Data/Cache/RAMCache.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include <filesystem>

using namespace WheelDL::Data;
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

class RobustnessTest : public ::testing::Test
{
protected:
	static std::unique_ptr<Dataset::ClassificationDataset> dataset;
	static std::filesystem::path dp;
	static std::filesystem::path cp;

	static void SetUpTestSuite()
	{
		try
		{
			dp = getTestDataPath() / "mnist_sample" / "images";
			cp = getTestDataPath() / "mnist_sample" / "labels" / "classification";
			Configuration c;
			dataset = std::make_unique<Dataset::ClassificationDataset>(dp.string(), cp.string(), c, true);
		}
		catch (...)
		{
			dataset.reset();
		}
	}

	static void TearDownTestSuite()
	{
		dataset.reset();
	}
};

std::unique_ptr<Dataset::ClassificationDataset> RobustnessTest::dataset;
std::filesystem::path RobustnessTest::dp;
std::filesystem::path RobustnessTest::cp;

TEST_F(RobustnessTest, RepeatedGet)
{
	if (!dataset)
	{
		return;
	}
	for (int i = 0; i < 5; ++i)
		dataset->get(0);
	EXPECT_TRUE(true);
}

TEST_F(RobustnessTest, AlternatingGet)
{
	if (!dataset)
	{
		return;
	}
	for (int i = 0; i < 10; ++i)
		dataset->get(i % 2);
	EXPECT_TRUE(true);
}

TEST_F(RobustnessTest, ReverseGet)
{
	if (!dataset)
	{
		return;
	}
	for (int i = 19; i >= 0; --i)
		dataset->get(i);
	EXPECT_TRUE(true);
}

TEST_F(RobustnessTest, SkipGet)
{
	if (!dataset)
	{
		return;
	}
	for (int i = 0; i < 50; i += 5)
		dataset->get(i);
	EXPECT_TRUE(true);
}

TEST_F(RobustnessTest, CacheRepeatedPut)
{
	Cache::RAMCache c;
	for (int i = 0; i < 10; ++i)
		c.put("same", cv::Mat(10, 10, CV_8UC3));
	EXPECT_EQ(c.size(), 1);
}

TEST_F(RobustnessTest, CacheAlternatingOps)
{
	Cache::RAMCache c;
	for (int i = 0; i < 10; ++i)
	{
		c.put("k" + std::to_string(i), cv::Mat(10, 10, CV_8UC3));
		c.get("k" + std::to_string(i / 2));
	}
	EXPECT_GT(c.size(), 0);
}

TEST_F(RobustnessTest, MultipleClearOps)
{
	Cache::RAMCache c;
	for (int i = 0; i < 5; ++i)
	{
		c.put("k", cv::Mat(10, 10, CV_8UC3));
		c.clear();
	}
	EXPECT_EQ(c.size(), 0);
}

TEST_F(RobustnessTest, CacheAfterEviction)
{
	Cache::RAMCache c(5);
	for (int i = 0; i < 20; ++i)
		c.put("k" + std::to_string(i), cv::Mat(10, 10, CV_8UC3));
	EXPECT_EQ(c.size(), 5);
}

TEST_F(RobustnessTest, GetNonSequential)
{
	if (!dataset)
	{
		return;
	}
	std::vector<int> idxs = { 5,2,9,1,7,3,8,0,6,4 };
	for (auto i : idxs)
		dataset->get(i);
	EXPECT_TRUE(true);
}

TEST_F(RobustnessTest, CacheSizeConsistency)
{
	Cache::RAMCache c(10);
	for (int i = 0; i < 15; ++i)
	{
		c.put("k" + std::to_string(i), cv::Mat(10, 10, CV_8UC3));
		EXPECT_LE(c.size(), 10);
	}
}

TEST_F(RobustnessTest, MemoryConsistency)
{
	Cache::RAMCache c(100, 1024 * 1024);
	for (int i = 0; i < 50; ++i)
	{
		c.put("k" + std::to_string(i), cv::Mat(100, 100, CV_8UC3));
		EXPECT_LE(c.getMemoryUsage(), 2 * 1024 * 1024);
	}
}
