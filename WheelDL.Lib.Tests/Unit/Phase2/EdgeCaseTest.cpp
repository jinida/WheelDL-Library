#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Dataset/ClassificationDataset.h"
#include "WheelDL.Lib/Data/Cache/RAMCache.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace WheelDL::Data;
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

class EdgeCaseTest : public ::testing::Test

{
};

TEST_F(EdgeCaseTest, EmptyCache)
{
	Cache::RAMCache c; EXPECT_EQ(c.size(), 0);
}
TEST_F(EdgeCaseTest, ZeroSizeCache)
{
	Cache::RAMCache c(0); EXPECT_EQ(c.size(), 0);
}
TEST_F(EdgeCaseTest, OneSizeCache)
{
	Cache::RAMCache c(1); c.put("k", cv::Mat(10, 10, CV_8UC3)); EXPECT_EQ(c.size(), 1);
}
TEST_F(EdgeCaseTest, NulloptGet)
{
	Cache::RAMCache c; EXPECT_FALSE(c.get("missing").has_value());
}
TEST_F(EdgeCaseTest, EmptyKeyCache)
{
	Cache::RAMCache c; c.put("", cv::Mat(10, 10, CV_8UC3)); EXPECT_TRUE(c.has(""));
}
TEST_F(EdgeCaseTest, LongKeyCache)
{
	Cache::RAMCache c; std::string k(10000, 'x'); c.put(k, cv::Mat(10, 10, CV_8UC3)); EXPECT_TRUE(c.has(k));
}
TEST_F(EdgeCaseTest, SpecialCharsKey)
{
	Cache::RAMCache c; c.put("@#$%^&*()", cv::Mat(10, 10, CV_8UC3)); EXPECT_TRUE(c.has("@#$%^&*()"));
}
TEST_F(EdgeCaseTest, UnicodeKey)
{
	Cache::RAMCache c; c.put("测试", cv::Mat(10, 10, CV_8UC3)); EXPECT_TRUE(c.has("测试"));
}
TEST_F(EdgeCaseTest, NewlineKey)
{
	Cache::RAMCache c; c.put("key\nwith\nnewlines", cv::Mat(10, 10, CV_8UC3)); EXPECT_TRUE(c.has("key\nwith\nnewlines"));
}
TEST_F(EdgeCaseTest, TabKey)
{
	Cache::RAMCache c; c.put("key\twith\ttabs", cv::Mat(10, 10, CV_8UC3)); EXPECT_TRUE(c.has("key\twith\ttabs"));
}
TEST_F(EdgeCaseTest, SmallImage1x1)
{
	Cache::RAMCache c; c.put("tiny", cv::Mat(1, 1, CV_8UC3)); EXPECT_TRUE(c.has("tiny"));
}
TEST_F(EdgeCaseTest, LargeImage4kx4k)
{
	Cache::RAMCache c; c.put("huge", cv::Mat(4096, 4096, CV_8UC3)); EXPECT_TRUE(c.has("huge"));
}
TEST_F(EdgeCaseTest, OverwriteSame)
{
	Cache::RAMCache c; c.put("k", cv::Mat(10, 10, CV_8UC3)); c.put("k", cv::Mat(20, 20, CV_8UC3)); EXPECT_EQ(c.size(), 1);
}
TEST_F(EdgeCaseTest, ClearEmpty)
{
	Cache::RAMCache c; c.clear(); EXPECT_EQ(c.size(), 0);
}
TEST_F(EdgeCaseTest, MultiClear)
{
	Cache::RAMCache c; c.put("k", cv::Mat(10, 10, CV_8UC3)); c.clear(); c.clear(); EXPECT_EQ(c.size(), 0);
}
TEST_F(EdgeCaseTest, GetAfterClear)
{
	Cache::RAMCache c; c.put("k", cv::Mat(10, 10, CV_8UC3)); c.clear(); EXPECT_FALSE(c.get("k").has_value());
}
TEST_F(EdgeCaseTest, HasAfterClear)
{
	Cache::RAMCache c; c.put("k", cv::Mat(10, 10, CV_8UC3)); c.clear(); EXPECT_FALSE(c.has("k"));
}
TEST_F(EdgeCaseTest, SizeAfterClear)
{
	Cache::RAMCache c; c.put("k", cv::Mat(10, 10, CV_8UC3)); c.clear(); EXPECT_EQ(c.size(), 0);
}
TEST_F(EdgeCaseTest, MemoryAfterClear)
{
	Cache::RAMCache c; c.put("k", cv::Mat(10, 10, CV_8UC3)); c.clear(); EXPECT_EQ(c.getMemoryUsage(), 0);
}
TEST_F(EdgeCaseTest, ReputAfterClear)
{
	Cache::RAMCache c; c.put("k", cv::Mat(10, 10, CV_8UC3)); c.clear(); c.put("k", cv::Mat(10, 10, CV_8UC3)); EXPECT_TRUE(c.has("k"));
}
TEST_F(EdgeCaseTest, MemoryLimit0)
{
	Cache::RAMCache c(100, 0); for (int i = 0; i < 200; ++i) c.put("k" + std::to_string(i), cv::Mat(100, 100, CV_8UC3)); EXPECT_LE(c.size(), 100);
}
TEST_F(EdgeCaseTest, SizeLimit0)
{
	Cache::RAMCache c(0, 1024 * 1024); c.put("k", cv::Mat(100, 100, CV_8UC3)); EXPECT_GT(c.size(), 0);
}
TEST_F(EdgeCaseTest, BothLimits0)
{
	Cache::RAMCache c(0, 0); c.put("k", cv::Mat(100, 100, CV_8UC3)); EXPECT_GT(c.size(), 0);
}
TEST_F(EdgeCaseTest, NegativeIndex)
{
	Configuration c; Dataset::ClassificationDataset d((getTestDataPath() / "mnist_sample" / "images").string(), (getTestDataPath() / "mnist_sample" / "labels" / "classification").string(), c, true); EXPECT_THROW(d.get(-1), std::out_of_range);
}
TEST_F(EdgeCaseTest, HugeIndex)
{
	Configuration c; Dataset::ClassificationDataset d((getTestDataPath() / "mnist_sample" / "images").string(), (getTestDataPath() / "mnist_sample" / "labels" / "classification").string(), c, true); EXPECT_THROW(d.get(999999), std::out_of_range);
}
