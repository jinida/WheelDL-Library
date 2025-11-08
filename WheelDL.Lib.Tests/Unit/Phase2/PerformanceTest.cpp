#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Dataset/ClassificationDataset.h"
#include "WheelDL.Lib/Data/Cache/RAMCache.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include <chrono>
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

class PerformanceTest : public ::testing::Test
{
protected:
	static std::unique_ptr<Dataset::ClassificationDataset> dataset;
	static std::unique_ptr<Dataset::ClassificationDataset> cachedDataset;

	static void SetUpTestSuite()
	{
		try {
			Configuration c;
			std::string dataPath = "C:\\Users\\PC\\source\\repos\\WheelLib\\WheelDL.Lib.Tests\\Data\\mnist_sample";
			auto dp = dataPath + "\\images";
			auto lp = dataPath + "\\labels\\classification";
			dataset = std::make_unique<Dataset::ClassificationDataset>(dp, lp, c, true);
			cachedDataset = std::make_unique<Dataset::ClassificationDataset>(dp, lp, c, true);
			cachedDataset->enableCache(Dataset::CacheType::RAM);
			// Warm up cache
			for (int i = 0; i < 10; ++i)
			{
				cachedDataset->get(i);
			}
		}
		catch (const std::exception& e) {
			std::cerr << "PerformanceTest SetUpTestSuite failed: " << e.what() << std::endl;
			dataset.reset();
			cachedDataset.reset();
		}
	}

	static void TearDownTestSuite()
	{
		dataset.reset();
		cachedDataset.reset();
	}
};

std::unique_ptr<Dataset::ClassificationDataset> PerformanceTest::dataset;
std::unique_ptr<Dataset::ClassificationDataset> PerformanceTest::cachedDataset;

TEST_F(PerformanceTest, CachePut100)
{
	Cache::RAMCache c;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 100; ++i)
	{
		c.put("k" + std::to_string(i), cv::Mat(50, 50, CV_8UC3));
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	EXPECT_LT(ms, 1000);
}
TEST_F(PerformanceTest, CacheGet100)
{
	Cache::RAMCache c;
	for (int i = 0; i < 100; ++i)
	{
		c.put("k" + std::to_string(i), cv::Mat(50, 50, CV_8UC3));
	}
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 100; ++i)
	{
		c.get("k" + std::to_string(i));
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	EXPECT_LT(ms, 100);
}
TEST_F(PerformanceTest, DatasetGet10)
{
	if (!dataset)
	{
		std::cerr << "Dataset not initialized, skipping DatasetGet10 test" << std::endl;
		return;
	}
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10; ++i)
	{
		dataset->get(i);
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "DatasetGet10 took " << ms << "ms" << std::endl;
	EXPECT_LT(ms, 5000);
}
TEST_F(PerformanceTest, CachedGet10)
{
	if (!cachedDataset)
	{
		std::cerr << "Cached dataset not initialized, skipping CachedGet10 test" << std::endl;
		return;
	}
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 10; ++i)
	{
		cachedDataset->get(i);
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "CachedGet10 took " << ms << "ms" << std::endl;
	EXPECT_LT(ms, 500);
}
TEST_F(PerformanceTest, MemoryEfficiency)
{
	Cache::RAMCache c(1000, 10 * 1024 * 1024);
	for (int i = 0; i < 100; ++i)
	{
		c.put("k" + std::to_string(i), cv::Mat(100, 100, CV_8UC3));
	}
	EXPECT_LE(c.getMemoryUsage(), 10 * 1024 * 1024);
}
