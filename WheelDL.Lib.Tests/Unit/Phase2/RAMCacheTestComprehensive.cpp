#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Cache/RAMCache.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

using namespace WheelDL::Data::Cache;

class RAMCacheTestComprehensive : public ::testing::Test

{
protected:
	std::unique_ptr<RAMCache> cache;

	void SetUp() override
	{
		cache = std::make_unique<RAMCache>(100, 20 * 1024 * 1024);  // Increased to 20MB to accommodate large images
	}

	cv::Mat createTestImage(int rows, int cols)
	{
		return cv::Mat(rows, cols, CV_8UC3, cv::Scalar(100, 150, 200));
	}
};

TEST_F(RAMCacheTestComprehensive, Construction_Default)
{
	RAMCache c;
	EXPECT_EQ(c.size(), 0);
}

TEST_F(RAMCacheTestComprehensive, Put_Single)
{
	cache->put("k1", createTestImage(100, 100));
	EXPECT_EQ(cache->size(), 1);
}

TEST_F(RAMCacheTestComprehensive, Get_Existing)
{
	cache->put("k1", createTestImage(100, 100));
	EXPECT_TRUE(cache->get("k1").has_value());
}

TEST_F(RAMCacheTestComprehensive, Get_Missing)
{
	EXPECT_FALSE(cache->get("missing").has_value());
}

TEST_F(RAMCacheTestComprehensive, Has_True)
{
	cache->put("k1", createTestImage(100, 100));
	EXPECT_TRUE(cache->has("k1"));
}

TEST_F(RAMCacheTestComprehensive, Has_False)
{
	EXPECT_FALSE(cache->has("missing"));
}

TEST_F(RAMCacheTestComprehensive, Clear_Empty)
{
	cache->clear();
	EXPECT_EQ(cache->size(), 0);
}

TEST_F(RAMCacheTestComprehensive, Clear_Populated)
{
	cache->put("k1", createTestImage(100, 100));
	cache->clear();
	EXPECT_EQ(cache->size(), 0);
}

TEST_F(RAMCacheTestComprehensive, Size_Zero)
{
	EXPECT_EQ(cache->size(), 0);
}

TEST_F(RAMCacheTestComprehensive, Size_Multiple)
{
	for (int i = 0; i < 10; ++i)
	{
		cache->put("k" + std::to_string(i), createTestImage(50, 50));
	}
	EXPECT_EQ(cache->size(), 10);
}

TEST_F(RAMCacheTestComprehensive, MemoryUsage_Zero)
{
	EXPECT_EQ(cache->getMemoryUsage(), 0);
}

TEST_F(RAMCacheTestComprehensive, MemoryUsage_NonZero)
{
	cache->put("k1", createTestImage(100, 100));
	EXPECT_GT(cache->getMemoryUsage(), 0);
}

TEST_F(RAMCacheTestComprehensive, LRU_Eviction)
{
	RAMCache small(3);
	small.put("k1", createTestImage(10, 10));
	small.put("k2", createTestImage(10, 10));
	small.put("k3", createTestImage(10, 10));
	small.put("k4", createTestImage(10, 10));
	EXPECT_EQ(small.size(), 3);
	EXPECT_FALSE(small.has("k1"));
}

TEST_F(RAMCacheTestComprehensive, LRU_AccessUpdate)
{
	RAMCache small(3);
	small.put("k1", createTestImage(10, 10));
	small.put("k2", createTestImage(10, 10));
	small.put("k3", createTestImage(10, 10));
	small.get("k1");
	small.put("k4", createTestImage(10, 10));
	EXPECT_TRUE(small.has("k1"));
}

TEST_F(RAMCacheTestComprehensive, Overwrite_SameKey)
{
	cache->put("k1", createTestImage(50, 50));
	cache->put("k1", createTestImage(100, 100));
	EXPECT_EQ(cache->size(), 1);
}

TEST_F(RAMCacheTestComprehensive, ThreadSafety_Put)
{
	std::vector<std::thread> threads;
	for (int t = 0; t < 5; ++t)
	{
		threads.emplace_back([this, t]()
			{
				for (int i = 0; i < 10; ++i)
				{
					cache->put("t" + std::to_string(t) + "_" + std::to_string(i), createTestImage(50, 50));
				}
			});
	}
	for (auto& t : threads) t.join();
	EXPECT_GT(cache->size(), 0);
}

TEST_F(RAMCacheTestComprehensive, ThreadSafety_Get)
{
	for (int i = 0; i < 20; ++i)
	{
		cache->put("k" + std::to_string(i), createTestImage(50, 50));
	}
	std::vector<std::thread> threads;
	std::atomic<int> count{ 0 };
	for (int t = 0; t < 5; ++t)
	{
		threads.emplace_back([this, &count]()
			{
				for (int i = 0; i < 20; ++i)
				{
					if (cache->get("k" + std::to_string(i)).has_value()) count++;
				}
			});
	}
	for (auto& t : threads) t.join();
	EXPECT_GT(count.load(), 0);
}

TEST_F(RAMCacheTestComprehensive, StressTest_Many)
{
	for (int i = 0; i < 100; ++i)
	{
		cache->put("k" + std::to_string(i), createTestImage(50, 50));
	}
	EXPECT_EQ(cache->size(), 100);
}

TEST_F(RAMCacheTestComprehensive, EmptyKey)
{
	cache->put("", createTestImage(50, 50));
	EXPECT_TRUE(cache->has(""));
}

TEST_F(RAMCacheTestComprehensive, LongKey)
{
	std::string long_key(1000, 'x');
	cache->put(long_key, createTestImage(50, 50));
	EXPECT_TRUE(cache->has(long_key));
}

TEST_F(RAMCacheTestComprehensive, SpecialChars_Key)
{
	cache->put("k@#$%", createTestImage(50, 50));
	EXPECT_TRUE(cache->has("k@#$%"));
}

TEST_F(RAMCacheTestComprehensive, Unicode_Key)
{
	cache->put("??", createTestImage(50, 50));
	EXPECT_TRUE(cache->has("??"));
}

TEST_F(RAMCacheTestComprehensive, Large_Image)
{
	cache->put("large", createTestImage(2000, 2000));
	EXPECT_TRUE(cache->has("large"));
}

TEST_F(RAMCacheTestComprehensive, Small_Image)
{
	cache->put("small", createTestImage(1, 1));
	EXPECT_TRUE(cache->has("small"));
}

TEST_F(RAMCacheTestComprehensive, Multiple_Clear)
{
	cache->put("k1", createTestImage(50, 50));
	cache->clear();
	cache->put("k2", createTestImage(50, 50));
	cache->clear();
	EXPECT_EQ(cache->size(), 0);
}
