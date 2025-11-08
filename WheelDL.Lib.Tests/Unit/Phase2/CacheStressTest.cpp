#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Cache/RAMCache.h"
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

using namespace WheelDL::Data::Cache;

class CacheStressTest : public ::testing::Test
{
protected:
	cv::Mat createImg(int s)
	{
		return cv::Mat(s, s, CV_8UC3, cv::Scalar(100, 100, 100));
	}
};

TEST_F(CacheStressTest, Put100)
{
	RAMCache c(200);
	for (int i = 0; i < 100; ++i)
		c.put("k" + std::to_string(i), createImg(50));
	EXPECT_EQ(c.size(), 100);
}
TEST_F(CacheStressTest, Put200)
{
	RAMCache c(300);
	for (int i = 0; i < 200; ++i)
		c.put("k" + std::to_string(i), createImg(50));
	EXPECT_EQ(c.size(), 200);
}
TEST_F(CacheStressTest, Put500)
{
	RAMCache c(600);
	for (int i = 0; i < 500; ++i)
		c.put("k" + std::to_string(i), createImg(50));
	EXPECT_EQ(c.size(), 500);
}
TEST_F(CacheStressTest, Get100)
{
	RAMCache c;
	for (int i = 0; i < 100; ++i)
		c.put("k" + std::to_string(i), createImg(50));
	for (int i = 0; i < 100; ++i)
		EXPECT_TRUE(c.get("k" + std::to_string(i)).has_value());
}
TEST_F(CacheStressTest, Evict)
{
	RAMCache c(10);
	for (int i = 0; i < 20; ++i)
		c.put("k" + std::to_string(i), createImg(50));
	EXPECT_EQ(c.size(), 10);
}
TEST_F(CacheStressTest, Clear100)
{
	RAMCache c;
	for (int i = 0; i < 100; ++i)
		c.put("k" + std::to_string(i), createImg(50));
	c.clear();
	EXPECT_EQ(c.size(), 0);
}
TEST_F(CacheStressTest, Mixed)
{
	RAMCache c(50);
	for (int i = 0; i < 100; ++i)
	{
		c.put("k" + std::to_string(i), createImg(50));
		if (i % 10 == 0)
			c.get("k" + std::to_string(i / 2));
	}
	EXPECT_GT(c.size(), 0);
}
TEST_F(CacheStressTest, LargeImg)
{
	RAMCache c;
	c.put("large", createImg(2000));
	EXPECT_TRUE(c.has("large"));
}

TEST_F(CacheStressTest, SmallImg)
{
	RAMCache c;
	c.put("small", createImg(1));
	EXPECT_TRUE(c.has("small"));
}
TEST_F(CacheStressTest, Memory1MB)
{
	RAMCache c(1000, 1024 * 1024);
	for (int i = 0; i < 100; ++i)
		c.put("k" + std::to_string(i), createImg(100));
	EXPECT_GT(c.size(), 0);
}

TEST_F(CacheStressTest, Memory10MB)
{
	RAMCache c(1000, 10 * 1024 * 1024);
	for (int i = 0; i < 100; ++i)
		c.put("k" + std::to_string(i), createImg(200));
	EXPECT_GT(c.size(), 0);
}
TEST_F(CacheStressTest, Thread2)
{
	RAMCache c;
	std::vector<std::thread> t;
	for (int i = 0; i < 2; ++i)
	{
		t.emplace_back([&, i]()
		{
			for (int j = 0; j < 50; ++j)
				c.put("t" + std::to_string(i) + "_" + std::to_string(j), createImg(50));
		});
	}
	for (auto& x : t)
		x.join();
	EXPECT_GT(c.size(), 0);
}
TEST_F(CacheStressTest, Thread4)
{
	RAMCache c;
	std::vector<std::thread> t;
	for (int i = 0; i < 4; ++i)
	{
		t.emplace_back([&, i]()
		{
			for (int j = 0; j < 25; ++j)
				c.put("t" + std::to_string(i) + "_" + std::to_string(j), createImg(50));
		});
	}
	for (auto& x : t)
		x.join();
	EXPECT_GT(c.size(), 0);
}

TEST_F(CacheStressTest, Thread8)
{
	RAMCache c;
	std::vector<std::thread> t;
	for (int i = 0; i < 8; ++i)
	{
		t.emplace_back([&, i]()
		{
			for (int j = 0; j < 12; ++j)
				c.put("t" + std::to_string(i) + "_" + std::to_string(j), createImg(50));
		});
	}
	for (auto& x : t)
		x.join();
	EXPECT_GT(c.size(), 0);
}
TEST_F(CacheStressTest, Overwrite)
{
	RAMCache c;
	for (int i = 0; i < 10; ++i)
	{
		c.put("k1", createImg(50 + i * 10));
	}
	EXPECT_EQ(c.size(), 1);
}
