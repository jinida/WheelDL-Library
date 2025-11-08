#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Cache/RAMCache.h"
#include <opencv2/opencv.hpp>

using namespace WheelDL::Data::Cache;

class RAMCacheTest : public ::testing::Test
{
protected:
    std::unique_ptr<RAMCache> cache;
    void SetUp() override
    {
        cache = std::make_unique<RAMCache>(100);
    }
    cv::Mat createTestImage(int r, int c)
    {
        return cv::Mat(r, c, CV_8UC3, cv::Scalar(100, 150, 200));
    }
};

TEST_F(RAMCacheTest, Put_Success)
{
    cache->put("k1", createTestImage(100, 100));
    EXPECT_EQ(cache->size(), 1);
}

TEST_F(RAMCacheTest, Get_Success)
{
    cache->put("k1", createTestImage(100, 100));
    EXPECT_TRUE(cache->get("k1").has_value());
}

TEST_F(RAMCacheTest, Clear_Success)
{
    cache->put("k1", createTestImage(100, 100));
    cache->clear();
    EXPECT_EQ(cache->size(), 0);
}
