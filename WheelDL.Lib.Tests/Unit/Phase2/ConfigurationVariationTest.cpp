#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include <filesystem>

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

class ConfigurationVariationTest : public ::testing::Test
{
protected:
	std::filesystem::path cd = getTestDataPath() / "Phase2TestConfigs";
};

TEST_F(ConfigurationVariationTest, DefaultConstruction)
{
	Configuration c;
	EXPECT_GT(c.getImageSize(), 0);
	EXPECT_GT(c.getBatchSize(), 0);
}

TEST_F(ConfigurationVariationTest, DefaultBatchSize)
{
	Configuration c;
	EXPECT_EQ(c.getBatchSize(), 16);
}

TEST_F(ConfigurationVariationTest, DefaultImageSize)
{
	Configuration c;
	EXPECT_EQ(c.getImageSize(), 640);
}

TEST_F(ConfigurationVariationTest, DefaultWorkers)
{
	Configuration c;
	EXPECT_GE(c.getWorkers(), 0);
}

TEST_F(ConfigurationVariationTest, DefaultSeed)
{
	Configuration c;
	EXPECT_GE(c.getSeed(), 0);
}

TEST_F(ConfigurationVariationTest, DefaultDegrees)
{
	Configuration c;
	EXPECT_GE(c.getDegrees(), 0.0f);
}

TEST_F(ConfigurationVariationTest, MultipleInstances)
{
	Configuration c1;
	Configuration c2;
	EXPECT_EQ(c1.getImageSize(), c2.getImageSize());
	EXPECT_EQ(c1.getBatchSize(), c2.getBatchSize());
}
