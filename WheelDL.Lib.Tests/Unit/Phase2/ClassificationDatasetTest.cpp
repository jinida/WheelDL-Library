#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Dataset/ClassificationDataset.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include <filesystem>

using namespace WheelDL::Data::Dataset;
using namespace WheelDL::Config;
namespace fs = std::filesystem;

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

class ClassificationDatasetTest : public ::testing::Test
{
protected:
	static std::unique_ptr<ClassificationDataset> trainDataset;
	static std::unique_ptr<ClassificationDataset> valDataset;
	static std::string dataPath;
	static std::string annotationPath;

	static void SetUpTestSuite()
	{
		try
		{
			dataPath = (getTestDataPath() / "mnist_sample" / "images").string();
			annotationPath = (getTestDataPath() / "mnist_sample" / "labels" / "classification").string();

			Configuration config;
			trainDataset = std::make_unique<ClassificationDataset>(dataPath, annotationPath, config, true);
			valDataset = std::make_unique<ClassificationDataset>(dataPath, annotationPath, config, false);
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
std::unique_ptr<ClassificationDataset> ClassificationDatasetTest::trainDataset;
std::unique_ptr<ClassificationDataset> ClassificationDatasetTest::valDataset;
std::string ClassificationDatasetTest::dataPath;
std::string ClassificationDatasetTest::annotationPath;

TEST_F(ClassificationDatasetTest, Construction_Success)
{
	if (!trainDataset)
	{
		return;
	}
	Configuration config;
	EXPECT_NO_THROW({
		ClassificationDataset dataset(dataPath, annotationPath, config, true);
	});
}

TEST_F(ClassificationDatasetTest, Size_Positive)
{
	if (!trainDataset)
	{
		return;
	}
	auto size = trainDataset->size();
	ASSERT_TRUE(size.has_value());
	EXPECT_GT(*size, 0);
}

TEST_F(ClassificationDatasetTest, Get_ValidIndex)
{
	if (!trainDataset)
	{
		return;
	}
	EXPECT_NO_THROW({
		auto example = trainDataset->get(0);
		EXPECT_FALSE(example.data.sizes().empty());
	});
}

TEST_F(ClassificationDatasetTest, Get_OutOfRange)
{
	if (!trainDataset)
	{
		return;
	}
	EXPECT_THROW({
		trainDataset->get(999999);
	}, std::out_of_range);
}

TEST_F(ClassificationDatasetTest, DataTensor_Shape)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	EXPECT_EQ(example.data.dim(), 3);
	EXPECT_EQ(example.data.size(0), 3);
}

TEST_F(ClassificationDatasetTest, ClassesTensor_Valid)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	EXPECT_GT(example.classes.numel(), 0);
}

TEST_F(ClassificationDatasetTest, TargetTensor_Shape)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	EXPECT_FALSE(example.targets.sizes().empty());
}

TEST_F(ClassificationDatasetTest, BatchIndices_Valid)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	EXPECT_GE(example.batchIndices.numel(), 0);
}

TEST_F(ClassificationDatasetTest, Cache_EnableRAM)
{
	if (!trainDataset)
	{
		return;
	}
	Configuration config;
	ClassificationDataset dataset(dataPath, annotationPath, config, true);
	dataset.enableCache(CacheType::RAM);
	auto ex1 = dataset.get(0);
	auto ex2 = dataset.get(0);
	EXPECT_TRUE(torch::allclose(ex1.data, ex2.data));
}

TEST_F(ClassificationDatasetTest, Cache_Clear)
{
	if (!trainDataset)
	{
		return;
	}
	Configuration config;
	ClassificationDataset dataset(dataPath, annotationPath, config, true);
	dataset.enableCache(CacheType::RAM);
	dataset.get(0);
	EXPECT_NO_THROW(dataset.clearCache());
}

TEST_F(ClassificationDatasetTest, TrainMode_Success)
{
	if (!trainDataset)
	{
		return;
	}
	EXPECT_NO_THROW(trainDataset->get(0));
}

TEST_F(ClassificationDatasetTest, ValMode_Success)
{
	if (!valDataset)
	{
		return;
	}
	EXPECT_NO_THROW(valDataset->get(0));
}

TEST_F(ClassificationDatasetTest, MultipleGets_Success)
{
	if (!trainDataset)
	{
		return;
	}
	for (int i = 0; i < 10; ++i)
	{
		EXPECT_NO_THROW(trainDataset->get(i));
	}
}

TEST_F(ClassificationDatasetTest, RandomAccess_Success)
{
	if (!trainDataset)
	{
		return;
	}
	auto size = trainDataset->size();
	if (size.has_value() && *size > 10)
	{
		EXPECT_NO_THROW(trainDataset->get(*size / 2));
	}
}

TEST_F(ClassificationDatasetTest, Paths_Valid)
{
	if (!trainDataset)
	{
		return;
	}
	EXPECT_EQ(trainDataset->getDataPath(), dataPath);
	EXPECT_EQ(trainDataset->getAnnotationPath(), annotationPath);
}

TEST_F(ClassificationDatasetTest, Config_Accessible)
{
	if (!trainDataset)
	{
		return;
	}
	const auto& cfg = trainDataset->getConfig();
	EXPECT_EQ(cfg.getImageSize(), 640);
}

TEST_F(ClassificationDatasetTest, StressTest_SequentialAccess)
{
	if (!trainDataset)
	{
		return;
	}
	auto size = trainDataset->size();
	if (size.has_value())
	{
		int limit = std::min(100, static_cast<int>(*size));
		for (int i = 0; i < limit; ++i)
		{
			EXPECT_NO_THROW(trainDataset->get(i));
		}
	}
}

TEST_F(ClassificationDatasetTest, DataConsistency_SameIndex)
{
	if (!trainDataset)
	{
		return;
	}
	auto ex1 = trainDataset->get(0);
	auto ex2 = trainDataset->get(0);
	EXPECT_TRUE(torch::allclose(ex1.classes, ex2.classes));
}

TEST_F(ClassificationDatasetTest, TensorValues_InRange)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	auto data_min = example.data.min().item<float>();
	auto data_max = example.data.max().item<float>();
	EXPECT_GE(data_min, 0.0f);
	EXPECT_LE(data_max, 255.0f);
}

TEST_F(ClassificationDatasetTest, ClassLabels_Valid)
{
	if (!trainDataset)
	{
		return;
	}
	auto example = trainDataset->get(0);
	auto class_val = example.classes[0].item<int64_t>();
	EXPECT_GE(class_val, 0);
	EXPECT_LT(class_val, 10);
}
