#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Dataset/ClassificationDataset.h"
#include "WheelDL.Lib/Data/Dataset/DetectionDataset.h"
#include "WheelDL.Lib/Data/Transforms/Collation.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include <torch/torch.h>
#include <filesystem>
#include <vector>

using namespace WheelDL::Data::Dataset;
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

class BatchGenerationTest : public ::testing::Test
{
protected:
	static std::unique_ptr<ClassificationDataset> classificationDataset;
	static std::unique_ptr<DetectionDataset> detectionDataset;
	static std::string dataPath;
	static std::string classificationPath;
	static std::string detectionPath;

	static void SetUpTestSuite()
	{
		try
		{
			auto testDataPath = getTestDataPath();
			dataPath = (testDataPath / "mnist_sample" / "images").string();
			classificationPath = (testDataPath / "mnist_sample" / "labels" / "classification").string();
			detectionPath = (testDataPath / "mnist_sample" / "labels" / "objectdetection").string();

			Configuration config;
			classificationDataset = std::make_unique<ClassificationDataset>(dataPath, classificationPath, config, true);
			detectionDataset = std::make_unique<DetectionDataset>(dataPath, detectionPath, config, true);
		}
		catch (const std::exception& e)
		{
			std::cerr << "SetUpTestSuite failed: " << e.what() << std::endl;
			classificationDataset.reset();
			detectionDataset.reset();
		}
	}

	static void TearDownTestSuite()
	{
		classificationDataset.reset();
		detectionDataset.reset();
	}
};

// Static member initialization
std::unique_ptr<ClassificationDataset> BatchGenerationTest::classificationDataset;
std::unique_ptr<DetectionDataset> BatchGenerationTest::detectionDataset;
std::string BatchGenerationTest::dataPath;
std::string BatchGenerationTest::classificationPath;
std::string BatchGenerationTest::detectionPath;

TEST_F(BatchGenerationTest, Collation_BatchSize2)
{
	if (!classificationDataset)
	{
		return;
	}
	// Check dataset is not empty
	ASSERT_GT(*classificationDataset->size(), 0) << "Dataset is empty - no images/labels loaded";
	ASSERT_GE(*classificationDataset->size(), 2) << "Dataset has less than 2 samples";

	std::vector<DataExample> examples;
	examples.push_back(classificationDataset->get(0));
	examples.push_back(classificationDataset->get(1));

	DataExampleCollation collation;
	auto batch = collation.apply_batch(examples);

	EXPECT_EQ(batch.data.size(0), 2);
	EXPECT_EQ(batch.classes.size(0), 2);
}

TEST_F(BatchGenerationTest, Collation_BatchSize4)
{
	if (!classificationDataset)
	{
		return;
	}
	ASSERT_GT(*classificationDataset->size(), 0) << "Dataset is empty";
	ASSERT_GE(*classificationDataset->size(), 4) << "Dataset has less than 4 samples";

	std::vector<DataExample> examples;
	for (int i = 0; i < 4; ++i)
	{
		examples.push_back(classificationDataset->get(i));
	}

	DataExampleCollation collation;
	auto batch = collation.apply_batch(examples);

	EXPECT_EQ(batch.data.size(0), 4);
	EXPECT_EQ(batch.classes.size(0), 4);
}

TEST_F(BatchGenerationTest, Batch_TensorShape)
{
	if (!classificationDataset)
	{
		return;
	}
	ASSERT_GT(*classificationDataset->size(), 0) << "Dataset is empty";
	ASSERT_GE(*classificationDataset->size(), 2) << "Dataset has less than 2 samples";

	std::vector<DataExample> examples;
	examples.push_back(classificationDataset->get(0));
	examples.push_back(classificationDataset->get(1));

	DataExampleCollation collation;
	auto batch = collation.apply_batch(examples);

	EXPECT_EQ(batch.data.dim(), 4);
}

TEST_F(BatchGenerationTest, Batch_DataStacked)
{
	if (!classificationDataset)
	{
		return;
	}
	ASSERT_GT(*classificationDataset->size(), 0) << "Dataset is empty";
	ASSERT_GE(*classificationDataset->size(), 2) << "Dataset has less than 2 samples";

	std::vector<DataExample> examples;
	examples.push_back(classificationDataset->get(0));
	examples.push_back(classificationDataset->get(1));

	DataExampleCollation collation;
	auto batch = collation.apply_batch(examples);

	EXPECT_EQ(batch.data.size(0), 2);
	EXPECT_GT(batch.data.size(1), 0);
}

TEST_F(BatchGenerationTest, Batch_TargetsConcatenated)
{
	if (!classificationDataset)
	{
		return;
	}
	ASSERT_GT(*classificationDataset->size(), 0) << "Dataset is empty";
	ASSERT_GE(*classificationDataset->size(), 2) << "Dataset has less than 2 samples";

	std::vector<DataExample> examples;
	examples.push_back(classificationDataset->get(0));
	examples.push_back(classificationDataset->get(1));

	DataExampleCollation collation;
	auto batch = collation.apply_batch(examples);

	EXPECT_GT(batch.targets.numel(), 0);
	EXPECT_GT(batch.batchIndices.numel(), 0);
}

TEST_F(BatchGenerationTest, Detection_Collation)
{
	if (!detectionDataset)
	{
		return;
	}
	ASSERT_GT(*detectionDataset->size(), 0) << "Dataset is empty";
	ASSERT_GE(*detectionDataset->size(), 2) << "Dataset has less than 2 samples";

	std::vector<DataExample> examples;
	examples.push_back(detectionDataset->get(0));
	examples.push_back(detectionDataset->get(1));

	DataExampleCollation collation;
	auto batch = collation.apply_batch(examples);

	EXPECT_EQ(batch.data.size(0), 2);
	EXPECT_GT(batch.targets.numel(), 0);
}

TEST_F(BatchGenerationTest, Collation_LargeBatch)
{
	if (!classificationDataset)
	{
		return;
	}
	ASSERT_GT(*classificationDataset->size(), 0) << "Dataset is empty";
	ASSERT_GE(*classificationDataset->size(), 8) << "Dataset has less than 8 samples";

	std::vector<DataExample> examples;
	for (int i = 0; i < 8; ++i)
	{
		examples.push_back(classificationDataset->get(i));
	}

	DataExampleCollation collation;
	auto batch = collation.apply_batch(examples);

	EXPECT_EQ(batch.data.size(0), 8);
	EXPECT_EQ(batch.classes.size(0), 8);
}
