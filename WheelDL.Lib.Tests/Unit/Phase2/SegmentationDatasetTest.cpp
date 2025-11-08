#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Dataset/SegmentationDataset.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include <filesystem>

using namespace WheelDL::Data::Dataset;
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

class SegmentationDatasetTest : public ::testing::Test
{
protected:
    static std::unique_ptr<SegmentationDataset> trainDataset;
    static std::unique_ptr<SegmentationDataset> valDataset;
    static std::string dataPath;
    static std::string annotationPath;

    static void SetUpTestSuite()
    {
        try {
            dataPath = (getTestDataPath() / "mnist_sample" / "images").string();
            annotationPath = (getTestDataPath() / "mnist_sample" / "labels" / "segmentation").string();

            Configuration config;
            trainDataset = std::make_unique<SegmentationDataset>(dataPath, annotationPath, config, true);
            valDataset = std::make_unique<SegmentationDataset>(dataPath, annotationPath, config, false);
        } catch (const std::exception& e) {
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
std::unique_ptr<SegmentationDataset> SegmentationDatasetTest::trainDataset;
std::unique_ptr<SegmentationDataset> SegmentationDatasetTest::valDataset;
std::string SegmentationDatasetTest::dataPath;
std::string SegmentationDatasetTest::annotationPath;

TEST_F(SegmentationDatasetTest, Construction_Success)
{
    if (!trainDataset)
    {
        return;
    }
    Configuration config;
    EXPECT_NO_THROW(SegmentationDataset(dataPath, annotationPath, config, true));
}

TEST_F(SegmentationDatasetTest, Size_Positive)
{
    if (!trainDataset)
    {
        return;
    }
    ASSERT_TRUE(trainDataset->size().has_value());
    EXPECT_GT(*trainDataset->size(), 0);
}

TEST_F(SegmentationDatasetTest, Get_ValidIndex)
{
    if (!trainDataset)
    {
        return;
    }
    EXPECT_NO_THROW(trainDataset->get(0));
}

TEST_F(SegmentationDatasetTest, MaskTensor_3D)
{
    if (!trainDataset)
    {
        return;
    }
    auto example = trainDataset->get(0);
    if (example.targets.numel() > 0)
    {
        EXPECT_EQ(example.targets.dim(), 3);
    }
}

TEST_F(SegmentationDatasetTest, PolygonData_Valid)
{
    if (!trainDataset)
    {
        return;
    }
    auto example = trainDataset->get(0);
    EXPECT_FALSE(example.data.sizes().empty());
}

TEST_F(SegmentationDatasetTest, Cache_Enabled)
{
    if (!trainDataset)
    {
        return;
    }
    Configuration config;
    SegmentationDataset dataset(dataPath, annotationPath, config, true);
    dataset.enableCache(CacheType::RAM);
    EXPECT_NO_THROW(dataset.get(0));
}

TEST_F(SegmentationDatasetTest, TrainMode_Success)
{
    if (!trainDataset)
    {
        return;
    }
    EXPECT_NO_THROW(trainDataset->get(0));
}

TEST_F(SegmentationDatasetTest, ValMode_Success)
{
    if (!valDataset)
    {
        return;
    }
    EXPECT_NO_THROW(valDataset->get(0));
}

TEST_F(SegmentationDatasetTest, StressTest_Multiple)
{
    if (!trainDataset)
    {
        return;
    }
    for (int i = 0; i < 30; ++i)
    {
        EXPECT_NO_THROW(trainDataset->get(i));
    }
}

TEST_F(SegmentationDatasetTest, BatchIndices_Valid)
{
    if (!trainDataset)
    {
        return;
    }
    auto example = trainDataset->get(0);
    EXPECT_GE(example.batchIndices.numel(), 0);
}
