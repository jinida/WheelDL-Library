#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Dataset/ClassificationDataset.h"
#include "WheelDL.Lib/Data/Dataset/DetectionDataset.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include <torch/torch.h>
#include <filesystem>

using namespace WheelDL::Data::Dataset;
using namespace WheelDL::Config;

namespace {
    std::filesystem::path getTestDataPath() {
        std::filesystem::path sourceDir = __FILE__;
        sourceDir = sourceDir.parent_path();
        while (sourceDir.filename() != "WheelDL.Lib.Tests" && sourceDir.has_parent_path()) {
            sourceDir = sourceDir.parent_path();
        }
        return sourceDir / "Data";
    }
}

class Phase2IntegrationTest : public ::testing::Test {
protected:
    std::string dataPath;
    std::string configDir;

    void SetUp() override {
        auto testData = getTestDataPath();
        dataPath = (testData / "mnist_sample").string();
        configDir = (testData / "Phase2TestConfigs").string();
    }
};

TEST_F(Phase2IntegrationTest, Classification_Full) {
    Configuration config;
    ClassificationDataset dataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "classification").string(),
        config, true);
    dataset.enableCache(CacheType::RAM);
    EXPECT_NO_THROW(dataset.get(0));
}

TEST_F(Phase2IntegrationTest, Detection_Full) {
    Configuration config;
    DetectionDataset dataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "objectdetection").string(),
        config, true);
    EXPECT_NO_THROW(dataset.get(0));
}

TEST_F(Phase2IntegrationTest, BatchGeneration_Small) {
    Configuration config;
    DetectionDataset dataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "objectdetection").string(),
        config, true);
    for (int i = 0; i < 5; ++i) {
        EXPECT_NO_THROW(dataset.get(i));
    }
}

TEST_F(Phase2IntegrationTest, StressTest_Sequential) {
    Configuration config;
    ClassificationDataset dataset(
        (std::filesystem::path(dataPath) / "images").string(),
        (std::filesystem::path(dataPath) / "labels" / "classification").string(),
        config, true);
    for (int i = 0; i < 100; ++i) {
        EXPECT_NO_THROW(dataset.get(i));
    }
}
