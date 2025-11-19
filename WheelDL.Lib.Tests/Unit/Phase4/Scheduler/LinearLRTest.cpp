#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Optimizer/Scheduler/LinearLR.h"
#include "WheelDL.Lib/Optimizer/OptimizerFactory.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <filesystem>

using namespace WheelDL::Optimizer;
using namespace WheelDL::Optimizer::Scheduler;
using namespace WheelDL::Config;

namespace {
	std::filesystem::path getTestDataPath() {
		// Get the directory containing this source file
		std::filesystem::path sourceDir = __FILE__;
		sourceDir = sourceDir.parent_path(); // Remove filename

		// Navigate to WheelDL.Lib.Tests/Data
		while (sourceDir.filename() != "WheelDL.Lib.Tests" && sourceDir.has_parent_path()) {
			sourceDir = sourceDir.parent_path();
		}
		return sourceDir / "Data" / "Phase4TestConfig";
	}
}

class LinearLRTest : public ::testing::Test {
protected:
    void SetUp() override {
        params.push_back(torch::randn({10, 10}, torch::requires_grad(true)));

        // Get test data path
        testDataPath = getTestDataPath();
        defaultModelPath = (testDataPath / "default_model.yaml").string();
        // Load configuration for linear LR
        config = std::make_shared<Configuration>();
        auto configPath = (testDataPath / "linear_lr_config.yaml").string();
        config->loadFromYaml(defaultModelPath, configPath, "");

        // Create optimizer
        auto optimizerPtr = OptimizerFactory::createFromConfig(params, *config);
        optimizer = optimizerPtr.get();
        optimizerOwner = std::move(optimizerPtr);
    }

    std::vector<torch::Tensor> params;
    torch::optim::Optimizer* optimizer;
    std::unique_ptr<torch::optim::Optimizer> optimizerOwner;
    std::shared_ptr<Configuration> config;
    std::filesystem::path testDataPath;
    std::string defaultModelPath;
};

TEST_F(LinearLRTest, InitialLR) {
    LinearLR scheduler(optimizer, *config);

    float initialLR = scheduler.getInitialLR();
    EXPECT_GT(initialLR, 0.0f);
}

TEST_F(LinearLRTest, LinearDecay) {
    LinearLR scheduler(optimizer, *config);

    float initialLR = scheduler.getInitialLR();
    float finalLR = scheduler.getFinalLR();
    int totalEpochs = scheduler.getTotalEpochs();
    int warmupEpochs = scheduler.getWarmupEpochs();

    // Test at quarter, half, three-quarter points
    int quarter = warmupEpochs + (totalEpochs - warmupEpochs) / 4;
    int half = warmupEpochs + (totalEpochs - warmupEpochs) / 2;
    int threeQuarter = warmupEpochs + 3 * (totalEpochs - warmupEpochs) / 4;

    scheduler.step(quarter);
    float quarterLR = scheduler.getCurrentLR();

    scheduler.step(half);
    float halfLR = scheduler.getCurrentLR();

    scheduler.step(threeQuarter);
    float threeQuarterLR = scheduler.getCurrentLR();

    // Linear decay should maintain consistent differences
    float diff1 = quarterLR - halfLR;
    float diff2 = halfLR - threeQuarterLR;

    EXPECT_NEAR(diff1, diff2, 0.001f);  // Differences should be similar
}

TEST_F(LinearLRTest, MidpointLR) {
    LinearLR scheduler(optimizer, *config);

    float initialLR = scheduler.getInitialLR();
    float finalLR = scheduler.getFinalLR();
    int totalEpochs = scheduler.getTotalEpochs();
    int warmupEpochs = scheduler.getWarmupEpochs();

    int midpoint = warmupEpochs + (totalEpochs - warmupEpochs) / 2;

    scheduler.step(midpoint);
    float midLR = scheduler.getCurrentLR();

    // At midpoint, should be average of initial and final
    float expectedMidLR = (initialLR + finalLR) / 2.0f;
    EXPECT_NEAR(midLR, expectedMidLR, 0.01f);
}

TEST_F(LinearLRTest, FinalEpochLR) {
    LinearLR scheduler(optimizer, *config);

    int totalEpochs = scheduler.getTotalEpochs();
    float finalLR = scheduler.getFinalLR();

    scheduler.step(totalEpochs - 1);
    float actualFinalLR = scheduler.getCurrentLR();

    EXPECT_NEAR(actualFinalLR, finalLR, 0.001f);
}

TEST_F(LinearLRTest, WarmupPhase) {
    LinearLR scheduler(optimizer, *config);

    int warmupEpochs = scheduler.getWarmupEpochs();

    if (warmupEpochs > 0) {
        scheduler.step(0);
        float firstLR = scheduler.getCurrentLR();

        scheduler.step(warmupEpochs - 1);
        float lastWarmupLR = scheduler.getCurrentLR();

        // Warmup should increase linearly
        EXPECT_LT(firstLR, lastWarmupLR);
    }
}

TEST_F(LinearLRTest, MonotonicDecayAfterWarmup) {
    LinearLR scheduler(optimizer, *config);

    int warmupEpochs = scheduler.getWarmupEpochs();
    int totalEpochs = scheduler.getTotalEpochs();

    float prevLR = scheduler.getInitialLR();

    for (int epoch = warmupEpochs; epoch < totalEpochs; ++epoch) {
        scheduler.step(epoch);
        float currentLR = scheduler.getCurrentLR();

        EXPECT_LE(currentLR, prevLR + 1e-5f);
        prevLR = currentLR;
    }
}

TEST_F(LinearLRTest, NullOptimizer) {
    EXPECT_THROW(
        LinearLR(nullptr, *config),
        WheelDL::Utils::ConfigurationException
    );
}
