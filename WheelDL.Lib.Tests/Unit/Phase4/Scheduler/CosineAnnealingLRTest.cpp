#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Optimizer/Scheduler/CosineAnnealingLR.h"
#include "WheelDL.Lib/Optimizer/OptimizerFactory.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <cmath>
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


class CosineAnnealingLRTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create dummy parameters
        params.push_back(torch::randn({10, 10}, torch::requires_grad(true)));

        // Get test data path
        testDataPath = getTestDataPath();
        defaultModelPath = (testDataPath / "default_model.yaml").string();

        // Load configuration for cosine LR
        config = std::make_shared<Configuration>();
        auto configPath = (testDataPath / "cosine_lr_config.yaml").string();
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

TEST_F(CosineAnnealingLRTest, InitialLR) {
    CosineAnnealingLR scheduler(optimizer, *config);

    float initialLR = scheduler.getInitialLR();
    EXPECT_GT(initialLR, 0.0f);
}

TEST_F(CosineAnnealingLRTest, CosineDecay) {
    CosineAnnealingLR scheduler(optimizer, *config);

    float initialLR = scheduler.getInitialLR();
    float finalLR = scheduler.getFinalLR();
    int totalEpochs = scheduler.getTotalEpochs();
    int warmupEpochs = scheduler.getWarmupEpochs();

    // Skip warmup and test main phase
    int testEpoch = warmupEpochs + (totalEpochs - warmupEpochs) / 2;

    scheduler.step(testEpoch);
    float midLR = scheduler.getCurrentLR();

    // At halfway point, cosine should be between initial and final
    EXPECT_LT(midLR, initialLR);
    EXPECT_GT(midLR, finalLR);
}

TEST_F(CosineAnnealingLRTest, FinalEpochLR) {
    CosineAnnealingLR scheduler(optimizer, *config);

    int totalEpochs = scheduler.getTotalEpochs();
    float finalLR = scheduler.getFinalLR();

    // Step to final epoch
    scheduler.step(totalEpochs - 1);
    float actualFinalLR = scheduler.getCurrentLR();

    // Should be close to final LR
    EXPECT_NEAR(actualFinalLR, finalLR, 0.001f);
}

TEST_F(CosineAnnealingLRTest, WarmupPhase) {
    CosineAnnealingLR scheduler(optimizer, *config);

    int warmupEpochs = scheduler.getWarmupEpochs();

    if (warmupEpochs > 0) {
        // First epoch should have low LR
        scheduler.step(0);
        float firstLR = scheduler.getCurrentLR();

        // Last warmup epoch should approach initial LR
        scheduler.step(warmupEpochs - 1);
        float lastWarmupLR = scheduler.getCurrentLR();

        EXPECT_LT(firstLR, lastWarmupLR);
    }
}

TEST_F(CosineAnnealingLRTest, MonotonicDecayAfterWarmup) {
    CosineAnnealingLR scheduler(optimizer, *config);

    int warmupEpochs = scheduler.getWarmupEpochs();
    int totalEpochs = scheduler.getTotalEpochs();

    float prevLR = scheduler.getInitialLR();

    // Check that LR decreases monotonically after warmup
    for (int epoch = warmupEpochs; epoch < totalEpochs; ++epoch) {
        scheduler.step(epoch);
        float currentLR = scheduler.getCurrentLR();

        EXPECT_LE(currentLR, prevLR + 1e-5f);  // Allow small numerical error
        prevLR = currentLR;
    }
}

TEST_F(CosineAnnealingLRTest, NullOptimizer) {
    EXPECT_THROW(
        CosineAnnealingLR(nullptr, *config),
        WheelDL::Utils::ConfigurationException
    );
}
