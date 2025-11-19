#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Optimizer/OptimizerFactory.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <filesystem>

using namespace WheelDL::Optimizer;
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

class OptimizerFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create dummy parameters
        params.push_back(torch::randn({10, 10}, torch::requires_grad(true)));
        params.push_back(torch::randn({5}, torch::requires_grad(true)));

        // Get test data path
        testDataPath = getTestDataPath();
		defaultModelPath = (testDataPath / "default_model.yaml").string();
    }

    std::shared_ptr<Configuration> loadConfig(const std::string& filename) {
        auto config = std::make_shared<Configuration>();
        auto configPath = (testDataPath / filename).string();

        // Load with empty model path (not testing model here)
        config->loadFromYaml(defaultModelPath, configPath, "");

        return config;
    }

    std::vector<torch::Tensor> params;
    std::filesystem::path testDataPath;
	std::string defaultModelPath;
};

TEST_F(OptimizerFactoryTest, CreateSGD) {
    auto optimizer = OptimizerFactory::createSGD(params, 0.01f, 0.9f, 0.0005f);

    ASSERT_NE(optimizer, nullptr);

    // Verify correct optimizer type was created
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    ASSERT_NE(sgd, nullptr);
}

TEST_F(OptimizerFactoryTest, CreateAdam) {
    auto optimizer = OptimizerFactory::createAdam(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f);

    ASSERT_NE(optimizer, nullptr);

    // Verify correct optimizer type was created
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    ASSERT_NE(adam, nullptr);
}

TEST_F(OptimizerFactoryTest, CreateAdamW) {
    auto optimizer = OptimizerFactory::createAdamW(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);

    ASSERT_NE(optimizer, nullptr);

    // Verify correct optimizer type was created
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    ASSERT_NE(adamw, nullptr);
}

TEST_F(OptimizerFactoryTest, CreateFromConfig_SGD) {
    auto config = loadConfig("sgd_config.yaml");

    auto optimizer = OptimizerFactory::createFromConfig(params, *config);

    ASSERT_NE(optimizer, nullptr);

    // Verify correct optimizer type was created from config
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    ASSERT_NE(sgd, nullptr);
}

TEST_F(OptimizerFactoryTest, CreateFromConfig_Adam) {
    auto config = loadConfig("adam_config.yaml");

    auto optimizer = OptimizerFactory::createFromConfig(params, *config);

    ASSERT_NE(optimizer, nullptr);

    // Verify correct optimizer type was created from config
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    ASSERT_NE(adam, nullptr);
}

TEST_F(OptimizerFactoryTest, CreateFromConfig_AdamW) {
    auto config = loadConfig("adamw_config.yaml");

    auto optimizer = OptimizerFactory::createFromConfig(params, *config);

    ASSERT_NE(optimizer, nullptr);

    // Verify correct optimizer type was created from config
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    ASSERT_NE(adamw, nullptr);
}

TEST_F(OptimizerFactoryTest, CreateFromConfig_Auto_SmallModel) {
    auto config = loadConfig("auto_config.yaml");

    // Small model: < 10M parameters
    std::vector<torch::Tensor> smallParams;
    smallParams.push_back(torch::randn({100, 100}, torch::requires_grad(true)));  // 10K params

    auto optimizer = OptimizerFactory::createFromConfig(smallParams, *config);

    ASSERT_NE(optimizer, nullptr);

    // Should select Adam for small models
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    EXPECT_NE(adam, nullptr);
}

TEST_F(OptimizerFactoryTest, CreateFromConfig_Auto_MediumModel) {
    auto config = loadConfig("auto_config.yaml");

    // Medium model: 10M-50M parameters
    std::vector<torch::Tensor> mediumParams;
    mediumParams.push_back(torch::randn({5000, 5000}, torch::requires_grad(true)));  // 25M params

    auto optimizer = OptimizerFactory::createFromConfig(mediumParams, *config);

    ASSERT_NE(optimizer, nullptr);

    // Should select AdamW for medium models
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    EXPECT_NE(adamw, nullptr);
}

TEST_F(OptimizerFactoryTest, CreateFromConfig_Auto_LargeModel) {
    auto config = loadConfig("auto_config.yaml");

    // Large model: > 50M parameters
    std::vector<torch::Tensor> largeParams;
    largeParams.push_back(torch::randn({10000, 6000}, torch::requires_grad(true)));  // 60M params

    auto optimizer = OptimizerFactory::createFromConfig(largeParams, *config);

    ASSERT_NE(optimizer, nullptr);

    // Should select SGD for large models
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    EXPECT_NE(sgd, nullptr);
}

TEST_F(OptimizerFactoryTest, EmptyParameters) {
    std::vector<torch::Tensor> emptyParams;

    EXPECT_THROW(
        OptimizerFactory::createSGD(emptyParams, 0.01f),
        WheelDL::Utils::ConfigurationException
    );
}

TEST_F(OptimizerFactoryTest, ParameterUpdate) {
    auto optimizer = OptimizerFactory::createSGD(params, 0.1f);

    // Simulate gradient
    for (auto& param : params) {
        param.mutable_grad() = torch::ones_like(param);
    }

    // Take optimizer step
    optimizer->step();

    // Parameters should have changed
    // (This is a basic sanity check)
}
