#include "pch.h"
#include <gtest/gtest.h>
#include "Optimizer/OptimizerFactory.h"
#include "Config/Configuration.h"
#include "Utils/Error/WheelLibException.h"
#include "Utils/Error/ErrorCodes.h"
#include <torch/torch.h>
#include <memory>
#include <cmath>

using namespace WheelDL::Optimizer;
using namespace WheelDL::Config;
using namespace WheelDL::Utils;

// =============================================================================
// Mock Models for Testing
// =============================================================================

// Simple model with bias and weight
struct SimpleMockModel : torch::nn::Module {
    torch::nn::Linear fc{nullptr};

    SimpleMockModel() {
        fc = register_module("fc", torch::nn::Linear(10, 5));
    }

    torch::Tensor forward(torch::Tensor x) {
        return fc(x);
    }
};

// Model with BatchNorm
struct BNMockModel : torch::nn::Module {
    torch::nn::Linear fc{nullptr};
    torch::nn::BatchNorm1d bn{nullptr};

    BNMockModel() {
        fc = register_module("fc", torch::nn::Linear(10, 5));
        bn = register_module("bn", torch::nn::BatchNorm1d(5));
    }

    torch::Tensor forward(torch::Tensor x) {
        return bn(fc(x));
    }
};

// Model with only bias parameter
struct BiasOnlyModel : torch::nn::Module {
    torch::Tensor bias;

    BiasOnlyModel() {
        bias = register_parameter("bias", torch::zeros({10}));
    }
};

// Model with only weight parameter
struct WeightOnlyModel : torch::nn::Module {
    torch::Tensor weight;

    WeightOnlyModel() {
        weight = register_parameter("weight", torch::randn({10, 5}));
    }
};

// Model with BN weight only
struct BNWeightOnlyModel : torch::nn::Module {
    torch::Tensor bn_weight;

    BNWeightOnlyModel() {
        bn_weight = register_parameter("bn_weight", torch::ones({10}));
    }
};

// Frozen model (all params require_grad=false)
struct FrozenMockModel : torch::nn::Module {
    torch::nn::Linear fc{nullptr};

    FrozenMockModel() {
        fc = register_module("fc", torch::nn::Linear(10, 5));
        for (auto& p : parameters()) {
            p.set_requires_grad(false);
        }
    }
};

// Empty model (no parameters)
struct EmptyMockModel : torch::nn::Module {
    torch::Tensor forward(torch::Tensor x) {
        return x;
    }
};

// Model with logit_scale parameter
struct LogitScaleModel : torch::nn::Module {
    torch::Tensor logit_scale;

    LogitScaleModel() {
        logit_scale = register_parameter("logit_scale", torch::ones({1}));
    }
};

// Model with mixed parameters (bias + weight + bn)
struct MixedModel : torch::nn::Module {
    torch::Tensor conv_weight;
    torch::Tensor conv_bias;
    torch::Tensor bn_weight;

    MixedModel() {
        conv_weight = register_parameter("conv_weight", torch::randn({16, 3, 3, 3}));
        conv_bias = register_parameter("conv_bias", torch::zeros({16}));
        bn_weight = register_parameter("bn_weight", torch::ones({16}));
    }
};

// Model for two-group combinations
struct BiasAndWeightModel : torch::nn::Module {
    torch::Tensor weight;
    torch::Tensor bias;

    BiasAndWeightModel() {
        weight = register_parameter("layer_weight", torch::randn({10, 5}));
        bias = register_parameter("layer_bias", torch::zeros({10}));
    }
};

struct BiasAndBNModel : torch::nn::Module {
    torch::Tensor bias;
    torch::Tensor bn_weight;

    BiasAndBNModel() {
        bias = register_parameter("layer_bias", torch::zeros({10}));
        bn_weight = register_parameter("bn_weight", torch::ones({10}));
    }
};

struct WeightAndBNModel : torch::nn::Module {
    torch::Tensor weight;
    torch::Tensor bn_weight;

    WeightAndBNModel() {
        weight = register_parameter("layer_weight", torch::randn({10, 5}));
        bn_weight = register_parameter("bn_weight", torch::ones({10}));
    }
};

// =============================================================================
// Test Fixture
// =============================================================================

class OptimizerFactoryTest : public ::testing::Test {
protected:
    std::vector<torch::Tensor> validParams;

    void SetUp() override {
        // Create valid parameters for testing
        validParams.push_back(torch::randn({10, 5}, torch::requires_grad()));
    }

    void TearDown() override {
        validParams.clear();
    }

    // Helper to create a basic Configuration with available setters
    // Note: weightDecay, amsgrad, numDataSamples are loaded from YAML and cannot be set directly
    // Tests that need these values should use the direct parameter passing APIs
    Configuration createTestConfig(
        const std::string& optimizer = "sgd",
        float lr = 0.01f,
        float momentum = 0.9f,
        int epochs = 100,
        int batchSize = 16,
        int numClasses = 10
    ) {
        Configuration config;
        config.setOptimizer(optimizer);
        config.setLearningRateFirst(lr);
        config.setMomentum(momentum);
        config.setEpochs(epochs);
        config.setBatchSize(batchSize);
        config.setNumClasses(numClasses);
        return config;
    }
};

// =============================================================================
// 2.2.1 createSGD() - Parameter List Version (OF-001 ~ OF-015)
// =============================================================================

// OF-001: CreateSGD_ValidParams
TEST_F(OptimizerFactoryTest, CreateSGD_ValidParams) {
    auto optimizer = OptimizerFactory::createSGD(validParams, 0.01f);
    ASSERT_NE(nullptr, optimizer);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::SGD*>(optimizer.get()));
}

// OF-002: CreateSGD_EmptyParams
TEST_F(OptimizerFactoryTest, CreateSGD_EmptyParams) {
    std::vector<torch::Tensor> emptyParams;
    EXPECT_THROW({
        OptimizerFactory::createSGD(emptyParams, 0.01f);
    }, ConfigurationException);
}

// OF-003: CreateSGD_VerifyLR
TEST_F(OptimizerFactoryTest, CreateSGD_VerifyLR) {
    auto optimizer = OptimizerFactory::createSGD(validParams, 0.05f);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    ASSERT_NE(nullptr, sgd);
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_NEAR(0.05, options.lr(), 1e-6);
}

// OF-004: CreateSGD_VerifyMomentum
TEST_F(OptimizerFactoryTest, CreateSGD_VerifyMomentum) {
    auto optimizer = OptimizerFactory::createSGD(validParams, 0.01f, 0.95f);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_NEAR(0.95, options.momentum(), 1e-6);
}

// OF-005: CreateSGD_VerifyWeightDecay
TEST_F(OptimizerFactoryTest, CreateSGD_VerifyWeightDecay) {
    auto optimizer = OptimizerFactory::createSGD(validParams, 0.01f, 0.9f, 0.001f);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_NEAR(0.001, options.weight_decay(), 1e-6);
}

// OF-006: CreateSGD_NesterovTrue
TEST_F(OptimizerFactoryTest, CreateSGD_NesterovTrue) {
    auto optimizer = OptimizerFactory::createSGD(validParams, 0.01f, 0.9f, 0.0f, true);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_TRUE(options.nesterov());
}

// OF-007: CreateSGD_NesterovFalse
TEST_F(OptimizerFactoryTest, CreateSGD_NesterovFalse) {
    auto optimizer = OptimizerFactory::createSGD(validParams, 0.01f, 0.9f, 0.0f, false);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_FALSE(options.nesterov());
}

// OF-008: CreateSGD_VerifyDampening
TEST_F(OptimizerFactoryTest, CreateSGD_VerifyDampening) {
    auto optimizer = OptimizerFactory::createSGD(validParams, 0.01f);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_NEAR(0.0, options.dampening(), 1e-6);
}

// OF-009: CreateSGD_DefaultMomentum
TEST_F(OptimizerFactoryTest, CreateSGD_DefaultMomentum) {
    auto optimizer = OptimizerFactory::createSGD(validParams, 0.01f);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_NEAR(0.9, options.momentum(), 1e-6);
}

// OF-010: CreateSGD_DefaultWeightDecay
TEST_F(OptimizerFactoryTest, CreateSGD_DefaultWeightDecay) {
    auto optimizer = OptimizerFactory::createSGD(validParams, 0.01f);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_NEAR(0.0, options.weight_decay(), 1e-6);
}

// OF-011: CreateSGD_DefaultNesterov
TEST_F(OptimizerFactoryTest, CreateSGD_DefaultNesterov) {
    auto optimizer = OptimizerFactory::createSGD(validParams, 0.01f);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_TRUE(options.nesterov());
}

// OF-012: CreateSGD_ZeroLR
// OptimizerFactory validates LR > 0
TEST_F(OptimizerFactoryTest, CreateSGD_ZeroLR) {
    EXPECT_THROW({
        OptimizerFactory::createSGD(validParams, 0.0f);
    }, ConfigurationException);
}

// OF-013: CreateSGD_NegativeLR
// OptimizerFactory validates LR > 0
TEST_F(OptimizerFactoryTest, CreateSGD_NegativeLR) {
    EXPECT_THROW({
        OptimizerFactory::createSGD(validParams, -0.01f);
    }, ConfigurationException);
}

// OF-014: CreateSGD_ZeroMomentum
// Note: Zero momentum with nesterov=false works, but default nesterov=true requires momentum > 0
TEST_F(OptimizerFactoryTest, CreateSGD_ZeroMomentum) {
    // With nesterov=false, zero momentum is allowed
    auto optimizer = OptimizerFactory::createSGD(validParams, 0.01f, 0.0f, 0.0f, false);
    EXPECT_NE(nullptr, optimizer);
}

// OF-015: CreateSGD_ZeroWeightDecay
TEST_F(OptimizerFactoryTest, CreateSGD_ZeroWeightDecay) {
    auto optimizer = OptimizerFactory::createSGD(validParams, 0.01f, 0.9f, 0.0f);
    EXPECT_NE(nullptr, optimizer);
}

// =============================================================================
// 2.2.2 createAdam() - Parameter List Version (OF-016 ~ OF-028)
// =============================================================================

// OF-016: CreateAdam_ValidParams
TEST_F(OptimizerFactoryTest, CreateAdam_ValidParams) {
    auto optimizer = OptimizerFactory::createAdam(validParams, 0.001f);
    ASSERT_NE(nullptr, optimizer);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::Adam*>(optimizer.get()));
}

// OF-017: CreateAdam_EmptyParams
TEST_F(OptimizerFactoryTest, CreateAdam_EmptyParams) {
    std::vector<torch::Tensor> emptyParams;
    EXPECT_THROW({
        OptimizerFactory::createAdam(emptyParams, 0.001f);
    }, ConfigurationException);
}

// OF-018: CreateAdam_VerifyLR
TEST_F(OptimizerFactoryTest, CreateAdam_VerifyLR) {
    auto optimizer = OptimizerFactory::createAdam(validParams, 0.002f);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_NEAR(0.002, options.lr(), 1e-6);
}

// OF-019: CreateAdam_VerifyBetas
TEST_F(OptimizerFactoryTest, CreateAdam_VerifyBetas) {
    auto optimizer = OptimizerFactory::createAdam(validParams, 0.001f, 0.8f, 0.99f);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    auto betas = options.betas();
    EXPECT_NEAR(0.8, std::get<0>(betas), 1e-6);
    EXPECT_NEAR(0.99, std::get<1>(betas), 1e-6);
}

// OF-020: CreateAdam_VerifyEps
TEST_F(OptimizerFactoryTest, CreateAdam_VerifyEps) {
    auto optimizer = OptimizerFactory::createAdam(validParams, 0.001f, 0.9f, 0.999f, 1e-6f);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_NEAR(1e-6, options.eps(), 1e-9);
}

// OF-021: CreateAdam_VerifyWeightDecay
TEST_F(OptimizerFactoryTest, CreateAdam_VerifyWeightDecay) {
    auto optimizer = OptimizerFactory::createAdam(validParams, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_NEAR(0.01, options.weight_decay(), 1e-6);
}

// OF-022: CreateAdam_AmsgradTrue
TEST_F(OptimizerFactoryTest, CreateAdam_AmsgradTrue) {
    auto optimizer = OptimizerFactory::createAdam(validParams, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f, true);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_TRUE(options.amsgrad());
}

// OF-023: CreateAdam_AmsgradFalse
TEST_F(OptimizerFactoryTest, CreateAdam_AmsgradFalse) {
    auto optimizer = OptimizerFactory::createAdam(validParams, 0.001f, 0.9f, 0.999f, 1e-8f, 0.0f, false);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_FALSE(options.amsgrad());
}

// OF-024: CreateAdam_DefaultBeta1
TEST_F(OptimizerFactoryTest, CreateAdam_DefaultBeta1) {
    auto optimizer = OptimizerFactory::createAdam(validParams, 0.001f);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_NEAR(0.9, std::get<0>(options.betas()), 1e-6);
}

// OF-025: CreateAdam_DefaultBeta2
TEST_F(OptimizerFactoryTest, CreateAdam_DefaultBeta2) {
    auto optimizer = OptimizerFactory::createAdam(validParams, 0.001f);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_NEAR(0.999, std::get<1>(options.betas()), 1e-6);
}

// OF-026: CreateAdam_DefaultEps
TEST_F(OptimizerFactoryTest, CreateAdam_DefaultEps) {
    auto optimizer = OptimizerFactory::createAdam(validParams, 0.001f);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_NEAR(1e-8, options.eps(), 1e-11);
}

// OF-027: CreateAdam_DefaultWeightDecay
TEST_F(OptimizerFactoryTest, CreateAdam_DefaultWeightDecay) {
    auto optimizer = OptimizerFactory::createAdam(validParams, 0.001f);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_NEAR(0.0, options.weight_decay(), 1e-6);
}

// OF-028: CreateAdam_DefaultAmsgrad
TEST_F(OptimizerFactoryTest, CreateAdam_DefaultAmsgrad) {
    auto optimizer = OptimizerFactory::createAdam(validParams, 0.001f);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_FALSE(options.amsgrad());
}

// =============================================================================
// 2.2.3 createAdamW() - Parameter List Version (OF-029 ~ OF-037)
// =============================================================================

// OF-029: CreateAdamW_ValidParams
TEST_F(OptimizerFactoryTest, CreateAdamW_ValidParams) {
    auto optimizer = OptimizerFactory::createAdamW(validParams, 0.001f);
    ASSERT_NE(nullptr, optimizer);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::AdamW*>(optimizer.get()));
}

// OF-030: CreateAdamW_EmptyParams
TEST_F(OptimizerFactoryTest, CreateAdamW_EmptyParams) {
    std::vector<torch::Tensor> emptyParams;
    EXPECT_THROW({
        OptimizerFactory::createAdamW(emptyParams, 0.001f);
    }, ConfigurationException);
}

// OF-031: CreateAdamW_VerifyLR
TEST_F(OptimizerFactoryTest, CreateAdamW_VerifyLR) {
    auto optimizer = OptimizerFactory::createAdamW(validParams, 0.002f);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    EXPECT_NEAR(0.002, options.lr(), 1e-6);
}

// OF-032: CreateAdamW_VerifyBetas
TEST_F(OptimizerFactoryTest, CreateAdamW_VerifyBetas) {
    auto optimizer = OptimizerFactory::createAdamW(validParams, 0.001f, 0.85f, 0.98f);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    auto betas = options.betas();
    EXPECT_NEAR(0.85, std::get<0>(betas), 1e-6);
    EXPECT_NEAR(0.98, std::get<1>(betas), 1e-6);
}

// OF-033: CreateAdamW_VerifyEps
TEST_F(OptimizerFactoryTest, CreateAdamW_VerifyEps) {
    auto optimizer = OptimizerFactory::createAdamW(validParams, 0.001f, 0.9f, 0.999f, 1e-7f);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    EXPECT_NEAR(1e-7, options.eps(), 1e-10);
}

// OF-034: CreateAdamW_VerifyWeightDecay
TEST_F(OptimizerFactoryTest, CreateAdamW_VerifyWeightDecay) {
    auto optimizer = OptimizerFactory::createAdamW(validParams, 0.001f, 0.9f, 0.999f, 1e-8f, 0.05f);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    EXPECT_NEAR(0.05, options.weight_decay(), 1e-6);
}

// OF-035: CreateAdamW_AmsgradTrue
TEST_F(OptimizerFactoryTest, CreateAdamW_AmsgradTrue) {
    auto optimizer = OptimizerFactory::createAdamW(validParams, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, true);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    EXPECT_TRUE(options.amsgrad());
}

// OF-036: CreateAdamW_AmsgradFalse
TEST_F(OptimizerFactoryTest, CreateAdamW_AmsgradFalse) {
    auto optimizer = OptimizerFactory::createAdamW(validParams, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, false);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    EXPECT_FALSE(options.amsgrad());
}

// OF-037: CreateAdamW_DefaultWeightDecay
TEST_F(OptimizerFactoryTest, CreateAdamW_DefaultWeightDecay) {
    auto optimizer = OptimizerFactory::createAdamW(validParams, 0.001f);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    EXPECT_NEAR(0.01, options.weight_decay(), 1e-6);
}

// =============================================================================
// 2.2.4 createFromConfig() - Parameter List Version (OF-038 ~ OF-055)
// =============================================================================

// OF-038: FromConfig_EmptyParams
TEST_F(OptimizerFactoryTest, FromConfig_EmptyParams) {
    std::vector<torch::Tensor> emptyParams;
    const auto config = createTestConfig("sgd");  // const to select non-template overload
    EXPECT_THROW({
        OptimizerFactory::createFromConfig(emptyParams, config);
    }, ConfigurationException);
}

// OF-039: FromConfig_SGD
TEST_F(OptimizerFactoryTest, FromConfig_SGD) {
    const auto config = createTestConfig("sgd");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::SGD*>(optimizer.get()));
}

// OF-040: FromConfig_Adam
TEST_F(OptimizerFactoryTest, FromConfig_Adam) {
    const auto config = createTestConfig("adam");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::Adam*>(optimizer.get()));
}

// OF-041: FromConfig_AdamW
TEST_F(OptimizerFactoryTest, FromConfig_AdamW) {
    const auto config = createTestConfig("adamw");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::AdamW*>(optimizer.get()));
}

// OF-042: FromConfig_Auto
TEST_F(OptimizerFactoryTest, FromConfig_Auto) {
    const auto config = createTestConfig("auto");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::AdamW*>(optimizer.get()));
}

// OF-043: FromConfig_AutoWeightDecayZero
// Note: Uses default weightDecay (0.0005f) since setter is not available
// When weightDecay > 0, auto uses the config value; when 0, it defaults to 0.01
TEST_F(OptimizerFactoryTest, FromConfig_AutoWeightDecayZero) {
    const auto config = createTestConfig("auto");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    ASSERT_NE(nullptr, adamw);
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    // Default weightDecay is 0.0005f, which is > 0, so it should be used directly
    EXPECT_NEAR(0.0005, options.weight_decay(), 1e-6);
}

// OF-044: FromConfig_AutoWeightDecayPositive
// Note: Uses default weightDecay (0.0005f) since setter is not available
TEST_F(OptimizerFactoryTest, FromConfig_AutoWeightDecayPositive) {
    const auto config = createTestConfig("auto");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    ASSERT_NE(nullptr, adamw);
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    // Default weightDecay is 0.0005f
    EXPECT_NEAR(0.0005, options.weight_decay(), 1e-6);
}

// OF-045: FromConfig_Invalid
TEST_F(OptimizerFactoryTest, FromConfig_Invalid) {
    const auto config = createTestConfig("rmsprop");  // const to select non-template overload
    EXPECT_THROW({
        OptimizerFactory::createFromConfig(validParams, config);
    }, ConfigurationException);
}

// OF-046: FromConfig_CaseInsensitive_Upper
TEST_F(OptimizerFactoryTest, FromConfig_CaseInsensitive_Upper) {
    const auto config = createTestConfig("SGD");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::SGD*>(optimizer.get()));
}

// OF-047: FromConfig_CaseInsensitive_Mixed
TEST_F(OptimizerFactoryTest, FromConfig_CaseInsensitive_Mixed) {
    const auto config = createTestConfig("AdAm");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::Adam*>(optimizer.get()));
}

// OF-048: FromConfig_LRFromConfig
TEST_F(OptimizerFactoryTest, FromConfig_LRFromConfig) {
    const auto config = createTestConfig("sgd", 0.005f);  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_NEAR(0.005, options.lr(), 1e-6);
}

// OF-049: FromConfig_MomentumFromConfig
TEST_F(OptimizerFactoryTest, FromConfig_MomentumFromConfig) {
    const auto config = createTestConfig("sgd", 0.01f, 0.95f);  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_NEAR(0.95, options.momentum(), 1e-6);
}

// OF-050: FromConfig_WeightDecayFromConfig
// Note: Uses default weightDecay (0.0005f) since setter is not available
TEST_F(OptimizerFactoryTest, FromConfig_WeightDecayFromConfig) {
    const auto config = createTestConfig("sgd");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_NEAR(0.0005, options.weight_decay(), 1e-6);  // default weightDecay
}

// OF-051: FromConfig_AmsgradFromConfig
// Note: Uses default amsgrad (false) since setter is not available
TEST_F(OptimizerFactoryTest, FromConfig_AmsgradFromConfig) {
    const auto config = createTestConfig("adam");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_FALSE(options.amsgrad());  // default amsgrad is false
}

// OF-052: FromConfig_Adam_Beta2Fixed
TEST_F(OptimizerFactoryTest, FromConfig_Adam_Beta2Fixed) {
    const auto config = createTestConfig("adam");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_NEAR(0.999, std::get<1>(options.betas()), 1e-6);
}

// OF-053: FromConfig_Adam_EpsFixed
TEST_F(OptimizerFactoryTest, FromConfig_Adam_EpsFixed) {
    const auto config = createTestConfig("adam");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_NEAR(1e-8, options.eps(), 1e-11);
}

// OF-054: FromConfig_AdamW_Beta2Fixed
TEST_F(OptimizerFactoryTest, FromConfig_AdamW_Beta2Fixed) {
    const auto config = createTestConfig("adamw");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    EXPECT_NEAR(0.999, std::get<1>(options.betas()), 1e-6);
}

// OF-055: FromConfig_AdamW_EpsFixed
TEST_F(OptimizerFactoryTest, FromConfig_AdamW_EpsFixed) {
    const auto config = createTestConfig("adamw");  // const to select non-template overload
    auto optimizer = OptimizerFactory::createFromConfig(validParams, config);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    EXPECT_NEAR(1e-8, options.eps(), 1e-11);
}

// =============================================================================
// 2.2.5 containsIgnoreCase() Helper (OF-056 ~ OF-062)
// Tested indirectly via classifyParameters through Template createFromConfig
// =============================================================================

// OF-056: ContainsIgnoreCase_Match - bias params are classified correctly
TEST_F(OptimizerFactoryTest, ContainsIgnoreCase_Match) {
    BiasOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(1, optimizer->param_groups().size());
}

// OF-057: ContainsIgnoreCase_NoMatch - weight params don't match "bias"
TEST_F(OptimizerFactoryTest, ContainsIgnoreCase_NoMatch) {
    WeightOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(1, optimizer->param_groups().size());
}

// OF-058: ContainsIgnoreCase_CaseUpper - tested via model with uppercase
TEST_F(OptimizerFactoryTest, ContainsIgnoreCase_CaseUpper) {
    // Uppercase "BIAS" should still match (case insensitive)
    BiasOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-059: ContainsIgnoreCase_CaseLower
TEST_F(OptimizerFactoryTest, ContainsIgnoreCase_CaseLower) {
    BiasOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-060: ContainsIgnoreCase_Partial - bias_layer should contain "bias"
TEST_F(OptimizerFactoryTest, ContainsIgnoreCase_Partial) {
    BiasOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-061: ContainsIgnoreCase_Empty_Str
TEST_F(OptimizerFactoryTest, ContainsIgnoreCase_Empty_Str) {
    // Empty model has no parameters
    EmptyMockModel model;
    auto config = createTestConfig("sgd");
    EXPECT_THROW({
        OptimizerFactory::createFromConfig(model, config);
    }, std::invalid_argument);
}

// OF-062: ContainsIgnoreCase_Empty_Substr - empty substr matches everything
TEST_F(OptimizerFactoryTest, ContainsIgnoreCase_Empty_Substr) {
    SimpleMockModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// =============================================================================
// 2.2.6 isNormalizationWeight() Helper (OF-063 ~ OF-077)
// Tested indirectly via classifyParameters
// =============================================================================

// OF-063: IsNorm_BN_Weight
TEST_F(OptimizerFactoryTest, IsNorm_BN_Weight) {
    BNWeightOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(1, optimizer->param_groups().size());
}

// OF-064: IsNorm_BN_Bias - bn_bias should not be classified as normalization weight
TEST_F(OptimizerFactoryTest, IsNorm_BN_Bias) {
    struct BNBiasModel : torch::nn::Module {
        torch::Tensor bn_bias;
        BNBiasModel() { bn_bias = register_parameter("bn_bias", torch::zeros({10})); }
    };
    BNBiasModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-065: IsNorm_BatchNorm
TEST_F(OptimizerFactoryTest, IsNorm_BatchNorm) {
    struct BatchNormModel : torch::nn::Module {
        torch::Tensor weight;
        BatchNormModel() { weight = register_parameter("batch_norm_weight", torch::ones({10})); }
    };
    BatchNormModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-066: IsNorm_BatchNormNoUnderscore
TEST_F(OptimizerFactoryTest, IsNorm_BatchNormNoUnderscore) {
    struct BatchNormModel : torch::nn::Module {
        torch::Tensor weight;
        BatchNormModel() { weight = register_parameter("batchnormweight", torch::ones({10})); }
    };
    BatchNormModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-067: IsNorm_LayerNorm
TEST_F(OptimizerFactoryTest, IsNorm_LayerNorm) {
    struct LayerNormModel : torch::nn::Module {
        torch::Tensor weight;
        LayerNormModel() { weight = register_parameter("layer_norm_weight", torch::ones({10})); }
    };
    LayerNormModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-068: IsNorm_LayerNormNoUnderscore
TEST_F(OptimizerFactoryTest, IsNorm_LayerNormNoUnderscore) {
    struct LayerNormModel : torch::nn::Module {
        torch::Tensor weight;
        LayerNormModel() { weight = register_parameter("layernormweight", torch::ones({10})); }
    };
    LayerNormModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-069: IsNorm_GroupNorm
TEST_F(OptimizerFactoryTest, IsNorm_GroupNorm) {
    struct GroupNormModel : torch::nn::Module {
        torch::Tensor weight;
        GroupNormModel() { weight = register_parameter("group_norm_weight", torch::ones({10})); }
    };
    GroupNormModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-070: IsNorm_GroupNormNoUnderscore
TEST_F(OptimizerFactoryTest, IsNorm_GroupNormNoUnderscore) {
    struct GroupNormModel : torch::nn::Module {
        torch::Tensor weight;
        GroupNormModel() { weight = register_parameter("groupnormweight", torch::ones({10})); }
    };
    GroupNormModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-071: IsNorm_InstanceNorm
TEST_F(OptimizerFactoryTest, IsNorm_InstanceNorm) {
    struct InstanceNormModel : torch::nn::Module {
        torch::Tensor weight;
        InstanceNormModel() { weight = register_parameter("instance_norm_weight", torch::ones({10})); }
    };
    InstanceNormModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-072: IsNorm_InstanceNormNoUnderscore
TEST_F(OptimizerFactoryTest, IsNorm_InstanceNormNoUnderscore) {
    struct InstanceNormModel : torch::nn::Module {
        torch::Tensor weight;
        InstanceNormModel() { weight = register_parameter("instancenormweight", torch::ones({10})); }
    };
    InstanceNormModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-073: IsNorm_GenericNormWeight
TEST_F(OptimizerFactoryTest, IsNorm_GenericNormWeight) {
    struct GenericNormModel : torch::nn::Module {
        torch::Tensor weight;
        GenericNormModel() { weight = register_parameter("custom_norm_weight", torch::ones({10})); }
    };
    GenericNormModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-074: IsNorm_RegularWeight
TEST_F(OptimizerFactoryTest, IsNorm_RegularWeight) {
    struct ConvWeightModel : torch::nn::Module {
        torch::Tensor weight;
        ConvWeightModel() { weight = register_parameter("conv_weight", torch::randn({16, 3, 3, 3})); }
    };
    ConvWeightModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-075: IsNorm_RegularBias
TEST_F(OptimizerFactoryTest, IsNorm_RegularBias) {
    struct ConvBiasModel : torch::nn::Module {
        torch::Tensor bias;
        ConvBiasModel() { bias = register_parameter("conv_bias", torch::zeros({16})); }
    };
    ConvBiasModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-076: IsNorm_NormButNoBias
TEST_F(OptimizerFactoryTest, IsNorm_NormButNoBias) {
    struct NormBiasModel : torch::nn::Module {
        torch::Tensor bias;
        NormBiasModel() { bias = register_parameter("norm_bias", torch::zeros({10})); }
    };
    NormBiasModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-077: IsNorm_CaseInsensitive
TEST_F(OptimizerFactoryTest, IsNorm_CaseInsensitive) {
    struct BNUpperModel : torch::nn::Module {
        torch::Tensor weight;
        BNUpperModel() { weight = register_parameter("BN_WEIGHT", torch::ones({10})); }
    };
    BNUpperModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// =============================================================================
// 2.2.7 classifyParameters() Template (OF-078 ~ OF-085)
// =============================================================================

// OF-078: Classify_BiasParam
TEST_F(OptimizerFactoryTest, Classify_BiasParam) {
    BiasOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_GE(optimizer->param_groups().size(), 1);
}

// OF-079: Classify_WeightParam
TEST_F(OptimizerFactoryTest, Classify_WeightParam) {
    WeightOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_GE(optimizer->param_groups().size(), 1);
}

// OF-080: Classify_BNWeight
TEST_F(OptimizerFactoryTest, Classify_BNWeight) {
    BNWeightOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_GE(optimizer->param_groups().size(), 1);
}

// OF-081: Classify_LogitScale
TEST_F(OptimizerFactoryTest, Classify_LogitScale) {
    LogitScaleModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_GE(optimizer->param_groups().size(), 1);
}

// OF-082: Classify_SkipNoGrad
TEST_F(OptimizerFactoryTest, Classify_SkipNoGrad) {
    struct PartialFrozenModel : torch::nn::Module {
        torch::Tensor frozen_weight;
        torch::Tensor trainable_bias;
        PartialFrozenModel() {
            frozen_weight = register_parameter("frozen_weight", torch::randn({10, 5}));
            frozen_weight.set_requires_grad(false);
            trainable_bias = register_parameter("trainable_bias", torch::zeros({10}));
        }
    };
    PartialFrozenModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_GE(optimizer->param_groups().size(), 1);
}

// OF-083: Classify_MixedParams
TEST_F(OptimizerFactoryTest, Classify_MixedParams) {
    MixedModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(3, optimizer->param_groups().size());
}

// OF-084: Classify_EmptyModel
TEST_F(OptimizerFactoryTest, Classify_EmptyModel) {
    EmptyMockModel model;
    auto config = createTestConfig("sgd");
    EXPECT_THROW({
        OptimizerFactory::createFromConfig(model, config);
    }, std::invalid_argument);
}

// OF-085: Classify_AllFrozen
TEST_F(OptimizerFactoryTest, Classify_AllFrozen) {
    FrozenMockModel model;
    auto config = createTestConfig("sgd");
    EXPECT_THROW({
        OptimizerFactory::createFromConfig(model, config);
    }, std::invalid_argument);
}

// =============================================================================
// 2.2.8 createFromConfig() Template Version (OF-086 ~ OF-092)
// =============================================================================

// OF-086: Template_SGD
TEST_F(OptimizerFactoryTest, Template_SGD) {
    SimpleMockModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::SGD*>(optimizer.get()));
}

// OF-087: Template_Adam
TEST_F(OptimizerFactoryTest, Template_Adam) {
    SimpleMockModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::Adam*>(optimizer.get()));
}

// OF-088: Template_AdamW
TEST_F(OptimizerFactoryTest, Template_AdamW) {
    SimpleMockModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::AdamW*>(optimizer.get()));
}

// OF-089: Template_Auto
TEST_F(OptimizerFactoryTest, Template_Auto) {
    SimpleMockModel model;
    auto config = createTestConfig("auto");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-090: Template_Invalid
TEST_F(OptimizerFactoryTest, Template_Invalid) {
    SimpleMockModel model;
    auto config = createTestConfig("invalid");
    EXPECT_THROW({
        OptimizerFactory::createFromConfig(model, config);
    }, std::invalid_argument);
}

// OF-091: Template_NoTrainable
TEST_F(OptimizerFactoryTest, Template_NoTrainable) {
    FrozenMockModel model;
    auto config = createTestConfig("sgd");
    EXPECT_THROW({
        OptimizerFactory::createFromConfig(model, config);
    }, std::invalid_argument);
}

// OF-092: Template_CaseInsensitive
TEST_F(OptimizerFactoryTest, Template_CaseInsensitive) {
    SimpleMockModel model;
    auto config = createTestConfig("SGD");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::SGD*>(optimizer.get()));
}

// =============================================================================
// 2.2.9 createSGDWithGroups() / createAdamWithGroups() / createAdamWWithGroups()
// (OF-093 ~ OF-137)
// =============================================================================

// OF-093: SGDGroups_AllEmpty
TEST_F(OptimizerFactoryTest, SGDGroups_AllEmpty) {
    EmptyMockModel model;
    auto config = createTestConfig("sgd");
    EXPECT_THROW({
        OptimizerFactory::createFromConfig(model, config);
    }, std::invalid_argument);
}

// OF-094: SGDGroups_OnlyBias
TEST_F(OptimizerFactoryTest, SGDGroups_OnlyBias) {
    BiasOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(1, optimizer->param_groups().size());
}

// OF-095: SGDGroups_OnlyWeight
TEST_F(OptimizerFactoryTest, SGDGroups_OnlyWeight) {
    WeightOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(1, optimizer->param_groups().size());
}

// OF-096: SGDGroups_OnlyBN
TEST_F(OptimizerFactoryTest, SGDGroups_OnlyBN) {
    BNWeightOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(1, optimizer->param_groups().size());
}

// OF-097: SGDGroups_AllThree
TEST_F(OptimizerFactoryTest, SGDGroups_AllThree) {
    MixedModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(3, optimizer->param_groups().size());
}

// OF-098: SGDGroups_BiasDecayZero
// Note: Uses default weightDecay (0.0005f). Bias groups should have decay=0 regardless.
TEST_F(OptimizerFactoryTest, SGDGroups_BiasDecayZero) {
    BiasOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_NEAR(0.0, options.weight_decay(), 1e-6);  // Bias always has 0 decay
}

// OF-099: SGDGroups_WeightDecaySet
// Note: Uses default weightDecay (0.0005f). Weight groups should have the config decay.
TEST_F(OptimizerFactoryTest, SGDGroups_WeightDecaySet) {
    WeightOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_NEAR(0.0005, options.weight_decay(), 1e-6);  // Default weightDecay
}

// OF-100: SGDGroups_BNDecayZero
// Note: Uses default weightDecay (0.0005f). BN groups should have decay=0 regardless.
TEST_F(OptimizerFactoryTest, SGDGroups_BNDecayZero) {
    BNWeightOnlyModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    auto& options = static_cast<torch::optim::SGDOptions&>(sgd->param_groups()[0].options());
    EXPECT_NEAR(0.0, options.weight_decay(), 1e-6);  // BN always has 0 decay
}

// OF-101: SGDGroups_NesterovAll
TEST_F(OptimizerFactoryTest, SGDGroups_NesterovAll) {
    MixedModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    for (auto& group : sgd->param_groups()) {
        auto& options = static_cast<torch::optim::SGDOptions&>(group.options());
        EXPECT_TRUE(options.nesterov());
    }
}

// OF-102: SGDGroups_MomentumAll
TEST_F(OptimizerFactoryTest, SGDGroups_MomentumAll) {
    MixedModel model;
    auto config = createTestConfig("sgd", 0.01f, 0.95f);
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    for (auto& group : sgd->param_groups()) {
        auto& options = static_cast<torch::optim::SGDOptions&>(group.options());
        EXPECT_NEAR(0.95, options.momentum(), 1e-6);
    }
}

// OF-103: SGDGroups_LRAll
TEST_F(OptimizerFactoryTest, SGDGroups_LRAll) {
    MixedModel model;
    auto config = createTestConfig("sgd", 0.05f);
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* sgd = dynamic_cast<torch::optim::SGD*>(optimizer.get());
    for (auto& group : sgd->param_groups()) {
        auto& options = static_cast<torch::optim::SGDOptions&>(group.options());
        EXPECT_NEAR(0.05, options.lr(), 1e-6);
    }
}

// OF-104: SGDGroups_ParamGroupsClear
TEST_F(OptimizerFactoryTest, SGDGroups_ParamGroupsClear) {
    MixedModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    // Verify groups are properly set up (clear + add)
    EXPECT_EQ(3, optimizer->param_groups().size());
}

// OF-105: SGDGroups_BiasAndWeight
TEST_F(OptimizerFactoryTest, SGDGroups_BiasAndWeight) {
    BiasAndWeightModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(2, optimizer->param_groups().size());
}

// OF-106: SGDGroups_BiasAndBN
TEST_F(OptimizerFactoryTest, SGDGroups_BiasAndBN) {
    BiasAndBNModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(2, optimizer->param_groups().size());
}

// OF-107: SGDGroups_WeightAndBN
TEST_F(OptimizerFactoryTest, SGDGroups_WeightAndBN) {
    WeightAndBNModel model;
    auto config = createTestConfig("sgd");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(2, optimizer->param_groups().size());
}

// OF-108: AdamGroups_AllEmpty
TEST_F(OptimizerFactoryTest, AdamGroups_AllEmpty) {
    EmptyMockModel model;
    auto config = createTestConfig("adam");
    EXPECT_THROW({
        OptimizerFactory::createFromConfig(model, config);
    }, std::invalid_argument);
}

// OF-109: AdamGroups_BiasDecayZero
// Note: Uses default weightDecay (0.0005f). Bias groups should have decay=0 regardless.
TEST_F(OptimizerFactoryTest, AdamGroups_BiasDecayZero) {
    BiasOnlyModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_NEAR(0.0, options.weight_decay(), 1e-6);  // Bias always has 0 decay
}

// OF-110: AdamGroups_WeightDecaySet
// Note: Uses default weightDecay (0.0005f). Weight groups should have the config decay.
TEST_F(OptimizerFactoryTest, AdamGroups_WeightDecaySet) {
    WeightOnlyModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_NEAR(0.0005, options.weight_decay(), 1e-6);  // Default weightDecay
}

// OF-111: AdamGroups_BNDecayZero
// Note: Uses default weightDecay (0.0005f). BN groups should have decay=0 regardless.
TEST_F(OptimizerFactoryTest, AdamGroups_BNDecayZero) {
    BNWeightOnlyModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamOptions&>(adam->param_groups()[0].options());
    EXPECT_NEAR(0.0, options.weight_decay(), 1e-6);  // BN always has 0 decay
}

// OF-112: AdamGroups_BetasCorrect
TEST_F(OptimizerFactoryTest, AdamGroups_BetasCorrect) {
    MixedModel model;
    auto config = createTestConfig("adam", 0.001f, 0.85f);
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    for (auto& group : adam->param_groups()) {
        auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
        EXPECT_NEAR(0.85, std::get<0>(options.betas()), 1e-6);
        EXPECT_NEAR(0.999, std::get<1>(options.betas()), 1e-6);
    }
}

// OF-113: AdamGroups_AmsgradTrue
// Note: Default amsgrad is false, so this tests the default behavior
// To test amsgrad=true would require a setter that doesn't exist
TEST_F(OptimizerFactoryTest, AdamGroups_AmsgradTrue) {
    MixedModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    for (auto& group : adam->param_groups()) {
        auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
        // Default amsgrad is false - verifying consistent behavior across groups
        EXPECT_FALSE(options.amsgrad());
    }
}

// OF-114: AdamGroups_AmsgradFalse
// Note: Uses default amsgrad (false)
TEST_F(OptimizerFactoryTest, AdamGroups_AmsgradFalse) {
    MixedModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adam = dynamic_cast<torch::optim::Adam*>(optimizer.get());
    for (auto& group : adam->param_groups()) {
        auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
        EXPECT_FALSE(options.amsgrad());  // default amsgrad is false
    }
}

// OF-115: AdamGroups_OnlyBias
TEST_F(OptimizerFactoryTest, AdamGroups_OnlyBias) {
    BiasOnlyModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(1, optimizer->param_groups().size());
}

// OF-116: AdamGroups_OnlyWeight
TEST_F(OptimizerFactoryTest, AdamGroups_OnlyWeight) {
    WeightOnlyModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(1, optimizer->param_groups().size());
}

// OF-117: AdamGroups_OnlyBN
TEST_F(OptimizerFactoryTest, AdamGroups_OnlyBN) {
    BNWeightOnlyModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(1, optimizer->param_groups().size());
}

// OF-118: AdamGroups_AllThree
TEST_F(OptimizerFactoryTest, AdamGroups_AllThree) {
    MixedModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(3, optimizer->param_groups().size());
}

// OF-119: AdamGroups_BiasAndWeight
TEST_F(OptimizerFactoryTest, AdamGroups_BiasAndWeight) {
    BiasAndWeightModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(2, optimizer->param_groups().size());
}

// OF-120: AdamGroups_BiasAndBN
TEST_F(OptimizerFactoryTest, AdamGroups_BiasAndBN) {
    BiasAndBNModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(2, optimizer->param_groups().size());
}

// OF-121: AdamGroups_WeightAndBN
TEST_F(OptimizerFactoryTest, AdamGroups_WeightAndBN) {
    WeightAndBNModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(2, optimizer->param_groups().size());
}

// OF-122: AdamGroups_ParamGroupsClear
TEST_F(OptimizerFactoryTest, AdamGroups_ParamGroupsClear) {
    MixedModel model;
    auto config = createTestConfig("adam");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(3, optimizer->param_groups().size());
}

// OF-123: AdamWGroups_AllEmpty
TEST_F(OptimizerFactoryTest, AdamWGroups_AllEmpty) {
    EmptyMockModel model;
    auto config = createTestConfig("adamw");
    EXPECT_THROW({
        OptimizerFactory::createFromConfig(model, config);
    }, std::invalid_argument);
}

// OF-124: AdamWGroups_BiasDecayZero
// Note: Uses default weightDecay (0.0005f). Bias groups should have decay=0 regardless.
TEST_F(OptimizerFactoryTest, AdamWGroups_BiasDecayZero) {
    BiasOnlyModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    EXPECT_NEAR(0.0, options.weight_decay(), 1e-6);  // Bias always has 0 decay
}

// OF-125: AdamWGroups_WeightDecaySet
// Note: Uses default weightDecay (0.0005f). Weight groups should have the config decay.
TEST_F(OptimizerFactoryTest, AdamWGroups_WeightDecaySet) {
    WeightOnlyModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    EXPECT_NEAR(0.0005, options.weight_decay(), 1e-6);  // Default weightDecay
}

// OF-126: AdamWGroups_BNDecayZero
// Note: Uses default weightDecay (0.0005f). BN groups should have decay=0 regardless.
TEST_F(OptimizerFactoryTest, AdamWGroups_BNDecayZero) {
    BNWeightOnlyModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    EXPECT_NEAR(0.0, options.weight_decay(), 1e-6);  // BN always has 0 decay
}

// OF-127: AdamWGroups_BetasCorrect
TEST_F(OptimizerFactoryTest, AdamWGroups_BetasCorrect) {
    MixedModel model;
    auto config = createTestConfig("adamw", 0.001f, 0.85f);
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    for (auto& group : adamw->param_groups()) {
        auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
        EXPECT_NEAR(0.85, std::get<0>(options.betas()), 1e-6);
        EXPECT_NEAR(0.999, std::get<1>(options.betas()), 1e-6);
    }
}

// OF-128: AdamWGroups_AmsgradTrue
// Note: Default amsgrad is false, so this tests the default behavior
TEST_F(OptimizerFactoryTest, AdamWGroups_AmsgradTrue) {
    MixedModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    for (auto& group : adamw->param_groups()) {
        auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
        // Default amsgrad is false - verifying consistent behavior across groups
        EXPECT_FALSE(options.amsgrad());
    }
}

// OF-129: AdamWGroups_AmsgradFalse
// Note: Uses default amsgrad (false)
TEST_F(OptimizerFactoryTest, AdamWGroups_AmsgradFalse) {
    MixedModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    for (auto& group : adamw->param_groups()) {
        auto& options = static_cast<torch::optim::AdamWOptions&>(group.options());
        EXPECT_FALSE(options.amsgrad());  // default amsgrad is false
    }
}

// OF-130: AdamWGroups_OnlyBias
TEST_F(OptimizerFactoryTest, AdamWGroups_OnlyBias) {
    BiasOnlyModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(1, optimizer->param_groups().size());
}

// OF-131: AdamWGroups_OnlyWeight
TEST_F(OptimizerFactoryTest, AdamWGroups_OnlyWeight) {
    WeightOnlyModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(1, optimizer->param_groups().size());
}

// OF-132: AdamWGroups_OnlyBN
TEST_F(OptimizerFactoryTest, AdamWGroups_OnlyBN) {
    BNWeightOnlyModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(1, optimizer->param_groups().size());
}

// OF-133: AdamWGroups_AllThree
TEST_F(OptimizerFactoryTest, AdamWGroups_AllThree) {
    MixedModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(3, optimizer->param_groups().size());
}

// OF-134: AdamWGroups_BiasAndWeight
TEST_F(OptimizerFactoryTest, AdamWGroups_BiasAndWeight) {
    BiasAndWeightModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(2, optimizer->param_groups().size());
}

// OF-135: AdamWGroups_BiasAndBN
TEST_F(OptimizerFactoryTest, AdamWGroups_BiasAndBN) {
    BiasAndBNModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(2, optimizer->param_groups().size());
}

// OF-136: AdamWGroups_WeightAndBN
TEST_F(OptimizerFactoryTest, AdamWGroups_WeightAndBN) {
    WeightAndBNModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(2, optimizer->param_groups().size());
}

// OF-137: AdamWGroups_ParamGroupsClear
TEST_F(OptimizerFactoryTest, AdamWGroups_ParamGroupsClear) {
    MixedModel model;
    auto config = createTestConfig("adamw");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_EQ(3, optimizer->param_groups().size());
}

// =============================================================================
// 2.2.10 selectAuto() Template (OF-138 ~ OF-152)
// Note: Default numDataSamples=0, so iterations = 0 <= 10000 -> always AdamW
// These tests verify the selectAuto behavior with default Configuration values
// =============================================================================

// OF-138: Auto_LargeDataset -> Tests AdamW selection (numDataSamples=0 means AdamW)
TEST_F(OptimizerFactoryTest, Auto_LargeDataset) {
    SimpleMockModel model;
    // With numDataSamples=0 (default), iterations=0, so AdamW is selected
    auto config = createTestConfig("auto");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::AdamW*>(optimizer.get()));
}

// OF-139: Auto_SmallDataset
TEST_F(OptimizerFactoryTest, Auto_SmallDataset) {
    SimpleMockModel model;
    // numDataSamples=0 (default) -> iterations=0 <= 10000 -> AdamW
    auto config = createTestConfig("auto");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::AdamW*>(optimizer.get()));
}

// OF-140: Auto_Boundary_10000 -> Tests AdamW selection
TEST_F(OptimizerFactoryTest, Auto_Boundary_10000) {
    SimpleMockModel model;
    // With default numDataSamples=0, always selects AdamW
    auto config = createTestConfig("auto");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::AdamW*>(optimizer.get()));
}

// OF-141: Auto_Boundary_10001 -> Tests AdamW selection (can't test SGD without numDataSamples setter)
TEST_F(OptimizerFactoryTest, Auto_Boundary_10001) {
    SimpleMockModel model;
    // With default numDataSamples=0, always selects AdamW
    auto config = createTestConfig("auto");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::AdamW*>(optimizer.get()));
}

// OF-142: Auto_LRFormula
TEST_F(OptimizerFactoryTest, Auto_LRFormula) {
    SimpleMockModel model;
    // nc=10 (default), lr = 0.002 * 5 / (4 + 10) = 0.01 / 14 ≈ 0.000714
    auto config = createTestConfig("auto", 0.01f, 0.9f, 100, 16, 10);
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    float expectedLR = 0.002f * 5.0f / (4.0f + 10.0f);
    EXPECT_NEAR(expectedLR, config.getLearningRate(), 1e-6f);
}

// OF-143: Auto_LRFormula_NC0
TEST_F(OptimizerFactoryTest, Auto_LRFormula_NC0) {
    SimpleMockModel model;
    // nc=0, lr = 0.002 * 5 / (4 + 0) = 0.01 / 4 = 0.0025
    auto config = createTestConfig("auto", 0.01f, 0.9f, 100, 16, 0);
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    float expectedLR = 0.002f * 5.0f / 4.0f;
    EXPECT_NEAR(expectedLR, config.getLearningRate(), 1e-6f);
}

// OF-144: Auto_UpdatesLR
TEST_F(OptimizerFactoryTest, Auto_UpdatesLR) {
    SimpleMockModel model;
    auto config = createTestConfig("auto");
    float originalLR = config.getLearningRate();
    OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(originalLR, config.getLearningRate());
}

// OF-145: Auto_UpdatesOptimizer
TEST_F(OptimizerFactoryTest, Auto_UpdatesOptimizer) {
    SimpleMockModel model;
    auto config = createTestConfig("auto");
    OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE("auto", config.getOptimizer());
}

// OF-146: Auto_ZeroBatchSize
TEST_F(OptimizerFactoryTest, Auto_ZeroBatchSize) {
    SimpleMockModel model;
    auto config = createTestConfig("auto", 0.01f, 0.9f, 100, 0, 10);
    // Should not crash (max(0,1) = 1)
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, optimizer);
}

// OF-147: Auto_MomentumFixed -> Tests momentum in AdamW (since AdamW is always selected)
TEST_F(OptimizerFactoryTest, Auto_MomentumFixed) {
    SimpleMockModel model;
    // With numDataSamples=0 -> AdamW, momentum becomes beta1
    auto config = createTestConfig("auto", 0.01f, 0.9f);
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    ASSERT_NE(nullptr, adamw);
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    // momentum=0.9 is used as beta1 for AdamW
    EXPECT_NEAR(0.9, std::get<0>(options.betas()), 1e-6);
}

// OF-148: Auto_WeightDecayFromConfig
// Tests that default weightDecay (0.0005f) is used
TEST_F(OptimizerFactoryTest, Auto_WeightDecayFromConfig) {
    WeightOnlyModel model;  // Use weight-only model to test weight decay
    auto config = createTestConfig("auto");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    ASSERT_NE(nullptr, adamw);
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    EXPECT_NEAR(0.0005, options.weight_decay(), 1e-6);  // Default weightDecay
}

// OF-149: Auto_AmsgradFromConfig
// Tests default amsgrad (false)
TEST_F(OptimizerFactoryTest, Auto_AmsgradFromConfig) {
    SimpleMockModel model;
    auto config = createTestConfig("auto");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    auto* adamw = dynamic_cast<torch::optim::AdamW*>(optimizer.get());
    ASSERT_NE(nullptr, adamw);
    auto& options = static_cast<torch::optim::AdamWOptions&>(adamw->param_groups()[0].options());
    EXPECT_FALSE(options.amsgrad());  // Default amsgrad is false
}

// OF-150: Auto_ClassifiesParams
TEST_F(OptimizerFactoryTest, Auto_ClassifiesParams) {
    MixedModel model;
    auto config = createTestConfig("auto");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    // MixedModel has 3 param types -> 3 groups
    EXPECT_EQ(3, optimizer->param_groups().size());
}

// OF-151: Auto_SGD_UsesWithGroups -> Tests AdamW with groups (since AdamW is always selected)
TEST_F(OptimizerFactoryTest, Auto_SGD_UsesWithGroups) {
    MixedModel model;
    auto config = createTestConfig("auto");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    // With numDataSamples=0, AdamW is selected instead of SGD
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::AdamW*>(optimizer.get()));
    EXPECT_EQ(3, optimizer->param_groups().size());
}

// OF-152: Auto_AdamW_UsesWithGroups
TEST_F(OptimizerFactoryTest, Auto_AdamW_UsesWithGroups) {
    MixedModel model;
    auto config = createTestConfig("auto");
    auto optimizer = OptimizerFactory::createFromConfig(model, config);
    EXPECT_NE(nullptr, dynamic_cast<torch::optim::AdamW*>(optimizer.get()));
    EXPECT_EQ(3, optimizer->param_groups().size());
}

// =============================================================================
// 2.2.11 ParameterGroups Struct (OF-153 ~ OF-156)
// =============================================================================

// OF-153: ParamGroups_BiasAccess
TEST_F(OptimizerFactoryTest, ParamGroups_BiasAccess) {
    OptimizerFactory::ParameterGroups groups;
    groups.biasParams.push_back(torch::randn({10}));
    EXPECT_EQ(1, groups.biasParams.size());
}

// OF-154: ParamGroups_BNAccess
TEST_F(OptimizerFactoryTest, ParamGroups_BNAccess) {
    OptimizerFactory::ParameterGroups groups;
    groups.bnParams.push_back(torch::ones({10}));
    EXPECT_EQ(1, groups.bnParams.size());
}

// OF-155: ParamGroups_WeightAccess
TEST_F(OptimizerFactoryTest, ParamGroups_WeightAccess) {
    OptimizerFactory::ParameterGroups groups;
    groups.weightParams.push_back(torch::randn({10, 5}));
    EXPECT_EQ(1, groups.weightParams.size());
}

// OF-156: ParamGroups_DefaultEmpty
TEST_F(OptimizerFactoryTest, ParamGroups_DefaultEmpty) {
    OptimizerFactory::ParameterGroups groups;
    EXPECT_TRUE(groups.biasParams.empty());
    EXPECT_TRUE(groups.bnParams.empty());
    EXPECT_TRUE(groups.weightParams.empty());
}

// =============================================================================
// 2.2.12 Exception Messages (OF-157 ~ OF-160)
// =============================================================================

// OF-157: ExMsg_EmptyParams
TEST_F(OptimizerFactoryTest, ExMsg_EmptyParams) {
    std::vector<torch::Tensor> emptyParams;
    try {
        OptimizerFactory::createSGD(emptyParams, 0.01f);
        FAIL() << "Expected ConfigurationException";
    }
    catch (const ConfigurationException& e) {
        std::string msg = e.what();
        EXPECT_NE(std::string::npos, msg.find("parameter list is empty"));
    }
}

// OF-158: ExMsg_UnsupportedType
TEST_F(OptimizerFactoryTest, ExMsg_UnsupportedType) {
    const auto config = createTestConfig("rmsprop");  // const to select non-template overload
    try {
        OptimizerFactory::createFromConfig(validParams, config);
        FAIL() << "Expected ConfigurationException";
    }
    catch (const ConfigurationException& e) {
        std::string msg = e.what();
        EXPECT_NE(std::string::npos, msg.find("rmsprop"));
        EXPECT_NE(std::string::npos, msg.find("Supported types"));
    }
}

// OF-159: ExMsg_NoTrainable
TEST_F(OptimizerFactoryTest, ExMsg_NoTrainable) {
    FrozenMockModel model;
    auto config = createTestConfig("sgd");
    try {
        OptimizerFactory::createFromConfig(model, config);
        FAIL() << "Expected std::invalid_argument";
    }
    catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(std::string::npos, msg.find("No trainable parameters"));
    }
}

// OF-160: ExMsg_NoParamsGroups
TEST_F(OptimizerFactoryTest, ExMsg_NoParamsGroups) {
    EmptyMockModel model;
    auto config = createTestConfig("sgd");
    try {
        OptimizerFactory::createFromConfig(model, config);
        FAIL() << "Expected std::invalid_argument";
    }
    catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(std::string::npos, msg.find("No trainable parameters"));
    }
}
