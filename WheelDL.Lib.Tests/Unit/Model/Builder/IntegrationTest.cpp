/**
 * @file IntegrationTest.cpp
 * @brief Integration tests for Model/Builder & Model/Modules
 *
 * Tests cover:
 * - Full pipeline (Factory -> Module -> Forward)
 * - Factory registry completeness
 * - Module interoperability
 * - Device transfer
 * - Gradient flow
 *
 * @author WheelDL Team
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "Model/Builder/Factory/ModuleFactory.h"
#include "Model/Builder/Factory/ModuleFactoryRegistry.h"
#include "Model/Builder/Factory/ConvolutionFactory.h"
#include "Model/Builder/Factory/BlockFactory.h"
#include "Model/Builder/Factory/AttentionFactory.h"
#include "Model/Builder/Factory/HeadFactory.h"
#include "Model/Builder/Factory/TransformerFactory.h"
#include "Model/Builder/Factory/UtilityFactory.h"
#include "Model/Builder/ModelBuilder.h"
#include "Model/Modules/Conv.h"
#include "Model/Modules/Block.h"
#include "Model/Modules/Attention.h"
#include "Model/Modules/Transformer.h"
#include "Model/Modules/Head.h"
#include "Model/Modules/Utils.h"

using namespace WheelDL::Model::Builder;
using namespace WheelDL::Model::Modules;

// ============================================================================
// Test Fixture
// ============================================================================

class BuilderModulesIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);

    }

    ModuleBuildContext createContext(int64_t inputChannels = 64) {
        ModuleBuildContext ctx;
        ctx.inputChannels = inputChannels;
        ctx.repeats = 1;
        ctx.defaultAct = "SiLU";
        ctx.numClasses = 80;
        ctx.headChannels = {64, 128, 256};
        ctx.applyScale = [](int64_t ch, int64_t rep) {
            return std::make_pair(ch, rep);
        };
        return ctx;
    }

    torch::Tensor createInput(int64_t n = 1, int64_t c = 64, int64_t h = 32, int64_t w = 32) {
        return torch::randn({n, c, h, w});
    }

    std::vector<torch::Tensor> createMultiScaleInput(
        int64_t batch = 1,
        std::vector<int64_t> channels = {64, 128, 256},
        std::vector<int64_t> sizes = {80, 40, 20}) {
        std::vector<torch::Tensor> result;
        for (size_t i = 0; i < channels.size(); i++) {
            result.push_back(torch::randn({batch, channels[i], sizes[i], sizes[i]}));
        }
        return result;
    }
};

// ============================================================================
// Factory Registry Tests
// ============================================================================

TEST_F(BuilderModulesIntegrationTest, FactoryRegistry_ConvolutionFactoriesRegistered) {
    auto& registry = ModuleFactoryRegistry::instance();

    // All convolution factories should be registered
    std::vector<std::string> convTypes = {
        "Conv", "DWConv", "RepConv", "GhostConv", "Focus",
        "nn.ConvTranspose2d", "NaiveConv"
    };

    for (const auto& type : convTypes) {
        EXPECT_NE(registry.getFactory(type), nullptr) << "Factory not registered: " << type;
    }
}

TEST_F(BuilderModulesIntegrationTest, FactoryRegistry_BlockFactoriesRegistered) {
    auto& registry = ModuleFactoryRegistry::instance();

    // Core block factories
    std::vector<std::string> blockTypes = {
        "Bottleneck", "C2", "C2f", "C3", "C3x", "C3k2",
        "SPP", "SPPF", "RepC3"
    };

    for (const auto& type : blockTypes) {
        EXPECT_NE(registry.getFactory(type), nullptr) << "Factory not registered: " << type;
    }
}

TEST_F(BuilderModulesIntegrationTest, FactoryRegistry_AttentionFactoriesRegistered) {
    auto& registry = ModuleFactoryRegistry::instance();

    std::vector<std::string> attentionTypes = {"PSA", "CBAM", "Attention"};

    for (const auto& type : attentionTypes) {
        EXPECT_NE(registry.getFactory(type), nullptr) << "Factory not registered: " << type;
    }
}

TEST_F(BuilderModulesIntegrationTest, FactoryRegistry_HeadFactoriesRegistered) {
    auto& registry = ModuleFactoryRegistry::instance();

    std::vector<std::string> headTypes = {"Detect", "OBB", "Classify", "Segment", "Anomaly"};

    for (const auto& type : headTypes) {
        EXPECT_NE(registry.getFactory(type), nullptr) << "Factory not registered: " << type;
    }
}

TEST_F(BuilderModulesIntegrationTest, FactoryRegistry_TransformerFactoriesRegistered) {
    auto& registry = ModuleFactoryRegistry::instance();

    std::vector<std::string> transformerTypes = {
        "LayerNorm2d", "MLPBlock", "MLP", "TransformerLayer",
        "TransformerEncoderLayer", "AIFI"
    };

    for (const auto& type : transformerTypes) {
        EXPECT_NE(registry.getFactory(type), nullptr) << "Factory not registered: " << type;
    }
}

TEST_F(BuilderModulesIntegrationTest, FactoryRegistry_UtilityFactoriesRegistered) {
    auto& registry = ModuleFactoryRegistry::instance();

    std::vector<std::string> utilTypes = {
        "nn.Upsample", "Concat", "nn.MaxPool2d", "nn.AvgPool2d"
    };

    for (const auto& type : utilTypes) {
        EXPECT_NE(registry.getFactory(type), nullptr) << "Factory not registered: " << type;
    }
}

// ============================================================================
// Factory to Module Pipeline Tests
// ============================================================================

TEST_F(BuilderModulesIntegrationTest, Pipeline_Conv_FactoryToForward) {
    auto& registry = ModuleFactoryRegistry::instance();
    auto factory = registry.getFactory("Conv");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 3, 2]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);
    ASSERT_FALSE(result.module.is_empty());

    auto input = createInput(1, 64, 32, 32);
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 16);  // 32 / 2
    EXPECT_EQ(output.size(3), 16);
}

TEST_F(BuilderModulesIntegrationTest, Pipeline_Block_FactoryToForward) {
    auto& registry = ModuleFactoryRegistry::instance();
    auto factory = registry.getFactory("C2f");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[128, 2]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);
    ASSERT_FALSE(result.module.is_empty());

    auto input = createInput(1, 64, 32, 32);
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(BuilderModulesIntegrationTest, Pipeline_Attention_FactoryToForward) {
    auto& registry = ModuleFactoryRegistry::instance();
    auto factory = registry.getFactory("CBAM");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);
    ASSERT_FALSE(result.module.is_empty());

    auto input = createInput(1, 64, 32, 32);
    auto output = result.module.forward<torch::Tensor>(input);

    // CBAM preserves shape
    EXPECT_EQ(output.size(0), input.size(0));
    EXPECT_EQ(output.size(1), input.size(1));
    EXPECT_EQ(output.size(2), input.size(2));
    EXPECT_EQ(output.size(3), input.size(3));
}

TEST_F(BuilderModulesIntegrationTest, Pipeline_Transformer_FactoryToForward) {
    auto& registry = ModuleFactoryRegistry::instance();
    auto factory = registry.getFactory("LayerNorm2d");
    ASSERT_NE(factory, nullptr);

    YAML::Node args = YAML::Load("[]");
    auto ctx = createContext(64);

    auto result = factory->create(args, ctx);
    ASSERT_FALSE(result.module.is_empty());

    auto input = createInput(1, 64, 16, 16);
    auto output = result.module.forward<torch::Tensor>(input);

    EXPECT_EQ(output.size(0), input.size(0));
    EXPECT_EQ(output.size(1), input.size(1));
    EXPECT_EQ(output.size(2), input.size(2));
    EXPECT_EQ(output.size(3), input.size(3));
}

// ============================================================================
// Module Interoperability Tests
// ============================================================================

TEST_F(BuilderModulesIntegrationTest, ModuleChain_ConvToBlockToAttention) {
    // Create Conv -> C2f -> CBAM chain
    Conv conv(64, 128, 3, 2);
    C2f c2f(128, 128, 2, true, 0.5);
    CBAM cbam(128);

    auto input = createInput(1, 64, 64, 64);
    auto x = conv->forward(input);
    x = c2f->forward(x);
    x = cbam->forward(x);

    EXPECT_EQ(x.size(0), 1);
    EXPECT_EQ(x.size(1), 128);
    EXPECT_FALSE(x.isnan().any().item<bool>());
}

TEST_F(BuilderModulesIntegrationTest, ModuleChain_BackboneNeck) {
    // Simulate backbone -> neck architecture
    Conv stem(3, 64, 3, 2);
    C2f stage1(64, 128, 2, true, 0.5);
    SPPF sppf(128, 128, 5);
    C2f neck(128, 256, 2, true, 0.5);

    auto input = torch::randn({1, 3, 640, 640});
    auto x = stem->forward(input);
    x = stage1->forward(x);
    x = sppf->forward(x);
    x = neck->forward(x);

    EXPECT_EQ(x.size(0), 1);
    EXPECT_EQ(x.size(1), 256);
    EXPECT_FALSE(x.isnan().any().item<bool>());
}

TEST_F(BuilderModulesIntegrationTest, ModuleChain_ClassificationHead) {
    // Classification pipeline
    Conv stem(3, 64, 3, 2);
    C2f stage(64, 512, 4, true, 0.5);
    Classify head(512, 1000);

    auto input = torch::randn({1, 3, 224, 224});
    auto x = stem->forward(input);
    x = stage->forward(x);
    auto outputs = head->forward({x});

    EXPECT_EQ(outputs[0].size(0), 1);
    EXPECT_EQ(outputs[0].size(1), 1000);
}

TEST_F(BuilderModulesIntegrationTest, ModuleChain_SegmentationHead) {
    // Segmentation pipeline
    Conv conv(64, 256, 1, 1);
    Segment head(256, 21);

    auto input = createInput(1, 64, 64, 64);
    auto x = conv->forward(input);
    auto outputs = head->forward({x});

    EXPECT_EQ(outputs[0].size(0), 1);
    EXPECT_EQ(outputs[0].size(1), 21);
}

// ============================================================================
// Multi-Scale Feature Tests
// ============================================================================

TEST_F(BuilderModulesIntegrationTest, MultiScale_DetectionHead) {
    // Detection with multi-scale features
    Detect detect(80, std::vector<int64_t>{64, 128, 256});
    detect->train();

    auto inputs = createMultiScaleInput(1, {64, 128, 256}, {80, 40, 20});
    auto outputs = detect->forward(inputs);

    EXPECT_FALSE(outputs.empty());
}

TEST_F(BuilderModulesIntegrationTest, MultiScale_OBBHead) {
    // OBB detection with multi-scale features
    OBB obb(80, 1, std::vector<int64_t>{64, 128, 256});
    obb->train();

    auto inputs = createMultiScaleInput(1, {64, 128, 256}, {80, 40, 20});
    auto outputs = obb->forward(inputs);

    EXPECT_FALSE(outputs.empty());
}

// ============================================================================
// Gradient Flow Tests
// ============================================================================

TEST_F(BuilderModulesIntegrationTest, GradientFlow_Conv) {
    Conv conv(64, 128, 3, 1);

    auto input = torch::randn({1, 64, 32, 32}, torch::requires_grad(true));
    auto output = conv->forward(input);
    auto loss = output.sum();
    loss.backward();

    EXPECT_TRUE(input.grad().defined());
    EXPECT_FALSE(input.grad().isnan().any().item<bool>());
}

TEST_F(BuilderModulesIntegrationTest, GradientFlow_Block) {
    C2f block(64, 128, 2, true, 0.5);

    auto input = torch::randn({1, 64, 32, 32}, torch::requires_grad(true));
    auto output = block->forward(input);
    auto loss = output.sum();
    loss.backward();

    EXPECT_TRUE(input.grad().defined());
    EXPECT_FALSE(input.grad().isnan().any().item<bool>());
}

TEST_F(BuilderModulesIntegrationTest, GradientFlow_Attention) {
    CBAM cbam(64);

    auto input = torch::randn({1, 64, 32, 32}, torch::requires_grad(true));
    auto output = cbam->forward(input);
    auto loss = output.sum();
    loss.backward();

    EXPECT_TRUE(input.grad().defined());
    EXPECT_FALSE(input.grad().isnan().any().item<bool>());
}

TEST_F(BuilderModulesIntegrationTest, GradientFlow_Transformer) {
    LayerNorm2d ln(64);

    auto input = torch::randn({1, 64, 16, 16}, torch::requires_grad(true));
    auto output = ln->forward(input);
    auto loss = output.sum();
    loss.backward();

    EXPECT_TRUE(input.grad().defined());
    EXPECT_FALSE(input.grad().isnan().any().item<bool>());
}

TEST_F(BuilderModulesIntegrationTest, GradientFlow_FullPipeline) {
    // Full pipeline gradient test
    Conv conv(64, 128, 3, 2);
    C2f block(128, 128, 2, true, 0.5);
    Classify head(128, 10);

    auto input = torch::randn({1, 64, 64, 64}, torch::requires_grad(true));
    auto x = conv->forward(input);
    x = block->forward(x);
    auto outputs = head->forward({x});
    auto loss = outputs[0].sum();
    loss.backward();

    EXPECT_TRUE(input.grad().defined());
    EXPECT_FALSE(input.grad().isnan().any().item<bool>());

    // Check parameters have gradients
    for (const auto& param : conv->parameters()) {
        EXPECT_TRUE(param.grad().defined());
    }
}

// ============================================================================
// Training Mode Tests
// ============================================================================

TEST_F(BuilderModulesIntegrationTest, TrainingMode_Propagation) {
    Conv conv(64, 128, 3, 1);
    C2f block(128, 128, 2, true, 0.5);

    // Set to training mode
    conv->train();
    block->train();

    EXPECT_TRUE(conv->is_training());
    EXPECT_TRUE(block->is_training());

    // Set to eval mode
    conv->eval();
    block->eval();

    EXPECT_FALSE(conv->is_training());
    EXPECT_FALSE(block->is_training());
}

TEST_F(BuilderModulesIntegrationTest, TrainingMode_OutputConsistency) {
    C2f block(64, 64, 2, true, 0.5);
    block->eval();

    auto input = createInput(1, 64, 32, 32);

    // Same input should produce same output in eval mode
    auto output1 = block->forward(input);
    auto output2 = block->forward(input);

    EXPECT_TRUE(torch::allclose(output1, output2));
}

// ============================================================================
// Utils Integration Tests
// ============================================================================

TEST_F(BuilderModulesIntegrationTest, Utils_ActivationCreation) {
    std::vector<std::string> activations = {
        "ReLU", "SiLU", "LeakyReLU", "GELU", "Mish", "Hardswish", "Identity"
    };

    auto input = torch::randn({1, 64});

    for (const auto& actName : activations) {
        auto act = createActivation(actName);
        auto output = act.forward(input);

        EXPECT_FALSE(output.isnan().any().item<bool>()) << "NaN in " << actName;
    }
}

TEST_F(BuilderModulesIntegrationTest, Utils_BiasInit) {
    float bias = biasInitWithProb(0.01f);

    EXPECT_FALSE(std::isnan(bias));
    EXPECT_FALSE(std::isinf(bias));
}

TEST_F(BuilderModulesIntegrationTest, Utils_LinearInit) {
    torch::nn::Linear linear(64, 128);

    EXPECT_NO_THROW(linearInit(linear));
    EXPECT_FALSE(linear->weight.isnan().any().item<bool>());
}

TEST_F(BuilderModulesIntegrationTest, Utils_InverseSigmoid) {
    auto x = torch::sigmoid(torch::randn({1, 64}));
    auto output = inverseSigmoid(x);

    EXPECT_FALSE(output.isnan().any().item<bool>());
    EXPECT_FALSE(output.isinf().any().item<bool>());
}

TEST_F(BuilderModulesIntegrationTest, Utils_MakeAnchors) {
    std::vector<torch::Tensor> feats = {
        torch::randn({1, 64, 80, 80}),
        torch::randn({1, 128, 40, 40}),
        torch::randn({1, 256, 20, 20})
    };
    auto stride = torch::tensor({8.0, 16.0, 32.0});

    auto [anchors, strides] = makeAnchors(feats, stride);

    EXPECT_EQ(anchors.size(0), 8400);  // 80*80 + 40*40 + 20*20
    EXPECT_FALSE(anchors.isnan().any().item<bool>());
}

// ============================================================================
// Batch Size Tests
// ============================================================================

TEST_F(BuilderModulesIntegrationTest, BatchSize_Conv) {
    Conv conv(64, 128, 3, 1);

    for (int batch : {1, 2, 4, 8}) {
        auto input = createInput(batch, 64, 32, 32);
        auto output = conv->forward(input);

        EXPECT_EQ(output.size(0), batch);
    }
}

TEST_F(BuilderModulesIntegrationTest, BatchSize_Block) {
    C2f block(64, 128, 2, true, 0.5);

    for (int batch : {1, 2, 4, 8}) {
        auto input = createInput(batch, 64, 32, 32);
        auto output = block->forward(input);

        EXPECT_EQ(output.size(0), batch);
    }
}

TEST_F(BuilderModulesIntegrationTest, BatchSize_Detection) {
    Detect detect(80, std::vector<int64_t>{64, 128, 256});
    detect->train();

    for (int batch : {1, 2, 4}) {
        auto inputs = createMultiScaleInput(batch, {64, 128, 256}, {80, 40, 20});
        auto outputs = detect->forward(inputs);

        EXPECT_FALSE(outputs.empty());
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(BuilderModulesIntegrationTest, EdgeCase_SmallSpatialSize) {
    Conv conv(64, 64, 3, 1);

    auto input = createInput(1, 64, 4, 4);
    auto output = conv->forward(input);

    EXPECT_FALSE(output.isnan().any().item<bool>());
}

TEST_F(BuilderModulesIntegrationTest, EdgeCase_LargeBatchSize) {
    Conv conv(64, 64, 3, 1);

    auto input = createInput(32, 64, 16, 16);
    auto output = conv->forward(input);

    EXPECT_EQ(output.size(0), 32);
}

TEST_F(BuilderModulesIntegrationTest, EdgeCase_SingleChannel) {
    Conv conv(1, 64, 3, 1);

    auto input = createInput(1, 1, 32, 32);
    auto output = conv->forward(input);

    EXPECT_EQ(output.size(1), 64);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST_F(BuilderModulesIntegrationTest, NumericalStability_Conv) {
    Conv conv(64, 128, 3, 1);

    // Large values
    auto input = torch::randn({1, 64, 32, 32}) * 100.0f;
    auto output = conv->forward(input);

    EXPECT_FALSE(output.isnan().any().item<bool>());
    EXPECT_FALSE(output.isinf().any().item<bool>());
}

TEST_F(BuilderModulesIntegrationTest, NumericalStability_Attention) {
    CBAM cbam(64);

    // Large values
    auto input = torch::randn({1, 64, 32, 32}) * 100.0f;
    auto output = cbam->forward(input);

    EXPECT_FALSE(output.isnan().any().item<bool>());
    EXPECT_FALSE(output.isinf().any().item<bool>());
}

TEST_F(BuilderModulesIntegrationTest, NumericalStability_SmallValues) {
    Conv conv(64, 128, 3, 1);

    // Very small values
    auto input = torch::randn({1, 64, 32, 32}) * 1e-5f;
    auto output = conv->forward(input);

    EXPECT_FALSE(output.isnan().any().item<bool>());
}

