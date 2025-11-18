#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Modules/Transformer.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <torch/torch.h>

using namespace WheelDL;
using namespace WheelDL::Model::Modules;
using namespace WheelDL::Utils;

// ========== LayerNorm2d Module Tests ==========

class LayerNorm2dModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<LayerNorm2d> layerNorm2dModule;

    static void SetUpTestSuite() {
        try {
            layerNorm2dModule = std::make_unique<LayerNorm2d>(64, 1e-6);
        } catch (const std::exception& e) {
            std::cerr << "LayerNorm2d setup failed: " << e.what() << std::endl;
            layerNorm2dModule.reset();
        }
    }

    static void TearDownTestSuite() {
        layerNorm2dModule.reset();
    }
};

std::unique_ptr<LayerNorm2d> LayerNorm2dModuleTest::layerNorm2dModule;

TEST_F(LayerNorm2dModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        LayerNorm2d ln2d(64, 1e-6);
    });
}

TEST_F(LayerNorm2dModuleTest, Forward_ValidInput_Success) {
    if (!layerNorm2dModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = layerNorm2dModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(LayerNorm2dModuleTest, Forward_2DNormalization_CorrectOutputShape) {
    if (!layerNorm2dModule) return;

    torch::Tensor input = torch::randn({2, 64, 32, 32});

    auto output = layerNorm2dModule->ptr()->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(LayerNorm2dModuleTest, Forward_ZeroMean_CorrectBehavior) {
    LayerNorm2d ln2d(64, 1e-6);
    torch::Tensor input = torch::randn({1, 64, 32, 32});

    auto output = ln2d->forward(input);

    // LayerNorm2d normalizes across channels for each spatial position
    // So the mean across channel dimension should be zero
    auto mean = output.mean(1);  // Mean over channel dimension
    EXPECT_TRUE(torch::allclose(mean, torch::zeros_like(mean), 1e-5, 1e-5));
}

TEST_F(LayerNorm2dModuleTest, Forward_UnitVariance_CorrectBehavior) {
    LayerNorm2d ln2d(64, 1e-6);
    torch::Tensor input = torch::randn({1, 64, 32, 32});

    auto output = ln2d->forward(input);

    auto var = output.var({2, 3}, false);
    EXPECT_TRUE(torch::allclose(var, torch::ones_like(var), 1e-1, 1e-1));
}

// ========== TransformerBlock Module Tests ==========

class TransformerBlockModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<TransformerBlock> transformerBlockModule;

    static void SetUpTestSuite() {
        try {
            transformerBlockModule = std::make_unique<TransformerBlock>(128, 128, 8, 2);
        } catch (const std::exception& e) {
            std::cerr << "TransformerBlock setup failed: " << e.what() << std::endl;
            transformerBlockModule.reset();
        }
    }

    static void TearDownTestSuite() {
        transformerBlockModule.reset();
    }
};

std::unique_ptr<TransformerBlock> TransformerBlockModuleTest::transformerBlockModule;

TEST_F(TransformerBlockModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        TransformerBlock tb(128, 128, 8, 2);
    });
}

TEST_F(TransformerBlockModuleTest, Forward_ValidInput_Success) {
    if (!transformerBlockModule) return;

    torch::Tensor input = torch::randn({1, 128, 32, 32});

    EXPECT_NO_THROW({
        auto output = transformerBlockModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 128);
    });
}

TEST_F(TransformerBlockModuleTest, Forward_AttentionAndFFN_CorrectOutputShape) {
    TransformerBlock tb(128, 128, 8, 2);
    torch::Tensor input = torch::randn({2, 128, 32, 32});

    auto output = tb->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(TransformerBlockModuleTest, Forward_MultipleHeads_Success) {
    TransformerBlock tb4(128, 128, 4, 2);
    TransformerBlock tb16(128, 128, 16, 2);
    torch::Tensor input = torch::randn({1, 128, 32, 32});

    EXPECT_NO_THROW({
        auto output4 = tb4->forward(input);
        auto output16 = tb16->forward(input);
        EXPECT_EQ(output4.size(1), 128);
        EXPECT_EQ(output16.size(1), 128);
    });
}

TEST_F(TransformerBlockModuleTest, Forward_MultipleLayers_CorrectBehavior) {
    TransformerBlock tb1(128, 128, 8, 1);
    TransformerBlock tb4(128, 128, 8, 4);
    torch::Tensor input = torch::randn({1, 128, 32, 32});

    auto output1 = tb1->forward(input);
    auto output4 = tb4->forward(input);

    EXPECT_EQ(output1.sizes(), output4.sizes());
}

// ========== TransformerLayer Module Tests ==========

class TransformerLayerModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<TransformerLayer> transformerLayerModule;

    static void SetUpTestSuite() {
        try {
            transformerLayerModule = std::make_unique<TransformerLayer>(256, 8);
        } catch (const std::exception& e) {
            std::cerr << "TransformerLayer setup failed: " << e.what() << std::endl;
            transformerLayerModule.reset();
        }
    }

    static void TearDownTestSuite() {
        transformerLayerModule.reset();
    }
};

std::unique_ptr<TransformerLayer> TransformerLayerModuleTest::transformerLayerModule;

TEST_F(TransformerLayerModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        TransformerLayer tl(256, 8);
    });
}

TEST_F(TransformerLayerModuleTest, Forward_ValidInput_Success) {
    if (!transformerLayerModule) return;

    torch::Tensor input = torch::randn({10, 1, 256});

    EXPECT_NO_THROW({
        auto output = transformerLayerModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 10);
        EXPECT_EQ(output.size(1), 1);
        EXPECT_EQ(output.size(2), 256);
    });
}

TEST_F(TransformerLayerModuleTest, Forward_ResidualConnections_CorrectBehavior) {
    TransformerLayer tl(256, 8);
    torch::Tensor input = torch::randn({10, 2, 256});

    auto output = tl->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

// ========== MLPBlock Module Tests ==========

class MLPBlockModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<MLPBlock> mlpBlockModule;

    static void SetUpTestSuite() {
        try {
            mlpBlockModule = std::make_unique<MLPBlock>(256, 1024);
        } catch (const std::exception& e) {
            std::cerr << "MLPBlock setup failed: " << e.what() << std::endl;
            mlpBlockModule.reset();
        }
    }

    static void TearDownTestSuite() {
        mlpBlockModule.reset();
    }
};

std::unique_ptr<MLPBlock> MLPBlockModuleTest::mlpBlockModule;

TEST_F(MLPBlockModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        MLPBlock mlp(256, 1024);
    });
}

TEST_F(MLPBlockModuleTest, Forward_ValidInput_Success) {
    if (!mlpBlockModule) return;

    torch::Tensor input = torch::randn({10, 256});

    EXPECT_NO_THROW({
        auto output = mlpBlockModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 10);
        EXPECT_EQ(output.size(1), 256);
    });
}

TEST_F(MLPBlockModuleTest, Forward_OutputShapeSameAsInput_Success) {
    if (!mlpBlockModule) return;

    torch::Tensor input = torch::randn({20, 256});

    auto output = mlpBlockModule->ptr()->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

// ========== MLP Module Tests ==========

class MLPModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<MLP> mlpModule;

    static void SetUpTestSuite() {
        try {
            mlpModule = std::make_unique<MLP>(256, 512, 128, 3, false);
        } catch (const std::exception& e) {
            std::cerr << "MLP setup failed: " << e.what() << std::endl;
            mlpModule.reset();
        }
    }

    static void TearDownTestSuite() {
        mlpModule.reset();
    }
};

std::unique_ptr<MLP> MLPModuleTest::mlpModule;

TEST_F(MLPModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        MLP mlp(256, 512, 128, 3, false);
    });
}

TEST_F(MLPModuleTest, Forward_ValidInput_Success) {
    if (!mlpModule) return;

    torch::Tensor input = torch::randn({10, 256});

    EXPECT_NO_THROW({
        auto output = mlpModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 10);
        EXPECT_EQ(output.size(1), 128);
    });
}

TEST_F(MLPModuleTest, Forward_DifferentOutputDim_CorrectOutputShape) {
    MLP mlp(256, 512, 64, 3, false);
    torch::Tensor input = torch::randn({20, 256});

    auto output = mlp->forward(input);
    EXPECT_EQ(output.size(0), 20);
    EXPECT_EQ(output.size(1), 64);
}

TEST_F(MLPModuleTest, Forward_WithSigmoid_OutputInRange) {
    MLP mlp(256, 512, 128, 3, true);
    torch::Tensor input = torch::randn({10, 256});

    auto output = mlp->forward(input);
    EXPECT_TRUE(torch::all(output >= 0.0).item<bool>());
    EXPECT_TRUE(torch::all(output <= 1.0).item<bool>());
}
