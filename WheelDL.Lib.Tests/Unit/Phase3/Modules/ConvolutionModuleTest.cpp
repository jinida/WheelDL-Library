#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Modules/Conv.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <torch/torch.h>

using namespace WheelDL;
using namespace WheelDL::Model::Modules;
using namespace WheelDL::Utils;

// ========== Conv Module Tests ==========

class ConvModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<Conv> convModule;

    static void SetUpTestSuite() {
        try {
            convModule = std::make_unique<Conv>(3, 64, 3, 1);
        } catch (const std::exception& e) {
            std::cerr << "Conv setup failed: " << e.what() << std::endl;
            convModule.reset();
        }
    }

    static void TearDownTestSuite() {
        convModule.reset();
    }
};

std::unique_ptr<Conv> ConvModuleTest::convModule;

TEST_F(ConvModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        Conv conv(3, 64, 3, 1);
    });
}

TEST_F(ConvModuleTest, Forward_ValidInput_Success) {
    if (!convModule) return;

    torch::Tensor input = torch::randn({1, 3, 64, 64});

    EXPECT_NO_THROW({
        auto output = convModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(ConvModuleTest, Forward_DifferentKernels_CorrectOutputShape) {
    torch::Tensor input = torch::randn({2, 3, 32, 32});

    // Kernel=1
    Conv conv1(3, 16, 1, 1);
    auto out1 = conv1->forward(input);
    EXPECT_EQ(out1.size(0), 2);
    EXPECT_EQ(out1.size(1), 16);
    EXPECT_EQ(out1.size(2), 32);
    EXPECT_EQ(out1.size(3), 32);

    // Kernel=3
    Conv conv3(3, 16, 3, 1);
    auto out3 = conv3->forward(input);
    EXPECT_EQ(out3.size(0), 2);
    EXPECT_EQ(out3.size(1), 16);

    // Kernel=5
    Conv conv5(3, 16, 5, 1);
    auto out5 = conv5->forward(input);
    EXPECT_EQ(out5.size(0), 2);
    EXPECT_EQ(out5.size(1), 16);
}

TEST_F(ConvModuleTest, Forward_DifferentStrides_CorrectOutputShape) {
    torch::Tensor input = torch::randn({1, 3, 64, 64});

    // Stride=1
    Conv conv1(3, 32, 3, 1);
    auto out1 = conv1->forward(input);
    EXPECT_EQ(out1.size(2), 64);
    EXPECT_EQ(out1.size(3), 64);

    // Stride=2
    Conv conv2(3, 32, 3, 2);
    auto out2 = conv2->forward(input);
    EXPECT_EQ(out2.size(2), 32);
    EXPECT_EQ(out2.size(3), 32);
}

TEST_F(ConvModuleTest, Forward_WithSiLUActivation_Success) {
    Conv conv(3, 64, 3, 1, std::nullopt, 1, 1, "SiLU");
    torch::Tensor input = torch::randn({1, 3, 32, 32});

    EXPECT_NO_THROW({
        auto output = conv->forward(input);
        EXPECT_FALSE(output.sizes().empty());
    });
}

TEST_F(ConvModuleTest, Forward_WithReLUActivation_Success) {
    Conv conv(3, 64, 3, 1, std::nullopt, 1, 1, "ReLU");
    torch::Tensor input = torch::randn({1, 3, 32, 32});

    EXPECT_NO_THROW({
        auto output = conv->forward(input);
        EXPECT_FALSE(output.sizes().empty());
    });
}

TEST_F(ConvModuleTest, Forward_NoActivation_Success) {
    Conv conv(3, 64, 3, 1, std::nullopt, 1, 1, "");
    torch::Tensor input = torch::randn({1, 3, 32, 32});

    EXPECT_NO_THROW({
        auto output = conv->forward(input);
        EXPECT_FALSE(output.sizes().empty());
    });
}

// ========== DWConv Module Tests ==========

class DWConvModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<DWConv> dwconvModule;

    static void SetUpTestSuite() {
        try {
            dwconvModule = std::make_unique<DWConv>(32, 64, 3, 1);
        } catch (const std::exception& e) {
            std::cerr << "DWConv setup failed: " << e.what() << std::endl;
            dwconvModule.reset();
        }
    }

    static void TearDownTestSuite() {
        dwconvModule.reset();
    }
};

std::unique_ptr<DWConv> DWConvModuleTest::dwconvModule;

TEST_F(DWConvModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        DWConv dwconv(32, 64, 3, 1);
    });
}

TEST_F(DWConvModuleTest, Forward_ValidInput_Success) {
    if (!dwconvModule) return;

    torch::Tensor input = torch::randn({1, 32, 64, 64});

    EXPECT_NO_THROW({
        auto output = dwconvModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(DWConvModuleTest, Forward_DepthwiseGrouping_CorrectBehavior) {
    DWConv dwconv(16, 32, 3, 1);
    torch::Tensor input = torch::randn({2, 16, 32, 32});

    auto output = dwconv->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 32);
}

// ========== Conv2 Module Tests ==========

class Conv2ModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<Conv2> conv2Module;

    static void SetUpTestSuite() {
        try {
            conv2Module = std::make_unique<Conv2>(3, 64, 3, 1);
        } catch (const std::exception& e) {
            std::cerr << "Conv2 setup failed: " << e.what() << std::endl;
            conv2Module.reset();
        }
    }

    static void TearDownTestSuite() {
        conv2Module.reset();
    }
};

std::unique_ptr<Conv2> Conv2ModuleTest::conv2Module;

TEST_F(Conv2ModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        Conv2 conv2(3, 64, 3, 1);
    });
}

TEST_F(Conv2ModuleTest, Forward_ValidInput_Success) {
    if (!conv2Module) return;

    torch::Tensor input = torch::randn({1, 3, 64, 64});

    EXPECT_NO_THROW({
        auto output = conv2Module->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(Conv2ModuleTest, Forward_DualPath_CorrectOutputShape) {
    Conv2 conv2(3, 32, 3, 1);
    torch::Tensor input = torch::randn({2, 3, 32, 32});

    auto output = conv2->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 32);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

// ========== RepConv Module Tests ==========

class RepConvModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<RepConv> repconvModule;

    static void SetUpTestSuite() {
        try {
            repconvModule = std::make_unique<RepConv>(32, 64, 3, 1, 1);
        } catch (const std::exception& e) {
            std::cerr << "RepConv setup failed: " << e.what() << std::endl;
            repconvModule.reset();
        }
    }

    static void TearDownTestSuite() {
        repconvModule.reset();
    }
};

std::unique_ptr<RepConv> RepConvModuleTest::repconvModule;

TEST_F(RepConvModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        RepConv repconv(32, 64, 3, 1, 1);
    });
}

TEST_F(RepConvModuleTest, Forward_TrainingMode_Success) {
    if (!repconvModule) return;

    repconvModule->ptr()->train();
    torch::Tensor input = torch::randn({1, 32, 64, 64});

    EXPECT_NO_THROW({
        auto output = repconvModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(RepConvModuleTest, Forward_EvalMode_Success) {
    if (!repconvModule) return;

    repconvModule->ptr()->eval();
    torch::Tensor input = torch::randn({1, 32, 64, 64});

    EXPECT_NO_THROW({
        auto output = repconvModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(RepConvModuleTest, Forward_DifferentModes_SameOutputShape) {
    RepConv repconv(16, 32, 3, 1, 1);
    torch::Tensor input = torch::randn({2, 16, 32, 32});

    repconv->train();
    auto trainOutput = repconv->forward(input);

    repconv->eval();
    auto evalOutput = repconv->forward(input);

    EXPECT_EQ(trainOutput.sizes(), evalOutput.sizes());
}

// ========== GhostConv Module Tests ==========

class GhostConvModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<GhostConv> ghostconvModule;

    static void SetUpTestSuite() {
        try {
            ghostconvModule = std::make_unique<GhostConv>(32, 64, 3, 1);
        } catch (const std::exception& e) {
            std::cerr << "GhostConv setup failed: " << e.what() << std::endl;
            ghostconvModule.reset();
        }
    }

    static void TearDownTestSuite() {
        ghostconvModule.reset();
    }
};

std::unique_ptr<GhostConv> GhostConvModuleTest::ghostconvModule;

TEST_F(GhostConvModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        GhostConv ghostconv(32, 64, 3, 1);
    });
}

TEST_F(GhostConvModuleTest, Forward_ValidInput_Success) {
    if (!ghostconvModule) return;

    torch::Tensor input = torch::randn({1, 32, 64, 64});

    EXPECT_NO_THROW({
        auto output = ghostconvModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(GhostConvModuleTest, Forward_ChannelSplitting_CorrectOutputShape) {
    GhostConv ghostconv(16, 48, 3, 1);
    torch::Tensor input = torch::randn({2, 16, 32, 32});

    auto output = ghostconv->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 48);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

// ========== Focus Module Tests ==========

class FocusModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<Focus> focusModule;

    static void SetUpTestSuite() {
        try {
            focusModule = std::make_unique<Focus>(3, 64, 3, 1);
        } catch (const std::exception& e) {
            std::cerr << "Focus setup failed: " << e.what() << std::endl;
            focusModule.reset();
        }
    }

    static void TearDownTestSuite() {
        focusModule.reset();
    }
};

std::unique_ptr<Focus> FocusModuleTest::focusModule;

TEST_F(FocusModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        Focus focus(3, 64, 3, 1);
    });
}

TEST_F(FocusModuleTest, Forward_ValidInput_Success) {
    if (!focusModule) return;

    torch::Tensor input = torch::randn({1, 3, 64, 64});

    EXPECT_NO_THROW({
        auto output = focusModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
    });
}

TEST_F(FocusModuleTest, Forward_FourXChannelExpansion_CorrectBehavior) {
    Focus focus(3, 32, 1, 1);
    torch::Tensor input = torch::randn({1, 3, 64, 64});

    auto output = focus->forward(input);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 32);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(FocusModuleTest, Forward_SpatialReduction_CorrectOutputShape) {
    Focus focus(3, 48, 1, 1);
    torch::Tensor input = torch::randn({2, 3, 128, 128});

    auto output = focus->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(2), 64);
    EXPECT_EQ(output.size(3), 64);
}

// ========== ConvTranspose Module Tests ==========

class ConvTransposeModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<ConvTranspose> convTransposeModule;

    static void SetUpTestSuite() {
        try {
            convTransposeModule = std::make_unique<ConvTranspose>(64, 32, 2, 2);
        } catch (const std::exception& e) {
            std::cerr << "ConvTranspose setup failed: " << e.what() << std::endl;
            convTransposeModule.reset();
        }
    }

    static void TearDownTestSuite() {
        convTransposeModule.reset();
    }
};

std::unique_ptr<ConvTranspose> ConvTransposeModuleTest::convTransposeModule;

TEST_F(ConvTransposeModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        ConvTranspose convt(64, 32, 2, 2);
    });
}

TEST_F(ConvTransposeModuleTest, Forward_ValidInput_Success) {
    if (!convTransposeModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = convTransposeModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 32);
    });
}

TEST_F(ConvTransposeModuleTest, Forward_UpsamplingVerification_CorrectOutputShape) {
    ConvTranspose convt(64, 32, 2, 2);
    torch::Tensor input = torch::randn({1, 64, 16, 16});

    auto output = convt->forward(input);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 32);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(ConvTransposeModuleTest, Forward_WithBatchNorm_Success) {
    ConvTranspose convt(64, 32, 2, 2, 0, true);
    torch::Tensor input = torch::randn({2, 64, 16, 16});

    EXPECT_NO_THROW({
        auto output = convt->forward(input);
        EXPECT_EQ(output.size(0), 2);
        EXPECT_EQ(output.size(1), 32);
    });
}

TEST_F(ConvTransposeModuleTest, Forward_WithoutBatchNorm_Success) {
    ConvTranspose convt(64, 32, 2, 2, 0, false);
    torch::Tensor input = torch::randn({2, 64, 16, 16});

    EXPECT_NO_THROW({
        auto output = convt->forward(input);
        EXPECT_EQ(output.size(0), 2);
        EXPECT_EQ(output.size(1), 32);
    });
}
