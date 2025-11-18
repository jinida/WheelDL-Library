#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Modules/Activation.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <torch/torch.h>

using namespace WheelDL;
using namespace WheelDL::Model::Modules;
using namespace WheelDL::Utils;

// ========== AGLU Module Tests ==========

class AGLUModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<AGLU> agluModule;

    static void SetUpTestSuite() {
        try {
            agluModule = std::make_unique<AGLU>();
        } catch (const std::exception& e) {
            std::cerr << "AGLU setup failed: " << e.what() << std::endl;
            agluModule.reset();
        }
    }

    static void TearDownTestSuite() {
        agluModule.reset();
    }
};

std::unique_ptr<AGLU> AGLUModuleTest::agluModule;

TEST_F(AGLUModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        AGLU aglu;
    });
}

TEST_F(AGLUModuleTest, Forward_ValidInput_Success) {
    if (!agluModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});
    
    EXPECT_NO_THROW({
        auto output = agluModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(AGLUModuleTest, Forward_OutputShapeSameAsInput_Success) {
    if (!agluModule) return;

    torch::Tensor input = torch::randn({2, 128, 16, 16});

    auto output = agluModule->ptr()->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(AGLUModuleTest, Forward_GatingMechanism_CorrectBehavior) {
    AGLU aglu;
    torch::Tensor input = torch::randn({1, 64, 32, 32});

    auto output = aglu->forward(input);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(AGLUModuleTest, Forward_PositiveOutputValues_Success) {
    AGLU aglu;
    torch::Tensor input = torch::randn({1, 64, 32, 32});

    auto output = aglu->forward(input);
    EXPECT_TRUE(torch::all(output >= 0.0).item<bool>());
}

TEST_F(AGLUModuleTest, Forward_GradientFlow_Success) {
    AGLU aglu;
    torch::Tensor input = torch::randn({1, 64, 32, 32}, torch::requires_grad(true));

    auto output = aglu->forward(input);
    auto loss = output.sum();

    EXPECT_NO_THROW({
        loss.backward();
        EXPECT_TRUE(input.grad().defined());
    });
}

// ========== Hardswish Module Tests ==========

class HardswishModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<Hardswish> hardswishModule;

    static void SetUpTestSuite() {
        try {
            hardswishModule = std::make_unique<Hardswish>();
        } catch (const std::exception& e) {
            std::cerr << "Hardswish setup failed: " << e.what() << std::endl;
            hardswishModule.reset();
        }
    }

    static void TearDownTestSuite() {
        hardswishModule.reset();
    }
};

std::unique_ptr<Hardswish> HardswishModuleTest::hardswishModule;

TEST_F(HardswishModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        Hardswish hardswish;
    });
}

TEST_F(HardswishModuleTest, Forward_ValidInput_Success) {
    if (!hardswishModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = hardswishModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(HardswishModuleTest, Forward_OutputShapeSameAsInput_Success) {
    if (!hardswishModule) return;

    torch::Tensor input = torch::randn({2, 128, 16, 16});

    auto output = hardswishModule->ptr()->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(HardswishModuleTest, Forward_ActivationFunction_CorrectBehavior) {
    Hardswish hardswish;
    torch::Tensor input = torch::randn({1, 64, 32, 32});

    auto output = hardswish->forward(input);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(HardswishModuleTest, Forward_VariousInputValues_CorrectOutputRange) {
    Hardswish hardswish;

    torch::Tensor negative_input = torch::full({1, 64, 32, 32}, -10.0);
    auto negative_output = hardswish->forward(negative_input);
    EXPECT_TRUE(torch::all(negative_output <= 0.0).item<bool>());

    torch::Tensor positive_input = torch::full({1, 64, 32, 32}, 10.0);
    auto positive_output = hardswish->forward(positive_input);
    EXPECT_TRUE(torch::all(positive_output >= 0.0).item<bool>());
}

TEST_F(HardswishModuleTest, Forward_GradientFlow_Success) {
    Hardswish hardswish;
    torch::Tensor input = torch::randn({1, 64, 32, 32}, torch::requires_grad(true));

    auto output = hardswish->forward(input);
    auto loss = output.sum();

    EXPECT_NO_THROW({
        loss.backward();
        EXPECT_TRUE(input.grad().defined());
    });
}
