/**
 * @file ActivationTest.cpp
 * @brief Unit tests for Activation modules and createActivation utility
 *
 * Tests cover:
 * - createActivation() for all supported activation types
 * - Forward pass for each activation
 * - AGLU learnable parameters
 * - Hardswish formula verification
 * - Case-insensitive activation names
 * - Unknown activations default to ReLU
 *
 * @author WheelDL Team
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "Model/Modules/Activation.h"
#include "Model/Modules/Utils.h"

using namespace WheelDL::Model::Modules;

// ============================================================================
// Test Fixture
// ============================================================================

class ActivationTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
    }

    torch::Tensor createTestInput(std::vector<int64_t> shape = {2, 16, 8, 8}) {
        return torch::randn(shape);
    }
};

// ============================================================================
// createActivation Tests - Standard Activations
// ============================================================================

TEST_F(ActivationTest, CreateActivation_ReLU) {
    auto act = createActivation("ReLU");

    EXPECT_TRUE(act.ptr());

    auto input = createTestInput();
    auto output = act.forward<torch::Tensor>(input);

    // ReLU output should be >= 0
    EXPECT_TRUE((output >= 0).all().item<bool>());
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(ActivationTest, CreateActivation_SiLU) {
    auto act = createActivation("SiLU");

    EXPECT_TRUE(act.ptr());

    auto input = createTestInput();
    auto output = act.forward<torch::Tensor>(input);

    EXPECT_EQ(output.sizes(), input.sizes());
    EXPECT_FALSE(output.isnan().any().item<bool>());
}

TEST_F(ActivationTest, CreateActivation_Swish) {
    // Swish is an alias for SiLU
    auto act = createActivation("Swish");

    EXPECT_TRUE(act.ptr());

    auto input = createTestInput();
    auto output = act.forward<torch::Tensor>(input);

    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(ActivationTest, CreateActivation_LeakyReLU) {
    auto act = createActivation("LeakyReLU");

    EXPECT_TRUE(act.ptr());

    auto input = createTestInput();
    auto output = act.forward<torch::Tensor>(input);

    EXPECT_EQ(output.sizes(), input.sizes());
    // LeakyReLU allows negative values (with slope 0.1)
}

TEST_F(ActivationTest, CreateActivation_GELU) {
    auto act = createActivation("GELU");

    EXPECT_TRUE(act.ptr());

    auto input = createTestInput();
    auto output = act.forward<torch::Tensor>(input);

    EXPECT_EQ(output.sizes(), input.sizes());
    EXPECT_FALSE(output.isnan().any().item<bool>());
}

TEST_F(ActivationTest, CreateActivation_Mish) {
    auto act = createActivation("Mish");

    EXPECT_TRUE(act.ptr());

    auto input = createTestInput();
    auto output = act.forward<torch::Tensor>(input);

    EXPECT_EQ(output.sizes(), input.sizes());
    EXPECT_FALSE(output.isnan().any().item<bool>());
}

TEST_F(ActivationTest, CreateActivation_Hardswish) {
    auto act = createActivation("Hardswish");

    EXPECT_TRUE(act.ptr());

    auto input = createTestInput();
    auto output = act.forward<torch::Tensor>(input);

    EXPECT_EQ(output.sizes(), input.sizes());
    EXPECT_FALSE(output.isnan().any().item<bool>());
}

TEST_F(ActivationTest, CreateActivation_AGLU) {
    auto act = createActivation("AGLU");

    EXPECT_TRUE(act.ptr());

    auto input = createTestInput();
    auto output = act.forward<torch::Tensor>(input);

    EXPECT_EQ(output.sizes(), input.sizes());
    EXPECT_FALSE(output.isnan().any().item<bool>());
}

// ============================================================================
// createActivation Tests - Identity/None/Empty
// ============================================================================

TEST_F(ActivationTest, CreateActivation_Identity) {
    auto act = createActivation("Identity");

    EXPECT_TRUE(act.ptr());

    auto input = createTestInput();
    auto output = act.forward<torch::Tensor>(input);

    // Identity should return exactly the same values
    EXPECT_TRUE(torch::allclose(output, input));
}

TEST_F(ActivationTest, CreateActivation_None) {
    auto act = createActivation("None");

    EXPECT_TRUE(act.ptr());

    auto input = createTestInput();
    auto output = act.forward<torch::Tensor>(input);

    // "None" should return Identity (same values)
    EXPECT_TRUE(torch::allclose(output, input));
}

TEST_F(ActivationTest, CreateActivation_Empty) {
    auto act = createActivation("");

    EXPECT_TRUE(act.ptr());

    auto input = createTestInput();
    auto output = act.forward<torch::Tensor>(input);

    // Empty string should return Identity (same values)
    EXPECT_TRUE(torch::allclose(output, input));
}

// ============================================================================
// createActivation Tests - Unknown/Default Behavior
// ============================================================================

TEST_F(ActivationTest, CreateActivation_Unknown_DefaultsToReLU) {
    // Unknown activation names should default to ReLU, NOT throw exception
    auto act = createActivation("UnknownActivation");

    EXPECT_TRUE(act.ptr());

    auto input = createTestInput();
    auto output = act.forward<torch::Tensor>(input);

    // Should behave like ReLU (all values >= 0)
    EXPECT_TRUE((output >= 0).all().item<bool>());
}

TEST_F(ActivationTest, CreateActivation_InvalidName_DefaultsToReLU) {
    auto act = createActivation("SomeRandomName123");

    EXPECT_TRUE(act.ptr());

    // Should not throw - defaults to ReLU
    auto input = createTestInput();
    EXPECT_NO_THROW({
        auto output = act.forward<torch::Tensor>(input);
        EXPECT_TRUE((output >= 0).all().item<bool>());
    });
}

// ============================================================================
// createActivation Tests - Case Insensitivity
// ============================================================================

TEST_F(ActivationTest, CreateActivation_CaseInsensitive_SILU) {
    auto act1 = createActivation("SILU");
    auto act2 = createActivation("silu");
    auto act3 = createActivation("SiLu");

    auto input = createTestInput();

    auto output1 = act1.forward<torch::Tensor>(input);
    auto output2 = act2.forward<torch::Tensor>(input);
    auto output3 = act3.forward<torch::Tensor>(input);

    // All should produce the same output
    EXPECT_TRUE(torch::allclose(output1, output2));
    EXPECT_TRUE(torch::allclose(output2, output3));
}

TEST_F(ActivationTest, CreateActivation_CaseInsensitive_RELU) {
    auto act1 = createActivation("RELU");
    auto act2 = createActivation("relu");
    auto act3 = createActivation("ReLU");

    auto input = createTestInput();

    auto output1 = act1.forward<torch::Tensor>(input);
    auto output2 = act2.forward<torch::Tensor>(input);
    auto output3 = act3.forward<torch::Tensor>(input);

    // All should produce the same output
    EXPECT_TRUE(torch::allclose(output1, output2));
    EXPECT_TRUE(torch::allclose(output2, output3));
}

TEST_F(ActivationTest, CreateActivation_CaseInsensitive_LeakyReLU) {
    auto act1 = createActivation("LEAKYRELU");
    auto act2 = createActivation("leakyrelu");
    auto act3 = createActivation("LeakyReLU");

    auto input = createTestInput();

    auto output1 = act1.forward<torch::Tensor>(input);
    auto output2 = act2.forward<torch::Tensor>(input);
    auto output3 = act3.forward<torch::Tensor>(input);

    EXPECT_TRUE(torch::allclose(output1, output2));
    EXPECT_TRUE(torch::allclose(output2, output3));
}

// ============================================================================
// Forward Pass Tests - All Activations
// ============================================================================

TEST_F(ActivationTest, Forward_AllActivations_ValidOutput) {
    std::vector<std::string> activations = {
        "ReLU", "SiLU", "LeakyReLU", "GELU", "Mish",
        "Hardswish", "AGLU", "Identity"
    };

    auto input = createTestInput();

    for (const auto& actName : activations) {
        auto act = createActivation(actName);
        auto output = act.forward<torch::Tensor>(input);

        EXPECT_EQ(output.sizes(), input.sizes())
            << "Failed for activation: " << actName;
        EXPECT_FALSE(output.isnan().any().item<bool>())
            << "NaN detected for activation: " << actName;
        EXPECT_FALSE(output.isinf().any().item<bool>())
            << "Inf detected for activation: " << actName;
    }
}

TEST_F(ActivationTest, Forward_AllActivations_BatchSize) {
    std::vector<std::string> activations = {
        "ReLU", "SiLU", "LeakyReLU", "GELU", "Mish",
        "Hardswish", "AGLU", "Identity"
    };

    auto input = torch::randn({8, 64, 16, 16});

    for (const auto& actName : activations) {
        auto act = createActivation(actName);
        auto output = act.forward<torch::Tensor>(input);

        EXPECT_EQ(output.size(0), 8)  // Batch preserved
            << "Batch size not preserved for: " << actName;
        EXPECT_EQ(output.size(1), 64)
            << "Channel size changed for: " << actName;
    }
}

// ============================================================================
// Gradient Tests
// ============================================================================

TEST_F(ActivationTest, Gradient_AllActivations_Backward) {
    std::vector<std::string> activations = {
        "ReLU", "SiLU", "LeakyReLU", "GELU", "Mish",
        "Hardswish", "AGLU"  // Exclude Identity for gradient test
    };

    for (const auto& actName : activations) {
        auto input = torch::randn({2, 8, 4, 4}, torch::requires_grad(true));
        auto act = createActivation(actName);
        auto output = act.forward<torch::Tensor>(input);

        auto loss = output.sum();
        loss.backward();

        EXPECT_TRUE(input.grad().defined())
            << "Gradient not defined for: " << actName;
        EXPECT_FALSE(input.grad().isnan().any().item<bool>())
            << "NaN gradient for: " << actName;
    }
}

// ============================================================================
// AGLU Specific Tests
// ============================================================================

TEST_F(ActivationTest, AGLU_LearnableParams) {
    AGLU aglu;

    // AGLU should have learnable parameters
    auto params = aglu->parameters();

    // Should have lambda and kappa parameters
    EXPECT_GE(params.size(), 2);

    // Parameters should be registered
    bool hasLambda = false;
    bool hasKappa = false;

    for (const auto& param : aglu->named_parameters()) {
        if (param.key() == "lambda") hasLambda = true;
        if (param.key() == "kappa") hasKappa = true;
    }

    EXPECT_TRUE(hasLambda) << "AGLU should have lambda parameter";
    EXPECT_TRUE(hasKappa) << "AGLU should have kappa parameter";
}

TEST_F(ActivationTest, AGLU_Forward) {
    AGLU aglu;

    auto input = torch::randn({1, 16, 4, 4});
    auto output = aglu->forward(input);

    EXPECT_EQ(output.sizes(), input.sizes());
    EXPECT_FALSE(output.isnan().any().item<bool>());
    EXPECT_FALSE(output.isinf().any().item<bool>());
}

TEST_F(ActivationTest, AGLU_PositiveOutput) {
    AGLU aglu;

    auto input = torch::randn({2, 8, 4, 4});
    auto output = aglu->forward(input);

    // AGLU output should be positive (due to exp)
    EXPECT_TRUE((output > 0).all().item<bool>());
}

// ============================================================================
// Hardswish Specific Tests
// ============================================================================

TEST_F(ActivationTest, Hardswish_Formula) {
    Hardswish hardswish;

    // Test formula: x * relu6(x + 3) / 6
    // where relu6(x) = min(max(x, 0), 6)
    auto input = torch::tensor({-4.0f, -3.0f, -1.0f, 0.0f, 1.0f, 3.0f, 4.0f});
    auto output = hardswish->forward(input);

    // Manual calculation for verification:
    // x=-4: -4 * clamp(-4+3, 0, 6) / 6 = -4 * 0 / 6 = 0
    // x=-3: -3 * clamp(-3+3, 0, 6) / 6 = -3 * 0 / 6 = 0
    // x=-1: -1 * clamp(-1+3, 0, 6) / 6 = -1 * 2 / 6 = -0.333...
    // x=0:  0 * clamp(0+3, 0, 6) / 6 = 0 * 3 / 6 = 0
    // x=1:  1 * clamp(1+3, 0, 6) / 6 = 1 * 4 / 6 = 0.666...
    // x=3:  3 * clamp(3+3, 0, 6) / 6 = 3 * 6 / 6 = 3
    // x=4:  4 * clamp(4+3, 0, 6) / 6 = 4 * 6 / 6 = 4

    auto expected = input * torch::clamp(input + 3.0f, 0.0f, 6.0f) / 6.0f;

    EXPECT_TRUE(torch::allclose(output, expected, 1e-5, 1e-5));
}

TEST_F(ActivationTest, Hardswish_BoundaryValues) {
    Hardswish hardswish;

    // Test boundary conditions
    auto input = torch::tensor({-3.0f, 3.0f});
    auto output = hardswish->forward(input);

    // At x=-3: output should be 0
    EXPECT_NEAR(output[0].item<float>(), 0.0f, 1e-5);

    // At x=3: output should be x (linear region)
    EXPECT_NEAR(output[1].item<float>(), 3.0f, 1e-5);
}

TEST_F(ActivationTest, Hardswish_Forward) {
    Hardswish hardswish;

    auto input = torch::randn({2, 16, 8, 8});
    auto output = hardswish->forward(input);

    EXPECT_EQ(output.sizes(), input.sizes());
    EXPECT_FALSE(output.isnan().any().item<bool>());
    EXPECT_FALSE(output.isinf().any().item<bool>());
}

// ============================================================================
// Different Input Shapes
// ============================================================================

TEST_F(ActivationTest, Activation_1DInput) {
    auto act = createActivation("ReLU");
    auto input = torch::randn({10});
    auto output = act.forward<torch::Tensor>(input);

    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(ActivationTest, Activation_2DInput) {
    auto act = createActivation("SiLU");
    auto input = torch::randn({4, 16});
    auto output = act.forward<torch::Tensor>(input);

    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(ActivationTest, Activation_3DInput) {
    auto act = createActivation("GELU");
    auto input = torch::randn({2, 8, 16});
    auto output = act.forward<torch::Tensor>(input);

    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST_F(ActivationTest, Activation_5DInput) {
    auto act = createActivation("Mish");
    auto input = torch::randn({2, 3, 4, 8, 8});
    auto output = act.forward<torch::Tensor>(input);

    EXPECT_EQ(output.sizes(), input.sizes());
}

