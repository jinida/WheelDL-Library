#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Optimizer/EMA/ModelEMA.h"
#include "WheelDL.Lib/Model/Task/BaseModel.h"
#include "WheelDL.Lib/Model/Loss/BaseLoss.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"

using namespace WheelDL::Optimizer::EMA;
using namespace WheelDL::Model;

// Simple test model for EMA testing
class SimpleTestModel : public BaseModel {
public:
    SimpleTestModel() {
        // Create a simple sequential model with some layers
        // Include BatchNorm1d to test buffer updates (running_mean, running_var)
        auto linear1 = torch::nn::Linear(10, 5);
        auto batchnorm = torch::nn::BatchNorm1d(5);
        auto relu = torch::nn::ReLU();
        auto linear2 = torch::nn::Linear(5, 2);

        _model = torch::nn::Sequential(linear1, batchnorm, relu, linear2);
        register_module("model", _model);

        _isInitialized = true;
        _taskType = WheelDL::TaskType::CLASSIFICATION;
    }

    std::unique_ptr<Loss::BaseLoss> initCriterion() override {
        return nullptr;  // Not needed for EMA tests
    }
};

class ModelEMATest : public ::testing::Test {
protected:
    void SetUp() override {
        model = std::make_shared<SimpleTestModel>();
    }

    std::shared_ptr<SimpleTestModel> model;
};

TEST_F(ModelEMATest, Construction) {
    EXPECT_NO_THROW({
        ModelEMA ema(*model);
    });
}

TEST_F(ModelEMATest, ConstructionWithParameters) {
    EXPECT_NO_THROW({
        ModelEMA ema(*model, 0.9999f, 2000, 0);
    });
}

TEST_F(ModelEMATest, InvalidMaxDecay) {
    EXPECT_THROW(
        ModelEMA(*model, 0.0f),
        WheelDL::Utils::ConfigurationException
    );

    EXPECT_THROW(
        ModelEMA(*model, 1.0f),
        WheelDL::Utils::ConfigurationException
    );

    EXPECT_THROW(
        ModelEMA(*model, 1.5f),
        WheelDL::Utils::ConfigurationException
    );
}

TEST_F(ModelEMATest, InvalidDecayRamp) {
    EXPECT_THROW(
        ModelEMA(*model, 0.9999f, 0),
        WheelDL::Utils::ConfigurationException
    );

    EXPECT_THROW(
        ModelEMA(*model, 0.9999f, -100),
        WheelDL::Utils::ConfigurationException
    );
}

TEST_F(ModelEMATest, InitialCopy) {
    ModelEMA ema(*model);

    auto emaModel = ema.getEMAModel();

    // EMA model should initially have same parameters as original
    auto sourceModel = model->getModel();
    auto originalParams = sourceModel->named_parameters();
    auto emaParams = emaModel->named_parameters();

    // Check that EMA model has parameters
    ASSERT_GT(emaParams.size(), 0);

    for (auto& origParam : originalParams) {
        auto emaIter = std::find_if(emaParams.begin(), emaParams.end(),
            [&](const auto& p) { return p.key() == origParam.key(); });

        ASSERT_NE(emaIter, emaParams.end());
        EXPECT_TRUE(torch::allclose(origParam.value(), emaIter->value()));
    }
}

TEST_F(ModelEMATest, DynamicDecay) {
    ModelEMA ema(*model, 0.9999f, 2000);

    // Initially decay should be 0
    EXPECT_FLOAT_EQ(ema.getDecay(), 0.0f);

    // After 1 update
    ema.update(*model);
    float decay1 = ema.getDecay();
    EXPECT_GT(decay1, 0.0f);
    EXPECT_LT(decay1, 0.9999f);

    // After 2000 updates (decay_ramp), decay should be close to max
    for (int i = 1; i < 2000; ++i) {
        ema.update(*model);
    }
    float decay2000 = ema.getDecay();
    EXPECT_GT(decay2000, decay1);
    // At decay_ramp, decay = max_decay * (1 - exp(-1)) ≈ max_decay * 0.632
    EXPECT_NEAR(decay2000, 0.9999f * (1.0f - std::exp(-1.0f)), 0.001f);
}

TEST_F(ModelEMATest, UpdateChangesParameters) {
    ModelEMA ema(*model, 0.9f, 10);  // Lower max decay and short ramp for visible changes

    auto emaModel = ema.getEMAModel();
    auto emaParams = emaModel->named_parameters();
    auto originalEMAWeight = emaParams.begin()->value().clone();

    // Modify model parameters
    auto sourceModel = model->getModel();
    auto params = sourceModel->parameters();
    for (auto& param : params) {
        param.data().mul_(2.0f);
    }

    // Update EMA multiple times to ramp up decay
    for (int i = 0; i < 10; ++i) {
        ema.update(*model);
    }

    // EMA parameters should have changed
    auto updatedEMAParams = emaModel->named_parameters();
    auto updatedEMAWeight = updatedEMAParams.begin()->value();
    EXPECT_FALSE(torch::allclose(originalEMAWeight, updatedEMAWeight));
}

TEST_F(ModelEMATest, UpdateFormula) {
    ModelEMA ema(*model, 0.9f, 100);

    auto emaModel = ema.getEMAModel();

    // Do several updates to ramp up decay
    for (int i = 0; i < 100; ++i) {
        ema.update(*model);
    }

    // Get EMA weight before next update
    auto emaWeightBefore = emaModel->named_parameters().begin()->value().clone();

    // Modify model weight
    auto sourceModel = model->getModel();
    auto sourceParams = sourceModel->named_parameters();
    auto firstSourceParam = sourceParams.begin();
    auto currentWeight = firstSourceParam->value().clone();
    firstSourceParam->value().data().add_(1.0f);  // Add 1 to all elements
    auto newWeight = firstSourceParam->value().clone();

    // The update() function uses decay based on current _updates (before increment)
    int currentUpdates = ema.getUpdates();  // This is 100
    float expectedDecay = 0.9f * (1.0f - std::exp(-static_cast<float>(currentUpdates) / 100.0f));

    // Update EMA
    ema.update(*model);

    // Get updated EMA weight
    auto updatedEMAWeight = emaModel->named_parameters().begin()->value();

    // Manually compute expected EMA: ema_new = decay * ema_old + (1 - decay) * current
    auto expected = emaWeightBefore * expectedDecay + newWeight * (1.0f - expectedDecay);

    EXPECT_TRUE(torch::allclose(updatedEMAWeight, expected, 1e-3));
}

TEST_F(ModelEMATest, MultipleUpdates) {
    ModelEMA ema(*model);

    EXPECT_EQ(ema.getUpdates(), 0);

    ema.update(*model);
    EXPECT_EQ(ema.getUpdates(), 1);

    ema.update(*model);
    EXPECT_EQ(ema.getUpdates(), 2);

    ema.update(*model);
    EXPECT_EQ(ema.getUpdates(), 3);
}

TEST_F(ModelEMATest, ResetUpdates) {
    ModelEMA ema(*model);

    ema.update(*model);
    ema.update(*model);
    EXPECT_EQ(ema.getUpdates(), 2);

    ema.resetUpdates();
    EXPECT_EQ(ema.getUpdates(), 0);
}

TEST_F(ModelEMATest, SetMaxDecay) {
    ModelEMA ema(*model, 0.9999f);

    EXPECT_FLOAT_EQ(ema.getMaxDecay(), 0.9999f);

    ema.setMaxDecay(0.999f);
    EXPECT_FLOAT_EQ(ema.getMaxDecay(), 0.999f);

    EXPECT_THROW(ema.setMaxDecay(0.0f), WheelDL::Utils::ConfigurationException);
    EXPECT_THROW(ema.setMaxDecay(1.0f), WheelDL::Utils::ConfigurationException);
}

TEST_F(ModelEMATest, SetDecayRamp) {
    ModelEMA ema(*model, 0.9999f, 2000);

    EXPECT_EQ(ema.getDecayRamp(), 2000);

    ema.setDecayRamp(1000);
    EXPECT_EQ(ema.getDecayRamp(), 1000);

    EXPECT_THROW(ema.setDecayRamp(0), WheelDL::Utils::ConfigurationException);
    EXPECT_THROW(ema.setDecayRamp(-100), WheelDL::Utils::ConfigurationException);
}

TEST_F(ModelEMATest, EMAModelIsInEvalMode) {
    ModelEMA ema(*model);

    auto emaModel = ema.getEMAModel();
    EXPECT_FALSE(emaModel->is_training());
}

TEST_F(ModelEMATest, EMAModelHasNoGradients) {
    ModelEMA ema(*model);

    auto emaModel = ema.getEMAModel();
    auto emaParams = emaModel->parameters();

    for (auto& param : emaParams) {
        EXPECT_FALSE(param.requires_grad());
    }
}

TEST_F(ModelEMATest, EnableDisable) {
    ModelEMA ema(*model, 0.9f, 10);

    EXPECT_TRUE(ema.isEnabled());

    auto emaModel = ema.getEMAModel();
    auto originalEMAWeight = emaModel->named_parameters().begin()->value().clone();

    // Modify model
    auto sourceModel = model->getModel();
    auto params = sourceModel->parameters();
    for (auto& param : params) {
        param.data().mul_(2.0f);
    }

    // Disable and update - should not change
    ema.setEnabled(false);
    EXPECT_FALSE(ema.isEnabled());

    int updatesBefore = ema.getUpdates();
    ema.update(*model);

    // Updates counter should not increase when disabled
    EXPECT_EQ(ema.getUpdates(), updatesBefore);

    // EMA parameters should not change when disabled
    auto emaWeight = emaModel->named_parameters().begin()->value();
    EXPECT_TRUE(torch::allclose(originalEMAWeight, emaWeight));

    // Enable and update - should change
    ema.setEnabled(true);
    for (int i = 0; i < 10; ++i) {
        ema.update(*model);
    }

    auto updatedWeight = emaModel->named_parameters().begin()->value();
    EXPECT_FALSE(torch::allclose(originalEMAWeight, updatedWeight));
}

TEST_F(ModelEMATest, SlowDecayPreservesParameters) {
    ModelEMA ema(*model, 0.9999f, 100);  // Very high max decay, short ramp

    auto emaModel = ema.getEMAModel();

    // Do many updates to reach high decay (close to max_decay)
    // With decay_ramp=100, after 1000 updates: decay ≈ 0.9999 * (1 - exp(-10)) ≈ 0.9999
    for (int i = 0; i < 1000; ++i) {
        ema.update(*model);
    }

    // At this point, decay should be very close to 0.9999
    float currentDecay = ema.getDecay();
    EXPECT_GT(currentDecay, 0.9998f);  // Should be very high

    auto originalEMAWeight = emaModel->named_parameters().begin()->value().clone();

    // Large change to model
    auto sourceModel = model->getModel();
    auto sourceParams = sourceModel->named_parameters();
    for (auto& param : sourceParams) {
        param.value().data().mul_(10.0f);
    }

    // Single update with high decay (slow update)
    ema.update(*model);

    auto updatedEMAWeight = emaModel->named_parameters().begin()->value();

    // With high decay (0.9999), EMA should change very slowly
    // ema_new = 0.9999 * ema_old + 0.0001 * current
    // So it should be much closer to original than to new weight
    auto firstSourceParam = sourceParams.begin();
    auto distToOriginal = (updatedEMAWeight - originalEMAWeight).abs().mean().item<float>();
    auto distToNew = (updatedEMAWeight - firstSourceParam->value()).abs().mean().item<float>();

    EXPECT_LT(distToOriginal, distToNew);
}

TEST_F(ModelEMATest, ResumeFromUpdateCount) {
    // Create EMA with initial update count
    ModelEMA ema(*model, 0.9999f, 2000, 1000);

    EXPECT_EQ(ema.getUpdates(), 1000);

    // Decay should be computed based on initial update count
    float expectedDecay = 0.9999f * (1.0f - std::exp(-1000.0f / 2000.0f));
    EXPECT_NEAR(ema.getDecay(), expectedDecay, 0.001f);
}

TEST_F(ModelEMATest, BufferUpdatePolicy) {
    // Test that buffers (like BatchNorm running_mean, running_var) are COPIED,
    // not updated with EMA decay like parameters

    ModelEMA ema(*model, 0.9f, 100);

    auto emaModel = ema.getEMAModel();
    auto sourceModel = model->getModel();

    // Do some updates to warm up
    for (int i = 0; i < 10; ++i) {
        ema.update(*model);
    }

    // Get buffers from both models
    auto sourceBuffers = sourceModel->named_buffers();
    auto emaBuffers = emaModel->named_buffers();

    // Verify we have buffers (from BatchNorm)
    ASSERT_GT(sourceBuffers.size(), 0);

    // Manually modify source model's buffer values
    for (auto& buffer : sourceBuffers) {
        if (buffer.key().find("running_mean") != std::string::npos ||
            buffer.key().find("running_var") != std::string::npos) {
            // Set to a distinctive value
            buffer.value().fill_(42.0f);
        }
    }

    // Update EMA
    ema.update(*model);

    // Verify that EMA buffers now EXACTLY match source buffers (not EMA'd)
    for (auto& sourceBuffer : sourceBuffers) {
        const std::string& name = sourceBuffer.key();
        auto emaIter = std::find_if(emaBuffers.begin(), emaBuffers.end(),
            [&name](const auto& b) { return b.key() == name; });

        if (emaIter != emaBuffers.end()) {
            const auto& sourceValue = sourceBuffer.value();
            const auto& emaValue = emaIter->value();

            // Buffers should be exactly equal (copied, not EMA'd)
            EXPECT_TRUE(torch::allclose(emaValue, sourceValue, 1e-6))
                << "Buffer " << name << " should be copied exactly, not updated with decay";
        }
    }
}

TEST_F(ModelEMATest, DeviceConsistency) {
    // Test that EMA model is on the same device as source model

    if (!torch::cuda::is_available()) {
        // Skip test if CUDA is not available
        return;
    }

    // Move model to CUDA
    model->to(torch::kCUDA);

    // Create EMA
    ModelEMA ema(*model);

    auto emaModel = ema.getEMAModel();

    // Verify EMA model parameters are on CUDA
    auto emaParams = emaModel->parameters();
    ASSERT_GT(emaParams.size(), 0);

    for (const auto& param : emaParams) {
        EXPECT_TRUE(param.is_cuda())
            << "EMA parameter should be on CUDA device";
    }

    // Verify buffers are also on CUDA
    auto emaBuffers = emaModel->buffers();
    for (const auto& buffer : emaBuffers) {
        EXPECT_TRUE(buffer.is_cuda())
            << "EMA buffer should be on CUDA device";
    }
}
