/**
 * @file HeadTest.cpp
 * @brief Unit tests for Head modules
 *
 * Tests cover:
 * - Detect, OBB, Classify, Segment
 * - Anomaly (EfficientAD, PatchCore)
 *
 * @author WheelDL Team
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "Model/Modules/Head.h"

using namespace WheelDL::Model::Modules;

// ============================================================================
// Test Fixture
// ============================================================================

class HeadTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
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
// Detect Tests
// ============================================================================

TEST_F(HeadTest, Detect_Constructor) {
    Detect detect(80, std::vector<int64_t>{64, 128, 256});

    EXPECT_TRUE(detect.ptr());
}

TEST_F(HeadTest, Detect_Constructor_DefaultNC) {
    Detect detect(80, std::vector<int64_t>{64, 128, 256});

    EXPECT_EQ(detect->nc, 80);
}

TEST_F(HeadTest, Detect_Constructor_CustomNC) {
    Detect detect(10, std::vector<int64_t>{64, 128, 256});

    EXPECT_EQ(detect->nc, 10);
}

TEST_F(HeadTest, Detect_Constructor_NumLayers) {
    Detect detect(80, std::vector<int64_t>{64, 128, 256});

    EXPECT_EQ(detect->nl, 3);  // 3 detection layers
}

TEST_F(HeadTest, Detect_Forward) {
    Detect detect(80, std::vector<int64_t>{64, 128, 256});
    detect->train();

    auto inputs = createMultiScaleInput(1, {64, 128, 256}, {80, 40, 20});
    auto outputs = detect->forward(inputs);

    EXPECT_FALSE(outputs.empty());
}

TEST_F(HeadTest, Detect_Forward_BatchSize) {
    Detect detect(80, std::vector<int64_t>{64, 128, 256});
    detect->train();

    auto inputs = createMultiScaleInput(4, {64, 128, 256}, {80, 40, 20});
    auto outputs = detect->forward(inputs);

    EXPECT_FALSE(outputs.empty());
}

TEST_F(HeadTest, Detect_BiasInit) {
    Detect detect(80, std::vector<int64_t>{64, 128, 256});

    EXPECT_NO_THROW(detect->biasInit());
}

TEST_F(HeadTest, Detect_RegMax) {
    Detect detect(80, std::vector<int64_t>{64, 128, 256});

    EXPECT_EQ(detect->regMax, 16);  // Default reg_max
}

TEST_F(HeadTest, Detect_OutputChannels) {
    int64_t nc = 80;
    int64_t regMax = 16;
    Detect detect(nc, std::vector<int64_t>{64, 128, 256});

    // no = nc + regMax * 4
    EXPECT_EQ(detect->no, nc + regMax * 4);
}

// ============================================================================
// OBB Tests
// ============================================================================

TEST_F(HeadTest, OBB_Constructor) {
    OBB obb(80, 1, std::vector<int64_t>{64, 128, 256});

    EXPECT_TRUE(obb.ptr());
}

TEST_F(HeadTest, OBB_Constructor_DefaultNE) {
    OBB obb(80, 1, std::vector<int64_t>{64, 128, 256});

    EXPECT_EQ(obb->ne, 1);
}

TEST_F(HeadTest, OBB_Constructor_CustomNE) {
    OBB obb(80, 2, std::vector<int64_t>{64, 128, 256});

    EXPECT_EQ(obb->ne, 2);
}

TEST_F(HeadTest, OBB_Forward) {
    OBB obb(80, 1, std::vector<int64_t>{64, 128, 256});
    obb->train();

    auto inputs = createMultiScaleInput(1, {64, 128, 256}, {80, 40, 20});
    auto outputs = obb->forward(inputs);

    EXPECT_FALSE(outputs.empty());
}

TEST_F(HeadTest, OBB_Forward_BatchSize) {
    OBB obb(80, 1, std::vector<int64_t>{64, 128, 256});
    obb->train();

    auto inputs = createMultiScaleInput(4, {64, 128, 256}, {80, 40, 20});
    auto outputs = obb->forward(inputs);

    EXPECT_FALSE(outputs.empty());
}

TEST_F(HeadTest, OBB_InheritsFromDetect) {
    OBB obb(80, 1, std::vector<int64_t>{64, 128, 256});

    // OBB should have nc from Detect
    EXPECT_EQ(obb->nc, 80);
}

// ============================================================================
// Classify Tests
// ============================================================================

TEST_F(HeadTest, Classify_Constructor) {
    Classify classify(512, 1000);

    EXPECT_TRUE(classify.ptr());
}

TEST_F(HeadTest, Classify_Constructor_CustomParams) {
    Classify classify(512, 100, 3, 2, 1, 1);

    EXPECT_TRUE(classify.ptr());
}

TEST_F(HeadTest, Classify_Forward) {
    Classify classify(512, 1000);

    auto input = createInput(1, 512, 7, 7);
    auto outputs = classify->forward({input});

    EXPECT_FALSE(outputs.empty());
    EXPECT_EQ(outputs[0].size(1), 1000);  // num classes
}

TEST_F(HeadTest, Classify_Forward_BatchSize) {
    Classify classify(512, 1000);

    auto input = createInput(4, 512, 7, 7);
    auto outputs = classify->forward({input});

    EXPECT_EQ(outputs[0].size(0), 4);
}

TEST_F(HeadTest, Classify_Forward_DifferentInputSize) {
    Classify classify(256, 100);

    auto input = createInput(1, 256, 14, 14);
    auto outputs = classify->forward({input});

    EXPECT_EQ(outputs[0].size(1), 100);
}

TEST_F(HeadTest, Classify_NoNaN) {
    Classify classify(512, 1000);

    auto input = createInput(1, 512, 7, 7);
    auto outputs = classify->forward({input});

    EXPECT_FALSE(outputs[0].isnan().any().item<bool>());
    EXPECT_FALSE(outputs[0].isinf().any().item<bool>());
}

// ============================================================================
// Segment Tests
// ============================================================================

TEST_F(HeadTest, Segment_Constructor) {
    Segment segment(256, 21);  // 21 classes for typical segmentation

    EXPECT_TRUE(segment.ptr());
}

TEST_F(HeadTest, Segment_Constructor_CustomActivation) {
    Segment segment(256, 21, "SiLU");

    EXPECT_TRUE(segment.ptr());
}

TEST_F(HeadTest, Segment_Forward) {
    Segment segment(256, 21);

    auto input = createInput(1, 256, 64, 64);
    auto outputs = segment->forward({input});

    EXPECT_FALSE(outputs.empty());
    EXPECT_EQ(outputs[0].size(1), 21);  // num classes
}

TEST_F(HeadTest, Segment_Forward_BatchSize) {
    Segment segment(256, 21);

    auto input = createInput(4, 256, 64, 64);
    auto outputs = segment->forward({input});

    EXPECT_EQ(outputs[0].size(0), 4);
}

TEST_F(HeadTest, Segment_Forward_SpatialPreserved) {
    Segment segment(256, 21);

    auto input = createInput(1, 256, 32, 32);
    auto outputs = segment->forward({input});

    EXPECT_EQ(outputs[0].size(2), 32);
    EXPECT_EQ(outputs[0].size(3), 32);
}

TEST_F(HeadTest, Segment_NoNaN) {
    Segment segment(256, 21);

    auto input = createInput(1, 256, 32, 32);
    auto outputs = segment->forward({input});

    EXPECT_FALSE(outputs[0].isnan().any().item<bool>());
    EXPECT_FALSE(outputs[0].isinf().any().item<bool>());
}

TEST_F(HeadTest, Segment_BinarySegmentation) {
    Segment segment(64, 1);  // Binary segmentation

    auto input = createInput(1, 64, 128, 128);
    auto outputs = segment->forward({input});

    EXPECT_EQ(outputs[0].size(1), 1);
}

// ============================================================================
// Multi-Class Tests
// ============================================================================

TEST_F(HeadTest, Detect_DifferentNumClasses) {
    std::vector<int64_t> numClasses = {10, 20, 80, 100};

    for (auto nc : numClasses) {
        Detect detect(nc, std::vector<int64_t>{64, 128, 256});
        EXPECT_EQ(detect->nc, nc);
    }
}

TEST_F(HeadTest, Classify_DifferentNumClasses) {
    std::vector<int64_t> numClasses = {10, 100, 1000};

    for (auto nc : numClasses) {
        Classify classify(512, nc);
        auto input = createInput(1, 512, 7, 7);
        auto outputs = classify->forward({input});
        EXPECT_EQ(outputs[0].size(1), nc);
    }
}

TEST_F(HeadTest, Segment_DifferentNumClasses) {
    std::vector<int64_t> numClasses = {2, 21, 150};

    for (auto nc : numClasses) {
        Segment segment(256, nc);
        auto input = createInput(1, 256, 32, 32);
        auto outputs = segment->forward({input});
        EXPECT_EQ(outputs[0].size(1), nc);
    }
}

// ============================================================================
// Channel Configuration Tests
// ============================================================================

TEST_F(HeadTest, Detect_DifferentChannelConfig) {
    // Test with different backbone channel configurations
    std::vector<std::vector<int64_t>> configs = {
        {32, 64, 128},
        {64, 128, 256},
        {128, 256, 512}
    };

    for (const auto& ch : configs) {
        Detect detect(80, ch);
        EXPECT_EQ(detect->nl, static_cast<int64_t>(ch.size()));
    }
}

TEST_F(HeadTest, Classify_DifferentInputChannels) {
    std::vector<int64_t> inputChannels = {256, 512, 1024, 2048};

    for (auto c1 : inputChannels) {
        Classify classify(c1, 1000);
        auto input = createInput(1, c1, 7, 7);
        auto outputs = classify->forward({input});
        EXPECT_EQ(outputs[0].size(1), 1000);
    }
}

TEST_F(HeadTest, Segment_DifferentInputChannels) {
    std::vector<int64_t> inputChannels = {64, 128, 256, 512};

    for (auto c1 : inputChannels) {
        Segment segment(c1, 21);
        auto input = createInput(1, c1, 32, 32);
        auto outputs = segment->forward({input});
        EXPECT_EQ(outputs[0].size(1), 21);
    }
}

// ============================================================================
// Training Mode Tests
// ============================================================================

TEST_F(HeadTest, Detect_TrainingMode) {
    Detect detect(80, std::vector<int64_t>{64, 128, 256});

    detect->train();
    EXPECT_TRUE(detect->is_training());

    detect->eval();
    EXPECT_FALSE(detect->is_training());
}

TEST_F(HeadTest, Classify_TrainingMode) {
    Classify classify(512, 1000);

    classify->train();
    EXPECT_TRUE(classify->is_training());

    classify->eval();
    EXPECT_FALSE(classify->is_training());
}

TEST_F(HeadTest, Segment_TrainingMode) {
    Segment segment(256, 21);

    segment->train();
    EXPECT_TRUE(segment->is_training());

    segment->eval();
    EXPECT_FALSE(segment->is_training());
}

// ============================================================================
// Gradient Tests
// ============================================================================

TEST_F(HeadTest, Classify_Gradient) {
    Classify classify(512, 1000);

    auto input = torch::randn({1, 512, 7, 7}, torch::requires_grad(true));
    auto outputs = classify->forward({input});
    auto loss = outputs[0].sum();
    loss.backward();

    EXPECT_TRUE(input.grad().defined());
    EXPECT_FALSE(input.grad().isnan().any().item<bool>());
}

TEST_F(HeadTest, Segment_Gradient) {
    Segment segment(256, 21);

    auto input = torch::randn({1, 256, 32, 32}, torch::requires_grad(true));
    auto outputs = segment->forward({input});
    auto loss = outputs[0].sum();
    loss.backward();

    EXPECT_TRUE(input.grad().defined());
    EXPECT_FALSE(input.grad().isnan().any().item<bool>());
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(HeadTest, Detect_SmallFeatureMap) {
    Detect detect(80, std::vector<int64_t>{64, 128, 256});
    detect->train();

    auto inputs = createMultiScaleInput(1, {64, 128, 256}, {20, 10, 5});
    auto outputs = detect->forward(inputs);

    EXPECT_FALSE(outputs.empty());
}

TEST_F(HeadTest, Classify_SmallSpatialSize) {
    Classify classify(512, 1000);
    // Use eval mode to avoid BatchNorm training issue with 1x1 spatial size
    classify->eval();

    auto input = createInput(1, 512, 1, 1);
    auto outputs = classify->forward({input});

    EXPECT_EQ(outputs[0].size(1), 1000);
}

TEST_F(HeadTest, Segment_LargeSpatialSize) {
    Segment segment(64, 21);

    auto input = createInput(1, 64, 256, 256);
    auto outputs = segment->forward({input});

    EXPECT_EQ(outputs[0].size(2), 256);
    EXPECT_EQ(outputs[0].size(3), 256);
}

// ============================================================================
// Two-Scale Detection Tests
// ============================================================================

TEST_F(HeadTest, Detect_TwoScales) {
    Detect detect(80, std::vector<int64_t>{64, 128});
    detect->train();

    auto inputs = createMultiScaleInput(1, {64, 128}, {40, 20});
    auto outputs = detect->forward(inputs);

    EXPECT_FALSE(outputs.empty());
}

TEST_F(HeadTest, OBB_TwoScales) {
    OBB obb(80, 1, std::vector<int64_t>{64, 128});
    obb->train();

    auto inputs = createMultiScaleInput(1, {64, 128}, {40, 20});
    auto outputs = obb->forward(inputs);

    EXPECT_FALSE(outputs.empty());
}

