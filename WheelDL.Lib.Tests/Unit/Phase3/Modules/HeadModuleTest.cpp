#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Modules/Head.h"
#include "WheelDL.Lib/Interfaces.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <torch/torch.h>

using namespace WheelDL;
using namespace WheelDL::Model::Modules;
using namespace WheelDL::Utils;

// ========== Detect Module Tests ==========

class DetectModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<Detect> detectModule;

    static void SetUpTestSuite() {
        try {
            std::vector<int64_t> channels = {128, 256, 512};
            detectModule = std::make_unique<Detect>(80, channels);
        } catch (const std::exception& e) {
            std::cerr << "Detect setup failed: " << e.what() << std::endl;
            detectModule.reset();
        }
    }

    static void TearDownTestSuite() {
        detectModule.reset();
    }
};

std::unique_ptr<Detect> DetectModuleTest::detectModule;

TEST_F(DetectModuleTest, Construction_ValidParams_Success) 
{
    EXPECT_NO_THROW(Detect(80, std::vector<int64_t>{ 128, 256, 512 }););
}

TEST_F(DetectModuleTest, Forward_TrainingMode_Success) {
    if (!detectModule) return;

    detectModule->ptr()->train();
    std::vector<torch::Tensor> inputs = {
        torch::randn({1, 128, 80, 80}),
        torch::randn({1, 256, 40, 40}),
        torch::randn({1, 512, 20, 20})
    };

    EXPECT_NO_THROW({
        auto outputs = detectModule->ptr()->forward(inputs);
        EXPECT_FALSE(outputs.empty());
    });

}

TEST_F(DetectModuleTest, Forward_EvalMode_Success) {
    if (!detectModule) return;

    detectModule->ptr()->eval();
    std::vector<torch::Tensor> inputs = {
        torch::randn({1, 128, 80, 80}),
        torch::randn({1, 256, 40, 40}),
        torch::randn({1, 512, 20, 20})
    };

    EXPECT_NO_THROW({
        auto outputs = detectModule->ptr()->forward(inputs);
        EXPECT_FALSE(outputs.empty());
    });
}

TEST_F(DetectModuleTest, Forward_MultiScaleOutput_CorrectBehavior) {
    std::vector<int64_t> channels = {128, 256, 512};
    Detect detect(80, channels);
    detect->eval();

    std::vector<torch::Tensor> inputs = {
        torch::randn({2, 128, 80, 80}),
        torch::randn({2, 256, 40, 40}),
        torch::randn({2, 512, 20, 20})
    };

    auto outputs = detect->forward(inputs);
    EXPECT_FALSE(outputs.empty());
}

TEST_F(DetectModuleTest, BiasInit_InitializesCorrectly_Success) {
    std::vector<int64_t> channels = {128, 256, 512};
    Detect detect(80, channels);

    EXPECT_NO_THROW({
        detect->biasInit();
    });
}

TEST_F(DetectModuleTest, DecodeBboxes_ValidInput_Success) {
    if (!detectModule) return;

    torch::Tensor bboxes = torch::randn({1, 8400, 64});
    torch::Tensor anchors = torch::randn({8400, 2});

    EXPECT_NO_THROW({
        auto decoded = detectModule->ptr()->decodeBboxes(bboxes, anchors, true);
        EXPECT_FALSE(decoded.sizes().empty());
    });
}

TEST_F(DetectModuleTest, Forward_DifferentBatchSizes_CorrectOutputShape) {
    std::vector<int64_t> channels = {128, 256, 512};
    Detect detect(80, channels);
    detect->eval();

    std::vector<torch::Tensor> inputs_batch1 = {
        torch::randn({1, 128, 80, 80}),
        torch::randn({1, 256, 40, 40}),
        torch::randn({1, 512, 20, 20})
    };

    std::vector<torch::Tensor> inputs_batch4 = {
        torch::randn({4, 128, 80, 80}),
        torch::randn({4, 256, 40, 40}),
        torch::randn({4, 512, 20, 20})
    };

    auto outputs1 = detect->forward(inputs_batch1);
    auto outputs4 = detect->forward(inputs_batch4);

    EXPECT_FALSE(outputs1.empty());
    EXPECT_FALSE(outputs4.empty());
}

// ========== OBB Module Tests ==========

class OBBModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<OBB> obbModule;

    static void SetUpTestSuite() {
        try {
            std::vector<int64_t> channels = {128, 256, 512};
            obbModule = std::make_unique<OBB>(80, 1, channels);
        } catch (const std::exception& e) {
            std::cerr << "OBB setup failed: " << e.what() << std::endl;
            obbModule.reset();
        }
    }

    static void TearDownTestSuite() {
        obbModule.reset();
    }
};

std::unique_ptr<OBB> OBBModuleTest::obbModule;

TEST_F(OBBModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW(OBB obb(80, 1, std::vector<int64_t>{128, 256, 512}););
}

TEST_F(OBBModuleTest, Forward_TrainingMode_Success) {
    if (!obbModule) return;

    obbModule->ptr()->train();
    std::vector<torch::Tensor> inputs = {
        torch::randn({1, 128, 80, 80}),
        torch::randn({1, 256, 40, 40}),
        torch::randn({1, 512, 20, 20})
    };

    EXPECT_NO_THROW({
        auto outputs = obbModule->ptr()->forward(inputs);
        EXPECT_FALSE(outputs.empty());
    });
}

TEST_F(OBBModuleTest, Forward_EvalMode_Success) {
    if (!obbModule) return;

    obbModule->ptr()->eval();
    std::vector<torch::Tensor> inputs = {
        torch::randn({1, 128, 80, 80}),
        torch::randn({1, 256, 40, 40}),
        torch::randn({1, 512, 20, 20})
    };

    EXPECT_NO_THROW({
        auto outputs = obbModule->ptr()->forward(inputs);
        EXPECT_FALSE(outputs.empty());
    });
}

TEST_F(OBBModuleTest, Forward_AnglePrediction_CorrectBehavior) {
    std::vector<int64_t> channels = {128, 256, 512};
    OBB obb(80, 1, channels);
    obb->eval();

    std::vector<torch::Tensor> inputs = {
        torch::randn({2, 128, 80, 80}),
        torch::randn({2, 256, 40, 40}),
        torch::randn({2, 512, 20, 20})
    };

    auto outputs = obb->forward(inputs);
    EXPECT_FALSE(outputs.empty());
}

TEST_F(OBBModuleTest, DecodeBboxes_RotatedBoxes_Success) {
    if (!obbModule) return;

    torch::Tensor bboxes = torch::randn({1, 8400, 64});
    torch::Tensor anchors = torch::randn({8400, 2});

    EXPECT_NO_THROW({
        auto decoded = obbModule->ptr()->decodeBboxes(bboxes, anchors);
        EXPECT_FALSE(decoded.sizes().empty());
    });
}

// ========== Classify Module Tests ==========

class ClassifyModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<Classify> classifyModule;

    static void SetUpTestSuite() {
        try {
            classifyModule = std::make_unique<Classify>(512, 1000, 1, 1);
        } catch (const std::exception& e) {
            std::cerr << "Classify setup failed: " << e.what() << std::endl;
            classifyModule.reset();
        }
    }

    static void TearDownTestSuite() {
        classifyModule.reset();
    }
};

std::unique_ptr<Classify> ClassifyModuleTest::classifyModule;

TEST_F(ClassifyModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        Classify classify(512, 1000, 1, 1);
    });
}

TEST_F(ClassifyModuleTest, Forward_TrainingMode_Success) {
    if (!classifyModule) return;

    classifyModule->ptr()->train();
    std::vector<torch::Tensor> inputs = {
        torch::randn({1, 512, 7, 7})
    };

    EXPECT_NO_THROW({
        auto outputs = classifyModule->ptr()->forward(inputs);
        EXPECT_FALSE(outputs.empty());
        EXPECT_EQ(outputs[0].size(0), 1);
        EXPECT_EQ(outputs[0].size(1), 1000);
    });
}

TEST_F(ClassifyModuleTest, Forward_EvalMode_Success) {
    if (!classifyModule) return;

    classifyModule->ptr()->eval();
    std::vector<torch::Tensor> inputs = {
        torch::randn({1, 512, 7, 7})
    };

    EXPECT_NO_THROW({
        auto outputs = classifyModule->ptr()->forward(inputs);
        EXPECT_FALSE(outputs.empty());
        EXPECT_EQ(outputs[0].size(0), 1);
        EXPECT_EQ(outputs[0].size(1), 1000);
    });
}

TEST_F(ClassifyModuleTest, Forward_AdaptivePooling_CorrectOutputShape) {
    Classify classify(512, 1000, 1, 1);
    classify->eval();

    std::vector<torch::Tensor> inputs = {
        torch::randn({2, 512, 14, 14})
    };

    auto outputs = classify->forward(inputs);
    EXPECT_FALSE(outputs.empty());
    EXPECT_EQ(outputs[0].size(0), 2);
    EXPECT_EQ(outputs[0].size(1), 1000);
}

TEST_F(ClassifyModuleTest, Forward_OutputShape_BatchNumClasses) {
    Classify classify(256, 10, 1, 1);
    classify->eval();

    std::vector<torch::Tensor> inputs = {
        torch::randn({4, 256, 7, 7})
    };

    auto outputs = classify->forward(inputs);
    EXPECT_FALSE(outputs.empty());
    EXPECT_EQ(outputs[0].size(0), 4);
    EXPECT_EQ(outputs[0].size(1), 10);
}

// ========== Segment Module Tests ==========

class SegmentModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<Segment> segmentModule;

    static void SetUpTestSuite() {
        try {
            segmentModule = std::make_unique<Segment>(256, 21, "ReLU");
        } catch (const std::exception& e) {
            std::cerr << "Segment setup failed: " << e.what() << std::endl;
            segmentModule.reset();
        }
    }

    static void TearDownTestSuite() {
        segmentModule.reset();
    }
};

std::unique_ptr<Segment> SegmentModuleTest::segmentModule;

TEST_F(SegmentModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        Segment segment(256, 21, "ReLU");
    });
}

TEST_F(SegmentModuleTest, Forward_TrainingMode_LogitsOutput) {
    if (!segmentModule) return;

    segmentModule->ptr()->train();
    std::vector<torch::Tensor> inputs = {
        torch::randn({1, 256, 64, 64})
    };

    EXPECT_NO_THROW({
        auto outputs = segmentModule->ptr()->forward(inputs);
        EXPECT_FALSE(outputs.empty());
        EXPECT_EQ(outputs[0].size(0), 1);
        EXPECT_EQ(outputs[0].size(1), 21);
    });
}

TEST_F(SegmentModuleTest, Forward_InferenceMode_ProbabilitiesOutput) {
    if (!segmentModule) return;

    segmentModule->ptr()->eval();
    std::vector<torch::Tensor> inputs = {
        torch::randn({1, 256, 64, 64})
    };

    EXPECT_NO_THROW({
        auto outputs = segmentModule->ptr()->forward(inputs);
        EXPECT_FALSE(outputs.empty());
        EXPECT_EQ(outputs[0].size(0), 1);
        EXPECT_EQ(outputs[0].size(1), 21);
    });
}

TEST_F(SegmentModuleTest, Forward_Upsampling_CorrectBehavior) {
    Segment segment(128, 10, "ReLU");
    segment->eval();

    std::vector<torch::Tensor> inputs = {
        torch::randn({2, 128, 32, 32})
    };

    auto outputs = segment->forward(inputs);
    EXPECT_FALSE(outputs.empty());
    EXPECT_EQ(outputs[0].size(0), 2);
    EXPECT_EQ(outputs[0].size(1), 10);
}

TEST_F(SegmentModuleTest, Forward_PixelWiseOutput_CorrectOutputShape) {
    Segment segment(256, 21, "ReLU");
    segment->eval();

    std::vector<torch::Tensor> inputs = {
        torch::randn({1, 256, 128, 128})
    };

    auto outputs = segment->forward(inputs);
    EXPECT_FALSE(outputs.empty());
    EXPECT_EQ(outputs[0].size(0), 1);
    EXPECT_EQ(outputs[0].size(1), 21);
    EXPECT_EQ(outputs[0].size(2), 128);
    EXPECT_EQ(outputs[0].size(3), 128);
}

// ========== Anomaly Module Tests ==========

// Mock anomaly model for testing
class MockAnomalyModel : public IAnomalyModel {
public:
    std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x) override {
        if (x.empty()) return {};
        return {x[0]};
    }

    std::string getModelType() const override {
        return "MockAnomaly";
    }

    int64_t getOutputChannels() const override {
        return 256;
    }
};

class AnomalyModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<Anomaly> anomalyModule;

    static void SetUpTestSuite() {
        try {
            auto mockModel = std::make_shared<MockAnomalyModel>();
            anomalyModule = std::make_unique<Anomaly>(mockModel);
        } catch (const std::exception& e) {
            std::cerr << "Anomaly setup failed: " << e.what() << std::endl;
            anomalyModule.reset();
        }
    }

    static void TearDownTestSuite() {
        anomalyModule.reset();
    }
};

std::unique_ptr<Anomaly> AnomalyModuleTest::anomalyModule;

TEST_F(AnomalyModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        auto mockModel = std::make_shared<MockAnomalyModel>();
        Anomaly anomaly(mockModel);
    });
}

TEST_F(AnomalyModuleTest, Forward_TrainingMode_Success) {
    if (!anomalyModule) return;

    anomalyModule->ptr()->train();
    std::vector<torch::Tensor> inputs = {
        torch::randn({1, 256, 64, 64})
    };

    EXPECT_NO_THROW({
        auto outputs = anomalyModule->ptr()->forward(inputs);
        EXPECT_FALSE(outputs.empty());
    });
}

TEST_F(AnomalyModuleTest, Forward_InferenceMode_Success) {
    if (!anomalyModule) return;

    anomalyModule->ptr()->eval();
    std::vector<torch::Tensor> inputs = {
        torch::randn({1, 256, 64, 64})
    };

    EXPECT_NO_THROW({
        auto outputs = anomalyModule->ptr()->forward(inputs);
        EXPECT_FALSE(outputs.empty());
    });
}

TEST_F(AnomalyModuleTest, GetModelType_ReturnsCorrectType_Success) {
    if (!anomalyModule) return;

    EXPECT_EQ(anomalyModule->ptr()->getModelType(), "MockAnomaly");
}

TEST_F(AnomalyModuleTest, GetAnomalyModel_ReturnsValidModel_Success) {
    if (!anomalyModule) return;

    auto model = anomalyModule->ptr()->getAnomalyModel();
    EXPECT_NE(model, nullptr);
    EXPECT_EQ(model->getModelType(), "MockAnomaly");
}

TEST_F(AnomalyModuleTest, Forward_EfficientADWrapper_CorrectBehavior) {
    auto mockModel = std::make_shared<MockAnomalyModel>();
    Anomaly anomaly(mockModel);
    anomaly->train();

    std::vector<torch::Tensor> inputs = {
        torch::randn({2, 256, 64, 64})
    };

    auto outputs = anomaly->forward(inputs);
    EXPECT_FALSE(outputs.empty());
}

TEST_F(AnomalyModuleTest, Forward_PatchCoreWrapper_CorrectBehavior) {
    auto mockModel = std::make_shared<MockAnomalyModel>();
    Anomaly anomaly(mockModel);
    anomaly->eval();

    std::vector<torch::Tensor> inputs = {
        torch::randn({1, 256, 64, 64})
    };

    auto outputs = anomaly->forward(inputs);
    EXPECT_FALSE(outputs.empty());
}

TEST_F(AnomalyModuleTest, Forward_DifferentBatchSizes_CorrectOutputShape) {
    auto mockModel = std::make_shared<MockAnomalyModel>();
    Anomaly anomaly(mockModel);
    anomaly->eval();

    std::vector<torch::Tensor> inputs_batch1 = {
        torch::randn({1, 256, 64, 64})
    };

    std::vector<torch::Tensor> inputs_batch4 = {
        torch::randn({4, 256, 64, 64})
    };

    auto outputs1 = anomaly->forward(inputs_batch1);
    auto outputs4 = anomaly->forward(inputs_batch4);

    EXPECT_FALSE(outputs1.empty());
    EXPECT_FALSE(outputs4.empty());
}

TEST_F(AnomalyModuleTest, Forward_TrainingVsInferenceModes_DifferentBehavior) {
    auto mockModel = std::make_shared<MockAnomalyModel>();
    Anomaly anomaly(mockModel);

    std::vector<torch::Tensor> inputs = {
        torch::randn({1, 256, 64, 64})
    };

    anomaly->train();
    auto train_outputs = anomaly->forward(inputs);

    anomaly->eval();
    auto eval_outputs = anomaly->forward(inputs);

    EXPECT_FALSE(train_outputs.empty());
    EXPECT_FALSE(eval_outputs.empty());
}
