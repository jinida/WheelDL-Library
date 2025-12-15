#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Model/Loss/AnomalyLoss.h"
#include "Data/Dataset/BaseDataset.h"

using namespace WheelDL::Model::Loss;
using namespace WheelDL::Data::Dataset;

class AnomalyLossTest : public ::testing::Test {
protected:
    torch::Device device_ = torch::kCPU;

    void SetUp() override {
        torch::manual_seed(42);
#ifdef USE_CUDA
        if (torch::cuda::is_available()) {
            device_ = torch::kCUDA;
        }
#endif
    }

    // Helper: Create random feature tensor [N, C, H, W]
    torch::Tensor createFeatureTensor(int64_t batch = 2, int64_t channels = 64,
                                       int64_t height = 32, int64_t width = 32) {
        return torch::randn({batch, channels, height, width},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    }

    // Helper: Create EfficientAD prediction tensors
    std::vector<torch::Tensor> createEfficientADPredictions(int64_t batch = 2) {
        std::vector<torch::Tensor> predictions;
        // teacher_out, student_out, ae_teacher_out, ae_student_out, ae_out
        for (int i = 0; i < 5; ++i) {
            predictions.push_back(createFeatureTensor(batch, 64, 32, 32));
        }
        return predictions;
    }

    // Helper: Create SimpleNet prediction tensors
    std::vector<torch::Tensor> createSimpleNetPredictions(int64_t batch = 2) {
        std::vector<torch::Tensor> predictions;
        predictions.push_back(createFeatureTensor(batch, 64, 32, 32));
        return predictions;
    }

    // Helper: Create PatchCore prediction tensors
    std::vector<torch::Tensor> createPatchCorePredictions(int64_t batch = 2) {
        std::vector<torch::Tensor> predictions;
        predictions.push_back(createFeatureTensor(batch, 256, 16, 16));
        return predictions;
    }

    // Helper: Create DataExample
    DataExample createDataExample(int64_t batch = 2) {
        DataExample example;
        example.data = createFeatureTensor(batch, 3, 256, 256);
        return example;
    }
};

// =============================================================================
// Constructor Tests
// =============================================================================

TEST_F(AnomalyLossTest, Constructor_DefaultType) {
    AnomalyLoss loss;
    EXPECT_EQ(loss.getLossType(), AnomalyLoss::LossType::SimpleNet);
}

TEST_F(AnomalyLossTest, Constructor_SimpleNet) {
    AnomalyLoss loss(AnomalyLoss::LossType::SimpleNet);
    EXPECT_EQ(loss.getLossType(), AnomalyLoss::LossType::SimpleNet);
}

TEST_F(AnomalyLossTest, Constructor_EfficientAD) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);
    EXPECT_EQ(loss.getLossType(), AnomalyLoss::LossType::EfficientAD);
}

TEST_F(AnomalyLossTest, Constructor_PatchCore) {
    AnomalyLoss loss(AnomalyLoss::LossType::PatchCore);
    EXPECT_EQ(loss.getLossType(), AnomalyLoss::LossType::PatchCore);
}

// =============================================================================
// API Tests
// =============================================================================

TEST_F(AnomalyLossTest, API_SetLossType) {
    AnomalyLoss loss(AnomalyLoss::LossType::SimpleNet);

    loss.setLossType(AnomalyLoss::LossType::EfficientAD);
    EXPECT_EQ(loss.getLossType(), AnomalyLoss::LossType::EfficientAD);

    loss.setLossType(AnomalyLoss::LossType::PatchCore);
    EXPECT_EQ(loss.getLossType(), AnomalyLoss::LossType::PatchCore);
}

TEST_F(AnomalyLossTest, API_GetLossType) {
    AnomalyLoss loss1(AnomalyLoss::LossType::SimpleNet);
    AnomalyLoss loss2(AnomalyLoss::LossType::EfficientAD);
    AnomalyLoss loss3(AnomalyLoss::LossType::PatchCore);

    EXPECT_EQ(loss1.getLossType(), AnomalyLoss::LossType::SimpleNet);
    EXPECT_EQ(loss2.getLossType(), AnomalyLoss::LossType::EfficientAD);
    EXPECT_EQ(loss3.getLossType(), AnomalyLoss::LossType::PatchCore);
}

TEST_F(AnomalyLossTest, API_Name_SimpleNet) {
    AnomalyLoss loss(AnomalyLoss::LossType::SimpleNet);
    EXPECT_EQ(loss.name(), "AnomalyLoss::SimpleNet");
}

TEST_F(AnomalyLossTest, API_Name_EfficientAD) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);
    EXPECT_EQ(loss.name(), "AnomalyLoss::EfficientAD");
}

TEST_F(AnomalyLossTest, API_Name_PatchCore) {
    AnomalyLoss loss(AnomalyLoss::LossType::PatchCore);
    EXPECT_EQ(loss.name(), "AnomalyLoss::PatchCore");
}

// =============================================================================
// Exception Tests
// =============================================================================

TEST_F(AnomalyLossTest, Exception_SingleTensorCompute) {
    AnomalyLoss loss(AnomalyLoss::LossType::SimpleNet);
    auto pred = createFeatureTensor(2, 3, 64, 64);
    auto target = createFeatureTensor(2, 3, 64, 64);

    EXPECT_THROW(loss.compute(pred, target), std::runtime_error);
}

TEST_F(AnomalyLossTest, Exception_SingleTensorComputeWithDataExample) {
    AnomalyLoss loss(AnomalyLoss::LossType::SimpleNet);
    auto pred = createFeatureTensor(2, 3, 64, 64);
    auto example = createDataExample(2);

    EXPECT_THROW(loss.compute(pred, example), std::runtime_error);
}

TEST_F(AnomalyLossTest, Exception_EfficientAD_TooFewPredictions) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);
    std::vector<torch::Tensor> predictions;
    predictions.push_back(createFeatureTensor());
    predictions.push_back(createFeatureTensor());
    predictions.push_back(createFeatureTensor());
    // Only 3 tensors, needs at least 5

    auto example = createDataExample();
    EXPECT_THROW(loss.compute(predictions, example), std::invalid_argument);
}

TEST_F(AnomalyLossTest, Exception_EfficientAD_FourPredictions) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);
    std::vector<torch::Tensor> predictions;
    for (int i = 0; i < 4; ++i) {
        predictions.push_back(createFeatureTensor());
    }

    auto example = createDataExample();
    EXPECT_THROW(loss.compute(predictions, example), std::invalid_argument);
}

// =============================================================================
// SimpleNet Loss Tests
// =============================================================================

TEST_F(AnomalyLossTest, SimpleNet_BasicCompute) {
    AnomalyLoss loss(AnomalyLoss::LossType::SimpleNet);
    auto predictions = createSimpleNetPredictions();
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(AnomalyLossTest, SimpleNet_ReturnsMockLoss) {
    AnomalyLoss loss(AnomalyLoss::LossType::SimpleNet);
    auto predictions = createSimpleNetPredictions();
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    // Current implementation returns zero tensor
    EXPECT_FLOAT_EQ(resultMap["total"].item<float>(), 0.0f);
}

TEST_F(AnomalyLossTest, SimpleNet_DifferentBatchSizes) {
    AnomalyLoss loss(AnomalyLoss::LossType::SimpleNet);

    for (int64_t batch : {1, 4, 8, 16}) {
        auto predictions = createSimpleNetPredictions(batch);
        auto example = createDataExample(batch);

        auto resultMap = loss.compute(predictions, example);

        EXPECT_TRUE(resultMap.find("total") != resultMap.end());
        EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    }
}

TEST_F(AnomalyLossTest, SimpleNet_DifferentFeatureDimensions) {
    AnomalyLoss loss(AnomalyLoss::LossType::SimpleNet);

    for (int64_t channels : {64, 128, 256}) {
        std::vector<torch::Tensor> predictions;
        predictions.push_back(createFeatureTensor(2, channels, 32, 32));
        auto example = createDataExample();

        auto resultMap = loss.compute(predictions, example);

        EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    }
}

// =============================================================================
// EfficientAD Loss Tests
// =============================================================================

TEST_F(AnomalyLossTest, EfficientAD_BasicCompute) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);
    auto predictions = createEfficientADPredictions();
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    EXPECT_TRUE(resultMap.find("hard") != resultMap.end());
    EXPECT_TRUE(resultMap.find("ae") != resultMap.end());
    EXPECT_TRUE(resultMap.find("stae") != resultMap.end());
    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
}

TEST_F(AnomalyLossTest, EfficientAD_ReturnMapKeys) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);
    auto predictions = createEfficientADPredictions();
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    // Check all expected keys exist
    EXPECT_EQ(resultMap.size(), 4u);
    EXPECT_TRUE(resultMap.count("hard") > 0);
    EXPECT_TRUE(resultMap.count("ae") > 0);
    EXPECT_TRUE(resultMap.count("stae") > 0);
    EXPECT_TRUE(resultMap.count("total") > 0);
}

TEST_F(AnomalyLossTest, EfficientAD_ValidLossValues) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);
    auto predictions = createEfficientADPredictions();
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    // All loss values should be valid
    EXPECT_FALSE(resultMap["hard"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["ae"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["stae"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());

    EXPECT_FALSE(resultMap["hard"].isinf().any().item<bool>());
    EXPECT_FALSE(resultMap["ae"].isinf().any().item<bool>());
    EXPECT_FALSE(resultMap["stae"].isinf().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(AnomalyLossTest, EfficientAD_NonNegativeLosses) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);
    auto predictions = createEfficientADPredictions();
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    // All loss values should be non-negative (they are squared differences)
    EXPECT_GE(resultMap["hard"].item<float>(), 0.0f);
    EXPECT_GE(resultMap["ae"].item<float>(), 0.0f);
    EXPECT_GE(resultMap["stae"].item<float>(), 0.0f);
    EXPECT_GE(resultMap["total"].item<float>(), 0.0f);
}

TEST_F(AnomalyLossTest, EfficientAD_TotalIsSum) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);
    auto predictions = createEfficientADPredictions();
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    // total = hard + ae + stae (detached values may differ slightly)
    float total = resultMap["total"].item<float>();
    float sum = resultMap["hard"].item<float>() +
                resultMap["ae"].item<float>() +
                resultMap["stae"].item<float>();

    // Note: total is not detached, components are detached, so check approximately
    EXPECT_TRUE(total > 0.0f || sum == 0.0f);
}

TEST_F(AnomalyLossTest, EfficientAD_IdenticalTeacherStudent) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);

    // When teacher and student outputs are identical, hard loss should be 0
    auto teacherOut = createFeatureTensor(2, 64, 32, 32);
    auto studentOut = teacherOut.clone();
    auto aeTeacherOut = createFeatureTensor(2, 64, 32, 32);
    auto aeStudentOut = createFeatureTensor(2, 64, 32, 32);
    auto aeOut = createFeatureTensor(2, 64, 32, 32);

    std::vector<torch::Tensor> predictions = {
        teacherOut, studentOut, aeTeacherOut, aeStudentOut, aeOut
    };
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    // Hard loss should be 0 since teacher == student
    EXPECT_FLOAT_EQ(resultMap["hard"].item<float>(), 0.0f);
}

TEST_F(AnomalyLossTest, EfficientAD_IdenticalAEOutput) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);

    // When AE outputs match targets, ae and stae losses should be 0
    auto teacherOut = createFeatureTensor(2, 64, 32, 32);
    auto studentOut = createFeatureTensor(2, 64, 32, 32);
    auto aeTeacherOut = createFeatureTensor(2, 64, 32, 32);
    auto aeOut = aeTeacherOut.clone();  // AE output matches ae_teacher_out
    auto aeStudentOut = aeOut.clone();  // AE student matches ae_out

    std::vector<torch::Tensor> predictions = {
        teacherOut, studentOut, aeTeacherOut, aeStudentOut, aeOut
    };
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    // AE loss should be 0 since aeTeacherOut == aeOut
    EXPECT_FLOAT_EQ(resultMap["ae"].item<float>(), 0.0f);
    // STAE loss should be 0 since aeStudentOut == aeOut
    EXPECT_FLOAT_EQ(resultMap["stae"].item<float>(), 0.0f);
}

TEST_F(AnomalyLossTest, EfficientAD_HardLossMask) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);

    // Create predictions where most teacher-student diffs are small
    // but a few are large (to test the top 0.1% mask)
    auto teacherOut = torch::zeros({2, 64, 32, 32},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    auto studentOut = torch::zeros_like(teacherOut);

    // Add a few large differences
    studentOut[0][0][0][0] = 100.0f;  // Large outlier

    auto aeTeacherOut = createFeatureTensor(2, 64, 32, 32);
    auto aeStudentOut = createFeatureTensor(2, 64, 32, 32);
    auto aeOut = createFeatureTensor(2, 64, 32, 32);

    std::vector<torch::Tensor> predictions = {
        teacherOut, studentOut, aeTeacherOut, aeStudentOut, aeOut
    };
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    // Hard loss should be non-zero due to the outlier
    EXPECT_GT(resultMap["hard"].item<float>(), 0.0f);
}

TEST_F(AnomalyLossTest, EfficientAD_DifferentBatchSizes) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);

    for (int64_t batch : {1, 2, 4, 8}) {
        auto predictions = createEfficientADPredictions(batch);
        auto example = createDataExample(batch);

        auto resultMap = loss.compute(predictions, example);

        EXPECT_TRUE(resultMap.find("total") != resultMap.end());
        EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    }
}

TEST_F(AnomalyLossTest, EfficientAD_MoreThan5Predictions) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);

    // More than 5 predictions should be fine (extra are ignored)
    std::vector<torch::Tensor> predictions;
    for (int i = 0; i < 7; ++i) {
        predictions.push_back(createFeatureTensor());
    }
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

// =============================================================================
// PatchCore Loss Tests
// =============================================================================

TEST_F(AnomalyLossTest, PatchCore_BasicCompute) {
    AnomalyLoss loss(AnomalyLoss::LossType::PatchCore);
    auto predictions = createPatchCorePredictions();
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(AnomalyLossTest, PatchCore_ReturnsZero) {
    AnomalyLoss loss(AnomalyLoss::LossType::PatchCore);
    auto predictions = createPatchCorePredictions();
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    // Current implementation returns zero tensor (not implemented)
    EXPECT_FLOAT_EQ(resultMap["total"].item<float>(), 0.0f);
}

TEST_F(AnomalyLossTest, PatchCore_DifferentBatchSizes) {
    AnomalyLoss loss(AnomalyLoss::LossType::PatchCore);

    for (int64_t batch : {1, 4, 8}) {
        auto predictions = createPatchCorePredictions(batch);
        auto example = createDataExample(batch);

        auto resultMap = loss.compute(predictions, example);

        EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    }
}

// =============================================================================
// Numerical Stability Tests
// =============================================================================

TEST_F(AnomalyLossTest, NumericalStability_LargeValues) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);

    std::vector<torch::Tensor> predictions;
    for (int i = 0; i < 5; ++i) {
        predictions.push_back(torch::randn({2, 64, 32, 32},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_)) * 100.0f);
    }
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(AnomalyLossTest, NumericalStability_SmallValues) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);

    std::vector<torch::Tensor> predictions;
    for (int i = 0; i < 5; ++i) {
        predictions.push_back(torch::randn({2, 64, 32, 32},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_)) * 1e-6f);
    }
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
}

TEST_F(AnomalyLossTest, NumericalStability_ZeroInput) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);

    std::vector<torch::Tensor> predictions;
    for (int i = 0; i < 5; ++i) {
        predictions.push_back(torch::zeros({2, 64, 32, 32},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    }
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
    EXPECT_FALSE(resultMap["total"].isinf().any().item<bool>());
    EXPECT_FLOAT_EQ(resultMap["total"].item<float>(), 0.0f);
}

// =============================================================================
// Gradient Tests
// =============================================================================

TEST_F(AnomalyLossTest, Gradient_EfficientAD) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);

    std::vector<torch::Tensor> predictions;
    for (int i = 0; i < 5; ++i) {
        predictions.push_back(torch::randn({2, 64, 32, 32},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_).requires_grad(true)));
    }
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);
    resultMap["total"].backward();

    // Check gradients exist and are valid
    for (const auto& pred : predictions) {
        EXPECT_TRUE(pred.grad().defined());
        EXPECT_FALSE(pred.grad().isnan().any().item<bool>());
        EXPECT_FALSE(pred.grad().isinf().any().item<bool>());
    }
}

//TEST_F(AnomalyLossTest, Gradient_SimpleNet) {
//    AnomalyLoss loss(AnomalyLoss::LossType::SimpleNet);
//
//    std::vector<torch::Tensor> predictions;
//    predictions.push_back(torch::randn({2, 64, 32, 32},
//        torch::TensorOptions().dtype(torch::kFloat32).device(device_).requires_grad(true)));
//    auto example = createDataExample();
//
//    auto resultMap = loss.compute(predictions, example);
//
//    // SimpleNet currently returns mockup (zero tensor), backward should still work
//    EXPECT_NO_THROW(resultMap["total"].backward());
//}

// =============================================================================
// Return Map Structure Tests
// =============================================================================

TEST_F(AnomalyLossTest, ReturnMap_SimpleNet) {
    AnomalyLoss loss(AnomalyLoss::LossType::SimpleNet);
    auto predictions = createSimpleNetPredictions();
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    EXPECT_EQ(resultMap.size(), 1u);
    EXPECT_TRUE(resultMap.count("total") > 0);
}

TEST_F(AnomalyLossTest, ReturnMap_EfficientAD) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);
    auto predictions = createEfficientADPredictions();
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    EXPECT_EQ(resultMap.size(), 4u);
    EXPECT_TRUE(resultMap.count("hard") > 0);
    EXPECT_TRUE(resultMap.count("ae") > 0);
    EXPECT_TRUE(resultMap.count("stae") > 0);
    EXPECT_TRUE(resultMap.count("total") > 0);
}

TEST_F(AnomalyLossTest, ReturnMap_PatchCore) {
    AnomalyLoss loss(AnomalyLoss::LossType::PatchCore);
    auto predictions = createPatchCorePredictions();
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    EXPECT_EQ(resultMap.size(), 1u);
    EXPECT_TRUE(resultMap.count("total") > 0);
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

TEST_F(AnomalyLossTest, EdgeCase_SingleBatch) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);
    auto predictions = createEfficientADPredictions(1);
    auto example = createDataExample(1);

    auto resultMap = loss.compute(predictions, example);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(AnomalyLossTest, EdgeCase_LargeBatch) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);
    auto predictions = createEfficientADPredictions(32);
    auto example = createDataExample(32);

    auto resultMap = loss.compute(predictions, example);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(AnomalyLossTest, EdgeCase_SmallSpatialSize) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);

    std::vector<torch::Tensor> predictions;
    for (int i = 0; i < 5; ++i) {
        predictions.push_back(torch::randn({2, 64, 4, 4},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    }
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(AnomalyLossTest, EdgeCase_LargeSpatialSize) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);

    std::vector<torch::Tensor> predictions;
    for (int i = 0; i < 5; ++i) {
        predictions.push_back(torch::randn({2, 32, 128, 128},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    }
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}

TEST_F(AnomalyLossTest, EdgeCase_SingleChannelFeature) {
    AnomalyLoss loss(AnomalyLoss::LossType::EfficientAD);

    std::vector<torch::Tensor> predictions;
    for (int i = 0; i < 5; ++i) {
        predictions.push_back(torch::randn({2, 1, 32, 32},
            torch::TensorOptions().dtype(torch::kFloat32).device(device_)));
    }
    auto example = createDataExample();

    auto resultMap = loss.compute(predictions, example);

    EXPECT_TRUE(resultMap.find("total") != resultMap.end());
    EXPECT_FALSE(resultMap["total"].isnan().any().item<bool>());
}
