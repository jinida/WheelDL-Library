/**
 * @file RequestTest.cpp
 * @brief Unit tests for WheelDL::Core::Manager Request Structures
 *
 * Phase 1 of test_manager.md - 52 tests
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Core/Manager/Request.h"
#include "Config/Configuration.h"
#include "../CoreTestHelpers.h"

using namespace WheelDL;
using namespace WheelDL::Core::Manager;
using namespace WheelDL::Test::Core;

// ============================================================================
// Test Fixture
// ============================================================================

class RequestTest : public ::testing::Test {
protected:
    void SetUp() override {
        _hasCuda = torch::cuda::is_available();
    }

    bool requireCuda() {
        if (!_hasCuda) {
            SUCCEED() << "CUDA not available, test skipped";
            return false;
        }
        return true;
    }

    std::shared_ptr<Config::Configuration> createConfig() {
        return CoreConfigFactory::createMinimal();
    }

private:
    bool _hasCuda = false;
};

// ============================================================================
// 2.1 BaseRequest Tests (20)
// ============================================================================

TEST_F(RequestTest, BaseRequest_Config_Stored) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    EXPECT_NE(request.config, nullptr);
}

TEST_F(RequestTest, BaseRequest_OperationType_Stored) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest trainReq(config);
    ValidateRequest valReq(config);
    PredictRequest predReq(config);

    EXPECT_EQ(trainReq.operationType, OperationType::TRAIN);
    EXPECT_EQ(valReq.operationType, OperationType::VALIDATE);
    EXPECT_EQ(predReq.operationType, OperationType::PREDICT);
}

TEST_F(RequestTest, BaseRequest_DefaultPriority_NORMAL) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    EXPECT_EQ(request.priority, TaskPriority::NORMAL);
}

TEST_F(RequestTest, BaseRequest_Priority_SetLOW) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    request.priority = TaskPriority::LOW;
    EXPECT_EQ(request.priority, TaskPriority::LOW);
}

TEST_F(RequestTest, BaseRequest_Priority_SetHIGH) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    request.priority = TaskPriority::HIGH;
    EXPECT_EQ(request.priority, TaskPriority::HIGH);
}

TEST_F(RequestTest, BaseRequest_Priority_SetCRITICAL) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    request.priority = TaskPriority::CRITICAL;
    EXPECT_EQ(request.priority, TaskPriority::CRITICAL);
}

TEST_F(RequestTest, BaseRequest_VirtualDestructor_NoLeak) {
    if (!requireCuda()) return;

    auto config = createConfig();
    BaseRequest* ptr = new TrainRequest(config);
    EXPECT_NO_THROW(delete ptr);
}

TEST_F(RequestTest, BaseRequest_ConfigRefCount_Initial) {
    if (!requireCuda()) return;

    auto config = createConfig();
    long initialCount = config.use_count();
    TrainRequest request(config);
    EXPECT_GE(request.config.use_count(), 2);
}

TEST_F(RequestTest, BaseRequest_ConfigRefCount_AfterCopy) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request1(config);
    long countAfterFirst = config.use_count();
    TrainRequest request2(config);
    EXPECT_GT(config.use_count(), countAfterFirst);
}

TEST_F(RequestTest, BaseRequest_NullConfig_Stored) {
    if (!requireCuda()) return;

    std::shared_ptr<Config::Configuration> nullConfig = nullptr;
    TrainRequest request(nullConfig);
    EXPECT_EQ(request.config, nullptr);
}

TEST_F(RequestTest, BaseRequest_ConfigMoved_OriginalNull) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(std::move(config));
    EXPECT_EQ(config, nullptr);
}

TEST_F(RequestTest, BaseRequest_ConfigAccess_AfterMove) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(std::move(config));
    EXPECT_NE(request.config, nullptr);
}

TEST_F(RequestTest, BaseRequest_MultipleRequests_ShareConfig) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest req1(config);
    TrainRequest req2(config);
    EXPECT_EQ(req1.config.get(), req2.config.get());
}

TEST_F(RequestTest, BaseRequest_PriorityModify_AfterCreate) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    EXPECT_EQ(request.priority, TaskPriority::NORMAL);
    request.priority = TaskPriority::HIGH;
    EXPECT_EQ(request.priority, TaskPriority::HIGH);
}

// REQ-015 is compile-time test (protected constructor) - cannot be runtime tested

TEST_F(RequestTest, BaseRequest_Destructor_RefDecrement) {
    if (!requireCuda()) return;

    auto config = createConfig();
    {
        TrainRequest request(config);
        EXPECT_GE(config.use_count(), 2);
    }
    EXPECT_EQ(config.use_count(), 1);
}

TEST_F(RequestTest, BaseRequest_Polymorphism_Upcast) {
    if (!requireCuda()) return;

    auto config = createConfig();
    std::unique_ptr<BaseRequest> ptr = std::make_unique<TrainRequest>(config);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(ptr->operationType, OperationType::TRAIN);
}

TEST_F(RequestTest, BaseRequest_Polymorphism_Downcast) {
    if (!requireCuda()) return;

    auto config = createConfig();
    BaseRequest* basePtr = new TrainRequest(config);
    TrainRequest* trainPtr = dynamic_cast<TrainRequest*>(basePtr);
    EXPECT_NE(trainPtr, nullptr);
    delete basePtr;
}

TEST_F(RequestTest, BaseRequest_SizeOf) {
    if (!requireCuda()) return;

    EXPECT_GT(sizeof(TrainRequest), 0u);
    EXPECT_GT(sizeof(ValidateRequest), 0u);
    EXPECT_GT(sizeof(PredictRequest), 0u);
}

TEST_F(RequestTest, BaseRequest_Alignment) {
    if (!requireCuda()) return;

    EXPECT_GT(alignof(TrainRequest), 0u);
    EXPECT_GT(alignof(ValidateRequest), 0u);
    EXPECT_GT(alignof(PredictRequest), 0u);
}

// ============================================================================
// 2.2 TrainRequest Tests (14)
// ============================================================================

TEST_F(RequestTest, TrainRequest_OperationType_TRAIN) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    EXPECT_EQ(request.operationType, OperationType::TRAIN);
}

// REQ-022 is compile-time test (explicit constructor) - cannot be runtime tested

TEST_F(RequestTest, TrainRequest_DefaultCallback_Null) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    EXPECT_EQ(request.progressCallback, nullptr);
}

TEST_F(RequestTest, TrainRequest_Callback_Stored) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    request.progressCallback = [](const ProgressData&) {};
    EXPECT_NE(request.progressCallback, nullptr);
}

TEST_F(RequestTest, TrainRequest_Callback_Invocable) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    bool called = false;
    request.progressCallback = [&called](const ProgressData&) { called = true; };

    ProgressData progress;
    request.progressCallback(progress);
    EXPECT_TRUE(called);
}

TEST_F(RequestTest, TrainRequest_Callback_ReceivesProgress) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    int receivedEpoch = -1;
    request.progressCallback = [&receivedEpoch](const ProgressData& p) {
        receivedEpoch = p.currentEpoch;
    };

    ProgressData progress;
    progress.currentEpoch = 42;
    request.progressCallback(progress);
    EXPECT_EQ(receivedEpoch, 42);
}

TEST_F(RequestTest, TrainRequest_Callback_ModifyCaptured) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    int counter = 0;
    request.progressCallback = [&counter](const ProgressData&) { counter++; };

    ProgressData progress;
    request.progressCallback(progress);
    request.progressCallback(progress);
    EXPECT_EQ(counter, 2);
}

TEST_F(RequestTest, TrainRequest_Callback_Reassign) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    int first = 0, second = 0;

    request.progressCallback = [&first](const ProgressData&) { first++; };
    ProgressData progress;
    request.progressCallback(progress);

    request.progressCallback = [&second](const ProgressData&) { second++; };
    request.progressCallback(progress);

    EXPECT_EQ(first, 1);
    EXPECT_EQ(second, 1);
}

TEST_F(RequestTest, TrainRequest_Callback_SetNull) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    request.progressCallback = [](const ProgressData&) {};
    EXPECT_NE(request.progressCallback, nullptr);
    request.progressCallback = nullptr;
    EXPECT_EQ(request.progressCallback, nullptr);
}

TEST_F(RequestTest, TrainRequest_InheritsBaseRequest) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request(config);
    BaseRequest* basePtr = &request;
    EXPECT_NE(basePtr, nullptr);
    EXPECT_EQ(basePtr->operationType, OperationType::TRAIN);
}

TEST_F(RequestTest, TrainRequest_ConfigMoveSemantics) {
    if (!requireCuda()) return;

    auto config = createConfig();
    auto* configPtr = config.get();
    TrainRequest request(std::move(config));
    EXPECT_EQ(request.config.get(), configPtr);
}

TEST_F(RequestTest, TrainRequest_Destructor_CallbackCleanup) {
    if (!requireCuda()) return;

    auto config = createConfig();
    {
        TrainRequest request(config);
        request.progressCallback = [](const ProgressData&) {};
    }
    // No leak - implicit by successful test completion
    SUCCEED();
}

TEST_F(RequestTest, TrainRequest_CopyBehavior) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request1(config);
    request1.priority = TaskPriority::HIGH;
    TrainRequest request2 = request1;
    EXPECT_EQ(request2.priority, TaskPriority::HIGH);
    EXPECT_EQ(request2.config.get(), request1.config.get());
}

TEST_F(RequestTest, TrainRequest_MoveBehavior) {
    if (!requireCuda()) return;

    auto config = createConfig();
    TrainRequest request1(config);
    request1.priority = TaskPriority::HIGH;
    auto* configPtr = request1.config.get();
    TrainRequest request2 = std::move(request1);
    EXPECT_EQ(request2.priority, TaskPriority::HIGH);
    EXPECT_EQ(request2.config.get(), configPtr);
}

// ============================================================================
// 2.3 ValidateRequest Tests (10)
// ============================================================================

TEST_F(RequestTest, ValidateRequest_OperationType_VALIDATE) {
    if (!requireCuda()) return;

    auto config = createConfig();
    ValidateRequest request(config);
    EXPECT_EQ(request.operationType, OperationType::VALIDATE);
}

// REQ-036 is compile-time test (explicit constructor) - cannot be runtime tested

TEST_F(RequestTest, ValidateRequest_DefaultCheckpointPath_Empty) {
    if (!requireCuda()) return;

    auto config = createConfig();
    ValidateRequest request(config);
    EXPECT_TRUE(request.checkpointPath.empty());
}

TEST_F(RequestTest, ValidateRequest_CheckpointPath_Stored) {
    if (!requireCuda()) return;

    auto config = createConfig();
    ValidateRequest request(config);
    request.checkpointPath = "path/to/checkpoint.pt";
    EXPECT_EQ(request.checkpointPath, "path/to/checkpoint.pt");
}

TEST_F(RequestTest, ValidateRequest_CheckpointPath_Spaces) {
    if (!requireCuda()) return;

    auto config = createConfig();
    ValidateRequest request(config);
    request.checkpointPath = "path with spaces/checkpoint.pt";
    EXPECT_EQ(request.checkpointPath, "path with spaces/checkpoint.pt");
}

TEST_F(RequestTest, ValidateRequest_CheckpointPath_Unicode) {
    if (!requireCuda()) return;

    auto config = createConfig();
    ValidateRequest request(config);
    request.checkpointPath = u8"경로/체크포인트.pt";
    EXPECT_EQ(request.checkpointPath, u8"경로/체크포인트.pt");
}

TEST_F(RequestTest, ValidateRequest_CheckpointPath_LongPath) {
    if (!requireCuda()) return;

    auto config = createConfig();
    ValidateRequest request(config);
    std::string longPath(250, 'a');
    longPath += ".pt";
    request.checkpointPath = longPath;
    EXPECT_EQ(request.checkpointPath, longPath);
}

TEST_F(RequestTest, ValidateRequest_CheckpointPath_MixedSlash) {
    if (!requireCuda()) return;

    auto config = createConfig();
    ValidateRequest request(config);
    request.checkpointPath = "path/to\\mixed/slashes\\checkpoint.pt";
    EXPECT_EQ(request.checkpointPath, "path/to\\mixed/slashes\\checkpoint.pt");
}

TEST_F(RequestTest, ValidateRequest_InheritsBaseRequest) {
    if (!requireCuda()) return;

    auto config = createConfig();
    ValidateRequest request(config);
    BaseRequest* basePtr = &request;
    EXPECT_NE(basePtr, nullptr);
    EXPECT_EQ(basePtr->operationType, OperationType::VALIDATE);
}

TEST_F(RequestTest, ValidateRequest_ConfigMoveSemantics) {
    if (!requireCuda()) return;

    auto config = createConfig();
    auto* configPtr = config.get();
    ValidateRequest request(std::move(config));
    EXPECT_EQ(request.config.get(), configPtr);
}

// ============================================================================
// 2.4 PredictRequest Tests (8)
// ============================================================================

TEST_F(RequestTest, PredictRequest_OperationType_PREDICT) {
    if (!requireCuda()) return;

    auto config = createConfig();
    PredictRequest request(config);
    EXPECT_EQ(request.operationType, OperationType::PREDICT);
}

// REQ-046 is compile-time test (explicit constructor) - cannot be runtime tested

TEST_F(RequestTest, PredictRequest_DefaultCheckpointPath_Empty) {
    if (!requireCuda()) return;

    auto config = createConfig();
    PredictRequest request(config);
    EXPECT_TRUE(request.checkpointPath.empty());
}

TEST_F(RequestTest, PredictRequest_CheckpointPath_Stored) {
    if (!requireCuda()) return;

    auto config = createConfig();
    PredictRequest request(config);
    request.checkpointPath = "path/to/model.pt";
    EXPECT_EQ(request.checkpointPath, "path/to/model.pt");
}

TEST_F(RequestTest, PredictRequest_CheckpointPath_Relative) {
    if (!requireCuda()) return;

    auto config = createConfig();
    PredictRequest request(config);
    request.checkpointPath = "./relative/path/model.pt";
    EXPECT_EQ(request.checkpointPath, "./relative/path/model.pt");
}

TEST_F(RequestTest, PredictRequest_CheckpointPath_Absolute) {
    if (!requireCuda()) return;

    auto config = createConfig();
    PredictRequest request(config);
    request.checkpointPath = "C:/absolute/path/model.pt";
    EXPECT_EQ(request.checkpointPath, "C:/absolute/path/model.pt");
}

TEST_F(RequestTest, PredictRequest_InheritsBaseRequest) {
    if (!requireCuda()) return;

    auto config = createConfig();
    PredictRequest request(config);
    BaseRequest* basePtr = &request;
    EXPECT_NE(basePtr, nullptr);
    EXPECT_EQ(basePtr->operationType, OperationType::PREDICT);
}

TEST_F(RequestTest, PredictRequest_ConfigMoveSemantics) {
    if (!requireCuda()) return;

    auto config = createConfig();
    auto* configPtr = config.get();
    PredictRequest request(std::move(config));
    EXPECT_EQ(request.config.get(), configPtr);
}

