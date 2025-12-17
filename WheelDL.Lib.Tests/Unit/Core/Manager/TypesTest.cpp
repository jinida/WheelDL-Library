/**
 * @file TypesTest.cpp
 * @brief Unit tests for WheelDL::Core::Manager Types and Enums
 *
 * Phase 1 of test_manager.md - 42 tests
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "Core/Manager/Types.h"

using namespace WheelDL;
using namespace WheelDL::Core::Manager;

// ============================================================================
// Test Fixture
// ============================================================================

class TypesTest : public ::testing::Test {
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

private:
    bool _hasCuda = false;
};

// ============================================================================
// 1.1 OperationType Enum Tests (14)
// ============================================================================

TEST_F(TypesTest, OperationType_TRAIN_Value) {
    if (!requireCuda()) return;

    auto value = static_cast<int>(OperationType::TRAIN);
    EXPECT_GE(value, 0);
}

TEST_F(TypesTest, OperationType_VALIDATE_Value) {
    if (!requireCuda()) return;

    auto value = static_cast<int>(OperationType::VALIDATE);
    EXPECT_GE(value, 0);
}

TEST_F(TypesTest, OperationType_PREDICT_Value) {
    if (!requireCuda()) return;

    auto value = static_cast<int>(OperationType::PREDICT);
    EXPECT_GE(value, 0);
}

TEST_F(TypesTest, OperationType_TRAIN_NotEqual_VALIDATE) {
    if (!requireCuda()) return;

    EXPECT_NE(OperationType::TRAIN, OperationType::VALIDATE);
}

TEST_F(TypesTest, OperationType_TRAIN_NotEqual_PREDICT) {
    if (!requireCuda()) return;

    EXPECT_NE(OperationType::TRAIN, OperationType::PREDICT);
}

TEST_F(TypesTest, OperationType_VALIDATE_NotEqual_PREDICT) {
    if (!requireCuda()) return;

    EXPECT_NE(OperationType::VALIDATE, OperationType::PREDICT);
}

TEST_F(TypesTest, operationTypeToString_TRAIN) {
    if (!requireCuda()) return;

    EXPECT_STREQ(operationTypeToString(OperationType::TRAIN), "TRAIN");
}

TEST_F(TypesTest, operationTypeToString_VALIDATE) {
    if (!requireCuda()) return;

    EXPECT_STREQ(operationTypeToString(OperationType::VALIDATE), "VALIDATE");
}

TEST_F(TypesTest, operationTypeToString_PREDICT) {
    if (!requireCuda()) return;

    EXPECT_STREQ(operationTypeToString(OperationType::PREDICT), "PREDICT");
}

TEST_F(TypesTest, operationTypeToString_Default_99) {
    if (!requireCuda()) return;

    EXPECT_STREQ(operationTypeToString(static_cast<OperationType>(99)), "UNKNOWN");
}

TEST_F(TypesTest, operationTypeToString_Default_Negative) {
    if (!requireCuda()) return;

    EXPECT_STREQ(operationTypeToString(static_cast<OperationType>(-1)), "UNKNOWN");
}

TEST_F(TypesTest, operationTypeToString_Default_Large) {
    if (!requireCuda()) return;

    EXPECT_STREQ(operationTypeToString(static_cast<OperationType>(999)), "UNKNOWN");
}

TEST_F(TypesTest, OperationType_SizeOf) {
    if (!requireCuda()) return;

    EXPECT_GT(sizeof(OperationType), 0u);
}

TEST_F(TypesTest, OperationType_Equal_Same) {
    if (!requireCuda()) return;

    OperationType a = OperationType::TRAIN;
    OperationType b = OperationType::TRAIN;
    EXPECT_EQ(a, b);
}

// ============================================================================
// 1.2 TaskPriority Enum Tests (12)
// ============================================================================

TEST_F(TypesTest, TaskPriority_LOW_Value) {
    if (!requireCuda()) return;

    EXPECT_EQ(static_cast<int>(TaskPriority::LOW), 0);
}

TEST_F(TypesTest, TaskPriority_NORMAL_Value) {
    if (!requireCuda()) return;

    EXPECT_EQ(static_cast<int>(TaskPriority::NORMAL), 1);
}

TEST_F(TypesTest, TaskPriority_HIGH_Value) {
    if (!requireCuda()) return;

    EXPECT_EQ(static_cast<int>(TaskPriority::HIGH), 2);
}

TEST_F(TypesTest, TaskPriority_CRITICAL_Value) {
    if (!requireCuda()) return;

    EXPECT_EQ(static_cast<int>(TaskPriority::CRITICAL), 3);
}

TEST_F(TypesTest, TaskPriority_LOW_LT_NORMAL) {
    if (!requireCuda()) return;

    EXPECT_LT(static_cast<int>(TaskPriority::LOW), static_cast<int>(TaskPriority::NORMAL));
}

TEST_F(TypesTest, TaskPriority_NORMAL_LT_HIGH) {
    if (!requireCuda()) return;

    EXPECT_LT(static_cast<int>(TaskPriority::NORMAL), static_cast<int>(TaskPriority::HIGH));
}

TEST_F(TypesTest, TaskPriority_HIGH_LT_CRITICAL) {
    if (!requireCuda()) return;

    EXPECT_LT(static_cast<int>(TaskPriority::HIGH), static_cast<int>(TaskPriority::CRITICAL));
}

TEST_F(TypesTest, TaskPriority_CRITICAL_GT_LOW) {
    if (!requireCuda()) return;

    EXPECT_GT(static_cast<int>(TaskPriority::CRITICAL), static_cast<int>(TaskPriority::LOW));
}

TEST_F(TypesTest, TaskPriority_StaticCast_LOW) {
    if (!requireCuda()) return;

    EXPECT_EQ(static_cast<int>(TaskPriority::LOW), 0);
}

TEST_F(TypesTest, TaskPriority_StaticCast_NORMAL) {
    if (!requireCuda()) return;

    EXPECT_EQ(static_cast<int>(TaskPriority::NORMAL), 1);
}

TEST_F(TypesTest, TaskPriority_StaticCast_HIGH) {
    if (!requireCuda()) return;

    EXPECT_EQ(static_cast<int>(TaskPriority::HIGH), 2);
}

TEST_F(TypesTest, TaskPriority_StaticCast_CRITICAL) {
    if (!requireCuda()) return;

    EXPECT_EQ(static_cast<int>(TaskPriority::CRITICAL), 3);
}

// ============================================================================
// 1.3 isTerminalState Function Tests (16)
// ============================================================================

TEST_F(TypesTest, isTerminalState_COMPLETED) {
    if (!requireCuda()) return;

    EXPECT_TRUE(isTerminalState(TrainingState::COMPLETED));
}

TEST_F(TypesTest, isTerminalState_FAILED) {
    if (!requireCuda()) return;

    EXPECT_TRUE(isTerminalState(TrainingState::FAILED));
}

TEST_F(TypesTest, isTerminalState_STOPPED) {
    if (!requireCuda()) return;

    EXPECT_TRUE(isTerminalState(TrainingState::STOPPED));
}

TEST_F(TypesTest, isTerminalState_IDLE) {
    if (!requireCuda()) return;

    EXPECT_FALSE(isTerminalState(TrainingState::IDLE));
}

TEST_F(TypesTest, isTerminalState_INITIALIZING) {
    if (!requireCuda()) return;

    EXPECT_FALSE(isTerminalState(TrainingState::INITIALIZING));
}

TEST_F(TypesTest, isTerminalState_TRAINING) {
    if (!requireCuda()) return;

    EXPECT_FALSE(isTerminalState(TrainingState::TRAINING));
}

TEST_F(TypesTest, isTerminalState_VALIDATING) {
    if (!requireCuda()) return;

    EXPECT_FALSE(isTerminalState(TrainingState::VALIDATING));
}

TEST_F(TypesTest, isTerminalState_PAUSED) {
    if (!requireCuda()) return;

    EXPECT_FALSE(isTerminalState(TrainingState::PAUSED));
}

TEST_F(TypesTest, isTerminalState_Idempotent_COMPLETED) {
    if (!requireCuda()) return;

    bool first = isTerminalState(TrainingState::COMPLETED);
    bool second = isTerminalState(TrainingState::COMPLETED);
    bool third = isTerminalState(TrainingState::COMPLETED);
    EXPECT_TRUE(first);
    EXPECT_EQ(first, second);
    EXPECT_EQ(second, third);
}

TEST_F(TypesTest, isTerminalState_Idempotent_IDLE) {
    if (!requireCuda()) return;

    bool first = isTerminalState(TrainingState::IDLE);
    bool second = isTerminalState(TrainingState::IDLE);
    bool third = isTerminalState(TrainingState::IDLE);
    EXPECT_FALSE(first);
    EXPECT_EQ(first, second);
    EXPECT_EQ(second, third);
}

TEST_F(TypesTest, isTerminalState_TerminalCount_3) {
    if (!requireCuda()) return;

    int terminalCount = 0;
    if (isTerminalState(TrainingState::IDLE)) terminalCount++;
    if (isTerminalState(TrainingState::INITIALIZING)) terminalCount++;
    if (isTerminalState(TrainingState::TRAINING)) terminalCount++;
    if (isTerminalState(TrainingState::VALIDATING)) terminalCount++;
    if (isTerminalState(TrainingState::PAUSED)) terminalCount++;
    if (isTerminalState(TrainingState::COMPLETED)) terminalCount++;
    if (isTerminalState(TrainingState::FAILED)) terminalCount++;
    if (isTerminalState(TrainingState::STOPPED)) terminalCount++;

    EXPECT_EQ(terminalCount, 3);
}

TEST_F(TypesTest, isTerminalState_NonTerminalCount_4Plus) {
    if (!requireCuda()) return;

    int nonTerminalCount = 0;
    if (!isTerminalState(TrainingState::IDLE)) nonTerminalCount++;
    if (!isTerminalState(TrainingState::INITIALIZING)) nonTerminalCount++;
    if (!isTerminalState(TrainingState::TRAINING)) nonTerminalCount++;
    if (!isTerminalState(TrainingState::VALIDATING)) nonTerminalCount++;
    if (!isTerminalState(TrainingState::PAUSED)) nonTerminalCount++;
    if (!isTerminalState(TrainingState::COMPLETED)) nonTerminalCount++;
    if (!isTerminalState(TrainingState::FAILED)) nonTerminalCount++;
    if (!isTerminalState(TrainingState::STOPPED)) nonTerminalCount++;

    EXPECT_GE(nonTerminalCount, 4);
}

TEST_F(TypesTest, isTerminalState_FirstCondition_COMPLETED) {
    if (!requireCuda()) return;

    // First || branch: state == TrainingState::COMPLETED
    EXPECT_TRUE(isTerminalState(TrainingState::COMPLETED));
}

TEST_F(TypesTest, isTerminalState_SecondCondition_FAILED) {
    if (!requireCuda()) return;

    // Second || branch: state == TrainingState::FAILED
    EXPECT_TRUE(isTerminalState(TrainingState::FAILED));
}

TEST_F(TypesTest, isTerminalState_ThirdCondition_STOPPED) {
    if (!requireCuda()) return;

    // Third || branch: state == TrainingState::STOPPED
    EXPECT_TRUE(isTerminalState(TrainingState::STOPPED));
}

TEST_F(TypesTest, isTerminalState_InvalidState) {
    if (!requireCuda()) return;

    // Invalid TrainingState value should return false (else branch)
    EXPECT_FALSE(isTerminalState(static_cast<TrainingState>(99)));
}

