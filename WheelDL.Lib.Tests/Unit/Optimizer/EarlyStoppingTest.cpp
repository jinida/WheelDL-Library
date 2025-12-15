#include "pch.h"
#include <gtest/gtest.h>
#include "Optimizer/EarlyStopping/EarlyStopping.h"
#include "Utils/Error/WheelLibException.h"
#include "Utils/Error/ErrorCodes.h"
#include <limits>
#include <cmath>

using namespace WheelDL::Optimizer::EarlyStopping;
using namespace WheelDL::Utils;

// =============================================================================
// Test Fixture
// =============================================================================

class EarlyStoppingTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// =============================================================================
// 3.2.1 Constructor Tests (ES-001 ~ ES-009)
// =============================================================================

// ES-001: Ctor_DefaultPatience
TEST_F(EarlyStoppingTest, Ctor_DefaultPatience) {
    EarlyStopping es;
    EXPECT_EQ(50, es.getPatience());
}

// ES-002: Ctor_CustomPatience
TEST_F(EarlyStoppingTest, Ctor_CustomPatience) {
    EarlyStopping es(100);
    EXPECT_EQ(100, es.getPatience());
}

// ES-003: Ctor_PatOne
TEST_F(EarlyStoppingTest, Ctor_PatOne) {
    EarlyStopping es(1);
    EXPECT_EQ(1, es.getPatience());
}

// ES-004: Ctor_ZeroPatience
TEST_F(EarlyStoppingTest, Ctor_ZeroPatience) {
    EXPECT_THROW({
        EarlyStopping es(0);
    }, ConfigurationException);
}

// ES-005: Ctor_NegativePatience
TEST_F(EarlyStoppingTest, Ctor_NegativePatience) {
    EXPECT_THROW({
        EarlyStopping es(-1);
    }, ConfigurationException);
}

// ES-006: Ctor_NegativeLarge
TEST_F(EarlyStoppingTest, Ctor_NegativeLarge) {
    EXPECT_THROW({
        EarlyStopping es(-100);
    }, ConfigurationException);
}

// ES-007: Ctor_InitialBestFitness
TEST_F(EarlyStoppingTest, Ctor_InitialBestFitness) {
    EarlyStopping es(10);
    EXPECT_EQ(-std::numeric_limits<float>::infinity(), es.getBestFitness());
}

// ES-008: Ctor_InitialCounter
TEST_F(EarlyStoppingTest, Ctor_InitialCounter) {
    EarlyStopping es(10);
    EXPECT_EQ(0, es.getCounter());
}

// ES-009: Ctor_InitialMayStopNext
TEST_F(EarlyStoppingTest, Ctor_InitialMayStopNext) {
    EarlyStopping es(10);
    EXPECT_FALSE(es.mayStopNext());
}

// =============================================================================
// 3.2.2 shouldStop() - Improvement Cases (ES-010 ~ ES-017)
// =============================================================================

// ES-010: ShouldStop_FirstCall
TEST_F(EarlyStoppingTest, ShouldStop_FirstCall) {
    EarlyStopping es(10);
    bool result = es.shouldStop(0.5f);
    EXPECT_FALSE(result);
    EXPECT_FLOAT_EQ(0.5f, es.getBestFitness());
}

// ES-011: ShouldStop_Improvement
TEST_F(EarlyStoppingTest, ShouldStop_Improvement) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);
    bool result = es.shouldStop(0.6f);
    EXPECT_FALSE(result);
    EXPECT_EQ(0, es.getCounter());
}

// ES-012: ShouldStop_ImproveTiny
TEST_F(EarlyStoppingTest, ShouldStop_ImproveTiny) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);
    bool result = es.shouldStop(0.5f + 1e-7f);
    EXPECT_FALSE(result);
    EXPECT_EQ(0, es.getCounter());
}

// ES-013: ShouldStop_ImproveLarge
TEST_F(EarlyStoppingTest, ShouldStop_ImproveLarge) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);
    bool result = es.shouldStop(100.5f);
    EXPECT_FALSE(result);
    EXPECT_EQ(0, es.getCounter());
}

// ES-014: ShouldStop_NegativeImprove
TEST_F(EarlyStoppingTest, ShouldStop_NegativeImprove) {
    EarlyStopping es(10);
    es.shouldStop(-0.5f);
    bool result = es.shouldStop(-0.3f);  // -0.3 > -0.5
    EXPECT_FALSE(result);
    EXPECT_EQ(0, es.getCounter());
}

// ES-015: ShouldStop_BestUpdated
TEST_F(EarlyStoppingTest, ShouldStop_BestUpdated) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);
    es.shouldStop(0.7f);
    EXPECT_FLOAT_EQ(0.7f, es.getBestFitness());
}

// ES-016: ShouldStop_CounterReset
TEST_F(EarlyStoppingTest, ShouldStop_CounterReset) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);
    // Decline
    es.shouldStop(0.4f);
    es.shouldStop(0.3f);
    EXPECT_EQ(2, es.getCounter());
    // Improve
    es.shouldStop(0.8f);
    EXPECT_EQ(0, es.getCounter());
}

// ES-017: ShouldStop_MayStopNextReset
TEST_F(EarlyStoppingTest, ShouldStop_MayStopNextReset) {
    EarlyStopping es(3);
    es.shouldStop(0.5f);
    es.shouldStop(0.4f);  // counter=1
    es.shouldStop(0.3f);  // counter=2, mayStopNext=true (patience-1)
    EXPECT_TRUE(es.mayStopNext());
    // Improve
    es.shouldStop(0.8f);
    EXPECT_FALSE(es.mayStopNext());
}

// =============================================================================
// 3.2.3 shouldStop() - No Improvement Cases (ES-018 ~ ES-023)
// =============================================================================

// ES-018: ShouldStop_NoImprove
TEST_F(EarlyStoppingTest, ShouldStop_NoImprove) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);
    es.shouldStop(0.4f);  // No improvement
    EXPECT_EQ(1, es.getCounter());
}

// ES-019: ShouldStop_Equal
TEST_F(EarlyStoppingTest, ShouldStop_Equal) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);
    es.shouldStop(0.5f);  // Equal = no improvement
    EXPECT_EQ(1, es.getCounter());
}

// ES-020: ShouldStop_CounterInc
TEST_F(EarlyStoppingTest, ShouldStop_CounterInc) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);
    es.shouldStop(0.4f);
    es.shouldStop(0.3f);
    es.shouldStop(0.2f);
    EXPECT_EQ(3, es.getCounter());
}

// ES-021: ShouldStop_NotYetStopped
TEST_F(EarlyStoppingTest, ShouldStop_NotYetStopped) {
    EarlyStopping es(5);
    es.shouldStop(0.5f);
    es.shouldStop(0.4f);
    es.shouldStop(0.3f);
    es.shouldStop(0.2f);
    // counter=3, patience=5
    EXPECT_FALSE(es.shouldStop(0.1f));  // counter=4, still < 5
}

// ES-022: ShouldStop_ExactPatience
TEST_F(EarlyStoppingTest, ShouldStop_ExactPatience) {
    EarlyStopping es(3);
    es.shouldStop(0.5f);  // best=0.5, counter=0
    es.shouldStop(0.4f);  // counter=1
    es.shouldStop(0.3f);  // counter=2
    bool result = es.shouldStop(0.2f);  // counter=3 == patience
    EXPECT_TRUE(result);
}

// ES-023: ShouldStop_ExceedPatience
TEST_F(EarlyStoppingTest, ShouldStop_ExceedPatience) {
    EarlyStopping es(2);
    es.shouldStop(0.5f);
    es.shouldStop(0.4f);  // counter=1
    es.shouldStop(0.3f);  // counter=2 == patience, returns true
    bool result = es.shouldStop(0.2f);  // counter=3 > patience
    EXPECT_TRUE(result);
}

// =============================================================================
// 3.2.4 mayStopNext Logic (ES-024 ~ ES-029)
// =============================================================================

// ES-024: MayStopNext_Initial
TEST_F(EarlyStoppingTest, MayStopNext_Initial) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);
    EXPECT_FALSE(es.mayStopNext());
}

// ES-025: MayStopNext_OneBeforePatience
TEST_F(EarlyStoppingTest, MayStopNext_OneBeforePatience) {
    EarlyStopping es(5);
    es.shouldStop(0.5f);
    for (int i = 0; i < 4; ++i) {
        es.shouldStop(0.4f);  // counter increments
    }
    // counter=4 == patience-1
    EXPECT_TRUE(es.mayStopNext());
}

// ES-026: MayStopNext_TwoBeforePatience
TEST_F(EarlyStoppingTest, MayStopNext_TwoBeforePatience) {
    EarlyStopping es(5);
    es.shouldStop(0.5f);
    es.shouldStop(0.4f);  // counter=1
    es.shouldStop(0.3f);  // counter=2
    es.shouldStop(0.2f);  // counter=3 == patience-2
    EXPECT_FALSE(es.mayStopNext());
}

// ES-027: MayStopNext_AtPatience
TEST_F(EarlyStoppingTest, MayStopNext_AtPatience) {
    EarlyStopping es(3);
    es.shouldStop(0.5f);
    es.shouldStop(0.4f);  // counter=1
    es.shouldStop(0.3f);  // counter=2
    es.shouldStop(0.2f);  // counter=3 >= patience-1
    EXPECT_TRUE(es.mayStopNext());
}

// ES-028: MayStopNext_PatOne_Initial
TEST_F(EarlyStoppingTest, MayStopNext_PatOne_Initial) {
    EarlyStopping es(1);
    es.shouldStop(0.5f);
    es.shouldStop(0.4f);  // counter=1 >= patience-1 (0)
    EXPECT_TRUE(es.mayStopNext());
}

// ES-029: MayStopNext_AfterImprove
TEST_F(EarlyStoppingTest, MayStopNext_AfterImprove) {
    EarlyStopping es(3);
    es.shouldStop(0.5f);
    es.shouldStop(0.4f);
    es.shouldStop(0.3f);  // counter=2 >= patience-1
    EXPECT_TRUE(es.mayStopNext());
    es.shouldStop(0.8f);  // Improvement
    EXPECT_FALSE(es.mayStopNext());
}

// =============================================================================
// 3.2.5 reset() (ES-030 ~ ES-034)
// =============================================================================

// ES-030: Reset_Counter
TEST_F(EarlyStoppingTest, Reset_Counter) {
    EarlyStopping es(20);
    es.shouldStop(0.5f);
    for (int i = 0; i < 10; ++i) {
        es.shouldStop(0.4f);
    }
    EXPECT_EQ(10, es.getCounter());
    es.reset();
    EXPECT_EQ(0, es.getCounter());
}

// ES-031: Reset_BestFitness
TEST_F(EarlyStoppingTest, Reset_BestFitness) {
    EarlyStopping es(10);
    es.shouldStop(0.9f);
    EXPECT_FLOAT_EQ(0.9f, es.getBestFitness());
    es.reset();
    EXPECT_EQ(-std::numeric_limits<float>::infinity(), es.getBestFitness());
}

// ES-032: Reset_MayStopNext
TEST_F(EarlyStoppingTest, Reset_MayStopNext) {
    EarlyStopping es(3);
    es.shouldStop(0.5f);
    es.shouldStop(0.4f);
    es.shouldStop(0.3f);  // mayStopNext=true
    EXPECT_TRUE(es.mayStopNext());
    es.reset();
    EXPECT_FALSE(es.mayStopNext());
}

// ES-033: Reset_FullState
TEST_F(EarlyStoppingTest, Reset_FullState) {
    EarlyStopping es(5);
    es.shouldStop(0.5f);
    es.shouldStop(0.4f);
    es.shouldStop(0.3f);
    es.shouldStop(0.2f);
    // counter=3, best=0.5, mayStopNext=true
    es.reset();
    EXPECT_EQ(0, es.getCounter());
    EXPECT_EQ(-std::numeric_limits<float>::infinity(), es.getBestFitness());
    EXPECT_FALSE(es.mayStopNext());
}

// ES-034: Reset_ThenContinue
TEST_F(EarlyStoppingTest, Reset_ThenContinue) {
    EarlyStopping es(3);
    es.shouldStop(0.5f);
    es.shouldStop(0.4f);
    es.shouldStop(0.3f);
    es.reset();
    // Fresh start
    bool result = es.shouldStop(0.1f);
    EXPECT_FALSE(result);
    EXPECT_FLOAT_EQ(0.1f, es.getBestFitness());
    EXPECT_EQ(0, es.getCounter());
}

// =============================================================================
// 3.2.6 setPatience() (ES-035 ~ ES-040)
// =============================================================================

// ES-035: SetPatience_Valid
TEST_F(EarlyStoppingTest, SetPatience_Valid) {
    EarlyStopping es(50);
    es.setPatience(100);
    EXPECT_EQ(100, es.getPatience());
}

// ES-036: SetPatience_Zero
TEST_F(EarlyStoppingTest, SetPatience_Zero) {
    EarlyStopping es(50);
    EXPECT_THROW({
        es.setPatience(0);
    }, ConfigurationException);
}

// ES-037: SetPatience_Negative
TEST_F(EarlyStoppingTest, SetPatience_Negative) {
    EarlyStopping es(50);
    EXPECT_THROW({
        es.setPatience(-1);
    }, ConfigurationException);
}

// ES-038: SetPatience_ReduceUpdates
TEST_F(EarlyStoppingTest, SetPatience_ReduceUpdates) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);
    for (int i = 0; i < 4; ++i) {
        es.shouldStop(0.4f);
    }
    // counter=4
    es.setPatience(5);  // counter=4 >= patience-1 (4)
    EXPECT_TRUE(es.mayStopNext());
}

// ES-039: SetPatience_IncreaseUpdates
TEST_F(EarlyStoppingTest, SetPatience_IncreaseUpdates) {
    EarlyStopping es(5);
    es.shouldStop(0.5f);
    for (int i = 0; i < 4; ++i) {
        es.shouldStop(0.4f);
    }
    // counter=4 >= patience-1 (4), mayStopNext=true
    EXPECT_TRUE(es.mayStopNext());
    es.setPatience(20);  // counter=4 < patience-1 (19)
    EXPECT_FALSE(es.mayStopNext());
}

// ES-040: SetPatience_SameValue
TEST_F(EarlyStoppingTest, SetPatience_SameValue) {
    EarlyStopping es(50);
    es.setPatience(50);
    EXPECT_EQ(50, es.getPatience());
}

// =============================================================================
// 3.2.7 Getter Methods (ES-041 ~ ES-044)
// =============================================================================

// ES-041: Get_BestFitness
TEST_F(EarlyStoppingTest, Get_BestFitness) {
    EarlyStopping es(10);
    es.shouldStop(0.75f);
    EXPECT_FLOAT_EQ(0.75f, es.getBestFitness());
}

// ES-042: Get_Counter
TEST_F(EarlyStoppingTest, Get_Counter) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);
    es.shouldStop(0.4f);
    es.shouldStop(0.3f);
    EXPECT_EQ(2, es.getCounter());
}

// ES-043: Get_Patience
TEST_F(EarlyStoppingTest, Get_Patience) {
    EarlyStopping es(77);
    EXPECT_EQ(77, es.getPatience());
}

// ES-044: Get_MayStopNext
TEST_F(EarlyStoppingTest, Get_MayStopNext) {
    EarlyStopping es(2);
    es.shouldStop(0.5f);
    EXPECT_FALSE(es.mayStopNext());
    es.shouldStop(0.4f);  // counter=1 >= patience-1 (1)
    EXPECT_TRUE(es.mayStopNext());
}

// =============================================================================
// 3.2.8 Integration Scenarios (ES-045 ~ ES-051)
// =============================================================================

// ES-045: Scenario_MonotonicImprove
TEST_F(EarlyStoppingTest, Scenario_MonotonicImprove) {
    EarlyStopping es(10);
    for (int i = 0; i < 100; ++i) {
        bool result = es.shouldStop(static_cast<float>(i) * 0.01f);
        EXPECT_FALSE(result);
        EXPECT_EQ(0, es.getCounter());
    }
}

// ES-046: Scenario_NoImprove
TEST_F(EarlyStoppingTest, Scenario_NoImprove) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);  // Initial best
    for (int i = 0; i < 9; ++i) {
        bool result = es.shouldStop(0.5f);  // No improvement
        EXPECT_FALSE(result);
    }
    bool finalResult = es.shouldStop(0.5f);  // counter=10
    EXPECT_TRUE(finalResult);
}

// ES-047: Scenario_Oscillating
TEST_F(EarlyStoppingTest, Scenario_Oscillating) {
    EarlyStopping es(5);
    es.shouldStop(0.5f);  // best=0.5
    es.shouldStop(0.3f);  // counter=1
    es.shouldStop(0.6f);  // best=0.6, counter=0
    es.shouldStop(0.4f);  // counter=1
    es.shouldStop(0.7f);  // best=0.7, counter=0
    EXPECT_EQ(0, es.getCounter());
    EXPECT_FLOAT_EQ(0.7f, es.getBestFitness());
}

// ES-048: Scenario_PlateauThenImprove
TEST_F(EarlyStoppingTest, Scenario_PlateauThenImprove) {
    EarlyStopping es(5);
    es.shouldStop(0.5f);
    // Plateau for patience-1 epochs
    for (int i = 0; i < 4; ++i) {
        es.shouldStop(0.5f);
    }
    EXPECT_TRUE(es.mayStopNext());
    // Improve just in time
    es.shouldStop(0.6f);
    EXPECT_EQ(0, es.getCounter());
    EXPECT_FALSE(es.mayStopNext());
}

// ES-049: Scenario_LargePatience
TEST_F(EarlyStoppingTest, Scenario_LargePatience) {
    EarlyStopping es(1000);
    es.shouldStop(0.5f);
    for (int i = 0; i < 999; ++i) {
        bool result = es.shouldStop(0.4f);
        EXPECT_FALSE(result);
    }
    bool finalResult = es.shouldStop(0.4f);  // counter=1000
    EXPECT_TRUE(finalResult);
}

// ES-050: Scenario_SmallPatience
TEST_F(EarlyStoppingTest, Scenario_SmallPatience) {
    EarlyStopping es(1);
    es.shouldStop(0.5f);
    bool result = es.shouldStop(0.4f);  // counter=1 == patience
    EXPECT_TRUE(result);
}

// ES-051: Scenario_ExactStop
TEST_F(EarlyStoppingTest, Scenario_ExactStop) {
    EarlyStopping es(5);
    es.shouldStop(0.5f);
    for (int i = 0; i < 4; ++i) {
        bool result = es.shouldStop(0.4f);
        EXPECT_FALSE(result);
    }
    bool result = es.shouldStop(0.4f);  // counter=5 == patience
    EXPECT_TRUE(result);
}

// =============================================================================
// 3.2.9 Edge Cases (ES-052 ~ ES-057)
// =============================================================================

// ES-052: Edge_InfinityFitness
TEST_F(EarlyStoppingTest, Edge_InfinityFitness) {
    EarlyStopping es(10);
    bool result = es.shouldStop(std::numeric_limits<float>::infinity());
    EXPECT_FALSE(result);
    EXPECT_EQ(std::numeric_limits<float>::infinity(), es.getBestFitness());
}

// ES-053: Edge_NegInfinityFitness
TEST_F(EarlyStoppingTest, Edge_NegInfinityFitness) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);  // best=0.5
    es.shouldStop(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(1, es.getCounter());  // No improvement
}

// ES-054: Edge_NaNFitness
TEST_F(EarlyStoppingTest, Edge_NaNFitness) {
    EarlyStopping es(10);
    es.shouldStop(0.5f);
    // NaN comparison: NaN > 0.5 is false
    es.shouldStop(std::nanf(""));
    EXPECT_EQ(1, es.getCounter());  // No improvement (NaN not > best)
}

// ES-055: Edge_VerySmallFitness
TEST_F(EarlyStoppingTest, Edge_VerySmallFitness) {
    EarlyStopping es(10);
    bool result = es.shouldStop(1e-38f);
    EXPECT_FALSE(result);
    EXPECT_FLOAT_EQ(1e-38f, es.getBestFitness());
}

// ES-056: Edge_VeryLargeFitness
TEST_F(EarlyStoppingTest, Edge_VeryLargeFitness) {
    EarlyStopping es(10);
    bool result = es.shouldStop(1e38f);
    EXPECT_FALSE(result);
    EXPECT_FLOAT_EQ(1e38f, es.getBestFitness());
}

// ES-057: Edge_ZeroFitness
TEST_F(EarlyStoppingTest, Edge_ZeroFitness) {
    EarlyStopping es(10);
    bool result = es.shouldStop(0.0f);
    EXPECT_FALSE(result);
    EXPECT_FLOAT_EQ(0.0f, es.getBestFitness());
}

// =============================================================================
// 3.2.10 Exception Details (ES-058 ~ ES-061)
// =============================================================================

// ES-058: ExType_Ctor
TEST_F(EarlyStoppingTest, ExType_Ctor) {
    try {
        EarlyStopping es(0);
        FAIL() << "Expected ConfigurationException";
    }
    catch (const ConfigurationException& e) {
        EXPECT_EQ(ErrorCode::INVALID_CONFIG, e.getErrorCode());
    }
}

// ES-059: ExType_SetPatience
TEST_F(EarlyStoppingTest, ExType_SetPatience) {
    EarlyStopping es(50);
    try {
        es.setPatience(0);
        FAIL() << "Expected ConfigurationException";
    }
    catch (const ConfigurationException& e) {
        EXPECT_EQ(ErrorCode::INVALID_CONFIG, e.getErrorCode());
    }
}

// ES-060: ExMsg_ContainsValue
TEST_F(EarlyStoppingTest, ExMsg_ContainsValue) {
    try {
        EarlyStopping es(-5);
        FAIL() << "Expected ConfigurationException";
    }
    catch (const ConfigurationException& e) {
        std::string msg = e.what();
        EXPECT_NE(std::string::npos, msg.find("-5"));
    }
}

// ES-061: ExCode_InvalidConfig
TEST_F(EarlyStoppingTest, ExCode_InvalidConfig) {
    try {
        EarlyStopping es(-10);
        FAIL() << "Expected ConfigurationException";
    }
    catch (const ConfigurationException& e) {
        EXPECT_EQ(ErrorCode::INVALID_CONFIG, e.getErrorCode());
    }
}
