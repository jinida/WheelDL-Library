#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Optimizer/EarlyStopping/EarlyStopping.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"

using namespace WheelDL::Optimizer::EarlyStopping;

class EarlyStoppingTest : public ::testing::Test {
protected:
    void SetUp() override {
        patience = 5;
        earlyStopping = std::make_unique<EarlyStopping>(patience);
    }

    int patience;
    std::unique_ptr<EarlyStopping> earlyStopping;
};

TEST_F(EarlyStoppingTest, Construction) {
    EXPECT_NO_THROW({
        EarlyStopping es(10);
    });

    EXPECT_EQ(earlyStopping->getPatience(), patience);
    EXPECT_EQ(earlyStopping->getCounter(), 0);
    EXPECT_FALSE(earlyStopping->mayStopNext());
}

TEST_F(EarlyStoppingTest, InvalidPatience) {
    EXPECT_THROW(
        EarlyStopping(0),
        WheelDL::Utils::ConfigurationException
    );

    EXPECT_THROW(
        EarlyStopping(-5),
        WheelDL::Utils::ConfigurationException
    );
}

TEST_F(EarlyStoppingTest, ImprovementResets) {
    EXPECT_FALSE(earlyStopping->shouldStop(0.5f));  // First fitness
    EXPECT_EQ(earlyStopping->getCounter(), 0);

    EXPECT_FALSE(earlyStopping->shouldStop(0.4f));  // Worse
    EXPECT_EQ(earlyStopping->getCounter(), 1);

    EXPECT_FALSE(earlyStopping->shouldStop(0.3f));  // Worse
    EXPECT_EQ(earlyStopping->getCounter(), 2);

    EXPECT_FALSE(earlyStopping->shouldStop(0.6f));  // Better!
    EXPECT_EQ(earlyStopping->getCounter(), 0);      // Counter reset
}

TEST_F(EarlyStoppingTest, StopAfterPatience) {
    float initialFitness = 0.5f;

    // Initial fitness
    EXPECT_FALSE(earlyStopping->shouldStop(initialFitness));

    // patience epochs without improvement
    for (int i = 0; i < patience - 1; ++i) {
        EXPECT_FALSE(earlyStopping->shouldStop(initialFitness - 0.1f));
        EXPECT_EQ(earlyStopping->getCounter(), i + 1);
    }

    // One more epoch should trigger stop
    EXPECT_TRUE(earlyStopping->shouldStop(initialFitness - 0.1f));
}

TEST_F(EarlyStoppingTest, MayStopNext) {
    float initialFitness = 0.5f;

    earlyStopping->shouldStop(initialFitness);
    EXPECT_FALSE(earlyStopping->mayStopNext());

    // patience - 2 epochs without improvement
    for (int i = 0; i < patience - 2; ++i) {
        earlyStopping->shouldStop(initialFitness - 0.1f);
        EXPECT_FALSE(earlyStopping->mayStopNext());
    }

    // patience - 1 epoch: should set mayStopNext
    earlyStopping->shouldStop(initialFitness - 0.1f);
    EXPECT_TRUE(earlyStopping->mayStopNext());

    // Final epoch: should still be true and trigger stop
    EXPECT_TRUE(earlyStopping->mayStopNext());
    EXPECT_TRUE(earlyStopping->shouldStop(initialFitness - 0.1f));
}

TEST_F(EarlyStoppingTest, BestFitnessTracking) {
    EXPECT_FALSE(earlyStopping->shouldStop(0.5f));
    EXPECT_FLOAT_EQ(earlyStopping->getBestFitness(), 0.5f);

    EXPECT_FALSE(earlyStopping->shouldStop(0.4f));  // Worse
    EXPECT_FLOAT_EQ(earlyStopping->getBestFitness(), 0.5f);  // Unchanged

    EXPECT_FALSE(earlyStopping->shouldStop(0.7f));  // Better
    EXPECT_FLOAT_EQ(earlyStopping->getBestFitness(), 0.7f);  // Updated

    EXPECT_FALSE(earlyStopping->shouldStop(0.6f));  // Worse
    EXPECT_FLOAT_EQ(earlyStopping->getBestFitness(), 0.7f);  // Unchanged
}

TEST_F(EarlyStoppingTest, Reset) {
    earlyStopping->shouldStop(0.5f);
    earlyStopping->shouldStop(0.4f);
    earlyStopping->shouldStop(0.3f);

    EXPECT_EQ(earlyStopping->getCounter(), 2);
    EXPECT_FLOAT_EQ(earlyStopping->getBestFitness(), 0.5f);

    earlyStopping->reset();

    EXPECT_EQ(earlyStopping->getCounter(), 0);
    EXPECT_FALSE(earlyStopping->mayStopNext());
    // Best fitness should be reset to -infinity
    EXPECT_LT(earlyStopping->getBestFitness(), -1e30f);
}

TEST_F(EarlyStoppingTest, SetPatience) {
    earlyStopping->setPatience(10);
    EXPECT_EQ(earlyStopping->getPatience(), 10);

    EXPECT_THROW(
        earlyStopping->setPatience(0),
        WheelDL::Utils::ConfigurationException
    );

    EXPECT_THROW(
        earlyStopping->setPatience(-1),
        WheelDL::Utils::ConfigurationException
    );
}

TEST_F(EarlyStoppingTest, ChangingPatienceUpdatesMayStopNext) {
    float initialFitness = 0.5f;

    earlyStopping->shouldStop(initialFitness);

    // patience - 1 epochs without improvement
    for (int i = 0; i < patience - 1; ++i) {
        earlyStopping->shouldStop(initialFitness - 0.1f);
    }

    EXPECT_TRUE(earlyStopping->mayStopNext());

    // Increase patience
    earlyStopping->setPatience(10);

    // mayStopNext should be updated
    EXPECT_FALSE(earlyStopping->mayStopNext());
}

TEST_F(EarlyStoppingTest, EqualFitnessCountsAsNoImprovement) {
    earlyStopping->shouldStop(0.5f);
    EXPECT_EQ(earlyStopping->getCounter(), 0);

    // Same fitness
    earlyStopping->shouldStop(0.5f);
    EXPECT_EQ(earlyStopping->getCounter(), 1);  // No improvement
}

TEST_F(EarlyStoppingTest, SmallImprovementResets) {
    earlyStopping->shouldStop(0.500f);
    EXPECT_EQ(earlyStopping->getCounter(), 0);

    earlyStopping->shouldStop(0.499f);
    EXPECT_EQ(earlyStopping->getCounter(), 1);

    // Even small improvement should reset
    earlyStopping->shouldStop(0.5001f);
    EXPECT_EQ(earlyStopping->getCounter(), 0);
}

TEST_F(EarlyStoppingTest, LongRunWithoutImprovement) {
    float initialFitness = 1.0f;

    earlyStopping->shouldStop(initialFitness);

    // Run for 2 * patience epochs without improvement
    for (int i = 0; i < patience - 1; ++i) {
        EXPECT_FALSE(earlyStopping->shouldStop(initialFitness - 0.1f));
    }

    // Should stop at patience
    EXPECT_TRUE(earlyStopping->shouldStop(initialFitness - 0.1f));

    // Continuing should still return true
    EXPECT_TRUE(earlyStopping->shouldStop(initialFitness - 0.1f));
}
