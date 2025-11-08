#include "pch.h"
#include "WheelDL.Lib/Utils/Common/Random.h"
#include <vector>
#include <algorithm>
#include <cmath>

using namespace WheelDL::Utils;

/**
 * @brief Random number generator tests
 */
class RandomTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use fixed seed for reproducible tests
        rng = std::make_unique<Random>(12345);
    }

    std::unique_ptr<Random> rng;
};

TEST_F(RandomTest, Constructor) {
    // Default constructor should work
    Random rng1;

    // Constructor with seed should work
    Random rng2(42);

    // Should generate different values with different instances
    int val1 = rng1.uniformInt(0, 1000);
    int val2 = rng2.uniformInt(0, 1000);

    // Very unlikely to be the same (though possible)
    // Just verify they're in valid range
    EXPECT_GE(val1, 0);
    EXPECT_LE(val1, 1000);
    EXPECT_GE(val2, 0);
    EXPECT_LE(val2, 1000);
}

TEST_F(RandomTest, SetSeed) {
    Random rng1(12345);
    Random rng2(12345);

    // Same seed should produce same sequence
    for (int i = 0; i < 10; ++i) {
        int val1 = rng1.uniformInt(0, 100);
        int val2 = rng2.uniformInt(0, 100);
        EXPECT_EQ(val1, val2);
    }

    // Reset seed of rng2
    rng2.setSeed(12345);

    // Should produce same sequence again
    Random rng3(12345);
    for (int i = 0; i < 10; ++i) {
        int val2 = rng2.uniformInt(0, 100);
        int val3 = rng3.uniformInt(0, 100);
        EXPECT_EQ(val2, val3);
    }
}

TEST_F(RandomTest, UniformInt) {
    // Test range [0, 10]
    for (int i = 0; i < 100; ++i) {
        int val = rng->uniformInt(0, 10);
        EXPECT_GE(val, 0);
        EXPECT_LE(val, 10);
    }

    // Test range [50, 60]
    for (int i = 0; i < 100; ++i) {
        int val = rng->uniformInt(50, 60);
        EXPECT_GE(val, 50);
        EXPECT_LE(val, 60);
    }

    // Test negative range [-10, 10]
    for (int i = 0; i < 100; ++i) {
        int val = rng->uniformInt(-10, 10);
        EXPECT_GE(val, -10);
        EXPECT_LE(val, 10);
    }
}

TEST_F(RandomTest, UniformIntDistribution) {
    // Generate many samples and check distribution
    std::vector<int> counts(10, 0);

    for (int i = 0; i < 10000; ++i) {
        int val = rng->uniformInt(0, 9);
        counts[val]++;
    }

    // Each value should appear roughly 1000 times (10000 / 10)
    // Allow 20% deviation
    for (int count : counts) {
        EXPECT_GT(count, 800);   // At least 800
        EXPECT_LT(count, 1200);  // At most 1200
    }
}

TEST_F(RandomTest, UniformFloat) {
    // Test range [0.0, 1.0]
    for (int i = 0; i < 100; ++i) {
        float val = rng->uniformFloat(0.0f, 1.0f);
        EXPECT_GE(val, 0.0f);
        EXPECT_LT(val, 1.0f);  // Exclusive upper bound
    }

    // Test range [5.0, 10.0]
    for (int i = 0; i < 100; ++i) {
        float val = rng->uniformFloat(5.0f, 10.0f);
        EXPECT_GE(val, 5.0f);
        EXPECT_LT(val, 10.0f);
    }

    // Test negative range [-1.0, 1.0]
    for (int i = 0; i < 100; ++i) {
        float val = rng->uniformFloat(-1.0f, 1.0f);
        EXPECT_GE(val, -1.0f);
        EXPECT_LT(val, 1.0f);
    }
}

TEST_F(RandomTest, UniformDouble) {
    // Test range [0.0, 1.0]
    for (int i = 0; i < 100; ++i) {
        double val = rng->uniformDouble(0.0, 1.0);
        EXPECT_GE(val, 0.0);
        EXPECT_LT(val, 1.0);
    }

    // Test range [100.0, 200.0]
    for (int i = 0; i < 100; ++i) {
        double val = rng->uniformDouble(100.0, 200.0);
        EXPECT_GE(val, 100.0);
        EXPECT_LT(val, 200.0);
    }
}

TEST_F(RandomTest, Normal) {
    // Generate many samples from normal distribution
    std::vector<float> samples;
    samples.reserve(10000);

    for (int i = 0; i < 10000; ++i) {
        float val = rng->normal(0.0f, 1.0f);  // Mean=0, StdDev=1
        samples.push_back(val);
    }

    // Calculate sample mean
    float sum = 0.0f;
    for (float val : samples) {
        sum += val;
    }
    float mean = sum / samples.size();

    // Calculate sample standard deviation
    float variance = 0.0f;
    for (float val : samples) {
        float diff = val - mean;
        variance += diff * diff;
    }
    float stddev = std::sqrt(variance / samples.size());

    // Mean should be close to 0 (within 0.05)
    EXPECT_NEAR(mean, 0.0f, 0.05f);

    // Standard deviation should be close to 1 (within 0.05)
    EXPECT_NEAR(stddev, 1.0f, 0.05f);
}

TEST_F(RandomTest, NormalWithCustomParameters) {
    // Test normal distribution with mean=10, stddev=2
    std::vector<float> samples;
    samples.reserve(10000);

    for (int i = 0; i < 10000; ++i) {
        float val = rng->normal(10.0f, 2.0f);
        samples.push_back(val);
    }

    // Calculate sample mean
    float sum = 0.0f;
    for (float val : samples) {
        sum += val;
    }
    float mean = sum / samples.size();

    // Mean should be close to 10 (within 0.1)
    EXPECT_NEAR(mean, 10.0f, 0.1f);
}

TEST_F(RandomTest, Bernoulli) {
    // Test with p=0.5
    int trueCount = 0;
    int totalCount = 10000;

    for (int i = 0; i < totalCount; ++i) {
        if (rng->bernoulli(0.5)) {
            trueCount++;
        }
    }

    // Should be roughly 50% (allow 3% deviation)
    double ratio = static_cast<double>(trueCount) / totalCount;
    EXPECT_NEAR(ratio, 0.5, 0.03);
}

TEST_F(RandomTest, BernoulliWithCustomProbability) {
    // Test with p=0.3
    int trueCount = 0;
    int totalCount = 10000;

    for (int i = 0; i < totalCount; ++i) {
        if (rng->bernoulli(0.3)) {
            trueCount++;
        }
    }

    // Should be roughly 30% (allow 3% deviation)
    double ratio = static_cast<double>(trueCount) / totalCount;
    EXPECT_NEAR(ratio, 0.3, 0.03);
}

TEST_F(RandomTest, Shuffle) {
    // Create a vector with known values
    std::vector<int> vec = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> original = vec;

    // Shuffle the vector
    rng->shuffle(vec);

    // Should contain same elements
    std::sort(vec.begin(), vec.end());
    EXPECT_EQ(vec, original);

    // Reset and shuffle again with same seed - should produce same result
    vec = original;
    rng->setSeed(12345);
    rng->shuffle(vec);

    std::vector<int> shuffled1 = vec;

    vec = original;
    rng->setSeed(12345);
    rng->shuffle(vec);

    EXPECT_EQ(vec, shuffled1);
}

TEST_F(RandomTest, GetGenerator) {
    // Test that we can access the internal generator
    std::mt19937& generator = rng->getGenerator();

    // Use it to generate a value
    std::uniform_int_distribution<int> dist(0, 100);
    int val = dist(generator);

    EXPECT_GE(val, 0);
    EXPECT_LE(val, 100);
}
