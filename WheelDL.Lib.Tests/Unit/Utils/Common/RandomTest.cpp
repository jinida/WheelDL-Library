#include "pch.h"
#include "Utils/Common/Random.h"
#include <thread>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <gtest/gtest.h>

using namespace WheelDL::Utils;

// =============================================================================
// Random Constructor Tests (RND-001 ~ RND-002)
// =============================================================================

// RND-001: Default constructor creates valid Random object
TEST(RandomTest, DefaultConstructor_CreatesValidObject) {
    EXPECT_NO_THROW({
        Random rng;
        int value = rng.uniformInt(0, 100);
        EXPECT_GE(value, 0);
        EXPECT_LE(value, 100);
    });
}

// RND-002: Constructor with seed produces reproducible sequences
TEST(RandomTest, Constructor_WithSeed_Reproducible) {
    const unsigned int seed = 12345;

    Random rng1(seed);
    Random rng2(seed);

    // Generate same sequence of numbers
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(rng1.uniformInt(0, 1000), rng2.uniformInt(0, 1000));
    }
}

// =============================================================================
// setSeed Tests (RND-003 ~ RND-004)
// =============================================================================

// RND-003: setSeed allows reproducing sequences
TEST(RandomTest, SetSeed_ReproducesSequence) {
    Random rng;
    const unsigned int seed = 54321;

    // Set seed and generate sequence
    rng.setSeed(seed);
    std::vector<int> sequence1;
    for (int i = 0; i < 10; ++i) {
        sequence1.push_back(rng.uniformInt(0, 1000));
    }

    // Reset with same seed
    rng.setSeed(seed);
    std::vector<int> sequence2;
    for (int i = 0; i < 10; ++i) {
        sequence2.push_back(rng.uniformInt(0, 1000));
    }

    EXPECT_EQ(sequence1, sequence2);
}

// RND-004: Different seeds produce different sequences
TEST(RandomTest, SetSeed_DifferentSeeds_DifferentSequences) {
    Random rng1;
    Random rng2;

    rng1.setSeed(11111);
    rng2.setSeed(22222);

    // Generate sequences
    std::vector<int> sequence1, sequence2;
    for (int i = 0; i < 10; ++i) {
        sequence1.push_back(rng1.uniformInt(0, 1000000));
        sequence2.push_back(rng2.uniformInt(0, 1000000));
    }

    // Sequences should be different
    EXPECT_NE(sequence1, sequence2);
}

// =============================================================================
// uniformInt Tests (RND-005 ~ RND-009)
// =============================================================================

// RND-005: uniformInt returns values within [min, max] range
TEST(RandomTest, UniformInt_WithinRange) {
    Random rng(42);
    const int min = 10;
    const int max = 50;

    for (int i = 0; i < 1000; ++i) {
        int value = rng.uniformInt(min, max);
        EXPECT_GE(value, min);
        EXPECT_LE(value, max);
    }
}

// RND-006: uniformInt throws for invalid range (min > max)
TEST(RandomTest, UniformInt_InvalidRange_Throws) {
    Random rng;
    EXPECT_THROW(rng.uniformInt(100, 10), std::invalid_argument);
}

// RND-007: uniformInt handles min == max (returns that value)
TEST(RandomTest, UniformInt_EqualMinMax) {
    Random rng;
    const int value = 42;

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(value, rng.uniformInt(value, value));
    }
}

// RND-008: uniformInt produces uniform distribution (chi-square test)
TEST(RandomTest, UniformInt_UniformDistribution) {
    Random rng(42);
    const int min = 0;
    const int max = 9;
    const int numSamples = 10000;

    std::map<int, int> counts;
    for (int i = min; i <= max; ++i) {
        counts[i] = 0;
    }

    for (int i = 0; i < numSamples; ++i) {
        counts[rng.uniformInt(min, max)]++;
    }

    // Expected count for each value
    double expected = static_cast<double>(numSamples) / (max - min + 1);

    // Chi-square test
    double chiSquare = 0.0;
    for (const auto& pair : counts) {
        double diff = pair.second - expected;
        chiSquare += (diff * diff) / expected;
    }

    // With 9 degrees of freedom, chi-square critical value at 0.05 is ~16.92
    EXPECT_LT(chiSquare, 20.0);
}

// RND-009: uniformInt covers boundary values
TEST(RandomTest, UniformInt_BoundaryValues) {
    Random rng(42);
    const int min = 0;
    const int max = 5;

    std::set<int> observedValues;
    for (int i = 0; i < 1000; ++i) {
        observedValues.insert(rng.uniformInt(min, max));
    }

    // All values in range should eventually appear
    for (int v = min; v <= max; ++v) {
        EXPECT_TRUE(observedValues.count(v) > 0)
            << "Value " << v << " was never generated";
    }
}

// =============================================================================
// uniformFloat Tests (RND-010 ~ RND-013)
// =============================================================================

// RND-010: uniformFloat returns values within [min, max) range
TEST(RandomTest, UniformFloat_WithinRange) {
    Random rng(42);
    const float min = 0.0f;
    const float max = 1.0f;

    for (int i = 0; i < 1000; ++i) {
        float value = rng.uniformFloat(min, max);
        EXPECT_GE(value, min);
        EXPECT_LT(value, max);  // Exclusive upper bound
    }
}

// RND-011: uniformFloat throws for invalid range
TEST(RandomTest, UniformFloat_InvalidRange_Throws) {
    Random rng;
    EXPECT_THROW(rng.uniformFloat(10.0f, 5.0f), std::invalid_argument);
}

// RND-012: uniformFloat handles negative ranges
TEST(RandomTest, UniformFloat_NegativeRange) {
    Random rng(42);
    const float min = -10.0f;
    const float max = -5.0f;

    for (int i = 0; i < 100; ++i) {
        float value = rng.uniformFloat(min, max);
        EXPECT_GE(value, min);
        EXPECT_LT(value, max);
    }
}

// RND-013: uniformFloat handles min == max
TEST(RandomTest, UniformFloat_EqualMinMax) {
    Random rng;
    const float value = 3.14f;

    float result = rng.uniformFloat(value, value);
    EXPECT_FLOAT_EQ(value, result);
}

// =============================================================================
// uniformDouble Tests (RND-014 ~ RND-015)
// =============================================================================

// RND-014: uniformDouble returns values within [min, max) range
TEST(RandomTest, UniformDouble_WithinRange) {
    Random rng(42);
    const double min = 0.0;
    const double max = 1.0;

    for (int i = 0; i < 1000; ++i) {
        double value = rng.uniformDouble(min, max);
        EXPECT_GE(value, min);
        EXPECT_LT(value, max);
    }
}

// RND-015: uniformDouble throws for invalid range
TEST(RandomTest, UniformDouble_InvalidRange_Throws) {
    Random rng;
    EXPECT_THROW(rng.uniformDouble(100.0, 50.0), std::invalid_argument);
}

// =============================================================================
// normal Tests (RND-016 ~ RND-019)
// =============================================================================

// RND-016: normal returns values (basic functionality)
TEST(RandomTest, Normal_ReturnsValues) {
    Random rng(42);

    std::vector<float> values;
    for (int i = 0; i < 100; ++i) {
        values.push_back(rng.normal(0.0f, 1.0f));
    }

    // Check that we get variety of values
    std::set<float> uniqueValues(values.begin(), values.end());
    EXPECT_GT(uniqueValues.size(), 50u);
}

// RND-017: normal distribution has correct mean (approximately)
TEST(RandomTest, Normal_CorrectMean) {
    Random rng(42);
    const float mean = 5.0f;
    const float stddev = 1.0f;
    const int numSamples = 10000;

    double sum = 0.0;
    for (int i = 0; i < numSamples; ++i) {
        sum += rng.normal(mean, stddev);
    }

    double sampleMean = sum / numSamples;
    EXPECT_NEAR(sampleMean, mean, 0.1);  // Within 0.1 of expected mean
}

// RND-018: normal distribution has correct stddev (approximately)
TEST(RandomTest, Normal_CorrectStddev) {
    Random rng(42);
    const float mean = 0.0f;
    const float stddev = 2.0f;
    const int numSamples = 10000;

    std::vector<float> values;
    for (int i = 0; i < numSamples; ++i) {
        values.push_back(rng.normal(mean, stddev));
    }

    // Calculate sample mean and stddev
    double sampleMean = std::accumulate(values.begin(), values.end(), 0.0) / numSamples;

    double sumSquaredDiff = 0.0;
    for (float v : values) {
        double diff = v - sampleMean;
        sumSquaredDiff += diff * diff;
    }
    double sampleStddev = std::sqrt(sumSquaredDiff / numSamples);

    EXPECT_NEAR(sampleStddev, stddev, 0.2);  // Within 0.2 of expected stddev
}

// RND-019: normal throws for negative stddev
TEST(RandomTest, Normal_NegativeStddev_Throws) {
    Random rng;
    EXPECT_THROW(rng.normal(0.0f, -1.0f), std::invalid_argument);
}

// =============================================================================
// bernoulli Tests (RND-020 ~ RND-023)
// =============================================================================

// RND-020: bernoulli returns true/false
TEST(RandomTest, Bernoulli_ReturnsBool) {
    Random rng(42);

    bool seenTrue = false;
    bool seenFalse = false;

    for (int i = 0; i < 100; ++i) {
        if (rng.bernoulli(0.5)) {
            seenTrue = true;
        }
        else {
            seenFalse = true;
        }
    }

    EXPECT_TRUE(seenTrue);
    EXPECT_TRUE(seenFalse);
}

// RND-021: bernoulli with probability 0.0 always returns false
TEST(RandomTest, Bernoulli_ProbabilityZero) {
    Random rng;

    for (int i = 0; i < 100; ++i) {
        EXPECT_FALSE(rng.bernoulli(0.0));
    }
}

// RND-022: bernoulli with probability 1.0 always returns true
TEST(RandomTest, Bernoulli_ProbabilityOne) {
    Random rng;

    for (int i = 0; i < 100; ++i) {
        EXPECT_TRUE(rng.bernoulli(1.0));
    }
}

// RND-023: bernoulli throws for invalid probability
TEST(RandomTest, Bernoulli_InvalidProbability_Throws) {
    Random rng;

    EXPECT_THROW(rng.bernoulli(-0.1), std::invalid_argument);
    EXPECT_THROW(rng.bernoulli(1.1), std::invalid_argument);
}

// RND-023b: bernoulli has correct probability distribution
TEST(RandomTest, Bernoulli_CorrectProbability) {
    Random rng(42);
    const double probability = 0.3;
    const int numSamples = 10000;

    int trueCount = 0;
    for (int i = 0; i < numSamples; ++i) {
        if (rng.bernoulli(probability)) {
            trueCount++;
        }
    }

    double observedProbability = static_cast<double>(trueCount) / numSamples;
    EXPECT_NEAR(observedProbability, probability, 0.02);
}

// =============================================================================
// shuffle Tests (RND-024 ~ RND-025)
// =============================================================================

// RND-024: shuffle modifies vector
TEST(RandomTest, Shuffle_ModifiesVector) {
    Random rng(42);

    std::vector<int> original = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> shuffled = original;

    rng.shuffle(shuffled);

    // Vector should contain same elements but likely different order
    std::vector<int> sortedShuffled = shuffled;
    std::sort(sortedShuffled.begin(), sortedShuffled.end());
    EXPECT_EQ(original, sortedShuffled);

    // With high probability, order should be different
    // (There's a 1/10! chance it's the same, but that's essentially zero)
    EXPECT_NE(original, shuffled);
}

// RND-025: shuffle preserves all elements
TEST(RandomTest, Shuffle_PreservesElements) {
    Random rng(42);

    std::vector<std::string> original = {"a", "b", "c", "d", "e"};
    std::vector<std::string> shuffled = original;

    rng.shuffle(shuffled);

    std::set<std::string> originalSet(original.begin(), original.end());
    std::set<std::string> shuffledSet(shuffled.begin(), shuffled.end());

    EXPECT_EQ(originalSet, shuffledSet);
}

// RND-025b: shuffle handles empty vector
TEST(RandomTest, Shuffle_EmptyVector) {
    Random rng;
    std::vector<int> empty;

    EXPECT_NO_THROW(rng.shuffle(empty));
    EXPECT_TRUE(empty.empty());
}

// RND-025c: shuffle handles single element vector
TEST(RandomTest, Shuffle_SingleElement) {
    Random rng;
    std::vector<int> single = {42};

    EXPECT_NO_THROW(rng.shuffle(single));
    EXPECT_EQ(1u, single.size());
    EXPECT_EQ(42, single[0]);
}

// =============================================================================
// Thread Safety Tests (RND-026)
// =============================================================================

// RND-026: Random is thread-safe for concurrent operations
TEST(RandomTest, ThreadSafety_ConcurrentOperations) {
    Random rng(42);
    std::atomic<bool> hasError{false};
    std::atomic<int> operationCount{0};

    const int numThreads = 4;
    const int opsPerThread = 100;
    std::vector<std::thread> threads;

    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&rng, &hasError, &operationCount, opsPerThread]() {
            try {
                for (int i = 0; i < opsPerThread; ++i) {
                    // Mix of different operations
                    int randInt = rng.uniformInt(0, 100);
                    if (randInt < 0 || randInt > 100) hasError.store(true);

                    float randFloat = rng.uniformFloat(0.0f, 1.0f);
                    if (randFloat < 0.0f || randFloat >= 1.0f) hasError.store(true);

                    rng.normal(0.0f, 1.0f);
                    rng.bernoulli(0.5);

                    operationCount.fetch_add(4);
                }
            }
            catch (...) {
                hasError.store(true);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_FALSE(hasError.load());
    EXPECT_EQ(numThreads * opsPerThread * 4, operationCount.load());
}

// =============================================================================
// getGenerator and getMutex Tests
// =============================================================================

// Test that getGenerator can be used with manual locking
TEST(RandomTest, GetGenerator_ManualLocking) {
    Random rng(42);

    {
        std::lock_guard<std::mutex> lock(rng.getMutex());
        std::mt19937& gen = rng.getGenerator();

        // Use generator directly
        std::uniform_int_distribution<int> dist(0, 100);
        int value = dist(gen);

        EXPECT_GE(value, 0);
        EXPECT_LE(value, 100);
    }
}
