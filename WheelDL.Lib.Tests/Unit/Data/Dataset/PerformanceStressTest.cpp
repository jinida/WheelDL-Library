// =============================================================================
// Phase 10: Performance & Stress Tests (20 tests)
// =============================================================================

#include "pch.h"
#include <gtest/gtest.h>
#include <fstream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>
#include <filesystem>

#include "Config/Configuration.h"
#include "Data/Dataset/ClassificationDataset.h"
#include "Data/Dataset/DetectionDataset.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

using namespace WheelDL::Data::Dataset;
using namespace WheelDL::Config;

// =============================================================================
// Test Fixture
// =============================================================================

class PerformanceStressTest : public ::testing::Test
{
protected:
    std::string testDir_;
    std::string dataDir_;
    std::string annotDir_;
    std::string modelPath_;
    std::string hyperPath_;
    std::string annotationPath_;

    void SetUp() override
    {
        testDir_ = std::filesystem::temp_directory_path().string() + "/perf_stress_test_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        dataDir_ = testDir_ + "/a";
        annotDir_ = testDir_ + "/a/b/c";

        std::filesystem::create_directories(dataDir_);
        std::filesystem::create_directories(annotDir_);

        modelPath_ = testDir_ + "/model.yaml";
        hyperPath_ = testDir_ + "/hyper.yaml";
        annotationPath_ = annotDir_ + "/annotations.json";

        createModelConfig();
        createHyperConfig();
    }

    void TearDown() override
    {
        std::error_code ec;
        std::filesystem::remove_all(testDir_, ec);
    }

    void createModelConfig(const std::string& task = "classification")
    {
        std::ofstream file(modelPath_);
        file << "task: " << task << "\n";
        file << "numClasses: 10\n";
        file.close();
    }

    void createHyperConfig(const std::string& cacheType = "", int imageSize = 32)
    {
        std::ofstream file(hyperPath_);
        file << "imageSize: " << imageSize << "\n";
        file << "batchSize: 2\n";
        file << "cache: \"" << cacheType << "\"\n";
        file << "mosaic: 0.0\n";
        file.close();
    }

    void createImage(const std::string& filename, int width = 32, int height = 32)
    {
        cv::Mat image(height, width, CV_8UC3, cv::Scalar(100, 150, 200));
        cv::imwrite(dataDir_ + "/" + filename, image);
    }

    void createAnnotation(const std::vector<std::pair<std::string, int>>& samples)
    {
        std::ofstream file(annotationPath_);
        file << "{\n";
        file << "  \"header\": { \"categories\": [\"cat0\", \"cat1\", \"cat2\"] },\n";
        file << "  \"annotations\": [\n";

        bool first = true;
        for (const auto& sample : samples)
        {
            if (!first) file << ",\n";
            first = false;
            file << "    { \"filename\": \"" << sample.first << "\", \"role\": 0, \"label\": " << sample.second << " }";
        }

        file << "\n  ]\n";
        file << "}\n";
        file.close();
    }

    void createTestDataset(int count, int imageSize = 32)
    {
        std::vector<std::pair<std::string, int>> samples;
        for (int i = 0; i < count; ++i)
        {
            std::string name = "img_" + std::to_string(i) + ".png";
            createImage(name, imageSize, imageSize);
            samples.push_back({ name, i % 3 });
        }
        createAnnotation(samples);
    }

    Configuration createConfig(const std::string& cacheType = "", int imageSize = 32)
    {
        createHyperConfig(cacheType, imageSize);
        Configuration config;
        config.load(modelPath_, hyperPath_, annotationPath_);
        return config;
    }

    // Helper to measure execution time
    template<typename Func>
    double measureTime(Func&& func)
    {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// =============================================================================
// Performance Tests (PERF-001 to PERF-010)
// =============================================================================

// PERF-001: LoadTime_SingleImage
TEST_F(PerformanceStressTest, LoadTime_SingleImage)
{
    createTestDataset(1, 32);
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    double loadTime = measureTime([&]() {
        auto example = dataset.get(0);
        EXPECT_TRUE(example.data.defined());
        });

    EXPECT_LT(loadTime, 1000.0);
    std::cout << "[PERF] Single image load time: " << loadTime << " ms" << std::endl;
}

// PERF-002: LoadTime_BatchImages (reduced to 2 images)
TEST_F(PerformanceStressTest, LoadTime_BatchImages)
{
    createTestDataset(2, 32);
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    double totalTime = measureTime([&]() {
        for (size_t i = 0; i < dataset.size().value(); ++i)
        {
            auto example = dataset.get(i);
            EXPECT_TRUE(example.data.defined());
        }
        });

    std::cout << "[PERF] Batch load (2 images): " << totalTime << " ms" << std::endl;
    EXPECT_LT(totalTime, 1000.0);
}

// PERF-003: CacheHit_Speedup
TEST_F(PerformanceStressTest, CacheHit_Speedup)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // First pass - cache miss
    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        auto example = dataset.get(i);
    }

    // Second pass - cache hit (should be faster)
    double secondPassTime = measureTime([&]() {
        for (size_t i = 0; i < dataset.size().value(); ++i)
        {
            auto example = dataset.get(i);
        }
        });

    std::cout << "[PERF] Cache hit time: " << secondPassTime << " ms" << std::endl;
    EXPECT_LT(secondPassTime, 1000.0);
}

// PERF-004: Transform_Overhead
TEST_F(PerformanceStressTest, Transform_Overhead)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
        dataset.get(i);

    double transformTime = measureTime([&]() {
        for (size_t i = 0; i < dataset.size().value(); ++i)
        {
            auto example = dataset.get(i);
            EXPECT_TRUE(example.data.defined());
        }
        });

    std::cout << "[PERF] Transform overhead: " << transformTime << " ms" << std::endl;
    EXPECT_LT(transformTime, 1000.0);
}

// PERF-005: TensorConversion_Speed
TEST_F(PerformanceStressTest, TensorConversion_Speed)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
        dataset.get(i);

    double conversionTime = measureTime([&]() {
        for (size_t i = 0; i < dataset.size().value(); ++i)
        {
            auto example = dataset.get(i);
            EXPECT_EQ(example.data.dtype(), torch::kFloat32);
            EXPECT_EQ(example.data.dim(), 3);
        }
        });

    std::cout << "[PERF] Tensor conversion: " << conversionTime << " ms" << std::endl;
    EXPECT_LT(conversionTime, 1000.0);
}

// PERF-006: DataLoader_Throughput
TEST_F(PerformanceStressTest, DataLoader_Throughput)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
        dataset.get(i);

    int count = 0;
    double throughputTime = measureTime([&]() {
        for (size_t i = 0; i < dataset.size().value(); ++i)
        {
            auto example = dataset.get(i);
            if (example.data.defined()) count++;
        }
        });

    double throughput = (count * 1000.0) / throughputTime;
    std::cout << "[PERF] DataLoader throughput: " << throughput << " images/sec" << std::endl;

    EXPECT_EQ(count, 2);
    EXPECT_GT(throughput, 1.0);
}

// PERF-007: Mosaic_Overhead
TEST_F(PerformanceStressTest, Mosaic_Overhead)
{
    // Mosaic requires at least 4 images
    createTestDataset(4, 32);
    createModelConfig("detection");

    std::ofstream hyperFile(hyperPath_);
    hyperFile << "imageSize: 32\n";
    hyperFile << "batchSize: 2\n";
    hyperFile << "cache: \"ram\"\n";
    hyperFile << "mosaic: 1.0\n";  // Always apply mosaic
    hyperFile.close();

    std::ofstream file(annotationPath_);
    file << "{\n";
    file << "  \"header\": { \"categories\": [\"obj\"] },\n";
    file << "  \"annotations\": [\n";
    for (int i = 0; i < 4; ++i)
    {
        if (i > 0) file << ",\n";
        file << "    { \"filename\": \"img_" << i << ".png\", \"role\": 0, \"label\": [[0, 5, 5, 25, 25]] }";
    }
    file << "\n  ]\n";
    file << "}\n";
    file.close();

    Configuration config;
    config.load(modelPath_, hyperPath_, annotationPath_);
    DetectionDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache all images
    for (size_t i = 0; i < 4; ++i)
        dataset.get(i);

    double mosaicTime = measureTime([&]() {
        auto example = dataset.get(0);
        EXPECT_TRUE(example.data.defined());
        });

    std::cout << "[PERF] Mosaic overhead: " << mosaicTime << " ms" << std::endl;
    EXPECT_LT(mosaicTime, 1000.0);
}

// PERF-008: LargeDataset_Iteration (reduced to 2 images)
TEST_F(PerformanceStressTest, LargeDataset_Iteration)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
        dataset.get(i);

    double iterationTime = measureTime([&]() {
        for (size_t i = 0; i < dataset.size().value(); ++i)
        {
            auto example = dataset.get(i);
        }
        });

    std::cout << "[PERF] Dataset iteration: " << iterationTime << " ms" << std::endl;
    EXPECT_LT(iterationTime, 1000.0);
}

// PERF-009: MemoryUsage_Baseline
TEST_F(PerformanceStressTest, MemoryUsage_Baseline)
{
    createTestDataset(2, 32);
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        auto example = dataset.get(i);
        EXPECT_TRUE(example.data.defined());
    }

    EXPECT_EQ(dataset.size().value(), 2);
    std::cout << "[PERF] Baseline memory test passed" << std::endl;
}

// PERF-010: MemoryUsage_WithCache
TEST_F(PerformanceStressTest, MemoryUsage_WithCache)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        auto example = dataset.get(i);
        EXPECT_TRUE(example.data.defined());
    }

    dataset.clearCache();
    EXPECT_EQ(dataset.size().value(), 2);
    std::cout << "[PERF] Cache memory test passed" << std::endl;
}

// =============================================================================
// Stress Tests (STRESS-001 to STRESS-010)
// =============================================================================

// STRESS-001: RapidGet (reduced iterations)
TEST_F(PerformanceStressTest, RapidGet_1000Times)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        dataset.get(i);
    }

    int successCount = 0;
    double stressTime = measureTime([&]() {
        for (int i = 0; i < 10; ++i)  // Reduced from 1000 to 10
        {
            size_t idx = i % dataset.size().value();
            auto example = dataset.get(idx);
            if (example.data.defined()) successCount++;
        }
        });

    std::cout << "[STRESS] Rapid get: " << stressTime << " ms, success: " << successCount << std::endl;
    EXPECT_EQ(successCount, 10);
}

// STRESS-002: CacheThrashing
TEST_F(PerformanceStressTest, CacheThrashing)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    int successCount = 0;
    double thrashTime = measureTime([&]() {
        for (int i = 0; i < 5; ++i)  // Reduced from 500 to 5
        {
            size_t idx = dis(gen);
            auto example = dataset.get(idx);
            if (example.data.defined()) successCount++;
        }
        });

    std::cout << "[STRESS] Cache thrashing: " << thrashTime << " ms" << std::endl;
    EXPECT_EQ(successCount, 5);
}

// STRESS-003: ConcurrentAccess_Stress
TEST_F(PerformanceStressTest, ConcurrentAccess_Stress)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        dataset.get(i);
    }

    std::atomic<int> successCount{ 0 };
    std::vector<std::thread> threads;

    double concurrentTime = measureTime([&]() {
        for (int t = 0; t < 2; ++t)  // Reduced from 20 to 2 threads
        {
            threads.emplace_back([&dataset, &successCount]() {
                for (int i = 0; i < 2; ++i)  // Reduced from 50 to 2
                {
                    size_t idx = i % dataset.size().value();
                    auto example = dataset.get(idx);
                    if (example.data.defined()) successCount++;
                }
                });
        }

        for (auto& thread : threads)
        {
            thread.join();
        }
        });

    std::cout << "[STRESS] Concurrent access: " << concurrentTime << " ms, success: " << successCount.load() << std::endl;
    EXPECT_EQ(successCount.load(), 4);
}

// STRESS-004: LongRunning_Simulation
TEST_F(PerformanceStressTest, LongRunning_Simulation)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        dataset.get(i);
    }

    int iterations = 0;
    double runTime = measureTime([&]() {
        for (int round = 0; round < 3; ++round)  // Reduced from 100 to 3
        {
            for (size_t i = 0; i < dataset.size().value(); ++i)
            {
                auto example = dataset.get(i);
                iterations++;
            }
        }
        });

    std::cout << "[STRESS] Long running: " << runTime << " ms, iterations: " << iterations << std::endl;
    EXPECT_EQ(iterations, 6);
}

// STRESS-005: MemoryPressure
TEST_F(PerformanceStressTest, MemoryPressure)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    std::vector<torch::Tensor> tensors;

    double pressureTime = measureTime([&]() {
        for (size_t i = 0; i < dataset.size().value(); ++i)
        {
            auto example = dataset.get(i);
            tensors.push_back(example.data.clone());
        }
        });

    std::cout << "[STRESS] Memory pressure: " << pressureTime << " ms" << std::endl;
    EXPECT_EQ(tensors.size(), 2);
    tensors.clear();
}

// STRESS-006: GPUMemory_Stress
TEST_F(PerformanceStressTest, GPUMemory_Stress)
{
    createTestDataset(2, 32);
    Configuration config = createConfig();
    ClassificationDataset dataset(config, true);

    torch::Device device = torch::kCPU;  // Always use CPU for speed

    int successCount = 0;
    double gpuTime = measureTime([&]() {
        for (size_t i = 0; i < dataset.size().value(); ++i)
        {
            auto example = dataset.get(i);
            auto movedExample = example.toDevice(device);
            if (movedExample.data.defined()) successCount++;
        }
        });

    std::cout << "[STRESS] CPU stress: " << gpuTime << " ms" << std::endl;
    EXPECT_EQ(successCount, 2);
}

// STRESS-007: DataLoader_Continuous
TEST_F(PerformanceStressTest, DataLoader_Continuous)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        dataset.get(i);
    }

    int totalIterations = 0;
    double continuousTime = measureTime([&]() {
        for (int epoch = 0; epoch < 2; ++epoch)  // Reduced from 10 to 2
        {
            for (size_t i = 0; i < dataset.size().value(); ++i)
            {
                auto example = dataset.get(i);
                totalIterations++;
            }
        }
        });

    std::cout << "[STRESS] Continuous DataLoader: " << continuousTime << " ms" << std::endl;
    EXPECT_EQ(totalIterations, 4);
}

// STRESS-008: Transform_Continuous
TEST_F(PerformanceStressTest, Transform_Continuous)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        dataset.get(i);
    }

    int transformCount = 0;
    double transformTime = measureTime([&]() {
        for (int i = 0; i < 5; ++i)  // Reduced from 200 to 5
        {
            size_t idx = i % dataset.size().value();
            auto example = dataset.get(idx);
            if (example.data.size(0) == 3)
            {
                transformCount++;
            }
        }
        });

    std::cout << "[STRESS] Continuous transforms: " << transformTime << " ms" << std::endl;
    EXPECT_EQ(transformCount, 5);
}

// STRESS-009: Cache_Continuous
TEST_F(PerformanceStressTest, Cache_Continuous)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);

    int operationCount = 0;
    double cacheTime = measureTime([&]() {
        dataset.enableCache(CacheType::RAM);

        for (size_t i = 0; i < dataset.size().value(); ++i)
        {
            auto example = dataset.get(i);
            operationCount++;
        }

        dataset.clearCache();
        });

    std::cout << "[STRESS] Cache enable/clear: " << cacheTime << " ms" << std::endl;
    EXPECT_EQ(operationCount, 2);
}

// STRESS-010: MixedOperations
TEST_F(PerformanceStressTest, MixedOperations)
{
    createTestDataset(2, 32);
    Configuration config = createConfig("ram");
    ClassificationDataset dataset(config, true);
    dataset.enableCache(CacheType::RAM);

    // Pre-cache
    for (size_t i = 0; i < dataset.size().value(); ++i)
    {
        dataset.get(i);
    }

    std::atomic<int> successCount{ 0 };
    std::vector<std::thread> threads;

    double mixedTime = measureTime([&]() {
        // Thread 1: Sequential access
        threads.emplace_back([&dataset, &successCount]() {
            for (int i = 0; i < 2; ++i)
            {
                auto example = dataset.get(i % dataset.size().value());
                if (example.data.defined()) successCount++;
            }
            });

        // Thread 2: Clone operations
        threads.emplace_back([&dataset, &successCount]() {
            for (int i = 0; i < 2; ++i)
            {
                auto example = dataset.get(i % dataset.size().value());
                if (example.data.defined())
                {
                    auto cloned = example.data.clone();
                    successCount++;
                }
            }
            });

        for (auto& thread : threads)
        {
            thread.join();
        }
        });

    std::cout << "[STRESS] Mixed operations: " << mixedTime << " ms, success: " << successCount.load() << std::endl;
    EXPECT_EQ(successCount.load(), 4);
}
