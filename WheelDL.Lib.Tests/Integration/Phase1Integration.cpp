#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Utils/Logger/Logger.h"
#include "WheelDL.Lib/Utils/ThreadPool/ThreadPool.h"
#include "WheelDL.Lib/Utils/Path/PathValidator.h"
#include "WheelDL.Lib/Utils/Profiler/PerformanceProfiler.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include "WheelDL.Lib/Utils/Error/ErrorCodes.h"
#include "WheelDL.Lib/Utils/Common/Types.h"
#include "WheelDL.Lib/Utils/Common/Constants.h"
#include "WheelDL.Lib/Config/Configuration.h"
#include "WheelDL.Lib/Config/YamlParser.h"
#include "WheelDL.Lib/Utils/Memory/MemoryManager.h"
#include "WheelDL.Lib/Utils/Memory/GPUMemoryGuard.h"
#include "WheelDL.Lib/Utils/Memory/GPUMemoryPool.h"
#include "WheelDL.Lib/Utils/Memory/DynamicBatchSizer.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <numeric>

using namespace WheelDL;
using namespace WheelDL::Utils;
namespace fs = std::filesystem;

/**
 * @class Phase1IntegrationTest
 * @brief Integration tests for Phase 1 components
 *
 * These tests verify that Phase 1 components work correctly together
 * in realistic usage scenarios:
 * - Logging with performance profiling
 * - Parallel file processing with thread pool
 * - Error handling across components
 * - Combined utility usage
 */
class Phase1IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        logger = Logger::getInstance();
        logger->setGlobalLogLevel(LogLevel::INFO);

        profiler = &PerformanceProfiler::getInstance();
        profiler->reset();
        profiler->setEnabled(true);

        // Create test directory
        testDir = fs::current_path() / "integration_test_data";
        fs::create_directories(testDir);
    }

    void TearDown() override {
        profiler->reset();

        if (logger) {
            logger->reset();
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        if (fs::exists(testDir)) {
            try {
                fs::remove_all(testDir);
            } catch (const std::exception&) {
            }
        }

        // Clean up log files
        try {
            if (fs::exists("integration_test.log")) {
                fs::remove("integration_test.log");
            }
            if (fs::exists("performance_report.json")) {
                fs::remove("performance_report.json");
            }
            if (fs::exists("performance_report.html")) {
                fs::remove("performance_report.html");
            }
        } catch (const std::exception&) {
            // Ignore cleanup errors
        }
    }

    std::shared_ptr<Logger> logger;
    PerformanceProfiler* profiler;
    fs::path testDir;
};

// ============================================================================
// Logging + Performance Profiling Integration
// ============================================================================

TEST_F(Phase1IntegrationTest, LoggingWithProfiling_SimpleWorkflow) {
    // Arrange
    logger->setLogFile((testDir / "test.log").string());

    // Act
    {
        auto timer = profiler->createScopedTimer("workflow");
        logger->info("Starting workflow");

        profiler->start("step1");
        logger->debug("Executing step 1");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        profiler->stop("step1");

        profiler->start("step2");
        logger->debug("Executing step 2");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        profiler->stop("step2");

        logger->info("Workflow completed");
    }

    // Assert
    auto workflowStats = profiler->getStatistics("workflow");
    auto step1Stats = profiler->getStatistics("step1");
    auto step2Stats = profiler->getStatistics("step2");

    EXPECT_GE(workflowStats.total, step1Stats.total + step2Stats.total);
    EXPECT_TRUE(fs::exists(testDir / "test.log"));
}

TEST_F(Phase1IntegrationTest, LoggingWithProfiling_ExportReport) {
    // Arrange & Act
    for (int i = 0; i < 10; ++i) {
        auto timer = profiler->createScopedTimer("iteration");
        logger->info("Iteration " + std::to_string(i));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // Export performance report
    profiler->exportToJSON((testDir / "report.json").string());
    profiler->exportToHTML((testDir / "report.html").string());

    // Assert
    EXPECT_TRUE(fs::exists(testDir / "report.json"));
    EXPECT_TRUE(fs::exists(testDir / "report.html"));

    auto stats = profiler->getStatistics("iteration");
    EXPECT_EQ(stats.count, 10);
}

// ============================================================================
// ThreadPool + Logging Integration
// ============================================================================

TEST_F(Phase1IntegrationTest, ThreadPoolWithLogging_ParallelTasks) {
    // Arrange
    ThreadPool pool(4);
    std::mutex logMutex;

    // Act
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 20; ++i) {
        futures.push_back(pool.enqueue([this, i, &logMutex]() {
            {
                std::lock_guard<std::mutex> lock(logMutex);
                logger->info("Task " + std::to_string(i) + " started");
            }

            int result = i * i;

            {
                std::lock_guard<std::mutex> lock(logMutex);
                logger->info("Task " + std::to_string(i) + " completed: " + std::to_string(result));
            }

            return result;
        }));
    }

    // Assert
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(futures[i].get(), i * i);
    }
}

TEST_F(Phase1IntegrationTest, ThreadPoolWithProfiling_MeasureParallelPerformance) {
    // Arrange
    ThreadPool pool(4);
    const int numTasks = 100;

    // Act
    {
        auto timer = profiler->createScopedTimer("parallel_execution");

        std::vector<std::future<void>> futures;
        for (int i = 0; i < numTasks; ++i) {
            futures.push_back(pool.enqueue([]() {
                // Simulate some work
                double sum = 0;
                for (int j = 0; j < 1000; ++j) {
                    sum += std::sqrt(j);
                }
            }));
        }

        for (auto& f : futures) {
            f.wait();
        }
    }

    // Assert
    auto stats = profiler->getStatistics("parallel_execution");
    EXPECT_GT(stats.total, 0.0);
    logger->info("Parallel execution completed in " + std::to_string(stats.total) + " ms");
}

// ============================================================================
// PathValidator + File Processing Integration
// ============================================================================

TEST_F(Phase1IntegrationTest, PathValidationWithFileProcessing_ValidateAndProcess) {
    // Arrange - Create test files
    std::vector<std::string> testFiles = {
        "data1.yaml", "data2.yaml", "data3.json", "readme.txt"
    };

    for (const auto& filename : testFiles) {
        std::ofstream file(testDir / filename);
        file << "test content";
    }

    // Act - Validate and filter YAML files
    std::vector<std::string> yamlFiles;
    for (const auto& entry : fs::directory_iterator(testDir)) {
        std::string path = entry.path().string();

        if (PathValidator::fileExists(path) &&
            PathValidator::hasValidExtension(path, {".yaml", ".yml"})) {
            yamlFiles.push_back(path);
            logger->info("Found YAML file: " + PathValidator::getFilename(path));
        }
    }

    // Assert
    EXPECT_EQ(yamlFiles.size(), 2);  // data1.yaml, data2.yaml
}

TEST_F(Phase1IntegrationTest, PathValidationWithLogging_DirectoryCreation) {
    // Arrange
    fs::path targetDir = testDir / "nested" / "directory" / "structure";

    // Act
    profiler->start("directory_creation");
    logger->info("Creating directory: " + targetDir.string());

    bool created = PathValidator::createDirectoryIfNotExists(targetDir.string());

    profiler->stop("directory_creation");
    logger->info("Directory creation completed in " +
                 std::to_string(profiler->getDuration("directory_creation")) + " ms");

    // Assert
    EXPECT_TRUE(created);
    EXPECT_TRUE(PathValidator::directoryExists(targetDir.string()));
}

// ============================================================================
// Exception Handling + Logging Integration
// ============================================================================

TEST_F(Phase1IntegrationTest, ExceptionHandlingWithLogging_ConfigurationError) {
    // Arrange & Act
    try {
        logger->error("Configuration validation failed");
        throw ConfigurationException(
            ErrorCode::INVALID_CONFIG,
            "Missing required field: model_path"
        );
    }
    catch (const ConfigurationException& ex) {
        // Log the error
        logger->error("Caught ConfigurationException: " + std::string(ex.what()));
        logger->error("Error code: " + ex.getErrorCodeString());

        // Assert
        EXPECT_EQ(ex.getErrorCode(), ErrorCode::INVALID_CONFIG);
        SUCCEED();
        return;
    }

    FAIL() << "Exception not caught";
}

TEST_F(Phase1IntegrationTest, ExceptionHandlingWithLogging_NestedExceptions) {
    // Arrange - Create nested exception scenario
    std::exception_ptr innerEx;

    try {
        logger->info("Attempting to load file");
        throw std::runtime_error("File not found: model.pth");
    }
    catch (...) {
        innerEx = std::current_exception();
        logger->error("File I/O error occurred");
    }

    // Act
    try {
        throw ModelException(
            ErrorCode::MODEL_LOAD_FAILED,
            "Failed to load pretrained model",
            innerEx
        );
    }
    catch (const ModelException& ex) {
        logger->error("Model loading failed: " + std::string(ex.what()));

        // Try to access inner exception
        if (ex.getInnerException()) {
            try {
                std::rethrow_exception(ex.getInnerException());
            }
            catch (const std::runtime_error& inner) {
                logger->error("Root cause: " + std::string(inner.what()));
            }
        }

        // Assert
        EXPECT_NE(ex.getInnerException(), nullptr);
        SUCCEED();
        return;
    }

    FAIL();
}

// ============================================================================
// ThreadPool + Exception Handling Integration
// ============================================================================

TEST_F(Phase1IntegrationTest, ThreadPoolWithExceptionHandling_TaskFailures) {
    // Arrange
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;

    // Act - Submit mix of successful and failing tasks
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool.enqueue([this, i]() -> int {
            if (i % 3 == 0) {
                logger->error("Task " + std::to_string(i) + " failing");
                throw DataException(
                    ErrorCode::DATA_LOAD_FAILED,
                    "Simulated failure for task " + std::to_string(i)
                );
            }
            logger->info("Task " + std::to_string(i) + " succeeded");
            return i * 2;
        }));
    }

    // Assert - Verify exception handling
    int successCount = 0;
    int failureCount = 0;

    for (int i = 0; i < 10; ++i) {
        try {
            int result = futures[i].get();
            successCount++;
            EXPECT_EQ(result, i * 2);
        }
        catch (const DataException& ex) {
            failureCount++;
            EXPECT_EQ(ex.getErrorCode(), ErrorCode::DATA_LOAD_FAILED);
        }
    }

    EXPECT_EQ(successCount, 6);   // i = 1,2,4,5,7,8
    EXPECT_EQ(failureCount, 4);   // i = 0,3,6,9
}

// ============================================================================
// Full Workflow Integration Tests
// ============================================================================

TEST_F(Phase1IntegrationTest, FullWorkflow_DataProcessingPipeline) {
    // Simulate a realistic data processing pipeline using all Phase 1 components

    // Setup
    logger->info("=== Starting Data Processing Pipeline ===");
    auto pipelineTimer = profiler->createScopedTimer("full_pipeline");

    // Step 1: Validate and prepare directories
    {
        auto timer = profiler->createScopedTimer("step1_setup");
        logger->info("Step 1: Setting up directories");

        fs::path inputDir = testDir / "input";
        fs::path outputDir = testDir / "output";

        if (!PathValidator::createDirectoryIfNotExists(inputDir.string())) {
            throw ConfigurationException(
                ErrorCode::INVALID_CONFIG,
                "Failed to create input directory"
            );
        }

        if (!PathValidator::createDirectoryIfNotExists(outputDir.string())) {
            throw ConfigurationException(
                ErrorCode::INVALID_CONFIG,
                "Failed to create output directory"
            );
        }

        logger->info("Directories created successfully");
    }

    // Step 2: Generate test data files
    {
        auto timer = profiler->createScopedTimer("step2_data_generation");
        logger->info("Step 2: Generating test data");

        for (int i = 0; i < 50; ++i) {
            std::string filename = "data_" + std::to_string(i) + ".txt";
            fs::path filepath = testDir / "input" / filename;

            std::ofstream file(filepath);
            file << "Sample data " << i << "\n";
            file << "Value: " << (i * 100) << "\n";
        }

        logger->info("Generated 50 test files");
    }

    // Step 3: Process files in parallel
    {
        auto timer = profiler->createScopedTimer("step3_parallel_processing");
        logger->info("Step 3: Processing files in parallel");

        ThreadPool pool(8);
        std::vector<std::future<int>> futures;

        int fileCount = 0;
        for (const auto& entry : fs::directory_iterator(testDir / "input")) {
            if (PathValidator::hasValidExtension(entry.path().string(), {".txt"})) {
                futures.push_back(pool.enqueue([this, entry]() -> int {
                    // Read input file
                    std::ifstream inFile(entry.path());
                    std::string content((std::istreambuf_iterator<char>(inFile)),
                                       std::istreambuf_iterator<char>());

                    // Process (simulate)
                    int lineCount = std::count(content.begin(), content.end(), '\n');

                    // Write output
                    std::string outputFilename = "processed_" +
                                                PathValidator::getFilename(entry.path().string());
                    fs::path outputPath = testDir / "output" / outputFilename;

                    std::ofstream outFile(outputPath);
                    outFile << "Processed: " << content;
                    outFile << "Line count: " << lineCount << "\n";

                    return lineCount;
                }));

                fileCount++;
            }
        }

        // Wait for all tasks
        int totalLines = 0;
        for (auto& f : futures) {
            try {
                totalLines += f.get();
            }
            catch (const std::exception& ex) {
                logger->error("File processing failed: " + std::string(ex.what()));
            }
        }

        logger->info("Processed " + std::to_string(fileCount) + " files, " +
                     std::to_string(totalLines) + " total lines");
    }

    // Step 4: Validate results
    {
        auto timer = profiler->createScopedTimer("step4_validation");
        logger->info("Step 4: Validating results");

        int outputFileCount = 0;
        for (const auto& entry : fs::directory_iterator(testDir / "output")) {
            if (PathValidator::fileExists(entry.path().string())) {
                outputFileCount++;
            }
        }

        if (outputFileCount != 50) {
            throw DataException(
                ErrorCode::DATA_PREPROCESSING_FAILED,
                "Expected 50 output files, got " + std::to_string(outputFileCount)
            );
        }

        logger->info("Validation passed: all files processed correctly");
    }

    logger->info("=== Pipeline Completed Successfully ===");

    // Generate performance report
    std::string report = profiler->report();
    logger->info("\n" + report);

    // Assert
    EXPECT_TRUE(fs::exists(testDir / "output"));

    // Verify all steps completed (don't check full_pipeline as it's still running)
    auto step1Stats = profiler->getStatistics("step1_setup");
    EXPECT_GT(step1Stats.total, 0.0);
    EXPECT_NO_THROW(profiler->getStatistics("step2_data_generation"));
    EXPECT_NO_THROW(profiler->getStatistics("step3_parallel_processing"));
    EXPECT_NO_THROW(profiler->getStatistics("step4_validation"));
}

TEST_F(Phase1IntegrationTest, FullWorkflow_ErrorRecovery) {
    // Simulate error handling and recovery

    logger->info("=== Testing Error Recovery Workflow ===");

    // Attempt 1: Will fail
    try {
        auto timer = profiler->createScopedTimer("attempt1");
        logger->info("Attempt 1: Trying to process non-existent file");

        std::string fakePath = (testDir / "nonexistent.txt").string();

        if (!PathValidator::fileExists(fakePath)) {
            throw DataException(
                ErrorCode::DATASET_NOT_FOUND,
                "File not found: " + fakePath
            );
        }

        FAIL() << "Should have thrown exception";
    }
    catch (const DataException& ex) {
        logger->warn("Attempt 1 failed: " + std::string(ex.what()));
        EXPECT_EQ(ex.getErrorCode(), ErrorCode::DATASET_NOT_FOUND);
    }

    // Attempt 2: Create file and retry
    try {
        auto timer = profiler->createScopedTimer("attempt2");
        logger->info("Attempt 2: Creating file and retrying");

        std::string filePath = (testDir / "recovery.txt").string();

        // Create the file
        std::ofstream file(filePath);
        file << "Recovery data";
        file.close();

        if (!PathValidator::fileExists(filePath)) {
            throw DataException(
                ErrorCode::DATASET_NOT_FOUND,
                "File still not found after creation"
            );
        }

        logger->info("Attempt 2 succeeded");
        SUCCEED();
    }
    catch (const DataException& ex) {
        logger->error("Attempt 2 failed: " + std::string(ex.what()));
        FAIL();
    }

    logger->info("=== Error Recovery Workflow Completed ===");

    // Verify both attempts were profiled
    EXPECT_NO_THROW(profiler->getStatistics("attempt1"));
    EXPECT_NO_THROW(profiler->getStatistics("attempt2"));
}

// ============================================================================
// Constants and Types Integration
// ============================================================================

TEST_F(Phase1IntegrationTest, ConstantsAndTypes_UsageInWorkflow) {
    // Demonstrate usage of Constants and Types

    logger->info("Library: " + std::string(Constants::LIBRARY_NAME));
    logger->info("Version: " + std::to_string(Constants::MAJOR_VERSION) + "." +
                               std::to_string(Constants::MINOR_VERSION) + "." +
                               std::to_string(Constants::PATCH_VERSION));

    // Simulate training configuration using constants
    HyperParameters hyperParams;
    hyperParams.batchSize = Constants::DEFAULT_BATCH_SIZE;
    hyperParams.epochs = Constants::DEFAULT_EPOCHS;
    hyperParams.learningRate = Constants::DEFAULT_LEARNING_RATE;
    hyperParams.weightDecay = Constants::DEFAULT_WEIGHT_DECAY;
    hyperParams.momentum = Constants::DEFAULT_MOMENTUM;
    hyperParams.warmupEpochs = Constants::DEFAULT_WARMUP_EPOCHS;
    hyperParams.optimizer = "Adam";
    hyperParams.scheduler = "CosineAnnealing";

    logger->info("Training configuration:");
    logger->info("  Batch size: " + std::to_string(hyperParams.batchSize));
    logger->info("  Epochs: " + std::to_string(hyperParams.epochs));
    logger->info("  Learning rate: " + std::to_string(hyperParams.learningRate));

    // Simulate task type
    TaskType currentTask = TaskType::DETECTION;
    logger->info("Task: " + std::string(taskTypeToString(currentTask)));

    // Simulate training state progression
    std::vector<TrainingState> states = {
        TrainingState::INITIALIZING,
        TrainingState::TRAINING,
        TrainingState::VALIDATING,
        TrainingState::COMPLETED
    };

    for (const auto& state : states) {
        logger->info("State: " + std::string(trainingStateToString(state)));
    }

    // Assert
    EXPECT_EQ(hyperParams.batchSize, 32);
    EXPECT_EQ(currentTask, TaskType::DETECTION);
}

// ============================================================================
// Configuration + YamlParser Integration
// ============================================================================

TEST_F(Phase1IntegrationTest, ConfigurationWithYamlParser_LoadAndValidate) {
    // Arrange - Load test YAML files from Data/TestConfigs
    // Search upward for WheelDL.Lib.Tests directory
    fs::path searchPath = fs::current_path();
    fs::path testProjectDir;

    // Search up to 5 levels for WheelDL.Lib.Tests directory
    for (int i = 0; i < 5; ++i) {
        fs::path candidatePath = searchPath / "WheelDL.Lib.Tests";
        if (fs::exists(candidatePath) && fs::is_directory(candidatePath)) {
            testProjectDir = candidatePath;
            break;
        }
        searchPath = searchPath.parent_path();
    }

    fs::path testConfigDir = testProjectDir / "Data" / "TestConfigs";
    fs::path modelPath = testConfigDir / "test_model.yaml";
    fs::path hyperparamPath = testConfigDir / "test_hyperparam.yaml";

    // Verify test files exist
    ASSERT_TRUE(fs::exists(modelPath)) << "Test model YAML not found: " << modelPath;
    ASSERT_TRUE(fs::exists(hyperparamPath)) << "Test hyperparam YAML not found: " << hyperparamPath;

    // Act
    profiler->start("config_load");
    logger->info("Loading configuration from YAML files");
    logger->info("  Model: " + modelPath.string());
    logger->info("  Hyperparam: " + hyperparamPath.string());
    Config::Configuration config;
    
    EXPECT_NO_THROW(config.loadFromYaml(modelPath.string(), hyperparamPath.string()));

    profiler->stop("config_load");
    logger->info("Configuration loaded in " +
                 std::to_string(profiler->getDuration("config_load")) + " ms");

    // Assert - Verify configuration values
    EXPECT_EQ(config.getTaskType(), TaskType::DETECTION);
    EXPECT_EQ(config.getNumClasses(), 10);
    EXPECT_EQ(config.getBatchSize(), 64);
    EXPECT_EQ(config.getEpochs(), 200);
    EXPECT_FLOAT_EQ(config.getLearningRate(), 0.001f);
    EXPECT_FLOAT_EQ(config.getMomentum(), 0.937f);
    EXPECT_FLOAT_EQ(config.getWeightDecay(), 0.0005f);
    EXPECT_EQ(config.getWarmupEpochs(), 3);
    EXPECT_EQ(config.getDevice(), "cuda:0");
    EXPECT_TRUE(config.useAMP());

    logger->info("All configuration values validated successfully");
}

TEST_F(Phase1IntegrationTest, ConfigurationWithYamlParser_InvalidYaml) {
    // Arrange - Load invalid YAML from test data
    // Search upward for WheelDL.Lib.Tests directory
    fs::path searchPath = fs::current_path();
    fs::path testProjectDir;

    for (int i = 0; i < 5; ++i) {
        fs::path candidatePath = searchPath / "WheelDL.Lib.Tests";
        if (fs::exists(candidatePath) && fs::is_directory(candidatePath)) {
            testProjectDir = candidatePath;
            break;
        }
        searchPath = searchPath.parent_path();
    }

    fs::path testConfigDir = testProjectDir / "Data" / "TestConfigs";
    fs::path invalidPath = testConfigDir / "invalid.yaml";

    ASSERT_TRUE(fs::exists(invalidPath)) << "Invalid test YAML not found: " << invalidPath;

    // Act & Assert
    logger->info("Testing invalid YAML handling");
    logger->info("  File: " + invalidPath.string());

    try {
        Config::Configuration config;
        config.loadFromYaml(invalidPath.string(), invalidPath.string());
        FAIL() << "Should have thrown exception for invalid YAML";
    }
    catch (const ConfigurationException& ex) {
        logger->info("Correctly caught ConfigurationException: " + std::string(ex.what()));
        EXPECT_EQ(ex.getErrorCode(), ErrorCode::CONFIG_PARSE_FAILED);
        SUCCEED();
    }
    catch (const std::exception& ex) {
        logger->info("Caught exception: " + std::string(ex.what()));
        SUCCEED();
    }
}

TEST_F(Phase1IntegrationTest, ConfigurationWithYamlParser_MultipleFormats) {
    // Test different YAML format variations from test data
    logger->info("Testing various YAML formats");

    // Search upward for WheelDL.Lib.Tests directory
    fs::path searchPath = fs::current_path();
    fs::path testProjectDir;

    for (int i = 0; i < 5; ++i) {
        fs::path candidatePath = searchPath / "WheelDL.Lib.Tests";
        if (fs::exists(candidatePath) && fs::is_directory(candidatePath)) {
            testProjectDir = candidatePath;
            break;
        }
        searchPath = searchPath.parent_path();
    }

    fs::path testConfigDir = testProjectDir / "Data" / "TestConfigs";

    std::vector<std::string> formatFiles = {
        "format1_device_string.yaml",
        "format2_device_list.yaml",
        "format3_cache_bool.yaml",
        "format4_cache_string.yaml"
    };

    for (const auto& filename : formatFiles) {
        fs::path yamlPath = testConfigDir / filename;

        ASSERT_TRUE(fs::exists(yamlPath)) << "Format test file not found: " << yamlPath;

        logger->info("Testing format: " + filename);

        Config::Configuration config;
        EXPECT_NO_THROW(config.loadFromYaml(yamlPath.string(), yamlPath.string()))
            << "Failed to load format: " << filename;

        logger->info("Successfully loaded format: " + filename);
    }
}

// ============================================================================
// GPU Memory Management Integration
// ============================================================================

TEST_F(Phase1IntegrationTest, GPUMemoryManagement_GuardAndManager)
{
    // Skip test if CUDA is not available
    if (!torch::cuda::is_available())
    {
        logger->info("CUDA not available, skipping GPU memory test");
        return;
    }

    logger->info("=== Testing GPU Memory Guard with Memory Manager ===");

    auto& memMgr = MemoryManager::getInstance();

    // Get initial state
    size_t initialUsed = memMgr.getGPUMemoryUsed();
    logger->info("Initial GPU memory used: " + std::to_string(initialUsed / (1024*1024)) + " MB");

    // Allocate memory with guard
    {
        profiler->start("gpu_allocation");
        logger->info("Allocating GPU memory with guard");

        GPUMemoryGuard guard;

        // Allocate some tensors
        std::vector<torch::Tensor> tensors;
        for (int i = 0; i < 10; ++i) {
            tensors.push_back(torch::randn({1024, 1024}, torch::kCUDA));
        }

        size_t peakUsed = memMgr.getGPUMemoryUsed();
        logger->info("Peak GPU memory used: " + std::to_string(peakUsed / (1024*1024)) + " MB");

        EXPECT_GT(peakUsed, initialUsed);

        profiler->stop("gpu_allocation");
        logger->info("GPU allocation completed in " +
                     std::to_string(profiler->getDuration("gpu_allocation")) + " ms");

        // Guard will clean up on scope exit
    }

    // Verify cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    size_t finalUsed = memMgr.getGPUMemoryUsed();
    logger->info("Final GPU memory used: " + std::to_string(finalUsed / (1024*1024)) + " MB");

    logger->info("=== GPU Memory Guard Test Completed ===");
}

TEST_F(Phase1IntegrationTest, GPUMemoryManagement_PoolAllocation) {
    if (!torch::cuda::is_available()) {
        logger->info("CUDA not available, skipping GPU memory test");
        return;
    }

    logger->info("=== Testing GPU Memory Pool ===");

    auto& memMgr = MemoryManager::getInstance();
    auto& pool = GPUMemoryPool::getInstance();

    pool.clear();

    // Allocate tensors from pool
    std::vector<torch::Tensor> batch;
    for (int i = 0; i < 10; ++i) {
        batch.push_back(pool.allocate({3, 224, 224}, torch::kFloat32, torch::kCUDA));
    }

    float memUsage = memMgr.getGPUMemoryUsagePercent();
    logger->info("GPU memory usage: " + std::to_string(memUsage) + "%");

    // Return tensors to pool
    for (auto& tensor : batch) {
        pool.release(tensor);
    }

    // Check pool statistics
    size_t poolSize = pool.getPoolSize();
    size_t totalAllocs = pool.getTotalAllocations();
    float cacheHitRate = pool.getCacheHitRate();

    logger->info("Pool statistics:");
    logger->info("  Pool size: " + std::to_string(poolSize));
    logger->info("  Total allocations: " + std::to_string(totalAllocs));
    logger->info("  Cache hit rate: " + std::to_string(cacheHitRate) + "%");

    EXPECT_GT(totalAllocs, 0);
    EXPECT_GE(cacheHitRate, 0.0f);
    EXPECT_LE(cacheHitRate, 100.0f);

    logger->info("=== GPU Memory Pool Test Completed ===");
}

TEST_F(Phase1IntegrationTest, GPUMemoryManagement_DynamicBatchSizer) {
    if (!torch::cuda::is_available()) {
        logger->info("CUDA not available, skipping GPU memory test");
        return;
    }

    logger->info("=== Testing Dynamic Batch Sizer ===");

    auto& memMgr = MemoryManager::getInstance();

    // Create dynamic batch sizer
    DynamicBatchSizer sizer(4, 128, 85.0f);

    logger->info("Min batch: " + std::to_string(sizer.getMinBatch()));
    logger->info("Max batch: " + std::to_string(sizer.getMaxBatch()));
    logger->info("Target memory: " + std::to_string(sizer.getTargetMemoryUsage()) + "%");

    // Get current memory usage
    float currentUsage = memMgr.getGPUMemoryUsagePercent();
    logger->info("Current GPU memory usage: " + std::to_string(currentUsage) + "%");

    // Test if adjustment is needed
    bool needsAdjustment = sizer.shouldAdjustBatchSize(currentUsage);
    logger->info("Needs adjustment: " + std::string(needsAdjustment ? "true" : "false"));

    // Compute optimal batch size (assuming 10MB per sample)
    size_t bytesPerSample = 10 * 1024 * 1024; // 10MB
    size_t optimalBatch = sizer.computeOptimalBatchSize(currentUsage, bytesPerSample);
    logger->info("Optimal batch size: " + std::to_string(optimalBatch));

    // Verify batch size is within bounds
    EXPECT_GE(optimalBatch, sizer.getMinBatch());
    EXPECT_LE(optimalBatch, sizer.getMaxBatch());

    logger->info("=== Dynamic Batch Sizer Test Completed ===");
}

TEST_F(Phase1IntegrationTest, GPUMemoryManagement_FullWorkflow) {
    if (!torch::cuda::is_available()) {
        logger->info("CUDA not available, skipping GPU memory test");
        return;
    }

    logger->info("=== Full GPU Memory Management Workflow ===");

    auto pipelineTimer = profiler->createScopedTimer("gpu_workflow");
    auto& memMgr = MemoryManager::getInstance();
    auto& pool = GPUMemoryPool::getInstance();

    pool.clear();

    // Step 1: Check available memory
    {
        auto timer = profiler->createScopedTimer("step1_memory_check");
        logger->info("Step 1: Checking available GPU memory");

        size_t totalMem = memMgr.getGPUMemoryTotal();
        size_t usedMem = memMgr.getGPUMemoryUsed();
        float usagePercent = memMgr.getGPUMemoryUsagePercent();

        logger->info("Total GPU memory: " + std::to_string(totalMem / (1024*1024)) + " MB");
        logger->info("Used GPU memory: " + std::to_string(usedMem / (1024*1024)) + " MB");
        logger->info("Usage: " + std::to_string(usagePercent) + "%");

        EXPECT_GT(totalMem, 0);
        EXPECT_GE(usagePercent, 0.0f);
        EXPECT_LE(usagePercent, 100.0f);
    }

    // Step 2: Initialize memory pool and batch sizer
    {
        auto timer = profiler->createScopedTimer("step2_initialization");
        logger->info("Step 2: Initializing memory pool and batch sizer");

        DynamicBatchSizer sizer(16, 256, 80.0f);

        // Pre-allocate some tensors to pool
        std::vector<torch::Tensor> preallocated;
        for (int i = 0; i < 10; ++i) {
            preallocated.push_back(pool.allocate({3, 224, 224}, torch::kFloat32, torch::kCUDA));
        }

        // Return to pool
        for (auto& tensor : preallocated) {
            pool.release(tensor);
        }

        logger->info("Pool pre-warmed with 10 tensors");
    }

    // Step 3: Simulate training with memory management
    {
        auto timer = profiler->createScopedTimer("step3_simulated_training");
        logger->info("Step 3: Simulating training with memory management");

        ThreadPool threadPool(4);
        DynamicBatchSizer sizer(32, 128, 85.0f);

        std::vector<std::future<void>> futures;

        for (int epoch = 0; epoch < 3; ++epoch) {
            logger->info("Epoch " + std::to_string(epoch));

            GPUMemoryGuard epochGuard;

            for (int batch = 0; batch < 10; ++batch) {
                futures.push_back(threadPool.enqueue([this, &pool, &memMgr, &sizer, epoch, batch]() {
                    // Get current memory usage and compute optimal batch size
                    float currentUsage = memMgr.getGPUMemoryUsagePercent();
                    size_t bytesPerSample = 5 * 1024 * 1024; // 5MB per sample estimate
                    size_t batchSize = sizer.computeOptimalBatchSize(currentUsage, bytesPerSample);

                    // Allocate batch from pool (limit to reasonable size)
                    std::vector<torch::Tensor> batchData;
                    size_t actualBatch = std::min(batchSize, static_cast<size_t>(10));
                    for (size_t i = 0; i < actualBatch; ++i) {
                        batchData.push_back(pool.allocate({3, 224, 224}, torch::kFloat32, torch::kCUDA));
                    }

                    // Simulate processing
                    torch::Tensor combined = torch::cat(batchData, 0);
                    torch::Tensor result = combined * 2.0f;

                    // Return to pool
                    for (auto& tensor : batchData) {
                        pool.release(tensor);
                    }

                    // Check if we should adjust batch size
                    float updatedUsage = memMgr.getGPUMemoryUsagePercent();
                    if (sizer.shouldAdjustBatchSize(updatedUsage)) {
                        size_t newOptimalBatch = sizer.computeOptimalBatchSize(updatedUsage, bytesPerSample);
                        logger->info("Adjusted batch size to: " + std::to_string(newOptimalBatch));
                    }
                }));
            }

            // Wait for epoch to complete
            for (auto& f : futures) {
                f.wait();
            }
            futures.clear();

            logger->info("Epoch " + std::to_string(epoch) + " completed");

            // Epoch guard will clean up cache
        }
    }

    // Step 4: Report statistics
    {
        auto timer = profiler->createScopedTimer("step4_reporting");
        logger->info("Step 4: Reporting statistics");

        size_t finalUsed = memMgr.getGPUMemoryUsed();
        float finalUsagePercent = memMgr.getGPUMemoryUsagePercent();

        size_t poolSize = pool.getPoolSize();
        size_t totalAllocs = pool.getTotalAllocations();
        float cacheHitRate = pool.getCacheHitRate();

        logger->info("Final GPU memory used: " + std::to_string(finalUsed / (1024*1024)) + " MB");
        logger->info("Final usage: " + std::to_string(finalUsagePercent) + "%");
        logger->info("Pool size: " + std::to_string(poolSize));
        logger->info("Total allocations: " + std::to_string(totalAllocs));
        logger->info("Cache hit rate: " + std::to_string(cacheHitRate) + "%");

        // Generate performance report
        std::string report = profiler->report();
        logger->info("\n" + report);

        EXPECT_GT(totalAllocs, 0);
        EXPECT_GE(cacheHitRate, 0.0f);
    }

    logger->info("=== Full GPU Memory Management Workflow Completed ===");
}
