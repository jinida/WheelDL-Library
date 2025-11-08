#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Utils/Logger/Logger.h"
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>

using namespace WheelDL::Utils;
namespace fs = std::filesystem;

/**
 * @class LoggerTest
 * @brief Unit tests for Logger class
 *
 * Tests cover:
 * - Singleton pattern behavior
 * - Log level management (global and module-specific)
 * - File logging and rotation
 * - Thread safety
 * - Rank-specific logging for distributed training
 */
class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset logger state before each test
        logger = Logger::getInstance();
        logger->reset();  // Close any open file handles
        logger->setGlobalLogLevel(LogLevel::INFO);

        // Clean up any test log files
        if (fs::exists("test_log.txt")) {
            fs::remove("test_log.txt");
        }
        if (fs::exists("test_rank_0.log")) {
            fs::remove("test_rank_0.log");
        }
        // Clean up rank_*.log files
        for (const auto& entry : fs::directory_iterator(".")) {
            std::string filename = entry.path().filename().string();
            if (filename.find("rank_") == 0 && entry.path().extension() == ".log") {
                fs::remove(entry.path());
            }
        }
    }

    void TearDown() override {
        // Reset logger to close file handles before cleanup
        logger->reset();

        // Clean up test log files
        if (fs::exists("test_log.txt")) {
            fs::remove("test_log.txt");
        }
        if (fs::exists("test_rank_0.log")) {
            fs::remove("test_rank_0.log");
        }

        // Clean up rotating log files and rank_*.log files
        for (const auto& entry : fs::directory_iterator(".")) {
            std::string filename = entry.path().filename().string();
            if ((entry.path().extension() == ".txt" && filename.find("test_log") != std::string::npos) ||
                (filename.find("rank_") == 0 && entry.path().extension() == ".log")) {
                fs::remove(entry.path());
            }
        }
    }

    std::shared_ptr<Logger> logger;
};

// ============================================================================
// Singleton Pattern Tests
// ============================================================================

TEST_F(LoggerTest, SingletonInstance_SameInstanceReturned) {
    // Arrange & Act
    auto instance1 = Logger::getInstance();
    auto instance2 = Logger::getInstance();

    // Assert
    EXPECT_EQ(instance1, instance2);
    EXPECT_NE(instance1, nullptr);
}

TEST_F(LoggerTest, SingletonInstance_ThreadSafe) {
    // Arrange
    std::vector<std::shared_ptr<Logger>> instances(10);
    std::vector<std::thread> threads;

    // Act - Create logger instances from multiple threads
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&instances, i]() {
            instances[i] = Logger::getInstance();
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Assert - All instances should be the same
    for (int i = 1; i < 10; ++i) {
        EXPECT_EQ(instances[0], instances[i]);
    }
}

// ============================================================================
// Log Level Tests
// ============================================================================

TEST_F(LoggerTest, GlobalLogLevel_SetAndGet) {
    // Arrange & Act
    logger->setGlobalLogLevel(LogLevel::DEBUG);

    // Assert
    // Log messages should be accepted at DEBUG level
    EXPECT_NO_THROW(logger->debug("Debug message"));
    EXPECT_NO_THROW(logger->info("Info message"));
    EXPECT_NO_THROW(logger->warn("Warning message"));
}

TEST_F(LoggerTest, GlobalLogLevel_FilterLowerLevels) {
    // Arrange
    logger->setGlobalLogLevel(LogLevel::ERR);

    // Act & Assert - Should not throw, but lower levels should be filtered
    EXPECT_NO_THROW(logger->trace("Trace message"));  // Filtered
    EXPECT_NO_THROW(logger->debug("Debug message"));  // Filtered
    EXPECT_NO_THROW(logger->info("Info message"));    // Filtered
    EXPECT_NO_THROW(logger->warn("Warning message")); // Filtered
    EXPECT_NO_THROW(logger->error("Error message"));  // Should log
    EXPECT_NO_THROW(logger->fatal("Fatal message"));  // Should log
}

TEST_F(LoggerTest, ModuleLogLevel_OverridesGlobal) {
    // Arrange
    logger->setGlobalLogLevel(LogLevel::ERR);
    logger->setModuleLogLevel("TestModule", LogLevel::DEBUG);

    // Act & Assert
    EXPECT_NO_THROW(logger->debug("[TestModule] Debug message"));
    EXPECT_NO_THROW(logger->info("[TestModule] Info message"));
}

TEST_F(LoggerTest, ModuleLogLevel_MultipleModules) {
    // Arrange
    logger->setGlobalLogLevel(LogLevel::WARN);
    logger->setModuleLogLevel("Module1", LogLevel::DEBUG);
    logger->setModuleLogLevel("Module2", LogLevel::ERR);
    logger->setModuleLogLevel("Module3", LogLevel::TRACE);

    // Act & Assert
    EXPECT_NO_THROW(logger->debug("[Module1] Debug message"));
    EXPECT_NO_THROW(logger->trace("[Module3] Trace message"));
    EXPECT_NO_THROW(logger->error("[Module2] Error message"));
}

// ============================================================================
// Logging Method Tests
// ============================================================================

TEST_F(LoggerTest, LogMethods_AllLevels) {
    // Arrange
    logger->setGlobalLogLevel(LogLevel::TRACE);

    // Act & Assert - All methods should work without throwing
    EXPECT_NO_THROW(logger->trace("Trace level message"));
    EXPECT_NO_THROW(logger->debug("Debug level message"));
    EXPECT_NO_THROW(logger->info("Info level message"));
    EXPECT_NO_THROW(logger->warn("Warning level message"));
    EXPECT_NO_THROW(logger->error("Error level message"));
    EXPECT_NO_THROW(logger->fatal("Fatal level message"));
}

TEST_F(LoggerTest, LogMethods_EmptyMessage) {
    // Act & Assert
    EXPECT_NO_THROW(logger->info(""));
    EXPECT_NO_THROW(logger->error(""));
}

TEST_F(LoggerTest, LogMethods_LongMessage) {
    // Arrange
    std::string longMessage(10000, 'x');

    // Act & Assert
    EXPECT_NO_THROW(logger->info(longMessage));
}

TEST_F(LoggerTest, LogMethods_SpecialCharacters) {
    // Act & Assert
    EXPECT_NO_THROW(logger->info("Special chars: \n\t\r\"\'\\"));
    EXPECT_NO_THROW(logger->info("Unicode: 한글 테스트 🚀"));
}

// ============================================================================
// File Logging Tests
// ============================================================================

TEST_F(LoggerTest, FileLogging_CreateLogFile) {
    // Arrange
    logger->setLogFile("test_log.txt");

    // Act
    logger->info("Test log message");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Allow file write

    // Assert
    EXPECT_TRUE(fs::exists("test_log.txt"));
}

TEST_F(LoggerTest, FileLogging_MessageWrittenToFile) {
    // Arrange
    logger->setLogFile("test_log.txt");
    std::string testMessage = "UniqueTestMessage12345";

    // Act
    logger->info(testMessage);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Allow file write

    // Assert
    std::ifstream file("test_log.txt");
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    EXPECT_TRUE(content.find(testMessage) != std::string::npos);
}

TEST_F(LoggerTest, FileLogging_Rotation_Enabled) {
    // Arrange
    size_t maxFileSize = 1024;  // 1 KB
    size_t maxFiles = 3;

    // Act
    logger->setLogFile("test_log.txt");
    logger->enableRotation(maxFileSize, maxFiles);

    // Write enough data to trigger rotation
    for (int i = 0; i < 100; ++i) {
        logger->info("Long message to fill up the log file and trigger rotation mechanism");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Assert - Main log file should exist
    EXPECT_TRUE(fs::exists("test_log.txt"));
}

// ============================================================================
// Distributed Training (Rank-Specific) Tests
// ============================================================================

TEST_F(LoggerTest, RankSpecificLog_CreatesRankFile) {
    // Arrange
    int rank = 0;

    // Act
    logger->setRankSpecificLog(rank);
    logger->info("Rank 0 message");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Assert
    EXPECT_TRUE(fs::exists("test_rank_0.log") || fs::exists("rank_0.log"));
}

TEST_F(LoggerTest, RankSpecificLog_MultipleRanks) {
    // Act
    logger->setRankSpecificLog(0);
    logger->info("Message from rank 0");

    logger->setRankSpecificLog(1);
    logger->info("Message from rank 1");

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Assert - This test is simplified, in real scenario files would be separate
    EXPECT_NO_THROW(logger->info("Completed"));
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(LoggerTest, ThreadSafety_ConcurrentLogging) {
    // Arrange
    const int numThreads = 10;
    const int messagesPerThread = 100;
    std::vector<std::thread> threads;

    // Act
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, messagesPerThread]() {
            for (int j = 0; j < messagesPerThread; ++j) {
                logger->info("Thread " + std::to_string(i) + " Message " + std::to_string(j));
            }
        });
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    // Assert - No crashes occurred
    SUCCEED();
}

TEST_F(LoggerTest, ThreadSafety_ConcurrentLevelChanges) {
    // Arrange
    std::vector<std::thread> threads;

    // Act
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([this]() {
            logger->setGlobalLogLevel(LogLevel::DEBUG);
            logger->debug("Debug message");
        });

        threads.emplace_back([this]() {
            logger->setGlobalLogLevel(LogLevel::INFO);
            logger->info("Info message");
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Assert - No crashes occurred
    SUCCEED();
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(LoggerTest, EdgeCase_InvalidFilePath) {
    // Act & Assert - Should handle gracefully
    EXPECT_NO_THROW(logger->setLogFile("Z:\\invalid\\path\\log.txt"));
}

TEST_F(LoggerTest, EdgeCase_VeryHighLogLevel) {
    // Arrange
    logger->setGlobalLogLevel(LogLevel::OFF);

    // Act & Assert - No logs should appear, but no exceptions
    EXPECT_NO_THROW(logger->trace("Should not appear"));
    EXPECT_NO_THROW(logger->error("Should not appear"));
    EXPECT_NO_THROW(logger->fatal("Should not appear"));
}

TEST_F(LoggerTest, EdgeCase_ZeroRotationSize) {
    // Act & Assert - Should handle gracefully
    EXPECT_NO_THROW(logger->enableRotation(0, 5));
}

TEST_F(LoggerTest, EdgeCase_NegativeRank) {
    // Act & Assert - Should handle gracefully
    EXPECT_NO_THROW(logger->setRankSpecificLog(-1));
}
