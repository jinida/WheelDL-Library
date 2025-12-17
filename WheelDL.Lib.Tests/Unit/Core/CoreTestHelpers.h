#pragma once

/**
 * @file CoreTestHelpers.h
 * @brief Test infrastructure for Core/Utils and Core/Callback tests
 *
 * Provides mock helpers, test fixtures, and utilities for testing:
 * - Checkpoint and CheckpointMetadata
 * - CallbackQueue, AsyncCallbackQueue, CallbackThrottler
 *
 * Phase 1 of test_core_utils.md
 */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <mutex>

// WheelDL includes
#include "Config/Configuration.h"
#include "Utils/Common/Types.h"
#include "Utils/Logger/Logger.h"
#include "Model/Task/BaseModel.h"
#include "Model/Loss/BaseLoss.h"
#include "Core/Utils/Checkpoint.h"
#include "Core/Utils/CheckpointMetadata.h"

namespace fs = std::filesystem;

namespace WheelDL {
namespace Test {
namespace Core {

// ============================================================================
// TempFileManager - Manages temporary files and directories for tests
// ============================================================================

/**
 * @class TempFileManager
 * @brief RAII manager for temporary files and directories
 *
 * Creates a unique temp directory on construction and cleans it up on destruction.
 * Use for checkpoint file tests that need file system access.
 */
class TempFileManager {
public:
    TempFileManager(const std::string& prefix = "WheelDL_CoreTest") {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        _tempDir = (fs::temp_directory_path() / (prefix + "_" + std::to_string(timestamp))).string();
        fs::create_directories(_tempDir);
    }

    ~TempFileManager() {
        cleanup();
    }

    // Non-copyable
    TempFileManager(const TempFileManager&) = delete;
    TempFileManager& operator=(const TempFileManager&) = delete;

    /**
     * @brief Get the temp directory path
     */
    std::string getTempDir() const { return _tempDir; }

    /**
     * @brief Get a unique file path within the temp directory
     * @param filename File name (e.g., "checkpoint.pt")
     */
    std::string getFilePath(const std::string& filename) const {
        return (fs::path(_tempDir) / filename).string();
    }

    /**
     * @brief Get a unique numbered file path
     * @param extension File extension (e.g., ".pt")
     */
    std::string getUniqueFilePath(const std::string& extension = ".pt") {
        return (fs::path(_tempDir) / ("file_" + std::to_string(++_counter) + extension)).string();
    }

    /**
     * @brief Create a subdirectory in the temp directory
     * @param subdir Subdirectory name
     * @return Path to the created subdirectory
     */
    std::string createSubDir(const std::string& subdir) {
        auto path = fs::path(_tempDir) / subdir;
        fs::create_directories(path);
        return path.string();
    }

    /**
     * @brief Create a file with content
     * @param filename File name
     * @param content File content
     * @return Path to the created file
     */
    std::string createFile(const std::string& filename, const std::string& content) {
        auto path = getFilePath(filename);
        std::ofstream file(path, std::ios::binary);
        file << content;
        file.close();
        return path;
    }

    /**
     * @brief Create an empty file
     */
    std::string createEmptyFile(const std::string& filename) {
        return createFile(filename, "");
    }

    /**
     * @brief Create a corrupted checkpoint file
     */
    std::string createCorruptedCheckpoint(const std::string& filename = "corrupted.pt") {
        return createFile(filename, "THIS_IS_NOT_A_VALID_CHECKPOINT_FILE_12345");
    }

    /**
     * @brief Clean up all temp files
     */
    void cleanup() {
        try {
            if (fs::exists(_tempDir)) {
                fs::remove_all(_tempDir);
            }
        }
        catch (...) {}
    }

private:
    std::string _tempDir;
    int _counter = 0;
};

// ============================================================================
// LogCapture - Captures log messages for verification (composition, not inheritance)
// ============================================================================

/**
 * @struct LogEntry
 * @brief Represents a captured log entry
 */
struct LogEntry {
    std::string level;
    std::string tag;
    std::string message;
};

/**
 * @class LogCapture
 * @brief Captures log messages for test verification
 *
 * Since Logger cannot be inherited (private constructor, no virtual methods),
 * this class wraps a real Logger and provides capture functionality.
 */
class LogCapture {
public:
    LogCapture(const std::string& name = "TestLogger")
        : _logger(WheelDL::Utils::Logger::create(name))
    {}

    WheelDL::Utils::Logger* get() { return _logger.get(); }

    void info(const std::string& tag, const std::string& message) {
        std::lock_guard<std::mutex> lock(_mutex);
        _entries.push_back({"INFO", tag, message});
        _logger->info(tag, message);
    }

    void warn(const std::string& tag, const std::string& message) {
        std::lock_guard<std::mutex> lock(_mutex);
        _entries.push_back({"WARN", tag, message});
        _logger->warn(tag, message);
    }

    void error(const std::string& tag, const std::string& message) {
        std::lock_guard<std::mutex> lock(_mutex);
        _entries.push_back({"ERROR", tag, message});
        _logger->error(tag, message);
    }

    void debug(const std::string& tag, const std::string& message) {
        std::lock_guard<std::mutex> lock(_mutex);
        _entries.push_back({"DEBUG", tag, message});
        _logger->debug(tag, message);
    }

    /**
     * @brief Get all captured log entries
     */
    std::vector<LogEntry> getEntries() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _entries;
    }

    /**
     * @brief Check if any log message contains the given text
     */
    bool hasMessageContaining(const std::string& text) const {
        std::lock_guard<std::mutex> lock(_mutex);
        for (const auto& entry : _entries) {
            if (entry.message.find(text) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Check if a specific level log was made
     */
    bool hasLogLevel(const std::string& level) const {
        std::lock_guard<std::mutex> lock(_mutex);
        for (const auto& entry : _entries) {
            if (entry.level == level) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Get count of logs at a specific level
     */
    size_t countLogLevel(const std::string& level) const {
        std::lock_guard<std::mutex> lock(_mutex);
        size_t count = 0;
        for (const auto& entry : _entries) {
            if (entry.level == level) {
                count++;
            }
        }
        return count;
    }

    /**
     * @brief Clear all captured entries
     */
    void clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        _entries.clear();
    }

private:
    std::unique_ptr<WheelDL::Utils::Logger> _logger;
    mutable std::mutex _mutex;
    std::vector<LogEntry> _entries;
};

// ============================================================================
// MockLoss - Simple loss for MockBaseModel
// ============================================================================

class MockLoss : public Model::Loss::BaseLoss {
public:
    MockLoss() = default;

    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const Data::Dataset::DataExample& target) override
    {
        auto device = prediction.device();
        auto loss = torch::tensor(0.5f, torch::TensorOptions().device(device));
        return {{"total", loss}, {"cls", loss}};
    }

    [[nodiscard]] std::string name() const override { return "MockLoss"; }
};

// ============================================================================
// MockBaseModel - Test model for checkpoint tests
// ============================================================================

/**
 * @class MockBaseModel
 * @brief Mock implementation of BaseModel for testing
 *
 * Provides minimal implementation for checkpoint tests.
 * Only overrides pure virtual method initCriterion().
 */
class MockBaseModel : public Model::BaseModel {
public:
    MockBaseModel() : Model::BaseModel() {
        // Create a simple sequential model
        _model = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)),
            torch::nn::BatchNorm2d(16),
            torch::nn::ReLU(),
            torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})),
            torch::nn::Flatten(),
            torch::nn::Linear(16, 10)
        );

        // Register modules
        for (size_t i = 0; i < _model->size(); ++i) {
            register_module("layer_" + std::to_string(i), _model->ptr(i));
        }

        // Initialize stride tensor
        _stride = torch::tensor({8.0f, 16.0f, 32.0f});
        _taskType = TaskType::CLASSIFICATION;
        _isInitialized = true;
    }

protected:
    std::unique_ptr<Model::Loss::BaseLoss> initCriterion() override {
        return std::make_unique<MockLoss>();
    }
};

// ============================================================================
// MetricsData Factory - Creates test MetricsData objects
// ============================================================================

/**
 * @class MetricsFactory
 * @brief Factory for creating test MetricsData objects
 */
class MetricsFactory {
public:
    /**
     * @brief Create default MetricsData with all zeros
     */
    static MetricsData createDefault() {
        return MetricsData{};
    }

    /**
     * @brief Create MetricsData with typical values
     */
    static MetricsData createTypical() {
        MetricsData m;
        m.loss = 0.5f;
        m.accuracy = 0.85f;
        m.precision = 0.82f;
        m.recall = 0.78f;
        m.f1Score = 0.80f;
        m.mAP = 0.75f;
        m.fitness = 0.80f;
        m.threshold = 0.5f;
        m.aucROC = 0.88f;
        return m;
    }

    /**
     * @brief Create MetricsData with perfect scores
     */
    static MetricsData createPerfect() {
        MetricsData m;
        m.loss = 0.0f;
        m.accuracy = 1.0f;
        m.precision = 1.0f;
        m.recall = 1.0f;
        m.f1Score = 1.0f;
        m.mAP = 1.0f;
        m.fitness = 1.0f;
        m.threshold = 0.5f;
        m.aucROC = 1.0f;
        return m;
    }

    /**
     * @brief Create MetricsData with custom fitness
     */
    static MetricsData createWithFitness(float fitness) {
        MetricsData m = createTypical();
        m.fitness = fitness;
        return m;
    }
};

// ============================================================================
// ProgressData Factory - Creates test ProgressData objects
// ============================================================================

/**
 * @class ProgressFactory
 * @brief Factory for creating test ProgressData objects
 */
class ProgressFactory {
public:
    /**
     * @brief Create default ProgressData
     */
    static ProgressData createDefault() {
        ProgressData p;
        p.stage = ProgressStage::TRAIN_BATCH;
        p.currentEpoch = 0;
        p.totalEpochs = 100;
        p.currentBatch = 0;
        p.totalBatches = 100;
        p.loss = 0.0f;
        p.metrics = MetricsFactory::createDefault();
        p.learningRate = 0.01f;
        p.gpuMemoryUsage = 0.0f;
        p.elapsedTime = 0.0;
        p.eta = 0.0;
        p.message = "";
        return p;
    }

    /**
     * @brief Create ProgressData for a specific epoch/batch
     */
    static ProgressData createAt(int epoch, int batch, int totalEpochs = 100, int totalBatches = 100) {
        ProgressData p = createDefault();
        p.currentEpoch = epoch;
        p.currentBatch = batch;
        p.totalEpochs = totalEpochs;
        p.totalBatches = totalBatches;
        return p;
    }

    /**
     * @brief Create ProgressData with train batch stage
     */
    static ProgressData createTrainBatch(int epoch, int batch, float loss) {
        ProgressData p = createAt(epoch, batch);
        p.stage = ProgressStage::TRAIN_BATCH;
        p.loss = loss;
        return p;
    }

    /**
     * @brief Create ProgressData with train epoch stage
     */
    static ProgressData createTrainEpoch(int epoch, const MetricsData& metrics) {
        ProgressData p = createAt(epoch, 0);
        p.stage = ProgressStage::TRAIN_EPOCH;
        p.metrics = metrics;
        return p;
    }

    /**
     * @brief Create ProgressData with validation epoch stage
     */
    static ProgressData createValEpoch(int epoch, const MetricsData& metrics) {
        ProgressData p = createAt(epoch, 0);
        p.stage = ProgressStage::VAL_EPOCH;
        p.metrics = metrics;
        return p;
    }
};

// ============================================================================
// Configuration Factory for Core Tests
// ============================================================================

/**
 * @class CoreConfigFactory
 * @brief Factory for creating test Configuration objects for Core tests
 */
class CoreConfigFactory {
public:
    /**
     * @brief Create a minimal Configuration for checkpoint tests
     */
    static std::shared_ptr<Config::Configuration> createMinimal() {
        TempFileManager temp("CoreConfig");

        // Create minimal dataset JSON
        std::string datasetPath = temp.createFile("dataset.json", R"({
    "header": {
        "type": "classification",
        "categories": ["class0", "class1", "class2", "class3", "class4",
                       "class5", "class6", "class7", "class8", "class9"]
    },
    "annotations": []
})");

        // Create minimal hyp YAML
        std::string hypPath = temp.createFile("hyp.yaml", R"(
epochs: 100
patience: 50
batch_size: 16
image_size: 224
device: "0"
workers: 0
optimizer: auto
seed: 42
deterministic: true
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 7.5
cls: 0.5
dfl: 1.5
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0
degrees: 0.0
translate: 0.0
scale: 0.0
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.0
mosaic: 0.0
cos_lr: false
amp: false
)");

        auto config = std::make_shared<Config::Configuration>();
        config->load("Config/Valid/model/cls_ConvNext.yaml", hypPath, datasetPath);

        return config;
    }

    /**
     * @brief Create Configuration with specific device string
     */
    static std::shared_ptr<Config::Configuration> createWithDevice(const std::string& device) {
        TempFileManager temp("CoreConfigDevice");

        std::string datasetPath = temp.createFile("dataset.json", R"({
    "header": {
        "type": "classification",
        "categories": ["class0", "class1", "class2"]
    },
    "annotations": []
})");

        std::string deviceStr = device.empty() ? "\"\"" : ("\"" + device + "\"");
        std::string hypPath = temp.createFile("hyp.yaml",
            "epochs: 100\n"
            "batch_size: 16\n"
            "image_size: 224\n"
            "device: " + deviceStr + "\n"
            "workers: 0\n"
            "optimizer: SGD\n"
            "lr0: 0.01\n"
            "lrf: 0.01\n"
            "momentum: 0.937\n"
            "weight_decay: 0.0005\n"
            "warmup_epochs: 3.0\n"
            "cos_lr: true\n"
            "amp: true\n"
        );

        auto config = std::make_shared<Config::Configuration>();
        config->load("Config/Valid/model/cls_ConvNext.yaml", hypPath, datasetPath);

        return config;
    }

    /**
     * @brief Create Configuration with multi-GPU device string
     */
    static std::shared_ptr<Config::Configuration> createMultiGPU() {
        return createWithDevice("0,1,2");
    }

    /**
     * @brief Create Configuration for CPU
     */
    static std::shared_ptr<Config::Configuration> createCPU() {
        return createWithDevice("cpu");
    }
};

// ============================================================================
// CUDA Test Fixture Base
// ============================================================================

/**
 * @class CUDATestFixture
 * @brief Base fixture for tests requiring CUDA
 */
class CUDATestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        _hasCuda = torch::cuda::is_available();
        if (_hasCuda) {
            _device = torch::Device(torch::kCUDA, 0);
            // Clear CUDA cache before each test
            torch::cuda::synchronize();
        }
        else {
            _device = torch::kCPU;
        }
    }

    void TearDown() override {
        if (_hasCuda) {
            torch::cuda::synchronize();
        }
    }

    bool hasCuda() const { return _hasCuda; }
    torch::Device device() const { return _device; }

    /**
     * @brief Skip test if CUDA not available (for gtest 1.8.1.7)
     * Since GTEST_SKIP is not available, we use SUCCEED() with early return
     */
    bool requireCuda() {
        if (!_hasCuda) {
            SUCCEED() << "CUDA not available, test skipped";
            return false;
        }
        return true;
    }

private:
    bool _hasCuda = false;
    torch::Device _device = torch::kCPU;
};

// ============================================================================
// Checkpoint Test Fixture
// ============================================================================

/**
 * @class CheckpointTestFixture
 * @brief Fixture for Checkpoint tests with common setup
 */
class CheckpointTestFixture : public CUDATestFixture {
protected:
    void SetUp() override {
        CUDATestFixture::SetUp();

        _tempManager = std::make_unique<TempFileManager>("CheckpointTest");
        _config = CoreConfigFactory::createMinimal();
        _logCapture = std::make_unique<LogCapture>("CheckpointTestLogger");
    }

    void TearDown() override {
        _tempManager.reset();
        CUDATestFixture::TearDown();
    }

    /**
     * @brief Create a mock model for testing
     */
    std::unique_ptr<MockBaseModel> createMockModel() {
        return std::make_unique<MockBaseModel>();
    }

    /**
     * @brief Create an SGD optimizer for the model
     */
    std::unique_ptr<torch::optim::SGD> createOptimizer(MockBaseModel& model) {
        return std::make_unique<torch::optim::SGD>(
            model.parameters(),
            torch::optim::SGDOptions(0.01).momentum(0.9)
        );
    }

    /**
     * @brief Create test metadata
     */
    WheelDL::Core::Utils::CheckpointMetadata createTestMetadata(int epoch = 10) {
        auto metrics = MetricsFactory::createTypical();
        return WheelDL::Core::Utils::CheckpointMetadata::fromConfiguration(
            *_config, epoch, metrics);
    }

    /**
     * @brief Save a test checkpoint and return its path
     */
    std::string saveTestCheckpoint(const std::string& filename = "test.pt") {
        auto model = createMockModel();
        auto optimizer = createOptimizer(*model);
        auto metadata = createTestMetadata();

        std::string path = _tempManager->getFilePath(filename);
        WheelDL::Core::Utils::Checkpoint::save(path, *model, *optimizer, metadata);

        return path;
    }

    /**
     * @brief Save a model-only checkpoint and return its path
     */
    std::string saveModelOnlyCheckpoint(const std::string& filename = "model_only.pt") {
        auto model = createMockModel();
        auto metadata = createTestMetadata();

        std::string path = _tempManager->getFilePath(filename);
        WheelDL::Core::Utils::Checkpoint::saveModelOnly(path, *model, metadata);

        return path;
    }

    TempFileManager& tempManager() { return *_tempManager; }
    std::shared_ptr<Config::Configuration> config() { return _config; }
    LogCapture& logCapture() { return *_logCapture; }
    WheelDL::Utils::Logger* logger() { return _logCapture->get(); }

private:
    std::unique_ptr<TempFileManager> _tempManager;
    std::shared_ptr<Config::Configuration> _config;
    std::unique_ptr<LogCapture> _logCapture;
};

// ============================================================================
// Callback Test Helpers
// ============================================================================

/**
 * @class CallbackTracker
 * @brief Tracks callback invocations for testing
 */
class CallbackTracker {
public:
    /**
     * @brief Get a callback that records invocations
     */
    ProgressCallback getCallback() {
        return [this](const ProgressData& data) {
            std::lock_guard<std::mutex> lock(_mutex);
            _invocations.push_back(data);
        };
    }

    /**
     * @brief Get a callback that throws std::exception
     */
    ProgressCallback getThrowingCallback() {
        return [](const ProgressData&) {
            throw std::runtime_error("Test exception");
        };
    }

    /**
     * @brief Get a callback that throws unknown exception
     */
    ProgressCallback getUnknownThrowingCallback() {
        return [](const ProgressData&) {
            throw 42;  // Non-standard exception
        };
    }

    /**
     * @brief Get count of invocations
     */
    size_t count() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _invocations.size();
    }

    /**
     * @brief Get all recorded invocations
     */
    std::vector<ProgressData> getInvocations() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _invocations;
    }

    /**
     * @brief Clear all recorded invocations
     */
    void clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        _invocations.clear();
    }

    /**
     * @brief Wait for a specific number of invocations
     * @param count Expected count
     * @param timeoutMs Timeout in milliseconds
     * @return true if count reached, false on timeout
     */
    bool waitForCount(size_t targetCount, int timeoutMs = 5000) {
        auto start = std::chrono::steady_clock::now();
        while (count() < targetCount) {
            if (std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start).count() > timeoutMs) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return true;
    }

private:
    mutable std::mutex _mutex;
    std::vector<ProgressData> _invocations;
};

} // namespace Core
} // namespace Test
} // namespace WheelDL
