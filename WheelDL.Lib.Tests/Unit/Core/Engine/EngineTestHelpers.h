/**
 * @file EngineTestHelpers.h
 * @brief Test infrastructure for Core/Engine tests
 *
 * Provides mock helpers, test fixtures, and utilities for testing:
 * - BasePredictor, BaseValidator, BaseTrainer
 * - Task-specific Predictor/Validator/Trainer classes
 *
 * Phase 1 of test_engine_and_etc.md
 */

#pragma once

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
#include <functional>

// WheelDL includes
#include "Config/Configuration.h"
#include "Utils/Common/Types.h"
#include "Utils/Logger/Logger.h"
#include "Utils/Profiler/PerformanceProfiler.h"
#include "Utils/Workspace/Workspace.h"
#include "Utils/Error/WheelLibException.h"
#include "Model/Task/BaseModel.h"
#include "Model/Loss/BaseLoss.h"
#include "Data/Dataset/BaseDataset.h"
#include "Core/Engine/BasePredictor.h"
#include "Core/Engine/BaseValidator.h"
#include "Core/Engine/BaseTrainer.h"
#include "Core/Utils/Checkpoint.h"
#include "Core/Utils/CheckpointMetadata.h"

namespace fs = std::filesystem;

namespace WheelDL {
namespace Test {
namespace Engine {

// ============================================================================
// TempFileManager - Manages temporary files and directories for tests
// ============================================================================

class TempFileManager {
public:
    TempFileManager(const std::string& prefix = "WheelDL_EngineTest") {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        _tempDir = (fs::temp_directory_path() / (prefix + "_" + std::to_string(timestamp))).string();
        fs::create_directories(_tempDir);
    }

    ~TempFileManager() {
        cleanup();
    }

    TempFileManager(const TempFileManager&) = delete;
    TempFileManager& operator=(const TempFileManager&) = delete;

    std::string getTempDir() const { return _tempDir; }

    std::string getFilePath(const std::string& filename) const {
        return (fs::path(_tempDir) / filename).string();
    }

    std::string getUniqueFilePath(const std::string& extension = ".pt") {
        return (fs::path(_tempDir) / ("file_" + std::to_string(++_counter) + extension)).string();
    }

    std::string createSubDir(const std::string& subdir) {
        auto path = fs::path(_tempDir) / subdir;
        fs::create_directories(path);
        return path.string();
    }

    std::string createFile(const std::string& filename, const std::string& content) {
        auto path = getFilePath(filename);
        std::ofstream file(path, std::ios::binary);
        file << content;
        file.close();
        return path;
    }

    std::string createEmptyFile(const std::string& filename) {
        return createFile(filename, "");
    }

    std::string createCorruptedCheckpoint(const std::string& filename = "corrupted.pt") {
        return createFile(filename, "THIS_IS_NOT_A_VALID_CHECKPOINT_FILE_12345");
    }

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
// MockLogger - Captures log messages for verification
// ============================================================================

struct LogEntry {
    std::string level;
    std::string tag;
    std::string message;
};

class MockLogger {
public:
    MockLogger(const std::string& name = "EngineTestLogger")
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

    std::vector<LogEntry> getEntries() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _entries;
    }

    bool hasMessageContaining(const std::string& text) const {
        std::lock_guard<std::mutex> lock(_mutex);
        for (const auto& entry : _entries) {
            if (entry.message.find(text) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    bool hasLogLevel(const std::string& level) const {
        std::lock_guard<std::mutex> lock(_mutex);
        for (const auto& entry : _entries) {
            if (entry.level == level) {
                return true;
            }
        }
        return false;
    }

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
// MockProfiler - Wraps PerformanceProfiler with tracking
// ============================================================================

class MockProfiler {
public:
    MockProfiler()
        : _profiler(WheelDL::Utils::PerformanceProfiler::create())
    {}

    WheelDL::Utils::PerformanceProfiler* get() { return _profiler.get(); }

    void start(const std::string& name) {
        std::lock_guard<std::mutex> lock(_mutex);
        _startCalls.push_back(name);
        _profiler->start(name);
    }

    void stop(const std::string& name) {
        std::lock_guard<std::mutex> lock(_mutex);
        _stopCalls.push_back(name);
        _profiler->stop(name);
    }

    bool wasStarted(const std::string& name) const {
        std::lock_guard<std::mutex> lock(_mutex);
        return std::find(_startCalls.begin(), _startCalls.end(), name) != _startCalls.end();
    }

    bool wasStopped(const std::string& name) const {
        std::lock_guard<std::mutex> lock(_mutex);
        return std::find(_stopCalls.begin(), _stopCalls.end(), name) != _stopCalls.end();
    }

    size_t startCount() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _startCalls.size();
    }

    size_t stopCount() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _stopCalls.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        _startCalls.clear();
        _stopCalls.clear();
        _profiler->reset();
    }

private:
    std::unique_ptr<WheelDL::Utils::PerformanceProfiler> _profiler;
    mutable std::mutex _mutex;
    std::vector<std::string> _startCalls;
    std::vector<std::string> _stopCalls;
};

// ============================================================================
// MockWorkspace - Test workspace with temp directories
// ============================================================================

class MockWorkspace {
public:
    MockWorkspace()
        : _tempManager("MockWorkspace")
    {
        _root = _tempManager.getTempDir();
        _weightsDir = _tempManager.createSubDir("weights");
        _logsDir = _tempManager.createSubDir("logs");
        _profilerDir = _tempManager.createSubDir("profiler");
        _resultDir = _tempManager.createSubDir("results");

        // Create a real workspace that points to our temp directories
        _workspace = std::make_unique<WheelDL::Utils::Workspace>(_tempManager.getTempDir(), "test", false);
    }

    WheelDL::Utils::Workspace* get() { return _workspace.get(); }

    std::string getRoot() const { return _root; }
    std::string getWeightsDir() const { return _weightsDir; }
    std::string getLogsDir() const { return _logsDir; }
    std::string getProfilerDir() const { return _profilerDir; }
    std::string getResultDir() const { return _resultDir; }

private:
    TempFileManager _tempManager;
    std::unique_ptr<WheelDL::Utils::Workspace> _workspace;
    std::string _root;
    std::string _weightsDir;
    std::string _logsDir;
    std::string _profilerDir;
    std::string _resultDir;
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
// MockBaseModel - Test model for engine tests
// ============================================================================

class MockBaseModel : public Model::BaseModel {
public:
    MockBaseModel() : Model::BaseModel() {
        _model = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 16, 3).padding(1)),
            torch::nn::BatchNorm2d(16),
            torch::nn::ReLU(),
            torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})),
            torch::nn::Flatten(),
            torch::nn::Linear(16, 10)
        );

        for (size_t i = 0; i < _model->size(); ++i) {
            register_module("layer_" + std::to_string(i), _model->ptr(i));
        }

        _stride = torch::tensor({8.0f, 16.0f, 32.0f});
        _taskType = TaskType::CLASSIFICATION;
        _isInitialized = true;
    }

    // Expose forward for testing
    std::vector<torch::Tensor> predict(const torch::Tensor& x) {
        return {_model->forward(x)};
    }

protected:
    std::unique_ptr<Model::Loss::BaseLoss> initCriterion() override {
        return std::make_unique<MockLoss>();
    }
};

// ============================================================================
// MetricsFactory - Creates test MetricsData objects
// ============================================================================

class MetricsFactory {
public:
    static MetricsData createDefault() {
        return MetricsData{};
    }

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

    static MetricsData createWithFitness(float fitness) {
        MetricsData m = createTypical();
        m.fitness = fitness;
        return m;
    }
};

// ============================================================================
// ProgressFactory - Creates test ProgressData objects
// ============================================================================

class ProgressFactory {
public:
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

    static ProgressData createAt(int epoch, int batch, int totalEpochs = 100, int totalBatches = 100) {
        ProgressData p = createDefault();
        p.currentEpoch = epoch;
        p.currentBatch = batch;
        p.totalEpochs = totalEpochs;
        p.totalBatches = totalBatches;
        return p;
    }

    static ProgressData createTrainBatch(int epoch, int batch, float loss) {
        ProgressData p = createAt(epoch, batch);
        p.stage = ProgressStage::TRAIN_BATCH;
        p.loss = loss;
        return p;
    }

    static ProgressData createTrainEpoch(int epoch, const MetricsData& metrics) {
        ProgressData p = createAt(epoch, 0);
        p.stage = ProgressStage::TRAIN_EPOCH;
        p.metrics = metrics;
        return p;
    }

    static ProgressData createValEpoch(int epoch, const MetricsData& metrics) {
        ProgressData p = createAt(epoch, 0);
        p.stage = ProgressStage::VAL_EPOCH;
        p.metrics = metrics;
        return p;
    }
};

// ============================================================================
// DataExampleFactory - Creates test DataExample objects
// ============================================================================

class DataExampleFactory {
public:
    static Data::Dataset::DataExample createClassification(int batchSize = 4, int numClasses = 10) {
        Data::Dataset::DataExample example;
        example.data = torch::randn({batchSize, 3, 224, 224});
        example.targets = torch::randint(0, numClasses, {batchSize});
        return example;
    }

    static Data::Dataset::DataExample createDetection(int batchSize = 4, int maxObjects = 10) {
        Data::Dataset::DataExample example;
        example.data = torch::randn({batchSize, 3, 640, 640});
        // Detection target: [batch_idx, class, x, y, w, h]
        example.targets = torch::rand({batchSize * maxObjects, 6});
        return example;
    }

    static Data::Dataset::DataExample createSegmentation(int batchSize = 4, int numClasses = 10) {
        Data::Dataset::DataExample example;
        example.data = torch::randn({batchSize, 3, 640, 640});
        example.targets = torch::randint(0, numClasses, {batchSize, 640, 640});
        return example;
    }

    static Data::Dataset::DataExample createAnomaly(int batchSize = 4) {
        Data::Dataset::DataExample example;
        example.data = torch::randn({batchSize, 3, 256, 256});
        example.targets = torch::randint(0, 2, {batchSize});  // 0=normal, 1=anomaly
        return example;
    }

    static Data::Dataset::DataExample createEmpty() {
        return Data::Dataset::DataExample{};
    }
};

// ============================================================================
// Configuration Factory for Engine Tests
// ============================================================================

class EngineConfigFactory {
public:
    static std::shared_ptr<Config::Configuration> createMinimal(TaskType taskType = TaskType::CLASSIFICATION) {
        TempFileManager temp("EngineConfig");

        std::string typeStr;
        switch (taskType) {
            case TaskType::CLASSIFICATION: typeStr = "classification"; break;
            case TaskType::DETECTION: typeStr = "detection"; break;
            case TaskType::SEGMENTATION: typeStr = "segmentation"; break;
            case TaskType::ANOMALY: typeStr = "anomaly"; break;
            case TaskType::OBB: typeStr = "obb"; break;
            default: typeStr = "classification"; break;
        }

        std::string datasetPath = temp.createFile("dataset.json", R"({
    "header": {
        "type": ")" + typeStr + R"(",
        "categories": ["class0", "class1", "class2", "class3", "class4",
                       "class5", "class6", "class7", "class8", "class9"]
    },
    "annotations": []
})");

        std::string hypPath = temp.createFile("hyp.yaml", R"(
epochs: 10
patience: 5
batch_size: 4
image_size: 224
device: "0"
workers: 0
optimizer: SGD
seed: 42
deterministic: true
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 1.0
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
freeze: 0
)");

        auto config = std::make_shared<Config::Configuration>();
        config->load("Config/Valid/model/cls_ConvNext.yaml", hypPath, datasetPath);

        return config;
    }

    static std::shared_ptr<Config::Configuration> createWithDevice(const std::string& device) {
        TempFileManager temp("EngineConfigDevice");

        std::string datasetPath = temp.createFile("dataset.json", R"({
    "header": {
        "type": "classification",
        "categories": ["class0", "class1", "class2"]
    },
    "annotations": []
})");

        std::string deviceStr = device.empty() ? "\"\"" : ("\"" + device + "\"");
        std::string hypPath = temp.createFile("hyp.yaml",
            "epochs: 10\n"
            "batch_size: 4\n"
            "image_size: 224\n"
            "device: " + deviceStr + "\n"
            "workers: 0\n"
            "optimizer: SGD\n"
            "lr0: 0.01\n"
            "lrf: 0.01\n"
            "momentum: 0.937\n"
            "weight_decay: 0.0005\n"
            "warmup_epochs: 1.0\n"
            "cos_lr: false\n"
            "amp: false\n"
        );

        auto config = std::make_shared<Config::Configuration>();
        config->load("Config/Valid/model/cls_ConvNext.yaml", hypPath, datasetPath);

        return config;
    }

    static std::shared_ptr<Config::Configuration> createCPU() {
        return createWithDevice("cpu");
    }

    static std::shared_ptr<Config::Configuration> createCUDA() {
        return createWithDevice("0");
    }
};

// ============================================================================
// TestableBasePredictor - Concrete implementation for testing
// ============================================================================

class TestableBasePredictor : public Core::Predictor::BasePredictor {
public:
    using SetupModelCallback = std::function<void()>;
    using PostprocessCallback = std::function<void(const std::vector<torch::Tensor>&, const std::tuple<int, int>&, const std::string&)>;
    using ExportResultsCallback = std::function<void(const std::string&)>;
    using ClearResultsCallback = std::function<void()>;

    TestableBasePredictor(
        std::shared_ptr<Config::Configuration> config,
        const std::string& checkpointPath,
        WheelDL::Utils::Logger* logger,
        WheelDL::Utils::PerformanceProfiler* profiler,
        std::atomic<bool>* stopFlag = nullptr)
        : BasePredictor(config, checkpointPath, logger, profiler, stopFlag)
    {}

    // Hook callbacks
    SetupModelCallback onSetupModel;
    PostprocessCallback onPostprocess;
    ExportResultsCallback onExportResults;
    ClearResultsCallback onClearResults;

    // Counters for verification
    int setupModelCallCount = 0;
    int postprocessCallCount = 0;
    int exportResultsCallCount = 0;
    int clearResultsCallCount = 0;

    // Expose protected members for testing
    Model::BaseModel* getModel() { return _model.get(); }
    void setModel(std::unique_ptr<Model::BaseModel> model) { _model = std::move(model); }
    bool isModelLoaded() const { return _isModelLoaded; }
    void setModelLoaded(bool loaded) { _isModelLoaded = loaded; }
    bool isWarmedUp() const { return _isWarmedUp; }
    void setWarmedUp(bool warmed) { _isWarmedUp = warmed; }
    float getThreshold() const { return _threshold; }

    // Expose protected methods for testing
    using BasePredictor::setupDevice;
    using BasePredictor::inference;
    using BasePredictor::warmup;
    using BasePredictor::isStopRequested;

protected:
    void setupModel() override {
        setupModelCallCount++;
        if (onSetupModel) {
            onSetupModel();
        } else {
            _model = std::make_unique<MockBaseModel>();
        }
    }

    void postprocess(
        const std::vector<torch::Tensor>& output,
        const std::tuple<int, int>& originalShape,
        const std::string& imagePath) override
    {
        postprocessCallCount++;
        if (onPostprocess) {
            onPostprocess(output, originalShape, imagePath);
        }
    }

    void exportResults(const std::string& outputDir) override {
        exportResultsCallCount++;
        if (onExportResults) {
            onExportResults(outputDir);
        }
    }

    void clearResults() override {
        clearResultsCallCount++;
        if (onClearResults) {
            onClearResults();
        }
    }
};

// ============================================================================
// TestableBaseValidator - Concrete implementation for testing
// ============================================================================

class TestableBaseValidator : public Core::Validator::BaseValidator {
public:
    using SetupModelCallback = std::function<std::unique_ptr<Model::BaseModel>()>;
    using SetupDataLoaderCallback = std::function<void()>;
    using PreprocessBatchCallback = std::function<Data::Dataset::DataExample(const Data::Dataset::DataExample&)>;
    using PostprocessBatchCallback = std::function<torch::Tensor(const std::vector<torch::Tensor>&)>;
    using ComputeMetricsCallback = std::function<MetricsData(const torch::Tensor&, const torch::Tensor&)>;

    TestableBaseValidator(
        std::shared_ptr<Config::Configuration> config,
        WheelDL::Utils::Logger* logger,
        WheelDL::Utils::PerformanceProfiler* profiler,
        std::atomic<bool>* stopFlag = nullptr)
        : BaseValidator(config, logger, profiler, stopFlag)
    {}

    // Hook callbacks
    SetupModelCallback onSetupModel;
    SetupDataLoaderCallback onSetupDataLoader;
    PreprocessBatchCallback onPreprocessBatch;
    PostprocessBatchCallback onPostprocessBatch;
    ComputeMetricsCallback onComputeMetrics;

    // Counters
    int setupModelCallCount = 0;
    int setupDataLoaderCallCount = 0;
    int preprocessBatchCallCount = 0;
    int postprocessBatchCallCount = 0;
    int computeMetricsCallCount = 0;

    // Expose protected methods
    using BaseValidator::setupDevice;
    using BaseValidator::isStopRequested;

protected:
    std::unique_ptr<Model::BaseModel> setupModel() override {
        setupModelCallCount++;
        if (onSetupModel) {
            return onSetupModel();
        }
        return std::make_unique<MockBaseModel>();
    }

    void setupDataLoader() override {
        setupDataLoaderCallCount++;
        if (onSetupDataLoader) {
            onSetupDataLoader();
        }
    }

    Data::Dataset::DataExample preprocessBatch(const Data::Dataset::DataExample& batch) override {
        preprocessBatchCallCount++;
        if (onPreprocessBatch) {
            return onPreprocessBatch(batch);
        }
        return batch;
    }

    torch::Tensor postprocessBatch(const std::vector<torch::Tensor>& prediction) override {
        postprocessBatchCallCount++;
        if (onPostprocessBatch) {
            return onPostprocessBatch(prediction);
        }
        return prediction.empty() ? torch::Tensor() : prediction[0];
    }

    MetricsData computeMetrics(const torch::Tensor& pred, const torch::Tensor& target) override {
        computeMetricsCallCount++;
        if (onComputeMetrics) {
            return onComputeMetrics(pred, target);
        }
        return MetricsFactory::createTypical();
    }
};

// ============================================================================
// TestableBaseTrainer - Concrete implementation for testing
// ============================================================================

class TestableBaseTrainer : public Core::Trainer::BaseTrainer {
public:
    using SetupModelCallback = std::function<void()>;
    using SetupDataLoadersCallback = std::function<void()>;
    using SetupValidatorCallback = std::function<void()>;
    using PreprocessBatchCallback = std::function<Data::Dataset::DataExample(const Data::Dataset::DataExample&)>;
    using CalculateFitnessCallback = std::function<float(const MetricsData&)>;
    using SetupPredictorCallback = std::function<std::unique_ptr<Core::Predictor::BasePredictor>(const std::string&)>;

    TestableBaseTrainer(
        std::shared_ptr<Config::Configuration> config,
        WheelDL::Utils::Logger* logger,
        WheelDL::Utils::Workspace* workspace,
        WheelDL::Utils::PerformanceProfiler* profiler,
        ProgressCallback progressCallback = nullptr,
        std::atomic<bool>* stopFlag = nullptr)
        : BaseTrainer(config, logger, workspace, profiler, progressCallback, stopFlag)
    {}

    // Hook callbacks
    SetupModelCallback onSetupModel;
    SetupDataLoadersCallback onSetupDataLoaders;
    SetupValidatorCallback onSetupValidator;
    PreprocessBatchCallback onPreprocessBatch;
    CalculateFitnessCallback onCalculateFitness;
    SetupPredictorCallback onSetupPredictor;

    // Counters
    int setupModelCallCount = 0;
    int setupDataLoadersCallCount = 0;
    int setupValidatorCallCount = 0;
    int preprocessBatchCallCount = 0;
    int calculateFitnessCallCount = 0;
    int setupPredictorCallCount = 0;

    // Expose protected members for testing
    Model::BaseModel* getModel() { return _model.get(); }
    void setModel(std::unique_ptr<Model::BaseModel> model) { _model = std::move(model); }
    Core::Validator::BaseValidator* getValidator() { return _validator.get(); }
    torch::optim::Optimizer* getOptimizer() { return _optimizer.get(); }
    Optimizer::Scheduler::LRScheduler* getScheduler() { return _scheduler.get(); }
    Optimizer::EMA::ModelEMA* getEMA() { return _ema.get(); }
    Optimizer::EarlyStopping::EarlyStopping* getEarlyStopping() { return _earlyStopping.get(); }
    torch::Device getDevice() const { return _device; }

    // Expose protected methods for testing
    using BaseTrainer::initializeSeeds;
    using BaseTrainer::setupDevice;
    using BaseTrainer::setupSaveDirectory;
    using BaseTrainer::freezeLayersIfNeeded;
    using BaseTrainer::setupOptimizer;
    using BaseTrainer::setupScheduler;
    using BaseTrainer::initializeEMA;
    using BaseTrainer::initializeEarlyStopping;
    using BaseTrainer::trainEpoch;
    using BaseTrainer::trainBatch;
    using BaseTrainer::saveCheckpoint;
    using BaseTrainer::loadCheckpoint;
    using BaseTrainer::finalValidation;
    using BaseTrainer::finalPrediction;
    using BaseTrainer::invokeProgressCallback;
    using BaseTrainer::isStopRequested;

    // Helper to set train data loader with mock data
    template<typename DataLoaderType>
    void setTestTrainDataLoader(DataLoaderType&& loader) {
        setTrainDataLoader(std::forward<DataLoaderType>(loader));
    }

protected:
    void setupModel() override {
        setupModelCallCount++;
        if (onSetupModel) {
            onSetupModel();
        } else {
            _model = std::make_unique<MockBaseModel>();
        }
    }

    void setupDataLoaders() override {
        setupDataLoadersCallCount++;
        if (onSetupDataLoaders) {
            onSetupDataLoaders();
        }
    }

    void setupValidator() override {
        setupValidatorCallCount++;
        if (onSetupValidator) {
            onSetupValidator();
        }
    }

    Data::Dataset::DataExample preprocessBatch(const Data::Dataset::DataExample& batch) override {
        preprocessBatchCallCount++;
        if (onPreprocessBatch) {
            return onPreprocessBatch(batch);
        }
        return batch;
    }

    float calculateFitness(const MetricsData& metrics) override {
        calculateFitnessCallCount++;
        if (onCalculateFitness) {
            return onCalculateFitness(metrics);
        }
        return metrics.fitness;
    }

    std::unique_ptr<Core::Predictor::BasePredictor> setupPredictor(const std::string& checkpointPath) override {
        setupPredictorCallCount++;
        if (onSetupPredictor) {
            return onSetupPredictor(checkpointPath);
        }
        return std::make_unique<TestableBasePredictor>(
            _config, checkpointPath, _logger, _profiler, _stopFlag);
    }
};

// ============================================================================
// CUDA Test Fixture Base
// ============================================================================

class CUDATestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        _hasCuda = torch::cuda::is_available();
        if (_hasCuda) {
            _device = torch::Device(torch::kCUDA, 0);
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
// Engine Test Fixture
// ============================================================================

class EngineTestFixture : public CUDATestFixture {
protected:
    void SetUp() override {
        CUDATestFixture::SetUp();

        _tempManager = std::make_unique<TempFileManager>("EngineTest");
        _mockLogger = std::make_unique<MockLogger>("EngineTestLogger");
        _mockProfiler = std::make_unique<MockProfiler>();
        _mockWorkspace = std::make_unique<MockWorkspace>();
        _config = EngineConfigFactory::createMinimal();
        _stopFlag = false;
    }

    void TearDown() override {
        _tempManager.reset();
        _mockLogger.reset();
        _mockProfiler.reset();
        _mockWorkspace.reset();
        CUDATestFixture::TearDown();
    }

    // Accessors
    TempFileManager& tempManager() { return *_tempManager; }
    MockLogger& mockLogger() { return *_mockLogger; }
    MockProfiler& mockProfiler() { return *_mockProfiler; }
    MockWorkspace& mockWorkspace() { return *_mockWorkspace; }
    std::shared_ptr<Config::Configuration> config() { return _config; }
    std::atomic<bool>* stopFlag() { return &_stopFlag; }

    WheelDL::Utils::Logger* logger() { return _mockLogger->get(); }
    WheelDL::Utils::PerformanceProfiler* profiler() { return _mockProfiler->get(); }
    WheelDL::Utils::Workspace* workspace() { return _mockWorkspace->get(); }

    // Helper to create mock model
    std::unique_ptr<MockBaseModel> createMockModel() {
        return std::make_unique<MockBaseModel>();
    }

    // Helper to create test checkpoint
    std::string createTestCheckpoint(const std::string& filename = "test.pt") {
        auto model = createMockModel();
        auto optimizer = std::make_unique<torch::optim::SGD>(
            model->parameters(),
            torch::optim::SGDOptions(0.01).momentum(0.9)
        );

        Core::Utils::CheckpointMetadata metadata{};
        metadata.wheelLibVersion = "0.1.0";
        metadata.libtorchVersion = "2.0.0";
        metadata.taskType = TaskType::CLASSIFICATION;
        metadata.epoch = 10;
        metadata.bestFitness = 0.85f;
        metadata.threshold = 0.5f;
        metadata.hyperParams["image_size"] = "224";
        metadata.hyperParams["batch_size"] = "4";

        std::string path = _tempManager->getFilePath(filename);
        Core::Utils::Checkpoint::save(path, *model, *optimizer, metadata);

        return path;
    }

private:
    std::unique_ptr<TempFileManager> _tempManager;
    std::unique_ptr<MockLogger> _mockLogger;
    std::unique_ptr<MockProfiler> _mockProfiler;
    std::unique_ptr<MockWorkspace> _mockWorkspace;
    std::shared_ptr<Config::Configuration> _config;
    std::atomic<bool> _stopFlag;
};

// ============================================================================
// Callback Tracker for Progress Callbacks
// ============================================================================

class ProgressCallbackTracker {
public:
    ProgressCallback getCallback() {
        return [this](const ProgressData& data) {
            std::lock_guard<std::mutex> lock(_mutex);
            _invocations.push_back(data);
        };
    }

    size_t count() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _invocations.size();
    }

    std::vector<ProgressData> getInvocations() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _invocations;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        _invocations.clear();
    }

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

} // namespace Engine
} // namespace Test
} // namespace WheelDL
