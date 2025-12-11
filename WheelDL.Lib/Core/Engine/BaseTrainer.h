#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>
#include <atomic>
#include <functional>
#include "../../Config/Configuration.h"
#include "../../Model/Task/BaseModel.h"
#include "../../Model/Loss/BaseLoss.h"
#include "../../Data/Dataset/BaseDataset.h"
#include "../../Optimizer/OptimizerFactory.h"
#include "../../Optimizer/Scheduler/LRScheduler.h"
#include "../../Optimizer/EMA/ModelEMA.h"
#include "../../Optimizer/EarlyStopping/EarlyStopping.h"
#include "../../Utils/Logger/Logger.h"
#include "../../Utils/Common/Timer.h"
#include "../../Utils/Profiler/PerformanceProfiler.h"
#include "../../Utils/Memory/MemoryManager.h"
#include "../../Utils/Common/Random.h"
#include "../../Utils/Workspace/Workspace.h"
#include "../Callback/AsyncCallbackQueue.h"
#include "../Callback/CallbackThrottler.h"
#include "../Engine/BaseValidator.h"
#include "../Utils/Checkpoint.h"
#include "../Utils/CheckpointMetadata.h"

// Forward declarations
namespace WheelDL {
    namespace Core {
        namespace Predictor {
            class BasePredictor;
        }
    }
}

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            /**
             * @class EMAGuard
             * @brief RAII guard for EMA apply/restore pattern
             *
             * Automatically applies EMA weights on construction and restores original
             * weights on destruction, ensuring exception-safe EMA operations.
             */
            class EMAGuard {
            public:
                EMAGuard(Optimizer::EMA::ModelEMA* ema, Model::BaseModel* model)
                    : _ema(ema), _model(model), _applied(false)
                {
                    if (_ema && _model) {
                        _ema->applyToModel(*_model);
                        _applied = true;
                    }
                }

                ~EMAGuard() {
                    if (_applied && _ema && _model) {
                        _ema->restoreOriginalParams(*_model);
                    }
                }

                // Non-copyable, non-movable
                EMAGuard(const EMAGuard&) = delete;
                EMAGuard& operator=(const EMAGuard&) = delete;
                EMAGuard(EMAGuard&&) = delete;
                EMAGuard& operator=(EMAGuard&&) = delete;

                bool isApplied() const { return _applied; }

            private:
                Optimizer::EMA::ModelEMA* _ema;
                Model::BaseModel* _model;
                bool _applied;
            };

            /**
             * @class BaseTrainer
             * @brief Template Method Pattern base class for training
             *
             * Provides complete training loop with hooks for task-specific customization.
             * Utilizes all existing WheelDL components to avoid duplication.
             *
             * Key features:
             * - Configuration-driven (no hardcoded values)
             * - Comprehensive logging via Utils::Logger
             * - Performance profiling via Utils::PerformanceProfiler
             * - Memory monitoring via Utils::MemoryManager
             * - Async callback system
             * - Checkpoint management
             * - AMP support
             * - EMA support
             * - Early stopping
             *
             * Derived classes must implement:
             * - setupModel()
             * - setupDataLoaders()
             * - setupValidator()
             * - preprocessBatch()
             * - calculateFitness()
             */
            class BaseTrainer {
            public:
                /**
                 * @brief Constructor with dependency injection
                 * @param config Configuration object containing all training settings
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param workspace Workspace instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param progressCallback Optional progress callback
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit BaseTrainer(
                    std::shared_ptr<Config::Configuration> config,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::Workspace* workspace,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    ProgressCallback progressCallback = nullptr,
                    std::atomic<bool>* stopFlag = nullptr);

                /**
                 * @brief Virtual destructor
                 */
                virtual ~BaseTrainer();

                // Disable copy and move
                BaseTrainer(const BaseTrainer&) = delete;
                BaseTrainer& operator=(const BaseTrainer&) = delete;
                BaseTrainer(BaseTrainer&&) = delete;
                BaseTrainer& operator=(BaseTrainer&&) = delete;

                /**
                 * @brief Main training loop (Template Method)
                 * @return Final metrics
                 */
                MetricsData train();

                /**
                 * @brief Check if stop has been requested
                 * @return true if stop was requested, false otherwise
                 */
                bool isStopRequested() const {
                    return _stopFlag && _stopFlag->load(std::memory_order_acquire);
                }

                /**
                 * @brief Get current epoch
                 */
                int getCurrentEpoch() const { return _currentEpoch; }

                /**
                 * @brief Get best fitness achieved
                 */
                float getBestFitness() const { return _bestFitness; }

                /**
                 * @brief Get save directory path
                 */
                std::string getSaveDirectory() const;

				void setCheckpoint(const std::string& checkpointPath) { _checkpointPath = checkpointPath; }
                void setPretrainedPath(const std::string& checkpointPath) { _checkpointPath = checkpointPath; _usePretrained = true; }

            protected:
                // ========== Hook Methods (Pure Virtual) ==========

                /**
                 * @brief Setup model architecture
                 *
                 * Create and initialize the model based on configuration.
                 * Should set _model member.
                 */
                virtual void setupModel() = 0;

                /**
                 * @brief Setup data loaders
                 *
                 * Create train and validation data loaders.
                 * Should call setDataLoaders() template method to store training loader.
                 */
                virtual void setupDataLoaders() = 0;

                /**
                 * @brief Setup validator
                 *
                 * Create validator instance and inject validation data loader.
                 * Should set _validator member.
                 */
                virtual void setupValidator() = 0;

                /**
                 * @brief Preprocess batch before forward pass
                 * @param batch Input batch from dataloader
                 * @return Preprocessed batch ready for model
                 */
                virtual WheelDL::Data::Dataset::DataExample preprocessBatch(
                    const WheelDL::Data::Dataset::DataExample& batch) = 0;

                // ========== Common Methods ==========

                /**
                 * @brief Initialize random seeds
                 *
                 * Sets seeds for reproducibility using Utils::Random and torch.
                 */
                void initializeSeeds();

                /**
                 * @brief Setup device (CPU/CUDA)
                 *
                 * Configures torch device based on Configuration.
                 */
                void setupDevice();

                /**
                 * @brief Setup save directory
                 *
                 * Creates output directory for checkpoints and logs.
                 */
                void setupSaveDirectory();

                /**
                 * @brief Freeze layers if specified in config
                 *
                 * Implements layer freezing based on configuration.
                 */
                void freezeLayersIfNeeded();

                /**
                 * @brief Setup AMP (Automatic Mixed Precision)
                 *
                 * TODO: Implement custom AMP support for LibTorch C++
                 * LibTorch C++ does not have native AMP like Python PyTorch
                 */
                void setupAMP();

                /**
                 * @brief Setup optimizer using OptimizerFactory
                 *
                 * Creates optimizer based on config.getOptimizer().
                 */
                void setupOptimizer();

                /**
                 * @brief Setup LR scheduler
                 *
                 * Creates appropriate scheduler based on config.useCosineLR().
                 */
                void setupScheduler();

                /**
                 * @brief Initialize EMA if enabled
                 *
                 * Creates ModelEMA instance based on configuration.
                 */
                void initializeEMA();

                /**
                 * @brief Initialize early stopping
                 *
                 * Creates EarlyStopping instance based on config.getPatience().
                 */
                void initializeEarlyStopping();

                /**
                 * @brief Set training data loader (template method for type erasure)
                 * @tparam DataLoaderType Type of data loader
                 * @param trainLoader Training data loader (unique_ptr)
                 */
                template<typename DataLoaderType>
                void setTrainDataLoader(DataLoaderType&& trainLoader) {
                    // Store loader in shared_ptr for lambda capture
                    auto trainLoaderPtr = std::make_shared<DataLoaderType>(std::forward<DataLoaderType>(trainLoader));

                    // Create type-erased iteration function
                    _trainBatchIterator = [trainLoaderPtr](std::function<void(const WheelDL::Data::Dataset::DataExample&, int)> callback) {
                        int batchIdx = 0;
                        for (auto& batch : *(*trainLoaderPtr)) {
                            callback(batch, batchIdx++);
                        }
                    };
                }

                /**
                 * @brief Train one epoch
                 * @param epoch Current epoch number
                 */
                void trainEpoch(int epoch);

                /**
                 * @brief Train one batch (helper for derived classes)
                 * @param batch Input batch
                 * @param batchIdx Batch index
                 */
                void trainBatch(const WheelDL::Data::Dataset::DataExample& batch, int batchIdx);

                /**
                 * @brief Save checkpoint
                 * @param epoch Current epoch
                 * @param metrics Current metrics data
                 * @param isBest Whether this is the best checkpoint
                 */
                void saveCheckpoint(int epoch, bool isBest);

                void saveTrainingResults();

                virtual MetricsData runValidation(int epoch, bool useEmaIfAvailable);

                /**
                 * @brief Perform final validation with best model
                 *
                 * Loads best.pt checkpoint (or uses current model if not exists)
                 * and performs final validation. Saves best_metrics.json.
                 * If best.pt didn't exist, saves it after validation.
                 */
                void finalValidation();

                /**
                 * @brief Perform final prediction with best model
                 *
                 * Delegates all prediction logic to the Predictor:
                 * - Creates predictor via setupPredictor()
                 * - Calls predictor->predictAndExport()
                 *
                 * The Predictor handles:
                 * - PredDataset creation
                 * - DataLoader iteration
                 * - Batch inference
                 * - Result export
                 */
                void finalPrediction();

                /**
                 * @brief Setup task-specific predictor (pure virtual hook)
                 *
                 * Derived classes must implement this to create their specific predictor.
                 * For example:
                 * - AnomalyTrainer: return std::make_unique<AnomalyPredictor>(_config, checkpointPath)
                 * - DetectionTrainer: return std::make_unique<DetectionPredictor>(_config, checkpointPath)
                 *
                 * @param checkpointPath Path to checkpoint file
                 * @return Unique pointer to task-specific predictor
                 */
                virtual std::unique_ptr<Predictor::BasePredictor> setupPredictor(
                    const std::string& checkpointPath) = 0;

                /**
                 * @brief Optimizer step with gradient clipping
                 */
                void optimizerStep();

                /**
                 * @brief Scheduler step
                 */
                void schedulerStep();

                /**
                 * @brief Invoke progress callback (with throttling)
                 * @param data Progress data to send
                 */
                void invokeProgressCallback(const ProgressData& data);

                /**
                 * @brief Calculate fitness from metrics (must be implemented by derived class)
                 * @param metrics Current metrics
                 * @return Fitness value (higher is better)
                 *
                 * Derived classes should implement task-specific fitness calculation.
                 * For example:
                 * - Detection: use mAP
                 * - Classification: use accuracy
                 * - Segmentation: use IoU
                 */
                virtual float calculateFitness(const MetricsData& metrics) = 0;

                /**
                 * @brief Load checkpoint into the trainer's model
                 * @param checkpointPath Path to checkpoint file (.pt)
                 * @param resumeTraining If true, also loads optimizer state for resume training
                 * @return Checkpoint metadata containing epoch, fitness, and other info
                 *
                 * This method loads model weights (and optionally optimizer state) from a checkpoint.
                 * The model must be initialized via setupModel() before calling this method.
                 *
                 * Usage:
                 * - loadCheckpoint("path/to/best.pt") - Load model only (inference/validation)
                 * - loadCheckpoint("path/to/last.pt", true) - Load model + optimizer (resume training)
                 */
                Utils::CheckpointMetadata loadCheckpoint(const std::string& checkpointPath, bool resumeTraining = false);

            protected:
                // ========== Configuration & Core Components ==========
                std::shared_ptr<Config::Configuration> _config;
                WheelDL::Utils::Logger* _logger;                      // Injected (owned by Context)
                WheelDL::Utils::Workspace* _workspace;                // Injected (owned by Context)
                WheelDL::Utils::PerformanceProfiler* _profiler;       // Injected (owned by Context)
                WheelDL::Utils::MemoryManager& _memoryManager;
                std::unique_ptr<WheelDL::Utils::Random> _random;

                // ========== Model & Training Components ==========
                std::unique_ptr<Model::BaseModel> _model;
                std::unique_ptr<Validator::BaseValidator> _validator;
                std::unique_ptr<torch::optim::Optimizer> _optimizer;
                std::unique_ptr<Optimizer::Scheduler::LRScheduler> _scheduler;
                std::unique_ptr<Optimizer::EMA::ModelEMA> _ema;
                std::unique_ptr<Optimizer::EarlyStopping::EarlyStopping> _earlyStopping;
                bool _usePretrained = false;

                // ========== Data Loaders ==========
                using BatchIteratorFunc = std::function<void(std::function<void(const WheelDL::Data::Dataset::DataExample&, int)>)>;
                BatchIteratorFunc _trainBatchIterator;
               
                // ========== Device & AMP ==========
                torch::Device _device;
                bool _useAMP;

                // TODO: Add Distributed Data Parallel (DDP) support
                // - bool _isDistributed;
                // - int _rank;
                // - int _worldSize;
                // - std::string _backend; // "nccl", "gloo"
                // - void initializeDistributed();
                // - void wrapModelDDP();
                // - void synchronizeGradients();
                // - void reduceMetrics(); // Average metrics across all ranks

                // ========== Training State ==========
                int _currentEpoch;
                int _totalEpochs;
                float _bestFitness;
                MetricsData _currentMetrics;
				std::vector<MetricsData> _epochMetrics;
				std::string _checkpointPath;

                // ========== Callback & Control ==========
                ProgressCallback _progressCallback;
                std::unique_ptr<Callback::AsyncCallbackQueue> _asyncCallbackQueue;
                std::unique_ptr<Callback::CallbackThrottler> _callbackThrottler;
                std::atomic<bool>* _stopFlag;                         // Injected (owned by Context)

                // ========== Timers ==========
                std::unique_ptr<WheelDL::Utils::Timer> _epochTimer;
                std::unique_ptr<WheelDL::Utils::Timer> _totalTimer;
            };

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
