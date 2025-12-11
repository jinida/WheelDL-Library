#pragma once

#include "../Engine/BaseTrainer.h"
#include "../../Model/Task/AnomalyModel.h"
#include "../../Data/Dataset/AnomalyDataset.h"
#include <functional>

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            /**
             * @class AnomalyTrainer
             * @brief Trainer for anomaly detection tasks
             *
             * Implements training for anomaly detection models:
             * - EfficientAD: Student-Teacher distillation with autoencoder
             * - PatchCore: Memory bank based anomaly detection
             * - SimpleNet: Simple feature-based anomaly detection
             *
             * Key features:
             * - Automatic model preparation (feature normalization, memory bank building)
             * - One-class learning (trains on normal samples only)
             * - Validation preparation (quantile normalization, memory bank subsampling)
             */
            class AnomalyTrainer : public BaseTrainer {
            public:
                /**
                 * @brief Constructor with dependency injection
                 * @param config Configuration object with anomaly detection settings
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param workspace Workspace instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param progressCallback Optional progress callback
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit AnomalyTrainer(
                    std::shared_ptr<Config::Configuration> config,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::Workspace* workspace,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    ProgressCallback progressCallback = nullptr,
                    std::atomic<bool>* stopFlag = nullptr);

                /**
                 * @brief Destructor
                 */
                ~AnomalyTrainer() override = default;

            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup anomaly detection model
                 *
                 * Creates AnomalyModel based on configuration model path.
                 */
                void setupModel() override;

                /**
                 * @brief Setup data loaders for anomaly detection
                 *
                 * Creates AnomalyDataset and DataLoader for training.
                 * Calls prepareTraining() after dataloader creation.
                 */
                void setupDataLoaders() override;

                /**
                 * @brief Setup validator for anomaly detection
                 *
                 * Creates AnomalyValidator and injects validation dataloader.
                 */
                void setupValidator() override;

                /**
                 * @brief Preprocess batch before forward pass
                 * @param batch Input batch from dataloader
                 * @return Preprocessed batch with data moved to device
                 */
                WheelDL::Data::Dataset::DataExample preprocessBatch(
                    const WheelDL::Data::Dataset::DataExample& batch) override;

                /**
                 * @brief Calculate fitness metric for checkpoint selection
                 *
                 * For anomaly detection, fitness is typically based on:
                 * - AUC-ROC (Area Under Curve - Receiver Operating Characteristic)
                 * - AP (Average Precision)
                 * - F1 Score
                 *
                 * @param metrics Validation metrics
                 * @return Fitness value (higher is better)
                 */
                float calculateFitness(const MetricsData& metrics) override;

                /**
                 * @brief Setup anomaly predictor for inference
                 *
                 * Creates AnomalyPredictor with checkpoint.
                 *
                 * @param checkpointPath Path to checkpoint file
                 * @return Unique pointer to AnomalyPredictor
                 */
                std::unique_ptr<Predictor::BasePredictor> setupPredictor(
                    const std::string& checkpointPath) override;

                /**
                 * @brief Run validation with prepareValidation call
                 *
                 * Overrides BaseTrainer::runValidation to call prepareValidation
                 * before each validation (required for EfficientAD quantile setup).
                 *
                 * @param epoch Current epoch number
                 * @param useEmaIfAvailable Whether to use EMA model if available
                 * @return Validation metrics
                 */
                MetricsData runValidation(int epoch, bool useEmaIfAvailable) override;

            private:
                bool _isModelPrepared;       ///< Whether model has been prepared for training

                /**
                 * @brief Callback to prepare validation (e.g., EfficientAD quantile setup)
                 *
                 * Set by setupDataLoaders(), uses 10% subset for efficiency
                 */
                std::function<void()> _prepareValidationCallback;

                /**
                 * @brief Batch iterator for 10% subset (used for quantile computation)
                 */
                Model::Modules::BatchIteratorFunc _prepareBatchIterator;
            };

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
