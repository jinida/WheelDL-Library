#pragma once

#include "../Engine/BaseTrainer.h"
#include "../../Model/Task/AnomalyModel.h"
#include "../../Data/Dataset/AnomalyDataset.h"

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
                 * @brief Constructor
                 * @param config Configuration object with anomaly detection settings
                 */
                explicit AnomalyTrainer(const std::shared_ptr<Config::Configuration> config);

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

            private:
                std::string _modelYamlPath;  ///< Path to model YAML configuration
                bool _isModelPrepared;       ///< Whether model has been prepared for training
                bool _isValidationPrepared;  ///< Whether model has been prepared for validation
            };

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
