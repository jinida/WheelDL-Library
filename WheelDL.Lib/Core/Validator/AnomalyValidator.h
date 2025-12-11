#pragma once

#include "../Engine/BaseValidator.h"
#include "../../Model/Task/AnomalyModel.h"

namespace WheelDL {
    namespace Core {
        namespace Validator {

            /**
             * @class AnomalyValidator
             * @brief Validator for anomaly detection tasks
             *
             * Computes metrics for anomaly detection:
             * - AUC-ROC (Area Under Curve - Receiver Operating Characteristic)
             * - AP (Average Precision)
             * - F1 Score
             * - Accuracy
             *
             * Supports multiple anomaly detection methods:
             * - EfficientAD: Student-Teacher distillation
             * - PatchCore: Memory bank based detection
             * - SimpleNet: Simple feature-based detection
             */
            class AnomalyValidator : public BaseValidator {
            public:
                /**
                 * @brief Constructor
                 * @param config Configuration object
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit AnomalyValidator(
                    std::shared_ptr<Config::Configuration> config,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    std::atomic<bool>* stopFlag = nullptr);

                /**
                 * @brief Destructor
                 */
                ~AnomalyValidator() override = default;

            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup model for standalone validation
                 * @return AnomalyModel instance
                 */
                std::unique_ptr<Model::BaseModel> setupModel() override;

                /**
                 * @brief Setup data loader (empty - uses dependency injection)
                 *
                 * AnomalyValidator uses DataLoader injected by AnomalyTrainer.
                 * This method is not used.
                 */
                void setupDataLoader() override;

                /**
                 * @brief Preprocess batch before forward pass
                 * @param batch Input batch from dataloader
                 * @return Preprocessed batch with data moved to device
                 */
                WheelDL::Data::Dataset::DataExample preprocessBatch(
                    const WheelDL::Data::Dataset::DataExample& batch) override;

                torch::Tensor postprocessBatch(
					const std::vector<torch::Tensor>& prediction) override;

                /**
                 * @brief Compute anomaly detection metrics
                 *
                 * @param pred Anomaly scores [N] or [N, H, W] (higher = more anomalous)
                 * @param target Ground truth labels [N] (0=normal, 1=anomaly)
                 * @return MetricsData with:
                 *         - fitness: AUC-ROC (primary metric)
                 *         - mAP: Average Precision
                 *         - accuracy: Classification accuracy
                 *         - f1Score: F1 Score
                 */
                MetricsData computeMetrics(
                    const torch::Tensor& pred,
                    const torch::Tensor& target) override;

            private:
                /**
                 * @brief Compute AUC-ROC score
                 * @param scores Anomaly scores [N]
                 * @param labels Ground truth labels [N] (0=normal, 1=anomaly)
                 * @return AUC-ROC value [0, 1]
                 */
                float computeAUCROC(const torch::Tensor& scores, const torch::Tensor& labels);

                /**
                 * @brief Compute Average Precision (AP)
                 * @param scores Anomaly scores [N]
                 * @param labels Ground truth labels [N] (0=normal, 1=anomaly)
                 * @return AP value [0, 1]
                 */
                float computeAveragePrecision(const torch::Tensor& scores, const torch::Tensor& labels);

                /**
                 * @brief Compute optimal metrics at best F1 threshold
                 * @param scores Anomaly scores [N]
                 * @param labels Ground truth labels [N] (0=normal, 1=anomaly)
                 * @param[out] threshold Optimal threshold
                 * @param[out] precision Precision at optimal threshold
                 * @param[out] recall Recall at optimal threshold
                 * @param[out] accuracy Accuracy at optimal threshold
                 * @return F1 Score value [0, 1]
                 */
                float computeOptimalMetrics(
                    const torch::Tensor& scores,
                    const torch::Tensor& labels,
                    float& threshold,
                    float& precision,
                    float& recall,
                    float& accuracy);
            };

        } // namespace Validator
    } // namespace Core
} // namespace WheelDL
