#pragma once

#include "../Engine/BaseValidator.h"
#include "../../Model/Task/ClassificationModel.h"

namespace WheelDL {
    namespace Core {
        namespace Validator {

            /**
             * @class ClassificationValidator
             * @brief Validator for image classification tasks
             *
             * Computes metrics for classification:
             * - Accuracy (Top-1)
             * - Precision, Recall, F1 Score (macro-averaged)
             *
             * Supports:
             * - Multi-class classification
             */
            class ClassificationValidator : public BaseValidator {
            public:
                /**
                 * @brief Constructor
                 * @param config Configuration object
                 */
                explicit ClassificationValidator(const std::shared_ptr<Config::Configuration> config);

                /**
                 * @brief Destructor
                 */
                ~ClassificationValidator() override = default;

            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup data loader (empty - uses dependency injection)
                 *
                 * ClassificationValidator uses DataLoader injected by ClassificationTrainer.
                 */
                void setupDataLoader() override;

                /**
                 * @brief Preprocess batch before forward pass
                 * @param batch Input batch from dataloader
                 * @return Preprocessed batch with data moved to device
                 */
                WheelDL::Data::Dataset::DataExample preprocessBatch(
                    const WheelDL::Data::Dataset::DataExample& batch) override;

                /**
                 * @brief Postprocess model output
                 * @param prediction Model output tensors
                 * @return Processed prediction tensor (class probabilities or logits)
                 */
                torch::Tensor postprocessBatch(
                    const std::vector<torch::Tensor>& prediction) override;

                /**
                 * @brief Compute classification metrics
                 *
                 * @param pred Model predictions [N, num_classes] (logits or probabilities)
                 * @param target Ground truth labels [N] (class indices)
                 * @return MetricsData with:
                 *         - accuracy: Classification accuracy
                 *         - fitness: Same as accuracy
                 *         - precision: Macro-averaged precision
                 *         - recall: Macro-averaged recall
                 *         - f1Score: Macro-averaged F1
                 */
                MetricsData computeMetrics(
                    const torch::Tensor& pred,
                    const torch::Tensor& target) override;

            private:
                /**
                 * @brief Compute precision, recall, F1 for multi-class
                 * @param pred Predicted class indices [N]
                 * @param target Ground truth labels [N]
                 * @param numClasses Number of classes
                 * @param[out] precision Macro-averaged precision
                 * @param[out] recall Macro-averaged recall
                 * @return Macro-averaged F1 score
                 */
                float computePrecisionRecallF1(
                    const torch::Tensor& pred,
                    const torch::Tensor& target,
                    int numClasses,
                    float& precision,
                    float& recall);

                float computeAUCROC(
                    const torch::Tensor& scores,
					const torch::Tensor& labels);
                int _numClasses;  ///< Number of classes
            };

        } // namespace Validator
    } // namespace Core
} // namespace WheelDL
