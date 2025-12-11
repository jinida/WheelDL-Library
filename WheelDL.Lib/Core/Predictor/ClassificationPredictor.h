#pragma once

#include "../Engine/BasePredictor.h"
#include "../../Model/Task/ClassificationModel.h"
#include <opencv2/opencv.hpp>

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            /**
             * @class ClassificationPredictor
             * @brief Predictor for image classification tasks
             *
             * Performs batch inference for classification models:
             * - Multi-class classification with softmax
             *
             * Output (exported to JSON):
             * - Predicted class ID
             * - Confidence score
             * - Class name (if available)
             */
            class ClassificationPredictor : public BasePredictor {
            public:
                /**
                 * @brief Constructor
                 * @param config Configuration object
                 * @param checkpointPath Path to trained model checkpoint
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit ClassificationPredictor(
                    std::shared_ptr<Config::Configuration> config,
                    const std::string& checkpointPath,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    std::atomic<bool>* stopFlag = nullptr);

                /**
                 * @brief Destructor
                 */
                ~ClassificationPredictor() override = default;

            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup classification model
                 *
                 * Creates ClassificationModel from configuration.
                 * Model will be loaded from checkpoint later.
                 */
                void setupModel() override;

                /**
                 * @brief Postprocess classification output and store results
                 *
                 * Converts logits to class prediction with softmax.
                 *
                 * @param output Model output logits [1, num_classes]
                 * @param originalShape Original input shape
                 * @param imagePath Image file path
                 */
                void postprocess(
                    const std::vector<torch::Tensor>& output,
                    const std::tuple<int, int>& originalShape,
                    const std::string& imagePath) override;

                /**
                 * @brief Export classification results to JSON
                 * @param outputDir Output directory
                 */
                void exportResults(const std::string& outputDir) override;

                /**
                 * @brief Clear internal results container
                 */
                void clearResults() override;

            private:
                // Internal result container
                struct ClassificationResult {
                    std::string imagePath;
                    float score;
                    int classId;
                    std::pair<int, int> originalShape;
                    float inferenceTimeMs;
                };

                std::vector<ClassificationResult> _results;

                int _numClasses;  ///< Number of classes
            };

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
