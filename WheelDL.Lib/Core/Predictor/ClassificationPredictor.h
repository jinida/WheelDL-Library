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
             * Performs inference for classification models:
             * - Multi-class classification with softmax
             *
             * Output:
             * - Predicted class ID
             * - Confidence score
             */
            class ClassificationPredictor : public BasePredictor {
            public:
                /**
                 * @brief Constructor
                 * @param config Configuration object
                 * @param checkpointPath Path to trained model checkpoint
                 */
                explicit ClassificationPredictor(
                    std::shared_ptr<Config::Configuration> config,
                    const std::string& checkpointPath = ""
                );

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
                 * @brief Preprocess input tensor
                 *
                 * Normalizes input and resizes to model input size.
                 * Expected input: [C, H, W] or [1, C, H, W]
                 *
                 * @param input Raw input tensor
                 * @return Preprocessed tensor [1, C, H, W]
                 */
                torch::Tensor preprocess(const torch::Tensor& input) override;

                /**
                 * @brief Postprocess classification output
                 *
                 * Converts logits to class prediction.
                 *
                 * @param output Model output logits [1, num_classes]
                 * @param originalShape Original input shape (not used for classification)
                 * @return PredictionResult with:
                 *         - classIds[0]: Predicted class index
                 *         - scores[0]: Confidence probability
                 */
                PredictionResult postprocess(
                    const std::vector<torch::Tensor>& output,
                    const std::tuple<int, int>& originalShape) override;

            private:
                std::string _modelYamlPath;  ///< Path to model YAML configuration
                int _numClasses;             ///< Number of classes

                // Normalization tensors
                torch::Tensor _mean;
                torch::Tensor _std;
                bool _useImageNetNorm;
            };

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
