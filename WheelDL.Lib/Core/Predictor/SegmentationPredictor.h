#pragma once

#include "../Engine/BasePredictor.h"
#include "../../Model/Task/SegmentationModel.h"
#include <opencv2/opencv.hpp>

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            /**
             * @class SegmentationPredictor
             * @brief Predictor for semantic/instance segmentation tasks
             *
             * Performs inference for segmentation models:
             * - Pixel-wise classification
             * - Multi-class segmentation with sigmoid activation
             * - Contour extraction from binary masks
             *
             * Output:
             * - Segmentation contours per class
             * - Class IDs
             * - Confidence scores (mean probability per segment)
             */
            class SegmentationPredictor : public BasePredictor {
            public:
                /**
                 * @brief Constructor
                 * @param config Configuration object
                 * @param checkpointPath Path to trained model checkpoint
                 */
                explicit SegmentationPredictor(
                    std::shared_ptr<Config::Configuration> config,
                    const std::string& checkpointPath = ""
                );

                /**
                 * @brief Destructor
                 */
                ~SegmentationPredictor() override = default;

            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup segmentation model
                 *
                 * Creates SegmentationModel from configuration.
                 * Model will be loaded from checkpoint later.
                 */
                void setupModel() override;

                /**
                 * @brief Preprocess input tensor
                 *
                 * Applies LetterBox resizing and normalization.
                 * Expected input: [C, H, W] or [1, C, H, W]
                 *
                 * @param input Raw input tensor
                 * @return Preprocessed tensor [1, C, H, W]
                 */
                torch::Tensor preprocess(const torch::Tensor& input) override;

                /**
                 * @brief Postprocess segmentation output
                 *
                 * Converts model output to segmentation contours.
                 * - Applies sigmoid to get probabilities
                 * - Thresholds to get binary masks
                 * - Extracts contours from masks
                 * - Scales contours to original image coordinates
                 *
                 * @param output Model output tensors (logits) [1, C, H, W]
                 * @param originalShape Original input shape for coordinate scaling
                 * @return PredictionResult with:
                 *         - contours: Segmentation contours per detected region
                 *         - classIds: Class indices for each contour
                 *         - scores: Mean confidence per contour
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
                float _minContourArea;
                bool _useImageNetNorm;
            };

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
