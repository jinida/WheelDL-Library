#pragma once

#include "../Engine/BasePredictor.h"
#include "../../Model/Task/AnomalyModel.h"
#include <opencv2/opencv.hpp>

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            /**
             * @class AnomalyPredictor
             * @brief Predictor for anomaly detection tasks
             *
             * Performs inference for anomaly detection models:
             * - EfficientAD: Student-Teacher distillation based detection
             * - PatchCore: Memory bank based detection
             * - SimpleNet: Simple feature-based detection
             *
             * Output:
             * - Anomaly score map (pixel-level)
             * - Image-level anomaly score
             * - Anomaly contours (segmentation of anomalous regions)
             */
            class AnomalyPredictor : public BasePredictor {
            public:
                /**
                 * @brief Constructor
                 * @param config Configuration object
                 * @param checkpointPath Path to trained model checkpoint
                 */
                explicit AnomalyPredictor(
                    std::shared_ptr<Config::Configuration> config,
                    const std::string& checkpointPath = ""
                );

                /**
                 * @brief Destructor
                 */
                ~AnomalyPredictor() override = default;

                /**
                 * @brief Set anomaly threshold
                 * @param threshold Threshold for anomaly detection [0, 1]
                 */
                void setThreshold(float threshold) {
                    _threshold = threshold;
                }

                /**
                 * @brief Get current threshold
                 */
                float getThreshold() const {
                    return _threshold;
                }

            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup anomaly detection model
                 *
                 * Creates AnomalyModel from configuration.
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
                 * @brief Postprocess anomaly detection output
                 *
                 * Converts anomaly score map to PredictionResult.
                 * Extracts contours for anomalous regions.
                 *
                 * @param output Anomaly score map [H, W] or [1, H, W]
                 * @param originalInput Original input tensor (for dimensions)
                 * @return PredictionResult with:
                 *         - scores: [image-level anomaly score]
                 *         - classIds: [0 or 1] (0=normal, 1=anomaly)
                 *         - contours: anomaly region contours
                 */
                PredictionResult postprocess(
                    const torch::Tensor& output,
                    const torch::Tensor& originalInput) override;

            private:
                /**
                 * @brief Extract contours from anomaly score map
                 *
                 * Applies threshold to create binary mask, then extracts contours.
                 *
                 * @param anomalyMap Anomaly score map [H, W]
                 * @param threshold Anomaly threshold [0, 1]
                 * @return Vector of contours
                 */
                std::vector<Contour> extractAnomalyContours(
                    const torch::Tensor& anomalyMap,
                    float threshold);

                /**
                 * @brief Compute image-level anomaly score
                 *
                 * Computes max or mean of anomaly score map.
                 *
                 * @param anomalyMap Anomaly score map [H, W]
                 * @return Image-level score [0, 1]
                 */
                float computeImageLevelScore(const torch::Tensor& anomalyMap);

            private:
                std::string _modelYamlPath;  /// Path to model YAML configuration
                float _threshold;            /// Anomaly detection threshold [0, 1]
                bool _isModelPrepared;       /// Whether model has been prepared
            };

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
