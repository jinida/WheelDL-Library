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
             * Performs batch inference for anomaly detection models:
             * - EfficientAD: Student-Teacher distillation based detection
             * - PatchCore: Memory bank based detection
             * - SimpleNet: Simple feature-based detection
             *
             * Output (exported to JSON + anomaly map images):
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
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit AnomalyPredictor(
                    std::shared_ptr<Config::Configuration> config,
                    const std::string& checkpointPath,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    std::atomic<bool>* stopFlag = nullptr);

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
                 * @brief Postprocess anomaly detection output and store results
                 *
                 * Converts anomaly score map to internal result container.
                 * Extracts contours for anomalous regions.
                 *
                 * @param output Anomaly score map [H, W] or [1, H, W]
                 * @param originalShape Original image shape
                 * @param imagePath Image file path
                 */
                void postprocess(
                    const std::vector<torch::Tensor>& output,
                    const std::tuple<int, int>& originalShape,
                    const std::string& imagePath) override;

                /**
                 * @brief Export anomaly detection results to JSON and images
                 * @param outputDir Output directory
                 */
                void exportResults(const std::string& outputDir) override;

                /**
                 * @brief Clear internal results container
                 */
                void clearResults() override;

            private:
                // Internal result container
                struct AnomalyResult {
                    std::string imagePath;
                    float anomalyScore;
                    int classId;  // 0=normal, 1=anomaly
                    cv::Mat anomalyMap;
                    std::pair<int, int> originalShape;
                    float inferenceTimeMs;
                };

                std::vector<AnomalyResult> _results;
            };

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
