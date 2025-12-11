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
             * Performs batch inference for segmentation models:
             * - Pixel-wise classification
             * - Multi-class segmentation with sigmoid activation
             * - Contour extraction from binary masks
             *
             * Output (exported to JSON):
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
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit SegmentationPredictor(
                    std::shared_ptr<Config::Configuration> config,
                    const std::string& checkpointPath,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    std::atomic<bool>* stopFlag = nullptr);

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
                 * @brief Postprocess segmentation output and store results
                 *
                 * Converts model output to segmentation contours:
                 * - Applies sigmoid to get probabilities
                 * - Thresholds to get binary masks
                 * - Extracts contours from masks
                 * - Scales contours to original image coordinates
                 *
                 * @param output Model output tensors (logits) [1, C, H, W]
                 * @param originalShape Original input shape for coordinate scaling
                 * @param imagePath Image file path
                 */
                void postprocess(
                    const std::vector<torch::Tensor>& output,
                    const std::tuple<int, int>& originalShape,
                    const std::string& imagePath) override;

                /**
                 * @brief Export segmentation results to JSON
                 * @param outputDir Output directory
                 */
                void exportResults(const std::string& outputDir) override;

                /**
                 * @brief Clear internal results container
                 */
                void clearResults() override;

            private:
                // Internal result container
                struct SegmentationResult {
                    std::string imagePath;
                    std::vector<Contour> contours;
                    std::vector<float> scores;
                    std::vector<int> classIds;
                    std::pair<int, int> originalShape;
                    float inferenceTimeMs;
                };

                std::vector<SegmentationResult> _results;

                int _numClasses;             ///< Number of classes
                float _minContourArea;       ///< Minimum contour area
            };

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
