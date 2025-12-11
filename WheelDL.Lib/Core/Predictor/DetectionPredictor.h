#pragma once

#include "../Engine/BasePredictor.h"
#include "../../Model/Task/DetectionModel.h"
#include "../../Model/Utils/IoU.h"
#include <opencv2/opencv.hpp>

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            /**
             * @class DetectionPredictor
             * @brief Predictor for object detection tasks
             *
             * Performs batch inference for YOLO-style detection models:
             * - Multi-scale anchor-free detection
             * - Non-Maximum Suppression (NMS) via Model::Utils::nonMaxSuppression
             * - Box decoding and coordinate scaling
             *
             * Output (exported to JSON):
             * - Bounding boxes (x1, y1, x2, y2)
             * - Class IDs
             * - Confidence scores
             */
            class DetectionPredictor : public BasePredictor {
            public:
                /**
                 * @brief Constructor
                 * @param config Configuration object
                 * @param checkpointPath Path to trained model checkpoint
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit DetectionPredictor(
                    std::shared_ptr<Config::Configuration> config,
                    const std::string& checkpointPath,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    std::atomic<bool>* stopFlag = nullptr);

                /**
                 * @brief Destructor
                 */
                ~DetectionPredictor() override = default;

            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup detection model
                 *
                 * Creates DetectionModel from configuration.
                 * Model will be loaded from checkpoint later.
                 */
                void setupModel() override;

                /**
                 * @brief Postprocess detection output and store results
                 *
                 * Decodes model output, applies NMS, scales boxes,
                 * and stores results in internal container.
                 *
                 * @param output Model output tensors
                 * @param originalShape Original input shape for coordinate scaling
                 * @param imagePath Image file path
                 */
                void postprocess(
                    const std::vector<torch::Tensor>& output,
                    const std::tuple<int, int>& originalShape,
                    const std::string& imagePath) override;

                /**
                 * @brief Export detection results to JSON
                 * @param outputDir Output directory
                 */
                void exportResults(const std::string& outputDir) override;

                /**
                 * @brief Clear internal results container
                 */
                void clearResults() override;

            private:
                /**
                 * @brief Scale boxes from model input size to original image size
                 * @param boxes Boxes in model input coordinates [N, 4]
                 * @param inputSize Model input size (square)
                 * @param originalShape Original image shape (height, width)
                 * @return Scaled boxes in original image coordinates
                 */
                torch::Tensor scaleBoxes(
                    const torch::Tensor& boxes,
                    int inputSize,
                    const std::tuple<int, int>& originalShape);

                // Internal result container
                struct DetectionResult {
                    std::string imagePath;
                    std::vector<BBox> boxes;
                    std::vector<float> scores;
                    std::vector<int> classIds;
                    std::pair<int, int> originalShape;
                    float inferenceTimeMs;
                };

                std::vector<DetectionResult> _results;

                int _numClasses;             ///< Number of classes
                float _confThresh;           ///< Confidence threshold
                float _iouThresh;            ///< IoU threshold for NMS
                int _maxDet;                 ///< Maximum detections per image
            };

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
