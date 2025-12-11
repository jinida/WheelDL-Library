#pragma once

#include "../Engine/BasePredictor.h"
#include "../../Model/Task/OBBModel.h"
#include "../../Model/Utils/IoU.h"

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            /**
             * @class OBBPredictor
             * @brief Predictor for oriented bounding box (OBB) detection tasks
             *
             * Performs batch inference for YOLO-OBB style rotated object detection models:
             * - Multi-scale anchor-free detection
             * - Non-Maximum Suppression for OBBs using Probiou
             * - Rotated box decoding and coordinate scaling
             *
             * Output (exported to JSON):
             * - Oriented bounding boxes (cx, cy, w, h, angle)
             * - Class IDs
             * - Confidence scores
             */
            class OBBPredictor : public BasePredictor {
            public:
                /**
                 * @brief Constructor
                 * @param config Configuration object
                 * @param checkpointPath Path to trained model checkpoint
                 * @param logger Logger instance (non-null, owned by Launcher)
                 * @param profiler PerformanceProfiler instance (non-null, owned by Launcher)
                 * @param stopFlag Atomic stop flag (optional, owned by Context)
                 */
                explicit OBBPredictor(
                    std::shared_ptr<Config::Configuration> config,
                    const std::string& checkpointPath,
                    WheelDL::Utils::Logger* logger,
                    WheelDL::Utils::PerformanceProfiler* profiler,
                    std::atomic<bool>* stopFlag = nullptr);

                /**
                 * @brief Destructor
                 */
                ~OBBPredictor() override = default;

            protected:
                // ========== Hook Methods Implementation ==========

                /**
                 * @brief Setup OBB detection model
                 *
                 * Creates OBBModel from configuration.
                 * Model will be loaded from checkpoint later.
                 */
                void setupModel() override;

                /**
                 * @brief Postprocess OBB detection output and store results
                 *
                 * Decodes model output, applies NMS using Probiou,
                 * scales OBBs, and stores results in internal container.
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
                 * @brief Export OBB detection results to JSON
                 * @param outputDir Output directory
                 */
                void exportResults(const std::string& outputDir) override;

                /**
                 * @brief Clear internal results container
                 */
                void clearResults() override;

            private:
                /**
                 * @brief Scale OBBs from model input size to original image size
                 * @param obbs OBBs in model input coordinates [N, 5]
                 * @param inputSize Model input size (square)
                 * @param originalShape Original image shape (height, width)
                 * @return Scaled OBBs in original image coordinates
                 */
                torch::Tensor scaleOBBs(
                    const torch::Tensor& obbs,
                    int inputSize,
                    const std::tuple<int, int>& originalShape);

                // Internal result container
                struct OBBResult {
                    std::string imagePath;
                    std::vector<OBB> orientedBoxes;
                    std::vector<float> scores;
                    std::vector<int> classIds;
                    std::pair<int, int> originalShape;
                    float inferenceTimeMs;
                };

                std::vector<OBBResult> _results;

                int _numClasses;             ///< Number of classes
                float _confThresh;           ///< Confidence threshold
                float _iouThresh;            ///< IoU threshold for NMS
                int _maxDet;                 ///< Maximum detections per image
            };

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
