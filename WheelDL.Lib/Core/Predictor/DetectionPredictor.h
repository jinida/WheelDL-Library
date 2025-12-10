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
             * Performs inference for YOLO-style detection models:
             * - Multi-scale anchor-free detection
             * - Non-Maximum Suppression (NMS) via Model::Utils::nonMaxSuppression
             * - Box decoding and coordinate scaling
             *
             * Output:
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
                 */
                explicit DetectionPredictor(
                    std::shared_ptr<Config::Configuration> config,
                    const std::string& checkpointPath = ""
                );

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
                 * @brief Postprocess detection output
                 *
                 * Decodes model output and applies NMS using Model::Utils::nonMaxSuppression.
                 *
                 * @param output Model output tensors (multi-scale predictions)
                 * @param originalShape Original input shape for coordinate scaling
                 * @return PredictionResult with:
                 *         - boxes: Bounding boxes in original image coordinates
                 *         - classIds: Class indices
                 *         - scores: Confidence scores
                 */
                PredictionResult postprocess(
                    const std::vector<torch::Tensor>& output,
                    const std::tuple<int, int>& originalShape) override;

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

                std::string _modelYamlPath;  ///< Path to model YAML configuration
                int _numClasses;             ///< Number of classes
                float _confThresh;           ///< Confidence threshold
                float _iouThresh;            ///< IoU threshold for NMS
                int _maxDet;                 ///< Maximum detections per image

                // Normalization tensors
                torch::Tensor _mean;
                torch::Tensor _std;
            };

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
