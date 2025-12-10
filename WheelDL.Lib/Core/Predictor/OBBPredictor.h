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
             * Performs inference for YOLO-OBB style rotated object detection models:
             * - Multi-scale anchor-free detection
             * - Non-Maximum Suppression for OBBs using Probiou
             * - Rotated box decoding and coordinate scaling
             *
             * Output:
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
                 */
                explicit OBBPredictor(
                    std::shared_ptr<Config::Configuration> config,
                    const std::string& checkpointPath = ""
                );

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
                 * @brief Postprocess OBB detection output
                 *
                 * Decodes model output and applies NMS using Probiou.
                 *
                 * @param output Model output tensors (multi-scale predictions)
                 * @param originalShape Original input shape for coordinate scaling
                 * @return PredictionResult with:
                 *         - orientedBoxes: OBBs in original image coordinates
                 *         - classIds: Class indices
                 *         - scores: Confidence scores
                 */
                PredictionResult postprocess(
                    const std::vector<torch::Tensor>& output,
                    const std::tuple<int, int>& originalShape) override;

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
