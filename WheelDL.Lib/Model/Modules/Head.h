#pragma once

#include <torch/torch.h>
#include <vector>
#include "Interfaces.h"
#include "Conv.h"
#include "Block.h"
#include "Transformer.h"

namespace WheelDL {
	namespace Model {
		namespace Modules {

			// Constants for Head modules
			namespace HeadConstants {
				constexpr int64_t MIN_CHANNEL_SIZE = 16;
				constexpr int64_t MAX_CLASS_CHANNELS = 100;
				constexpr int64_t DEFAULT_REG_MAX = 16;
				constexpr double DEFAULT_INPUT_SIZE = 640.0;
				constexpr int64_t CLASSIFY_HIDDEN_CHANNELS = 256;
			}

			/**
			 * @brief YOLO Detect head for object detection models
			 *
			 * This class implements the detection head used in YOLO models for predicting
			 * bounding boxes and class probabilities.
			 */

			class IHeadBlockImpl : public torch::nn::Module
			{
			public:
				virtual std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x) = 0;
			};

			class DetectImpl : public IHeadBlockImpl {
			public:
				/**
				 * @brief Initialize the YOLO detection layer
				 *
				 * @param nc Number of classes
				 * @param ch Tuple of channel sizes from backbone feature maps
				 */
				DetectImpl(int64_t nc = 80, const std::vector<int64_t>& ch = {});

				/**
				 * @brief Perform forward pass
				 *
				 * @param x List of feature maps from different detection layers
				 * @return torch::Tensor or std::vector<torch::Tensor> depending on training mode
				 */
				std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);

				/**
				 * @brief Initialize detection head biases
				 */
				void biasInit();

				/**
				 * @brief Decode bounding boxes from predictions
				 *
				 * @param bboxes Predicted bounding boxes
				 * @param anchors Anchor points
				 * @param xywh Output format (xywh or xyxy)
				 * @return torch::Tensor Decoded bounding boxes
				 */
				torch::Tensor decodeBboxes(const torch::Tensor& bboxes, const torch::Tensor& anchors, bool xywh = true);

				// Public members
				bool export_ = false;

				int64_t nc;        // number of classes
				int64_t nl;        // number of detection layers
				int64_t regMax;    // DFL channels
				int64_t no;        // number of outputs per anchor

				torch::Tensor stride;
				torch::Tensor anchors;
				torch::Tensor strides;
				torch::Tensor shape;

				// Default input size for bias initialization (can be changed before biasInit)
				double inputSize = 640.0;

			protected:
				torch::Tensor _inference(const std::vector<torch::Tensor>& x);

				torch::nn::ModuleList _cv2 = nullptr;  // box regression layers
				torch::nn::ModuleList _cv3 = nullptr;  // classification layers
				torch::nn::AnyModule _dfl;   // DFL layer

				// Track input shape for anchor regeneration when input size changes
				std::vector<std::vector<int64_t>> _lastInputShapes;
			};

			TORCH_MODULE(Detect);

			/**
			 * @brief YOLO OBB detection head for detection with rotation
			 *
			 * This class extends the Detect head to include oriented bounding box prediction.
			 */
			class OBBImpl : public DetectImpl {
			public:
				/**
				 * @brief Initialize OBB detection head
				 *
				 * @param nc Number of classes
				 * @param ne Number of extra parameters (angle)
				 * @param ch Channel sizes from backbone
				 */
				OBBImpl(int64_t nc = 80, int64_t ne = 1, const std::vector<int64_t>& ch = {});

				/**
				 * @brief Perform forward pass
				 *
				 * @param x List of feature maps
				 * @return Detections with rotation angles
				 */
				std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);

				/**
				 * @brief Decode rotated bounding boxes
				 *
				 * @param bboxes Predicted bounding boxes
				 * @param anchors Anchor points
				 * @return torch::Tensor Decoded rotated bounding boxes
				 */
				torch::Tensor decodeBboxes(const torch::Tensor& bboxes, const torch::Tensor& anchors);

				int64_t ne;  // number of extra parameters
				torch::Tensor angle;

			protected:
				torch::nn::ModuleList _cv4 = nullptr;  // angle prediction layers
			};

			TORCH_MODULE(OBB);

			/**
			 * @brief YOLO classification head
			 *
			 * Transforms feature maps into class predictions.
			 */
			class ClassifyImpl : public IHeadBlockImpl {
			public:
				/**
				 * @brief Initialize classification head
				 *
				 * @param c1 Number of input channels
				 * @param c2 Number of output classes
				 * @param k Kernel size
				 * @param s Stride
				 * @param p Padding
				 * @param g Groups
				 */
				ClassifyImpl(int64_t c1, int64_t c2, int64_t k = 1, int64_t s = 1,
					std::optional<int64_t> p = std::nullopt, int64_t g = 1);

				std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x) override;

				bool export_ = false;

			protected:
				torch::Tensor _forward(torch::Tensor x);
				Conv _conv = nullptr;
				torch::nn::AdaptiveAvgPool2d _pool = nullptr;
				torch::nn::Dropout _drop = nullptr;
				torch::nn::Linear _linear = nullptr;
			};

			TORCH_MODULE(Classify);

			/**
			 * @brief Segmentation head
			 *
			 * Simple segmentation head with Conv(k=3) -> Conv(k=1) structure.
			 * During training, outputs logits (before sigmoid).
			 * During inference, outputs probabilities (after sigmoid).
			 */
			class SegmentImpl : public IHeadBlockImpl {
			public:
				/**
				 * @brief Initialize segmentation head
				 *
				 * @param c1 Number of input channels
				 * @param c2 Number of output channels (num_classes)
				 * @param act Activation function (default: "ReLU")
				 */
				SegmentImpl(int64_t c1, int64_t c2, const std::string& act = "ReLU");

				/**
				 * @brief Forward pass for segmentation
				 *
				 * @param x Input feature maps (vector format)
				 * @return std::vector<torch::Tensor> Segmentation output
				 *         - Training: logits [N, num_classes, H, W]
				 *         - Inference: probabilities [N, num_classes, H, W] (after sigmoid)
				 */
				std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x) override;
			protected:
				/**
				 * @brief Forward pass for single tensor
				 *
				 * @param x Input tensor
				 * @return torch::Tensor Segmentation output
				 */
				torch::Tensor _forward(torch::Tensor x);
				Conv _conv1 = nullptr;
				torch::nn::Conv2d _conv2 = nullptr;
				bool export_ = false;
			};

			TORCH_MODULE(Segment);

			class AnomalyImpl : public IHeadBlockImpl {
			public:
				/**
				 * @brief Initialize anomaly detection head with injected model
				 *
				 * @param anomalyModel Shared pointer to the anomaly detection model
				 */
				explicit AnomalyImpl(std::shared_ptr<IAnomalyModel> anomalyModel);

				/**
				 * @brief Forward pass
				 *
				 * Delegates to the underlying anomaly detection model.
				 * Input/output format depends on model type and training mode.
				 *
				 * @param x Input tensors from backbone
				 * @return std::vector<torch::Tensor> Model outputs
				 */
				std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x) override;

				/**
				 * @brief Get model type name
				 */
				std::string getModelType() const {
					return _anomalyModel ? _anomalyModel->getModelType() : "Unknown";
				}

				/**
				 * @brief Get the underlying anomaly model
				 * @return Shared pointer to the anomaly model
				 */
				std::shared_ptr<IAnomalyModel> getAnomalyModel() const {
					return _anomalyModel;
				}

				bool export_ = false;

			private:
				std::shared_ptr<IAnomalyModel> _anomalyModel;  ///< Anomaly detection model
			};

			TORCH_MODULE(Anomaly);

		} // namespace Modules
	} // namespace Model
} // namespace WheelDL
