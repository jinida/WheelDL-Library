#pragma once

#include <torch/torch.h>
#include <vector>
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
				constexpr int64_t EFFICIENTNET_B0_CHANNELS = 1280;
			}

			/**
			 * @brief YOLO Detect head for object detection models
			 *
			 * This class implements the detection head used in YOLO models for predicting
			 * bounding boxes and class probabilities.
			 */
			class DetectImpl : public torch::nn::Module {
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
			class ClassifyImpl : public torch::nn::Module {
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

				/**
				 * @brief Perform forward pass
				 *
				 * @param x Input tensor or list of tensors
				 * @return torch::Tensor Class predictions
				 */
				torch::Tensor forward(torch::Tensor x);

				torch::Tensor forwardMulti(std::vector < torch::Tensor> x);

				bool export_ = false;

			protected:
				Conv _conv = nullptr;
				torch::nn::AdaptiveAvgPool2d _pool = nullptr;
				torch::nn::Dropout _drop = nullptr;
				torch::nn::Linear _linear = nullptr;
			};

			TORCH_MODULE(Classify);

		} // namespace Modules
	} // namespace Model
} // namespace WheelDL
