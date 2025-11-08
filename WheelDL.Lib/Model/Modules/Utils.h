#pragma once

#include <torch/torch.h>
#include <vector>

namespace WheelDL {
    namespace Model {
        namespace Modules {

            /**
             * @brief Initialize conv/fc bias value according to a given probability value
             *
             * Calculates the bias initialization value based on a prior probability.
             * Commonly used in object detection models to initialize classification layers.
             *
             * @param priorProb Prior probability for bias initialization (default: 0.01)
             * @return float Bias initialization value
             */
            float biasInitWithProb(float priorProb = 0.01f);

            /**
             * @brief Initialize the weights and biases of a linear module
             *
             * Initializes weights using uniform distribution within bounds calculated
             * from the input dimension.
             *
             * @param module Linear module to initialize
             */
            void linearInit(torch::nn::Linear& module);

            /**
             * @brief Create an activation module by name
             *
             * Creates an activation function module based on the provided name string.
             * Supported activations: ReLU, SiLU, LeakyReLU, GELU, Mish, Hardswish, Identity, AGLU
             * Empty string returns Identity (no activation).
             * Unknown names default to ReLU.
             *
             * @param actName Name of the activation function (case-insensitive)
             * @return torch::nn::AnyModule Activation module
             */
            torch::nn::AnyModule createActivation(const std::string& actName = "ReLU");

            /**
             * @brief Calculate the inverse sigmoid function for a tensor
             *
             * Applies the inverse of the sigmoid function to a tensor.
             * Useful in attention mechanisms and coordinate transformations.
             *
             * @param x Input tensor with values in range [0, 1]
             * @param eps Small epsilon value to prevent numerical instability (default: 1e-5)
             * @return torch::Tensor Tensor after applying inverse sigmoid
             */
            torch::Tensor inverseSigmoid(const torch::Tensor& x, float eps = 1e-5f);

            /**
             * @brief Multi-scale deformable attention PyTorch implementation
             *
             * Performs deformable attention across multiple feature map scales.
             *
             * @param value Value tensor with shape (bs, numKeys, numHeads, embedDims)
             * @param valueSpatialShapes Spatial shapes with shape (numLevels, 2)
             * @param samplingLocations Sampling locations with shape (bs, numQueries, numHeads, numLevels, numPoints, 2)
             * @param attentionWeights Attention weights with shape (bs, numQueries, numHeads, numLevels, numPoints)
             * @return torch::Tensor Output tensor with shape (bs, numQueries, embedDims)
             */
            torch::Tensor multiScaleDeformableAttnPytorch(
                const torch::Tensor& value,
                const torch::Tensor& valueSpatialShapes,
                const torch::Tensor& samplingLocations,
                const torch::Tensor& attentionWeights
            );

            /**
             * @brief Convert distance predictions to bounding boxes
             *
             * Decodes distance predictions from anchor-based detectors into bounding box coordinates.
             *
             * @param distance Distance tensor with shape (bs, 4*reg_max, h*w)
             * @param anchors Anchor points tensor with shape (h*w, 2)
             * @param xywh If true, output format is xywh; otherwise xyxy (default: true)
             * @param dim Dimension along which to operate (default: 1)
             * @return torch::Tensor Decoded bounding boxes
             */
            torch::Tensor dist2bbox(
                const torch::Tensor& distance,
                const torch::Tensor& anchors,
                bool xywh = true,
                int64_t dim = 1
            );

            /**
             * @brief Convert distance predictions to rotated bounding boxes
             *
             * Decodes distance predictions and rotation angles into oriented bounding box coordinates.
             *
             * @param distance Distance tensor with shape (bs, 4*reg_max, h*w)
             * @param angle Rotation angle tensor
             * @param anchors Anchor points tensor with shape (h*w, 2)
             * @param dim Dimension along which to operate (default: 1)
             * @return torch::Tensor Decoded rotated bounding boxes
             */
            torch::Tensor dist2rbox(
                const torch::Tensor& distance,
                const torch::Tensor& angle,
                const torch::Tensor& anchors,
                int64_t dim = 1
            );

            /**
             * @brief Generate anchor points and stride tensors
             *
             * Creates anchor points for each feature map level along with corresponding strides.
             *
             * @param feats Vector of feature map tensors from different levels
             * @param stride Stride tensor for each level
             * @param gridCellOffset Grid cell offset (default: 0.5)
             * @return std::pair<torch::Tensor, torch::Tensor> Pair of (anchors, strides)
             */
            std::pair<torch::Tensor, torch::Tensor> makeAnchors(
                const std::vector<torch::Tensor>& feats,
                const torch::Tensor& stride,
                double gridCellOffset = 0.5
            );

        } // namespace Modules
    } // namespace Model
} // namespace WheelDL
