#include "pch.h"
#include "OBBModel.h"
#include "../Builder/ModelBuilder.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <iostream>

namespace WheelDL {
    namespace Model {

        using namespace Loss;
        using namespace Builder;
        using namespace Config;

        OBBModel::OBBModel(std::shared_ptr<Configuration> config,
            const std::string& modelYamlPath)
            : _boxGain(0.0f), _clsGain(0.0f), _dflGain(0.0f)
        {
            _taskType = TaskType::OBB;

            if (!config) {
                throw WheelDL::Utils::ConfigurationException(
                    WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                    "Configuration is null"
                );
            }

            if (modelYamlPath.empty()) {
                throw WheelDL::Utils::ConfigurationException(
                    WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                    "Model YAML path is empty"
                );
            }

            try {
                // Store configuration
                _config = config;

                // Verify task type
                if (config->getTaskType() != TaskType::OBB) {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type is not OBB"
                    );
                }

                // Get loss weights from configuration
                _boxGain = config->getBoxGain();
                _clsGain = config->getClsGain();
                _dflGain = config->getDFLGain();

                // Get model parameters from configuration
                auto numClasses = config->getNumClasses();
                auto imageSize = config->getImageSize();

                // Create model builder with parameters
                ModelBuilder builder;
                builder.setYamlPath(modelYamlPath)
                    .setNumClasses(numClasses)
                    .setImageSize(imageSize);

                // Build model
                auto model = builder.build();

                // Set the model
                setModel(model);

                // Get metadata from builder
                setFromIndices(builder.getFromIndices());
                setSaveIndices(builder.getSaveIndices());

                // Calculate stride for OBB detection layers
                // Use smaller fixed size for efficiency (256 is sufficient to determine stride)
                _stride = calculateStride(256);
                auto maxStride = _stride.max().item<float>();
                auto adjustedImageSize = static_cast<int>(
                    std::ceil(static_cast<float>(imageSize) / maxStride) * maxStride
                );

                if (adjustedImageSize != imageSize) {
                    std::cerr << "Warning: Adjusted image size from "
                        << imageSize << " to " << adjustedImageSize
                        << " to be compatible with model stride." << std::endl;
                    _config->setImageSize(adjustedImageSize);
                }

                // Initialize criterion
                _criterion = initCriterion();
                _isInitialized = true;

            }
            catch (const std::exception& e) {
                throw WheelDL::Utils::ModelException(
                    WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                    "Failed to initialize OBBModel: " + std::string(e.what())
                );
            }
        }

        bool OBBModel::loadPretrained(const std::string& weightsPath)
        {
            try {
                loadWeights(weightsPath);
                return true;
            }
            catch (const std::exception& e) {
                std::cerr << "Warning: Failed to load pretrained weights: "
                    << e.what() << std::endl;
                return false;
            }
        }

        std::unique_ptr<BaseLoss> OBBModel::initCriterion()
        {
            if (_stride.numel() == 0) {
                throw WheelDL::Utils::ModelException(
                    WheelDL::Utils::ErrorCode::MODEL_INVALID_ARCHITECTURE,
                    "Stride not initialized - model architecture may be invalid"
                );
            }

            // Create and return OBB loss
            return std::make_unique<OBBLoss>(
                _config->getNumClasses(),
                _stride,
                _boxGain,
                _clsGain,
                _dflGain
            );
        }

        torch::Tensor OBBModel::calculateStride(int64_t imageSize)
        {
            // Create a dummy input to calculate stride
            torch::NoGradGuard no_grad;
            auto dummyInput = torch::zeros({1, 3, imageSize, imageSize});

            // Move to same device as model
            if (!_model->parameters().empty() && _model->parameters().begin()->is_cuda()) {
                dummyInput = dummyInput.cuda();
            }

            // Run forward pass to get predictions
            std::vector<torch::Tensor> predictions;
            try {
                predictions = predict(dummyInput);
            }
            catch (const std::exception& e) {
                // If predict fails, try with standard forward for OBB
                // OBB models typically output 3 scales like detection
                predictions = forward(dummyInput);
            }

            // Calculate stride for each prediction scale
            std::vector<float> strideValues;

            for (const auto& pred : predictions) {
                // For OBB, predictions shape is [batch, channels, height, width]
                // where channels = 4*regMax + numClasses + 1 (angle)
                if (pred.dim() == 4) {
                    auto outputSize = pred.size(2);  // height of feature map
                    float stride = static_cast<float>(imageSize) / static_cast<float>(outputSize);
                    strideValues.push_back(stride);
                }
                // For concatenated format [batch, channels, num_anchors]
                else if (pred.dim() == 3) {
                    // For standard YOLO-OBB with 3 scales
                    // Default stride values for P3, P4, P5
                    strideValues = {8.0f, 16.0f, 32.0f};
                    break;
                }
            }

            // If we couldn't determine stride, use default values
            if (strideValues.empty()) {
                strideValues = {8.0f, 16.0f, 32.0f};  // Standard YOLO strides
            }

            // Convert to tensor
            return torch::tensor(strideValues);
        }

    } // namespace Model
} // namespace WheelDL