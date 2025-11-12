#include "pch.h"
#include "DetectionModel.h"
#include "../Builder/ModelBuilder.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <iostream>

namespace WheelDL {
    namespace Model {

        using namespace Loss;
        using namespace Builder;
        using namespace Config;

        DetectionModel::DetectionModel(std::shared_ptr<Configuration> config,
            const std::string& modelYamlPath)
            : _boxGain(0.0f), _clsGain(0.0f), _dflGain(0.0f)
        {
            _taskType = TaskType::DETECTION;

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
                if (config->getTaskType() != TaskType::DETECTION) {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type is not DETECTION"
                    );
                }

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

                // Calculate stride for detection layers
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

                _criterion = initCriterion();
                _isInitialized = true;

            }
            catch (const std::exception& e) {
                throw WheelDL::Utils::ModelException(
                    WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                    "Failed to initialize DetectionModel: " + std::string(e.what())
                );
            }
        }

        bool DetectionModel::loadPretrained(const std::string& weightsPath)
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

        std::unique_ptr<BaseLoss> DetectionModel::initCriterion()
        {
            if (_stride.numel() == 0) {
                throw WheelDL::Utils::ModelException(
                    WheelDL::Utils::ErrorCode::MODEL_INVALID_ARCHITECTURE,
                    "Stride not initialized - model architecture may be invalid"
                );
            }

            // Create and return detection loss
            return std::make_unique<DetectionLoss>(
                _config->getNumClasses(),
                _stride,
                _boxGain,
                _clsGain,
                _dflGain
            );
        }

        torch::Tensor DetectionModel::calculateStride(int64_t imageSize)
        {
            // Create a dummy input to calculate stride
            torch::NoGradGuard no_grad;
            auto dummyInput = torch::zeros({1, 3, imageSize, imageSize});

            // Move to same device as model
            if (!_model->parameters().empty() && _model->parameters().begin()->is_cuda())
            {
                dummyInput = dummyInput.cuda();
            }

            // Run forward pass to get predictions
            std::vector<torch::Tensor> predictions;

            predictions = predict(dummyInput);
            std::vector<float> strideValues;

            for (const auto& pred : predictions) {
                // For detection, predictions shape is [batch, channels, height, width]
                if (pred.dim() == 4) {
                    auto outputSize = pred.size(2);  // height of feature map
                    float stride = static_cast<float>(imageSize) / static_cast<float>(outputSize);
                    strideValues.push_back(stride);
                }
            }

            // If we couldn't determine stride, use default values
            if (strideValues.empty())
            {
                strideValues = {8.0f, 16.0f, 32.0f};
            }

            // Convert to tensor
            return torch::tensor(strideValues);
        }

    } // namespace Model
} // namespace WheelDL