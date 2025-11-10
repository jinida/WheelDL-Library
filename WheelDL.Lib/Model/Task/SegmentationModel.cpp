#include "pch.h"
#include "SegmentationModel.h"
#include "../Builder/ModelBuilder.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <iostream>

namespace WheelDL {
    namespace Model {

        using namespace Loss;
        using namespace Builder;
        using namespace Config;

        SegmentationModel::SegmentationModel(std::shared_ptr<Configuration> config,
            const std::string& modelYamlPath)
            : _diceWeight(0.5f)
        {
            _taskType = TaskType::SEGMENTATION;

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
                if (config->getTaskType() != TaskType::SEGMENTATION) {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type is not SEGMENTATION"
                    );
                }

                // Get loss parameters from configuration if available
                // For now, use default value
                _diceWeight = 0.5f;  // Balance between CE and Dice loss

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
                _saveIndices = builder.getSaveIndices();
                _headInputIndices = builder.getHeadInputIndices();

                // Initialize criterion
                _criterion = initCriterion();
                _isInitialized = true;

            }
            catch (const std::exception& e) {
                throw WheelDL::Utils::ModelException(
                    WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                    "Failed to initialize SegmentationModel: " + std::string(e.what())
                );
            }
        }

        bool SegmentationModel::loadPretrained(const std::string& weightsPath)
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

        std::unique_ptr<BaseLoss> SegmentationModel::initCriterion()
        {
            // Create and return segmentation loss
            return std::make_unique<SegmentationLoss>(
                _config->getNumClasses(),
                _diceWeight
            );
        }

    } // namespace Model
} // namespace WheelDL