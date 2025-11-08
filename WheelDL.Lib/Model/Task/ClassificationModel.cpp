#include "ClassificationModel.h"
#include "../Builder/ModelBuilder.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <iostream>

namespace WheelDL {
    namespace Model {

        using namespace Loss;
        using namespace Builder;
        using namespace Config;

        ClassificationModel::ClassificationModel(std::shared_ptr<Configuration> config,
            const std::string& modelYamlPath)
            : _lossType(ClassificationLoss::LossType::CROSS_ENTROPY)
        {
            _taskType = TaskType::CLASSIFICATION;

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
                if (config->getTaskType() != TaskType::CLASSIFICATION) {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type is not CLASSIFICATION"
                    );
                }

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
                    "Failed to initialize ClassificationModel: " + std::string(e.what())
                );
            }
        }

        bool ClassificationModel::loadPretrained(const std::string& weightsPath)
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

        std::unique_ptr<BaseLoss> ClassificationModel::initCriterion()
        {
            // Get loss parameters from configuration
            float labelSmoothing = 0.0f;

            // Create and return classification loss
            return std::make_unique<ClassificationLoss>(
                _lossType,
                _config->getNumClasses(),
                labelSmoothing
            );
        }
    } // namespace Model
} // namespace WheelDL