#include "pch.h"
#include "AnomalyModel.h"
#include "../Builder/ModelBuilder.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <iostream>

namespace WheelDL {
    namespace Model {

        using namespace Loss;
        using namespace Builder;
        using namespace Config;

        AnomalyModel::AnomalyModel(std::shared_ptr<Configuration> config,
            const std::string& modelYamlPath)
            : _lossType(AnomalyLoss::LossType::MSE),
            _ssimWeight(0.0f)
        {
            _taskType = TaskType::ANOMALY;

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
                if (config->getTaskType() != TaskType::ANOMALY) {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type is not ANOMALY"
                    );
                }

                // Get loss parameters from configuration if available
                // For now, use default values
                _ssimWeight = 0.0f;

                // Get model parameters from configuration
                auto numClasses = config->getNumClasses();  // May be 1 for anomaly detection
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

                // Initialize criterion
                _criterion = initCriterion();
                _isInitialized = true;

            }
            catch (const std::exception& e) {
                throw WheelDL::Utils::ModelException(
                    WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                    "Failed to initialize AnomalyModel: " + std::string(e.what())
                );
            }
        }

        bool AnomalyModel::loadPretrained(const std::string& weightsPath)
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

        std::unique_ptr<BaseLoss> AnomalyModel::initCriterion()
        {
            // Create and return anomaly loss
            return std::make_unique<AnomalyLoss>(
                _lossType,
                _ssimWeight
            );
        }

    } // namespace Model
} // namespace WheelDL