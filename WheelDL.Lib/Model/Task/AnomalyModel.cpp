#include "pch.h"
#include "AnomalyModel.h"
#include "../Modules/Model.h"
#include "../Modules/Head.h"
#include "../Builder/ModelBuilder.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <iostream>

namespace WheelDL {
    namespace Model {

        using namespace Loss;
        using namespace Builder;
        using namespace Config;

        AnomalyModel::AnomalyModel(std::shared_ptr<Configuration> config)
            : _lossType(AnomalyLoss::LossType::EfficientAD)
        {
            _taskType = TaskType::ANOMALY;

            if (!config) {
                throw WheelDL::Utils::ConfigurationException(
                    WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                    "Configuration is null"
                );
            }

			auto modelYamlPath = config->getModelPath();
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

                determineLossType();

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
                _lossType
            );
        }

        void AnomalyModel::determineLossType()
        {
            for (int i = 0; i < _model->size(); ++i)
            {
                auto module = _model->ptr(i);
                // Check if module is a Head block
                auto headBlock = std::dynamic_pointer_cast<Modules::AnomalyImpl>(module);
                if (headBlock)
                {
                    // Extract and store anomaly model
                    _anomalyModel = headBlock->getAnomalyModel();

                    std::string modelType = headBlock->getModelType();
                    if (modelType == "EfficientAD") {
                        _lossType = AnomalyLoss::LossType::EfficientAD;
						_config->setIsEfficientAD(true);
                        return;
                    }
                    else if (modelType == "PatchCore") 
                    {
                        _lossType = AnomalyLoss::LossType::PatchCore;
						_config->setIsPatchCore(true);
                        return;
                    }
                    else if (modelType == "SimpleNet") {
                        _lossType = AnomalyLoss::LossType::SimpleNet;
                        return;
                    }
                    else {
                        throw WheelDL::Utils::ConfigurationException(
                            WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                            "Unsupported anomaly model type: " + modelType
                        );
                    }
                }
            }

            // No Anomaly head found
            throw WheelDL::Utils::ModelException(
                WheelDL::Utils::ErrorCode::MODEL_INVALID_ARCHITECTURE,
                "No Anomaly head found in model sequence"
            );
        }
    } // namespace Model
} // namespace WheelDL