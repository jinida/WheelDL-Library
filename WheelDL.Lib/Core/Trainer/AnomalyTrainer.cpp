#include "pch.h"
#include "WheelDL.Lib/Data/Transforms/Collation.h"
#include "AnomalyTrainer.h"
#include "../Validator/AnomalyValidator.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            AnomalyTrainer::AnomalyTrainer(const std::shared_ptr<Config::Configuration> config)
                : BaseTrainer(config)
                , _isModelPrepared(false)
                , _isValidationPrepared(false)
            {
                _logger->info("AnomalyTrainer", "Initializing anomaly detection trainer");

                // Get model YAML path from configuration
                _modelYamlPath = config->getModelPath();
                if (_modelYamlPath.empty()) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Model YAML path is not specified in configuration"
                    );
                }

                // Verify task type
                if (config->getTaskType() != TaskType::ANOMALY) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be ANOMALY for AnomalyTrainer"
                    );
                }

                _logger->info("AnomalyTrainer", "Model YAML: " + _modelYamlPath);
            }

            void AnomalyTrainer::setupModel()
            {
                _profiler.start("setup_model");
                _logger->info("AnomalyTrainer", "Setting up anomaly detection model");

                try {
                    // Create AnomalyModel with config and model YAML path
                    _model = std::make_unique<Model::AnomalyModel>(_config);
                    _model->to(_device);
                    _logger->info("AnomalyTrainer", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup anomaly model: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_model");
            }

            void AnomalyTrainer::setupDataLoaders()
            {
                _profiler.start("setup_dataloaders");
                _logger->info("AnomalyTrainer", "Setting up data loaders");

                try {
                    auto trainDataset = std::make_shared<Data::Dataset::AnomalyDataset>(*_config, true);

                    _logger->info("AnomalyTrainer", "Training dataset created with " +
                        std::to_string(trainDataset->size().value_or(0)) + " samples");

                    // Create DataLoader with LibTorch
                    auto trainLoader = torch::data::make_data_loader(
                        std::move(trainDataset->map(Data::DataExampleCollation())),
                        torch::data::DataLoaderOptions()
                            .batch_size(_config->getBatchSize())
                            .workers(_config->getWorkers())
                            .drop_last(false)
                    );

                    // Prepare model for training BEFORE moving the loader
                    // This computes teacher feature params (EfficientAD) or builds memory bank (PatchCore)
                    if (!_isModelPrepared) {
                        _profiler.start("prepare_model_training");
                        _logger->info("AnomalyTrainer", "Preparing model for training...");

                        try {
                            auto anomalyModel = dynamic_cast<Model::AnomalyModel*>(_model.get());
                            if (!anomalyModel) {
                                throw Utils::ModelException(
                                    Utils::ErrorCode::MODEL_INVALID_ARCHITECTURE,
                                    "Model is not an AnomalyModel"
                                );
                            }

                            // Use the trainLoader we just created for preparation
                            anomalyModel->prepareTraining(*trainLoader);

                            _isModelPrepared = true;
                            _logger->info("AnomalyTrainer", "Model preparation completed");
                        }
                        catch (const std::exception& e) {
                            _logger->error("AnomalyTrainer", "Failed to prepare model: " + std::string(e.what()));
                            throw;
                        }

                        _profiler.stop("prepare_model_training");
                    }

                    // Now move the loader to the trainer
                    setTrainDataLoader(std::move(trainLoader));

                    _logger->info("AnomalyTrainer", "Data loaders setup completed");
                }
                catch (const std::exception& e) {
                    throw Utils::DataException(
                        Utils::ErrorCode::DATA_LOAD_FAILED,
                        "Failed to setup data loaders: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_dataloaders");
            }

            void AnomalyTrainer::setupValidator()
            {
                _profiler.start("setup_validator");
                _logger->info("AnomalyTrainer", "Setting up validator");

                try 
                {
                    // Create AnomalyValidator
                    _validator = std::make_unique<Validator::AnomalyValidator>(_config);

                    // Create validation dataset
                    auto valDataset = std::make_shared<Data::Dataset::AnomalyDataset>(*_config, false);

                    _logger->info("AnomalyTrainer", "Validation dataset created with " +
                                 std::to_string(valDataset->size().value_or(0)) + " samples");

                    // Create validation DataLoader
                    auto valLoader = torch::data::make_data_loader(
                        std::move(valDataset->map(Data::DataExampleCollation())),
                        torch::data::DataLoaderOptions() 
                            .batch_size(_config->getBatchSize())
                            .workers(_config->getWorkers())
                            .drop_last(false)
                    );

                    // Prepare model for validation BEFORE moving the loader
                    // This computes quantiles (EfficientAD)
                    if (!_isValidationPrepared) {
                        _profiler.start("prepare_model_validation");
                        _logger->info("AnomalyTrainer", "Preparing model for validation...");

                        try {
                            auto anomalyModel = dynamic_cast<Model::AnomalyModel*>(_model.get());
                            if (!anomalyModel) {
                                throw Utils::ModelException(
                                    Utils::ErrorCode::MODEL_INVALID_ARCHITECTURE,
                                    "Model is not an AnomalyModel"
                                );
                            }

                            // Use the valLoader we just created for preparation
                            anomalyModel->prepareValidation(*valLoader);

                            _isValidationPrepared = true;
                            _logger->info("AnomalyTrainer", "Validation preparation completed");
                        }
                        catch (const std::exception& e) {
                            _logger->error("AnomalyTrainer", "Failed to prepare validation: " + std::string(e.what()));
                            throw;
                        }

                        _profiler.stop("prepare_model_validation");
                    }

                    // Now inject data loader into validator
                    auto anomalyValidator = dynamic_cast<Validator::AnomalyValidator*>(_validator.get());
                    if (anomalyValidator) 
                    {
                        anomalyValidator->setDataLoader(std::move(valLoader));
                    }

                    _logger->info("AnomalyTrainer", "Validator setup completed");
                }
                catch (const std::exception& e) {
                    _logger->error("AnomalyTrainer", "Failed to setup validator: " + std::string(e.what()));
                    _validator.reset();  // Clear validator on error
                }

                _profiler.stop("setup_validator");
            }

            WheelDL::Data::Dataset::DataExample AnomalyTrainer::preprocessBatch(
                const WheelDL::Data::Dataset::DataExample& batch)
            {
                _profiler.start("preprocess_batch");

                // Move data to device
                WheelDL::Data::Dataset::DataExample preprocessed;
                preprocessed.data = batch.data.to(_device);

                // For anomaly detection, targets are typically the same as data (reconstruction)
                // or anomaly labels (0=normal, 1=anomaly)
                if (batch.targets.defined())
                {
                    preprocessed.targets = batch.targets.to(_device);
                }

                // Copy other fields
                preprocessed.classes = batch.classes.to(_device);

                _profiler.stop("preprocess_batch");
                return preprocessed;
            }

            float AnomalyTrainer::calculateFitness(const MetricsData& metrics)
            {
                // For anomaly detection, fitness is typically:
                // - AUC-ROC (Area Under Curve)
                // - AP (Average Precision)
                // - F1 Score
                //
                // Since MetricsData contains fitness field, use it directly
                // If fitness is not computed, use 1.0 - loss as fallback

                if (metrics.fitness > 0.0f) {
                    return metrics.fitness;
                }
                else if (metrics.mAP > 0.0f) {
                    // Use mAP as fitness (for PatchCore, etc.)
                    return metrics.mAP;
                }
                else {
                    // Fallback: use negative loss (lower loss = higher fitness)
                    return 1.0f / (1.0f + metrics.loss);
                }
            }


        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
