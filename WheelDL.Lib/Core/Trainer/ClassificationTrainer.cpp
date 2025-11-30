#include "pch.h"
#include "ClassificationTrainer.h"
#include "../Validator/ClassificationValidator.h"
#include "../Predictor/ClassificationPredictor.h"
#include "../../Data/Transforms/Collation.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <nlohmann/json.hpp>
#include <fstream>

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            ClassificationTrainer::ClassificationTrainer(const std::shared_ptr<Config::Configuration> config)
                : BaseTrainer(config)
            {
                _logger->info("ClassificationTrainer", "Initializing classification trainer");

                // Get model YAML path from configuration
                _modelYamlPath = config->getModelPath();
                if (_modelYamlPath.empty()) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Model YAML path is not specified in configuration"
                    );
                }

                // Verify task type
                if (config->getTaskType() != TaskType::CLASSIFICATION) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be CLASSIFICATION for ClassificationTrainer"
                    );
                }

                _logger->info("ClassificationTrainer", "Model YAML: " + _modelYamlPath);
            }

            void ClassificationTrainer::setupModel()
            {
                _profiler.start("setup_model");
                _logger->info("ClassificationTrainer", "Setting up classification model");

                try {
                    // Create ClassificationModel with config
                    _model = std::make_unique<Model::ClassificationModel>(_config);
                    _model->to(_device);
                    _logger->info("ClassificationTrainer", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup classification model: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_model");
            }

            void ClassificationTrainer::setupDataLoaders()
            {
                _profiler.start("setup_dataloaders");
                _logger->info("ClassificationTrainer", "Setting up data loaders");

                try {
                    auto trainDataset = std::make_shared<Data::Dataset::ClassificationDataset>(*_config, true);

                    _logger->info("ClassificationTrainer", "Training dataset created with " +
                        std::to_string(trainDataset->size().value_or(0)) + " samples");

                    // Create DataLoader with LibTorch
                    auto trainLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                        std::move(trainDataset->map(Data::DataExampleCollation())),
                        torch::data::DataLoaderOptions()
                            .batch_size(_config->getBatchSize())
                            .workers(_config->getWorkers())
                            .drop_last(true)
                    );

                    // Inject training DataLoader into trainer
                    setTrainDataLoader(std::move(trainLoader));

                    _logger->info("ClassificationTrainer", "Data loaders setup completed");
                }
                catch (const std::exception& e) {
                    throw Utils::DataException(
                        Utils::ErrorCode::DATA_LOAD_FAILED,
                        "Failed to setup data loaders: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_dataloaders");
            }

            void ClassificationTrainer::setupValidator()
            {
                _profiler.start("setup_validator");
                _logger->info("ClassificationTrainer", "Setting up validator");

                try 
                {
                    _validator = std::make_unique<Validator::ClassificationValidator>(_config);

                    // Create validation dataset
                    auto valDataset = std::make_shared<Data::Dataset::ClassificationDataset>(*_config, false);

                    _logger->info("ClassificationTrainer", "Validation dataset created with " +
                        std::to_string(valDataset->size().value_or(0)) + " samples");

                    // Create validation DataLoader
                    auto valLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                        std::move(valDataset->map(Data::DataExampleCollation())),
                        torch::data::DataLoaderOptions()
                            .batch_size(_config->getBatchSize())
                            .workers(_config->getWorkers())
                            .drop_last(false)
                    );

                    // Inject data loader into validator
                    auto clsValidator = dynamic_cast<Validator::ClassificationValidator*>(_validator.get());
                    if (clsValidator) {
                        clsValidator->setDataLoader(std::move(valLoader));
                    }

                    _logger->info("ClassificationTrainer", "Validator setup completed");
                }
                catch (const std::exception& e) {
                    _logger->error("ClassificationTrainer", "Failed to setup validator: " + std::string(e.what()));
                    _validator.reset();
                }

                _profiler.stop("setup_validator");
            }

            WheelDL::Data::Dataset::DataExample ClassificationTrainer::preprocessBatch(
                const WheelDL::Data::Dataset::DataExample& batch)
            {
                _profiler.start("preprocess_batch");

                // Move data to device
                WheelDL::Data::Dataset::DataExample preprocessed;
                preprocessed.data = batch.data.to(_device);

                if (batch.targets.defined()) {
                    preprocessed.targets = batch.targets.to(_device);
                }

                preprocessed.classes = batch.classes.to(_device);

                _profiler.stop("preprocess_batch");
                return preprocessed;
            }

            float ClassificationTrainer::calculateFitness(const MetricsData& metrics)
            {
                // For classification, use accuracy as primary fitness metric
				return metrics.accuracy * 0.5 + metrics.f1Score * 0.5;
            }

            std::unique_ptr<Predictor::BasePredictor> ClassificationTrainer::setupPredictor(
                const std::string& checkpointPath)
            {
                _logger->info("ClassificationTrainer", "Setting up classification predictor with checkpoint: " + checkpointPath);

                try {
                    auto predictor = std::make_unique<Predictor::ClassificationPredictor>(_config, checkpointPath);
                    _logger->info("ClassificationTrainer", "Classification predictor setup completed");
                    return predictor;
                }
                catch (const std::exception& e) {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup classification predictor: " + std::string(e.what())
                    );
                }
            }

            void ClassificationTrainer::exportPredictionResults(
                const std::vector<PredictionResult>& results,
                const std::vector<std::string>& imagePaths)
            {
                namespace fs = std::filesystem;

                fs::path resultsPath = _workspace->getResultDir();
                _logger->info("ClassificationTrainer", "Exporting classification prediction results to: " + resultsPath.string());

                try {
                    nlohmann::json predictionsJson = nlohmann::json::array();
                    predictionsJson.get<nlohmann::json::array_t>().reserve(results.size());
                    auto classNames = _config->getClassNames();
                    for (size_t i = 0; i < results.size(); ++i) {
                        const auto& result = results[i];
                        const auto& imagePath = imagePaths[i];

                        nlohmann::json predJson;
                        predJson["image_path"] = imagePath;
                        predJson["inference_time_ms"] = result.inferenceTime;

                        if (!result.classIds.empty()) 
                        {
                            predJson["predicted_class"] = result.classIds[0];
                            predJson["label"] = classNames[result.classIds[0]];
                        }

                        if (!result.scores.empty()) 
                        {
                            predJson["confidence"] = result.scores[0];
                        }

                        predictionsJson.push_back(predJson);
                    }

                    fs::path predictionsPath = resultsPath / "predictions.json";
                    std::ofstream jsonFile(predictionsPath);
                    if (!jsonFile.is_open()) {
                        throw Utils::DataException(
                            Utils::ErrorCode::FILE_IO_ERROR,
                            "Failed to create predictions.json file"
                        );
                    }

                    nlohmann::json rootJson;
                    rootJson["predictions"] = predictionsJson;
                    rootJson["total_predictions"] = results.size();
                    jsonFile << std::setw(4) << rootJson << std::endl;
                    jsonFile.close();

                    if (jsonFile.fail()) {
                        throw Utils::DataException(
                            Utils::ErrorCode::FILE_IO_ERROR,
                            "Failed to write predictions.json"
                        );
                    }
                    _logger->info("ClassificationTrainer", "Exported " + std::to_string(results.size()) +
                        " prediction results to " + predictionsPath.string());
                }
                catch (const std::exception& e) {
                    throw Utils::DataException(
                        Utils::ErrorCode::FILE_IO_ERROR,
                        "Failed to export classification prediction results: " + std::string(e.what())
                    );
                }
            }

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
