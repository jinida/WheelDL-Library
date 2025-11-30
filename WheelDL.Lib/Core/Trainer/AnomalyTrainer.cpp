#include "pch.h"
#include "WheelDL.Lib/Data/Transforms/Collation.h"
#include "AnomalyTrainer.h"
#include "../Validator/AnomalyValidator.h"
#include "../Predictor/AnomalyPredictor.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include "../../Utils/Export/PredictionExporter.h"
#include "../../Utils/Export/ImageExporter.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>
#include <chrono>
#include <algorithm>

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            AnomalyTrainer::AnomalyTrainer(const std::shared_ptr<Config::Configuration> config)
                : BaseTrainer(config)
                , _isModelPrepared(false)
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
                _config->setImageNetNorm(true);
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
					auto trainLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
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

					// Inject training DataLoader into trainer
                    setTrainDataLoader(std::move(trainLoader));

                    if (_config->IsEfficientAD())
                    {
                        // Create 10% subset dataset for efficient quantile computation
                        auto prepareDataset = std::make_shared<Data::Dataset::PredDataset>(*_config, true);
                        size_t subsetSize = std::max(size_t(1), trainDataset->size().value_or(0) / 10);
                        prepareDataset->limitSamples(subsetSize);

                        _logger->info("AnomalyTrainer", "Prepare dataset: " +
                            std::to_string(prepareDataset->size().value_or(0)) + " samples (10% subset)");

                        auto prepareLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                            std::move(prepareDataset->map(Data::PredDataExampleCollation())),
                            torch::data::DataLoaderOptions()
                            .batch_size(_config->getBatchSize())
                            .workers(_config->getWorkers())
                            .drop_last(false)
                        );

                        // Store in shared_ptr and create batch iterator
                        auto prepareLoaderPtr = std::make_shared<decltype(prepareLoader)>(std::move(prepareLoader));
                        _prepareBatchIterator = [prepareLoaderPtr](std::function<void(const Data::Dataset::PredDataExample&, int)> callback) {
                            int batchIdx = 0;
                            for (auto& batch : *(*prepareLoaderPtr)) {
                                callback(batch, batchIdx++);
                            }
                        };

                        _prepareValidationCallback = [this]() {
                            auto anomalyModel = dynamic_cast<Model::AnomalyModel*>(_model.get());
                            if (anomalyModel && _prepareBatchIterator)
                            {
                                anomalyModel->eval();
                                _logger->info("AnomalyTrainer", "Preparing validation (quantile setup)...");
                                anomalyModel->prepareValidation(_prepareBatchIterator);
                            }
                        };
                    }
                    // Now move the loader to the trainer

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
                    _validator = std::make_unique<Validator::AnomalyValidator>(_config);

                    // Create validation dataset
                    auto valDataset = std::make_shared<Data::Dataset::AnomalyDataset>(*_config, false);

                    _logger->info("AnomalyTrainer", "Validation dataset created with " +
                                 std::to_string(valDataset->size().value_or(0)) + " samples");

                    // Create validation DataLoader as shared_ptr
                    auto valLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                        std::move(valDataset->map(Data::DataExampleCollation())),
                        torch::data::DataLoaderOptions()
                            .batch_size(_config->getBatchSize())
                            .workers(_config->getWorkers())
                            .drop_last(false)
                    );

                    // Inject data loader into validator
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
				return metrics.aucROC * 0.5 + metrics.f1Score * 0.5;
            }

            std::unique_ptr<Predictor::BasePredictor> AnomalyTrainer::setupPredictor(const std::string& checkpointPath)
            {
                _logger->info("AnomalyTrainer", "Setting up anomaly predictor with checkpoint: " + checkpointPath);

                try {
                    // Create AnomalyPredictor with configuration and checkpoint path
                    auto predictor = std::make_unique<Predictor::AnomalyPredictor>(_config, checkpointPath);

                    _logger->info("AnomalyTrainer", "Anomaly predictor setup completed");
                    return predictor;
                }
                catch (const std::exception& e) {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup anomaly predictor: " + std::string(e.what())
                    );
                }
            }

            void AnomalyTrainer::exportPredictionResults(
                const std::vector<PredictionResult>& results,
                const std::vector<std::string>& imagePaths)
            {
                namespace fs = std::filesystem;

                fs::path resultsPath = _workspace->getResultDir();
                _logger->info("AnomalyTrainer", "Exporting anomaly prediction results to: " + resultsPath.string());

                try
                {
                    fs::path anomalyMapsDir = resultsPath / "anomaly_maps";
                    if (!fs::exists(anomalyMapsDir))
                    {
                        fs::create_directories(anomalyMapsDir);
                    }

                    nlohmann::json predictionsJson = nlohmann::json::array();

                    predictionsJson.get<nlohmann::json::array_t>().reserve(results.size());

                    for (size_t i = 0; i < results.size(); ++i)
                    {
                        const auto& result = results[i];
                        const auto& imagePath = imagePaths[i];

                        nlohmann::json predJson;
                        predJson["image_path"] = imagePath;
                        predJson["inference_time_ms"] = result.inferenceTime;

                        if (!result.scores.empty()) {
                            predJson["anomaly_score"] = result.scores[0];
                        }

                        if (!result.classIds.empty()) {
                            predJson["class_id"] = result.classIds[0];
                            predJson["label"] = (result.classIds[0] == 0) ? "normal" : "anomaly";
                        }

                        if (!result.anomalyMap.empty()) {
                            std::hash<std::string> hasher;
                            size_t hashValue = hasher(imagePath);
                            std::stringstream ss;
                            ss << std::hex << hashValue;

                            std::string filename = ss.str() + ".jpg";
                            fs::path mapPath = anomalyMapsDir / filename;

                            std::vector<uchar> buffer;
                            std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 80 };

                            bool encoded = cv::imencode(".jpg", result.anomalyMap, buffer, params);

                            if (encoded) {
                                std::ofstream outFile(mapPath, std::ios::binary);
                                if (outFile.is_open()) {
                                    outFile.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
                                    outFile.close();

                                    predJson["anomaly_map"] = "./anomaly_maps/" + filename;
                                    _logger->debug("AnomalyTrainer", "Saved anomaly map: " + filename);
                                }
                                else {
                                    _logger->warn("AnomalyTrainer", "Failed to save anomaly map to disk for: " + imagePath);
                                }
                            }
                            else {
                                _logger->warn("AnomalyTrainer", "Failed to encode anomaly map for: " + imagePath);
                            }
                        }

                        predictionsJson.push_back(predJson);
                    }

                    fs::path predictionsPath = resultsPath / "predictions.json";
                    std::ofstream jsonFile(predictionsPath);
                    if (!jsonFile.is_open())
                    {
                        throw Utils::DataException(
                            Utils::ErrorCode::FILE_IO_ERROR,
                            "Failed to create predictions.json file"
                        );
                    }

                    nlohmann::json rootJson;
                    rootJson["predictions"] = predictionsJson;
                    rootJson["total_predictions"] = results.size();
                    jsonFile << std::setw(4) << rootJson << std::endl;
                    jsonFile << std::setw(4) << rootJson << std::endl;
                    jsonFile.close();

                    if (jsonFile.fail()) {
                        throw Utils::DataException(
                            Utils::ErrorCode::FILE_IO_ERROR,
                            "Failed to write predictions.json"
                        );
                    }
                    _logger->info("AnomalyTrainer", "Exported " + std::to_string(results.size()) +
                        " prediction results to " + predictionsPath.string());
                }
                catch (const std::exception& e) {
                    throw Utils::DataException(
                        Utils::ErrorCode::FILE_IO_ERROR,
                        "Failed to export anomaly prediction results: " + std::string(e.what())
                    );
                }
            }

            MetricsData AnomalyTrainer::runValidation(int epoch, bool useEmaIfAvailable)
            {
                // Call prepareValidation before each validation
                // This is required for EfficientAD to setup quantile normalization
                if (_prepareValidationCallback)
                {
                    _prepareValidationCallback();
                }

                // Call base class validation
                return BaseTrainer::runValidation(epoch, useEmaIfAvailable);
            }

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
