#include "pch.h"
#include "OBBTrainer.h"
#include "../Validator/OBBValidator.h"
#include "../Predictor/OBBPredictor.h"
#include "../../Data/Transforms/Collation.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <nlohmann/json.hpp>
#include <fstream>

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            OBBTrainer::OBBTrainer(const std::shared_ptr<Config::Configuration> config)
                : BaseTrainer(config)
            {
                _logger->info("OBBTrainer", "Initializing OBB detection trainer");

                // Get model YAML path from configuration
                _modelYamlPath = config->getModelPath();
                if (_modelYamlPath.empty()) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Model YAML path is not specified in configuration"
                    );
                }

                // Verify task type
                if (config->getTaskType() != TaskType::OBB) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be OBB for OBBTrainer"
                    );
                }

                _logger->info("OBBTrainer", "Model YAML: " + _modelYamlPath);
            }

            void OBBTrainer::setupModel()
            {
                _profiler.start("setup_model");
                _logger->info("OBBTrainer", "Setting up OBB detection model");

                try {
                    // Create OBBModel with config
                    _model = std::make_unique<Model::OBBModel>(_config);
                    _model->to(_device);
                    _logger->info("OBBTrainer", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup OBB detection model: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_model");
            }

            void OBBTrainer::setupDataLoaders()
            {
                _profiler.start("setup_dataloaders");
                _logger->info("OBBTrainer", "Setting up data loaders");

                try {
                    auto trainDataset = std::make_shared<Data::Dataset::OBBDataset>(*_config, true);

                    _logger->info("OBBTrainer", "Training dataset created with " +
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

                    _logger->info("OBBTrainer", "Data loaders setup completed");
                }
                catch (const std::exception& e) {
                    throw Utils::DataException(
                        Utils::ErrorCode::DATA_LOAD_FAILED,
                        "Failed to setup data loaders: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_dataloaders");
            }

            void OBBTrainer::setupValidator()
            {
                _profiler.start("setup_validator");
                _logger->info("OBBTrainer", "Setting up validator");

                try
                {
                    _validator = std::make_unique<Validator::OBBValidator>(_config);

                    // Create validation dataset
                    auto valDataset = std::make_shared<Data::Dataset::OBBDataset>(*_config, false);

                    _logger->info("OBBTrainer", "Validation dataset created with " +
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
                    auto obbValidator = dynamic_cast<Validator::OBBValidator*>(_validator.get());
                    if (obbValidator) {
                        obbValidator->setDataLoader(std::move(valLoader));
                    }

                    _logger->info("OBBTrainer", "Validator setup completed");
                }
                catch (const std::exception& e) {
                    _logger->error("OBBTrainer", "Failed to setup validator: " + std::string(e.what()));
                    _validator.reset();
                }

                _profiler.stop("setup_validator");
            }

            WheelDL::Data::Dataset::DataExample OBBTrainer::preprocessBatch(
                const WheelDL::Data::Dataset::DataExample& batch)
            {
                WheelDL::Data::Dataset::DataExample preprocessed;
                preprocessed.data = batch.data.to(_device);

                auto batchIdx = batch.batchIndices.view({ -1, 1 });
                auto classes = batch.classes.view({ -1, 1 });
                auto bboxes = batch.targets;  // [num_gt, 5] for OBB: cx, cy, w, h, angle

                auto targets = torch::cat({
                    batchIdx.to(_device),
                    classes.to(_device),
                    bboxes.to(_device)
                    }, 1);

                preprocessed.targets = targets;
                return preprocessed;
            }

            float OBBTrainer::calculateFitness(const MetricsData& metrics)
            {
                return metrics.mAP;
            }

            std::unique_ptr<Predictor::BasePredictor> OBBTrainer::setupPredictor(
                const std::string& checkpointPath)
            {
                _logger->info("OBBTrainer", "Setting up OBB predictor with checkpoint: " + checkpointPath);

                try
                {
                    auto predictor = std::make_unique<Predictor::OBBPredictor>(_config, checkpointPath);
                    _logger->info("OBBTrainer", "OBB predictor setup completed");
                    return predictor;
                }
                catch (const std::exception& e)
                {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup OBB predictor: " + std::string(e.what())
                    );
                }
            }

            void OBBTrainer::exportPredictionResults(
                const std::vector<PredictionResult>& results,
                const std::vector<std::string>& imagePaths)
            {
                namespace fs = std::filesystem;

                fs::path resultsPath = _workspace->getResultDir();
                _logger->info("OBBTrainer", "Exporting OBB prediction results to: " + resultsPath.string());

                try {
                    nlohmann::json predictionsJson = nlohmann::json::array();
                    predictionsJson.get<nlohmann::json::array_t>().reserve(results.size());
                    auto classNames = _config->getClassNames();

                    for (size_t i = 0; i < results.size(); ++i)
                    {
                        const auto& result = results[i];
                        const auto& imagePath = imagePaths[i];

                        nlohmann::json predJson;
                        predJson["image_path"] = imagePath;
                        predJson["inference_time_ms"] = result.inferenceTime;
                        predJson["num_detections"] = result.orientedBoxes.size();

                        // Export OBB detections
                        nlohmann::json detectionsJson = nlohmann::json::array();
                        for (size_t j = 0; j < result.orientedBoxes.size(); ++j)
                        {
                            nlohmann::json detJson;

                            // Oriented Bounding Box
                            const auto& obb = result.orientedBoxes[j];
                            detJson["obb"] = {
                                {"cx", obb.cx},
                                {"cy", obb.cy},
                                {"width", obb.width},
                                {"height", obb.height},
                                {"angle", obb.angle}
                            };

                            // Class info
                            if (j < result.classIds.size()) {
                                int classId = static_cast<int>(result.classIds[j]);
                                detJson["class_id"] = classId;

                                auto it = classNames.find(classId);
                                if (it != classNames.end()) {
                                    detJson["class_name"] = it->second;
                                }
                            }

                            // Confidence
                            if (j < result.scores.size()) {
                                detJson["confidence"] = result.scores[j];
                            }

                            detectionsJson.push_back(detJson);
                        }

                        predJson["detections"] = detectionsJson;
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
                    rootJson["total_images"] = results.size();

                    // Calculate total detections
                    size_t totalDetections = 0;
                    for (const auto& result : results) {
                        totalDetections += result.orientedBoxes.size();
                    }
                    rootJson["total_detections"] = totalDetections;

                    jsonFile << std::setw(4) << rootJson << std::endl;
                    jsonFile.close();

                    if (jsonFile.fail()) {
                        throw Utils::DataException(
                            Utils::ErrorCode::FILE_IO_ERROR,
                            "Failed to write predictions.json"
                        );
                    }
                    _logger->info("OBBTrainer", "Exported " + std::to_string(results.size()) +
                        " images with " + std::to_string(totalDetections) +
                        " total OBB detections to " + predictionsPath.string());
                }
                catch (const std::exception& e) {
                    throw Utils::DataException(
                        Utils::ErrorCode::FILE_IO_ERROR,
                        "Failed to export OBB prediction results: " + std::string(e.what())
                    );
                }
            }

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
