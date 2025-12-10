#include "pch.h"
#include "SegmentationTrainer.h"
#include "../Validator/SegmentationValidator.h"
#include "../Predictor/SegmentationPredictor.h"
#include "../../Data/Transforms/Collation.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <nlohmann/json.hpp>
#include <fstream>

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            SegmentationTrainer::SegmentationTrainer(const std::shared_ptr<Config::Configuration> config)
                : BaseTrainer(config)
            {
                _logger->info("SegmentationTrainer", "Initializing segmentation trainer");

                // Get model YAML path from configuration
                _modelYamlPath = config->getModelPath();
                if (_modelYamlPath.empty()) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Model YAML path is not specified in configuration"
                    );
                }

                // Verify task type
                if (config->getTaskType() != TaskType::SEGMENTATION) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be SEGMENTATION for SegmentationTrainer"
                    );
                }

                _logger->info("SegmentationTrainer", "Model YAML: " + _modelYamlPath);
            }

            void SegmentationTrainer::setupModel()
            {
                _profiler.start("setup_model");
                _logger->info("SegmentationTrainer", "Setting up segmentation model");

                try {
                    // Create SegmentationModel with config
                    _model = std::make_unique<Model::SegmentationModel>(_config);
                    _model->to(_device);
                    _logger->info("SegmentationTrainer", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup segmentation model: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_model");
            }

            void SegmentationTrainer::setupDataLoaders()
            {
                _profiler.start("setup_dataloaders");
                _logger->info("SegmentationTrainer", "Setting up data loaders");

                try {
                    auto trainDataset = std::make_shared<Data::Dataset::SegmentationDataset>(*_config, true);

                    _logger->info("SegmentationTrainer", "Training dataset created with " +
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

                    _logger->info("SegmentationTrainer", "Data loaders setup completed");
                }
                catch (const std::exception& e) {
                    throw Utils::DataException(
                        Utils::ErrorCode::DATA_LOAD_FAILED,
                        "Failed to setup data loaders: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_dataloaders");
            }

            void SegmentationTrainer::setupValidator()
            {
                _profiler.start("setup_validator");
                _logger->info("SegmentationTrainer", "Setting up validator");

                try
                {
                    _validator = std::make_unique<Validator::SegmentationValidator>(_config);

                    // Create validation dataset
                    auto valDataset = std::make_shared<Data::Dataset::SegmentationDataset>(*_config, false);

                    _logger->info("SegmentationTrainer", "Validation dataset created with " +
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
                    auto segValidator = dynamic_cast<Validator::SegmentationValidator*>(_validator.get());
                    if (segValidator) {
                        segValidator->setDataLoader(std::move(valLoader));
                    }

                    _logger->info("SegmentationTrainer", "Validator setup completed");
                }
                catch (const std::exception& e) {
                    _logger->error("SegmentationTrainer", "Failed to setup validator: " + std::string(e.what()));
                    _validator.reset();
                }

                _profiler.stop("setup_validator");
            }

            WheelDL::Data::Dataset::DataExample SegmentationTrainer::preprocessBatch(
                const WheelDL::Data::Dataset::DataExample& batch)
            {
                _profiler.start("preprocess_batch");

                // Move data to device
                WheelDL::Data::Dataset::DataExample preprocessed;
                preprocessed.data = batch.data.to(_device);

                // Segmentation targets are already in [N, C, H, W] one-hot encoded format
                if (batch.targets.defined()) {
                    preprocessed.targets = batch.targets.to(_device);
                }

                if (batch.classes.defined()) {
                    preprocessed.classes = batch.classes.to(_device);
                }

                _profiler.stop("preprocess_batch");
                return preprocessed;
            }

            float SegmentationTrainer::calculateFitness(const MetricsData& metrics)
            {
                return metrics.mAP * 0.3f + metrics.f1Score * 0.7f;
            }

            std::unique_ptr<Predictor::BasePredictor> SegmentationTrainer::setupPredictor(
                const std::string& checkpointPath)
            {
                _logger->info("SegmentationTrainer", "Setting up segmentation predictor with checkpoint: " + checkpointPath);

                try {
                    auto predictor = std::make_unique<Predictor::SegmentationPredictor>(_config, checkpointPath);
                    _logger->info("SegmentationTrainer", "Segmentation predictor setup completed");
                    return predictor;
                }
                catch (const std::exception& e) {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup segmentation predictor: " + std::string(e.what())
                    );
                }
            }

            void SegmentationTrainer::exportPredictionResults(
                const std::vector<PredictionResult>& results,
                const std::vector<std::string>& imagePaths)
            {
                namespace fs = std::filesystem;

                fs::path resultsPath = _workspace->getResultDir();
                _logger->info("SegmentationTrainer", "Exporting segmentation prediction results to: " + resultsPath.string());

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
                        predJson["num_segments"] = result.contours.size();

                        // Export segmentation contours
                        nlohmann::json segmentsJson = nlohmann::json::array();
                        for (size_t j = 0; j < result.contours.size(); ++j)
                        {
                            nlohmann::json segJson;

                            // Contour points
                            const auto& contour = result.contours[j];
                            nlohmann::json pointsJson = nlohmann::json::array();
                            for (size_t k = 0; k + 1 < contour.points.size(); k += 2)
                            {

                                pointsJson.push_back(contour.points[k]);
                                pointsJson.push_back(contour.points[k + 1]);
                            }
                            segJson["contour"] = pointsJson;

                            // Class info
                            if (j < result.classIds.size()) {
                                int classId = static_cast<int>(result.classIds[j]);
                                segJson["class_id"] = classId;

                                auto it = classNames.find(classId);
                                if (it != classNames.end()) {
                                    segJson["class_name"] = it->second;
                                }
                            }

                            // Confidence
                            if (j < result.scores.size()) {
                                segJson["confidence"] = result.scores[j];
                            }

                            segmentsJson.push_back(segJson);
                        }

                        predJson["segments"] = segmentsJson;
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

                    // Calculate total segments
                    size_t totalSegments = 0;
                    for (const auto& result : results) {
                        totalSegments += result.contours.size();
                    }
                    rootJson["total_segments"] = totalSegments;

                    jsonFile << std::setw(4) << rootJson << std::endl;
                    jsonFile.close();

                    if (jsonFile.fail()) {
                        throw Utils::DataException(
                            Utils::ErrorCode::FILE_IO_ERROR,
                            "Failed to write predictions.json"
                        );
                    }
                    _logger->info("SegmentationTrainer", "Exported " + std::to_string(results.size()) +
                        " images with " + std::to_string(totalSegments) +
                        " total segments to " + predictionsPath.string());
                }
                catch (const std::exception& e) {
                    throw Utils::DataException(
                        Utils::ErrorCode::FILE_IO_ERROR,
                        "Failed to export segmentation prediction results: " + std::string(e.what())
                    );
                }
            }

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
