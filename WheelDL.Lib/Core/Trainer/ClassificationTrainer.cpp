#include "pch.h"
#include "ClassificationTrainer.h"
#include "../Validator/ClassificationValidator.h"
#include "../Predictor/ClassificationPredictor.h"
#include "../../Data/Transforms/Collation.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            ClassificationTrainer::ClassificationTrainer(
                std::shared_ptr<Config::Configuration> config,
                WheelDL::Utils::Logger* logger,
                WheelDL::Utils::Workspace* workspace,
                WheelDL::Utils::PerformanceProfiler* profiler,
                ProgressCallback progressCallback,
                std::atomic<bool>* stopFlag)
                : BaseTrainer(config, logger, workspace, profiler, progressCallback, stopFlag)
            {
                _logger->info("ClassificationTrainer", "Initializing classification trainer");

                if (config->getTaskType() != TaskType::CLASSIFICATION) {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be CLASSIFICATION for ClassificationTrainer"
                    );
                }
            }

            void ClassificationTrainer::setupModel()
            {
                _profiler->start("setup_model");
                _logger->info("ClassificationTrainer", "Setting up classification model");

                try {
                    // Create ClassificationModel with config
                    _model = std::make_unique<Model::ClassificationModel>(_config);
                    _model->to(_device);
                    _logger->info("ClassificationTrainer", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw WheelDL::Utils::ModelException(
                        WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup classification model: " + std::string(e.what())
                    );
                }

                _profiler->stop("setup_model");
            }

            void ClassificationTrainer::setupDataLoaders()
            {
                _profiler->start("setup_dataloaders");
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
                    throw WheelDL::Utils::DataException(
                        WheelDL::Utils::ErrorCode::DATA_LOAD_FAILED,
                        "Failed to setup data loaders: " + std::string(e.what())
                    );
                }

                _profiler->stop("setup_dataloaders");
            }

            void ClassificationTrainer::setupValidator()
            {
                _profiler->start("setup_validator");
                _logger->info("ClassificationTrainer", "Setting up validator");

                try 
                {
                    _validator = std::make_unique<Validator::ClassificationValidator>(_config, _logger, _profiler, _stopFlag);

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

                _profiler->stop("setup_validator");
            }

            WheelDL::Data::Dataset::DataExample ClassificationTrainer::preprocessBatch(
                const WheelDL::Data::Dataset::DataExample& batch)
            {
                _profiler->start("preprocess_batch");

                // Move data to device
                WheelDL::Data::Dataset::DataExample preprocessed;
                preprocessed.data = batch.data.to(_device);

                if (batch.targets.defined()) {
                    preprocessed.targets = batch.targets.to(_device);
                }

                preprocessed.classes = batch.classes.to(_device);

                _profiler->stop("preprocess_batch");
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
                    auto predictor = std::make_unique<Predictor::ClassificationPredictor>(_config, checkpointPath, _logger, _profiler);
                    _logger->info("ClassificationTrainer", "Classification predictor setup completed");
                    return predictor;
                }
                catch (const std::exception& e) {
                    throw WheelDL::Utils::ModelException(
                        WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup classification predictor: " + std::string(e.what())
                    );
                }
            }

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
