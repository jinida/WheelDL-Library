#include "pch.h"
#include "DetectionTrainer.h"
#include "../Validator/DetectionValidator.h"
#include "../Predictor/DetectionPredictor.h"
#include "../../Data/Transforms/Collation.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"

namespace WheelDL {
    namespace Core {
        namespace Trainer {
            DetectionTrainer::DetectionTrainer(
                std::shared_ptr<Config::Configuration> config,
                WheelDL::Utils::Logger* logger,
                WheelDL::Utils::Workspace* workspace,
                WheelDL::Utils::PerformanceProfiler* profiler,
                ProgressCallback progressCallback,
                std::atomic<bool>* stopFlag)
                : BaseTrainer(config, logger, workspace, profiler, progressCallback, stopFlag)
            {
                _logger->info("DetectionTrainer", "Initializing detection trainer");

                // Verify task type
                if (config->getTaskType() != TaskType::DETECTION) {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be DETECTION for DetectionTrainer"
                    );
                }
            }

            void DetectionTrainer::setupModel()
            {
                _profiler->start("setup_model");
                _logger->info("DetectionTrainer", "Setting up detection model");

                try {
                    // Create DetectionModel with config
                    _model = std::make_unique<Model::DetectionModel>(_config);
                    _model->to(_device);
                    _logger->info("DetectionTrainer", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw WheelDL::Utils::ModelException(
                        WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup detection model: " + std::string(e.what())
                    );
                }

                _profiler->stop("setup_model");
            }

            void DetectionTrainer::setupDataLoaders()
            {
                _profiler->start("setup_dataloaders");
                _logger->info("DetectionTrainer", "Setting up data loaders");

                try {
                    auto trainDataset = std::make_shared<Data::Dataset::DetectionDataset>(*_config, true);

                    _logger->info("DetectionTrainer", "Training dataset created with " +
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

                    _logger->info("DetectionTrainer", "Data loaders setup completed");
                }
                catch (const std::exception& e) {
                    throw WheelDL::Utils::DataException(
                        WheelDL::Utils::ErrorCode::DATA_LOAD_FAILED,
                        "Failed to setup data loaders: " + std::string(e.what())
                    );
                }

                _profiler->stop("setup_dataloaders");
            }

            void DetectionTrainer::setupValidator()
            {
                _profiler->start("setup_validator");
                _logger->info("DetectionTrainer", "Setting up validator");

                try
                {
                    _validator = std::make_unique<Validator::DetectionValidator>(_config, _logger, _profiler, _stopFlag);

                    // Create validation dataset
                    auto valDataset = std::make_shared<Data::Dataset::DetectionDataset>(*_config, false);

                    _logger->info("DetectionTrainer", "Validation dataset created with " +
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
                    auto detValidator = dynamic_cast<Validator::DetectionValidator*>(_validator.get());
                    if (detValidator) {           
                        detValidator->setDataLoader(std::move(valLoader));
                    }

                    _logger->info("DetectionTrainer", "Validator setup completed");
                }
                catch (const std::exception& e) {
                    _logger->error("DetectionTrainer", "Failed to setup validator: " + std::string(e.what()));
                    _validator.reset();
                }

                _profiler->stop("setup_validator");
            }

            WheelDL::Data::Dataset::DataExample DetectionTrainer::preprocessBatch(
                const WheelDL::Data::Dataset::DataExample& batch)
            {
                _profiler->start("preprocess_batch");

                WheelDL::Data::Dataset::DataExample preprocessed;
                preprocessed.data = batch.data.to(_device);

                auto batchIdx = batch.batchIndices.view({ -1, 1 });
                auto classes = batch.classes.view({ -1, 1 });
                auto bboxes = batch.targets;  // [num_gt, 4]

                auto targets = torch::cat({
                    batchIdx.to(_device),
                    classes.to(_device),
                    bboxes.to(_device)
                    }, 1);

				preprocessed.targets = targets;

                _profiler->stop("preprocess_batch");
                return preprocessed;
            }

            float DetectionTrainer::calculateFitness(const MetricsData& metrics)
            {
                return metrics.mAP;
            }

            std::unique_ptr<Predictor::BasePredictor> DetectionTrainer::setupPredictor(
                const std::string& checkpointPath)
            {
                _logger->info("DetectionTrainer", "Setting up detection predictor with checkpoint: " + checkpointPath);

                try 
                {
                    auto predictor = std::make_unique<Predictor::DetectionPredictor>(_config, checkpointPath, _logger, _profiler);
                    _logger->info("DetectionTrainer", "Detection predictor setup completed");
                    return predictor;
                }
                catch (const std::exception& e)
                {
                    throw WheelDL::Utils::ModelException(
                        WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup detection predictor: " + std::string(e.what())
                    );
                }
            }

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
