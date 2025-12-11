#include "pch.h"
#include "OBBTrainer.h"
#include "../Validator/OBBValidator.h"
#include "../Predictor/OBBPredictor.h"
#include "../../Data/Transforms/Collation.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            OBBTrainer::OBBTrainer(
                std::shared_ptr<Config::Configuration> config,
                WheelDL::Utils::Logger* logger,
                WheelDL::Utils::Workspace* workspace,
                WheelDL::Utils::PerformanceProfiler* profiler,
                ProgressCallback progressCallback,
                std::atomic<bool>* stopFlag)
                : BaseTrainer(config, logger, workspace, profiler, progressCallback, stopFlag)
            {
                _logger->info("OBBTrainer", "Initializing OBB detection trainer");

                // Verify task type
                if (config->getTaskType() != TaskType::OBB) {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be OBB for OBBTrainer"
                    );
                }
            }

            void OBBTrainer::setupModel()
            {
                _profiler->start("setup_model");
                _logger->info("OBBTrainer", "Setting up OBB detection model");

                try {
                    // Create OBBModel with config
                    _model = std::make_unique<Model::OBBModel>(_config);
                    _model->to(_device);
                    _logger->info("OBBTrainer", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw WheelDL::Utils::ModelException(
                        WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup OBB detection model: " + std::string(e.what())
                    );
                }

                _profiler->stop("setup_model");
            }

            void OBBTrainer::setupDataLoaders()
            {
                _profiler->start("setup_dataloaders");
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
                    throw WheelDL::Utils::DataException(
                        WheelDL::Utils::ErrorCode::DATA_LOAD_FAILED,
                        "Failed to setup data loaders: " + std::string(e.what())
                    );
                }

                _profiler->stop("setup_dataloaders");
            }

            void OBBTrainer::setupValidator()
            {
                _profiler->start("setup_validator");
                _logger->info("OBBTrainer", "Setting up validator");

                try
                {
                    _validator = std::make_unique<Validator::OBBValidator>(_config, _logger, _profiler, _stopFlag);

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

                _profiler->stop("setup_validator");
            }

            WheelDL::Data::Dataset::DataExample OBBTrainer::preprocessBatch(
                const WheelDL::Data::Dataset::DataExample& batch)
            {
                _profiler->start("preprocess_batch");

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

                _profiler->stop("preprocess_batch");
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
                    auto predictor = std::make_unique<Predictor::OBBPredictor>(_config, checkpointPath, _logger, _profiler);
                    _logger->info("OBBTrainer", "OBB predictor setup completed");
                    return predictor;
                }
                catch (const std::exception& e)
                {
                    throw WheelDL::Utils::ModelException(
                        WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup OBB predictor: " + std::string(e.what())
                    );
                }
            }
        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
