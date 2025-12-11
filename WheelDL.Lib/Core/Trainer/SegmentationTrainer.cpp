#include "pch.h"
#include "SegmentationTrainer.h"
#include "../Validator/SegmentationValidator.h"
#include "../Predictor/SegmentationPredictor.h"
#include "../../Data/Transforms/Collation.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            SegmentationTrainer::SegmentationTrainer(
                std::shared_ptr<Config::Configuration> config,
                WheelDL::Utils::Logger* logger,
                WheelDL::Utils::Workspace* workspace,
                WheelDL::Utils::PerformanceProfiler* profiler,
                ProgressCallback progressCallback,
                std::atomic<bool>* stopFlag)
                : BaseTrainer(config, logger, workspace, profiler, progressCallback, stopFlag)
            {
                _logger->info("SegmentationTrainer", "Initializing segmentation trainer");

                // Verify task type
                if (config->getTaskType() != TaskType::SEGMENTATION) {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be SEGMENTATION for SegmentationTrainer"
                    );
                }
            }

            void SegmentationTrainer::setupModel()
            {
                _profiler->start("setup_model");
                _logger->info("SegmentationTrainer", "Setting up segmentation model");

                try {
                    // Create SegmentationModel with config
                    _model = std::make_unique<Model::SegmentationModel>(_config);
                    _model->to(_device);
                    _logger->info("SegmentationTrainer", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw WheelDL::Utils::ModelException(
                        WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup segmentation model: " + std::string(e.what())
                    );
                }

                _profiler->stop("setup_model");
            }

            void SegmentationTrainer::setupDataLoaders()
            {
                _profiler->start("setup_dataloaders");
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
                    throw WheelDL::Utils::DataException(
                        WheelDL::Utils::ErrorCode::DATA_LOAD_FAILED,
                        "Failed to setup data loaders: " + std::string(e.what())
                    );
                }

                _profiler->stop("setup_dataloaders");
            }

            void SegmentationTrainer::setupValidator()
            {
                _profiler->start("setup_validator");
                _logger->info("SegmentationTrainer", "Setting up validator");

                try
                {
                    _validator = std::make_unique<Validator::SegmentationValidator>(_config, _logger, _profiler, _stopFlag);

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

                _profiler->stop("setup_validator");
            }

            WheelDL::Data::Dataset::DataExample SegmentationTrainer::preprocessBatch(
                const WheelDL::Data::Dataset::DataExample& batch)
            {
                _profiler->start("preprocess_batch");

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

                _profiler->stop("preprocess_batch");
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
                    auto predictor = std::make_unique<Predictor::SegmentationPredictor>(_config, checkpointPath, _logger, _profiler);
                    _logger->info("SegmentationTrainer", "Segmentation predictor setup completed");
                    return predictor;
                }
                catch (const std::exception& e) {
                    throw WheelDL::Utils::ModelException(
                        WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup segmentation predictor: " + std::string(e.what())
                    );
                }
            }

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
