#include "pch.h"
#include "DetectionTrainer.h"
#include "../Validator/DetectionValidator.h"
#include "../Predictor/DetectionPredictor.h"
#include "../../Data/Transforms/Collation.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <nlohmann/json.hpp>
#include <fstream>

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            static int s_debugImageCounter = 0;
            static std::string s_debugImagePrefix = "debug_batch";
            static bool s_enableDebugVisualization = true;

            void DetectionTrainer::drawBBoxesOnBatch(
                const torch::Tensor& batchData,      // [B, C, H, W], 0~1 scaled
                const torch::Tensor& batchIndices,   // [num_gt]
                const torch::Tensor& bboxes,         // [num_gt, 4], XYWH, 0~1 scaled
                const torch::Tensor& classes)        // [num_gt]
            {
                if (!s_enableDebugVisualization) return;

                auto data = batchData.cpu().contiguous();
                auto batchIdx = batchIndices.cpu().contiguous();
                auto boxes = bboxes.cpu().contiguous();
                auto cls = classes.cpu().contiguous();

                int B = data.size(0);
                int C = data.size(1);
                int H = data.size(2);
                int W = data.size(3);

                for (int b = 0; b < B; ++b)
                {
                    auto imgTensor = data[b]
                        .permute({ 1, 2, 0 })
                        .contiguous()
                        .mul(255)
                        .clamp(0, 255)
                        .to(torch::kUInt8)
                        .clone();

                    cv::Mat img(H, W, CV_8UC3, imgTensor.data_ptr<uint8_t>());
                    cv::Mat imgBGR;
                    cv::cvtColor(img, imgBGR, cv::COLOR_RGB2BGR);  // RGB -> BGR (OpenCV¿ë)

                    auto batchMask = (batchIdx == b);
                    auto indices = torch::nonzero(batchMask).squeeze(1);

                    if (indices.numel() > 0)
                    {
                        auto batchBoxes = boxes.index_select(0, indices);
                        auto batchClasses = cls.index_select(0, indices);

                        int numBoxes = batchBoxes.size(0);

                        for (int i = 0; i < numBoxes; ++i)
                        {
                            float cx = batchBoxes[i][0].item<float>() * W;
                            float cy = batchBoxes[i][1].item<float>() * H;
                            float bw = batchBoxes[i][2].item<float>() * W;
                            float bh = batchBoxes[i][3].item<float>() * H;

                            int x1 = static_cast<int>(cx - bw / 2.0f);
                            int y1 = static_cast<int>(cy - bh / 2.0f);
                            int x2 = static_cast<int>(cx + bw / 2.0f);
                            int y2 = static_cast<int>(cy + bh / 2.0f);

                            x1 = std::max(0, std::min(x1, W - 1));
                            y1 = std::max(0, std::min(y1, H - 1));
                            x2 = std::max(0, std::min(x2, W - 1));
                            y2 = std::max(0, std::min(y2, H - 1));

                            int classId = batchClasses[i].item<int>();
                            cv::Scalar color = cv::Scalar(
                                (classId * 67) % 256,
                                (classId * 123) % 256,
                                (classId * 91) % 256
                            );

                            cv::rectangle(imgBGR, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

                            std::string label = "cls:" + std::to_string(classId);
                            int baseline = 0;
                            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                            cv::rectangle(imgBGR,
                                cv::Point(x1, y1 - labelSize.height - 5),
                                cv::Point(x1 + labelSize.width, y1),
                                color, cv::FILLED);
                            cv::putText(imgBGR, label, cv::Point(x1, y1 - 3),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                        }
                    }

                    std::string filename = s_debugImagePrefix + "_" +
                        std::to_string(s_debugImageCounter) + "_" +
                        std::to_string(b) + ".jpg";
                    cv::imwrite(filename, imgBGR);
                }

                s_debugImageCounter++;
            }

            DetectionTrainer::DetectionTrainer(const std::shared_ptr<Config::Configuration> config)
                : BaseTrainer(config)
            {
                _logger->info("DetectionTrainer", "Initializing detection trainer");

                // Get model YAML path from configuration
                _modelYamlPath = config->getModelPath();
                if (_modelYamlPath.empty()) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Model YAML path is not specified in configuration"
                    );
                }

                // Verify task type
                if (config->getTaskType() != TaskType::DETECTION) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be DETECTION for DetectionTrainer"
                    );
                }

                _logger->info("DetectionTrainer", "Model YAML: " + _modelYamlPath);
            }

            void DetectionTrainer::setupModel()
            {
                _profiler.start("setup_model");
                _logger->info("DetectionTrainer", "Setting up detection model");

                try {
                    // Create DetectionModel with config
                    _model = std::make_unique<Model::DetectionModel>(_config);
                    _model->to(_device);
                    _logger->info("DetectionTrainer", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup detection model: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_model");
            }

            void DetectionTrainer::setupDataLoaders()
            {
                _profiler.start("setup_dataloaders");
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
                    throw Utils::DataException(
                        Utils::ErrorCode::DATA_LOAD_FAILED,
                        "Failed to setup data loaders: " + std::string(e.what())
                    );
                }

                _profiler.stop("setup_dataloaders");
            }

            void DetectionTrainer::setupValidator()
            {
                _profiler.start("setup_validator");
                _logger->info("DetectionTrainer", "Setting up validator");

                try
                {
                    _validator = std::make_unique<Validator::DetectionValidator>(_config);

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

                _profiler.stop("setup_validator");
            }

            WheelDL::Data::Dataset::DataExample DetectionTrainer::preprocessBatch(
                const WheelDL::Data::Dataset::DataExample& batch)
            {
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
                /*drawBBoxesOnBatch(
                    batch.data,
                    batch.batchIndices,
                    batch.targets,
                    batch.classes
                );*/
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
                    auto predictor = std::make_unique<Predictor::DetectionPredictor>(_config, checkpointPath);
                    _logger->info("DetectionTrainer", "Detection predictor setup completed");
                    return predictor;
                }
                catch (const std::exception& e) 
                {
                    throw Utils::ModelException(
                        Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup detection predictor: " + std::string(e.what())
                    );
                }
            }

            void DetectionTrainer::exportPredictionResults(
                const std::vector<PredictionResult>& results,
                const std::vector<std::string>& imagePaths)
            {
                namespace fs = std::filesystem;

                fs::path resultsPath = _workspace->getResultDir();
                _logger->info("DetectionTrainer", "Exporting detection prediction results to: " + resultsPath.string());

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
                        predJson["num_detections"] = result.boxes.size();

                        // Export detections
                        nlohmann::json detectionsJson = nlohmann::json::array();
                        for (size_t j = 0; j < result.boxes.size(); ++j) 
                        {
                            nlohmann::json detJson;

                            // Bounding box
                            const auto& box = result.boxes[j];
                            detJson["bbox"] = {
                                {"x1", box.x1},
                                {"y1", box.y1},
                                {"x2", box.x2},
                                {"y2", box.y2}
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
                        totalDetections += result.boxes.size();
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
                    _logger->info("DetectionTrainer", "Exported " + std::to_string(results.size()) +
                        " images with " + std::to_string(totalDetections) +
                        " total detections to " + predictionsPath.string());
                }
                catch (const std::exception& e) {
                    throw Utils::DataException(
                        Utils::ErrorCode::FILE_IO_ERROR,
                        "Failed to export detection prediction results: " + std::string(e.what())
                    );
                }
            }

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
