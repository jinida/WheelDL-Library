#include "pch.h"
#include "ClassificationPredictor.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            ClassificationPredictor::ClassificationPredictor(
                std::shared_ptr<Config::Configuration> config,
                const std::string& checkpointPath,
                WheelDL::Utils::Logger* logger,
                WheelDL::Utils::PerformanceProfiler* profiler,
                std::atomic<bool>* stopFlag)
                : BasePredictor(config, checkpointPath, logger, profiler, stopFlag)
                , _numClasses(config->getNumClasses())
            {
                _logger->info("ClassificationPredictor", "Initializing classification predictor");

                // Verify task type
                if (config->getTaskType() != TaskType::CLASSIFICATION) {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be CLASSIFICATION for ClassificationPredictor"
                    );
                }

                _logger->info("ClassificationPredictor", "Number of classes: " + std::to_string(_numClasses));

                // Setup model
                setupModel();

                // Load checkpoint if provided
                if (!checkpointPath.empty()) {
                    loadCheckpoint(checkpointPath);
                }

                _logger->info("ClassificationPredictor", "Classification predictor initialized");
            }

            void ClassificationPredictor::setupModel()
            {
                _profiler->start("setup_model");
                _logger->info("ClassificationPredictor", "Setting up classification model");

                try {
                    _model = std::make_unique<Model::ClassificationModel>(_config);
                    _logger->info("ClassificationPredictor", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw WheelDL::Utils::ModelException(
                        WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup classification model: " + std::string(e.what())
                    );
                }

                _profiler->stop("setup_model");
            }

            void ClassificationPredictor::postprocess(
                const std::vector<torch::Tensor>& output,
                const std::tuple<int, int>& originalShape,
                const std::string& imagePath)
            {
                try
                {
                    if (output.empty()) {
                        return;
                    }

                    // output[0]: logits [1, num_classes]
                    const torch::Tensor& logits = output[0];

                    // Apply softmax to get probabilities
                    auto probabilities = torch::softmax(logits, 1);  // [1, num_classes]

                    // Get top-1 prediction (argmax)
                    auto maxResult = probabilities.max(1);
                    auto maxScore = std::get<0>(maxResult).squeeze();  // scalar
                    auto maxIndex = std::get<1>(maxResult).squeeze();  // scalar

                    // Move to CPU and extract values
                    float confidence = maxScore.cpu().item<float>();
                    int64_t classId = maxIndex.cpu().item<int64_t>();

                    // Store in internal container
                    ClassificationResult result;
                    result.imagePath = imagePath;
                    result.score = confidence;
                    result.classId = static_cast<int>(classId);
                    result.originalShape = {std::get<0>(originalShape), std::get<1>(originalShape)};
                    result.inferenceTimeMs = static_cast<float>(_profiler->getDuration("inference"));

                    _results.push_back(std::move(result));
                }
                catch (const std::exception& e) {
                    _logger->error("ClassificationPredictor", "Postprocessing failed: " + std::string(e.what()));
                }
            }

            void ClassificationPredictor::exportResults(const std::string& outputDir)
            {
                namespace fs = std::filesystem;

                _profiler->start("export_results");

                fs::path resultsPath = fs::path(outputDir);
                fs::create_directories(resultsPath);

                nlohmann::json predictionsJson = nlohmann::json::array();
                predictionsJson.get<nlohmann::json::array_t>().reserve(_results.size());
                auto classNames = _config->getClassNames();

                for (const auto& result : _results) {
                    nlohmann::json predJson;
                    predJson["image_path"] = result.imagePath;
                    predJson["inference_time_ms"] = result.inferenceTimeMs;
                    predJson["predicted_class"] = result.classId;
                    predJson["label"] = classNames[result.classId];
                    predJson["confidence"] = result.score;

                    predictionsJson.push_back(predJson);
                }

                // Write to file
                fs::path outPath = resultsPath / "predictions.json";

                nlohmann::json rootJson;
                rootJson["predictions"] = predictionsJson;
                rootJson["total_predictions"] = _results.size();

                std::ofstream file(outPath);
                if (!file.is_open()) {
                    _logger->error("ClassificationPredictor", "Failed to create predictions.json file");
                    _profiler->stop("export_results");
                    return;
                }
                file << std::setw(4) << rootJson << std::endl;
                file.close();

                if (file.fail()) {
                    _logger->error("ClassificationPredictor", "Failed to write predictions.json file");
                }

                _logger->info("ClassificationPredictor", "Exported " +
                    std::to_string(_results.size()) + " prediction results to " + outPath.string());

                _profiler->stop("export_results");
            }

            void ClassificationPredictor::clearResults()
            {
                _results.clear();
            }

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
