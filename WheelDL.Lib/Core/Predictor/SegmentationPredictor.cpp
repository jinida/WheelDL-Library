#include "pch.h"
#include "SegmentationPredictor.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace WheelDL {
    namespace Core {
        namespace Predictor {

            SegmentationPredictor::SegmentationPredictor(
                std::shared_ptr<Config::Configuration> config,
                const std::string& checkpointPath,
                WheelDL::Utils::Logger* logger,
                WheelDL::Utils::PerformanceProfiler* profiler,
                std::atomic<bool>* stopFlag)
                : BasePredictor(config, checkpointPath, logger, profiler, stopFlag)
                , _numClasses(config->getNumClasses())
                , _minContourArea(5.0f)
            {
                _logger->info("SegmentationPredictor", "Initializing segmentation predictor");

                // Verify task type
                if (config->getTaskType() != TaskType::SEGMENTATION) {
                    throw WheelDL::Utils::ConfigurationException(
                        WheelDL::Utils::ErrorCode::INVALID_CONFIG,
                        "Configuration task type must be SEGMENTATION for SegmentationPredictor"
                    );
                }

                _logger->info("SegmentationPredictor", "Number of classes: " + std::to_string(_numClasses));

                // Setup model
                setupModel();

                // Load checkpoint if provided
                if (!checkpointPath.empty()) {
                    loadCheckpoint(checkpointPath);
                }

                _logger->info("SegmentationPredictor", "Confidence threshold: " + std::to_string(_threshold));
                _logger->info("SegmentationPredictor", "Segmentation predictor initialized");
            }

            void SegmentationPredictor::setupModel()
            {
                _profiler->start("setup_model");
                _logger->info("SegmentationPredictor", "Setting up segmentation model");

                try {
                    _model = std::make_unique<Model::SegmentationModel>(_config);
                    _logger->info("SegmentationPredictor", "Model setup completed");
                }
                catch (const std::exception& e) {
                    throw WheelDL::Utils::ModelException(
                        WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
                        "Failed to setup segmentation model: " + std::string(e.what())
                    );
                }

                _profiler->stop("setup_model");
            }

            void SegmentationPredictor::postprocess(
                const std::vector<torch::Tensor>& output,
                const std::tuple<int, int>& originalShape,
                const std::string& imagePath)
            {
                if (output.empty()) {
                    return;
                }

                try
                {
                    int origH = std::get<0>(originalShape);
                    int origW = std::get<1>(originalShape);
                    int inputSize = _config->getImageSize();

                    float epsilon = 1e-6f;
                    float clampedThresh = std::max(epsilon, std::min(_threshold, 1.0f - epsilon));
                    float logitThreshold = std::log(clampedThresh / (1.0f - clampedThresh));

                    auto rawOutput = output[0].squeeze(0).cpu().contiguous();
                    int maskH = rawOutput.size(1);
                    int maskW = rawOutput.size(2);
                    int64_t maskArea = static_cast<int64_t>(maskH) * maskW;

                    float scale = std::min(
                        static_cast<float>(inputSize) / origH,
                        static_cast<float>(inputSize) / origW
                    );

                    float padX = (inputSize - origW * scale) / 2.0f;
                    float padY = (inputSize - origH * scale) / 2.0f;

                    float maskScaleX = maskW / static_cast<float>(inputSize);
                    float maskScaleY = maskH / static_cast<float>(inputSize);

                    const float* logitsPtr = rawOutput.data_ptr<float>();

                    // Store in internal container
                    SegmentationResult result;
                    result.imagePath = imagePath;
                    result.originalShape = {origH, origW};
                    result.inferenceTimeMs = static_cast<float>(_profiler->getDuration("inference"));

                    cv::Mat maskMat;
                    std::vector<std::vector<cv::Point>> cvContours;

                    for (int c = 0; c < _numClasses; ++c)
                    {
                        const float* classLogits = logitsPtr + c * maskArea;

                        cv::Mat logitMat(maskH, maskW, CV_32FC1, const_cast<float*>(classLogits));

                        cv::threshold(logitMat, maskMat, logitThreshold, 255, cv::THRESH_BINARY);
                        maskMat.convertTo(maskMat, CV_8UC1);

                        cv::findContours(maskMat, cvContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                        for (const auto& originalContour : cvContours)
                        {
                            if (originalContour.size() < 3) continue;

                            double area = cv::contourArea(originalContour);

                            double scaledArea = area / (scale * maskScaleX * scale * maskScaleY);
                            if (scaledArea < _minContourArea) continue;

                            std::vector<cv::Point> cvContour = originalContour;
                            const int MAX_POINTS = 500;

                            if (cvContour.size() > MAX_POINTS) {
                                double perimeter = cv::arcLength(originalContour, true);
                                double eps = perimeter * 0.005;

                                cv::approxPolyDP(originalContour, cvContour, eps, true);

                                if (cvContour.size() > MAX_POINTS) {
                                    int step = cvContour.size() / MAX_POINTS + 1;

                                    std::vector<cv::Point> subsampledContour;
                                    for (size_t i = 0; i < cvContour.size(); i += step) {
                                        subsampledContour.push_back(cvContour[i]);
                                    }
                                    cvContour = subsampledContour;
                                }
                            }

                            cv::Rect bbox = cv::boundingRect(cvContour);
                            bbox &= cv::Rect(0, 0, maskW, maskH);

                            Contour contour;
                            contour.points.reserve(cvContour.size() * 2);

                            for (const auto& pt : cvContour) {
                                float x = (pt.x / maskScaleX - padX) / scale;
                                float y = (pt.y / maskScaleY - padY) / scale;

                                x = std::max(0.0f, std::min(x, static_cast<float>(origW)));
                                y = std::max(0.0f, std::min(y, static_cast<float>(origH)));

                                contour.points.push_back(x);
                                contour.points.push_back(y);
                            }

                            float sumProb = 0.0f;
                            int count = 0;

                            for (int y = bbox.y; y < bbox.y + bbox.height; ++y) {
                                const float* row = classLogits + y * maskW;
                                for (int x = bbox.x; x < bbox.x + bbox.width; ++x) {
                                    float lg = row[x];
                                    if (lg > logitThreshold) {
                                        float prob = 1.0f / (1.0f + std::exp(-lg));
                                        sumProb += prob;
                                        ++count;
                                    }
                                }
                            }

                            result.contours.push_back(std::move(contour));
                            result.classIds.push_back(c);
                            result.scores.push_back(count > 0 ? sumProb / count : 0.0f);
                        }
                    }

                    _results.push_back(std::move(result));
                }
                catch (const std::exception& e) {
                    _logger->error("SegmentationPredictor", "Postprocessing failed: " + std::string(e.what()));
                }
            }

            void SegmentationPredictor::exportResults(const std::string& outputDir)
            {
                _profiler->start("export_results");

                nlohmann::json predictionsJson = nlohmann::json::array();
                predictionsJson.get<nlohmann::json::array_t>().reserve(_results.size());
                auto classNames = _config->getClassNames();

                for (const auto& result : _results) {
                    nlohmann::json predJson;
                    predJson["image_path"] = result.imagePath;
                    predJson["inference_time_ms"] = result.inferenceTimeMs;
                    predJson["num_segments"] = result.contours.size();

                    nlohmann::json segmentsJson = nlohmann::json::array();
                    for (size_t i = 0; i < result.contours.size(); ++i) {
                        nlohmann::json segJson;

                        // Contour points
                        const auto& contour = result.contours[i];
                        nlohmann::json pointsJson = nlohmann::json::array();
                        for (size_t k = 0; k + 1 < contour.points.size(); k += 2) {
                            pointsJson.push_back(contour.points[k]);
                            pointsJson.push_back(contour.points[k + 1]);
                        }
                        segJson["contour"] = pointsJson;

                        // Class info
                        int classId = result.classIds[i];
                        segJson["class_id"] = classId;

                        auto it = classNames.find(classId);
                        if (it != classNames.end()) {
                            segJson["class_name"] = it->second;
                        }

                        // Confidence
                        segJson["confidence"] = result.scores[i];

                        segmentsJson.push_back(segJson);
                    }

                    predJson["segments"] = segmentsJson;
                    predictionsJson.push_back(predJson);
                }

                // Write to file
                std::filesystem::path outPath = std::filesystem::path(outputDir) / "predictions.json";
                std::filesystem::create_directories(outputDir);

                nlohmann::json rootJson;
                rootJson["predictions"] = predictionsJson;
                rootJson["total_images"] = _results.size();

                // Calculate total segments
                size_t totalSegments = 0;
                for (const auto& result : _results) {
                    totalSegments += result.contours.size();
                }
                rootJson["total_segments"] = totalSegments;

                std::ofstream file(outPath);
                if (!file.is_open()) {
                    _logger->error("SegmentationPredictor", "Failed to create predictions.json file");
                    _profiler->stop("export_results");
                    return;
                }
                file << std::setw(4) << rootJson << std::endl;
                file.close();

                if (file.fail()) {
                    _logger->error("SegmentationPredictor", "Failed to write predictions.json file");
                }

                _logger->info("SegmentationPredictor", "Exported " +
                    std::to_string(_results.size()) + " images with " +
                    std::to_string(totalSegments) + " total segments to " + outPath.string());

                _profiler->stop("export_results");
            }

            void SegmentationPredictor::clearResults()
            {
                _results.clear();
            }

        } // namespace Predictor
    } // namespace Core
} // namespace WheelDL
