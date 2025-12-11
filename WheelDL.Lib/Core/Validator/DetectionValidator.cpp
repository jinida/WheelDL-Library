#include "pch.h"
#include "DetectionValidator.h"
#include "../../Data/Dataset/DetectionDataset.h"
#include "../../Data/Transforms/Collation.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <algorithm>
#include <numeric>

namespace WheelDL {
    namespace Core {
        namespace Validator {

            DetectionValidator::DetectionValidator(
                std::shared_ptr<Config::Configuration> config,
                WheelDL::Utils::Logger* logger,
                WheelDL::Utils::PerformanceProfiler* profiler,
                std::atomic<bool>* stopFlag)
                : BaseValidator(config, logger, profiler, stopFlag)
                , _numClasses(config->getNumClasses())
                , _confThresh(0.5f)  // Low threshold for mAP calculation
                , _iouThresh(config->getIoU())
                , _maxDet(config->getMaxDet())
            {
                _logger->info("DetectionValidator",
                    "Detection validator initialized with " +
                    std::to_string(_numClasses) + " classes, IoU threshold: " +
                    std::to_string(_iouThresh));
            }

            std::unique_ptr<Model::BaseModel> DetectionValidator::setupModel()
            {
                _logger->info("DetectionValidator", "Creating DetectionModel");
                return std::make_unique<Model::DetectionModel>(_config);
            }

            void DetectionValidator::setupDataLoader()
            {
                if (_batchIterator) {
                    _logger->info("DetectionValidator",
                        "Using injected DataLoader for validation");
                    return;
                }

                auto valDataset = std::make_shared<Data::Dataset::DetectionDataset>(*_config, false);

                _logger->info("DetectionValidator",
                    "Validation dataset created with " +
                    std::to_string(valDataset->size().value_or(0)) + " samples");

                auto valLoader = torch::data::make_data_loader(
                    std::move(valDataset->map(Data::DataExampleCollation())),
                    torch::data::DataLoaderOptions()
                        .batch_size(_config->getBatchSize())
                        .workers(_config->getWorkers())
                        .drop_last(false)
                );

                setDataLoader(std::move(valLoader));
            }

            WheelDL::Data::Dataset::DataExample DetectionValidator::preprocessBatch(
                const WheelDL::Data::Dataset::DataExample& batch)
            {
                WheelDL::Data::Dataset::DataExample preprocessed;
                preprocessed.data = batch.data.to(_device);

                auto batchIdx = batch.batchIndices.view({ -1, 1 });
                auto classes = batch.classes.view({ -1, 1 });
                auto bboxes = batch.targets;

                auto targets = torch::cat({
                    batchIdx.to(_device),
                    classes.to(_device),
                    bboxes.to(_device)
                    }, 1);

                preprocessed.targets = targets;
				return preprocessed;
            }

            torch::Tensor DetectionValidator::postprocessBatch(
                const std::vector<torch::Tensor>& prediction)
            {
                if (prediction.empty()) {
                    return torch::empty({0, 6});
                }

                torch::Tensor pred = prediction[0]; // [N, 8400, 6]
                return pred;
            }

            MetricsData DetectionValidator::computeMetrics(
                const torch::Tensor& pred,
                const torch::Tensor& target)
            {
                _profiler->start("compute_metrics");
                MetricsData metrics{};
                try
                {
                    auto predBboxes = Model::Utils::nonMaxSuppression(pred, _confThresh, _iouThresh, _maxDet);
                    const float imageSize = static_cast<float>(_config->getImageSize());
                    const int numImages = static_cast<int>(predBboxes.size());
                    const int numClasses = _config->getNumClasses();

                    auto targetCPU = target.cpu().contiguous();
                    const float* targetPtr = targetCPU.data_ptr<float>();
                    const int totalTargets = static_cast<int>(targetCPU.size(0));

                    constexpr int NT = 10;
                    constexpr float thresholds[NT] = { 0.5f, 0.55f, 0.6f, 0.65f, 0.7f, 0.75f, 0.8f, 0.85f, 0.9f, 0.95f };

                    std::vector<float> apPerClass(numClasses, 0.0f);
                    std::vector<float> ap50PerClass(numClasses, 0.0f);
                    std::vector<int> gtCountPerClass(numClasses, 0);

                    std::vector<std::vector<float>> gtBoxes(totalTargets);
                    std::vector<int> gtImgIdx(totalTargets);
                    std::vector<int> gtClassId(totalTargets);
                    std::vector<std::array<bool, NT>> gtMatched(totalTargets);

                    for (int i = 0; i < totalTargets; ++i) {
                        const float* row = targetPtr + i * 6;
                        int imgIdx = static_cast<int>(row[0]);
                        int classId = static_cast<int>(row[1]);
                        float cx = row[2] * imageSize;
                        float cy = row[3] * imageSize;
                        float w = row[4] * imageSize;
                        float h = row[5] * imageSize;

                        gtImgIdx[i] = imgIdx;
                        gtClassId[i] = classId;
                        gtBoxes[i] = { cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f };
                        gtMatched[i].fill(false);

                        if (classId >= 0 && classId < numClasses) {
                            gtCountPerClass[classId]++;
                        }
                    }

                    std::vector<int> predImgIdx;
                    std::vector<int> predClassId;
                    std::vector<float> predConf;
                    std::vector<std::vector<float>> predBoxes;

                    for (int b = 0; b < numImages; ++b) {
                        auto& boxes = predBboxes[b];
                        if (boxes.size(0) == 0) continue;
                        auto boxesCPU = boxes.cpu().contiguous();
                        const float* ptr = boxesCPU.data_ptr<float>();
                        int numPred = static_cast<int>(boxes.size(0));

                        for (int i = 0; i < numPred; ++i) {
                            const float* p = ptr + i * 6;
                            predImgIdx.push_back(b);
                            predBoxes.push_back({ p[0], p[1], p[2], p[3] });
                            predConf.push_back(p[4]);
                            predClassId.push_back(static_cast<int>(p[5]));
                        }
                    }

                    int totalPreds = static_cast<int>(predConf.size());

                    std::vector<int> sortedIdx(totalPreds);
                    std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
                    std::sort(sortedIdx.begin(), sortedIdx.end(), [&](int a, int b) {
                        return predConf[a] > predConf[b];
                    });
                    
                    for (int c = 0; c < numClasses; ++c) 
                    {
                        int numGT = gtCountPerClass[c];
                        if (numGT == 0) continue;

                        std::vector<int> classGtIdx;
                        for (int i = 0; i < totalTargets; ++i) {
                            if (gtClassId[i] == c) classGtIdx.push_back(i);
                        }

                        std::vector<int> classPredIdx;
                        for (int i : sortedIdx) {
                            if (predClassId[i] == c) classPredIdx.push_back(i);
                        }

                        int numPred = static_cast<int>(classPredIdx.size());
                        if (numPred == 0) continue;

                        float apSum = 0.0f;

                        for (int t = 0; t < NT; ++t) 
                        {
                            float thresh = thresholds[t];

                            for (int gi : classGtIdx) {
                                gtMatched[gi][t] = false;
                            }

                            std::vector<float> prec(numPred);
                            std::vector<float> rec(numPred);
                            int tp = 0, fp = 0;

                            for (int pi = 0; pi < numPred; ++pi) {
                                int predIdx = classPredIdx[pi];
                                int pImg = predImgIdx[predIdx];
                                const auto& pBox = predBoxes[predIdx];

                                int bestGT = -1;
                                float bestIoU = thresh;

                                for (int gi : classGtIdx) {
                                    if (gtImgIdx[gi] != pImg) continue;
                                    if (gtMatched[gi][t]) continue;

                                    const auto& gBox = gtBoxes[gi];

                                    float ix1 = std::max(pBox[0], gBox[0]);
                                    float iy1 = std::max(pBox[1], gBox[1]);
                                    float ix2 = std::min(pBox[2], gBox[2]);
                                    float iy2 = std::min(pBox[3], gBox[3]);

                                    if (ix2 <= ix1 || iy2 <= iy1) continue;

                                    float inter = (ix2 - ix1) * (iy2 - iy1);
                                    float pArea = (pBox[2] - pBox[0]) * (pBox[3] - pBox[1]);
                                    float gArea = (gBox[2] - gBox[0]) * (gBox[3] - gBox[1]);
                                    float iou = inter / (pArea + gArea - inter);

                                    if (iou >= bestIoU) {
                                        bestIoU = iou;
                                        bestGT = gi;
                                    }
                                }

                                if (bestGT >= 0) {
                                    gtMatched[bestGT][t] = true;
                                    tp++;
                                }
                                else {
                                    fp++;
                                }

                                prec[pi] = static_cast<float>(tp) / (tp + fp);
                                rec[pi] = static_cast<float>(tp) / numGT;
                            }

                            // monotonic decreasing precision
                            for (int i = numPred - 2; i >= 0; --i) {
                                prec[i] = std::max(prec[i], prec[i + 1]);
                            }

                            float ap = 0.0f;
                            float prevRec = 0.0f;
                            for (int i = 0; i < numPred; ++i) {
                                ap += prec[i] * (rec[i] - prevRec);
                                prevRec = rec[i];
                            }

                            apSum += ap;
                            if (t == 0) ap50PerClass[c] = ap;
                        }

                        apPerClass[c] = apSum / NT;
                    }

                    float mAP = 0.0f;
                    float mAP50 = 0.0f;
                    int validClasses = 0;

                    for (int c = 0; c < numClasses; ++c) 
                    {
                        if (gtCountPerClass[c] > 0) 
                        {
                            mAP += apPerClass[c];
                            mAP50 += ap50PerClass[c];
                            validClasses++;
                        }
                    }

                    if (validClasses > 0)
                    {
                        metrics.mAP = mAP / validClasses;
                        metrics.aucROC = mAP50 / validClasses;
                    }

                    int totalTP = 0;
                    for (int i = 0; i < totalTargets; ++i) {
                        if (gtMatched[i][0]) totalTP++;
                    }

                    if (totalPreds > 0) {
                        metrics.precision = static_cast<float>(totalTP) / totalPreds;
                    }
                    if (totalTargets > 0) {
                        metrics.recall = static_cast<float>(totalTP) / totalTargets;
                    }

                    _logger->info("DetectionValidator",
                        "mAP@0.5:0.95: " + std::to_string(metrics.mAP) +
                        " | mAP@0.5: " + std::to_string(metrics.aucROC) +
                        " | Precision: " + std::to_string(metrics.precision) +
                        " | Recall: " + std::to_string(metrics.recall));
                }
                catch (const std::exception& e) {
                    _logger->error("DetectionValidator", "Failed to compute metrics: " + std::string(e.what()));
                }

                _profiler->stop("compute_metrics");
                return metrics;
            }
        } // namespace Validator
    } // namespace Core
} // namespace WheelDL
