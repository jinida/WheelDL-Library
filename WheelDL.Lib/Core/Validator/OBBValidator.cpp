#include "pch.h"
#include "OBBValidator.h"
#include "../../Model/Task/OBBModel.h"
#include "../../Data/Dataset/OBBDataset.h"
#include "../../Data/Transforms/Collation.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <algorithm>
#include <numeric>

namespace WheelDL {
    namespace Core {
        namespace Validator {

            OBBValidator::OBBValidator(
                std::shared_ptr<Config::Configuration> config,
                WheelDL::Utils::Logger* logger,
                WheelDL::Utils::PerformanceProfiler* profiler,
                std::atomic<bool>* stopFlag)
                : BaseValidator(config, logger, profiler, stopFlag)
                , _numClasses(config->getNumClasses())
                , _confThresh(0.2f)  // Low threshold for mAP calculation
                , _iouThresh(config->getIoU())
                , _maxDet(config->getMaxDet())
            {
                _logger->info("OBBValidator",
                    "OBB validator initialized with " +
                    std::to_string(_numClasses) + " classes, IoU threshold: " +
                    std::to_string(_iouThresh));
            }

            std::unique_ptr<Model::BaseModel> OBBValidator::setupModel()
            {
                _logger->info("OBBValidator", "Creating OBBModel");
                return std::make_unique<Model::OBBModel>(_config);
            }

            void OBBValidator::setupDataLoader()
            {
                if (_batchIterator) {
                    _logger->info("OBBValidator",
                        "Using injected DataLoader for validation");
                    return;
                }

                auto valDataset = std::make_shared<Data::Dataset::OBBDataset>(*_config, false);

                _logger->info("OBBValidator",
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

            WheelDL::Data::Dataset::DataExample OBBValidator::preprocessBatch(
                const WheelDL::Data::Dataset::DataExample& batch)
            {
                WheelDL::Data::Dataset::DataExample preprocessed;
                preprocessed.data = batch.data.to(_device);

                auto batchIdx = batch.batchIndices.view({ -1, 1 });
                auto classes = batch.classes.view({ -1, 1 });
                auto bboxes = batch.targets;  // [num_gt, 5] for OBB

                auto targets = torch::cat({
                    batchIdx.to(_device),
                    classes.to(_device),
                    bboxes.to(_device)
                    }, 1);

                preprocessed.targets = targets;
                return preprocessed;
            }

            torch::Tensor OBBValidator::postprocessBatch(
                const std::vector<torch::Tensor>& prediction)
            {
                if (prediction.empty()) {
                    return torch::empty({0, 7});
                }

                torch::Tensor pred = prediction[0]; 
                
                return pred;
            }

            MetricsData OBBValidator::computeMetrics(
                const torch::Tensor& pred,
                const torch::Tensor& target)
            {
                _profiler->start("compute_metrics");
                MetricsData metrics{};

                try
                {
                    const float imageSize = static_cast<float>(_config->getImageSize());
                    const int numClasses = _config->getNumClasses();

                    auto targetCPU = target.cpu().contiguous();
                    const int totalTargets = static_cast<int>(targetCPU.size(0));

                    if (totalTargets == 0) {
                        _logger->info("OBBValidator", "No ground truth targets");
                        _profiler->stop("compute_metrics");
                        return metrics;
                    }

                    if (pred.numel() == 0 || pred.size(0) == 0) {
                        _logger->info("OBBValidator", "No predictions");
                        _profiler->stop("compute_metrics");
                        return metrics;
                    }

                    auto predOBBs = Model::Utils::nonMaxSuppressionOBB(pred, _confThresh, _iouThresh, _maxDet);
                    const int numImages = static_cast<int>(predOBBs.size());

                    if (numImages == 0) {
                        _logger->info("OBBValidator", "No predictions after NMS");
                        _profiler->stop("compute_metrics");
                        return metrics;
                    }

                    const float* targetPtr = targetCPU.data_ptr<float>();

                    constexpr int NT = 10;
                    constexpr float thresholds[NT] = { 0.5f, 0.55f, 0.6f, 0.65f, 0.7f, 0.75f, 0.8f, 0.85f, 0.9f, 0.95f };

                    std::vector<float> apPerClass(numClasses, 0.0f);
                    std::vector<float> ap50PerClass(numClasses, 0.0f);
                    std::vector<int> gtCountPerClass(numClasses, 0);

                    struct GTBox {
                        int imgIdx;
                        int classId;
                        float cx, cy, w, h, angle;
                        std::array<bool, NT> matched;
                    };
                    std::vector<GTBox> gtBoxes(totalTargets);

                    for (int i = 0; i < totalTargets; ++i) {
                        const float* row = targetPtr + i * 7;
                        auto& gt = gtBoxes[i];
                        gt.imgIdx = static_cast<int>(row[0]);
                        gt.classId = static_cast<int>(row[1]);
                        gt.cx = row[2] * imageSize;
                        gt.cy = row[3] * imageSize;
                        gt.w = row[4] * imageSize;
                        gt.h = row[5] * imageSize;
                        gt.angle = row[6];
                        gt.matched.fill(false);

                        if (gt.classId >= 0 && gt.classId < numClasses) {
                            gtCountPerClass[gt.classId]++;
                        }
                    }

                    struct PredBox {
                        int imgIdx;
                        int classId;
                        float cx, cy, w, h, angle;
                        float conf;
                    };
                    std::vector<PredBox> predBoxes;
                    predBoxes.reserve(numImages * _maxDet);

                    for (int b = 0; b < numImages; ++b) {
                        auto& obbs = predOBBs[b];
                        if (obbs.numel() == 0 || obbs.size(0) == 0) continue;

                        auto obbsCPU = obbs.cpu().contiguous();
                        const float* ptr = obbsCPU.data_ptr<float>();
                        const int numPred = static_cast<int>(obbs.size(0));

                        for (int i = 0; i < numPred; ++i) {
                            const float* p = ptr + i * 7;
                            predBoxes.push_back({
                                b,
                                static_cast<int>(p[6]),
                                p[0], p[1], p[2], p[3], p[4],
                                p[5]
                                });
                        }
                    }

                    const int totalPreds = static_cast<int>(predBoxes.size());

                    if (totalPreds == 0) {
                        _logger->info("OBBValidator",
                            "mAP@0.5:0.95: 0 | mAP@0.5: 0 | Precision: 0 | Recall: 0");
                        _profiler->stop("compute_metrics");
                        return metrics;
                    }

                    std::vector<int> sortedIdx(totalPreds);
                    std::iota(sortedIdx.begin(), sortedIdx.end(), 0);
                    std::sort(sortedIdx.begin(), sortedIdx.end(), [&](int a, int b) {
                        return predBoxes[a].conf > predBoxes[b].conf;
                    });

                    std::vector<std::vector<int>> gtIdxByClass(numClasses);
                    std::vector<std::vector<int>> predIdxByClass(numClasses);

                    for (int i = 0; i < totalTargets; ++i) {
                        int c = gtBoxes[i].classId;
                        if (c >= 0 && c < numClasses) {
                            gtIdxByClass[c].push_back(i);
                        }
                    }

                    for (int idx : sortedIdx) {
                        int c = predBoxes[idx].classId;
                        if (c >= 0 && c < numClasses) {
                            predIdxByClass[c].push_back(idx);
                        }
                    }

                    for (int c = 0; c < numClasses; ++c)
                    {
                        const int numGT = gtCountPerClass[c];
                        if (numGT == 0) continue;

                        const auto& classGtIdx = gtIdxByClass[c];
                        const auto& classPredIdx = predIdxByClass[c];
                        const int numPred = static_cast<int>(classPredIdx.size());

                        if (numPred == 0) continue;

                        std::vector<float> predData;
                        std::vector<float> gtData;
                        std::vector<std::pair<int, int>> pairMapping;

                        predData.reserve(numPred * classGtIdx.size() * 5);
                        gtData.reserve(numPred * classGtIdx.size() * 5);
                        pairMapping.reserve(numPred * classGtIdx.size());

                        for (int pi = 0; pi < numPred; ++pi) {
                            const auto& pBox = predBoxes[classPredIdx[pi]];

                            for (size_t gi = 0; gi < classGtIdx.size(); ++gi) {
                                const auto& gBox = gtBoxes[classGtIdx[gi]];

                                if (pBox.imgIdx != gBox.imgIdx) continue;

                                predData.insert(predData.end(), { pBox.cx, pBox.cy, pBox.w, pBox.h, pBox.angle });
                                gtData.insert(gtData.end(), { gBox.cx, gBox.cy, gBox.w, gBox.h, gBox.angle });
                                pairMapping.push_back({ pi, static_cast<int>(gi) });
                            }
                        }

                        std::vector<std::vector<std::pair<int, float>>> predToGtIoU(numPred);

                        if (!pairMapping.empty()) {
                            const int numPairs = static_cast<int>(pairMapping.size());
                            auto predTensor = torch::from_blob(predData.data(), { numPairs, 5 }, torch::kFloat32).clone();
                            auto gtTensor = torch::from_blob(gtData.data(), { numPairs, 5 }, torch::kFloat32).clone();

                            auto iouTensor = Model::Utils::probiou(predTensor, gtTensor).cpu();
                            const float* iouPtr = iouTensor.data_ptr<float>();

                            for (int i = 0; i < numPairs; ++i) {
                                int pi = pairMapping[i].first;
                                int gi = pairMapping[i].second;
                                predToGtIoU[pi].push_back({ gi, iouPtr[i] });
                            }
                        }

                        float apSum = 0.0f;

                        for (int t = 0; t < NT; ++t)
                        {
                            const float thresh = thresholds[t];

                            for (int gi : classGtIdx) {
                                gtBoxes[gi].matched[t] = false;
                            }

                            std::vector<float> prec(numPred);
                            std::vector<float> rec(numPred);
                            int tp = 0, fp = 0;

                            for (int pi = 0; pi < numPred; ++pi) {
                                int bestGtLocalIdx = -1;
                                float bestIoU = thresh;

                                for (const auto& [gi, iou] : predToGtIoU[pi]) {
                                    if (gtBoxes[classGtIdx[gi]].matched[t]) continue;

                                    if (iou >= bestIoU) {
                                        bestIoU = iou;
                                        bestGtLocalIdx = gi;
                                    }
                                }

                                if (bestGtLocalIdx >= 0) {
                                    gtBoxes[classGtIdx[bestGtLocalIdx]].matched[t] = true;
                                    tp++;
                                }
                                else {
                                    fp++;
                                }

                                prec[pi] = static_cast<float>(tp) / (tp + fp);
                                rec[pi] = static_cast<float>(tp) / numGT;
                            }

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
                        if (gtBoxes[i].matched[0]) totalTP++;
                    }

                    if (totalPreds > 0) {
                        metrics.precision = static_cast<float>(totalTP) / totalPreds;
                    }
                    if (totalTargets > 0) {
                        metrics.recall = static_cast<float>(totalTP) / totalTargets;
                    }

                    _logger->info("OBBValidator",
                        "mAP@0.5:0.95: " + std::to_string(metrics.mAP) +
                        " | mAP@0.5: " + std::to_string(metrics.aucROC) +
                        " | Precision: " + std::to_string(metrics.precision) +
                        " | Recall: " + std::to_string(metrics.recall));
                }
                catch (const std::exception& e) {
                    _logger->error("OBBValidator", "Failed to compute metrics: " + std::string(e.what()));
                }

                _profiler->stop("compute_metrics");
                return metrics;
            }
        } // namespace Validator
    } // namespace Core
} // namespace WheelDL
