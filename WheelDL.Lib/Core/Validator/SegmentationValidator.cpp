#include "pch.h"
#include "SegmentationValidator.h"
#include "../../Data/Dataset/SegmentationDataset.h"
#include "../../Data/Transforms/Collation.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <algorithm>
#include <numeric>

namespace WheelDL {
    namespace Core {
        namespace Validator {

            SegmentationValidator::SegmentationValidator(const std::shared_ptr<Config::Configuration> config)
                : BaseValidator(config, torch::Device(torch::kCPU))
                , _numClasses(config->getNumClasses())
            {
                _logger->info("SegmentationValidator",
                    "Segmentation validator initialized with " +
                    std::to_string(_numClasses) + " classes");
            }

            void SegmentationValidator::setupDataLoader()
            {
                if (_batchIterator) {
                    _logger->info("SegmentationValidator",
                        "Using injected DataLoader for validation");
                    return;
                }

                auto valDataset = std::make_shared<Data::Dataset::SegmentationDataset>(*_config, false);

                _logger->info("SegmentationValidator",
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

            WheelDL::Data::Dataset::DataExample SegmentationValidator::preprocessBatch(
                const WheelDL::Data::Dataset::DataExample& batch)
            {
                WheelDL::Data::Dataset::DataExample preprocessed;
                preprocessed.data = batch.data.to(_device);

                // Segmentation targets are one-hot encoded [N, C, H, W]
                if (batch.targets.defined()) {
                    preprocessed.targets = batch.targets.to(_device);
                }

                if (batch.classes.defined()) {
                    preprocessed.classes = batch.classes.to(_device);
                }

                return preprocessed;
            }

            torch::Tensor SegmentationValidator::postprocessBatch(
                const std::vector<torch::Tensor>& prediction)
            {
                if (prediction.empty()) {
                    return torch::empty({0});
                }

                // Apply sigmoid to convert logits to probabilities
                // prediction[0]: [N, C, H, W] logits
                torch::Tensor probs = torch::sigmoid(prediction[0]);
                return probs;
            }

            MetricsData SegmentationValidator::computeMetrics(
                const torch::Tensor& pred,
                const torch::Tensor& target)
            {
                _profiler.start("compute_metrics");

                MetricsData metrics{};

                try
                {
                    if (pred.dim() != 4 || target.dim() != 4) {
                        _profiler.stop("compute_metrics");
                        return metrics;
                    }

                    const int64_t totalBatch = pred.size(0);
                    const int64_t numClasses = pred.size(1);
                    const int64_t chunkSize = 32;
                    const float smooth = 1e-6f;

                    constexpr int NT = 17;
                    constexpr float THRESH_MIN = 0.1f;
                    constexpr float THRESH_STEP = 0.05f;

                    std::vector<std::vector<double>> accTP(NT, std::vector<double>(numClasses, 0.0));
                    std::vector<std::vector<double>> accPred(NT, std::vector<double>(numClasses, 0.0));
                    std::vector<double> accTarget(numClasses, 0.0);
                    std::vector<int64_t> accCorrect(NT, 0);
                    int64_t totalPixels = 0;

                    for (int64_t start = 0; start < totalBatch; start += chunkSize) {
                        int64_t end = std::min(start + chunkSize, totalBatch);
            
                        auto pc = pred.slice(0, start, end).cpu();
                        auto tc = target.slice(0, start, end);

                        if (pc.size(2) != tc.size(2) || pc.size(3) != tc.size(3)) {
                            pc = torch::nn::functional::interpolate(pc,
                                torch::nn::functional::InterpolateFuncOptions()
                                    .size(std::vector<int64_t>{tc.size(2), tc.size(3)})
                                    .mode(torch::kBilinear)
                                    .align_corners(false));
                        }

                        auto tb = tc.to(torch::kBool);
                        auto targetSum = tb.sum({0, 2, 3}).cpu();
            
                        for (int64_t c = 0; c < numClasses; ++c) {
                            accTarget[c] += targetSum[c].item<float>();
                        }
            
                        totalPixels += pc.numel();

                        for (int t = 0; t < NT; ++t) {
                            float thresh = THRESH_MIN + t * THRESH_STEP;
                            auto pb = (pc > thresh);

                            auto tp = (pb & tb).sum({0, 2, 3}).cpu();
                            auto predSum = pb.sum({0, 2, 3}).cpu();

                            for (int64_t c = 0; c < numClasses; ++c) {
                                accTP[t][c] += tp[c].item<double>();
                                accPred[t][c] += predSum[c].item<double>();
                            }

                            accCorrect[t] += pb.eq(tb).sum().item<int64_t>();
                        }
                    }

                    float bestThreshold = 0.5f;
                    float bestF1 = 0.0f;
                    int bestIdx = NT / 2;

                    for (int t = 0; t < NT; ++t) {
                        double sumPrec = 0.0, sumRec = 0.0;
                        int valid = 0;

                        for (int64_t c = 0; c < numClasses; ++c) {
                            if (accTarget[c] > 0) {
                                double tp = accTP[t][c];
                                double fp = accPred[t][c] - tp;
                                double fn = accTarget[c] - tp;

                                sumPrec += (tp + smooth) / (tp + fp + smooth);
                                sumRec += (tp + smooth) / (tp + fn + smooth);
                                ++valid;
                            }
                        }

                        if (valid > 0) {
                            float prec = static_cast<float>(sumPrec / valid);
                            float rec = static_cast<float>(sumRec / valid);
                            float f1 = 2.0f * prec * rec / (prec + rec + smooth);

                            if (f1 > bestF1) {
                                bestF1 = f1;
                                bestThreshold = THRESH_MIN + t * THRESH_STEP;
                                bestIdx = t;
                            }
                        }
                    }

                    double sumIoU = 0.0, sumPrec = 0.0, sumRec = 0.0;
                    int valid = 0;

                    for (int64_t c = 0; c < numClasses; ++c) {
                        if (accTarget[c] > 0) {
                            double tp = accTP[bestIdx][c];
                            double fp = accPred[bestIdx][c] - tp;
                            double fn = accTarget[c] - tp;

                            sumIoU += (tp + smooth) / (tp + fp + fn + smooth);
                            sumPrec += (tp + smooth) / (tp + fp + smooth);
                            sumRec += (tp + smooth) / (tp + fn + smooth);
                            ++valid;
                        }
                    }

                    if (valid > 0) {
                        metrics.mAP = static_cast<float>(sumIoU / valid);
                        metrics.precision = static_cast<float>(sumPrec / valid);
                        metrics.recall = static_cast<float>(sumRec / valid);
                        metrics.f1Score = 2.0f * metrics.precision * metrics.recall 
                                        / (metrics.precision + metrics.recall + smooth);
                    }

                    metrics.accuracy = totalPixels > 0 
                        ? static_cast<float>(accCorrect[bestIdx]) / totalPixels 
                        : 0.0f;
                    metrics.fitness = metrics.mAP;
                    metrics.threshold = bestThreshold;

                    _logger->info("SegmentationValidator",
                        "mIoU: " + std::to_string(metrics.mAP) +
                        " | F1: " + std::to_string(metrics.f1Score) +
                        " | Thresh: " + std::to_string(bestThreshold));
                }
                catch (const std::exception& e) {
                    _logger->error("SegmentationValidator", std::string(e.what()));
                }

                _profiler.stop("compute_metrics");
                return metrics;
            }
        } // namespace Validator
    } // namespace Core
} // namespace WheelDL
