#include "pch.h"
#include "ClassificationValidator.h"
#include "../../Data/Dataset/ClassificationDataset.h"
#include "../../Data/Transforms/Collation.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <algorithm>
#include <numeric>

namespace WheelDL {
    namespace Core {
        namespace Validator {

            ClassificationValidator::ClassificationValidator(const std::shared_ptr<Config::Configuration> config)
                : BaseValidator(config, torch::Device(torch::kCPU))
                , _numClasses(config->getNumClasses())
            {
                _logger->info("ClassificationValidator",
                    "Classification validator initialized with " +
                    std::to_string(_numClasses) + " classes");
            }

            void ClassificationValidator::setupDataLoader()
            {
                if (_batchIterator) {
                    _logger->info("ClassificationValidator",
                        "Using injected DataLoader for validation");
                    return;
                }
                auto valDataset = std::make_shared<Data::Dataset::ClassificationDataset>(*_config, false);

                _logger->info("ClassificationValidator",
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

            WheelDL::Data::Dataset::DataExample ClassificationValidator::preprocessBatch(
                const WheelDL::Data::Dataset::DataExample& batch)
            {
                WheelDL::Data::Dataset::DataExample preprocessed;
                preprocessed.data = batch.data.to(_device);

                if (batch.targets.defined()) {
                    preprocessed.targets = batch.targets.to(_device);
                }

                preprocessed.classes = batch.classes.to(_device);

                return preprocessed;
            }

            torch::Tensor ClassificationValidator::postprocessBatch(
                const std::vector<torch::Tensor>& prediction)
            {
                return prediction[0];
            }

            MetricsData ClassificationValidator::computeMetrics(
                const torch::Tensor& pred,
                const torch::Tensor& target)
            {
                _profiler.start("compute_metrics");

                MetricsData metrics;
                metrics.loss = 0.0f;

                try
                {
                    torch::Tensor predictions = pred;
                    torch::Tensor labels = target;

                    if (labels.dim() > 1) {
                        labels = labels.flatten();
                    }

                    if (predictions.size(0) != labels.size(0)) {
                        int64_t minSize = std::min(predictions.size(0), labels.size(0));
                        predictions = predictions.slice(0, 0, minSize);
                        labels = labels.slice(0, 0, minSize);
                    }

                    auto predCPU = predictions.cpu().contiguous();
                    auto labelsCPU = labels.cpu().contiguous().to(torch::kLong);
                    auto predClasses = std::get<1>(predCPU.max(1));

                    auto correct = predClasses.eq(labelsCPU);
                    metrics.accuracy = correct.to(torch::kFloat32).mean().item<float>();

                    float precision = 0.0f;
                    float recall = 0.0f;
                    metrics.f1Score = computePrecisionRecallF1(predClasses, labelsCPU, _numClasses, precision, recall);
                    metrics.precision = precision;
                    metrics.recall = recall;
                    metrics.fitness = metrics.accuracy;
                    metrics.aucROC = computeAUCROC(predCPU, labelsCPU);

                    _logger->info("ClassificationValidator",
                        "Metrics - Accuracy: " + std::to_string(metrics.accuracy) +
                        " | F1: " + std::to_string(metrics.f1Score) +
                        " | Precision: " + std::to_string(metrics.precision) +
                        " | Recall: " + std::to_string(metrics.recall) +
						" | AUC-ROC: " + std::to_string(metrics.aucROC));

                }
                catch (const std::exception& e) {
                    _logger->error("ClassificationValidator", "Failed to compute metrics: " + std::string(e.what()));
                    metrics.accuracy = 0.0f;
                    metrics.fitness = 0.0f;
                    metrics.f1Score = 0.0f;
                    metrics.precision = 0.0f;
                    metrics.recall = 0.0f;
                }

                _profiler.stop("compute_metrics");
                return metrics;
            }

            float ClassificationValidator::computePrecisionRecallF1(
                const torch::Tensor& pred,
                const torch::Tensor& target,
                int numClasses,
                float& outPrecision,
                float& outRecall)
            {
                try {
                    int64_t count = pred.size(0);
                    if (count == 0) {
                        outPrecision = outRecall = 0.0f;
                        return 0.0f;
                    }

                    auto predCPU = pred.to(torch::kCPU, torch::kInt64, /*non_blocking=*/false, /*copy=*/false).contiguous();
                    auto targetCPU = target.to(torch::kCPU, torch::kInt64, false, false).contiguous();

                    auto validMask = (targetCPU >= 0) & (targetCPU < numClasses)
                        & (predCPU >= 0) & (predCPU < numClasses);

                    auto validPred = predCPU.masked_select(validMask);
                    auto validTarget = targetCPU.masked_select(validMask);

                    auto indices = validTarget * static_cast<int64_t>(numClasses) + validPred;
                    auto confusion = torch::bincount(indices, {}, static_cast<int64_t>(numClasses) * numClasses)
                        .reshape({ numClasses, numClasses }).to(torch::kFloat32);

                    auto tp = confusion.diag();
                    auto fp = confusion.sum(0) - tp;
                    auto fn = confusion.sum(1) - tp;

                    auto precisionDenom = (tp + fp).clamp_min(1e-7f);
                    auto recallDenom = (tp + fn).clamp_min(1e-7f);

                    auto precision = tp / precisionDenom;
                    auto recall = tp / recallDenom;
                    auto f1 = 2.0f * tp / (precisionDenom + recallDenom);  // 간소화된 F1 공식

                    auto supportMask = (tp + fn) > 0;
                    int64_t validClasses = supportMask.sum().item<int64_t>();

                    if (validClasses > 0) {
                        outPrecision = precision.masked_select(supportMask).mean().item<float>();
                        outRecall = recall.masked_select(supportMask).mean().item<float>();
                        return f1.masked_select(supportMask).mean().item<float>();
                    }

                    outPrecision = outRecall = 0.0f;
                    return 0.0f;
                }
                catch (const std::exception& e) {
                    _logger->error("ClassificationValidator",
                        "Failed to compute precision/recall/F1: " + std::string(e.what()));
                    outPrecision = outRecall = 0.0f;
                    return 0.0f;
                }
            }

            float ClassificationValidator::computeAUCROC(
                const torch::Tensor& pred,
                const torch::Tensor& target)
            {
                try {
                    if (pred.dim() != 2) throw std::invalid_argument("Pred must be [N, num_classes]");
                    if (target.dim() != 1) throw std::invalid_argument("Target must be [N]");

                    int64_t N = pred.size(0);
                    int64_t C = pred.size(1);
                    if (N == 0) return 0.0f;

                    auto prob = torch::softmax(pred, 1).to(torch::kCPU, torch::kFloat32).contiguous();
                    auto labels = torch::nn::functional::one_hot(target.to(torch::kLong), C)
                        .to(torch::kFloat32).contiguous();

                    auto P = labels.sum(0);
                    auto N_ = static_cast<float>(N) - P;
                    auto validMask = (P > 0) & (N_ > 0);

                    if (validMask.sum().item<int64_t>() == 0) return 0.0f;

                    auto sortedIdx = prob.argsort(0, true);
                    auto sortedLabels = labels.gather(0, sortedIdx);

                    auto ranks = (static_cast<float>(N) - torch::arange(N, torch::kFloat32).unsqueeze(1));
                    auto rankSum = (ranks * sortedLabels).sum(0);

                    auto auc = (rankSum - P * (P + 1) / 2.0f) / (P * N_).clamp_min(1e-7f);

                    return auc.masked_select(validMask).mean().item<float>();
                }
                catch (const std::exception& e) {
                    _logger->error("ClassificationValidator", "AUC-ROC failed: " + std::string(e.what()));
                    return 0.0f;
                }
            }
        }
    }
}
