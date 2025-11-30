#include "pch.h"
#include "AnomalyValidator.h"
#include "../../Data/Dataset/AnomalyDataset.h"
#include "../../Data/Transforms/Collation.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <algorithm>
#include <numeric>

namespace WheelDL {
	namespace Core {
		namespace Validator {

			AnomalyValidator::AnomalyValidator(const std::shared_ptr<Config::Configuration> config)
				: BaseValidator(config, torch::Device(torch::kCPU))
			{
				_logger->info("AnomalyValidator", "Anomaly detection validator initialized");
			}

			void AnomalyValidator::setupDataLoader()
			{
				if (_batchIterator) {
					_logger->info("AnomalyValidator",
						"Using injected DataLoader for validation");
					return;

				}

				auto valDataset = std::make_shared<Data::Dataset::AnomalyDataset>(*_config, false);
				_logger->info("AnomalyTrainer", "Validation dataset created with " +
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

			WheelDL::Data::Dataset::DataExample AnomalyValidator::preprocessBatch(
				const WheelDL::Data::Dataset::DataExample& batch)
			{
				WheelDL::Data::Dataset::DataExample preprocessed;
				preprocessed.data = batch.data.to(_device);
				if (batch.targets.defined())
				{
					preprocessed.targets = batch.targets.to(_device);
				}
				preprocessed.classes = batch.classes.to(_device);

				return preprocessed;
			}

			torch::Tensor AnomalyValidator::postprocessBatch(const std::vector<torch::Tensor>& prediction)
			{
				return _config->IsEfficientAD() ? prediction[5] : prediction[0];
			}

			MetricsData AnomalyValidator::computeMetrics(
				const torch::Tensor& pred,
				const torch::Tensor& target)
			{
				_profiler.start("compute_metrics");

				MetricsData metrics;
				metrics.loss = 0.0f;

				try
				{
					torch::Tensor anomalyScores = pred;
					torch::Tensor labels = target;

					if (labels.dim() > 1) {
						labels = labels.flatten();
					}

					if (anomalyScores.size(0) != labels.size(0))
					{
						_logger->warn("AnomalyValidator", "Mismatch between scores and labels size. " +
							std::string("Scores: ") + std::to_string(anomalyScores.size(0)) +
							", Labels: " + std::to_string(labels.size(0)));

						int minSize = std::min(anomalyScores.size(0), labels.size(0));
						anomalyScores = anomalyScores.slice(0, 0, minSize);
						labels = labels.slice(0, 0, minSize);
					}

					auto scoresCPU = anomalyScores.cpu().contiguous().to(torch::kFloat32);
					auto labelsCPU = labels.cpu().contiguous().to(torch::kFloat32);

					metrics.aucROC = computeAUCROC(scoresCPU, labelsCPU);
					metrics.mAP = computeAveragePrecision(scoresCPU, labelsCPU);

					// Compute all metrics at optimal F1 threshold
					float optimalThreshold = 0.5f;
					float precision = 0.0f;
					float recall = 0.0f;
					float accuracy = 0.0f;

					metrics.f1Score = computeOptimalMetrics(
						scoresCPU, labelsCPU,
						optimalThreshold, precision, recall, accuracy);

					metrics.threshold = optimalThreshold;
					metrics.precision = precision;
					metrics.recall = recall;
					metrics.accuracy = accuracy;

					_logger->info("AnomalyValidator", "Metrics - AUC-ROC: " + std::to_string(metrics.aucROC) +
						" | AP: " + std::to_string(metrics.mAP) +
						" | F1: " + std::to_string(metrics.f1Score) +
						" | Acc: " + std::to_string(metrics.accuracy));
				}
				catch (const std::exception& e)
				{
					_logger->error("AnomalyValidator", "Failed to compute metrics: " + std::string(e.what()));
					metrics.fitness = 0.0f;
					metrics.mAP = 0.0f;
					metrics.accuracy = 0.0f;
					metrics.f1Score = 0.0f;
				}

				_profiler.stop("compute_metrics");
				return metrics;
			}

			float AnomalyValidator::computeAUCROC(const torch::Tensor& scores,
				const torch::Tensor& labels)
			{
				try {
					int count = scores.size(0);
					if (count == 0) {
						return 0.0f;
					}
					struct ScoreItem { float score; int label; };
					std::vector<ScoreItem> items;
					items.reserve(count);
					const float* scoreData = scores.data_ptr<float>();
					const float* labelData = labels.data_ptr<float>();
					for (int i = 0; i < count; i++) {
						items.push_back({
							scoreData[i],
							(labelData[i] > 0.5f) ? 1 : 0
							});
					}
					std::sort(items.begin(), items.end(),
						[](const ScoreItem& a, const ScoreItem& b) {
							return a.score > b.score;
						});
					int numPositive = 0;
					int numNegative = 0;
					for (const auto& item : items) {
						if (item.label)
							numPositive++;
						else
							numNegative++;
					}
					if (numPositive == 0 || numNegative == 0) {
						return 0.5f;
					}
					float auc = 0.0f;
					int truePositive = 0;
					int falsePositive = 0;
					float prevTpr = 0.0f;
					float prevFpr = 0.0f;
					int i = 0;
					while (i < count) 
					{
						float currentScore = items[i].score;

						int tpInc = 0;
						int fpInc = 0;
						int j = i;

						while (j < count && items[j].score == currentScore) {
							if (items[j].label)
								tpInc++;
							else
								fpInc++;
							j++;
						}

						truePositive += tpInc;
						falsePositive += fpInc;

						float tpr = static_cast<float>(truePositive) / numPositive;
						float fpr = static_cast<float>(falsePositive) / numNegative;

						auc += (fpr - prevFpr) * (tpr + prevTpr) * 0.5f;

						prevTpr = tpr;
						prevFpr = fpr;

						i = j;
					}
					return auc;
				}
				catch (const std::exception& e) 
				{
					_logger->error("AnomalyValidator",
						"Failed to compute AUC-ROC: " + std::string(e.what()));
					return 0.0f;
				}
			}

			float AnomalyValidator::computeAveragePrecision(const torch::Tensor& scores,
				const torch::Tensor& labels)
			{
				try {
					int count = scores.size(0);
					if (count == 0) {
						return 0.0f;
					}

					const float* scoreData = scores.data_ptr<float>();
					const float* labelData = labels.data_ptr<float>();

					std::vector<int> indices(count);
					std::iota(indices.begin(), indices.end(), 0);

					std::sort(indices.begin(), indices.end(),
						[&](int a, int b) { return scoreData[a] > scoreData[b]; });

					int numPos = 0;
					for (int i = 0; i < count; i++) {
						if (labelData[i] > 0.5f) {
							numPos++;
						}
					}

					if (numPos == 0) {
						return 0.0f;
					}

					float ap = 0.0f;
					int truePos = 0;

					// AP = �� precision@k for all positive labels / numPos
					for (int i = 0; i < count; i++) {
						int idx = indices[i];

						if (labelData[idx] > 0.5f) {
							truePos++;
							float precision = static_cast<float>(truePos) / (i + 1);
							ap += precision;
						}
					}

					ap /= numPos;
					return ap;
				}
				catch (const std::exception& e) {
					_logger->error("AnomalyValidator",
						"Failed to compute Average Precision: " + std::string(e.what()));
					return 0.0f;
				}
			}

			float AnomalyValidator::computeOptimalMetrics(
				const torch::Tensor& scores,
				const torch::Tensor& labels,
				float& threshold,
				float& outPrecision,
				float& outRecall,
				float& outAccuracy)
			{
				try {
					int count = scores.size(0);
					if (count == 0) {
						threshold = 0.5f;
						outPrecision = 0.0f;
						outRecall = 0.0f;
						outAccuracy = 0.0f;
						return 0.0f;
					}

					const float* scoreData = scores.data_ptr<float>();
					const float* labelData = labels.data_ptr<float>();

					int totalPos = 0;
					int totalNeg = 0;
					for (int i = 0; i < count; i++) {
						if (labelData[i] > 0.5f) {
							totalPos++;
						} else {
							totalNeg++;
						}
					}

					if (totalPos == 0) {
						threshold = 0.5f;
						outPrecision = 0.0f;
						outRecall = 0.0f;
						outAccuracy = static_cast<float>(totalNeg) / count;
						return 0.0f;
					}

					std::vector<int> indices(count);
					std::iota(indices.begin(), indices.end(), 0);

					std::sort(indices.begin(), indices.end(),
						[&](int a, int b) { return scoreData[a] > scoreData[b]; });

					float bestF1 = 0.0f;
					float bestThreshold = 0.0f;
					float bestPrecision = 0.0f;
					float bestRecall = 0.0f;
					float bestAccuracy = 0.0f;

					int cumTP = 0;

					for (int i = 0; i < count; i++) {
						int idx = indices[i];

						if (labelData[idx] > 0.5f) {
							cumTP++;
						}

						int predictedPos = i + 1;  // samples with score >= scoreData[idx]
						int predictedNeg = count - predictedPos;
						int fp = predictedPos - cumTP;
						int tn = totalNeg - fp;

						float precision = static_cast<float>(cumTP) / predictedPos;
						float recall = static_cast<float>(cumTP) / totalPos;
						float accuracy = static_cast<float>(cumTP + tn) / count;

						float f1 = 0.0f;
						if (precision + recall > 0) {
							f1 = 2.0f * precision * recall / (precision + recall);
						}

						if (f1 > bestF1) {
							bestF1 = f1;
							bestThreshold = scoreData[idx];
							bestPrecision = precision;
							bestRecall = recall;
							bestAccuracy = accuracy;
						}
					}

					threshold = bestThreshold;
					outPrecision = bestPrecision;
					outRecall = bestRecall;
					outAccuracy = bestAccuracy;
					return bestF1;
				}
				catch (const std::exception& e) {
					_logger->error("AnomalyValidator",
						"Failed to compute optimal metrics: " + std::string(e.what()));
					threshold = 0.5f;
					outPrecision = 0.0f;
					outRecall = 0.0f;
					outAccuracy = 0.0f;
					return 0.0f;
				}
			}
		}
	}
}