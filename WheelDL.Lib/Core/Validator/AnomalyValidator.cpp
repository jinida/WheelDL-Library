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
				return prediction[0];
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

					metrics.fitness = computeAUCROC(scoresCPU, labelsCPU);
					metrics.mAP = computeAveragePrecision(scoresCPU, labelsCPU);

					float optimalThreshold = 0.5f;
					metrics.f1Score = computeF1Score(scoresCPU, labelsCPU, optimalThreshold);

					torch::Tensor predictions = (scoresCPU > optimalThreshold).to(torch::kFloat32);
					torch::Tensor correct = (predictions == labelsCPU).to(torch::kFloat32);
					metrics.accuracy = correct.mean().item<float>();

					_logger->info("AnomalyValidator", "Metrics - AUC-ROC: " + std::to_string(metrics.fitness) +
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

			float AnomalyValidator::computeAUCROC(const torch::Tensor& scores, const torch::Tensor& labels)
			{
				try {
					int n = scores.size(0);
					if (n == 0) {
						return 0.0f;
					}

					const float* scoresData = scores.data_ptr<float>();
					const float* labelsData = labels.data_ptr<float>();

					std::vector<int> indices(n);
					std::iota(indices.begin(), indices.end(), 0);

					std::sort(indices.begin(), indices.end(), [&](int i, int j) {
						return scoresData[i] > scoresData[j];
						});

					int numPos = 0;
					int numNeg = 0;
					for (int i = 0; i < n; ++i) {
						if (labelsData[i] > 0.5f) {
							numPos++;
						}
						else {
							numNeg++;
						}
					}

					if (numPos == 0 || numNeg == 0) {
						return 0.5f;
					}

					float auc = 0.0f;
					int truePos = 0;
					int falsePos = 0;
					float prevTPR = 0.0f;
					float prevFPR = 0.0f;

					for (int i = 0; i < n; ++i) {
						int idx = indices[i];
						if (labelsData[idx] > 0.5f) {
							truePos++;
						}
						else {
							falsePos++;
						}

						float tpr = static_cast<float>(truePos) / numPos;
						float fpr = static_cast<float>(falsePos) / numNeg;

						auc += (fpr - prevFPR) * (tpr + prevTPR) / 2.0f;

						prevTPR = tpr;
						prevFPR = fpr;
					}

					return auc;
				}
				catch (const std::exception& e) {
					_logger->error("AnomalyValidator", "Failed to compute AUC-ROC: " + std::string(e.what()));
					return 0.0f;
				}
			}

			float AnomalyValidator::computeAveragePrecision(const torch::Tensor& scores, const torch::Tensor& labels)
			{
				try {
					int n = scores.size(0);
					if (n == 0) {
						return 0.0f;
					}

					const float* scoresData = scores.data_ptr<float>();
					const float* labelsData = labels.data_ptr<float>();

					std::vector<int> indices(n);
					std::iota(indices.begin(), indices.end(), 0);

					std::sort(indices.begin(), indices.end(), [&](int i, int j) {
						return scoresData[i] > scoresData[j];
						});

					int numPos = 0;
					for (int i = 0; i < n; ++i) {
						if (labelsData[i] > 0.5f) {
							numPos++;
						}
					}

					if (numPos == 0) {
						return 0.0f;
					}

					float ap = 0.0f;
					int truePos = 0;

					for (int i = 0; i < n; ++i) {
						int idx = indices[i];
						if (labelsData[idx] > 0.5f) {
							truePos++;
							float precision = static_cast<float>(truePos) / (i + 1);
							ap += precision;
						}
					}

					ap /= numPos;
					return ap;
				}
				catch (const std::exception& e) {
					_logger->error("AnomalyValidator", "Failed to compute Average Precision: " + std::string(e.what()));
					return 0.0f;
				}
			}

			float AnomalyValidator::computeF1Score(const torch::Tensor& scores, const torch::Tensor& labels, float& threshold)
			{
				try {
					int n = scores.size(0);
					if (n == 0)
					{
						threshold = 0.5f;
						return 0.0f;
					}

					const float* scoresData = scores.data_ptr<float>();
					const float* labelsData = labels.data_ptr<float>();

					int totalPos = 0;
					for (int i = 0; i < n; ++i) {
						if (labelsData[i] > 0.5f) totalPos++;
					}

					if (totalPos == 0) {
						threshold = 0.5f;
						return 0.0f;
					}

					std::vector<int> indices(n);
					std::iota(indices.begin(), indices.end(), 0);

					std::sort(indices.begin(), indices.end(), [&](int i, int j) {
						return scoresData[i] > scoresData[j];
						});

					float bestF1 = 0.0f;
					float bestThreshold = 0.0f;
					int cumTP = 0;

					for (int i = 0; i < n; ++i) {
						int idx = indices[i];

						if (labelsData[idx] > 0.5f) {
							cumTP++;
						}

						int predictedPos = i + 1;
						float precision = static_cast<float>(cumTP) / predictedPos;
						float recall = static_cast<float>(cumTP) / totalPos;

						float f1 = 0.0f;
						if (precision + recall > 0) {
							f1 = 2.0f * precision * recall / (precision + recall);
						}

						if (f1 > bestF1) {
							bestF1 = f1;
							bestThreshold = scoresData[idx];
						}
					}

					threshold = bestThreshold;
					return bestF1;
				}
				catch (const std::exception& e)
				{
					_logger->error("AnomalyValidator", "Failed to compute F1 Score: " + std::string(e.what()));
					threshold = 0.5f;
					return 0.0f;
				}
			}

		}
	}
}