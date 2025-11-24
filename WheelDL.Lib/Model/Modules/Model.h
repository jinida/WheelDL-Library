#pragma once

#include <torch/torch.h>
#include "Interfaces.h"
#include <unordered_map>

namespace WheelDL {
	namespace Model {
		namespace Modules {

			/**
			 * @brief EfficientAD Model for Anomaly Detection
			 *
			 * Reference: EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies
			 * https://arxiv.org/abs/2303.14535
			 *
			 * Architecture:
			 * - Teacher Network (PDN): Pre-trained, frozen during training
			 * - Student Network (PDN): Learns to mimic teacher on normal samples
			 * - AutoEncoder: Reconstructs images for additional anomaly signal
			 *
			 * Training:
			 * - Teacher is frozen and automatically loaded from Assets folder
			 * - Student + AE are trained with:
			 *   - Hard loss: Focus on difficult samples (quantile-based)
			 *   - AE loss: Teacher vs AE reconstruction
			 *   - STAE loss: Student vs AE consistency
			 *
			 * Inference:
			 * - Computes anomaly maps from teacher-student and teacher-ae distances
			 * - Uses quantile normalization for stable scoring
			 */
			class EfficientADImpl : public IAnomalyModel, public IFeaturePreparable {
			public:
				/**
				 * @brief Construct EfficientAD model
				 *
				 * Automatically loads pre-trained teacher weights from Assets folder:
				 * - small=true: loads WheelDL.Lib/Assets/teacher_small.pth
				 * - small=false: loads WheelDL.Lib/Assets/teacher_medium.pth
				 *
				 * Teacher is automatically frozen (no gradient computation).
				 *
				 * @param outChannels Number of output channels for PDN networks (default: 384)
				 * @param small Use small PDN variant (default: true)
				 * @throws std::runtime_error if teacher weights file not found
				 */
				explicit EfficientADImpl(int64_t outChannels = 384, bool small = true);

				/**
				 * @brief Forward pass
				 *
				 * Training mode (expects vector with 2 tensors):
				 *   Input: [image, ae_image]
				 *   Output: [teacher_out, student_out, ae_teacher_out, ae_student_out, ae_out]
				 *
				 * Inference mode (expects vector with 1 tensor):
				 *   Input: [image]
				 *   Output: [map_st, map_ae] - Anomaly maps
				 *
				 * @param x Input tensor(s)
				 * @return std::vector<torch::Tensor> Output tensor(s)
				 */
				std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x) override;

				/**
				 * @brief Get model type identifier
				 * @return Model type string
				 */
				std::string getModelType() const override { return "EfficientAD"; }

				/**
				 * @brief Get output channels
				 * @return Number of output channels
				 */
				int64_t getOutputChannels() const override { return outChannels; }

				/**
				 * @brief Set teacher feature normalization parameters
				 *
				 * Computes mean and std of teacher features on normal training data.
				 * Must be called after loading teacher weights and before training.
				 *
				 * @param loader DataLoader with normal training samples
				 */
				template<typename DataLoader>
				void setFeatureParams(DataLoader& loader)
				{
					// Set to eval mode for feature extraction
					_teacher->eval();

					std::vector<torch::Tensor> meanOutputs;
					std::vector<torch::Tensor> meanDistances;

					// First pass: compute mean
					{
						torch::NoGradGuard noGrad;
						for (auto& batch : loader) {
							auto image = batch.data.data();
							auto features = _teacher->forward(image);
							meanOutputs.push_back(torch::mean(features, { 0, 2, 3 }));
						}
					}

					// IMPORTANT: Use set_() to update registered buffers in-place
					_teacherMean.set_(torch::mean(torch::stack(meanOutputs), 0)
						.unsqueeze(0).unsqueeze(-1).unsqueeze(-1));

					// Second pass: compute std
					{
						torch::NoGradGuard noGrad;
						for (auto& batch : loader)
						{
							auto image = batch.data.data();
							auto features = _teacher->forward(image);
							auto distances = torch::mean(torch::pow(features - _teacherMean, 2), { 0, 2, 3 });
							meanDistances.push_back(distances);
						}
					}

					auto channelVar = torch::mean(torch::stack(meanDistances), 0)
						.unsqueeze(0).unsqueeze(-1).unsqueeze(-1);
					_teacherStd.set_(torch::sqrt(channelVar + 1e-6f));
				}

				/**
				 * @brief Set quantile normalization parameters
				 *
				 * Computes quantiles of student and AE anomaly maps on normal training data.
				 * Must be called after training and before inference.
				 *
				 * @param loader DataLoader with normal training samples
				 */
				template<typename DataLoader>
				void setQuantiles(DataLoader& loader)
				{
					bool isTraining = this->is_training();
					if (isTraining)
					{
						this->eval();
					}

					{
						torch::NoGradGuard noGrad;
						std::vector<torch::Tensor> studentMaps;
						std::vector<torch::Tensor> aeMaps;

						for (auto& batch : loader)
						{
							auto image = batch.data.data();
							auto outputs = this->forward({ image });
							studentMaps.push_back(outputs[0].flatten());
							aeMaps.push_back(outputs[1].flatten());
						}
						auto allStudentMaps = torch::cat(studentMaps);
						auto allAeMaps = torch::cat(aeMaps);
						// IMPORTANT: Use set_() to update registered buffers in-place
						_qStStart.set_(torch::quantile(allStudentMaps, 0.9f));
						_qStEnd.set_(torch::quantile(allStudentMaps, 0.995f));
						_qAeStart.set_(torch::quantile(allAeMaps, 0.9f));
						_qAeEnd.set_(torch::quantile(allAeMaps, 0.995f));
					}

					if (isTraining)
					{
						this->train();
					}
				}

				/**
				 * @brief Load teacher network weights
				 *
				 * @param path Path to teacher weights file (.pt or .pth)
				 */
				void loadTeacherWeights(const std::string& path);

				/**
				 * @brief Freeze teacher network parameters
				 *
				 * Call this after loading teacher weights to ensure it's not trained.
				 */
				void freezeTeacher();

			private:
				/**
				 * @brief Build PDN (Patch Description Network)
				 *
				 * @param outCh Number of output channels
				 * @param small Use small variant (fewer channels)
				 * @return torch::nn::Sequential PDN network
				 */
				torch::nn::Sequential buildPDN(int64_t outCh, bool small);

				/**
				 * @brief Build AutoEncoder network
				 *
				 * @param outCh Number of output channels
				 * @return torch::nn::Sequential AutoEncoder network
				 */
				torch::nn::Sequential buildAutoEncoder(int64_t outCh);

				// Sub-networks
				torch::nn::Sequential _teacher = nullptr;   ///< Teacher network (frozen)
				torch::nn::Sequential _student = nullptr;   ///< Student network (trainable)
				torch::nn::Sequential _ae = nullptr;        ///< AutoEncoder (trainable)

				// Normalization buffers (saved in state_dict)
				torch::Tensor _teacherMean;  ///< Teacher feature mean [1, C, 1, 1]
				torch::Tensor _teacherStd;   ///< Teacher feature std [1, C, 1, 1]

				// Quantile buffers for inference normalization (saved in state_dict)
				torch::Tensor _qStStart;    ///< Student map quantile start (scalar)
				torch::Tensor _qStEnd;      ///< Student map quantile end (scalar)
				torch::Tensor _qAeStart;    ///< AE map quantile start (scalar)
				torch::Tensor _qAeEnd;      ///< AE map quantile end (scalar)
				// Public members
				int64_t outChannels;  ///< Number of output channels
			};

			TORCH_MODULE(EfficientAD);

			// ============================================================================
			// PatchCore Implementation
			// ============================================================================

			/**
			 * @brief PatchCore Model for Anomaly Detection
			 *
			 * Reference: Towards Total Recall in Industrial Anomaly Detection
			 * https://arxiv.org/abs/2106.08265
			 *
			 * Architecture:
			 * - Pretrained Feature Extractor (WideResNet50): Frozen, extracts multi-scale features
			 * - Memory Bank: Stores normal patch embeddings from training data
			 * - Coreset Subsampling: k-center greedy algorithm for memory efficiency
			 *
			 * Training:
			 * - No gradient computation (training-free)
			 * - Extract features from normal images
			 * - Build memory bank with coreset subsampling
			 * - Size-based memory management (target: ~150MB)
			 *
			 * Inference:
			 * - Extract features from test image
			 * - Nearest neighbor search in memory bank
			 * - Compute anomaly maps and scores
			 */
			class PatchCoreImpl : public IAnomalyModel, public IFeaturePreparable {
			public:
				/**
				 * @brief Construct PatchCore model
				 *
				 * @param backbonePath Path to pretrained feature extractor (TorchScript)
				 * @param numNeighbors Number of nearest neighbors for scoring (default: 9)
				 * @param maxMemoryBankPatches Maximum patches in memory bank for 150MB target (default: 25600)
				 */
				explicit PatchCoreImpl(
					int64_t numNeighbors = 9,
					int64_t maxMemoryBankPatches = 25600
				);

				/**
				 * @brief Forward pass
				 *
				 * Training mode:
				 *   Input: [image]
				 *   Output: [embedding] - Adds to memory bank
				 *
				 * Inference mode:
				 *   Input: [image]
				 *   Output: [anomaly_score, anomaly_map]
				 *
				 * @param x Input tensor(s)
				 * @return Output tensor(s)
				 */
				std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x) override;

				/**
				 * @brief Get model type identifier
				 * @return Model type string
				 */
				std::string getModelType() const override { return "PatchCore"; }

				/**
				 * @brief Get output channels (embedding dimension)
				 * @return Embedding dimension
				 */
				int64_t getOutputChannels() const override { return _embeddingDim; }

				/**
				 * @brief Perform coreset subsampling on memory bank
				 *
				 * Uses k-center greedy algorithm to reduce memory bank size.
				 * Target size is controlled by maxMemoryBankPatches.
				 */
				void subsampleMemoryBank();

				/**
				 * @brief Get current memory bank size
				 * @return Number of patches in memory bank
				 */
				int64_t getMemoryBankSize() const { return _memoryBank.size(0); }

				/**
				 * @brief Override to() to move JIT module
				 *
				 * JIT modules are not automatically moved by torch::nn::Module::to()
				 * because they cannot be registered with register_module().
				 */
				void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false) override;
				void to(torch::Device device, bool non_blocking = false) override;
				void to(torch::Dtype dtype, bool non_blocking = false) override;

				/**
				 * @brief Build memory bank from training data
				 *
				 * Extracts features from all training images and builds memory bank.
				 * Automatically performs coreset subsampling if size exceeds limit.
				 * Must be called before inference.
				 *
				 * @param loader DataLoader with normal training samples
				 */
				template<typename DataLoader>
				void buildMemoryBank(DataLoader& loader)
				{
					// Set to eval mode
					bool wasTraining = this->is_training();
					this->eval();

					std::vector<torch::Tensor> allEmbeddings;
					constexpr int64_t maxMemoryBytes = 5LL * 1024 * 1024 * 1024; // 5GB limit
					int64_t currentMemoryBytes = 0;
					int64_t batchCount = 0;

					{
						auto device = _memoryBank.device();
						torch::NoGradGuard noGrad;
						for (auto& batch : loader) {
							batchCount++;
							auto image = batch.data.data();
							image = image.to(device);
							// Extract features
							auto features = extractFeatures(image);
							auto reshapedEmbedding = reshapeEmbedding(features);

							allEmbeddings.push_back(reshapedEmbedding);

							// Calculate memory usage
							currentMemoryBytes += reshapedEmbedding.numel() * reshapedEmbedding.element_size();

							// Check if we've exceeded memory limit
							if (currentMemoryBytes > maxMemoryBytes) 
							{
								// Concatenate accumulated embeddings
								auto tempMemoryBank = torch::cat(allEmbeddings, 0);

								// random sampling half of the data to reduce memory
								auto numToSelect = tempMemoryBank.size(0) / 2;
								auto randIndices = torch::randperm(tempMemoryBank.size(0)).slice(0, 0, numToSelect).to(torch::kLong);
								tempMemoryBank = tempMemoryBank.index_select(0, randIndices);

								// Clear and restart with subsampled data
								allEmbeddings.clear();
								allEmbeddings.push_back(tempMemoryBank);
								currentMemoryBytes = tempMemoryBank.numel() * tempMemoryBank.element_size();
							}
						}
					}

					// Concatenate all embeddings
					if (allEmbeddings.empty()) {
						// No embeddings collected - loader was empty or had no data
						// Keep memory bank as empty [0, _embeddingDim]
					} else {
						// IMPORTANT: Use set_() to update the registered buffer in-place
						// Do NOT reassign with = as it breaks buffer registration
						_memoryBank.set_(torch::cat(allEmbeddings, 0));

						// Final subsample if exceeds limit (using k-center greedy)
						if (_memoryBank.size(0) > _maxMemoryBankPatches)
						{
							subsampleMemoryBank();
						}
					}

					// Restore training mode
					if (wasTraining) {
						this->train();
					}
				}

			private:
				/**
				 * @brief Initialize Gaussian blur convolution
				 */
				void initializeGaussianBlur();
				
				/**
				 * @brief Extract features using backbone
				 *
				 * @param x Input images [B, C, H, W]
				 * @return Feature tensor [B, C, H, W]
				 */
				torch::Tensor extractFeatures(const torch::Tensor& x);

				/**
				 * @brief Reshape embedding for patch-wise processing
				 *
				 * @param embedding [B, C, H, W]
				 * @return Reshaped [B*H*W, C]
				 */
				static torch::Tensor reshapeEmbedding(const torch::Tensor& embedding);

				/**
				 * @brief Compute Euclidean distance between two sets of vectors
				 *
				 * @param x [N, D]
				 * @param y [M, D]
				 * @return Distance matrix [N, M]
				 */
				static torch::Tensor euclideanDist(const torch::Tensor& x, const torch::Tensor& y);

				/**
				 * @brief Find nearest neighbors in memory bank
				 *
				 * @param embedding Query embeddings [N, D]
				 * @param nNeighbors Number of neighbors
				 * @return Tuple of (distances, indices)
				 */
				std::tuple<torch::Tensor, torch::Tensor> nearestNeighbors(
					const torch::Tensor& embedding,
					int64_t nNeighbors
				);

				/**
				 * @brief Compute image-level anomaly score
				 *
				 * @param patchScores Patch-level scores [B, N]
				 * @param locations Nearest neighbor indices [B, N]
				 * @param embedding Query embeddings [B*N, D]
				 * @return Image-level scores [B]
				 */
				torch::Tensor computeAnomalyScore(
					const torch::Tensor& patchScores,
					const torch::Tensor& locations,
					const torch::Tensor& embedding
				);

				/**
				 * @brief Generate anomaly map with Gaussian blur
				 *
				 * @param patchScores Patch scores [B, 1, H, W]
				 * @param imageSize Target output size
				 * @return Anomaly map [B, 1, H, W]
				 */
				torch::Tensor generateAnomalyMap(
					const torch::Tensor& patchScores,
					const std::vector<int64_t>& imageSize
				);

				// Model components
				torch::jit::script::Module _featureExtractor;  ///< Pretrained backbone (frozen)
				torch::nn::AvgPool2d _featurePooler = nullptr; ///< Feature pooling layer
				torch::nn::Conv2d _gaussianBlur = nullptr;   ///< Gaussian blur convolution

				// Configuration
				int64_t _numNeighbors;                        ///< Number of nearest neighbors
				int64_t _maxMemoryBankPatches;                ///< Maximum patches in memory bank
				int64_t _embeddingDim;                        ///< Embedding dimension

				// Memory bank (registered as buffer for state_dict)
				torch::Tensor _memoryBank;                    ///< Memory bank [N, D]
			};

			TORCH_MODULE(PatchCore);

		} // namespace Modules
	} // namespace Model
} // namespace WheelDL
