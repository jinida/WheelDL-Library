#include "pch.h"
#include "Model.h"
#include <stdexcept>
#include <filesystem>
#include <torch/script.h>
#include <iostream>

namespace WheelDL {
	namespace Model {
		namespace Modules {

			// ============================================================================
			// EfficientADImpl Implementation
			// ============================================================================

			EfficientADImpl::EfficientADImpl(int64_t outChannels, bool small)
				: outChannels(outChannels)
			{
				// Build networks
				_teacher = buildPDN(outChannels, small);
				_student = buildPDN(outChannels * 2, small);  // Student outputs 2x channels
				_ae = buildAutoEncoder(outChannels);

				// Register networks
				register_module("teacher", _teacher);
				register_module("student", _student);
				register_module("ae", _ae);

				// Initialize buffers (will be computed later via setFeatureParams)
				_teacherMean = register_buffer("teacher_mean", torch::zeros({ 1, outChannels, 1, 1 }));
				_teacherStd = register_buffer("teacher_std", torch::ones({ 1, outChannels, 1, 1 }));

				// Initialize quantile buffers (will be set later via setQuantiles)
				_qStStart = register_buffer("q_st_start", torch::tensor(0.0f));
				_qStEnd = register_buffer("q_st_end", torch::tensor(0.0f));
				_qAeStart = register_buffer("q_ae_start", torch::tensor(0.0f));
				_qAeEnd = register_buffer("q_ae_end", torch::tensor(0.0f));

				// Auto-load teacher weights from Assets folder
				std::string teacherPath = small ?
					"teacher_small.pt" :
					"teacher_medium.pt";

				// Try to load teacher weights, but don't fail if not found
				try 
				{
					loadTeacherWeights(teacherPath);
				}
				catch (const std::exception& e) 
				{
					freezeTeacher();
				}
			}

			torch::nn::Sequential EfficientADImpl::buildPDN(int64_t outCh, bool small)
			{
				torch::nn::Sequential net;

				if (small) {
					// Small PDN variant
					net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 128, 4)));
					net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
					net->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)));

					net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 4)));
					net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
					net->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)));

					net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3)));
					net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

					net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, outCh, 4)));
				}
				else {
					// Large PDN variant
					net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 256, 4)));
					net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
					net->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)));

					net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 4)));
					net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
					net->push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)));

					net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 1)));
					net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

					net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3)));
					net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

					net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(512, outCh, 4)));
					net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

					net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(outCh, outCh, 1)));
				}

				return net;
			}

			torch::nn::Sequential EfficientADImpl::buildAutoEncoder(int64_t outCh)
			{
				torch::nn::Sequential net;

				// Encoder - 5x stride=2 downsampling (32x reduction)
				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 4).stride(2).padding(1)));
				net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 4).stride(2).padding(1)));
				net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 4).stride(2).padding(1)));
				net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 4).stride(2).padding(1)));
				net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 4).stride(2).padding(1)));
				net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

				// Bottleneck - reduces to 1x1
				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 8)));

				// Decoder - 5x scale_factor=2 upsampling
				net->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions()
					.size(std::vector<int64_t>{3, 3})
					.mode(torch::kBilinear).align_corners(false)));
				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 4).stride(1).padding(2)));
				net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
				net->push_back(torch::nn::Dropout(0.2));

				net->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions()
					.size(std::vector<int64_t>{8, 8})
					.mode(torch::kBilinear).align_corners(false)));
				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 4).stride(1).padding(2)));
				net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
				net->push_back(torch::nn::Dropout(0.2));

				net->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions()
					.size(std::vector<int64_t>{15, 15})
					.mode(torch::kBilinear).align_corners(false)));
				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 4).stride(1).padding(2)));
				net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
				net->push_back(torch::nn::Dropout(0.2));

				net->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions()
					.size(std::vector<int64_t>{32, 32})
					.mode(torch::kBilinear).align_corners(false)));
				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 4).stride(1).padding(2)));
				net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
				net->push_back(torch::nn::Dropout(0.2));

				net->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions()
					.size(std::vector<int64_t>{63, 63})
					.mode(torch::kBilinear).align_corners(false)));
				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 4).stride(1).padding(2)));
				net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
				net->push_back(torch::nn::Dropout(0.2));

				net->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions()
					.size(std::vector<int64_t>{127, 127})
					.mode(torch::kBilinear).align_corners(false)));
				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 4).stride(1).padding(2)));
				net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
				net->push_back(torch::nn::Dropout(0.2));

				net->push_back(torch::nn::Upsample(torch::nn::UpsampleOptions()
					.size(std::vector<int64_t>{56, 56})
					.mode(torch::kBilinear).align_corners(false)));
				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)));
				net->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
				net->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(64, outCh, 1).stride(1).padding(1)));

				return net;
			}

			std::vector<torch::Tensor> EfficientADImpl::forward(std::vector<torch::Tensor> x)
			{
				if (x.empty()) {
					throw std::invalid_argument("EfficientADImpl::forward - input vector is empty");
				}

				auto inputShape = x[0].sizes().vec();
				
				if (inputShape[1] == 6)
				{
					auto splitResult = torch::split(x[0], 3, 1);
					auto& input = splitResult[0];
					auto& aeInput = splitResult[1];

					// Teacher forward (no gradient)
					torch::Tensor teacherOut, aeTeacherOut;
					{
						torch::NoGradGuard noGrad;
						teacherOut = (_teacher->forward(input) - _teacherMean) / _teacherStd;
						aeTeacherOut = (_teacher->forward(aeInput) - _teacherMean) / _teacherStd;
					}

					// Student forward
					auto studentOutFull = _student->forward(input);
					auto studentOut = studentOutFull.narrow(1, 0, outChannels);  // First half

					// AE forward
					auto aeStudentOutFull = _student->forward(aeInput);
					auto aeStudentOut = aeStudentOutFull.narrow(1, outChannels, outChannels);  // Second half

					auto aeOut = _ae->forward(aeInput);

					auto targetSize = teacherOut.sizes().slice(2);
					aeOut = torch::nn::functional::interpolate(aeOut,
						torch::nn::functional::InterpolateFuncOptions()
						.size(std::vector<int64_t>{targetSize[0], targetSize[1]})
						.mode(torch::kBilinear)
						.align_corners(false));

					// Return all outputs for loss computation
					return { teacherOut, studentOut, aeTeacherOut, aeStudentOut, aeOut };
				}
				else 
				{
					// Inference mode: expects [image]
					auto& input = x[0];

					torch::Tensor teacherOut, studentOut, aeOut;
					{
						torch::NoGradGuard noGrad;
						teacherOut = (_teacher->forward(input) - _teacherMean) / _teacherStd;

						auto studentOutFull = _student->forward(input);
						studentOut = studentOutFull.narrow(1, 0, outChannels);  // First half

						aeOut = _ae->forward(input);
						auto targetSize = teacherOut.sizes().slice(2);
						aeOut = torch::nn::functional::interpolate(aeOut,
							torch::nn::functional::InterpolateFuncOptions()
							.size(std::vector<int64_t>{targetSize[0], targetSize[1]})
							.mode(torch::kBilinear)
							.align_corners(false));
					}

					// Compute anomaly maps
					auto mapSt = torch::mean(torch::pow(teacherOut - studentOut, 2), 1, true);
					auto mapAe = torch::mean(torch::pow(teacherOut - aeOut, 2), 1, true);

					mapSt = (mapSt - _qStStart) / (_qStEnd - _qStStart + 1e-6f);
					mapAe = (mapAe - _qAeStart) / (_qAeEnd - _qAeStart + 1e-6f);

					auto anomalyMap = 0.5f * (mapSt + mapAe);
					anomalyMap = torch::nn::functional::pad(anomalyMap,
						torch::nn::functional::PadFuncOptions({ 4, 4, 4, 4 }).mode(torch::kReflect));
					anomalyMap = torch::nn::functional::interpolate(anomalyMap,
						torch::nn::functional::InterpolateFuncOptions()
						.size(std::vector<int64_t>{ input.size(2), input.size(3) })
						.mode(torch::kBilinear)
						.align_corners(false));

					auto predScore = torch::max(anomalyMap);

					return { predScore, anomalyMap };
				}
			}

			void EfficientADImpl::loadTeacherWeights(const std::string& path)
			{
				// Check if file exists
				if (!std::filesystem::exists(path))
				{
					throw std::runtime_error("Teacher weights file not found: " + path);
				}

				try {
					// Load as state_dict
					torch::load(_teacher, path);
					_teacher->to(torch::kFloat32);
					_teacher->to(torch::kCPU);
					
					// Freeze teacher immediately after loading
					freezeTeacher();
				}
				catch (const std::exception& e) {
					throw std::runtime_error("Failed to load teacher weights from " + path + ": " + e.what());
				}
			}

			void EfficientADImpl::freezeTeacher()
			{
				// Freeze all teacher parameters (no gradient computation)
				for (auto& param : _teacher->parameters()) {
					param.set_requires_grad(false);
				}

				// Set to eval mode (disable dropout, batchnorm updates, etc.)
				_teacher->eval();
			}

		// ============================================================================
		// PatchCoreImpl Implementation
		// ============================================================================

		PatchCoreImpl::PatchCoreImpl(
			int64_t numNeighbors,
			int64_t maxMemoryBankPatches
		)
			: _numNeighbors(numNeighbors)
			, _maxMemoryBankPatches(maxMemoryBankPatches)
			, _embeddingDim(1536)
		{
			_featureExtractor = torch::jit::load("wide_resnet50_2.pt");

			_featurePooler = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(3).stride(1).padding(1));
			register_module("feature_pooler", _featurePooler);

			_memoryBank = register_buffer("memory_bank", torch::empty({0, _embeddingDim}));
			initializeGaussianBlur();
		}

		void PatchCoreImpl::initializeGaussianBlur() 
		{
			const int64_t kernelSize = 33;
			const double sigma = 4.0;
			const int64_t padding = (kernelSize - 1) / 2;
			
			_gaussianBlur = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, kernelSize)
				.padding(padding)
				.bias(false));

			register_module("gaussian_blur", _gaussianBlur);
			
			double ksizeHalf = (kernelSize - 1) * 0.5;
			auto x = torch::linspace(-ksizeHalf, ksizeHalf, kernelSize);
			auto pdf = torch::exp(-0.5 * (x / sigma).pow(2));
			auto kernel1d = pdf / pdf.sum();
			auto kernel2d = torch::mm(kernel1d.unsqueeze(1), kernel1d.unsqueeze(0));

			kernel2d = kernel2d.unsqueeze(0).unsqueeze(0);
			_gaussianBlur->weight.data().copy_(kernel2d);
			_gaussianBlur->weight.requires_grad_(false);
		}
		

		torch::Tensor PatchCoreImpl::extractFeatures(const torch::Tensor& x)
		{
			std::vector<torch::jit::IValue> inputs;
			inputs.push_back(x);

			torch::Tensor features;
			{
				torch::NoGradGuard noGrad;
				auto output = _featureExtractor.forward(inputs);

				if (output.isList())
				{
					auto featureTuple = output.toTensorList();//.toList();
					auto layer2 = featureTuple.get(0);
					auto layer3 = featureTuple.get(1);

					layer2 = _featurePooler->forward(layer2);
					layer3 = _featurePooler->forward(layer3);

					layer3 = torch::nn::functional::interpolate(layer3,
						torch::nn::functional::InterpolateFuncOptions()
						.size(std::vector<int64_t>{layer2.size(2), layer2.size(3)})
						.mode(torch::kBilinear).align_corners(false));

					features = torch::cat({ layer2, layer3 }, 1);
				}
				else {
					throw std::runtime_error("PatchCore backbone must return tuple of features");
				}
			}

			return features;
		}

		std::vector<torch::Tensor> PatchCoreImpl::forward(std::vector<torch::Tensor> x)
		{
			if (x.empty()) {
				throw std::invalid_argument("PatchCoreImpl::forward - input vector is empty");
			}

			if (_memoryBank.size(0) == 0) {
				throw std::runtime_error("Memory bank is empty");
			}

			auto input = x[0];
			auto outputSize = input.sizes().slice(2);

			auto features = extractFeatures(input);

			int64_t batchSize = features.size(0);
			int64_t height = features.size(2);
			int64_t width = features.size(3);

			auto embedding = reshapeEmbedding(features);

			auto [patchScores, locations] = nearestNeighbors(embedding, 1);

			patchScores = patchScores.reshape({ batchSize, -1 });
			locations = locations.reshape({ batchSize, -1 });

			auto predScore = computeAnomalyScore(patchScores, locations, embedding);

			patchScores = patchScores.reshape({ batchSize, 1, height, width });

			auto anomalyMap = generateAnomalyMap(patchScores,
				std::vector<int64_t>{outputSize[0], outputSize[1]});

			return { predScore, anomalyMap };
		}

		torch::Tensor PatchCoreImpl::reshapeEmbedding(const torch::Tensor& embedding)
		{
			int64_t embeddingSize = embedding.size(1);
			return embedding.permute({ 0, 2, 3, 1 }).reshape({ -1, embeddingSize });
		}

		torch::Tensor PatchCoreImpl::euclideanDist(const torch::Tensor& x, const torch::Tensor& y)
		{
			auto xNorm = x.pow(2).sum(-1, true);
			auto yNorm = y.pow(2).sum(-1, true);
			
			auto res = xNorm - 2 * torch::mm(x, y.t()) + yNorm.t();
			return res.clamp_min_(1e-12).sqrt_();
		}


		std::tuple<torch::Tensor, torch::Tensor> PatchCoreImpl::nearestNeighbors(
			const torch::Tensor& embedding,
			int64_t nNeighbors
		)
		{
			auto distances = euclideanDist(embedding, _memoryBank);

			if (nNeighbors == 1) {
				auto [patchScores, locations] = distances.min(1);
				return { patchScores, locations };
			}
			else {
				return distances.topk(nNeighbors, 1, false);
			}
		}

		torch::Tensor PatchCoreImpl::computeAnomalyScore(
			const torch::Tensor& patchScores,
			const torch::Tensor& locations,
			const torch::Tensor& embedding
		)
		{
			return patchScores.amax(1);
		}

		torch::Tensor PatchCoreImpl::generateAnomalyMap(
			const torch::Tensor& patchScores,
			const std::vector<int64_t>& imageSize
		)
		{
			auto anomalyMap = torch::nn::functional::interpolate(patchScores,
				torch::nn::functional::InterpolateFuncOptions()
				.size(imageSize)
				.mode(torch::kBilinear)
				.align_corners(false));

			return _gaussianBlur->forward(anomalyMap);
		}

		void PatchCoreImpl::subsampleMemoryBank() {
			const int64_t n = _memoryBank.size(0);
			const int64_t k = _maxMemoryBankPatches;

			if (n == 0) throw std::runtime_error("Memory bank is empty");
			if (n <= k) return;

			if (!_memoryBank.is_contiguous()) {
				_memoryBank = _memoryBank.contiguous();
			}

			const auto device = _memoryBank.device();
			const auto dtype = _memoryBank.dtype();

			auto selfNorms = (_memoryBank * _memoryBank).sum(1);

			auto selectedIdxs = torch::empty({ k }, torch::dtype(torch::kLong).device(device));

			auto firstIdxTensor = torch::randint(n, { 1 }, torch::dtype(torch::kLong).device(device));
			int64_t firstIdx = firstIdxTensor.item<int64_t>();

			selectedIdxs[0] = firstIdx;

			auto selectedVec = _memoryBank.index({ firstIdx });
			auto dotProduct = torch::matmul(_memoryBank, selectedVec);

			auto minDistSq = selfNorms + selfNorms[firstIdx] - 2.0 * dotProduct;
			minDistSq.clamp_min_(0.0);

			for (int64_t i = 1; i < k; ++i) {
				int64_t maxIdx = minDistSq.argmax().item<int64_t>();
				selectedIdxs[i] = maxIdx;

				auto newSelectedVec = _memoryBank.index({ maxIdx });
				auto newDot = torch::matmul(_memoryBank, newSelectedVec);

				auto newDist = selfNorms + selfNorms[maxIdx] - 2.0 * newDot;
				newDist.clamp_min_(0.0);

				minDistSq = torch::minimum(minDistSq, newDist);
			}

			// IMPORTANT: Use set_() to update the registered buffer in-place
			// Do NOT reassign with = as it breaks buffer registration
			_memoryBank.set_(_memoryBank.index_select(0, selectedIdxs));
		}

		void PatchCoreImpl::to(torch::Device device, torch::Dtype dtype, bool non_blocking)
		{
			// Move registered modules and buffers
			torch::nn::Module::to(device, dtype, non_blocking);
			// Manually move JIT module (not registered with register_module)
			_featureExtractor.to(device, dtype, non_blocking);
		}

		void PatchCoreImpl::to(torch::Device device, bool non_blocking)
		{
			// Move registered modules and buffers
			torch::nn::Module::to(device, non_blocking);
			// Manually move JIT module (not registered with register_module)
			_featureExtractor.to(device, non_blocking);
		}

		void PatchCoreImpl::to(torch::Dtype dtype, bool non_blocking)
		{
			// Move registered modules and buffers
			torch::nn::Module::to(dtype, non_blocking);
			_featureExtractor.to(dtype, non_blocking);
		}
	} // namespace Modules
} // namespace Model
} // namespace WheelDL
