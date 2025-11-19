#include "pch.h"
#include "BaseModel.h"
#include "../Loss/BaseLoss.h"
#include "../Loss/DetectionLoss.h"
#include "../Loss/ClassificationLoss.h"
#include "../Loss/SegmentationLoss.h"
#include "../Loss/OBBLoss.h"
#include "../Modules/Conv.h"
#include "../Modules/Block.h"
#include "../Modules/Head.h"
#include "../Builder/ModelBuilder.h"
#include "../../Config/YamlParser.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <torch/script.h>
#include <torch/serialize.h>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

namespace WheelDL {
	namespace Model {
		using namespace Modules;
		BaseModel::BaseModel()
			: _taskType(WheelDL::TaskType::UNKNOWN)
			, _isInitialized(false)
			, _inplace(true)
			, _training(false)
		{
			_stride = torch::tensor({ 32.0 });  // Default stride
		}

		std::vector<torch::Tensor> BaseModel::forward(const torch::Tensor& x) 
		{
			return predict(x);
		}

		std::unordered_map<std::string, torch::Tensor> BaseModel::forward(const Data::Dataset::DataExample& data) 
		{
			return loss(data);
		}

		std::vector<torch::Tensor> BaseModel::predict(const torch::Tensor& x)
		{
			if (_model->is_empty()) {
				return { x };
			}

			if (_saveIndices.empty() && _fromIndices.empty()) {
				return { _model->forward(x) };
			}

			// Build a set of indices that need to be saved
			std::unordered_set<size_t> indicesToSave;
			indicesToSave.insert(0);  // Always save input (Layer 0)

			// Add all saveIndices
			for (int64_t idx : _saveIndices) 
			{
				indicesToSave.insert(static_cast<size_t>(idx));
			}

			// Storage for layer outputs (sparse - only saved layers)
			std::unordered_map<size_t, torch::Tensor> layerOutputs;
			layerOutputs[0] = x;  // Layer 0 is the input image
			torch::Tensor previousOutput = x;  // Keep track of previous layer output

			// Lambda to resolve negative layer indices
			auto resolveLayerId = [&](int64_t layerId, size_t currentLayer) -> size_t {
				if (layerId < 0) {
					// -1 means previous layer, -2 means two layers back, etc.
					return currentLayer + layerId;
				}
				return static_cast<size_t>(layerId);
				};

			// Forward through all modules (backbone + head)
			for (size_t i = 0; i < _model->size(); ++i)
			{
				size_t currentLayerId = i + 1;  // Layer ID (1-based, 0 is input)
				torch::Tensor currentInput;

				bool isHead = (i == _model->size() - 1);
				// Determine input based on _fromIndices
				if (!_fromIndices.empty() && i < _fromIndices.size())
				{
					const auto& fromList = _fromIndices[i];
					if (fromList.empty() || (fromList.size() == 1 && fromList[0] == -1))
					{
						// Use output from previous layer
						currentInput = previousOutput;
					}
					else if (fromList.size() == 1)
					{
						// Single input from specific layer
						int64_t fromIdx = fromList[0];
						size_t resolvedIdx = resolveLayerId(fromIdx, currentLayerId);

						auto it = layerOutputs.find(resolvedIdx);
						// ModelBuilder validated this - should always succeed
						assert(it != layerOutputs.end() && it->second.defined());
						currentInput = it->second;
					}
					else
					{
						// Multiple inputs (for Concat or multi-input heads)
						std::vector<torch::Tensor> inputs;
						for (int64_t fromIdx : fromList)
						{
							size_t resolvedIdx = resolveLayerId(fromIdx, currentLayerId);

							// Special case: -1 (previous layer) always uses previousOutput
							if (fromIdx == -1)
							{
								inputs.push_back(previousOutput);
							}
							else
							{
								auto it = layerOutputs.find(resolvedIdx);
								// ModelBuilder validated this - should always succeed
								assert(it != layerOutputs.end() && it->second.defined());
								inputs.push_back(it->second);
							}
						}

						if (isHead)
						{
							// ModelBuilder validated last layer is IHeadBlockImpl
							return _model[i]->as<Modules::IHeadBlockImpl>()->forward(inputs);
						}
						else
						{
							// Backbone module with multiple inputs (e.g., Concat)
							auto module = _model[i];

							if (auto* concatModule = dynamic_cast<Modules::ConcatImpl*>(module.get()))
							{
								currentInput = concatModule->forward(inputs);
								previousOutput = currentInput;
								if (indicesToSave.find(currentLayerId) != indicesToSave.end())
								{
									layerOutputs[currentLayerId] = currentInput;
								}
								continue;
							}
							else
							{
								currentInput = inputs[0];
							}
						}
					}
				}
				else
				{
					currentInput = previousOutput;
				}

				if (isHead)
				{
					return _model[i]->as<Modules::IHeadBlockImpl>()->forward({ currentInput });
				}

				auto module = _model[i]->as<IBlockImpl>();
				torch::Tensor output = module->forward(currentInput);

				// Update previous output
				previousOutput = output;

				// Store output only if it's in saveIndices
				if (indicesToSave.find(currentLayerId) != indicesToSave.end())
				{
					layerOutputs[currentLayerId] = output;
				}
			}

			// Return last layer output
			return { previousOutput };
		}

		std::unordered_map<std::string, torch::Tensor> BaseModel::loss(const Data::Dataset::DataExample& batch, const std::vector<torch::Tensor>& preds)
		{
			if (!_criterion)
			{
				_criterion = initCriterion();
			}

			std::vector<torch::Tensor> predictions = preds.empty() ? forward(batch.data) : preds;
			return _criterion->compute(predictions, batch);
		}

		void BaseModel::loadWeights(const std::string& weights) {
			try {
				// Load model state dict from file
				torch::serialize::InputArchive archive;
				archive.load_from(weights);

				// Load tensors from archive
				std::unordered_map<std::string, torch::Tensor> stateDict;
				for (const auto& param : named_parameters()) 
				{
					const std::string& name = param.key();
					torch::Tensor tensor;

					// Try to read the tensor from archive
					try {
						archive.read(name, tensor);
						stateDict[name] = tensor;
					}
					catch (...) {
						std::cerr << "Warning: Parameter " << name << " not found in checkpoint." << std::endl;
						continue;
					}
				}

				// Apply loaded weights
				loadWeights(stateDict);

			}
			catch (const std::exception& e) {
				throw WheelDL::Utils::ModelException(
					WheelDL::Utils::ErrorCode::MODEL_LOAD_FAILED,
					"Failed to load weights from " + weights + ": " + e.what()
				);
			}
		}

		void BaseModel::loadWeights(const std::unordered_map<std::string, torch::Tensor>& stateDict) {
			// Get current model parameters
			for (auto& param : named_parameters()) {
				const std::string& name = param.key();
				torch::Tensor& tensor = param.value();

				auto it = stateDict.find(name);
				if (it != stateDict.end()) {
					// Check shape compatibility
					if (tensor.sizes() == it->second.sizes()) {
						tensor.data().copy_(it->second);
					}
				}
			}
		}

		void BaseModel::saveWeights(const std::string& path) const {
			try {
				// Save model state dict
				torch::serialize::OutputArchive archive;

				for (const auto& param : named_parameters()) {
					const std::string& name = param.key();
					const torch::Tensor& tensor = param.value();
					archive.write(name, tensor);
				}

				archive.save_to(path);

			}
			catch (const std::exception& e) {
				throw WheelDL::Utils::ModelException(
					WheelDL::Utils::ErrorCode::MODEL_SAVE_FAILED,
					"Failed to save weights to " + path + ": " + e.what()
				);
			}
		}

		void BaseModel::applyToTensors(const std::function<torch::Tensor(const torch::Tensor&)>& fn) {
			// Apply function to stride and other model tensors
			if (_stride.defined()) {
				_stride = fn(_stride);
			}

			// Apply to module parameters
			for (auto& param : parameters()) {
				param = fn(param);
			}
		}

		int64_t BaseModel::countParameters() const {
			int64_t count = 0;
			for (const auto& p : parameters()) 
			{
				count += p.numel();
			}
			return count;
		}

		double BaseModel::countFlops(const torch::IntArrayRef& inputSize) const {
			// This is a simplified FLOP counting
			// For more accurate counting, would need to traverse each layer type

			double flops = 0.0;
			int64_t batch = inputSize[0];
			int64_t channels = inputSize[1];
			int64_t height = inputSize[2];
			int64_t width = inputSize[3];

			for (const auto& module : modules()) {
				// Check if it's a Conv2d
				if (auto conv = dynamic_cast<const torch::nn::Conv2dImpl*>(module.get())) {
					auto options = conv->options;
					int64_t kernelOps = options.kernel_size()->at(0) * options.kernel_size()->at(1);
					int64_t inChannels = options.in_channels();
					int64_t outChannels = options.out_channels();

					// Approximate output size
					int64_t outH = height / options.stride()->at(0);
					int64_t outW = width / options.stride()->at(1);

					flops += batch * kernelOps * inChannels * outChannels * outH * outW;

					// Update for next layer
					height = outH;
					width = outW;
				}
				// Could add more layer types here (Linear, etc.)
			}

			// Multiply-accumulate operations count as 2 FLOPs
			return flops * 2.0;
		}

		void BaseModel::to(torch::Device device, torch::Dtype dtype) {
			// Move module (parameters and buffers) to device
			torch::nn::Module::to(device, dtype);

			// Also move loss function if it exists
			if (_criterion) {
				_criterion->to(device);
			}
			_model->to(device, dtype);
		}

		void BaseModel::to(torch::Device device) {
			torch::nn::Module::to(device);
			if (_criterion) {
				_criterion->to(device);
			}
			_model->to(device);
		}

	} // namespace Model
} // namespace WheelDL
