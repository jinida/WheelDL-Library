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

namespace WheelDL {
    namespace Model {

        using namespace Modules;

        // ==================== BaseModel Implementation ====================

        BaseModel::BaseModel()
            : _taskType(WheelDL::TaskType::UNKNOWN)
            , _isInitialized(false)
            , _inplace(true)
            , _training(false)
        {
            _stride = torch::tensor({ 32.0 });  // Default stride
        }

        std::vector<torch::Tensor> BaseModel::forward(const torch::Tensor& x) {
            // Inference mode
            return predict(x);
        }

        std::unordered_map<std::string, torch::Tensor> BaseModel::forward(const Data::Dataset::DataExample& data) {
            // Training mode - compute loss
            return loss(data);
        }

        std::vector<torch::Tensor> BaseModel::predict(const torch::Tensor& x)
        {
            if (_model->is_empty()) {
                return {x};
            }

            if (_saveIndices.empty() && _fromIndices.empty())
            {
                // Simple forward pass - wrap in vector
                return {_model->forward(x)};
            }

            // Head is always the last module
            size_t headIdx = _model->size() - 1;

            // Storage for layer outputs (y in original)
            std::vector<torch::Tensor> layerOutputs(_model->size());
            torch::Tensor currentInput = x;

            // Forward through backbone (all modules except head)
            for (size_t i = 0; i < headIdx; ++i) {
                // Determine input based on _fromIndices
                if (!_fromIndices.empty() && i < _fromIndices.size()) 
                {
                    int64_t fromIdx = _fromIndices[i];
                    if (fromIdx != -1) 
                    {
                        if (fromIdx >= 0 && static_cast<size_t>(fromIdx) < layerOutputs.size())
                        {
                            currentInput = layerOutputs[fromIdx];
                        }
                        // Could also handle multiple inputs here for concat layers
                    }
                    // else: use currentInput from previous layer
                }

                // Forward through module
                auto module = _model[i]->as<torch::nn::AnyModule>();
                currentInput = module->forward(currentInput);

                // Save output if in save indices
                if (std::find(_saveIndices.begin(), _saveIndices.end(), i) != _saveIndices.end())
                {
                    layerOutputs[i] = currentInput;
                }
            }

            // Head inference
            if (headIdx < _model->size()) 
            {
                std::vector<torch::Tensor> headInputs;
                if (!_headInputIndices.empty()) {
                    for (int64_t idx : _headInputIndices) {
                        if (idx >= 0 && static_cast<size_t>(idx) < layerOutputs.size() &&
                            layerOutputs[idx].defined()) 
                        {
                            headInputs.push_back(layerOutputs[idx]);
                        }
                    }
                } 
                else 
                {
                    // Fallback: collect all saved outputs
                    for (size_t i = 0; i < layerOutputs.size(); ++i) 
                    {
                        if (layerOutputs[i].defined() &&
                            std::find(_saveIndices.begin(), _saveIndices.end(), i) != _saveIndices.end()) {
                            headInputs.push_back(layerOutputs[i]);
                        }
                    }
                }

                if (!headInputs.empty()) {
                    auto headModule = _model[headIdx];

                    if (auto* detectHead = dynamic_cast<Modules::DetectImpl*>(headModule.get()))
                    {
                        // DetectImpl returns vector<Tensor>
                        return detectHead->forward(headInputs);
                    }
                    else if (auto* obbHead = dynamic_cast<Modules::OBBImpl*>(headModule.get())) {
                        // OBBImpl also returns vector<Tensor>
                        return obbHead->forward(headInputs);
                    }
                    else if (auto* classifyHead = dynamic_cast<Modules::ClassifyImpl*>(headModule.get())) {
                        // Classify returns single tensor - wrap in vector
                        torch::Tensor output;
                        if (headInputs.size() > 1) {
                            output = classifyHead->forwardMulti(headInputs);
                        }
                        else {
                            output = classifyHead->forward(headInputs[0]);
                        }
                        return {output};
                    }
                    else {
                        // Unknown head type - try generic forward
                        auto anyModule = headModule->as<torch::nn::AnyModule>();
                        if (headInputs.size() > 1) {
                            return {headInputs.back()};
                        } else {
                            return {anyModule->forward(headInputs[0])};
                        }
                    }
                }
            }

            // Default: return current input wrapped in vector
            return {currentInput};
        }

        std::unordered_map<std::string, torch::Tensor> BaseModel::loss(
            const Data::Dataset::DataExample& batch,
            const std::vector<torch::Tensor>& preds)
        {
            if (!_criterion)
            {
                _criterion = initCriterion();
            }

            // Compute predictions if not provided
            std::vector<torch::Tensor> predictions = preds.empty() ? forward(batch.data) : preds;

            // Compute loss using the criterion
            // Loss functions need to be updated to accept vector<Tensor>
            // For now, pass the predictions vector to the loss
            return _criterion->compute(predictions, batch);
        }

        void BaseModel::loadWeights(const std::string& weights) {
            try {
                // Load model state dict from file
                torch::serialize::InputArchive archive;
                archive.load_from(weights);

                // Load tensors from archive
                std::unordered_map<std::string, torch::Tensor> stateDict;
                for (const auto& param : named_parameters()) {
                    const std::string& name = param.key();
                    torch::Tensor tensor;

                    // Try to read the tensor from archive
                    try {
                        archive.read(name, tensor);
                        stateDict[name] = tensor;
                    }
                    catch (...) {
                        // Parameter not found in checkpoint, skip
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
            for (const auto& p : parameters()) {
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

    } // namespace Model
} // namespace WheelDL