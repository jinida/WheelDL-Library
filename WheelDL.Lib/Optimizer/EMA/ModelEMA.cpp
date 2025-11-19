#include "pch.h"
#include "ModelEMA.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"

namespace WheelDL {
    namespace Optimizer {
        namespace EMA {

            ModelEMA::ModelEMA(const Model::BaseModel& model,
                             float maxDecay,
                             int decayRamp,
                             int updateCount)
                : _maxDecay(maxDecay)
                , _decayRamp(decayRamp)
                , _updates(updateCount)
                , _enabled(true)
            {
                if (maxDecay <= 0.0f || maxDecay >= 1.0f) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Max decay must be in range (0, 1), got: " + std::to_string(maxDecay)
                    );
                }

                if (decayRamp <= 0) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Decay ramp must be > 0, got: " + std::to_string(decayRamp)
                    );
                }

                // Clone the model's _model member (torch::nn::Sequential)
                // We need to use a workaround since LibTorch doesn't have direct clone()
                auto sourceModel = model.getModel();
                if (!sourceModel) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Source model is null or not initialized"
                    );
                }

                // Create a deep copy by cloning the module
                _emaModel = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(
                    sourceModel->clone()
                );

                if (!_emaModel) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Failed to clone model"
                    );
                }

                // Set EMA model to eval mode and disable gradients
                _emaModel->eval();
                for (auto& param : _emaModel->parameters()) {
                    param.set_requires_grad(false);
                }
            }

            void ModelEMA::update(const Model::BaseModel& model)
            {
                if (!_enabled) {
                    return;
                }

                torch::NoGradGuard no_grad;

                float decay = getDecay();

                // Get named parameters from both models
                auto sourceModel = model.getModel();
                if (!sourceModel) {
                    return;
                }

                auto currentParams = sourceModel->named_parameters();
                auto emaParams = _emaModel->named_parameters();

                // Update each EMA parameter
                for (auto& currentParam : currentParams) {
                    const std::string& name = currentParam.key();
                    const torch::Tensor& currentTensor = currentParam.value();

                    // Find corresponding EMA parameter
                    auto emaIter = std::find_if(emaParams.begin(), emaParams.end(),
                        [&name](const auto& p) { return p.key() == name; });

                    if (emaIter != emaParams.end()) {
                        torch::Tensor& emaTensor = emaIter->value();
                        updateParameter(emaTensor, currentTensor, decay);
                    }
                }

                // Copy buffers (e.g., running_mean, running_var, num_batches_tracked in BatchNorm)
                // Buffers are COPIED directly, NOT updated with EMA decay
                // This ensures BatchNorm statistics reflect current model state
                auto currentBuffers = sourceModel->named_buffers();
                auto emaBuffers = _emaModel->named_buffers();

                for (auto& currentBuffer : currentBuffers) {
                    const std::string& name = currentBuffer.key();
                    const torch::Tensor& currentTensor = currentBuffer.value();

                    auto emaIter = std::find_if(emaBuffers.begin(), emaBuffers.end(),
                        [&name](const auto& b) { return b.key() == name; });

                    if (emaIter != emaBuffers.end()) {
                        torch::Tensor& emaTensor = emaIter->value();
                        // Copy buffers directly without applying EMA
                        emaTensor.copy_(currentTensor);
                    }
                }

                _updates++;
            }

            float ModelEMA::getDecay() const
            {
                // Dynamic decay: max_decay * (1 - exp(-updates / decay_ramp))
                // This ramps up the decay over time
                if (_updates == 0) {
                    return 0.0f;
                }
                return _maxDecay * (1.0f - std::exp(-static_cast<float>(_updates) / _decayRamp));
            }

            void ModelEMA::setMaxDecay(float maxDecay)
            {
                if (maxDecay <= 0.0f || maxDecay >= 1.0f) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Max decay must be in range (0, 1), got: " + std::to_string(maxDecay)
                    );
                }
                _maxDecay = maxDecay;
            }

            void ModelEMA::setDecayRamp(int decayRamp)
            {
                if (decayRamp <= 0) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Decay ramp must be > 0, got: " + std::to_string(decayRamp)
                    );
                }
                _decayRamp = decayRamp;
            }

            void ModelEMA::updateParameter(torch::Tensor& emaParam,
                                          const torch::Tensor& currentParam,
                                          float decay)
            {
                // EMA formula: ema = decay * ema + (1 - decay) * current
                // Equivalent to: ema += (1 - decay) * (current - ema)
                // This is more numerically stable

                if (!emaParam.defined() || !currentParam.defined()) {
                    return;
                }

                // Ensure both tensors are on the same device
                if (emaParam.device() != currentParam.device()) {
                    return;  // Skip if devices don't match
                }

                // Ensure both tensors have the same shape
                if (emaParam.sizes() != currentParam.sizes()) {
                    return;  // Skip if shapes don't match
                }

                // Apply EMA update using .data() to avoid in-place operation errors
                emaParam.data().mul_(decay).add_(currentParam.data(), 1.0f - decay);
            }

        } // namespace EMA
    } // namespace Optimizer
} // namespace WheelDL
