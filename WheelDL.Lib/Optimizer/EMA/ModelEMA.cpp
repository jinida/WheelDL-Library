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

                auto sourceModel = model.getModel();
                if (!sourceModel) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Source model is null or not initialized"
                    );
                }

                // Copy all parameters (detached and cloned)
                for (const auto& param : sourceModel->named_parameters()) {
                    _emaParameters[param.key()] = param.value().detach().clone();
                    _emaParameters[param.key()].set_requires_grad(false);
                }

                // Copy all buffers (detached and cloned)
                for (const auto& buffer : sourceModel->named_buffers()) {
                    _emaBuffers[buffer.key()] = buffer.value().detach().clone();
                }
            }

            void ModelEMA::update(const Model::BaseModel& model)
            {
                if (!_enabled) {
                    return;
                }

                torch::NoGradGuard no_grad;

                _updates++;
                float decay = getDecay();

                auto sourceModel = model.getModel();
                if (!sourceModel) {
                    return;
                }

                // Update EMA parameters (only floating point tensors)
                for (const auto& param : sourceModel->named_parameters()) {
                    const std::string& name = param.key();
                    const torch::Tensor& currentTensor = param.value();

                    if (!currentTensor.is_floating_point()) {
                        continue;
                    }

                    auto it = _emaParameters.find(name);
                    if (it != _emaParameters.end()) {
                        updateParameter(it->second, currentTensor, decay);
                    }
                }

                // Update EMA buffers (only floating point, e.g., BatchNorm running stats)
                for (const auto& buffer : sourceModel->named_buffers()) {
                    const std::string& name = buffer.key();
                    const torch::Tensor& currentTensor = buffer.value();

                    // Only apply EMA to floating point tensors
                    if (!currentTensor.is_floating_point()) 
                    {
                        continue;
                    }

                    auto it = _emaBuffers.find(name);
                    if (it != _emaBuffers.end()) {
                        updateParameter(it->second, currentTensor, decay);
                    }
                }
            }

            void ModelEMA::applyToModel(Model::BaseModel& model)
            {
                torch::NoGradGuard no_grad;

                auto targetModel = model.getModel();
                if (!targetModel) {
                    return;
                }

                // Backup current parameters
                _backupParameters.clear();
                for (const auto& param : targetModel->named_parameters()) {
                    _backupParameters[param.key()] = param.value().detach().clone();
                }

                // Backup current buffers
                _backupBuffers.clear();
                for (const auto& buffer : targetModel->named_buffers()) {
                    _backupBuffers[buffer.key()] = buffer.value().detach().clone();
                }

                // Apply EMA parameters to model
                for (auto& param : targetModel->named_parameters()) {
                    const std::string& name = param.key();
                    auto it = _emaParameters.find(name);
                    if (it != _emaParameters.end()) {
                        param.value().data().copy_(it->second);
                    }
                }

                // Apply EMA buffers to model
                for (auto& buffer : targetModel->named_buffers()) {
                    const std::string& name = buffer.key();
                    auto it = _emaBuffers.find(name);
                    if (it != _emaBuffers.end()) {
                        buffer.value().copy_(it->second);
                    }
                }
            }

            void ModelEMA::restoreOriginalParams(Model::BaseModel& model)
            {
                torch::NoGradGuard no_grad;

                auto targetModel = model.getModel();
                if (!targetModel) {
                    return;
                }

                // Restore original parameters
                for (auto& param : targetModel->named_parameters()) 
                {
                    const std::string& name = param.key();
                    auto it = _backupParameters.find(name);
                    if (it != _backupParameters.end()) {
                        param.value().data().copy_(it->second);
                    }
                }

                // Restore original buffers
                for (auto& buffer : targetModel->named_buffers()) 
                {
                    const std::string& name = buffer.key();
                    auto it = _backupBuffers.find(name);
                    if (it != _backupBuffers.end()) {
                        buffer.value().copy_(it->second);
                    }
                }

                // Clear backups
                _backupParameters.clear();
                _backupBuffers.clear();
            }

            float ModelEMA::getDecay() const
            {
                // Dynamic decay: max_decay * (1 - exp(-updates / decay_ramp))
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
                    // Move EMA param to current param's device
                    emaParam = emaParam.to(currentParam.device());
                }

                // Ensure both tensors have the same shape
                if (emaParam.sizes() != currentParam.sizes()) {
                    return;  // Skip if shapes don't match
                }

                // Apply EMA update
                emaParam.mul_(decay).add_(currentParam.detach(), 1.0f - decay);
            }

        } // namespace EMA
    } // namespace Optimizer
} // namespace WheelDL
