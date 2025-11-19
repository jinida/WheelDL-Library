#include "pch.h"
#include "LRScheduler.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <cmath>
#include <algorithm>

namespace WheelDL {
    namespace Optimizer {
        namespace Scheduler {

            LRScheduler::LRScheduler(torch::optim::Optimizer* optimizer,
                                     const Config::Configuration& config)
                : _optimizer(optimizer)
            {
                if (!_optimizer) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Optimizer pointer is null"
                    );
                }

                _initialLR = config.getLearningRate();
                _finalLR = _initialLR * config.getLRFinalFraction();
                _totalEpochs = config.getEpochs();
                _warmupEpochs = static_cast<int>(std::ceil(config.getWarmupEpochs()));

                // Validate settings
                if (_initialLR <= 0.0f) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Initial learning rate must be positive"
                    );
                }

                if (_totalEpochs <= 0) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Total epochs must be positive"
                    );
                }

                if (_warmupEpochs < 0 || _warmupEpochs > _totalEpochs) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Warmup epochs must be between 0 and total epochs"
                    );
                }
            }

            void LRScheduler::step(int epoch)
            {
                float lr;

                if (epoch < _warmupEpochs) {
                    // Warmup phase: linear ramp-up
                    lr = computeWarmupLR(epoch);
                }
                else {
                    // Main scheduling phase: delegate to derived class
                    lr = computeLR(epoch);
                }

                setLR(lr);
            }

            float LRScheduler::getCurrentLR() const
            {
                // Get LR from first parameter group
                auto& param_groups = _optimizer->param_groups();
                if (param_groups.empty()) {
                    return 0.0f;
                }

                // Access learning rate from param_groups based on optimizer type
                if (auto* sgd = dynamic_cast<torch::optim::SGD*>(_optimizer)) {
                    return static_cast<torch::optim::SGDOptions&>(param_groups[0].options()).lr();
                }
                else if (auto* adam = dynamic_cast<torch::optim::Adam*>(_optimizer)) {
                    return static_cast<torch::optim::AdamOptions&>(param_groups[0].options()).lr();
                }
                else if (auto* adamw = dynamic_cast<torch::optim::AdamW*>(_optimizer)) {
                    return static_cast<torch::optim::AdamWOptions&>(param_groups[0].options()).lr();
                }

                // Default: return initial LR
                return _initialLR;
            }

            void LRScheduler::setLR(float lr)
            {
                // Clamp learning rate to reasonable bounds
                lr = std::max(1e-8f, std::min(lr, 1.0f));

                // Set LR in all parameter groups
                auto& param_groups = _optimizer->param_groups();
                for (auto& group : param_groups) {
                    // Set learning rate based on optimizer type
                    if (auto* sgd = dynamic_cast<torch::optim::SGD*>(_optimizer)) {
                        static_cast<torch::optim::SGDOptions&>(group.options()).lr(lr);
                    }
                    else if (auto* adam = dynamic_cast<torch::optim::Adam*>(_optimizer)) {
                        static_cast<torch::optim::AdamOptions&>(group.options()).lr(lr);
                    }
                    else if (auto* adamw = dynamic_cast<torch::optim::AdamW*>(_optimizer)) {
                        static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr);
                    }
                }
            }

            float LRScheduler::computeWarmupLR(int epoch) const
            {
                if (_warmupEpochs == 0) {
                    return _initialLR;
                }

                // Linear ramp-up from 0 to _initialLR
                float progress = static_cast<float>(epoch + 1) / static_cast<float>(_warmupEpochs);
                return _initialLR * progress;
            }

        } // namespace Scheduler
    } // namespace Optimizer
} // namespace WheelDL
