#include "pch.h"
#include "OptimizerFactory.h"
#include "../Utils/Error/WheelLibException.h"
#include "../Utils/Error/ErrorCodes.h"
#include <algorithm>
#include <cctype>

namespace WheelDL {
    namespace Optimizer {

        std::unique_ptr<torch::optim::Optimizer> OptimizerFactory::createFromConfig(
            const std::vector<torch::Tensor>& parameters,
            const Config::Configuration& config
        ) {
            if (parameters.empty()) {
                throw Utils::ConfigurationException(
                    Utils::ErrorCode::INVALID_CONFIG,
                    "Cannot create optimizer: parameter list is empty"
                );
            }

            std::string optimizerType = config.getOptimizer();
            std::transform(optimizerType.begin(), optimizerType.end(),
                optimizerType.begin(), ::tolower);

            float lr = config.getLearningRate();
            float momentum = config.getMomentum();
            float weightDecay = config.getWeightDecay();
            bool amsgrad = config.getAmsgrad();

            if (optimizerType == "sgd") {
                return createSGD(parameters, lr, momentum, weightDecay);
            }
            else if (optimizerType == "adam") {
                return createAdam(parameters, lr, momentum, 0.999f, 1e-8f, weightDecay, amsgrad);
            }
            else if (optimizerType == "adamw") {
                return createAdamW(parameters, lr, momentum, 0.999f, 1e-8f, weightDecay, amsgrad);
            }
            else if (optimizerType == "auto") {
                // Fallback for auto without model - use AdamW as default
                return createAdamW(parameters, lr, momentum, 0.999f, 1e-8f,
                    weightDecay > 0.0f ? weightDecay : 0.01f, amsgrad);
            }
            else {
                throw Utils::ConfigurationException(
                    Utils::ErrorCode::INVALID_CONFIG,
                    "Unsupported optimizer type: " + config.getOptimizer() +
                    ". Supported types: SGD, Adam, AdamW, auto"
                );
            }
        }

        std::unique_ptr<torch::optim::Optimizer> OptimizerFactory::createSGD(
            const std::vector<torch::Tensor>& parameters,
            float lr,
            float momentum,
            float weightDecay,
            bool nesterov
        ) {
            if (parameters.empty()) {
                throw Utils::ConfigurationException(
                    Utils::ErrorCode::INVALID_CONFIG,
                    "Cannot create optimizer: parameter list is empty"
                );
            }

            if (lr <= 0.0f) {
                throw Utils::ConfigurationException(
                    Utils::ErrorCode::INVALID_CONFIG,
                    "Learning rate must be positive, got: " + std::to_string(lr)
                );
            }

            torch::optim::SGDOptions options(lr);
            options.momentum(momentum);
            options.weight_decay(weightDecay);
            options.nesterov(nesterov);
            options.dampening(0.0);

            return std::make_unique<torch::optim::SGD>(parameters, options);
        }

        std::unique_ptr<torch::optim::Optimizer> OptimizerFactory::createAdam(
            const std::vector<torch::Tensor>& parameters,
            float lr,
            float beta1,
            float beta2,
            float eps,
            float weightDecay,
            bool amsgrad
        ) {
            if (parameters.empty()) {
                throw Utils::ConfigurationException(
                    Utils::ErrorCode::INVALID_CONFIG,
                    "Cannot create optimizer: parameter list is empty"
                );
            }

            if (lr <= 0.0f) {
                throw Utils::ConfigurationException(
                    Utils::ErrorCode::INVALID_CONFIG,
                    "Learning rate must be positive, got: " + std::to_string(lr)
                );
            }

            torch::optim::AdamOptions options(lr);
            options.betas(std::make_tuple(beta1, beta2));
            options.eps(eps);
            options.weight_decay(weightDecay);
            options.amsgrad(amsgrad);

            return std::make_unique<torch::optim::Adam>(parameters, options);
        }

        std::unique_ptr<torch::optim::Optimizer> OptimizerFactory::createAdamW(
            const std::vector<torch::Tensor>& parameters,
            float lr,
            float beta1,
            float beta2,
            float eps,
            float weightDecay,
            bool amsgrad
        ) {
            if (parameters.empty()) {
                throw Utils::ConfigurationException(
                    Utils::ErrorCode::INVALID_CONFIG,
                    "Cannot create optimizer: parameter list is empty"
                );
            }

            if (lr <= 0.0f) {
                throw Utils::ConfigurationException(
                    Utils::ErrorCode::INVALID_CONFIG,
                    "Learning rate must be positive, got: " + std::to_string(lr)
                );
            }

            torch::optim::AdamWOptions options(lr);
            options.betas(std::make_tuple(beta1, beta2));
            options.eps(eps);
            options.weight_decay(weightDecay);
            options.amsgrad(amsgrad);

            return std::make_unique<torch::optim::AdamW>(parameters, options);
        }

    } // namespace Optimizer
} // namespace WheelDL