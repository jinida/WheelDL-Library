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

            // Get optimizer type and convert to lowercase for case-insensitive comparison
            std::string optimizerType = config.getOptimizer();
            std::transform(optimizerType.begin(), optimizerType.end(), optimizerType.begin(),
                [](unsigned char c) { return std::tolower(c); });

            float lr = config.getLearningRate();
            float momentum = config.getMomentum();
            float weightDecay = config.getWeightDecay();
            bool amsgrad = config.getAmsgrad();

            // Create optimizer based on type
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
                return selectAuto(parameters, config);
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

            torch::optim::SGDOptions options(lr);
            options.momentum(momentum);
            options.weight_decay(weightDecay);
            options.nesterov(nesterov);
            options.dampening(0.0);  // Standard SGD with momentum

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

            torch::optim::AdamWOptions options(lr);
            options.betas(std::make_tuple(beta1, beta2));
            options.eps(eps);
            options.weight_decay(weightDecay);
            options.amsgrad(amsgrad);

            return std::make_unique<torch::optim::AdamW>(parameters, options);
        }

        std::unique_ptr<torch::optim::Optimizer> OptimizerFactory::selectAuto(
            const std::vector<torch::Tensor>& parameters,
            const Config::Configuration& config
        ) {
            // Calculate total number of parameters
            int64_t totalParams = 0;
            for (const auto& param : parameters) {
                totalParams += param.numel();
            }

            float lr = config.getLearningRate();
            float momentum = config.getMomentum();
            float weightDecay = config.getWeightDecay();
            bool amsgrad = config.getAmsgrad();
            // Selection strategy:
            // - Small models (< 10M params): Adam (faster convergence)
            // - Medium models (10M - 50M params): AdamW (better generalization)
            // - Large models (> 50M params): SGD with momentum (better final performance)

            constexpr int64_t SMALL_MODEL_THRESHOLD = 10'000'000;   // 10M parameters
            constexpr int64_t LARGE_MODEL_THRESHOLD = 50'000'000;   // 50M parameters

            if (totalParams < SMALL_MODEL_THRESHOLD) {
                // Small model: use Adam
                return createAdam(parameters, lr, momentum, 0.999f, 1e-8f,
                    weightDecay > 0.0f ? weightDecay : 0.0f, amsgrad);
            }
            else if (totalParams < LARGE_MODEL_THRESHOLD) {
                // Medium model: use AdamW with moderate weight decay
                return createAdamW(parameters, lr, momentum, 0.999f, 1e-8f,
                    weightDecay > 0.0f ? weightDecay : 0.01f, amsgrad);
            }
            else {
                // Large model: use SGD with Nesterov momentum
                return createSGD(parameters, lr, momentum, weightDecay);
            }
        }

    } // namespace Optimizer
} // namespace WheelDL
