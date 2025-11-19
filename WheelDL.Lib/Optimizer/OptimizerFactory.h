#pragma once

#include "../Config/Configuration.h"
#include <torch/torch.h>
#include <memory>
#include <vector>

namespace WheelDL {
    namespace Optimizer {

        /**
         * @class OptimizerFactory
         * @brief Factory class for creating PyTorch optimizers from configuration
         *
         * Supports:
         * - SGD (Stochastic Gradient Descent)
         * - Adam
         * - AdamW (Adam with decoupled weight decay)
         * - Auto (automatically selects based on task/model)
         */
        class OptimizerFactory {
        public:
            /**
             * @brief Create optimizer from configuration
             *
             * @param parameters Model parameters to optimize
             * @param config Configuration containing optimizer settings
             * @return std::unique_ptr<torch::optim::Optimizer> Created optimizer
             * @throws std::invalid_argument if optimizer type is unsupported
             */
            static std::unique_ptr<torch::optim::Optimizer> createFromConfig(
                const std::vector<torch::Tensor>& parameters,
                const Config::Configuration& config
            );

            /**
             * @brief Create SGD optimizer
             *
             * @param parameters Model parameters
             * @param lr Learning rate
             * @param momentum Momentum factor (default: 0.9)
             * @param weightDecay Weight decay (L2 penalty, default: 0.0005)
             * @param nesterov Use Nesterov momentum (default: true)
             * @return std::unique_ptr<torch::optim::Optimizer> SGD optimizer
             */
            static std::unique_ptr<torch::optim::Optimizer> createSGD(
                const std::vector<torch::Tensor>& parameters,
                float lr,
                float momentum = 0.9f,
                float weightDecay = 0.0005f,
                bool nesterov = true
            );

            /**
             * @brief Create Adam optimizer
             *
             * @param parameters Model parameters
             * @param lr Learning rate
             * @param beta1 First moment decay rate (default: 0.9)
             * @param beta2 Second moment decay rate (default: 0.999)
             * @param eps Epsilon for numerical stability (default: 1e-8)
             * @param weightDecay Weight decay (default: 0.0)
             * @return std::unique_ptr<torch::optim::Optimizer> Adam optimizer
             */
            static std::unique_ptr<torch::optim::Optimizer> createAdam(
                const std::vector<torch::Tensor>& parameters,
                float lr,
                float beta1 = 0.9f,
                float beta2 = 0.999f,
                float eps = 1e-8f,
                float weightDecay = 0.0f,
                bool amsgrad = false
            );

            /**
             * @brief Create AdamW optimizer
             *
             * AdamW implements Adam with decoupled weight decay regularization.
             *
             * @param parameters Model parameters
             * @param lr Learning rate
             * @param beta1 First moment decay rate (default: 0.9)
             * @param beta2 Second moment decay rate (default: 0.999)
             * @param eps Epsilon for numerical stability (default: 1e-8)
             * @param weightDecay Weight decay (default: 0.01)
             * @return std::unique_ptr<torch::optim::Optimizer> AdamW optimizer
             */
            static std::unique_ptr<torch::optim::Optimizer> createAdamW(
                const std::vector<torch::Tensor>& parameters,
                float lr,
                float beta1 = 0.9f,
                float beta2 = 0.999f,
                float eps = 1e-8f,
                float weightDecay = 0.01f,
                bool amsgrad = false
            );

        private:
            /**
             * @brief Automatically select optimizer based on configuration
             *
             * Selection strategy:
             * - Small models: Adam/AdamW
             * - Large models: SGD with momentum
             *
             * @param parameters Model parameters
             * @param config Configuration
             * @return std::unique_ptr<torch::optim::Optimizer> Selected optimizer
             */
            static std::unique_ptr<torch::optim::Optimizer> selectAuto(
                const std::vector<torch::Tensor>& parameters,
                const Config::Configuration& config
            );
        };

    } // namespace Optimizer
} // namespace WheelDL
