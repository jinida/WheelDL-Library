#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_set>
#include "../Config/Configuration.h"

namespace WheelDL {
    namespace Optimizer {

        class OptimizerFactory {
        public:
            // ============================================================================
            // Parameter Groups Structure
            // ============================================================================
            struct ParameterGroups {
                std::vector<torch::Tensor> biasParams;    // bias (no decay)
                std::vector<torch::Tensor> bnParams;      // BatchNorm weight (no decay)
                std::vector<torch::Tensor> weightParams;  // weight (with decay)
            };

            // ============================================================================
            // Create Optimizer from Config (with model for parameter grouping)
            // ============================================================================
            template<typename ModuleType>
            static std::unique_ptr<torch::optim::Optimizer> createFromConfig(
                ModuleType& model,
                Config::Configuration& config
            ) {
                // Get optimizer type
                std::string optimizerType = config.getOptimizer();
                std::transform(optimizerType.begin(), optimizerType.end(),optimizerType.begin(), ::tolower);
                float lr = config.getLearningRate();
                float momentum = config.getMomentum();
                float weightDecay = config.getWeightDecay();
                bool amsgrad = config.getAmsgrad();

                // Classify parameters into groups
                auto groups = classifyParameters(model);

                if (groups.biasParams.empty() &&
                    groups.bnParams.empty() &&
                    groups.weightParams.empty()) {
                    throw std::invalid_argument("No trainable parameters found in model");
                }

                // Create optimizer based on type
                if (optimizerType == "sgd") {
                    return createSGDWithGroups(groups, lr, momentum, weightDecay);
                }
                else if (optimizerType == "adam") {
                    return createAdamWithGroups(groups, lr, momentum, weightDecay, amsgrad);
                }
                else if (optimizerType == "adamw") {
                    return createAdamWWithGroups(groups, lr, momentum, weightDecay, amsgrad);
                }
                else if (optimizerType == "auto") {
                    return selectAuto(model, config);
                }
                else {
                    throw std::invalid_argument(
                        "Unsupported optimizer type: " + config.getOptimizer() +
                        ". Supported types: SGD, Adam, AdamW, auto"
                    );
                }
            }

            template<typename ModuleType>
            static std::unique_ptr<torch::optim::Optimizer> selectAuto(
                ModuleType& model,
                Config::Configuration& config
            ) {
                auto numSamples = config.getNumDataSamples();
                auto epochs = config.getEpochs();
                auto batchSize = config.getBatchSize();
                auto nc = config.getNumClasses();

                // Calculate iterations
                int64_t iterations = (numSamples / std::max(batchSize, 1)) * epochs;

                float lr;
                float momentum = 0.9f;
                float weightDecay = config.getWeightDecay();
                std::string optimizerName;

                // Auto selection logic (from Python YOLO)
                if (iterations > 10000) {
                    optimizerName = "SGD";
                    lr = 0.01f;
                }
                else {
                    optimizerName = "AdamW";
                    // lr_fit = 0.002 * 5 / (4 + nc)
                    lr = 0.002f * 5.0f / (4.0f + static_cast<float>(nc));
                }
                
                config.setLearningRateFirst(lr);
				config.setOptimizer(optimizerName);

                // Classify parameters into groups
                auto groups = classifyParameters(model);

                // Create optimizer based on selection
                if (optimizerName == "SGD") {
                    return createSGDWithGroups(groups, lr, momentum, weightDecay);
                }
                else {
                    return createAdamWWithGroups(groups, lr, momentum, weightDecay, config.getAmsgrad());
                }
            }

            static std::unique_ptr<torch::optim::Optimizer> createFromConfig(
                const std::vector<torch::Tensor>& parameters,
                const Config::Configuration& config
            );

            static std::unique_ptr<torch::optim::Optimizer> createSGD(
                const std::vector<torch::Tensor>& parameters,
                float lr,
                float momentum = 0.9f,
                float weightDecay = 0.0f,
                bool nesterov = true
            );

            static std::unique_ptr<torch::optim::Optimizer> createAdam(
                const std::vector<torch::Tensor>& parameters,
                float lr,
                float beta1 = 0.9f,
                float beta2 = 0.999f,
                float eps = 1e-8f,
                float weightDecay = 0.0f,
                bool amsgrad = false
            );

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
            template<typename ModuleType>
            static ParameterGroups classifyParameters(ModuleType& model) {
                ParameterGroups groups;

                for (const auto& named_param : model.named_parameters()) {
                    const auto& paramName = named_param.key();
                    auto& param = named_param.value();

                    if (!param.requires_grad()) {
                        continue;
                    }

                    if (containsIgnoreCase(paramName, "bias")) {
                        // Bias parameters (no decay)
                        groups.biasParams.push_back(param);
                    }
                    else if (containsIgnoreCase(paramName, "logit_scale")) {
                        // ContrastiveHead logit_scale (no decay)
                        groups.bnParams.push_back(param);
                    }
                    else if (isNormalizationWeight(paramName)) {
                        // BatchNorm, LayerNorm, etc. weights (no decay)
                        groups.bnParams.push_back(param);
                    }
                    else {
                        // Regular weights (with decay)
                        groups.weightParams.push_back(param);
                    }
                }

                return groups;
            }

            static bool containsIgnoreCase(const std::string& str, const std::string& substr) {
                std::string lowerStr = str;
                std::string lowerSubstr = substr;
                std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
                std::transform(lowerSubstr.begin(), lowerSubstr.end(), lowerSubstr.begin(), ::tolower);
                return lowerStr.find(lowerSubstr) != std::string::npos;
            }

            static bool isNormalizationWeight(const std::string& paramName) {
                std::string lowerName = paramName;
                std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

                // Check for common normalization layer patterns
                bool isNormLayer =
                    lowerName.find("bn") != std::string::npos ||
                    lowerName.find("batch_norm") != std::string::npos ||
                    lowerName.find("batchnorm") != std::string::npos ||
                    lowerName.find("layer_norm") != std::string::npos ||
                    lowerName.find("layernorm") != std::string::npos ||
                    lowerName.find("group_norm") != std::string::npos ||
                    lowerName.find("groupnorm") != std::string::npos ||
                    lowerName.find("instance_norm") != std::string::npos ||
                    lowerName.find("instancenorm") != std::string::npos ||
                    (lowerName.find("norm") != std::string::npos &&
                        lowerName.find("weight") != std::string::npos);

                return isNormLayer && lowerName.find("weight") != std::string::npos;
            }

            static std::unique_ptr<torch::optim::Optimizer> createSGDWithGroups(
                const ParameterGroups& groups,
                float lr,
                float momentum,
                float weightDecay
            ) {
                auto defaultOptions = torch::optim::SGDOptions(lr)
                    .momentum(momentum)
                    .nesterov(true)
                    .weight_decay(0.0);

                std::vector<torch::Tensor> allParams;
                for (const auto& p : groups.biasParams) allParams.push_back(p);
                for (const auto& p : groups.weightParams) allParams.push_back(p);
                for (const auto& p : groups.bnParams) allParams.push_back(p);

                if (allParams.empty()) {
                    throw std::invalid_argument("No parameters to optimize");
                }

                auto optimizer = std::make_unique<torch::optim::SGD>(
                    std::vector<torch::Tensor>{allParams[0]}, defaultOptions
                );
                optimizer->param_groups().clear();

                if (!groups.biasParams.empty()) {
                    torch::optim::OptimizerParamGroup group(groups.biasParams);
                    group.set_options(std::make_unique<torch::optim::SGDOptions>(
                        torch::optim::SGDOptions(lr)
                        .momentum(momentum)
                        .nesterov(true)
                        .weight_decay(0.0)
                    ));
                    optimizer->add_param_group(group);
                }

                if (!groups.weightParams.empty()) {
                    torch::optim::OptimizerParamGroup group(groups.weightParams);
                    group.set_options(std::make_unique<torch::optim::SGDOptions>(
                        torch::optim::SGDOptions(lr)
                        .momentum(momentum)
                        .nesterov(true)
                        .weight_decay(weightDecay)
                    ));
                    optimizer->add_param_group(group);
                }

                if (!groups.bnParams.empty()) {
                    torch::optim::OptimizerParamGroup group(groups.bnParams);
                    group.set_options(std::make_unique<torch::optim::SGDOptions>(
                        torch::optim::SGDOptions(lr)
                        .momentum(momentum)
                        .nesterov(true)
                        .weight_decay(0.0)
                    ));
                    optimizer->add_param_group(group);
                }

                return optimizer;
            }

            static std::unique_ptr<torch::optim::Optimizer> createAdamWithGroups(
                const ParameterGroups& groups,
                float lr,
                float momentum,
                float weightDecay,
                bool amsgrad
            ) {
                auto defaultOptions = torch::optim::AdamOptions(lr)
                    .betas(std::make_tuple(momentum, 0.999))
                    .weight_decay(0.0)
                    .amsgrad(amsgrad);

                std::vector<torch::Tensor> allParams;
                for (const auto& p : groups.biasParams) allParams.push_back(p);
                for (const auto& p : groups.weightParams) allParams.push_back(p);
                for (const auto& p : groups.bnParams) allParams.push_back(p);

                if (allParams.empty()) {
                    throw std::invalid_argument("No parameters to optimize");
                }

                auto optimizer = std::make_unique<torch::optim::Adam>(
                    std::vector<torch::Tensor>{allParams[0]}, defaultOptions
                );
                optimizer->param_groups().clear();

                if (!groups.biasParams.empty()) {
                    torch::optim::OptimizerParamGroup group(groups.biasParams);
                    group.set_options(std::make_unique<torch::optim::AdamOptions>(
                        torch::optim::AdamOptions(lr)
                        .betas(std::make_tuple(momentum, 0.999))
                        .weight_decay(0.0)
                        .amsgrad(amsgrad)
                    ));
                    optimizer->add_param_group(group);
                }

                if (!groups.weightParams.empty()) {
                    torch::optim::OptimizerParamGroup group(groups.weightParams);
                    group.set_options(std::make_unique<torch::optim::AdamOptions>(
                        torch::optim::AdamOptions(lr)
                        .betas(std::make_tuple(momentum, 0.999))
                        .weight_decay(weightDecay)
                        .amsgrad(amsgrad)
                    ));
                    optimizer->add_param_group(group);
                }

                if (!groups.bnParams.empty()) {
                    torch::optim::OptimizerParamGroup group(groups.bnParams);
                    group.set_options(std::make_unique<torch::optim::AdamOptions>(
                        torch::optim::AdamOptions(lr)
                        .betas(std::make_tuple(momentum, 0.999))
                        .weight_decay(0.0)
                        .amsgrad(amsgrad)
                    ));
                    optimizer->add_param_group(group);
                }

                return optimizer;
            }


            static std::unique_ptr<torch::optim::Optimizer> createAdamWWithGroups(
                const ParameterGroups& groups,
                float lr,
                float momentum,
                float weightDecay,
                bool amsgrad
            ) {
                auto defaultOptions = torch::optim::AdamWOptions(lr)
                    .betas(std::make_tuple(momentum, 0.999))
                    .weight_decay(0.0)
                    .amsgrad(amsgrad);

                std::vector<torch::Tensor> allParams;
                for (const auto& p : groups.biasParams) allParams.push_back(p);
                for (const auto& p : groups.weightParams) allParams.push_back(p);
                for (const auto& p : groups.bnParams) allParams.push_back(p);

                if (allParams.empty()) {
                    throw std::invalid_argument("No parameters to optimize");
                }

                auto optimizer = std::make_unique<torch::optim::AdamW>(
                    std::vector<torch::Tensor>{allParams[0]}, defaultOptions
                );
                optimizer->param_groups().clear();

                if (!groups.biasParams.empty()) {
                    torch::optim::OptimizerParamGroup group(groups.biasParams);
                    group.set_options(std::make_unique<torch::optim::AdamWOptions>(
                        torch::optim::AdamWOptions(lr)
                        .betas(std::make_tuple(momentum, 0.999))
                        .weight_decay(0.0)
                        .amsgrad(amsgrad)
                    ));
                    optimizer->add_param_group(group);
                }

                if (!groups.weightParams.empty()) {
                    torch::optim::OptimizerParamGroup group(groups.weightParams);
                    group.set_options(std::make_unique<torch::optim::AdamWOptions>(
                        torch::optim::AdamWOptions(lr)
                        .betas(std::make_tuple(momentum, 0.999))
                        .weight_decay(weightDecay)
                        .amsgrad(amsgrad)
                    ));
                    optimizer->add_param_group(group);
                }

                if (!groups.bnParams.empty()) {
                    torch::optim::OptimizerParamGroup group(groups.bnParams);
                    group.set_options(std::make_unique<torch::optim::AdamWOptions>(
                        torch::optim::AdamWOptions(lr)
                        .betas(std::make_tuple(momentum, 0.999))
                        .weight_decay(0.0)
                        .amsgrad(amsgrad)
                    ));
                    optimizer->add_param_group(group);
                }

                return optimizer;
            }
        };
    } // namespace Optimizer
} // namespace WheelDL