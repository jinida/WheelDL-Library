#pragma once
#include <torch/nn/module.h>
#include <vector>
#include <string>

namespace WheelDL
{
    namespace Model
    {
        namespace Modules
        {
            class IBlockImpl : public torch::nn::Module
            {
            public:
                virtual torch::Tensor forward(torch::Tensor x) = 0;
            };

            /**
             * @brief Interface for anomaly detection models
             */
            class IAnomalyModel : public torch::nn::Module
            {
            public:
                virtual std::vector<torch::Tensor> forward(std::vector<torch::Tensor>& x) = 0;
                virtual std::string getModelType() const = 0;
                virtual int64_t getOutputChannels() const = 0;
                virtual ~IAnomalyModel() = default;
            };

            /**
             * @brief Interface for models that need feature preparation
             *
             * Note: This is a marker interface. Actual methods are templates
             * defined in the implementing class.
             */
            class IFeaturePreparable
            {
            public:
                virtual ~IFeaturePreparable() = default;
            };
        }
    }
}