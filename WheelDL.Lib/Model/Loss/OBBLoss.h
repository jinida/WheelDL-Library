#pragma once

#include "BaseLoss.h"
#include "../Utils/TaskAlignedAssigner.h"
#include <torch/torch.h>
#include <memory>

namespace WheelDL {
    namespace Model {
        namespace Loss {

            class DFLoss;
            class OrientedBboxLoss;

            class OBBLoss : public BaseLoss {
            public:
                explicit OBBLoss(
                    int64_t numClasses,
                    int64_t imageSize,
                    const torch::Tensor& stride,
                    float boxGain = 7.5f,
                    float clsGain = 0.5f,
                    float dflGain = 1.5f,
                    int64_t talTopk = 10
                );

                ~OBBLoss();

                [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
                    const torch::Tensor& prediction,
                    const Data::Dataset::DataExample& target) override;

                [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
                    const std::vector<torch::Tensor>& predictions,
                    const Data::Dataset::DataExample& target) override;

                [[nodiscard]] std::string name() const override {
                    return "OBBLoss";
                }

                void setLossWeights(float box, float cls, float dfl) {
                    _boxGain = box;
                    _clsGain = cls;
                    _dflGain = dfl;
                }

                [[nodiscard]] std::tuple<float, float, float> getLossWeights() const {
                    return std::make_tuple(_boxGain, _clsGain, _dflGain);
                }

                void to(const torch::Device& device);

            private:
                [[nodiscard]] torch::Tensor preprocess(
                    const torch::Tensor& targets,
                    int64_t batchSize
                );

                [[nodiscard]] torch::Tensor decodeBbox(
                    const torch::Tensor& predDist,
                    const torch::Tensor& predAngle
                );

                void initializeAnchors();

            private:
                std::unique_ptr<BCEWithLogitsLoss> _bce;
                std::unique_ptr<OrientedBboxLoss> _bboxLoss;
                std::unique_ptr<Utils::OBBTaskAlignedAssigner> _assigner;

                int64_t _numClasses;
                int64_t _regMax;
                int64_t _imageSize;
                int64_t _numOutputs;

                torch::Tensor _stride;
                std::vector<float> _strideValuesCPU;
                std::vector<std::pair<int64_t, int64_t>> _featShapes;

                torch::Tensor _anchorPoints;
                torch::Tensor _strideTensor;
                torch::Tensor _scaleTensor;
                torch::Tensor _proj;

                float _boxGain;
                float _clsGain;
                float _dflGain;

                bool _useDfl;
                torch::Device _device;
            };

        }
    }
}