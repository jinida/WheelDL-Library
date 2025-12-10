// OBBLoss.cpp
#include "pch.h"
#include "OBBLoss.h"
#include "../Utils/IoU.h"
#include <cmath>
#include <stdexcept>

namespace WheelDL {
    namespace Model {
        namespace Loss {

            class OrientedBboxLoss {
            public:
                explicit OrientedBboxLoss(int64_t regMax) {
                    if (regMax > 1) {
                        _dflLoss = std::make_unique<DFLoss>(regMax);
                    }
                }

                std::tuple<torch::Tensor, torch::Tensor> forward(
                    const torch::Tensor& predDist,
                    const torch::Tensor& predBboxes,
                    const torch::Tensor& anchorPoints,
                    const torch::Tensor& targetBboxes,
                    const torch::Tensor& targetScores,
                    const torch::Tensor& targetScoresSum,
                    const torch::Tensor& fgMask
                ) {
                    auto weight = targetScores.sum(-1).index({ fgMask });

                    auto predBboxesFg = predBboxes.index({ fgMask });
                    auto targetBboxesFg = targetBboxes.index({ fgMask });

                    auto iou = Utils::probiou(predBboxesFg, targetBboxesFg);

                    auto lossIoU = ((1.0f - iou) * weight).sum() / targetScoresSum;

                    torch::Tensor lossDfl;
                    if (_dflLoss) 
                    {
                        auto targetXywh = targetBboxes.slice(2, 0, 4);
                        auto targetXyxy = Utils::xywh2xyxy(targetXywh);

                        auto targetLtrb = Utils::bbox2dist(
                            anchorPoints, targetXyxy, _dflLoss->getRegMax() - 1
                        );

                        auto predDistFg = predDist.index({ fgMask }).view({ -1, _dflLoss->getRegMax() });
                        auto targetLtrbFg = targetLtrb.index({ fgMask });

                        auto dflLossVal = _dflLoss->compute(predDistFg, targetLtrbFg);
                        lossDfl = (dflLossVal.view(-1) * weight).sum() / targetScoresSum;
                    }
                    else 
                    {
                        lossDfl = torch::zeros({ 1 }, predDist.options());
                    }

                    return std::make_tuple(lossIoU, lossDfl);
                }

            private:
                std::unique_ptr<DFLoss> _dflLoss;
            };

            OBBLoss::OBBLoss(
                int64_t numClasses,
                int64_t imageSize,
                const torch::Tensor& stride,
                float boxGain,
                float clsGain,
                float dflGain,
                int64_t talTopk
            )
                : _numClasses(numClasses)
                , _regMax(16)
                , _imageSize(imageSize)
                , _stride(stride.clone())
                , _boxGain(boxGain)
                , _clsGain(clsGain)
                , _dflGain(dflGain)
                , _useDfl(true)
                , _device(stride.device())
            {
                _bce = std::make_unique<BCEWithLogitsLoss>("none");
                _bboxLoss = std::make_unique<OrientedBboxLoss>(_regMax);
                _assigner = std::make_unique<Utils::OBBTaskAlignedAssigner>(
                    talTopk, numClasses, 0.5f, 6.0f
                );
                _numOutputs = _regMax * 4 + numClasses + 1;

                auto strideCpu = _stride.cpu();
                auto strideAccessor = strideCpu.accessor<float, 1>();
                _strideValuesCPU.reserve(strideCpu.size(0));

                for (int64_t i = 0; i < strideCpu.size(0); ++i) {
                    float s = strideAccessor[i];
                    _strideValuesCPU.push_back(s);

                    auto strideVal = static_cast<int64_t>(s);
                    auto featSize = _imageSize / strideVal;
                    _featShapes.emplace_back(featSize, featSize);
                }

                initializeAnchors();
            }

            OBBLoss::~OBBLoss() = default;

            void OBBLoss::initializeAnchors() {
                _proj = torch::arange(_regMax, torch::TensorOptions().dtype(torch::kFloat32).device(_device));

                std::tie(_anchorPoints, _strideTensor) = Utils::makeAnchors(
                    _featShapes, _stride, torch::kFloat32, _device, 0.5f
                );

                auto imgSizeF = static_cast<float>(_imageSize);
                _scaleTensor = torch::tensor(
                    { imgSizeF, imgSizeF, imgSizeF, imgSizeF },
                    torch::TensorOptions().dtype(torch::kFloat32).device(_device)
                );
            }

            void OBBLoss::to(const torch::Device& device) {
                _device = device;
                _stride = _stride.to(device);
                initializeAnchors();
            }

            torch::Tensor OBBLoss::preprocess(
                const torch::Tensor& targets,
                int64_t batchSize
            ) {
                if (targets.size(0) == 0) {
                    return torch::zeros({ batchSize, 0, 6 },
                        torch::TensorOptions().device(_device));
                }

                auto i = targets.select(1, 0).to(torch::kLong);

                auto uniqueResult = torch::_unique2(i, true, true, true);
                auto uniqueIndices = std::get<0>(uniqueResult);
                auto counts = std::get<2>(uniqueResult);

                auto maxCount = counts.max().item<int64_t>();
                auto out = torch::zeros({ batchSize, maxCount, 6 },
                    torch::TensorOptions().device(_device));

                if (targets.size(0) > 0 && batchSize > 0) {
                    auto uniqueIndicesCPU = uniqueIndices.cpu().contiguous();
                    auto countsCPU = counts.cpu().contiguous();

                    const int64_t numUnique = uniqueIndicesCPU.size(0);
                    const auto* uniquePtr = uniqueIndicesCPU.data_ptr<int64_t>();
                    const auto* countsPtr = countsCPU.data_ptr<int64_t>();

                    for (int64_t idx = 0; idx < numUnique; ++idx) {
                        auto batchIdx = uniquePtr[idx];
                        if (batchIdx < batchSize) {
                            auto mask = (i == batchIdx);
                            auto n = countsPtr[idx];

                            if (n > 0) {
                                auto bboxes = targets.index({ mask }).slice(1, 2, 7);

                                auto bboxesXywh = bboxes.slice(1, 0, 4);
                                bboxesXywh = bboxesXywh * _scaleTensor.unsqueeze(0);

                                auto angle = bboxes.slice(1, 4, 5);
                                auto scaledBboxes = torch::cat({ bboxesXywh, angle }, 1);

                                auto classLabels = targets.index({ mask }).slice(1, 1, 2);
                                out[batchIdx].slice(0, 0, n) = torch::cat({ classLabels, scaledBboxes }, 1);
                            }
                        }
                    }
                }

                return out;
            }

            torch::Tensor OBBLoss::decodeBbox(
                const torch::Tensor& predDist,
                const torch::Tensor& predAngle
            ) {
                torch::Tensor predDistDecoded;
                if (_useDfl) {
                    auto b = predDist.size(0);
                    auto a = predDist.size(1);
                    auto c = predDist.size(2);

                    auto predDistReshaped = predDist.view({ b, a, 4, c / 4 });
                    auto predDistSoftmax = torch::softmax(predDistReshaped, 3);
                    predDistDecoded = torch::matmul(predDistSoftmax, _proj);
                }
                else {
                    predDistDecoded = predDist;
                }

                auto rbox = Utils::dist2rbox(predDistDecoded, predAngle, _anchorPoints);
                return torch::cat({ rbox, predAngle }, -1);
            }

            std::unordered_map<std::string, torch::Tensor> OBBLoss::compute(
                const std::vector<torch::Tensor>& predictions,
                const Data::Dataset::DataExample& target
            ) {
                if (predictions.empty()) {
                    throw std::invalid_argument("OBBLoss::compute: predictions cannot be empty");
                }

                if (predictions[0].sizes().size() == 3)
                {
                    return compute(std::vector<torch::Tensor>(predictions.begin() + 1, predictions.end()), target);
                }

                bool hasAngleTensor = (predictions.back().dim() == 3);
                size_t numScales = hasAngleTensor ? predictions.size() - 1 : predictions.size();

                if (numScales == 0) {
                    throw std::invalid_argument("OBBLoss::compute: no detection predictions found");
                }

                auto batchSize = predictions[0].size(0);
                auto channels = predictions[0].size(1);

                int64_t totalAnchors = 0;
                for (size_t i = 0; i < numScales; ++i) {
                    totalAnchors += predictions[i].size(2) * predictions[i].size(3);
                }

                int64_t outputChannels = hasAngleTensor ? channels + 1 : channels;
                auto concatenated = torch::empty(
                    { batchSize, outputChannels, totalAnchors },
                    predictions[0].options()
                );

                int64_t offset = 0;
                for (size_t i = 0; i < numScales; ++i) {
                    const auto& pred = predictions[i];
                    auto numAnchors = pred.size(2) * pred.size(3);

                    concatenated.slice(1, 0, channels).slice(2, offset, offset + numAnchors).copy_(
                        pred.reshape({ batchSize, channels, numAnchors })
                    );
                    offset += numAnchors;
                }

                if (hasAngleTensor) {
                    const auto& angleTensor = predictions.back();
                    concatenated.slice(1, channels, channels + 1).copy_(angleTensor);
                }

                return compute(concatenated, target);
            }

            std::unordered_map<std::string, torch::Tensor> OBBLoss::compute(
                const torch::Tensor& prediction,
                const Data::Dataset::DataExample& target
            ) {
                auto loss = torch::zeros({ 3 }, torch::TensorOptions()
                    .dtype(torch::kFloat32)
                    .device(_device));

                auto predAngle = prediction.slice(1, _regMax * 4 + _numClasses, _regMax * 4 + _numClasses + 1);
                predAngle = predAngle.permute({ 0, 2, 1 }).contiguous();

                auto predDistri = prediction.slice(1, 0, _regMax * 4).permute({ 0, 2, 1 }).contiguous();
                auto predScores = prediction.slice(1, _regMax * 4, _regMax * 4 + _numClasses).permute({ 0, 2, 1 }).contiguous();

                auto batchSize = predScores.size(0);

                auto targets = target.targets;
                auto imgSizeF = static_cast<float>(_imageSize);

                auto rw = targets.select(1, 4) * imgSizeF;
                auto rh = targets.select(1, 5) * imgSizeF;
                auto validMask = (rw >= 2.0f) & (rh >= 2.0f);
                targets = targets.index({ validMask });

                targets = preprocess(targets, batchSize);

#if TORCH_VERSION_MAJOR < 2
                auto targetParts = targets.split_with_sizes({ 1, 5 }, 2);
#else
                auto targetParts = targets.split({ 1, 5 }, 2);
#endif
                auto gtLabels = targetParts[0];
                auto gtBboxes = targetParts[1];

                auto maskGt = gtBboxes.sum(2, true) > 0.0f;

                auto predBboxes = decodeBbox(predDistri, predAngle);

                auto bboxesForAssigner = predBboxes.clone().detach();
                bboxesForAssigner.slice(2, 0, 4).mul_(_strideTensor);

                auto [targetLabels, targetBboxes, targetScores, fgMask, targetGtIdx] = _assigner->forward(
                    predScores.detach().sigmoid(),
                    bboxesForAssigner,
                    _anchorPoints * _strideTensor,
                    gtLabels,
                    gtBboxes,
                    maskGt
                );

                auto targetScoresSum = targetScores.sum().clamp_min(1.0f);

                auto bceLossMap = _bce->compute(predScores, targetScores);
                loss[1] = bceLossMap.at("total").sum() / targetScoresSum;

                auto numFg = fgMask.sum();
                if (numFg.item<int64_t>() > 0) {
                    targetBboxes.slice(2, 0, 4).div_(_strideTensor);

                    auto [boxLoss, dflLoss] = _bboxLoss->forward(
                        predDistri,
                        predBboxes,
                        _anchorPoints,
                        targetBboxes,
                        targetScores,
                        targetScoresSum,
                        fgMask
                    );

                    loss[0] = boxLoss;
                    loss[2] = dflLoss;
                }
                else {
                    loss[0] += (predAngle * 0.0f).sum();
                }

                loss[0].mul_(_boxGain);
                loss[1].mul_(_clsGain);
                loss[2].mul_(_dflGain);

                return {
                    {"box", loss[0]},
                    {"cls", loss[1]},
                    {"dfl", loss[2]},
                    {"total", loss.sum() * static_cast<float>(batchSize)}
                };
            }
        }
    }
}