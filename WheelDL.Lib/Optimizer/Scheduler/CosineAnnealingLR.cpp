#include "pch.h"
#include "CosineAnnealingLR.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace WheelDL {
    namespace Optimizer {
        namespace Scheduler {

            CosineAnnealingLR::CosineAnnealingLR(torch::optim::Optimizer* optimizer,
                                                 const Config::Configuration& config)
                : LRScheduler(optimizer, config)
            {
            }

            float CosineAnnealingLR::computeLR(int epoch) const
            {
                float progress = computeProgress(epoch);

                // Cosine annealing formula
                float cosine_decay = 0.5f * (1.0f + std::cos(static_cast<float>(M_PI) * progress));
                float lr = _finalLR + (_initialLR - _finalLR) * cosine_decay;

                return lr;
            }

            float CosineAnnealingLR::computeProgress(int epoch) const
            {
                // Progress from end of warmup to end of training
                int effectiveEpoch = epoch - _warmupEpochs;
                int effectiveTotal = _totalEpochs - _warmupEpochs;

                if (effectiveTotal <= 0) {
                    return 1.0f;  // No training after warmup
                }

                float progress = static_cast<float>(effectiveEpoch) / static_cast<float>(effectiveTotal);

                // Clamp to [0, 1]
                return std::max(0.0f, std::min(1.0f, progress));
            }

        } // namespace Scheduler
    } // namespace Optimizer
} // namespace WheelDL
