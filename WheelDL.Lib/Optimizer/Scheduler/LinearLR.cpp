#include "pch.h"
#include "LinearLR.h"
#include <algorithm>

namespace WheelDL {
    namespace Optimizer {
        namespace Scheduler {

            LinearLR::LinearLR(torch::optim::Optimizer* optimizer,
                              const Config::Configuration& config)
                : LRScheduler(optimizer, config)
            {
            }

            float LinearLR::computeLR(int epoch) const
            {
                float progress = computeProgress(epoch);

                // Linear decay formula
                float lr = _initialLR - (_initialLR - _finalLR) * progress;

                return lr;
            }

            float LinearLR::computeProgress(int epoch) const
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
