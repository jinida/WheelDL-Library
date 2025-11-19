#pragma once

#include "LRScheduler.h"

namespace WheelDL {
    namespace Optimizer {
        namespace Scheduler {

            /**
             * @class LinearLR
             * @brief Linear decay learning rate scheduler
             *
             * Implements linear decay from initial LR to final LR.
             * After warmup, learning rate follows:
             *   lr = initial_lr - (initial_lr - final_lr) * progress
             *
             * This provides a simple, steady decay strategy suitable for
             * fine-tuning or when stable convergence is preferred.
             */
            class LinearLR : public LRScheduler {
            public:
                /**
                 * @brief Constructor
                 *
                 * @param optimizer Pointer to optimizer
                 * @param config Configuration containing scheduler settings
                 */
                LinearLR(torch::optim::Optimizer* optimizer,
                        const Config::Configuration& config);

                /**
                 * @brief Destructor
                 */
                ~LinearLR() override = default;

            protected:
                /**
                 * @brief Compute learning rate using linear decay
                 *
                 * @param epoch Current epoch (0-indexed)
                 * @return float Computed learning rate
                 */
                float computeLR(int epoch) const override;

            private:
                /**
                 * @brief Compute progress ratio (0.0 to 1.0)
                 *
                 * @param epoch Current epoch
                 * @return float Progress from warmup end to total epochs
                 */
                float computeProgress(int epoch) const;
            };

        } // namespace Scheduler
    } // namespace Optimizer
} // namespace WheelDL
