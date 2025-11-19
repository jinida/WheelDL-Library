#pragma once

#include "LRScheduler.h"

namespace WheelDL {
    namespace Optimizer {
        namespace Scheduler {

            /**
             * @class CosineAnnealingLR
             * @brief Cosine annealing learning rate scheduler
             *
             * Implements cosine annealing decay from initial LR to final LR.
             * After warmup, learning rate follows:
             *   lr = final_lr + (initial_lr - final_lr) * 0.5 * (1 + cos(pi * progress))
             *
             * Reference: "SGDR: Stochastic Gradient Descent with Warm Restarts"
             * https://arxiv.org/abs/1608.03983
             */
            class CosineAnnealingLR : public LRScheduler {
            public:
                /**
                 * @brief Constructor
                 *
                 * @param optimizer Pointer to optimizer
                 * @param config Configuration containing scheduler settings
                 */
                CosineAnnealingLR(torch::optim::Optimizer* optimizer,
                                 const Config::Configuration& config);

                /**
                 * @brief Destructor
                 */
                ~CosineAnnealingLR() override = default;

            protected:
                /**
                 * @brief Compute learning rate using cosine annealing
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
