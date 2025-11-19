#pragma once

#include "../../Config/Configuration.h"
#include <torch/torch.h>
#include <memory>

namespace WheelDL {
    namespace Optimizer {
        namespace Scheduler {

            /**
             * @class LRScheduler
             * @brief Base class for learning rate schedulers
             *
             * Implements warmup phase and delegates main scheduling to derived classes.
             * Supports:
             * - Warmup with linear ramp-up
             * - Configurable initial and final learning rates
             * - Virtual computeLR() for custom scheduling strategies
             */
            class LRScheduler {
            public:
                /**
                 * @brief Constructor
                 *
                 * @param optimizer Pointer to optimizer (scheduler does not take ownership)
                 * @param config Configuration containing scheduler settings
                 */
                LRScheduler(torch::optim::Optimizer* optimizer,
                           const Config::Configuration& config);

                /**
                 * @brief Virtual destructor
                 */
                virtual ~LRScheduler() = default;

                /**
                 * @brief Update learning rate for the given epoch
                 *
                 * Applies warmup during initial epochs, then delegates to computeLR().
                 *
                 * @param epoch Current epoch (0-indexed)
                 */
                virtual void step(int epoch);

                /**
                 * @brief Get current learning rate
                 *
                 * @return float Current LR from optimizer
                 */
                float getCurrentLR() const;

                /**
                 * @brief Get initial learning rate
                 *
                 * @return float Initial LR (lr0)
                 */
                float getInitialLR() const { return _initialLR; }

                /**
                 * @brief Get final learning rate
                 *
                 * @return float Final LR (lr0 * lrf)
                 */
                float getFinalLR() const { return _finalLR; }

                /**
                 * @brief Get total epochs
                 *
                 * @return int Total training epochs
                 */
                int getTotalEpochs() const { return _totalEpochs; }

                /**
                 * @brief Get warmup epochs
                 *
                 * @return int Warmup epochs
                 */
                int getWarmupEpochs() const { return _warmupEpochs; }

            protected:
                /**
                 * @brief Compute learning rate for given epoch
                 *
                 * Override this in derived classes to implement custom scheduling.
                 * This is called after warmup phase.
                 *
                 * @param epoch Current epoch (0-indexed)
                 * @return float Computed learning rate
                 */
                virtual float computeLR(int epoch) const = 0;

                /**
                 * @brief Set learning rate in optimizer
                 *
                 * @param lr New learning rate
                 */
                void setLR(float lr);

                /**
                 * @brief Compute warmup learning rate
                 *
                 * Linear ramp-up from 0 to initial LR.
                 *
                 * @param epoch Current epoch (0-indexed)
                 * @return float Warmup learning rate
                 */
                float computeWarmupLR(int epoch) const;

            protected:
                torch::optim::Optimizer* _optimizer;  ///< Pointer to optimizer (non-owning)
                float _initialLR;                      ///< Initial learning rate (lr0)
                float _finalLR;                        ///< Final learning rate (lr0 * lrf)
                int _totalEpochs;                      ///< Total number of epochs
                int _warmupEpochs;                     ///< Number of warmup epochs
            };

        } // namespace Scheduler
    } // namespace Optimizer
} // namespace WheelDL
