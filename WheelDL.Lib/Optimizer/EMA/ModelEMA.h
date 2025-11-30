#pragma once

#include "../../Model/Task/BaseModel.h"
#include <torch/torch.h>
#include <memory>
#include <cmath>
#include <unordered_map>

namespace WheelDL {
    namespace Optimizer {
        namespace EMA {

            /**
             * @class ModelEMA
             * @brief Exponential Moving Average for model parameters
             *
             * Maintains a smoothed version of model weights using exponential moving average.
             * This helps improve generalization and stability during inference.
             *
             * Formula: ema_param = decay * ema_param + (1 - decay) * current_param
             * Decay: max_decay * (1 - exp(-updates / decay_ramp))
             *
             * Reference: "Mean teachers are better role models"
             * https://arxiv.org/abs/1703.01780
             *
             * Note: This implementation uses parameter copying instead of model cloning
             * to avoid requiring torch::nn::Cloneable support in all model modules.
             */
            class ModelEMA {
            public:
                /**
                 * @brief Constructor
                 *
                 * Copies all parameters and buffers from the model for EMA tracking.
                 *
                 * @param model Model to track with EMA
                 * @param maxDecay Maximum decay rate (default: 0.9999)
                 * @param decayRamp Time constant to ramp up decay (default: 2000)
                 * @param updateCount Initial update count for resuming (default: 0)
                 * @throws ConfigurationException if model is null or parameters are invalid
                 */
                explicit ModelEMA(const Model::BaseModel& model,
                                 float maxDecay = 0.9999f,
                                 int decayRamp = 2000,
                                 int updateCount = 0);

                /**
                 * @brief Update EMA parameters with current model parameters
                 *
                 * Should be called after each training step or batch.
                 *
                 * @param model Current model with updated parameters
                 */
                void update(const Model::BaseModel& model);

                /**
                 * @brief Apply EMA parameters to the model
                 *
                 * Backs up current model parameters and applies EMA parameters.
                 * Call restoreOriginalParams() to restore the original parameters.
                 *
                 * @param model Model to apply EMA parameters to
                 */
                void applyToModel(Model::BaseModel& model);

                /**
                 * @brief Restore original parameters to the model
                 *
                 * Restores the parameters that were backed up during applyToModel().
                 *
                 * @param model Model to restore original parameters to
                 */
                void restoreOriginalParams(Model::BaseModel& model);

                /**
                 * @brief Get number of updates performed
                 *
                 * @return int Update count
                 */
                int getUpdates() const { return _updates; }

                /**
                 * @brief Get current decay rate (dynamic)
                 *
                 * Decay ramps up over time: max_decay * (1 - exp(-updates / decay_ramp))
                 *
                 * @return float Current decay rate
                 */
                float getDecay() const;

                /**
                 * @brief Get maximum decay rate
                 *
                 * @return float Maximum decay rate
                 */
                float getMaxDecay() const { return _maxDecay; }

                /**
                 * @brief Get decay ramp parameter
                 *
                 * @return int Decay ramp time constant
                 */
                int getDecayRamp() const { return _decayRamp; }

                /**
                 * @brief Set maximum decay rate
                 *
                 * @param maxDecay New maximum decay rate (must be in (0, 1))
                 */
                void setMaxDecay(float maxDecay);

                /**
                 * @brief Set decay ramp parameter
                 *
                 * @param decayRamp New decay ramp (must be > 0)
                 */
                void setDecayRamp(int decayRamp);

                /**
                 * @brief Reset update counter
                 */
                void resetUpdates() { _updates = 0; }

                /**
                 * @brief Enable/disable EMA updates
                 *
                 * @param enabled True to enable, false to disable
                 */
                void setEnabled(bool enabled) { _enabled = enabled; }

                /**
                 * @brief Check if EMA is enabled
                 *
                 * @return bool True if enabled
                 */
                bool isEnabled() const { return _enabled; }

            private:
                /**
                 * @brief Update single parameter with EMA
                 *
                 * @param emaParam EMA parameter to update
                 * @param currentParam Current parameter value
                 * @param decay Decay rate to use
                 */
                void updateParameter(torch::Tensor& emaParam,
                                   const torch::Tensor& currentParam,
                                   float decay);

            private:
                std::unordered_map<std::string, torch::Tensor> _emaParameters;   ///< EMA parameters (name -> tensor)
                std::unordered_map<std::string, torch::Tensor> _emaBuffers;      ///< EMA buffers (name -> tensor)
                std::unordered_map<std::string, torch::Tensor> _backupParameters; ///< Backup of original parameters
                std::unordered_map<std::string, torch::Tensor> _backupBuffers;    ///< Backup of original buffers
                float _maxDecay;                   ///< Maximum decay rate
                int _decayRamp;                    ///< Decay ramp time constant
                int _updates;                      ///< Number of updates
                bool _enabled;                     ///< Whether EMA is enabled
            };

        } // namespace EMA
    } // namespace Optimizer
} // namespace WheelDL
