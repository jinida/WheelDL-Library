#pragma once

namespace WheelDL {
    namespace Optimizer {
        namespace EarlyStopping {

            /**
             * @class EarlyStopping
             * @brief Early stopping mechanism to prevent overfitting
             *
             * Monitors validation fitness and stops training if no improvement
             * is observed for a specified number of epochs (patience).
             *
             * Fitness is typically validation mAP, accuracy, or negative loss.
             * Higher fitness values are considered better.
             */
            class EarlyStopping {
            public:
                /**
                 * @brief Constructor
                 *
                 * @param patience Number of epochs to wait for improvement (default: 50)
                 */
                explicit EarlyStopping(int patience = 50);

                /**
                 * @brief Check if training should stop
                 *
                 * Updates internal state based on current fitness.
                 *
                 * @param fitness Current validation fitness (higher is better)
                 * @return bool True if training should stop
                 */
                bool shouldStop(float fitness);

                /**
                 * @brief Check if next epoch might trigger stop
                 *
                 * Useful for scheduling final validations or checkpoints.
                 *
                 * @return bool True if next epoch without improvement will stop training
                 */
                bool mayStopNext() const { return _mayStopNext; }

                /**
                 * @brief Get best fitness observed
                 *
                 * @return float Best fitness value
                 */
                float getBestFitness() const { return _bestFitness; }

                /**
                 * @brief Get current patience counter
                 *
                 * @return int Epochs without improvement
                 */
                int getCounter() const { return _counter; }

                /**
                 * @brief Get patience threshold
                 *
                 * @return int Maximum epochs without improvement
                 */
                int getPatience() const { return _patience; }

                /**
                 * @brief Reset early stopping state
                 *
                 * Clears counter and best fitness. Useful when resuming training.
                 */
                void reset();

                /**
                 * @brief Set new patience value
                 *
                 * @param patience New patience (must be positive)
                 */
                void setPatience(int patience);

            private:
                int _patience;         ///< Patience (epochs without improvement)
                float _bestFitness;    ///< Best fitness observed
                int _counter;          ///< Current counter (epochs without improvement)
                bool _mayStopNext;     ///< True if next epoch might trigger stop
            };

        } // namespace EarlyStopping
    } // namespace Optimizer
} // namespace WheelDL
