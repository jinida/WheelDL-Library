#include "pch.h"
#include "EarlyStopping.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include <limits>
#include <algorithm>

namespace WheelDL {
    namespace Optimizer {
        namespace EarlyStopping {

            EarlyStopping::EarlyStopping(int patience)
                : _patience(patience)
                , _bestFitness(-std::numeric_limits<float>::infinity())
                , _counter(0)
                , _mayStopNext(false)
            {
                if (patience <= 0) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Patience must be positive, got: " + std::to_string(patience)
                    );
                }
            }

            bool EarlyStopping::shouldStop(float fitness)
            {
                // Check if current fitness is better than best
                if (fitness > _bestFitness) {
                    // Improvement found
                    _bestFitness = fitness;
                    _counter = 0;
                    _mayStopNext = false;
                    return false;
                }

                // No improvement
                _counter++;

                // Update mayStopNext flag
                _mayStopNext = (_counter >= _patience - 1);

                // Check if patience exceeded
                return _counter >= _patience;
            }

            void EarlyStopping::reset()
            {
                _bestFitness = -std::numeric_limits<float>::infinity();
                _counter = 0;
                _mayStopNext = false;
            }

            void EarlyStopping::setPatience(int patience)
            {
                if (patience <= 0) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::INVALID_CONFIG,
                        "Patience must be positive, got: " + std::to_string(patience)
                    );
                }
                _patience = patience;

                // Update mayStopNext based on new patience
                _mayStopNext = (_counter >= _patience - 1);
            }

        } // namespace EarlyStopping
    } // namespace Optimizer
} // namespace WheelDL
