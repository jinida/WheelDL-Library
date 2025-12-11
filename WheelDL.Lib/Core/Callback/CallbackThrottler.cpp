#include "pch.h"
#include "CallbackThrottler.h"

namespace WheelDL {
    namespace Core {
        namespace Callback {

            CallbackThrottler::CallbackThrottler(int minIntervalMs)
                : _minIntervalMs(minIntervalMs)
                , _timer()
            {
                if (minIntervalMs < 0) {
                    throw WheelDL::Utils::WheelLibException(
                        WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
                        "Minimum interval must be non-negative"
                    );
                }
            }

            bool CallbackThrottler::shouldInvoke() const
            {
                std::lock_guard<std::mutex> lock(_mutex);
                return _timer.elapsedMilliseconds() >= _minIntervalMs;
            }

            void CallbackThrottler::recordInvocation()
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _timer.reset();
            }

            bool CallbackThrottler::tryInvoke()
            {
                std::lock_guard<std::mutex> lock(_mutex);

                if (_timer.elapsedMilliseconds() >= _minIntervalMs) {
                    _timer.reset();
                    return true;
                }
                return false;
            }

            void CallbackThrottler::setMinInterval(int ms)
            {
                if (ms < 0) {
                    throw WheelDL::Utils::WheelLibException(
                        WheelDL::Utils::ErrorCode::INVALID_ARGUMENT,
                        "Minimum interval must be non-negative"
                    );
                }

                std::lock_guard<std::mutex> lock(_mutex);
                _minIntervalMs = ms;
            }

            int CallbackThrottler::getMinInterval() const
            {
                std::lock_guard<std::mutex> lock(_mutex);
                return _minIntervalMs;
            }

        } // namespace Callback
    } // namespace Core
} // namespace WheelDL
