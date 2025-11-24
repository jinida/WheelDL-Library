#pragma once

#include <mutex>
#include "../../Utils/Common/Timer.h"
#include "../../Utils/Error/WheelLibException.h"

namespace WheelDL {
    namespace Core {
        namespace Callback {

            /**
             * @class CallbackThrottler
             * @brief Rate limiter for callback invocations
             *
             * This class prevents callbacks from being invoked too frequently,
             * which can improve performance by reducing overhead from UI updates
             * or logging operations.
             */
            class CallbackThrottler {
            public:
                /**
                 * @brief Constructor
                 * @param minIntervalMs Minimum interval between invocations in milliseconds (default: 100ms)
                 */
                explicit CallbackThrottler(int minIntervalMs = 100);

                /**
                 * @brief Check if callback should be invoked based on throttle interval
                 * @return true if sufficient time has passed since last invocation, false otherwise
                 */
                bool shouldInvoke() const;

                /**
                 * @brief Record that a callback invocation occurred
                 *
                 * This updates the last invocation time to the current time.
                 * Should be called after successfully invoking the callback.
                 */
                void recordInvocation();

                /**
                 * @brief Atomically check and record invocation
                 * @return true if callback should be invoked, false otherwise
                 *
                 * This method combines shouldInvoke() and recordInvocation() into a single
                 * atomic operation, preventing race conditions in multi-threaded scenarios.
                 */
                bool tryInvoke();

                /**
                 * @brief Set the minimum interval between invocations
                 * @param ms Minimum interval in milliseconds
                 */
                void setMinInterval(int ms);

                /**
                 * @brief Get the current minimum interval
                 * @return Minimum interval in milliseconds
                 */
                int getMinInterval() const;

            private:
                int _minIntervalMs;
                Utils::Timer _timer;
                mutable std::mutex _mutex;
            };

        } // namespace Callback
    } // namespace Core
} // namespace WheelDL
