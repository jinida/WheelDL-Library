#pragma once

#include <chrono>
#include <atomic>

namespace WheelDL {
	namespace Utils {

		/**
		 * @class Timer
		 * @brief Thread-safe timer for measuring elapsed time
		 *
		 * Uses atomic operations to ensure thread-safe access to the start time.
		 * Multiple threads can safely call elapsed methods while another thread calls reset().
		 *
		 * Usage:
		 * @code
		 * Timer timer;
		 * // ... do some work ...
		 * double elapsed = timer.elapsedMilliseconds();
		 * @endcode
		 */
		class Timer {
		public:
			/**
			 * @brief Constructor - starts the timer
			 */
			Timer();

			/**
			 * @brief Reset the timer to current time (thread-safe)
			 */
			void reset();

			/**
			 * @brief Get elapsed time in milliseconds (thread-safe)
			 * @return double Elapsed time in milliseconds
			 */
			double elapsedMilliseconds() const;

			/**
			 * @brief Get elapsed time in seconds (thread-safe)
			 * @return double Elapsed time in seconds
			 */
			double elapsedSeconds() const;

			/**
			 * @brief Get elapsed time in microseconds (thread-safe)
			 * @return double Elapsed time in microseconds
			 */
			double elapsedMicroseconds() const;

		private:
			// Atomic storage of start time as nanoseconds since epoch
			// This allows lock-free thread-safe access
			std::atomic<int64_t> _startTimeNanos;

			// Helper to get current time as nanoseconds
			static int64_t getCurrentTimeNanos();
		};

	} // namespace Utils
} // namespace WheelDL
