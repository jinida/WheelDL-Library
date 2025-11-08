#include "pch.h"
#include "Timer.h"

namespace WheelDL {
	namespace Utils {

		int64_t Timer::getCurrentTimeNanos() {
			auto now = std::chrono::high_resolution_clock::now();
			auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
				now.time_since_epoch()
			).count();
			return nanos;
		}

		Timer::Timer()
			: _startTimeNanos(getCurrentTimeNanos()) {
		}

		void Timer::reset() {
			// Atomic store with release semantics for thread-safety
			_startTimeNanos.store(getCurrentTimeNanos(), std::memory_order_release);
		}

		double Timer::elapsedMilliseconds() const {
			// Atomic load with acquire semantics for thread-safety
			int64_t startNanos = _startTimeNanos.load(std::memory_order_acquire);
			int64_t currentNanos = getCurrentTimeNanos();
			int64_t elapsedNanos = currentNanos - startNanos;

			// Convert nanoseconds to milliseconds
			return elapsedNanos / 1'000'000.0;
		}

		double Timer::elapsedSeconds() const {
			// Direct implementation for efficiency (no intermediate call)
			int64_t startNanos = _startTimeNanos.load(std::memory_order_acquire);
			int64_t currentNanos = getCurrentTimeNanos();
			int64_t elapsedNanos = currentNanos - startNanos;

			// Convert nanoseconds to seconds
			return elapsedNanos / 1'000'000'000.0;
		}

		double Timer::elapsedMicroseconds() const {
			// Atomic load with acquire semantics for thread-safety
			int64_t startNanos = _startTimeNanos.load(std::memory_order_acquire);
			int64_t currentNanos = getCurrentTimeNanos();
			int64_t elapsedNanos = currentNanos - startNanos;

			// Convert nanoseconds to microseconds
			return elapsedNanos / 1'000.0;
		}

	} // namespace Utils
} // namespace WheelDL
