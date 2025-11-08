#pragma once

#include <string>
#include <unordered_map>
#include <deque>
#include <chrono>
#include <mutex>
#include <memory>
#include <atomic>

namespace WheelDL {
namespace Utils {

// Forward declaration
class PerformanceProfiler;

/**
 * @class ScopedTimer
 * @brief RAII timer for automatic profiling
 *
 * Automatically starts timer on construction and stops on destruction.
 * Use with PerformanceProfiler::createScopedTimer().
 */
class ScopedTimer {
public:
    /**
     * @brief Construct and start timer
     * @param profiler Reference to PerformanceProfiler
     * @param name Timer name
     */
    ScopedTimer(PerformanceProfiler& profiler, const std::string& name);

    /**
     * @brief Destructor - automatically stops timer
     */
    ~ScopedTimer();

    // Disable copy and move
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

private:
    PerformanceProfiler& _profiler;
    std::string _name;
};

/**
 * @class PerformanceProfiler
 * @brief Singleton-based performance profiling system
 *
 * Measures execution time of code sections and generates reports.
 * - Manual start/stop timers
 * - RAII-based scoped timers
 * - Statistical reporting (min/max/avg)
 * - Export to JSON/HTML
 *
 * @note Thread-safe Singleton implementation
 */
class PerformanceProfiler {
public:
    // Get Singleton instance
    static PerformanceProfiler& getInstance();

    // Disable copy and move
    PerformanceProfiler(const PerformanceProfiler&) = delete;
    PerformanceProfiler& operator=(const PerformanceProfiler&) = delete;
    PerformanceProfiler(PerformanceProfiler&&) = delete;
    PerformanceProfiler& operator=(PerformanceProfiler&&) = delete;

    /**
     * @brief Start a named timer
     * @param name Timer name
     */
    void start(const std::string& name);

    /**
     * @brief Stop a named timer and record duration
     * @param name Timer name
     */
    void stop(const std::string& name);

    /**
     * @brief Get duration of last measurement
     * @param name Timer name
     * @return double Duration in milliseconds
     */
    double getDuration(const std::string& name) const;

    /**
     * @brief Get statistics for a timer
     * @param name Timer name
     * @return Statistics struct with count, average, min, max, total
     */
    struct Statistics {
        size_t count;
        double average;
        double minimum;
        double maximum;
        double total;
    };
    Statistics getStatistics(const std::string& name) const;

    /**
     * @brief Generate text report
     * @return std::string Report text
     */
    std::string report() const;

    /**
     * @brief Export timings to JSON file
     * @param filePath Output file path
     */
    void exportToJSON(const std::string& filePath) const;

    /**
     * @brief Export timings to HTML report
     * @param filePath Output file path
     */
    void exportToHTML(const std::string& filePath) const;

    /**
     * @brief Reset all timing data
     */
    void reset();

    /**
     * @brief Enable or disable profiler
     * @param enabled True to enable, false to disable
     */
    void setEnabled(bool enabled) {
        _enabled.store(enabled, std::memory_order_release);
    }

    /**
     * @brief Check if profiler is enabled
     * @return bool True if enabled
     */
    bool isEnabled() const {
        return _enabled.load(std::memory_order_acquire);
    }

    /**
     * @brief Set maximum number of timings to keep per timer (prevents unbounded growth)
     * @param maxTimings Maximum number of timing records (0 = unlimited, default: 1000)
     */
    void setMaxTimingsPerTimer(size_t maxTimings) {
        std::lock_guard<std::mutex> lock(_mutex);
        _maxTimingsPerTimer = maxTimings;
    }

    /**
     * @brief Create RAII scoped timer
     * @param name Timer name
     * @return ScopedTimer RAII timer object
     */
    ScopedTimer createScopedTimer(const std::string& name);

private:
    // Private constructor (Singleton)
    PerformanceProfiler();
    ~PerformanceProfiler() = default;

    // Helper function to calculate statistics from durations
    struct StatsResult {
        double total;
        double minVal;
        double maxVal;
        double avg;
    };
    StatsResult calculateStats(const std::deque<double>& durations) const;

    // Timing records (name -> list of durations in milliseconds)
    // Using deque instead of vector for efficient O(1) front removal (memory limit implementation)
    std::unordered_map<std::string, std::deque<double>> _timings;

    // Active timers (name -> start time point)
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> _activeTimers;

    // Thread-safety
    mutable std::mutex _mutex;

    // Enable/disable flag (atomic for lock-free access)
    std::atomic<bool> _enabled;

    // Maximum timings per timer (prevents unbounded memory growth)
    size_t _maxTimingsPerTimer;
};

} // namespace Utils
} // namespace WheelDL
