#include "pch.h"
#include "PerformanceProfiler.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <stdexcept>
#include <iostream>
#include <filesystem>

namespace WheelDL {
namespace Utils {

// ScopedTimer implementation
ScopedTimer::ScopedTimer(PerformanceProfiler& profiler, const std::string& name)
    : _profiler(profiler)
    , _name(name)
{
    _profiler.start(_name);
}

ScopedTimer::~ScopedTimer() {
    _profiler.stop(_name);
}

// PerformanceProfiler implementation
PerformanceProfiler::PerformanceProfiler()
    : _enabled(true)
    , _maxTimingsPerTimer(1000)  // Default: keep last 1000 timings
{
}

PerformanceProfiler& PerformanceProfiler::getInstance() {
    static PerformanceProfiler instance;
    return instance;
}

void PerformanceProfiler::start(const std::string& name) {
    // Fast lock-free check for enabled state
    if (!_enabled.load(std::memory_order_acquire)) return;

    // Validate timer name
    if (name.empty()) {
        throw std::invalid_argument("Timer name cannot be empty");
    }

    // Capture start time before acquiring lock for better accuracy
    auto startTime = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(_mutex);

    // Check for duplicate start (same timer started twice without stop)
    if (_activeTimers.find(name) != _activeTimers.end()) {
        throw std::logic_error("Timer '" + name + "' is already running. Call stop() before starting again.");
    }

    _activeTimers[name] = startTime;
}

void PerformanceProfiler::stop(const std::string& name) {
    // Fast lock-free check for enabled state
    if (!_enabled.load(std::memory_order_acquire)) return;

    // Validate timer name
    if (name.empty()) {
        std::cerr << "Warning: PerformanceProfiler::stop() called with empty timer name" << std::endl;
        return;
    }

    // Capture end time immediately for accurate measurement
    auto endTime = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(_mutex);

    auto it = _activeTimers.find(name);
    if (it == _activeTimers.end()) {
        // Timer not found or not started - log warning
        // This indicates a programming error (stop called without start)
        std::cerr << "Warning: PerformanceProfiler::stop() called for timer '"
                  << name << "' that was not started" << std::endl;
        return;
    }

    // Calculate duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        endTime - it->second).count() / 1000.0;

    // Record timing with bounded growth
    auto& timingVec = _timings[name];
    timingVec.push_back(duration);

    // Limit memory growth by keeping only last N timings
    if (_maxTimingsPerTimer > 0 && timingVec.size() > _maxTimingsPerTimer) {
        // Remove oldest timing (first element) - O(1) operation
        timingVec.pop_front();
    }

    // Remove from active timers
    _activeTimers.erase(it);
}

double PerformanceProfiler::getDuration(const std::string& name) const {
    std::lock_guard<std::mutex> lock(_mutex);

    auto it = _timings.find(name);
    if (it == _timings.end() || it->second.empty()) {
        throw std::runtime_error("No timing data for: " + name);
    }

    return it->second.back();  // Return last recorded duration
}

// Helper function to calculate statistics from durations
PerformanceProfiler::StatsResult PerformanceProfiler::calculateStats(const std::deque<double>& durations) const {
    StatsResult result;
    result.total = 0.0;
    result.minVal = durations.front();
    result.maxVal = durations.front();

    for (const auto& duration : durations) {
        result.total += duration;
        if (duration < result.minVal) result.minVal = duration;
        if (duration > result.maxVal) result.maxVal = duration;
    }
    result.avg = result.total / durations.size();

    return result;
}

PerformanceProfiler::Statistics PerformanceProfiler::getStatistics(const std::string& name) const {
    std::lock_guard<std::mutex> lock(_mutex);

    auto it = _timings.find(name);
    if (it == _timings.end() || it->second.empty()) {
        return Statistics{0, 0.0, 0.0, 0.0, 0.0};
    }

    const auto& durations = it->second;
    auto stats_result = calculateStats(durations);

    Statistics stats;
    stats.count = durations.size();
    stats.total = stats_result.total;
    stats.minimum = stats_result.minVal;
    stats.maximum = stats_result.maxVal;
    stats.average = stats_result.avg;

    return stats;
}

std::string PerformanceProfiler::report() const {
    std::lock_guard<std::mutex> lock(_mutex);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);

    oss << "Performance Profile Report\n";
    oss << std::string(80, '=') << "\n";
    oss << std::left << std::setw(30) << "Operation"
        << std::right << std::setw(10) << "Count"
        << std::setw(12) << "Avg (ms)"
        << std::setw(12) << "Min (ms)"
        << std::setw(12) << "Max (ms)"
        << std::setw(14) << "Total (ms)" << "\n";
    oss << std::string(80, '-') << "\n";

    for (const auto& [name, durations] : _timings) {
        if (durations.empty()) continue;

        // Calculate all statistics using helper function
        auto stats = calculateStats(durations);

        oss << std::left << std::setw(30) << name
            << std::right << std::setw(10) << durations.size()
            << std::setw(12) << stats.avg
            << std::setw(12) << stats.minVal
            << std::setw(12) << stats.maxVal
            << std::setw(14) << stats.total << "\n";
    }

    oss << std::string(80, '=') << "\n";

    return oss.str();
}

void PerformanceProfiler::exportToJSON(const std::string& filePath) const {
    std::lock_guard<std::mutex> lock(_mutex);

    // Write to temporary file first for atomicity
    std::string tempPath = filePath + ".tmp";
    {
        std::ofstream file(tempPath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + tempPath);
        }

        file << "{\n";
        file << "  \"timings\": {\n";

        bool first = true;
        for (const auto& [name, durations] : _timings) {
            if (!first) file << ",\n";
            first = false;

            file << "    \"" << name << "\": {\n";
            file << "      \"count\": " << durations.size() << ",\n";

            if (!durations.empty()) {
                // Calculate all statistics using helper function
                auto stats = calculateStats(durations);

                file << "      \"average_ms\": " << stats.avg << ",\n";
                file << "      \"min_ms\": " << stats.minVal << ",\n";
                file << "      \"max_ms\": " << stats.maxVal << ",\n";
                file << "      \"total_ms\": " << stats.total << "\n";
            }

            file << "    }";
        }

        file << "\n  }\n";
        file << "}\n";
        file.close();

        // Check if write succeeded
        if (file.fail())
        {
            std::filesystem::remove(tempPath);
            throw std::runtime_error("Failed to write to file: " + tempPath);
        }
    }

    // Atomically replace old file with new file
    std::filesystem::rename(tempPath, filePath);
}

void PerformanceProfiler::exportToHTML(const std::string& filePath) const {
    std::lock_guard<std::mutex> lock(_mutex);

    std::ofstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    file << "<!DOCTYPE html>\n";
    file << "<html>\n<head>\n";
    file << "<title>Performance Profile Report</title>\n";
    file << "<style>\n";
    file << "body { font-family: Arial, sans-serif; margin: 20px; }\n";
    file << "h1 { color: #333; }\n";
    file << "table { border-collapse: collapse; width: 100%; margin-top: 20px; }\n";
    file << "th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }\n";
    file << "th { background-color: #4CAF50; color: white; }\n";
    file << "tr:nth-child(even) { background-color: #f2f2f2; }\n";
    file << "td:first-child { text-align: left; }\n";
    file << "</style>\n";
    file << "</head>\n<body>\n";

    file << "<h1>Performance Profile Report</h1>\n";
    file << "<table>\n";
    file << "<tr><th>Operation</th><th>Count</th><th>Avg (ms)</th><th>Min (ms)</th><th>Max (ms)</th><th>Total (ms)</th></tr>\n";

    for (const auto& [name, durations] : _timings) {
        if (durations.empty()) continue;

        // Calculate all statistics using helper function
        auto stats = calculateStats(durations);

        file << "<tr>";
        file << "<td>" << name << "</td>";
        file << "<td>" << durations.size() << "</td>";
        file << std::fixed << std::setprecision(3);
        file << "<td>" << stats.avg << "</td>";
        file << "<td>" << stats.minVal << "</td>";
        file << "<td>" << stats.maxVal << "</td>";
        file << "<td>" << stats.total << "</td>";
        file << "</tr>\n";
    }

    file << "</table>\n";
    file << "</body>\n</html>\n";
    // file automatically closed by RAII when going out of scope
}

void PerformanceProfiler::reset() {
    std::lock_guard<std::mutex> lock(_mutex);
    _timings.clear();
    _activeTimers.clear();
}

ScopedTimer PerformanceProfiler::createScopedTimer(const std::string& name) {
    return ScopedTimer(*this, name);
}

} // namespace Utils
} // namespace WheelDL
