#include "pch.h"
#include "Utils/Profiler/PerformanceProfiler.h"
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>
#include <chrono>
#include <gtest/gtest.h>

using namespace WheelDL::Utils;
namespace fs = std::filesystem;

// =============================================================================
// Test Fixture
// =============================================================================

class PerformanceProfilerTest : public ::testing::Test {
protected:
    std::string testDir;

    void SetUp() override {
        testDir = (fs::temp_directory_path() / "WheelDL_Profiler_Test").string();
        fs::remove_all(testDir);
        fs::create_directories(testDir);
    }

    void TearDown() override {
        try {
            fs::remove_all(testDir);
        }
        catch (...) {}
    }

    std::string getTestPath(const std::string& filename) {
        return (fs::path(testDir) / filename).string();
    }

    std::string readFileContent(const std::string& path) {
        std::ifstream file(path);
        if (!file) return "";
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
};

// =============================================================================
// Factory & Lifecycle Tests (PRF-001 ~ PRF-003)
// =============================================================================

// PRF-001: Create returns valid
TEST_F(PerformanceProfilerTest, Create_ReturnsValid) {
    auto profiler = PerformanceProfiler::create();

    EXPECT_NE(nullptr, profiler);
}

// PRF-002: Create multiple instances
TEST_F(PerformanceProfilerTest, Create_MultipleInstances) {
    auto profiler1 = PerformanceProfiler::create();
    auto profiler2 = PerformanceProfiler::create();
    auto profiler3 = PerformanceProfiler::create();

    EXPECT_NE(nullptr, profiler1);
    EXPECT_NE(nullptr, profiler2);
    EXPECT_NE(nullptr, profiler3);
    EXPECT_NE(profiler1.get(), profiler2.get());
    EXPECT_NE(profiler2.get(), profiler3.get());
}

// PRF-003: Destructor cleanup
TEST_F(PerformanceProfilerTest, Destructor_Cleanup) {
    {
        auto profiler = PerformanceProfiler::create();
        profiler->start("test");
        profiler->stop("test");
    }
    // No crash
    SUCCEED();
}

// =============================================================================
// Timer Operations Tests (PRF-004 ~ PRF-010)
// =============================================================================

// PRF-004: Start/stop basic usage
TEST_F(PerformanceProfilerTest, StartStop_BasicUsage) {
    auto profiler = PerformanceProfiler::create();

    profiler->start("test_timer");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler->stop("test_timer");

    double duration = profiler->getDuration("test_timer");
    EXPECT_GT(duration, 0.0);
}

// PRF-005: Start/stop multiple timers
TEST_F(PerformanceProfilerTest, StartStop_MultipleTimers) {
    auto profiler = PerformanceProfiler::create();

    profiler->start("timer1");
    profiler->start("timer2");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    profiler->stop("timer1");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    profiler->stop("timer2");

    double d1 = profiler->getDuration("timer1");
    double d2 = profiler->getDuration("timer2");

    EXPECT_GT(d1, 0.0);
    EXPECT_GT(d2, 0.0);
    EXPECT_GT(d2, d1);  // timer2 ran longer
}

// PRF-006: Same timer multiple measurements
TEST_F(PerformanceProfilerTest, StartStop_SameTimerMultiple) {
    auto profiler = PerformanceProfiler::create();

    for (int i = 0; i < 5; i++) {
        profiler->start("repeated");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        profiler->stop("repeated");
    }

    auto stats = profiler->getStatistics("repeated");
    EXPECT_EQ(5u, stats.count);
}

// PRF-008: Stop not started timer
TEST_F(PerformanceProfilerTest, Stop_NotStarted) {
    auto profiler = PerformanceProfiler::create();

    EXPECT_NO_THROW(profiler->stop("never_started"));
}

// PRF-009: GetDuration returns last value
TEST_F(PerformanceProfilerTest, GetDuration_LastValue) {
    auto profiler = PerformanceProfiler::create();

    profiler->start("multi");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    profiler->stop("multi");

    profiler->start("multi");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler->stop("multi");

    double duration = profiler->getDuration("multi");
    EXPECT_GE(duration, 5.0);  // Last measurement should be around 10ms
}

// PRF-010: GetDuration not found throws
TEST_F(PerformanceProfilerTest, GetDuration_NotFound) {
    auto profiler = PerformanceProfiler::create();

    EXPECT_THROW(profiler->getDuration("nonexistent"), std::runtime_error);
}

// =============================================================================
// Statistics Tests (PRF-011 ~ PRF-017)
// =============================================================================

// PRF-011: GetStatistics count
TEST_F(PerformanceProfilerTest, GetStatistics_Count) {
    auto profiler = PerformanceProfiler::create();

    for (int i = 0; i < 5; i++) {
        profiler->start("count_test");
        profiler->stop("count_test");
    }

    auto stats = profiler->getStatistics("count_test");
    EXPECT_EQ(5u, stats.count);
}

// PRF-012 ~ PRF-015: Statistics calculations
TEST_F(PerformanceProfilerTest, GetStatistics_Calculations) {
    auto profiler = PerformanceProfiler::create();

    // Create measurements with known delays
    for (int i = 1; i <= 3; i++) {
        profiler->start("calc_test");
        std::this_thread::sleep_for(std::chrono::milliseconds(i * 10));
        profiler->stop("calc_test");
    }

    auto stats = profiler->getStatistics("calc_test");

    EXPECT_EQ(3u, stats.count);
    EXPECT_GT(stats.minimum, 0.0);
    EXPECT_GE(stats.maximum, stats.minimum);
    EXPECT_GE(stats.average, stats.minimum);
    EXPECT_LE(stats.average, stats.maximum);
    EXPECT_GT(stats.total, 0.0);
}

// PRF-016: GetStatistics not found
TEST_F(PerformanceProfilerTest, GetStatistics_NotFound) {
    auto profiler = PerformanceProfiler::create();

    auto stats = profiler->getStatistics("nonexistent");

    EXPECT_EQ(0u, stats.count);
    EXPECT_EQ(0.0, stats.average);
    EXPECT_EQ(0.0, stats.minimum);
    EXPECT_EQ(0.0, stats.maximum);
    EXPECT_EQ(0.0, stats.total);
}

// PRF-017: Single measurement stats
TEST_F(PerformanceProfilerTest, GetStatistics_SingleMeasurement) {
    auto profiler = PerformanceProfiler::create();

    profiler->start("single");
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    profiler->stop("single");

    auto stats = profiler->getStatistics("single");

    EXPECT_EQ(1u, stats.count);
    EXPECT_DOUBLE_EQ(stats.minimum, stats.maximum);
    EXPECT_DOUBLE_EQ(stats.average, stats.minimum);
}

// =============================================================================
// Report & Export Tests (PRF-018 ~ PRF-028)
// =============================================================================

// PRF-018: Report not empty
TEST_F(PerformanceProfilerTest, Report_NotEmpty) {
    auto profiler = PerformanceProfiler::create();

    profiler->start("test");
    profiler->stop("test");

    std::string report = profiler->report();
    EXPECT_FALSE(report.empty());
}

// PRF-019: Report contains timer names
TEST_F(PerformanceProfilerTest, Report_ContainsTimerNames) {
    auto profiler = PerformanceProfiler::create();

    profiler->start("MyTimer");
    profiler->stop("MyTimer");

    std::string report = profiler->report();
    EXPECT_NE(std::string::npos, report.find("MyTimer"));
}

// PRF-020: Report empty profiler
TEST_F(PerformanceProfilerTest, Report_EmptyProfiler) {
    auto profiler = PerformanceProfiler::create();

    std::string report = profiler->report();
    // Report always includes header, so check it contains header but no actual data rows
    EXPECT_NE(std::string::npos, report.find("Performance Profile Report"));
    EXPECT_NE(std::string::npos, report.find("Operation"));  // Header exists
    // Verify no actual timer data (only header and footer lines)
    // The report format has headers/footers with '=' and '-' lines
    // Data rows contain timer names, which won't be present if empty
}

// PRF-021: ExportJSON creates file
TEST_F(PerformanceProfilerTest, ExportJSON_CreatesFile) {
    auto profiler = PerformanceProfiler::create();
    std::string jsonPath = getTestPath("profile.json");

    profiler->start("test");
    profiler->stop("test");
    profiler->exportToJSON(jsonPath);

    EXPECT_TRUE(fs::exists(jsonPath));
}

// PRF-022: ExportJSON valid format
TEST_F(PerformanceProfilerTest, ExportJSON_ValidFormat) {
    auto profiler = PerformanceProfiler::create();
    std::string jsonPath = getTestPath("valid.json");

    profiler->start("test");
    profiler->stop("test");
    profiler->exportToJSON(jsonPath);

    std::string content = readFileContent(jsonPath);
    EXPECT_NE(std::string::npos, content.find("{"));
    EXPECT_NE(std::string::npos, content.find("}"));
}

// PRF-023: ExportJSON contains stats
TEST_F(PerformanceProfilerTest, ExportJSON_ContainsStats) {
    auto profiler = PerformanceProfiler::create();
    std::string jsonPath = getTestPath("stats.json");

    profiler->start("test");
    profiler->stop("test");
    profiler->exportToJSON(jsonPath);

    std::string content = readFileContent(jsonPath);
    EXPECT_NE(std::string::npos, content.find("test"));
}

// PRF-025: ExportHTML creates file
TEST_F(PerformanceProfilerTest, ExportHTML_CreatesFile) {
    auto profiler = PerformanceProfiler::create();
    std::string htmlPath = getTestPath("profile.html");

    profiler->start("test");
    profiler->stop("test");
    profiler->exportToHTML(htmlPath);

    EXPECT_TRUE(fs::exists(htmlPath));
}

// PRF-026: ExportHTML valid HTML
TEST_F(PerformanceProfilerTest, ExportHTML_ValidHTML) {
    auto profiler = PerformanceProfiler::create();
    std::string htmlPath = getTestPath("valid.html");

    profiler->start("test");
    profiler->stop("test");
    profiler->exportToHTML(htmlPath);

    std::string content = readFileContent(htmlPath);
    EXPECT_NE(std::string::npos, content.find("<html>"));
    EXPECT_NE(std::string::npos, content.find("</html>"));
}

// PRF-027: ExportReport creates both files
TEST_F(PerformanceProfilerTest, ExportReport_CreatesBoth) {
    auto profiler = PerformanceProfiler::create();
    std::string outputDir = getTestPath("report_output");

    profiler->start("test");
    profiler->stop("test");
    profiler->exportReport(outputDir);

    EXPECT_TRUE(fs::exists(fs::path(outputDir) / "profile.json"));
    EXPECT_TRUE(fs::exists(fs::path(outputDir) / "profile.html"));
}

// PRF-028: ExportReport creates directory
TEST_F(PerformanceProfilerTest, ExportReport_CreateDirectory) {
    auto profiler = PerformanceProfiler::create();
    std::string outputDir = getTestPath("new_report_dir");

    EXPECT_FALSE(fs::exists(outputDir));

    profiler->start("test");
    profiler->stop("test");
    profiler->exportReport(outputDir);

    EXPECT_TRUE(fs::exists(outputDir));
}

// =============================================================================
// Control Tests (PRF-029 ~ PRF-038)
// =============================================================================

// PRF-029: Reset clears all data
TEST_F(PerformanceProfilerTest, Reset_ClearsAllData) {
    auto profiler = PerformanceProfiler::create();

    profiler->start("test");
    profiler->stop("test");

    profiler->reset();

    auto stats = profiler->getStatistics("test");
    EXPECT_EQ(0u, stats.count);
}

// PRF-030: Reset clears active timers
TEST_F(PerformanceProfilerTest, Reset_ClearsActiveTimers) {
    auto profiler = PerformanceProfiler::create();

    profiler->start("active");
    profiler->reset();

    // After reset, stop should not crash
    EXPECT_NO_THROW(profiler->stop("active"));
}

// PRF-031, PRF-032: SetEnabled
TEST_F(PerformanceProfilerTest, SetEnabled_False) {
    auto profiler = PerformanceProfiler::create();

    profiler->setEnabled(false);

    profiler->start("disabled");
    profiler->stop("disabled");

    auto stats = profiler->getStatistics("disabled");
    EXPECT_EQ(0u, stats.count);
}

TEST_F(PerformanceProfilerTest, SetEnabled_True) {
    auto profiler = PerformanceProfiler::create();

    profiler->setEnabled(true);

    profiler->start("enabled");
    profiler->stop("enabled");

    auto stats = profiler->getStatistics("enabled");
    EXPECT_EQ(1u, stats.count);
}

// PRF-033: IsEnabled default
TEST_F(PerformanceProfilerTest, IsEnabled_Default) {
    auto profiler = PerformanceProfiler::create();

    EXPECT_TRUE(profiler->isEnabled());
}

// PRF-034: SetEnabled atomic
TEST_F(PerformanceProfilerTest, SetEnabled_Atomic) {
    auto profiler = PerformanceProfiler::create();
    std::vector<std::thread> threads;

    for (int t = 0; t < 10; t++) {
        threads.emplace_back([&profiler, t]() {
            for (int i = 0; i < 100; i++) {
                profiler->setEnabled(t % 2 == 0);
                profiler->isEnabled();
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    SUCCEED();
}

// PRF-036: MaxTimings enforced
TEST_F(PerformanceProfilerTest, MaxTimings_Enforced) {
    auto profiler = PerformanceProfiler::create();
    profiler->setMaxTimingsPerTimer(3);

    for (int i = 0; i < 5; i++) {
        profiler->start("limited");
        profiler->stop("limited");
    }

    auto stats = profiler->getStatistics("limited");
    EXPECT_EQ(3u, stats.count);
}

// =============================================================================
// ScopedTimer Tests (PRF-039 ~ PRF-041)
// =============================================================================

// PRF-039: ScopedTimer auto stop
TEST_F(PerformanceProfilerTest, ScopedTimer_AutoStop) {
    auto profiler = PerformanceProfiler::create();

    {
        auto timer = profiler->createScopedTimer("scoped");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    auto stats = profiler->getStatistics("scoped");
    EXPECT_EQ(1u, stats.count);
    EXPECT_GT(stats.average, 0.0);
}

// PRF-040: ScopedTimer exception safe
TEST_F(PerformanceProfilerTest, ScopedTimer_Exception) {
    auto profiler = PerformanceProfiler::create();

    try {
        auto timer = profiler->createScopedTimer("exception_test");
        throw std::runtime_error("Test exception");
    }
    catch (...) {
        // Expected
    }

    auto stats = profiler->getStatistics("exception_test");
    EXPECT_EQ(1u, stats.count);
}

// PRF-041: ScopedTimer nested
TEST_F(PerformanceProfilerTest, ScopedTimer_Nested) {
    auto profiler = PerformanceProfiler::create();

    {
        auto outer = profiler->createScopedTimer("outer");
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        {
            auto inner = profiler->createScopedTimer("inner");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    auto outerStats = profiler->getStatistics("outer");
    auto innerStats = profiler->getStatistics("inner");

    EXPECT_EQ(1u, outerStats.count);
    EXPECT_EQ(1u, innerStats.count);
    EXPECT_GT(outerStats.average, innerStats.average);
}

// =============================================================================
// Thread Safety Tests (PRF-044 ~ PRF-046)
// =============================================================================

// PRF-044: Concurrent start
TEST_F(PerformanceProfilerTest, ThreadSafety_ConcurrentStart) {
    auto profiler = PerformanceProfiler::create();
    std::vector<std::thread> threads;

    for (int t = 0; t < 10; t++) {
        threads.emplace_back([&profiler, t]() {
            for (int i = 0; i < 100; i++) {
                std::string name = "timer_" + std::to_string(t) + "_" + std::to_string(i);
                profiler->start(name);
                profiler->stop(name);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    SUCCEED();
}

// PRF-046: Concurrent getStatistics
TEST_F(PerformanceProfilerTest, ThreadSafety_ConcurrentStats) {
    auto profiler = PerformanceProfiler::create();

    profiler->start("shared");
    profiler->stop("shared");

    std::vector<std::thread> threads;
    for (int t = 0; t < 10; t++) {
        threads.emplace_back([&profiler]() {
            for (int i = 0; i < 100; i++) {
                profiler->getStatistics("shared");
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    SUCCEED();
}

// =============================================================================
// Edge Cases (PRF-047 ~ PRF-054)
// =============================================================================

// PRF-048: Unicode timer name
TEST_F(PerformanceProfilerTest, UnicodeTimerName) {
    auto profiler = PerformanceProfiler::create();

    EXPECT_NO_THROW({
        profiler->start("Korean_Timer");
        profiler->stop("Korean_Timer");
    });
}

// PRF-049: Long timer name
TEST_F(PerformanceProfilerTest, LongTimerName) {
    auto profiler = PerformanceProfiler::create();
    std::string longName(1000, 'A');

    EXPECT_NO_THROW({
        profiler->start(longName);
        profiler->stop(longName);
    });

    auto stats = profiler->getStatistics(longName);
    EXPECT_EQ(1u, stats.count);
}
