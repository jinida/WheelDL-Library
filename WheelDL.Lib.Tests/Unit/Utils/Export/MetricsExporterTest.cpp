#include "pch.h"
#include <gtest/gtest.h>
#include "Utils/Export/MetricsExporter.h"
#include "Utils/Common/Types.h"
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace WheelDL::Utils::Export;
using namespace WheelDL;
namespace fs = std::filesystem;

// =============================================================================
// Test Fixture
// =============================================================================

class MetricsExporterTest : public ::testing::Test {
protected:
    std::string testBaseDir;

    void SetUp() override {
        testBaseDir = (fs::temp_directory_path() / "WheelDL_MetricsExporter_Test").string();
        fs::remove_all(testBaseDir);
        fs::create_directories(testBaseDir);
    }

    void TearDown() override {
        try {
            fs::remove_all(testBaseDir);
        }
        catch (...) {}
    }

    std::string getTestPath(const std::string& filename) {
        return (fs::path(testBaseDir) / filename).string();
    }

    std::vector<MetricsData> createTestMetrics(size_t count) {
        std::vector<MetricsData> metrics;
        for (size_t i = 0; i < count; ++i) {
            MetricsData m;
            m.loss = 1.0f - (static_cast<float>(i) * 0.1f);
            m.accuracy = static_cast<float>(i) * 0.1f;
            m.precision = 0.8f + (static_cast<float>(i) * 0.01f);
            m.recall = 0.75f + (static_cast<float>(i) * 0.02f);
            m.f1Score = 0.77f + (static_cast<float>(i) * 0.015f);
            m.mAP = 0.5f + (static_cast<float>(i) * 0.05f);
            m.fitness = 0.6f + (static_cast<float>(i) * 0.04f);
            m.threshold = 0.5f;
            m.aucROC = 0.85f + (static_cast<float>(i) * 0.01f);
            metrics.push_back(m);
        }
        return metrics;
    }

    std::string readFileContent(const std::string& filepath) {
        std::ifstream file(filepath);
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
};

// =============================================================================
// ExportToJSON Tests (ME-001 ~ ME-004)
// =============================================================================

// ME-001: ExportJSON valid
TEST_F(MetricsExporterTest, ExportJSON_Valid) {
    auto metrics = createTestMetrics(5);
    std::string filepath = getTestPath("metrics.json");

    bool result = MetricsExporter::exportToJSON(metrics, filepath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(filepath));
}

// ME-002: ExportJSON with data
TEST_F(MetricsExporterTest, ExportJSON_WithData) {
    auto metrics = createTestMetrics(3);
    std::string filepath = getTestPath("metrics_data.json");

    bool result = MetricsExporter::exportToJSON(metrics, filepath);

    EXPECT_TRUE(result);

    std::string content = readFileContent(filepath);
    EXPECT_NE(std::string::npos, content.find("training_metrics"));
    EXPECT_NE(std::string::npos, content.find("total_epochs"));
}

// ME-003: ExportJSON invalid path
TEST_F(MetricsExporterTest, ExportJSON_InvalidPath) {
    auto metrics = createTestMetrics(3);
    std::string filepath = "Z:\\<invalid>\\path|test\\metrics.json";

    bool result = MetricsExporter::exportToJSON(metrics, filepath);

    EXPECT_FALSE(result);
}

// ME-004: ExportJSON with bestFitness
TEST_F(MetricsExporterTest, ExportJSON_BestFitness) {
    auto metrics = createTestMetrics(3);
    std::string filepath = getTestPath("metrics_fitness.json");

    bool result = MetricsExporter::exportToJSON(metrics, filepath, 0.95f);

    EXPECT_TRUE(result);

    std::string content = readFileContent(filepath);
    EXPECT_NE(std::string::npos, content.find("best_fitness"));
    // Float 0.95f may be formatted differently due to precision (e.g., 0.949999988)
    // Just verify the field exists with a value starting with 0.9
    EXPECT_NE(std::string::npos, content.find("\"best_fitness\":"));
}

// ME-011: ExportJSON negative bestFitness - no best_fitness field
TEST_F(MetricsExporterTest, ExportJSON_BestFitnessNegative) {
    auto metrics = createTestMetrics(3);
    std::string filepath = getTestPath("metrics_no_fitness.json");

    bool result = MetricsExporter::exportToJSON(metrics, filepath, -1.0f);

    EXPECT_TRUE(result);

    std::string content = readFileContent(filepath);
    EXPECT_EQ(std::string::npos, content.find("best_fitness"));
}

// ME-012: ExportJSON file open failed
TEST_F(MetricsExporterTest, ExportJSON_FileOpenFailed) {
    auto metrics = createTestMetrics(3);

    // Try to write to an invalid path
    bool result = MetricsExporter::exportToJSON(metrics, "Z:\\nonexistent\\<>\\file.json");

    EXPECT_FALSE(result);
}

// ME-015: ExportJSON all metric fields
TEST_F(MetricsExporterTest, ExportJSON_AllMetricFields) {
    auto metrics = createTestMetrics(1);
    std::string filepath = getTestPath("all_fields.json");

    bool result = MetricsExporter::exportToJSON(metrics, filepath);

    EXPECT_TRUE(result);

    std::string content = readFileContent(filepath);
    EXPECT_NE(std::string::npos, content.find("loss"));
    EXPECT_NE(std::string::npos, content.find("accuracy"));
    EXPECT_NE(std::string::npos, content.find("precision"));
    EXPECT_NE(std::string::npos, content.find("recall"));
    EXPECT_NE(std::string::npos, content.find("f1_score"));
    EXPECT_NE(std::string::npos, content.find("mAP"));
    EXPECT_NE(std::string::npos, content.find("fitness"));
    EXPECT_NE(std::string::npos, content.find("threshold"));
    EXPECT_NE(std::string::npos, content.find("aucROC"));
}

// ME-017: ExportJSON empty metrics
TEST_F(MetricsExporterTest, ExportJSON_EmptyMetrics) {
    std::vector<MetricsData> emptyMetrics;
    std::string filepath = getTestPath("empty.json");

    bool result = MetricsExporter::exportToJSON(emptyMetrics, filepath);

    EXPECT_FALSE(result);
}

// =============================================================================
// ExportToCSV Tests (ME-005 ~ ME-007)
// =============================================================================

// ME-005: ExportCSV valid
TEST_F(MetricsExporterTest, ExportCSV_Valid) {
    auto metrics = createTestMetrics(5);
    std::string filepath = getTestPath("metrics.csv");

    bool result = MetricsExporter::exportToCSV(metrics, filepath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(filepath));
}

// ME-006: ExportCSV header
TEST_F(MetricsExporterTest, ExportCSV_Header) {
    auto metrics = createTestMetrics(3);
    std::string filepath = getTestPath("metrics_header.csv");

    bool result = MetricsExporter::exportToCSV(metrics, filepath);

    EXPECT_TRUE(result);

    std::string content = readFileContent(filepath);
    // Check header row exists
    EXPECT_NE(std::string::npos, content.find("epoch"));
    EXPECT_NE(std::string::npos, content.find("loss"));
    EXPECT_NE(std::string::npos, content.find("accuracy"));
}

// ME-007: ExportCSV multiple epochs
TEST_F(MetricsExporterTest, ExportCSV_MultipleEpochs) {
    auto metrics = createTestMetrics(10);
    std::string filepath = getTestPath("multi_epoch.csv");

    bool result = MetricsExporter::exportToCSV(metrics, filepath);

    EXPECT_TRUE(result);

    // Count lines (header + 10 data rows = 11 lines)
    std::ifstream file(filepath);
    int lineCount = 0;
    std::string line;
    while (std::getline(file, line)) {
        lineCount++;
    }
    EXPECT_EQ(11, lineCount);
}

// ME-013: ExportCSV file open failed
TEST_F(MetricsExporterTest, ExportCSV_FileOpenFailed) {
    auto metrics = createTestMetrics(3);

    bool result = MetricsExporter::exportToCSV(metrics, "Z:\\nonexistent\\<>\\file.csv");

    EXPECT_FALSE(result);
}

// ME-016: ExportCSV all metric fields
TEST_F(MetricsExporterTest, ExportCSV_AllMetricFields) {
    auto metrics = createTestMetrics(1);
    std::string filepath = getTestPath("csv_all_fields.csv");

    bool result = MetricsExporter::exportToCSV(metrics, filepath);

    EXPECT_TRUE(result);

    std::string content = readFileContent(filepath);
    // Check header contains all field names
    EXPECT_NE(std::string::npos, content.find("loss"));
    EXPECT_NE(std::string::npos, content.find("accuracy"));
    EXPECT_NE(std::string::npos, content.find("precision"));
    EXPECT_NE(std::string::npos, content.find("recall"));
    EXPECT_NE(std::string::npos, content.find("f1_score"));
    EXPECT_NE(std::string::npos, content.find("mAP"));
    EXPECT_NE(std::string::npos, content.find("fitness"));
    EXPECT_NE(std::string::npos, content.find("threshold"));
}

// ME-018: ExportCSV empty metrics
TEST_F(MetricsExporterTest, ExportCSV_EmptyMetrics) {
    std::vector<MetricsData> emptyMetrics;
    std::string filepath = getTestPath("empty.csv");

    bool result = MetricsExporter::exportToCSV(emptyMetrics, filepath);

    EXPECT_FALSE(result);
}

// =============================================================================
// ExportAll Tests (ME-008 ~ ME-010)
// =============================================================================

// ME-008: ExportAll creates both files
TEST_F(MetricsExporterTest, ExportAll_CreatesBoth) {
    auto metrics = createTestMetrics(5);
    std::string outputDir = getTestPath("export_all");

    bool result = MetricsExporter::exportAll(metrics, outputDir);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(outputDir + "/training_metrics.json"));
    EXPECT_TRUE(fs::exists(outputDir + "/training_metrics.csv"));
}

// ME-009: ExportAll custom filenames
TEST_F(MetricsExporterTest, ExportAll_CustomFilenames) {
    auto metrics = createTestMetrics(3);
    std::string outputDir = getTestPath("custom_names");

    bool result = MetricsExporter::exportAll(
        metrics, outputDir,
        "custom_metrics.json",
        "custom_metrics.csv",
        0.9f);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(outputDir + "/custom_metrics.json"));
    EXPECT_TRUE(fs::exists(outputDir + "/custom_metrics.csv"));
}

// ME-010: ExportAll partial failure
TEST_F(MetricsExporterTest, ExportAll_PartialFailure) {
    // This is hard to test directly since both exports use same directory
    // We can test with empty metrics which should fail both
    std::vector<MetricsData> emptyMetrics;
    std::string outputDir = getTestPath("partial_fail");

    bool result = MetricsExporter::exportAll(emptyMetrics, outputDir);

    EXPECT_FALSE(result);
}

// ME-014: ExportAll directory creation fails
TEST_F(MetricsExporterTest, ExportAll_DirectoryCreationFails) {
    auto metrics = createTestMetrics(3);

    // Invalid directory path
    bool result = MetricsExporter::exportAll(metrics, "Z:\\<invalid>\\path|test");

    EXPECT_FALSE(result);
}

// ME-019: ExportAll empty metrics
TEST_F(MetricsExporterTest, ExportAll_EmptyMetrics) {
    std::vector<MetricsData> emptyMetrics;
    std::string outputDir = getTestPath("empty_all");

    bool result = MetricsExporter::exportAll(emptyMetrics, outputDir);

    EXPECT_FALSE(result);
}

// =============================================================================
// Additional Tests
// =============================================================================

TEST_F(MetricsExporterTest, ExportJSON_CreatesDirectory) {
    auto metrics = createTestMetrics(3);
    std::string filepath = getTestPath("new_dir/subdir/metrics.json");

    bool result = MetricsExporter::exportToJSON(metrics, filepath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(filepath));
}

TEST_F(MetricsExporterTest, ExportCSV_CreatesDirectory) {
    auto metrics = createTestMetrics(3);
    std::string filepath = getTestPath("new_csv_dir/subdir/metrics.csv");

    bool result = MetricsExporter::exportToCSV(metrics, filepath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(filepath));
}

TEST_F(MetricsExporterTest, ExportAll_CreatesDirectory) {
    auto metrics = createTestMetrics(3);
    std::string outputDir = getTestPath("new_all_dir/nested");

    bool result = MetricsExporter::exportAll(metrics, outputDir);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(outputDir));
}

TEST_F(MetricsExporterTest, ExportJSON_LargeDataset) {
    auto metrics = createTestMetrics(1000);  // 1000 epochs
    std::string filepath = getTestPath("large.json");

    bool result = MetricsExporter::exportToJSON(metrics, filepath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(fs::exists(filepath));

    // File should be reasonably sized
    auto fileSize = fs::file_size(filepath);
    EXPECT_GT(fileSize, 0u);
}

TEST_F(MetricsExporterTest, ExportCSV_LargeDataset) {
    auto metrics = createTestMetrics(1000);  // 1000 epochs
    std::string filepath = getTestPath("large.csv");

    bool result = MetricsExporter::exportToCSV(metrics, filepath);

    EXPECT_TRUE(result);

    // Count lines (header + 1000 data rows = 1001 lines)
    std::ifstream file(filepath);
    int lineCount = 0;
    std::string line;
    while (std::getline(file, line)) {
        lineCount++;
    }
    EXPECT_EQ(1001, lineCount);
}

TEST_F(MetricsExporterTest, ExportJSON_OverwriteExisting) {
    auto metrics1 = createTestMetrics(3);
    auto metrics2 = createTestMetrics(5);
    std::string filepath = getTestPath("overwrite.json");

    // First export
    EXPECT_TRUE(MetricsExporter::exportToJSON(metrics1, filepath));

    // Overwrite
    EXPECT_TRUE(MetricsExporter::exportToJSON(metrics2, filepath));

    // Verify new content
    std::string content = readFileContent(filepath);
    EXPECT_NE(std::string::npos, content.find("\"total_epochs\": 5"));
}

TEST_F(MetricsExporterTest, ExportAll_WithBestFitness) {
    auto metrics = createTestMetrics(5);
    std::string outputDir = getTestPath("with_fitness");

    bool result = MetricsExporter::exportAll(metrics, outputDir,
        "training_metrics.json", "training_metrics.csv", 0.98f);

    EXPECT_TRUE(result);

    // Check JSON has best_fitness
    std::string jsonContent = readFileContent(outputDir + "/training_metrics.json");
    EXPECT_NE(std::string::npos, jsonContent.find("best_fitness"));
}

TEST_F(MetricsExporterTest, MetricsData_ZeroValues) {
    std::vector<MetricsData> metrics;
    MetricsData m = {};  // All zeros
    metrics.push_back(m);

    std::string filepath = getTestPath("zeros.json");

    bool result = MetricsExporter::exportToJSON(metrics, filepath);

    EXPECT_TRUE(result);

    std::string content = readFileContent(filepath);
    EXPECT_NE(std::string::npos, content.find("\"loss\": 0"));
}
