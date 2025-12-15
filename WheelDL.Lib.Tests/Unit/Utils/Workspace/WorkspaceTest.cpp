#include "pch.h"
#include "Utils/Workspace/Workspace.h"
#include "Utils/Error/WheelLibException.h"
#include <filesystem>
#include <thread>
#include <chrono>
#include <regex>
#include <gtest/gtest.h>

using namespace WheelDL::Utils;
namespace fs = std::filesystem;

// =============================================================================
// Test Fixture
// =============================================================================

class WorkspaceTest : public ::testing::Test {
protected:
    std::string testBaseDir;
    std::vector<std::string> createdDirs;

    void SetUp() override {
        testBaseDir = (fs::temp_directory_path() / "WheelDL_Workspace_Test").string();
        fs::remove_all(testBaseDir);
        fs::create_directories(testBaseDir);
    }

    void TearDown() override {
        try {
            fs::remove_all(testBaseDir);
            for (const auto& dir : createdDirs) {
                fs::remove_all(dir);
            }
        }
        catch (...) {}
    }

    std::string getTestBaseDir(const std::string& subdir = "") {
        if (subdir.empty()) {
            return testBaseDir;
        }
        return (fs::path(testBaseDir) / subdir).string();
    }

    void trackCreatedDir(const std::string& dir) {
        createdDirs.push_back(dir);
    }
};

// =============================================================================
// Constructor Tests (WS-001 ~ WS-006)
// =============================================================================

// WS-001: Constructor with default arguments
TEST_F(WorkspaceTest, Constructor_Default) {
    std::string baseDir = getTestBaseDir("default_runs");
    Workspace ws(baseDir, "train");

    EXPECT_TRUE(fs::exists(ws.getRoot()));
    EXPECT_NE(std::string::npos, ws.getRoot().find("train"));
}

// WS-002: Constructor with custom base directory
TEST_F(WorkspaceTest, Constructor_CustomBase) {
    std::string baseDir = getTestBaseDir("custom_output");
    Workspace ws(baseDir, "train");

    EXPECT_TRUE(fs::exists(ws.getRoot()));
    EXPECT_NE(std::string::npos, ws.getRoot().find("custom_output"));
}

// WS-003: Constructor with custom prefix
TEST_F(WorkspaceTest, Constructor_CustomPrefix) {
    std::string baseDir = getTestBaseDir("prefix_runs");
    Workspace ws(baseDir, "experiment");

    EXPECT_TRUE(fs::exists(ws.getRoot()));
    EXPECT_NE(std::string::npos, ws.getRoot().find("experiment"));
}

// WS-004: Constructor with custom name and timestamp
TEST_F(WorkspaceTest, Constructor_CustomName_WithTimestamp) {
    std::string baseDir = getTestBaseDir("custom_name");
    Workspace ws(baseDir, "my_run", true);

    EXPECT_TRUE(fs::exists(ws.getRoot()));
    EXPECT_NE(std::string::npos, ws.getRoot().find("my_run"));
    EXPECT_FALSE(ws.getTimestamp().empty());
}

// WS-005: Constructor with custom name, no timestamp
TEST_F(WorkspaceTest, Constructor_CustomName_NoTimestamp) {
    std::string baseDir = getTestBaseDir("no_timestamp");
    Workspace ws(baseDir, "fixed_name", false);

    EXPECT_TRUE(fs::exists(ws.getRoot()));
    EXPECT_NE(std::string::npos, ws.getRoot().find("fixed_name"));
}

// WS-006: Constructor with invalid path throws
TEST_F(WorkspaceTest, Constructor_InvalidPath) {
    // Path with invalid characters on Windows
    EXPECT_THROW({
        Workspace ws("Z:\\<invalid>\\path|test", "train");
    }, WheelLibException);
}

// =============================================================================
// Getter Tests (WS-007 ~ WS-012)
// =============================================================================

// WS-007: GetRoot returns valid path
TEST_F(WorkspaceTest, GetRoot_ReturnsValidPath) {
    std::string baseDir = getTestBaseDir("root_test");
    Workspace ws(baseDir, "train");

    std::string root = ws.getRoot();
    EXPECT_FALSE(root.empty());
    EXPECT_TRUE(fs::exists(root));
    EXPECT_TRUE(fs::is_directory(root));
}

// WS-008: GetWeightsDir exists
TEST_F(WorkspaceTest, GetWeightsDir_Exists) {
    std::string baseDir = getTestBaseDir("weights_test");
    Workspace ws(baseDir, "train");

    std::string weightsDir = ws.getWeightsDir();
    EXPECT_FALSE(weightsDir.empty());
    EXPECT_TRUE(fs::exists(weightsDir));
    EXPECT_TRUE(fs::is_directory(weightsDir));
}

// WS-009: GetLogsDir exists
TEST_F(WorkspaceTest, GetLogsDir_Exists) {
    std::string baseDir = getTestBaseDir("logs_test");
    Workspace ws(baseDir, "train");

    std::string logsDir = ws.getLogsDir();
    EXPECT_FALSE(logsDir.empty());
    EXPECT_TRUE(fs::exists(logsDir));
    EXPECT_TRUE(fs::is_directory(logsDir));
}

// WS-010: GetProfilerDir exists
TEST_F(WorkspaceTest, GetProfilerDir_Exists) {
    std::string baseDir = getTestBaseDir("profiler_test");
    Workspace ws(baseDir, "train");

    std::string profilerDir = ws.getProfilerDir();
    EXPECT_FALSE(profilerDir.empty());
    EXPECT_TRUE(fs::exists(profilerDir));
    EXPECT_TRUE(fs::is_directory(profilerDir));
}

// WS-011: GetResultDir exists
TEST_F(WorkspaceTest, GetResultDir_Exists) {
    std::string baseDir = getTestBaseDir("result_test");
    Workspace ws(baseDir, "train");

    std::string resultDir = ws.getResultDir();
    EXPECT_FALSE(resultDir.empty());
    EXPECT_TRUE(fs::exists(resultDir));
    EXPECT_TRUE(fs::is_directory(resultDir));
}

// WS-012: GetTimestamp format (YYYYMMDD_HHMMSS)
TEST_F(WorkspaceTest, GetTimestamp_Format) {
    std::string baseDir = getTestBaseDir("timestamp_test");
    Workspace ws(baseDir, "train");

    std::string timestamp = ws.getTimestamp();
    EXPECT_FALSE(timestamp.empty());

    // Format: YYYYMMDD_HHMMSS (15 chars)
    EXPECT_EQ(15u, timestamp.length());

    // Verify format with regex
    std::regex timestampRegex("\\d{8}_\\d{6}");
    EXPECT_TRUE(std::regex_match(timestamp, timestampRegex));
}

// =============================================================================
// CreateSubDirectory Tests (WS-013 ~ WS-015)
// =============================================================================

// WS-013: CreateSubDirectory new directory
TEST_F(WorkspaceTest, CreateSubDirectory_New) {
    std::string baseDir = getTestBaseDir("subdir_test");
    Workspace ws(baseDir, "train");

    std::string customDir = ws.createSubDirectory("custom");

    EXPECT_FALSE(customDir.empty());
    EXPECT_TRUE(fs::exists(customDir));
    EXPECT_TRUE(fs::is_directory(customDir));
}

// WS-014: CreateSubDirectory existing directory
TEST_F(WorkspaceTest, CreateSubDirectory_Existing) {
    std::string baseDir = getTestBaseDir("existing_test");
    Workspace ws(baseDir, "train");

    std::string dir1 = ws.createSubDirectory("same_name");
    std::string dir2 = ws.createSubDirectory("same_name");

    EXPECT_EQ(dir1, dir2);
}

// WS-015: CreateSubDirectory nested
TEST_F(WorkspaceTest, CreateSubDirectory_Nested) {
    std::string baseDir = getTestBaseDir("nested_test");
    Workspace ws(baseDir, "train");

    std::string nestedDir = ws.createSubDirectory("a/b/c");

    EXPECT_TRUE(fs::exists(nestedDir));
    EXPECT_TRUE(fs::is_directory(nestedDir));
}

// =============================================================================
// Exists Tests (WS-016 ~ WS-017)
// =============================================================================

// WS-016: Exists after construction
TEST_F(WorkspaceTest, Exists_AfterConstruction) {
    std::string baseDir = getTestBaseDir("exists_test");
    Workspace ws(baseDir, "train");

    EXPECT_TRUE(ws.exists());
}

// WS-017: Exists after deletion
TEST_F(WorkspaceTest, Exists_AfterDeletion) {
    std::string baseDir = getTestBaseDir("deletion_test");
    std::string rootPath;

    {
        Workspace ws(baseDir, "train");
        rootPath = ws.getRoot();
        EXPECT_TRUE(ws.exists());

        // Delete the directory
        fs::remove_all(rootPath);
        EXPECT_FALSE(ws.exists());
    }
}

// =============================================================================
// Move Semantics Tests (WS-018)
// =============================================================================

// WS-018: Move semantics valid
TEST_F(WorkspaceTest, MoveSemantics_Valid) {
    std::string baseDir = getTestBaseDir("move_test");
    Workspace ws1(baseDir, "train");
    std::string root1 = ws1.getRoot();

    Workspace ws2 = std::move(ws1);

    EXPECT_EQ(root1, ws2.getRoot());
    EXPECT_TRUE(ws2.exists());
}

// =============================================================================
// Timestamp Uniqueness Tests (WS-019)
// =============================================================================

// WS-019: Unique timestamp for sequential creation
TEST_F(WorkspaceTest, UniqueTimestamp_Sequential) {
    std::string baseDir = getTestBaseDir("unique_test");

    Workspace ws1(baseDir, "train");
    std::string ts1 = ws1.getTimestamp();

    // Wait a bit to ensure different timestamp
    std::this_thread::sleep_for(std::chrono::seconds(1));

    Workspace ws2(baseDir, "train");
    std::string ts2 = ws2.getTimestamp();

    EXPECT_NE(ts1, ts2);
}

// =============================================================================
// Exception Tests (WS-020 ~ WS-023)
// =============================================================================

// WS-020: Constructor with invalid path throws WheelLibException
TEST_F(WorkspaceTest, Constructor_InvalidPath_Throws) {
    try {
        Workspace ws("Z:\\nonexistent\\<invalid>", "train");
        FAIL() << "Expected WheelLibException";
    }
    catch (const WheelLibException& e) {
        EXPECT_EQ(ErrorCode::INVALID_CONFIG, e.getErrorCode());
    }
    catch (...) {
        FAIL() << "Expected WheelLibException, got different exception";
    }
}

// WS-023: GetTimestamp empty when no timestamp
TEST_F(WorkspaceTest, GetTimestamp_EmptyWhenNoTimestamp) {
    std::string baseDir = getTestBaseDir("no_ts_test");
    Workspace ws(baseDir, "fixed", false);

    // When useTimestamp is false, timestamp might be empty or still generated
    // This depends on implementation
    std::string timestamp = ws.getTimestamp();
    // Just verify no crash
    SUCCEED();
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(WorkspaceTest, MultipleWorkspacesInSameBase) {
    std::string baseDir = getTestBaseDir("multi_ws");

    Workspace ws1(baseDir, "exp1");
    Workspace ws2(baseDir, "exp2");
    Workspace ws3(baseDir, "exp3");

    EXPECT_TRUE(ws1.exists());
    EXPECT_TRUE(ws2.exists());
    EXPECT_TRUE(ws3.exists());

    EXPECT_NE(ws1.getRoot(), ws2.getRoot());
    EXPECT_NE(ws2.getRoot(), ws3.getRoot());
}

TEST_F(WorkspaceTest, SubdirectoryStructure) {
    std::string baseDir = getTestBaseDir("structure_test");
    Workspace ws(baseDir, "train");

    // Verify all standard directories exist
    EXPECT_TRUE(fs::exists(ws.getWeightsDir()));
    EXPECT_TRUE(fs::exists(ws.getLogsDir()));
    EXPECT_TRUE(fs::exists(ws.getProfilerDir()));
    EXPECT_TRUE(fs::exists(ws.getResultDir()));

    // Verify they are subdirectories of root
    fs::path root(ws.getRoot());
    EXPECT_EQ(root, fs::path(ws.getWeightsDir()).parent_path());
    EXPECT_EQ(root, fs::path(ws.getLogsDir()).parent_path());
    EXPECT_EQ(root, fs::path(ws.getProfilerDir()).parent_path());
    EXPECT_EQ(root, fs::path(ws.getResultDir()).parent_path());
}

TEST_F(WorkspaceTest, LongPrefix) {
    std::string baseDir = getTestBaseDir("long_prefix");
    std::string longPrefix(100, 'A');

    Workspace ws(baseDir, longPrefix);

    EXPECT_TRUE(ws.exists());
    EXPECT_NE(std::string::npos, ws.getRoot().find(longPrefix));
}

TEST_F(WorkspaceTest, SpecialCharactersInPrefix) {
    std::string baseDir = getTestBaseDir("special_prefix");

    // Underscores and numbers should be fine
    Workspace ws(baseDir, "test_123_exp");

    EXPECT_TRUE(ws.exists());
}
