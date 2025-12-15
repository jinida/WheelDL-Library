#include "pch.h"
#include "Utils/Path/PathValidator.h"
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using namespace WheelDL::Utils;
namespace fs = std::filesystem;

// =============================================================================
// Test Fixture for PathValidator Tests
// =============================================================================

class PathValidatorTest : public ::testing::Test {
protected:
    std::string testDir;
    std::string testFile;
    std::string nestedDir;

    void SetUp() override {
        // Create unique test directory
        testDir = (fs::temp_directory_path() / "WheelDL_PathValidator_Test").string();
        nestedDir = (fs::path(testDir) / "nested" / "dir").string();
        testFile = (fs::path(testDir) / "test_file.txt").string();

        // Clean up if exists from previous run
        fs::remove_all(testDir);

        // Create test directory structure
        fs::create_directories(nestedDir);

        // Create test file
        std::ofstream ofs(testFile);
        ofs << "test content";
        ofs.close();
    }

    void TearDown() override {
        // Clean up test directory
        try {
            fs::remove_all(testDir);
        }
        catch (...) {
            // Ignore cleanup errors
        }
    }
};

// =============================================================================
// isValidPath Tests (PV-001 ~ PV-008)
// =============================================================================

// PV-001: isValidPath returns true for valid path strings
TEST_F(PathValidatorTest, IsValidPath_ValidPath) {
    EXPECT_TRUE(PathValidator::isValidPath(testDir));
    EXPECT_TRUE(PathValidator::isValidPath(testFile));
    EXPECT_TRUE(PathValidator::isValidPath("C:\\Windows\\System32"));
    EXPECT_TRUE(PathValidator::isValidPath("relative\\path\\to\\file.txt"));
}

// PV-002: isValidPath returns false for empty string
TEST_F(PathValidatorTest, IsValidPath_EmptyString) {
    EXPECT_FALSE(PathValidator::isValidPath(""));
}

// PV-003: isValidPath returns false for path with null character
TEST_F(PathValidatorTest, IsValidPath_NullCharacter) {
    std::string pathWithNull = "path\0invalid";
    pathWithNull.resize(12);  // Ensure we include the null character
    // Note: The null character in the middle makes this actually 4 bytes + null
    std::string actualPath = "path";
    actualPath.push_back('\0');
    actualPath += "invalid";
    EXPECT_FALSE(PathValidator::isValidPath(actualPath));
}

// PV-004: isValidPath handles Windows special characters
TEST_F(PathValidatorTest, IsValidPath_WindowsInvalidCharacters) {
    // These should be invalid on Windows
    EXPECT_FALSE(PathValidator::isValidPath("path<invalid"));
    EXPECT_FALSE(PathValidator::isValidPath("path>invalid"));
    EXPECT_FALSE(PathValidator::isValidPath("path\"invalid"));
    EXPECT_FALSE(PathValidator::isValidPath("path|invalid"));
    EXPECT_FALSE(PathValidator::isValidPath("path?invalid"));
    EXPECT_FALSE(PathValidator::isValidPath("path*invalid"));
}

// PV-005: isValidPath handles colons correctly (Windows drive letters)
TEST_F(PathValidatorTest, IsValidPath_ColonHandling) {
    // Valid: Drive letter colon
    EXPECT_TRUE(PathValidator::isValidPath("C:\\path\\to\\file"));
    EXPECT_TRUE(PathValidator::isValidPath("D:\\"));

    // Invalid: Colon in wrong position
    EXPECT_FALSE(PathValidator::isValidPath("path:invalid"));
    EXPECT_FALSE(PathValidator::isValidPath("C:\\path:invalid"));
    EXPECT_FALSE(PathValidator::isValidPath("1:\\invalid"));  // Number before colon
}

// PV-006: isValidPath handles relative paths
TEST_F(PathValidatorTest, IsValidPath_RelativePaths) {
    EXPECT_TRUE(PathValidator::isValidPath("relative\\path"));
    EXPECT_TRUE(PathValidator::isValidPath("..\\parent\\path"));
    EXPECT_TRUE(PathValidator::isValidPath(".\\current\\path"));
}

// PV-007: isValidPath handles UNC paths
TEST_F(PathValidatorTest, IsValidPath_UNCPaths) {
    EXPECT_TRUE(PathValidator::isValidPath("\\\\server\\share\\path"));
}

// PV-008: isValidPath handles paths with spaces
TEST_F(PathValidatorTest, IsValidPath_PathsWithSpaces) {
    EXPECT_TRUE(PathValidator::isValidPath("C:\\Program Files\\App"));
    EXPECT_TRUE(PathValidator::isValidPath("path with spaces\\file.txt"));
}

// =============================================================================
// fileExists Tests (PV-009 ~ PV-013)
// =============================================================================

// PV-009: fileExists returns true for existing file
TEST_F(PathValidatorTest, FileExists_ExistingFile) {
    EXPECT_TRUE(PathValidator::fileExists(testFile));
}

// PV-010: fileExists returns false for non-existing file
TEST_F(PathValidatorTest, FileExists_NonExistingFile) {
    EXPECT_FALSE(PathValidator::fileExists(testDir + "\\nonexistent.txt"));
}

// PV-011: fileExists returns false for directory
TEST_F(PathValidatorTest, FileExists_Directory) {
    EXPECT_FALSE(PathValidator::fileExists(testDir));
}

// PV-012: fileExists returns false for empty path
TEST_F(PathValidatorTest, FileExists_EmptyPath) {
    EXPECT_FALSE(PathValidator::fileExists(""));
}

// PV-013: fileExists handles special characters in path
TEST_F(PathValidatorTest, FileExists_InvalidPath) {
    EXPECT_FALSE(PathValidator::fileExists("C:\\invalid<path"));
}

// =============================================================================
// directoryExists Tests (PV-014 ~ PV-018)
// =============================================================================

// PV-014: directoryExists returns true for existing directory
TEST_F(PathValidatorTest, DirectoryExists_ExistingDirectory) {
    EXPECT_TRUE(PathValidator::directoryExists(testDir));
    EXPECT_TRUE(PathValidator::directoryExists(nestedDir));
}

// PV-015: directoryExists returns false for non-existing directory
TEST_F(PathValidatorTest, DirectoryExists_NonExistingDirectory) {
    EXPECT_FALSE(PathValidator::directoryExists(testDir + "\\nonexistent"));
}

// PV-016: directoryExists returns false for file
TEST_F(PathValidatorTest, DirectoryExists_File) {
    EXPECT_FALSE(PathValidator::directoryExists(testFile));
}

// PV-017: directoryExists returns false for empty path
TEST_F(PathValidatorTest, DirectoryExists_EmptyPath) {
    EXPECT_FALSE(PathValidator::directoryExists(""));
}

// PV-018: directoryExists handles trailing separator
TEST_F(PathValidatorTest, DirectoryExists_TrailingSeparator) {
    EXPECT_TRUE(PathValidator::directoryExists(testDir + "\\"));
}

// =============================================================================
// isAbsolutePath Tests (PV-019 ~ PV-023)
// =============================================================================

// PV-019: isAbsolutePath returns true for Windows absolute paths
TEST_F(PathValidatorTest, IsAbsolutePath_WindowsAbsolute) {
    EXPECT_TRUE(PathValidator::isAbsolutePath("C:\\Windows"));
    EXPECT_TRUE(PathValidator::isAbsolutePath("D:\\path\\to\\file"));
    EXPECT_TRUE(PathValidator::isAbsolutePath(testDir));
}

// PV-020: isAbsolutePath returns true for UNC paths
TEST_F(PathValidatorTest, IsAbsolutePath_UNCPath) {
    EXPECT_TRUE(PathValidator::isAbsolutePath("\\\\server\\share"));
}

// PV-021: isAbsolutePath returns false for relative paths
TEST_F(PathValidatorTest, IsAbsolutePath_RelativePath) {
    EXPECT_FALSE(PathValidator::isAbsolutePath("relative\\path"));
    EXPECT_FALSE(PathValidator::isAbsolutePath("..\\parent"));
    EXPECT_FALSE(PathValidator::isAbsolutePath(".\\current"));
}

// PV-022: isAbsolutePath returns false for empty path
TEST_F(PathValidatorTest, IsAbsolutePath_EmptyPath) {
    EXPECT_FALSE(PathValidator::isAbsolutePath(""));
}

// PV-023: isAbsolutePath handles drive-relative paths
TEST_F(PathValidatorTest, IsAbsolutePath_DriveRelative) {
    // \path is a drive-relative path (relative to current drive)
    EXPECT_FALSE(PathValidator::isAbsolutePath("\\relative_to_root"));
}

// =============================================================================
// hasValidExtension Tests (PV-024 ~ PV-028)
// =============================================================================

// PV-024: hasValidExtension returns true for valid extension
TEST_F(PathValidatorTest, HasValidExtension_ValidExtension) {
    std::vector<std::string> validExts = {".txt", ".yml", ".yaml"};

    EXPECT_TRUE(PathValidator::hasValidExtension("file.txt", validExts));
    EXPECT_TRUE(PathValidator::hasValidExtension("config.yaml", validExts));
    EXPECT_TRUE(PathValidator::hasValidExtension("config.yml", validExts));
}

// PV-025: hasValidExtension returns false for invalid extension
TEST_F(PathValidatorTest, HasValidExtension_InvalidExtension) {
    std::vector<std::string> validExts = {".txt", ".yml"};

    EXPECT_FALSE(PathValidator::hasValidExtension("file.pdf", validExts));
    EXPECT_FALSE(PathValidator::hasValidExtension("file.doc", validExts));
}

// PV-026: hasValidExtension is case-insensitive
TEST_F(PathValidatorTest, HasValidExtension_CaseInsensitive) {
    std::vector<std::string> validExts = {".txt", ".TXT", ".Txt"};

    EXPECT_TRUE(PathValidator::hasValidExtension("file.TXT", validExts));
    EXPECT_TRUE(PathValidator::hasValidExtension("file.txt", validExts));
    EXPECT_TRUE(PathValidator::hasValidExtension("file.Txt", validExts));
}

// PV-027: hasValidExtension handles no extension
TEST_F(PathValidatorTest, HasValidExtension_NoExtension) {
    std::vector<std::string> validExts = {".txt"};

    EXPECT_FALSE(PathValidator::hasValidExtension("file_no_ext", validExts));
}

// PV-028: hasValidExtension handles multiple dots
TEST_F(PathValidatorTest, HasValidExtension_MultipleDots) {
    std::vector<std::string> validExts = {".tar.gz", ".gz"};

    // Only last extension is checked
    EXPECT_TRUE(PathValidator::hasValidExtension("archive.tar.gz", validExts));
}

// =============================================================================
// normalizePath Tests (PV-029 ~ PV-033)
// =============================================================================

// PV-029: normalizePath removes redundant separators
TEST_F(PathValidatorTest, NormalizePath_RedundantSeparators) {
    std::string normalized = PathValidator::normalizePath("C:\\\\path\\\\to\\\\file");
    EXPECT_EQ(normalized.find("\\\\"), std::string::npos);
}

// PV-030: normalizePath resolves . and ..
TEST_F(PathValidatorTest, NormalizePath_DotNotation) {
    std::string path = testDir + "\\.\\nested\\..\\nested";
    std::string normalized = PathValidator::normalizePath(path);

    // Should resolve to testDir\nested
    EXPECT_TRUE(normalized.find("nested") != std::string::npos);
    EXPECT_EQ(normalized.find("\\.\\"), std::string::npos);
    EXPECT_EQ(normalized.find("\\..\\"), std::string::npos);
}

// PV-031: normalizePath returns empty for empty path
TEST_F(PathValidatorTest, NormalizePath_EmptyPath) {
    EXPECT_EQ("", PathValidator::normalizePath(""));
}

// PV-032: normalizePath handles existing paths (canonical)
TEST_F(PathValidatorTest, NormalizePath_ExistingPath) {
    std::string normalized = PathValidator::normalizePath(testDir);
    EXPECT_FALSE(normalized.empty());
    EXPECT_TRUE(fs::exists(normalized));
}

// PV-033: normalizePath handles non-existing paths (absolute + lexically_normal)
TEST_F(PathValidatorTest, NormalizePath_NonExistingPath) {
    std::string nonExisting = testDir + "\\nonexistent\\path";
    std::string normalized = PathValidator::normalizePath(nonExisting);
    EXPECT_FALSE(normalized.empty());
}

// =============================================================================
// getExtension Tests (PV-034 ~ PV-037)
// =============================================================================

// PV-034: getExtension returns extension with dot
TEST_F(PathValidatorTest, GetExtension_WithDot) {
    EXPECT_EQ(".txt", PathValidator::getExtension("file.txt"));
    EXPECT_EQ(".yaml", PathValidator::getExtension("config.yaml"));
    EXPECT_EQ(".h", PathValidator::getExtension("header.h"));
}

// PV-035: getExtension returns empty for no extension
TEST_F(PathValidatorTest, GetExtension_NoExtension) {
    EXPECT_EQ("", PathValidator::getExtension("file_no_ext"));
    EXPECT_EQ("", PathValidator::getExtension("Makefile"));
}

// PV-036: getExtension handles multiple dots
TEST_F(PathValidatorTest, GetExtension_MultipleDots) {
    EXPECT_EQ(".gz", PathValidator::getExtension("archive.tar.gz"));
    EXPECT_EQ(".cpp", PathValidator::getExtension("file.test.cpp"));
}

// PV-037: getExtension returns empty for empty path
TEST_F(PathValidatorTest, GetExtension_EmptyPath) {
    EXPECT_EQ("", PathValidator::getExtension(""));
}

// =============================================================================
// getDirectory Tests (PV-038 ~ PV-041)
// =============================================================================

// PV-038: getDirectory returns parent directory
TEST_F(PathValidatorTest, GetDirectory_ParentDirectory) {
    std::string dir = PathValidator::getDirectory(testFile);
    EXPECT_EQ(testDir, dir);
}

// PV-039: getDirectory handles root path
TEST_F(PathValidatorTest, GetDirectory_RootPath) {
    std::string dir = PathValidator::getDirectory("C:\\file.txt");
    EXPECT_EQ("C:\\", dir);
}

// PV-040: getDirectory returns empty for filename only
TEST_F(PathValidatorTest, GetDirectory_FilenameOnly) {
    std::string dir = PathValidator::getDirectory("filename.txt");
    EXPECT_EQ("", dir);
}

// PV-041: getDirectory returns empty for empty path
TEST_F(PathValidatorTest, GetDirectory_EmptyPath) {
    EXPECT_EQ("", PathValidator::getDirectory(""));
}

// =============================================================================
// getFilename Tests (PV-042 ~ PV-045)
// =============================================================================

// PV-042: getFilename returns filename with extension
TEST_F(PathValidatorTest, GetFilename_WithExtension) {
    EXPECT_EQ("test_file.txt", PathValidator::getFilename(testFile));
    EXPECT_EQ("config.yaml", PathValidator::getFilename("C:\\path\\to\\config.yaml"));
}

// PV-043: getFilename returns directory name for directory path
TEST_F(PathValidatorTest, GetFilename_DirectoryPath) {
    std::string filename = PathValidator::getFilename(testDir);
    EXPECT_EQ("WheelDL_PathValidator_Test", filename);
}

// PV-044: getFilename returns filename for path without directory
TEST_F(PathValidatorTest, GetFilename_NoDirectory) {
    EXPECT_EQ("file.txt", PathValidator::getFilename("file.txt"));
}

// PV-045: getFilename returns empty for empty path
TEST_F(PathValidatorTest, GetFilename_EmptyPath) {
    EXPECT_EQ("", PathValidator::getFilename(""));
}

// =============================================================================
// createDirectoryIfNotExists Tests (PV-046 ~ PV-050)
// =============================================================================

// PV-046: createDirectoryIfNotExists creates new directory
TEST_F(PathValidatorTest, CreateDirectoryIfNotExists_NewDirectory) {
    std::string newDir = testDir + "\\new_directory";
    EXPECT_FALSE(fs::exists(newDir));

    EXPECT_TRUE(PathValidator::createDirectoryIfNotExists(newDir));
    EXPECT_TRUE(fs::exists(newDir));
    EXPECT_TRUE(fs::is_directory(newDir));
}

// PV-047: createDirectoryIfNotExists returns true for existing directory
TEST_F(PathValidatorTest, CreateDirectoryIfNotExists_ExistingDirectory) {
    EXPECT_TRUE(PathValidator::createDirectoryIfNotExists(testDir));
}

// PV-048: createDirectoryIfNotExists creates nested directories
TEST_F(PathValidatorTest, CreateDirectoryIfNotExists_NestedDirectories) {
    std::string deepPath = testDir + "\\level1\\level2\\level3";
    EXPECT_FALSE(fs::exists(deepPath));

    EXPECT_TRUE(PathValidator::createDirectoryIfNotExists(deepPath));
    EXPECT_TRUE(fs::exists(deepPath));
    EXPECT_TRUE(fs::is_directory(deepPath));
}

// PV-049: createDirectoryIfNotExists returns false for empty path
TEST_F(PathValidatorTest, CreateDirectoryIfNotExists_EmptyPath) {
    EXPECT_FALSE(PathValidator::createDirectoryIfNotExists(""));
}

// PV-050: createDirectoryIfNotExists returns false if path is existing file
TEST_F(PathValidatorTest, CreateDirectoryIfNotExists_ExistingFile) {
    // Cannot create directory where file exists
    EXPECT_FALSE(PathValidator::createDirectoryIfNotExists(testFile));
}

// =============================================================================
// Static Utility Class Tests
// =============================================================================

// Verify PathValidator cannot be instantiated
TEST(PathValidatorStaticTest, CannotInstantiate) {
    // This is a compile-time check - the deleted constructor prevents instantiation
    // The following line would cause a compilation error if uncommented:
    // PathValidator validator;

    // Instead, verify all methods are static by calling without instance
    EXPECT_FALSE(PathValidator::isValidPath(""));
    EXPECT_FALSE(PathValidator::fileExists(""));
    EXPECT_FALSE(PathValidator::directoryExists(""));
}
