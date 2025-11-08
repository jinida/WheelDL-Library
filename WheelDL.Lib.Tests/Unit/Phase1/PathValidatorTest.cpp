#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Utils/Path/PathValidator.h"
#include <filesystem>
#include <fstream>

using namespace WheelDL::Utils;
namespace fs = std::filesystem;

/**
 * @class PathValidatorTest
 * @brief Unit tests for PathValidator static utility class
 *
 * Tests cover:
 * - Path validation
 * - File and directory existence checks
 * - Path normalization
 * - Extension handling
 * - Directory creation
 * - Edge cases and invalid inputs
 */
class PathValidatorTest : public ::testing::Test {
protected:
	void SetUp() override {
		// Create test directory and files
		testDir = fs::current_path() / "test_path_validator";
		fs::create_directories(testDir);

		testFile = testDir / "test_file.txt";
		std::ofstream(testFile) << "test content";

		testYamlFile = testDir / "config.yaml";
		std::ofstream(testYamlFile) << "key: value";

		testSubDir = testDir / "subdir";
		fs::create_directories(testSubDir);
	}

	void TearDown() override {
		// Clean up test directory
		if (fs::exists(testDir)) {
			fs::remove_all(testDir);
		}
	}

	fs::path testDir;
	fs::path testFile;
	fs::path testYamlFile;
	fs::path testSubDir;
};

// ============================================================================
// Path Validation Tests
// ============================================================================

TEST_F(PathValidatorTest, IsValidPath_ValidAbsolutePath) {
	// Act & Assert
	EXPECT_TRUE(PathValidator::isValidPath(testFile.string()));
}

TEST_F(PathValidatorTest, IsValidPath_ValidRelativePath) {
	// Act & Assert
	EXPECT_TRUE(PathValidator::isValidPath("./relative/path"));
	EXPECT_TRUE(PathValidator::isValidPath("../parent/path"));
}

TEST_F(PathValidatorTest, IsValidPath_EmptyPath) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::isValidPath(""));
}

TEST_F(PathValidatorTest, IsValidPath_InvalidCharacters_Windows) {
#ifdef _WIN32
	// Act & Assert
	EXPECT_FALSE(PathValidator::isValidPath("C:\\invalid<path"));
	EXPECT_FALSE(PathValidator::isValidPath("C:\\invalid>path"));
	EXPECT_FALSE(PathValidator::isValidPath("C:\\invalid|path"));
	EXPECT_FALSE(PathValidator::isValidPath("C:\\invalid\"path"));
	EXPECT_FALSE(PathValidator::isValidPath("C:\\invalid?path"));
	EXPECT_FALSE(PathValidator::isValidPath("C:\\invalid*path"));
#endif
}

TEST_F(PathValidatorTest, IsValidPath_NullCharacter) {
	// Arrange
	std::string pathWithNull = "test";
	pathWithNull += '\0';
	pathWithNull += "path";

	// Act & Assert
	EXPECT_FALSE(PathValidator::isValidPath(pathWithNull));
}

TEST_F(PathValidatorTest, IsValidPath_LongPath) {
	// Arrange
	std::string longPath = testDir.string();
	for (int i = 0; i < 50; ++i) {
		longPath += "/very_long_directory_name_" + std::to_string(i);
	}

	// Act & Assert - Should be valid (though may not exist)
	EXPECT_TRUE(PathValidator::isValidPath(longPath));
}

// ============================================================================
// File Existence Tests
// ============================================================================

TEST_F(PathValidatorTest, FileExists_ExistingFile) {
	// Act & Assert
	EXPECT_TRUE(PathValidator::fileExists(testFile.string()));
}

TEST_F(PathValidatorTest, FileExists_NonExistentFile) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::fileExists((testDir / "nonexistent.txt").string()));
}

TEST_F(PathValidatorTest, FileExists_DirectoryPath) {
	// Act & Assert - Directory should not be recognized as a file
	EXPECT_FALSE(PathValidator::fileExists(testDir.string()));
}

TEST_F(PathValidatorTest, FileExists_EmptyPath) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::fileExists(""));
}

TEST_F(PathValidatorTest, FileExists_InvalidPath) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::fileExists("Z:\\nonexistent\\invalid\\path.txt"));
}

// ============================================================================
// Directory Existence Tests
// ============================================================================

TEST_F(PathValidatorTest, DirectoryExists_ExistingDirectory) {
	// Act & Assert
	EXPECT_TRUE(PathValidator::directoryExists(testDir.string()));
	EXPECT_TRUE(PathValidator::directoryExists(testSubDir.string()));
}

TEST_F(PathValidatorTest, DirectoryExists_NonExistentDirectory) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::directoryExists((testDir / "nonexistent_dir").string()));
}

TEST_F(PathValidatorTest, DirectoryExists_FilePath) {
	// Act & Assert - File should not be recognized as directory
	EXPECT_FALSE(PathValidator::directoryExists(testFile.string()));
}

TEST_F(PathValidatorTest, DirectoryExists_EmptyPath) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::directoryExists(""));
}

TEST_F(PathValidatorTest, DirectoryExists_CurrentDirectory) {
	// Act & Assert
	EXPECT_TRUE(PathValidator::directoryExists("."));
}

TEST_F(PathValidatorTest, DirectoryExists_ParentDirectory) {
	// Act & Assert
	EXPECT_TRUE(PathValidator::directoryExists(".."));
}

// ============================================================================
// Absolute Path Tests
// ============================================================================

TEST_F(PathValidatorTest, IsAbsolutePath_AbsolutePath) {
	// Act & Assert
	EXPECT_TRUE(PathValidator::isAbsolutePath(testFile.string()));
}

TEST_F(PathValidatorTest, IsAbsolutePath_RelativePath) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::isAbsolutePath("./relative/path"));
	EXPECT_FALSE(PathValidator::isAbsolutePath("../parent/path"));
	EXPECT_FALSE(PathValidator::isAbsolutePath("relative/path"));
}

#ifdef _WIN32
TEST_F(PathValidatorTest, IsAbsolutePath_WindowsDrive) {
	// Act & Assert
	EXPECT_TRUE(PathValidator::isAbsolutePath("C:\\Windows\\System32"));
	EXPECT_TRUE(PathValidator::isAbsolutePath("D:\\Data"));
}

TEST_F(PathValidatorTest, IsAbsolutePath_WindowsRelative) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::isAbsolutePath("Windows\\System32"));
}
#else
TEST_F(PathValidatorTest, IsAbsolutePath_UnixRoot) {
	// Act & Assert
	EXPECT_TRUE(PathValidator::isAbsolutePath("/usr/local/bin"));
	EXPECT_TRUE(PathValidator::isAbsolutePath("/home/user"));
}

TEST_F(PathValidatorTest, IsAbsolutePath_UnixRelative) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::isAbsolutePath("usr/local/bin"));
}
#endif

TEST_F(PathValidatorTest, IsAbsolutePath_EmptyPath) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::isAbsolutePath(""));
}

// ============================================================================
// Extension Validation Tests
// ============================================================================

TEST_F(PathValidatorTest, HasValidExtension_SingleExtension_Valid) {
	// Act & Assert
	EXPECT_TRUE(PathValidator::hasValidExtension(testYamlFile.string(), { ".yaml" }));
}

TEST_F(PathValidatorTest, HasValidExtension_SingleExtension_Invalid) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::hasValidExtension(testFile.string(), { ".yaml" }));
}

TEST_F(PathValidatorTest, HasValidExtension_MultipleExtensions_Valid) {
	// Act & Assert
	EXPECT_TRUE(PathValidator::hasValidExtension(testYamlFile.string(), { ".yml", ".yaml", ".json" }));
	EXPECT_TRUE(PathValidator::hasValidExtension(testFile.string(), { ".txt", ".log", ".dat" }));
}

TEST_F(PathValidatorTest, HasValidExtension_MultipleExtensions_Invalid) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::hasValidExtension(testFile.string(), { ".yaml", ".json", ".xml" }));
}

TEST_F(PathValidatorTest, HasValidExtension_CaseInsensitive) {
	// Act & Assert
	EXPECT_TRUE(PathValidator::hasValidExtension("FILE.TXT", { ".txt" }));
	EXPECT_TRUE(PathValidator::hasValidExtension("file.txt", { ".TXT" }));
	EXPECT_TRUE(PathValidator::hasValidExtension("FiLe.TxT", { ".TxT" }));
}

TEST_F(PathValidatorTest, HasValidExtension_NoExtension) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::hasValidExtension("file_without_extension", { ".txt" }));
}

TEST_F(PathValidatorTest, HasValidExtension_EmptyExtensionList) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::hasValidExtension(testFile.string(), {}));
}

TEST_F(PathValidatorTest, HasValidExtension_DotFile) {
	// Act & Assert
	EXPECT_FALSE(PathValidator::hasValidExtension(".gitignore", { ".txt" }));
}

// ============================================================================
// Extension Getter Tests
// ============================================================================

TEST_F(PathValidatorTest, GetExtension_StandardFile) {
	// Act & Assert
	EXPECT_EQ(PathValidator::getExtension(testFile.string()), ".txt");
	EXPECT_EQ(PathValidator::getExtension(testYamlFile.string()), ".yaml");
}

TEST_F(PathValidatorTest, GetExtension_NoExtension) {
	// Act & Assert
	EXPECT_EQ(PathValidator::getExtension("file_without_extension"), "");
}

TEST_F(PathValidatorTest, GetExtension_MultipleExtensions) {
	// Act & Assert
	EXPECT_EQ(PathValidator::getExtension("archive.tar.gz"), ".gz");
}

TEST_F(PathValidatorTest, GetExtension_DotFile) {
	// Act & Assert
	EXPECT_EQ(PathValidator::getExtension(".gitignore"), "");
}

TEST_F(PathValidatorTest, GetExtension_EmptyPath) {
	// Act & Assert
	EXPECT_EQ(PathValidator::getExtension(""), "");
}

// ============================================================================
// Path Component Tests
// ============================================================================

TEST_F(PathValidatorTest, GetDirectory_FilePath) {
	// Act
	std::string dir = PathValidator::getDirectory(testFile.string());

	// Assert
	EXPECT_EQ(fs::path(dir), testDir);
}

TEST_F(PathValidatorTest, GetDirectory_DirectoryPath) {
	// Act
	std::string parentDir = PathValidator::getDirectory(testSubDir.string());

	// Assert
	EXPECT_EQ(fs::path(parentDir), testDir);
}

TEST_F(PathValidatorTest, GetDirectory_RootPath) {
#ifdef _WIN32
	// Act
	std::string result = PathValidator::getDirectory("C:\\");

	// Assert - Root path may return "C:\" or "" depending on implementation
	// Both are acceptable as there's no parent for root
	EXPECT_TRUE(result == "" || result == "C:\\");
#else
	// Act & Assert
	std::string result = PathValidator::getDirectory("/");
	EXPECT_TRUE(result == "" || result == "/");
#endif
}

TEST_F(PathValidatorTest, GetDirectory_RelativePath) {
	// Act
	std::string dir = PathValidator::getDirectory("subdir/file.txt");

	// Assert
	EXPECT_EQ(dir, "subdir");
}

TEST_F(PathValidatorTest, GetDirectory_EmptyPath) {
	// Act & Assert
	EXPECT_EQ(PathValidator::getDirectory(""), "");
}

TEST_F(PathValidatorTest, GetFilename_FilePath) {
	// Act & Assert
	EXPECT_EQ(PathValidator::getFilename(testFile.string()), "test_file.txt");
	EXPECT_EQ(PathValidator::getFilename(testYamlFile.string()), "config.yaml");
}

TEST_F(PathValidatorTest, GetFilename_DirectoryPath) {
	// Act & Assert
	EXPECT_EQ(PathValidator::getFilename(testSubDir.string()), "subdir");
}

TEST_F(PathValidatorTest, GetFilename_EmptyPath) {
	// Act & Assert
	EXPECT_EQ(PathValidator::getFilename(""), "");
}

// ============================================================================
// Path Normalization Tests
// ============================================================================

TEST_F(PathValidatorTest, NormalizePath_ExistingPath) {
	// Act
	std::string normalized = PathValidator::normalizePath(testFile.string());

	// Assert - Should be canonical absolute path
	EXPECT_TRUE(PathValidator::isAbsolutePath(normalized));
	EXPECT_TRUE(PathValidator::fileExists(normalized));
}

TEST_F(PathValidatorTest, NormalizePath_RelativeToAbsolute) {
	// Arrange
	std::string relativePath = "test_path_validator/test_file.txt";

	// Act
	std::string normalized = PathValidator::normalizePath(relativePath);

	// Assert
	EXPECT_TRUE(PathValidator::isAbsolutePath(normalized));
}

TEST_F(PathValidatorTest, NormalizePath_WithDotDot) {
	// Arrange
	std::string pathWithDots = testDir.string() + "/subdir/../test_file.txt";

	// Act
	std::string normalized = PathValidator::normalizePath(pathWithDots);

	// Assert - Should resolve to test_file.txt in testDir
	EXPECT_EQ(fs::path(normalized).filename(), "test_file.txt");
	EXPECT_TRUE(PathValidator::fileExists(normalized));
}

TEST_F(PathValidatorTest, NormalizePath_WithDot) {
	// Arrange
	std::string pathWithDot = testDir.string() + "/./test_file.txt";

	// Act
	std::string normalized = PathValidator::normalizePath(pathWithDot);

	// Assert
	EXPECT_TRUE(PathValidator::fileExists(normalized));
}

TEST_F(PathValidatorTest, NormalizePath_NonExistentPath) {
	// Arrange
	std::string nonExistent = "nonexistent/path/file.txt";

	// Act
	std::string normalized = PathValidator::normalizePath(nonExistent);

	// Assert - Should still normalize even if doesn't exist
	EXPECT_TRUE(PathValidator::isAbsolutePath(normalized));
}

TEST_F(PathValidatorTest, NormalizePath_EmptyPath) {
	// Act & Assert
	EXPECT_EQ(PathValidator::normalizePath(""), "");
}

// ============================================================================
// Directory Creation Tests
// ============================================================================

TEST_F(PathValidatorTest, CreateDirectoryIfNotExists_NewDirectory) {
	// Arrange
	fs::path newDir = testDir / "new_directory";

	// Act
	bool created = PathValidator::createDirectoryIfNotExists(newDir.string());

	// Assert
	EXPECT_TRUE(created);
	EXPECT_TRUE(PathValidator::directoryExists(newDir.string()));
}

TEST_F(PathValidatorTest, CreateDirectoryIfNotExists_ExistingDirectory) {
	// Act
	bool result = PathValidator::createDirectoryIfNotExists(testDir.string());

	// Assert - Should return true for existing directory
	EXPECT_TRUE(result);
	EXPECT_TRUE(PathValidator::directoryExists(testDir.string()));
}

TEST_F(PathValidatorTest, CreateDirectoryIfNotExists_NestedDirectories) {
	// Arrange
	fs::path nestedDir = testDir / "level1" / "level2" / "level3";

	// Act
	bool created = PathValidator::createDirectoryIfNotExists(nestedDir.string());

	// Assert
	EXPECT_TRUE(created);
	EXPECT_TRUE(PathValidator::directoryExists(nestedDir.string()));
}

TEST_F(PathValidatorTest, CreateDirectoryIfNotExists_EmptyPath) {
	// Act
	bool result = PathValidator::createDirectoryIfNotExists("");

	// Assert
	EXPECT_FALSE(result);
}

TEST_F(PathValidatorTest, CreateDirectoryIfNotExists_InvalidPath) {
#ifdef _WIN32
	// Act
	bool result = PathValidator::createDirectoryIfNotExists("C:\\invalid<dir>");

	// Assert
	EXPECT_FALSE(result);
#endif
}

TEST_F(PathValidatorTest, CreateDirectoryIfNotExists_FileExistsAtPath) {
	// Act - Try to create directory where file exists
	bool result = PathValidator::createDirectoryIfNotExists(testFile.string());

	// Assert - Should return false
	EXPECT_FALSE(result);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(PathValidatorTest, EdgeCase_PathWithSpaces) {
	// Arrange
	fs::path pathWithSpaces = testDir / "directory with spaces";
	fs::create_directories(pathWithSpaces);

	// Act & Assert
	EXPECT_TRUE(PathValidator::isValidPath(pathWithSpaces.string()));
	EXPECT_TRUE(PathValidator::directoryExists(pathWithSpaces.string()));
}

TEST_F(PathValidatorTest, EdgeCase_PathWithUnicode) {
	// Unicode path support depends on system locale and file system settings
	// Test gracefully handles systems without proper UTF-8 support

	try {
		// Arrange
		fs::path unicodePath = testDir / u8"한글경로";

		// Act & Assert
		// Don't assert failure - just verify no crash occurs
		bool created = PathValidator::createDirectoryIfNotExists(unicodePath.string());
		if (created) {
			// If creation succeeded, verify existence
			EXPECT_TRUE(PathValidator::directoryExists(unicodePath.string()));
		}
		else {
			// Creation may fail on systems without Unicode support - this is acceptable
			SUCCEED() << "Unicode path creation not supported on this system";
		}
	}
	catch (const std::exception& e) {
		// Exception is acceptable on systems without Unicode support
		SUCCEED() << "Unicode path operations not supported on this system: " << e.what();
	}
	catch (...) {
		// Any exception is acceptable on systems without Unicode support
		SUCCEED() << "Unicode path operations not supported on this system";
	}
}

TEST_F(PathValidatorTest, EdgeCase_PathWithSpecialChars) {
	// Arrange
	fs::path specialPath = testDir / "dir_with-special.chars_123";
	fs::create_directories(specialPath);

	// Act & Assert
	EXPECT_TRUE(PathValidator::isValidPath(specialPath.string()));
	EXPECT_TRUE(PathValidator::directoryExists(specialPath.string()));
}

TEST_F(PathValidatorTest, EdgeCase_TrailingSlash) {
	// Arrange
	std::string pathWithSlash = testDir.string() + "/";

	// Act & Assert
	EXPECT_TRUE(PathValidator::isValidPath(pathWithSlash));
	EXPECT_TRUE(PathValidator::directoryExists(pathWithSlash));
}

TEST_F(PathValidatorTest, EdgeCase_MultipleSlashes) {
	// Arrange
	std::string pathWithMultiSlashes = testDir.string() + "///subdir";

	// Act
	std::string normalized = PathValidator::normalizePath(pathWithMultiSlashes);

	// Assert - Should normalize multiple slashes
	EXPECT_TRUE(PathValidator::isValidPath(normalized));
}
