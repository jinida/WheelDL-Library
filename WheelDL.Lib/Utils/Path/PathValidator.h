#pragma once

#include <string>
#include <vector>
#include <filesystem>

namespace WheelDL {
namespace Utils {

/**
 * @class PathValidator
 * @brief Static utility class for path validation and manipulation
 *
 * Provides file system path validation, existence checking,
 * and path normalization utilities.
 *
 * @note All methods are static - no instance needed
 */
class PathValidator {
public:
    // Delete constructor (static utility class)
    PathValidator() = delete;
    ~PathValidator() = delete;

    /**
     * @brief Check if path is valid
     * @param path Path to validate
     * @return bool True if path is valid
     */
    static bool isValidPath(const std::string& path);

    /**
     * @brief Check if file exists
     * @param path File path to check
     * @return bool True if file exists
     */
    static bool fileExists(const std::string& path);

    /**
     * @brief Check if directory exists
     * @param path Directory path to check
     * @return bool True if directory exists
     */
    static bool directoryExists(const std::string& path);

    /**
     * @brief Check if path is absolute
     * @param path Path to check
     * @return bool True if absolute path
     */
    static bool isAbsolutePath(const std::string& path);

    /**
     * @brief Check if file has valid extension
     * @param path File path to check
     * @param validExts List of valid extensions (e.g., {".yaml", ".yml"})
     * @return bool True if extension is in validExts
     */
    static bool hasValidExtension(const std::string& path,
                                  const std::vector<std::string>& validExts);

    /**
     * @brief Normalize path (convert to standard format)
     * @param path Path to normalize
     * @return std::string Normalized path
     */
    static std::string normalizePath(const std::string& path);

    /**
     * @brief Get file extension
     * @param path File path
     * @return std::string File extension (including dot, e.g., ".txt")
     */
    static std::string getExtension(const std::string& path);

    /**
     * @brief Get directory from path
     * @param path File or directory path
     * @return std::string Directory path
     */
    static std::string getDirectory(const std::string& path);

    /**
     * @brief Get filename from path (without directory)
     * @param path File path
     * @return std::string Filename
     */
    static std::string getFilename(const std::string& path);

    /**
     * @brief Create directory if it doesn't exist
     * @param path Directory path to create
     * @return bool True if directory was created or already exists
     */
    static bool createDirectoryIfNotExists(const std::string& path);

private:
    // Helper to check for invalid characters
    static bool hasInvalidCharacters(const std::string& path);
};

} // namespace Utils
} // namespace WheelDL
