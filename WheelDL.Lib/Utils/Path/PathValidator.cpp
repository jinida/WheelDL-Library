#include "pch.h"
#include "PathValidator.h"
#include <algorithm>
#include <cctype>
#include <iostream>

namespace WheelDL {
namespace Utils {

namespace fs = std::filesystem;

bool PathValidator::isValidPath(const std::string& path) {
    if (path.empty()) {
        return false;
    }

    // Check for invalid characters
    if (hasInvalidCharacters(path)) {
        return false;
    }

    // Try to create path object (will throw if invalid)
    try {
        fs::path p(path);
        return true;
    }
    catch (...) {
        return false;
    }
}

bool PathValidator::fileExists(const std::string& path) {
    if (path.empty()) {
        return false;
    }

    try {
        fs::path p(path);
        return fs::exists(p) && fs::is_regular_file(p);
    }
    catch (...) {
        return false;
    }
}

bool PathValidator::directoryExists(const std::string& path) {
    if (path.empty()) {
        return false;
    }

    try {
        fs::path p(path);
        return fs::exists(p) && fs::is_directory(p);
    }
    catch (...) {
        return false;
    }
}

bool PathValidator::isAbsolutePath(const std::string& path) {
    if (path.empty()) {
        return false;
    }

    try {
        fs::path p(path);
        return p.is_absolute();
    }
    catch (...) {
        return false;
    }
}

bool PathValidator::hasValidExtension(const std::string& path,
                                      const std::vector<std::string>& validExts) {
    std::string ext = getExtension(path);

    // Convert extension to lowercase once
    std::string extLower = ext;
    std::transform(extLower.begin(), extLower.end(), extLower.begin(),
                  [](unsigned char c) { return std::tolower(c); });

    // Check if extension is in valid list using character-by-character comparison
    // to avoid creating a copy of each validExt string
    for (const auto& validExt : validExts) {
        // Quick size check first
        if (extLower.size() != validExt.size()) {
            continue;
        }

        // Compare character by character (converting validExt chars to lowercase on the fly)
        bool match = true;
        for (size_t i = 0; i < extLower.size(); ++i) {
            if (extLower[i] != std::tolower(static_cast<unsigned char>(validExt[i]))) {
                match = false;
                break;
            }
        }

        if (match) {
            return true;
        }
    }

    return false;
}

std::string PathValidator::normalizePath(const std::string& path) {
    if (path.empty()) {
        return "";
    }

    try {
        fs::path p(path);
        // Convert to preferred format and make canonical if exists
        if (fs::exists(p)) {
            return fs::canonical(p).string();
        } else {
            return fs::absolute(p).lexically_normal().string();
        }
    }
    catch (const std::exception& e) {
        // Log normalization failure with specific error message
        std::cerr << "Warning: Path normalization failed for '" << path
                  << "': " << e.what() << ". Using original path." << std::endl;
        return path;
    }
    catch (...) {
        // Log normalization failure for unknown exceptions
        std::cerr << "Warning: Path normalization failed for '" << path
                  << "': Unknown error. Using original path." << std::endl;
        return path;
    }
}

std::string PathValidator::getExtension(const std::string& path) {
    if (path.empty()) {
        return "";
    }

    try {
        fs::path p(path);
        return p.extension().string();
    }
    catch (...) {
        return "";
    }
}

std::string PathValidator::getDirectory(const std::string& path) {
    if (path.empty()) {
        return "";
    }

    try {
        fs::path p(path);
        return p.parent_path().string();
    }
    catch (...) {
        return "";
    }
}

std::string PathValidator::getFilename(const std::string& path) {
    if (path.empty()) {
        return "";
    }

    try {
        fs::path p(path);
        return p.filename().string();
    }
    catch (...) {
        return "";
    }
}

bool PathValidator::createDirectoryIfNotExists(const std::string& path) {
    if (path.empty()) {
        return false;
    }

    try {
        fs::path p(path);

        // If already exists and is a directory, return true
        if (fs::exists(p)) {
            return fs::is_directory(p);
        }

        // Create directory and all parent directories
        return fs::create_directories(p);
    }
    catch (...) {
        return false;
    }
}

bool PathValidator::hasInvalidCharacters(const std::string& path) {
    // Check for null character
    if (path.find('\0') != std::string::npos) {
        return true;
    }

    // Platform-specific invalid characters
#ifdef _WIN32
    // Windows invalid characters: < > " | ? *
    // Note: Colon (:) is allowed after drive letter (e.g., C:\)
    // Use find_first_of for O(n) complexity instead of O(n*m)
    if (path.find_first_of("<>\"|?*") != std::string::npos) {
        return true;
    }

    // Check for colon in invalid positions in a single pass
    // Colon is valid only after drive letter (position 1)
    // We iterate once and check all conditions together
    bool hasValidDriveLetter = false;
    for (size_t i = 0; i < path.length(); ++i) {
        if (path[i] == ':') {
            // First colon: check if it's a valid drive letter
            if (i == 1 && std::isalpha(static_cast<unsigned char>(path[0]))) {
                hasValidDriveLetter = true;
                continue;  // This colon is valid, continue checking for additional colons
            }
            // Any other colon position is invalid (including additional colons)
            return true;
        }
    }
#endif

    return false;
}

} // namespace Utils
} // namespace WheelDL
