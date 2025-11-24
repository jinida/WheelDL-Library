#include "pch.h"
#include "Workspace.h"
#include "../Logger/Logger.h"
#include "../Path/PathValidator.h"
#include "../Error/WheelLibException.h"
#include "../Error/ErrorCodes.h"
#include <sstream>
#include <iomanip>
#include <chrono>

namespace WheelDL {
    namespace Utils {

        Workspace::Workspace(const std::string& baseDir, const std::string& prefix)
        {
            auto logger = Logger::getInstance();

            // Validate base directory path
            if (!PathValidator::isValidPath(baseDir)) {
                logger->error("Workspace", "Invalid base directory path: " + baseDir);
                throw WheelLibException(
                    ErrorCode::INVALID_CONFIG,
                    "Invalid base directory path: " + baseDir
                );
            }

            // Generate timestamp
            _timestamp = generateTimestamp();

            // Create workspace root path: baseDir/prefix_timestamp
            std::ostringstream oss;
            oss << baseDir << "/" << prefix << "_" << _timestamp;
            _workspaceRoot = oss.str();

            logger->info("Workspace", "Creating workspace: " + _workspaceRoot);

            // Create root directory using PathValidator
            if (!PathValidator::createDirectoryIfNotExists(_workspaceRoot)) {
                logger->error("Workspace", "Failed to create workspace directory: " + _workspaceRoot);
                throw WheelLibException(
                    ErrorCode::FILE_IO_ERROR,
                    "Failed to create workspace directory: " + _workspaceRoot
                );
            }

            // Create standard subdirectories
            createStandardDirectories();

            logger->info("Workspace", "Workspace created successfully");
        }

        Workspace::Workspace(
            const std::string& baseDir,
            const std::string& customName,
            bool useTimestamp)
        {
            auto logger = Logger::getInstance();

            // Validate base directory path
            if (!PathValidator::isValidPath(baseDir)) {
                logger->error("Workspace", "Invalid base directory path: " + baseDir);
                throw WheelLibException(
                    ErrorCode::INVALID_CONFIG,
                    "Invalid base directory path: " + baseDir
                );
            }

            // Generate workspace name
            std::ostringstream oss;
            oss << baseDir << "/" << customName;

            if (useTimestamp) {
                _timestamp = generateTimestamp();
                oss << "_" << _timestamp;
            } else {
                _timestamp = "";
            }

            _workspaceRoot = oss.str();

            logger->info("Workspace", "Creating custom workspace: " + _workspaceRoot);

            // Create root directory using PathValidator
            if (!PathValidator::createDirectoryIfNotExists(_workspaceRoot)) {
                logger->error("Workspace", "Failed to create workspace directory: " + _workspaceRoot);
                throw WheelLibException(
                    ErrorCode::FILE_IO_ERROR,
                    "Failed to create workspace directory: " + _workspaceRoot
                );
            }

            // Create standard subdirectories
            createStandardDirectories();

            logger->info("Workspace", "Custom workspace created successfully");
        }

        std::string Workspace::createSubDirectory(const std::string& subDirName)
        {
            auto logger = Logger::getInstance();
            std::string subDirPath = _workspaceRoot + "/" + subDirName;

            // Use PathValidator to create directory
            if (!PathValidator::createDirectoryIfNotExists(subDirPath)) {
                logger->error("Workspace", "Failed to create subdirectory: " + subDirPath);
                throw WheelLibException(
                    ErrorCode::FILE_IO_ERROR,
                    "Failed to create subdirectory: " + subDirPath
                );
            }

            logger->debug("Workspace", "Created subdirectory: " + subDirPath);
            return subDirPath;
        }

        bool Workspace::exists() const
        {
            return PathValidator::directoryExists(_workspaceRoot);
        }

        std::string Workspace::generateTimestamp()
        {
            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            std::tm tm_now;

            #ifdef _WIN32
                localtime_s(&tm_now, &time_t_now);
            #else
                localtime_r(&time_t_now, &tm_now);
            #endif

            std::ostringstream oss;
            oss << std::put_time(&tm_now, "%Y%m%d_%H%M%S");
            return oss.str();
        }

        void Workspace::createStandardDirectories()
        {
            auto logger = Logger::getInstance();

            // Create weights directory
            _weightsDir = _workspaceRoot + "/weights";
            if (!PathValidator::createDirectoryIfNotExists(_weightsDir)) {
                logger->error("Workspace", "Failed to create weights directory: " + _weightsDir);
                throw WheelLibException(
                    ErrorCode::FILE_IO_ERROR,
                    "Failed to create weights directory: " + _weightsDir
                );
            }
            logger->debug("Workspace", "Created weights directory: " + _weightsDir);

            // Create logs directory
            _logsDir = _workspaceRoot + "/logs";
            if (!PathValidator::createDirectoryIfNotExists(_logsDir)) {
                logger->error("Workspace", "Failed to create logs directory: " + _logsDir);
                throw WheelLibException(
                    ErrorCode::FILE_IO_ERROR,
                    "Failed to create logs directory: " + _logsDir
                );
            }
            logger->debug("Workspace", "Created logs directory: " + _logsDir);

            // Create profiler directory
            _profilerDir = _workspaceRoot + "/profiler";
            if (!PathValidator::createDirectoryIfNotExists(_profilerDir)) {
                logger->error("Workspace", "Failed to create profiler directory: " + _profilerDir);
                throw WheelLibException(
                    ErrorCode::FILE_IO_ERROR,
                    "Failed to create profiler directory: " + _profilerDir
                );
            }
            logger->debug("Workspace", "Created profiler directory: " + _profilerDir);
        }

    } // namespace Utils
} // namespace WheelDL
