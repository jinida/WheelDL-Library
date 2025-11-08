#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>

// Forward declaration for spdlog
namespace spdlog {
	class logger;
}

namespace WheelDL {
	namespace Utils {

		// Log level enumeration
		enum class LogLevel {
			TRACE = 0,
			DEBUG = 1,
			INFO = 2,
			WARN = 3,
			ERR = 4,
			FATAL = 5,
			OFF = 6
		};

		/**
		 * @class Logger
		 * @brief Singleton-based logging system
		 *
		 * Logging system based on spdlog library with features:
		 * - Global log level configuration
		 * - Per-module log level configuration
		 * - Log file rotation
		 * - Rank-specific log files for distributed training
		 *
		 * @note Thread-safe Singleton implementation
		 */
		class Logger {
		public:
			// Get Singleton instance
			static std::shared_ptr<Logger> getInstance();

			// Disable copy and move
			Logger(const Logger&) = delete;
			Logger& operator=(const Logger&) = delete;
			Logger(Logger&&) = delete;
			Logger& operator=(Logger&&) = delete;

			// Set global log level
			void setGlobalLogLevel(LogLevel level);

			// Set per-module log level
			void setModuleLogLevel(const std::string& module, LogLevel level);

			// Set log file path
			void setLogFile(const std::string& filePath);

			// Enable log rotation (maxFileSize: bytes, maxFiles: number of files)
			void enableRotation(size_t maxFileSize, size_t maxFiles);

			// Set rank-specific log file for distributed training
			void setRankSpecificLog(int rank);

			// Log output functions
			void trace(const std::string& message);
			void trace(const std::string& module, const std::string& message);

			void debug(const std::string& message);
			void debug(const std::string& module, const std::string& message);

			void info(const std::string& message);
			void info(const std::string& module, const std::string& message);

			void warn(const std::string& message);
			void warn(const std::string& module, const std::string& message);

			void error(const std::string& message);
			void error(const std::string& module, const std::string& message);

			void fatal(const std::string& message);
			void fatal(const std::string& module, const std::string& message);

			// Get current global log level (thread-safe)
			LogLevel getGlobalLogLevel() const {
				std::lock_guard<std::recursive_mutex> lock(_mutex);
				return _globalLogLevel;
			}

			// Reset logger to console-only mode (closes file handles)
			void reset();

			// Public destructor (required for shared_ptr, construction still private)
			~Logger() = default;

		private:
			// Private constructor (Singleton)
			Logger();

			// spdlog logger instance
			std::shared_ptr<spdlog::logger> _spdlogger;

			// Global log level
			LogLevel _globalLogLevel;

			// Per-module log levels (module name -> log level)
			std::unordered_map<std::string, LogLevel> _moduleLogLevels;

			// Log file path
			std::string _logFilePath;

			// Rotation settings
			size_t _rotationSize;
			size_t _rotationFiles;
			bool _rotationEnabled;

			// Thread-safety (recursive to allow shouldLog to acquire lock within locked methods)
			mutable std::recursive_mutex _mutex;

			// Singleton instance
			static std::shared_ptr<Logger> _instance;
			static std::once_flag _initFlag;

			// Helper functions
			bool shouldLog(const std::string& module, LogLevel level) const;
			void initializeSpdlog();
			void recreateLogger();
			void logMessage(LogLevel level, const std::string& module, const std::string& message);
		};

	} // namespace Utils
} // namespace WheelDL
