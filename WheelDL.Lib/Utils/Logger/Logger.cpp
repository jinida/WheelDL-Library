#include "pch.h"
#include "Logger.h"
#include "Utils/Path/PathValidator.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <iostream>

namespace WheelDL {
	namespace Utils {

		// Static member initialization
		std::shared_ptr<Logger> Logger::_instance = nullptr;
		std::once_flag Logger::_initFlag;

		Logger::Logger()
			: _globalLogLevel(LogLevel::INFO)
			, _rotationSize(0)
			, _rotationFiles(0)
			, _rotationEnabled(false)
		{
			initializeSpdlog();
		}

		std::shared_ptr<Logger> Logger::getInstance() {
			std::call_once(_initFlag, []() {
				_instance.reset(new Logger());
				});
			return _instance;
		}

		void Logger::initializeSpdlog() {
			try {
				// Create default console logger
				auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
				console_sink->set_level(spdlog::level::trace);

				_spdlogger = std::make_shared<spdlog::logger>("WheelDL", console_sink);
				_spdlogger->set_level(spdlog::level::info);
				_spdlogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

				// Register as default logger
				spdlog::set_default_logger(_spdlogger);
			}
			catch (const std::exception& e) {
				std::cerr << "CRITICAL: Failed to initialize logger: " << e.what() << std::endl;
				std::cerr << "Logging will be disabled. Application may continue but without logging." << std::endl;
				// Set _spdlogger to nullptr to ensure all logging checks fail gracefully
				_spdlogger = nullptr;
			}
		}

		void Logger::recreateLogger() {
			try {
				std::vector<spdlog::sink_ptr> sinks;

				// Console sink
				auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
				console_sink->set_level(spdlog::level::trace);
				sinks.push_back(console_sink);

				// File sink (with rotation or basic)
				if (!_logFilePath.empty()) {
					if (_rotationEnabled && _rotationSize > 0) {
						auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
							_logFilePath, _rotationSize, _rotationFiles);
						file_sink->set_level(spdlog::level::trace);
						sinks.push_back(file_sink);
					}
					else {
						auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(_logFilePath);
						file_sink->set_level(spdlog::level::trace);
						sinks.push_back(file_sink);
					}
				}

				_spdlogger = std::make_shared<spdlog::logger>("WheelDL", sinks.begin(), sinks.end());
				_spdlogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

				// Enable automatic flushing for immediate file writes
				_spdlogger->flush_on(spdlog::level::trace);

				// Apply global log level
				spdlog::level::level_enum spdLevel;
				switch (_globalLogLevel) {
				case LogLevel::TRACE: spdLevel = spdlog::level::trace; break;
				case LogLevel::DEBUG: spdLevel = spdlog::level::debug; break;
				case LogLevel::INFO:  spdLevel = spdlog::level::info; break;
				case LogLevel::WARN:  spdLevel = spdlog::level::warn; break;
				case LogLevel::ERR: spdLevel = spdlog::level::err; break;
				case LogLevel::FATAL: spdLevel = spdlog::level::critical; break;
				case LogLevel::OFF:   spdLevel = spdlog::level::off; break;
				default:              spdLevel = spdlog::level::info; break;
				}
				_spdlogger->set_level(spdLevel);

				spdlog::set_default_logger(_spdlogger);
			}
			catch (const std::exception& e) {
				std::cerr << "ERROR: Failed to recreate logger: " << e.what() << std::endl;
				std::cerr << "Logger configuration may be invalid. Reverting to console-only mode." << std::endl;
				// Attempt to create a fallback console-only logger
				try {
					auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
					_spdlogger = std::make_shared<spdlog::logger>("WheelDL", console_sink);
					_spdlogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
				}
				catch (...) {
					// If even fallback fails, disable logging
					_spdlogger = nullptr;
					std::cerr << "CRITICAL: Unable to create fallback logger. Logging disabled." << std::endl;
				}
			}
		}

		void Logger::setGlobalLogLevel(LogLevel level) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			_globalLogLevel = level;
			recreateLogger();
		}

		void Logger::reset() {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			_logFilePath.clear();
			_rotationEnabled = false;
			_rotationSize = 0;
			_rotationFiles = 0;
			recreateLogger();
		}

		void Logger::setModuleLogLevel(const std::string& module, LogLevel level) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			_moduleLogLevels[module] = level;
		}

		void Logger::setLogFile(const std::string& filePath) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			_logFilePath = filePath;
			recreateLogger();
		}

		void Logger::enableRotation(size_t maxFileSize, size_t maxFiles) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			_rotationSize = maxFileSize;
			_rotationFiles = maxFiles;
			_rotationEnabled = true;
			recreateLogger();
		}

		void Logger::setRankSpecificLog(int rank) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);

			// Handle negative ranks by creating unique log file name
			// This prevents multiple processes from writing to the same file
			std::string rankStr;
			if (rank < 0) {
				// Use timestamp to create unique identifier for negative ranks
				auto now = std::chrono::system_clock::now();
				auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
				rankStr = "neg_" + std::to_string(-rank) + "_" + std::to_string(timestamp);
				std::cerr << "Warning: Negative rank (" << rank << ") provided. Using unique identifier: " << rankStr << std::endl;
			} else {
				rankStr = std::to_string(rank);
			}

			// If no log file path is set, create default path
			if (_logFilePath.empty()) {
				_logFilePath = "rank_" + rankStr + ".log";
			}
			else {
				// Validate that the path doesn't contain invalid characters
				// Use PathValidator for consistent and proper path validation (handles Windows drive letters correctly)
				if (!PathValidator::isValidPath(_logFilePath)) {
					std::cerr << "Error: Log file path contains invalid characters or invalid path format" << std::endl;
					std::cerr << "Using default rank-specific log path instead." << std::endl;
					_logFilePath = "rank_" + rankStr + ".log";
					recreateLogger();
					return;
				}

				// Remove file extension and add rank
				size_t dotPos = _logFilePath.find_last_of('.');
				// Ensure dotPos is not at the beginning (hidden files on Unix)
				if (dotPos != std::string::npos && dotPos > 0) {
					_logFilePath = _logFilePath.substr(0, dotPos)
						+ "_rank_" + rankStr
						+ _logFilePath.substr(dotPos);
				}
				else {
					_logFilePath += "_rank_" + rankStr;
				}
			}
			recreateLogger();
		}

		bool Logger::shouldLog(const std::string& module, LogLevel level) const {
			// Thread-safe: acquire lock to protect access to _moduleLogLevels and _globalLogLevel
			// Uses recursive_mutex to allow this to be called from within other locked methods
			std::lock_guard<std::recursive_mutex> lock(_mutex);

			// Check module-specific log level
			if (!module.empty()) {
				auto it = _moduleLogLevels.find(module);
				if (it != _moduleLogLevels.end()) {
					return static_cast<int>(level) >= static_cast<int>(it->second);
				}
			}

			// Check global log level
			return static_cast<int>(level) >= static_cast<int>(_globalLogLevel);
		}

		// Helper macro to reduce code duplication in logging
		#define WHEEL_LOG_WITH_LEVEL(spdlog_func) \
			do { \
				if (module.empty()) { \
					_spdlogger->spdlog_func(message); \
				} else { \
					_spdlogger->spdlog_func("[{}] {}", module, message); \
				} \
			} while(0)

		// Helper function to reduce code duplication
		void Logger::logMessage(LogLevel level, const std::string& module, const std::string& message) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			if (!_spdlogger || !shouldLog(module, level)) {
				return;
			}

			// Map LogLevel to spdlog level and log function
			switch (level) {
			case LogLevel::TRACE:
				WHEEL_LOG_WITH_LEVEL(trace);
				break;
			case LogLevel::DEBUG:
				WHEEL_LOG_WITH_LEVEL(debug);
				break;
			case LogLevel::INFO:
				WHEEL_LOG_WITH_LEVEL(info);
				break;
			case LogLevel::WARN:
				WHEEL_LOG_WITH_LEVEL(warn);
				break;
			case LogLevel::ERR:
				WHEEL_LOG_WITH_LEVEL(error);
				break;
			case LogLevel::FATAL:
				WHEEL_LOG_WITH_LEVEL(critical);
				break;
			default:
				break;
			}
		}

		#undef WHEEL_LOG_WITH_LEVEL

		// TRACE level logging
		void Logger::trace(const std::string& message) {
			logMessage(LogLevel::TRACE, "", message);
		}

		void Logger::trace(const std::string& module, const std::string& message) {
			logMessage(LogLevel::TRACE, module, message);
		}

		// DEBUG level logging
		void Logger::debug(const std::string& message) {
			logMessage(LogLevel::DEBUG, "", message);
		}

		void Logger::debug(const std::string& module, const std::string& message) {
			logMessage(LogLevel::DEBUG, module, message);
		}

		// INFO level logging
		void Logger::info(const std::string& message) {
			logMessage(LogLevel::INFO, "", message);
		}

		void Logger::info(const std::string& module, const std::string& message) {
			logMessage(LogLevel::INFO, module, message);
		}

		// WARN level logging
		void Logger::warn(const std::string& message) {
			logMessage(LogLevel::WARN, "", message);
		}

		void Logger::warn(const std::string& module, const std::string& message) {
			logMessage(LogLevel::WARN, module, message);
		}

		// ERROR level logging
		void Logger::error(const std::string& message) {
			logMessage(LogLevel::ERR, "", message);
		}

		void Logger::error(const std::string& module, const std::string& message) {
			logMessage(LogLevel::ERR, module, message);
		}

		// FATAL level logging
		void Logger::fatal(const std::string& message) {
			logMessage(LogLevel::FATAL, "", message);
		}

		void Logger::fatal(const std::string& module, const std::string& message) {
			logMessage(LogLevel::FATAL, module, message);
		}

	} // namespace Utils
} // namespace WheelDL
