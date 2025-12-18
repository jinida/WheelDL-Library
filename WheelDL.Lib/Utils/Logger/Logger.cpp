#include "pch.h"
#include "Logger.h"
#include "Utils/Path/PathValidator.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace WheelDL {
	namespace Utils {

		// ========== Helper Functions ==========

		namespace {
			std::string generateTimestampedLogPath() {
				auto now = std::chrono::system_clock::now();
				auto time_t_now = std::chrono::system_clock::to_time_t(now);
				auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
					now.time_since_epoch()) % 1000;

				std::tm tm_now;
#ifdef _WIN32
				localtime_s(&tm_now, &time_t_now);
#else
				localtime_r(&time_t_now, &tm_now);
#endif

				std::ostringstream oss;
				oss << "runs/app/app_"
					<< std::put_time(&tm_now, "%y%m%d%H%M%S")
					<< std::setfill('0') << std::setw(3) << ms.count()
					<< ".log";
				return oss.str();
			}
		}

		// ========== Factory Methods ==========

		std::unique_ptr<Logger> Logger::create(const std::string& name) {
			return std::unique_ptr<Logger>(new Logger(name));
		}

		Logger& Logger::getDefault() {
			static Logger defaultLogger("default");
			static bool initialized = false;
			if (!initialized) {
				// Create runs/app directory
				PathValidator::createDirectoryIfNotExists("runs/app");
				// Generate timestamped log file path
				defaultLogger.setLogFile(generateTimestampedLogPath());
				initialized = true;
			}
			return defaultLogger;
		}

		// ========== Constructor / Destructor ==========

		Logger::Logger(const std::string& name)
			: _name(name)
			, _globalLogLevel(LogLevel::INFO)
			, _rotationSize(0)
			, _rotationFiles(0)
			, _rotationEnabled(false)
		{
			initializeSpdlog();
		}

		Logger::~Logger() {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			if (_spdlogger) {
				_spdlogger->flush();
				spdlog::drop(_name);
			}
		}

		// ========== Initialization ==========

		void Logger::initializeSpdlog() {
			try {
#ifdef WHEELDL_STATIC
				// Debug/ReleaseLib: Enable console output
				auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
				console_sink->set_level(spdlog::level::trace);
				_spdlogger = std::make_shared<spdlog::logger>(_name, console_sink);
#else
				// Release DLL: No console output
				_spdlogger = std::make_shared<spdlog::logger>(_name);
#endif
				_spdlogger->set_level(spdlog::level::info);
				_spdlogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");

				spdlog::register_logger(_spdlogger);
			}
			catch (const std::exception& e) {
#ifdef WHEELDL_STATIC
				std::cerr << "CRITICAL: Failed to initialize logger '" << _name << "': " << e.what() << std::endl;
#endif
				_spdlogger = nullptr;
			}
		}

		void Logger::recreateLogger() {
			try {
				if (_spdlogger) {
					_spdlogger->flush();
					spdlog::drop(_name);
				}

				std::vector<spdlog::sink_ptr> sinks;

#ifdef WHEELDL_STATIC
				// Debug/ReleaseLib: Enable console output
				auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
				console_sink->set_level(spdlog::level::trace);
				sinks.push_back(console_sink);
#endif

				if (!_logFilePath.empty()) {
					// Extract parent directory from file path and create it
					std::string parentDir = PathValidator::getDirectory(_logFilePath);
					if (!parentDir.empty()) {
						PathValidator::createDirectoryIfNotExists(parentDir);
					}

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

				_spdlogger = std::make_shared<spdlog::logger>(_name, sinks.begin(), sinks.end());
				_spdlogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
				_spdlogger->flush_on(spdlog::level::trace);

				// Apply log level
				spdlog::level::level_enum spdLevel;
				switch (_globalLogLevel) {
				case LogLevel::TRACE: spdLevel = spdlog::level::trace; break;
				case LogLevel::DEBUG: spdLevel = spdlog::level::debug; break;
				case LogLevel::INFO:  spdLevel = spdlog::level::info; break;
				case LogLevel::WARN:  spdLevel = spdlog::level::warn; break;
				case LogLevel::ERR:   spdLevel = spdlog::level::err; break;
				case LogLevel::FATAL: spdLevel = spdlog::level::critical; break;
				case LogLevel::OFF:   spdLevel = spdlog::level::off; break;
				default:              spdLevel = spdlog::level::info; break;
				}
				_spdlogger->set_level(spdLevel);

				spdlog::register_logger(_spdlogger);
			}
			catch (const std::exception& e) {
#ifdef WHEELDL_STATIC
				std::cerr << "ERROR: Failed to recreate logger '" << _name << "': " << e.what() << std::endl;
#endif
				// Fallback
				try {
#ifdef WHEELDL_STATIC
					// Debug/ReleaseLib: Fallback to console-only
					auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
					_spdlogger = std::make_shared<spdlog::logger>(_name, console_sink);
#else
					// Release DLL: Fallback to empty logger
					_spdlogger = std::make_shared<spdlog::logger>(_name);
#endif
					_spdlogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
				}
				catch (...) {
					_spdlogger = nullptr;
#ifdef WHEELDL_STATIC
					std::cerr << "CRITICAL: Unable to create fallback logger. Logging disabled." << std::endl;
#endif
				}
			}
		}

		// ========== Configuration ==========

		void Logger::setLogFile(const std::string& filePath) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			_logFilePath = filePath;
			recreateLogger();
		}

		void Logger::setGlobalLogLevel(LogLevel level) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			_globalLogLevel = level;
			recreateLogger();
		}

		void Logger::setModuleLogLevel(const std::string& module, LogLevel level) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			_moduleLogLevels[module] = level;
		}

		void Logger::enableRotation(size_t maxFileSize, size_t maxFiles) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			_rotationSize = maxFileSize;
			_rotationFiles = maxFiles;
			_rotationEnabled = true;
			recreateLogger();
		}

		void Logger::reset() {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			_logFilePath.clear();
			_rotationEnabled = false;
			_rotationSize = 0;
			_rotationFiles = 0;
			_moduleLogLevels.clear();
			recreateLogger();
		}

		// ========== Query ==========

		std::string Logger::getName() const {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			return _name;
		}

		LogLevel Logger::getGlobalLogLevel() const {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			return _globalLogLevel;
		}

		bool Logger::isValid() const {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			return _spdlogger != nullptr;
		}

		// ========== Internal Helpers ==========

		bool Logger::shouldLog(const std::string& module, LogLevel level) const {
			// Note: caller must hold lock
			if (!module.empty()) {
				auto it = _moduleLogLevels.find(module);
				if (it != _moduleLogLevels.end()) {
					return static_cast<int>(level) >= static_cast<int>(it->second);
				}
			}
			return static_cast<int>(level) >= static_cast<int>(_globalLogLevel);
		}

		void Logger::logImpl(LogLevel level, const std::string& module, const std::string& message) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			if (!_spdlogger || !shouldLog(module, level)) {
				return;
			}

			std::string formatted = module.empty() ? message : "[" + module + "] " + message;

			switch (level) {
			case LogLevel::TRACE: _spdlogger->trace(formatted); break;
			case LogLevel::DEBUG: _spdlogger->debug(formatted); break;
			case LogLevel::INFO:  _spdlogger->info(formatted); break;
			case LogLevel::WARN:  _spdlogger->warn(formatted); break;
			case LogLevel::ERR:   _spdlogger->error(formatted); break;
			case LogLevel::FATAL: _spdlogger->critical(formatted); break;
			default: break;
			}
		}

		// ========== Logging (Simple String) ==========

		void Logger::trace(const std::string& module, const std::string& message) {
			logImpl(LogLevel::TRACE, module, message);
		}

		void Logger::debug(const std::string& module, const std::string& message) {
			logImpl(LogLevel::DEBUG, module, message);
		}

		void Logger::info(const std::string& module, const std::string& message) {
			logImpl(LogLevel::INFO, module, message);
		}

		void Logger::warn(const std::string& module, const std::string& message) {
			logImpl(LogLevel::WARN, module, message);
		}

		void Logger::error(const std::string& module, const std::string& message) {
			logImpl(LogLevel::ERR, module, message);
		}

		void Logger::fatal(const std::string& module, const std::string& message) {
			logImpl(LogLevel::FATAL, module, message);
		}

	} // namespace Utils
} // namespace WheelDL
