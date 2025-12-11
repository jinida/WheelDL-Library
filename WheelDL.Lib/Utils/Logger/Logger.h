#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <spdlog/spdlog.h>

namespace WheelDL {
	namespace Utils {

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
		 * @brief Ownership-based logging system using unique_ptr
		 *
		 * Design principles:
		 * - Created via factory as unique_ptr (Logger::create)
		 * - Automatic destruction when owner is destroyed (RAII)
		 * - Pass raw pointer for sharing (no ownership transfer)
		 * - Copy/move disabled (mutex thread-safety)
		 *
		 * Usage:
		 * @code
		 * // Create task-specific Logger (owned by Launcher)
		 * auto logger = Logger::create("task_001");
		 * logger->setLogFile("logs/task_001.log");
		 * trainer->setLogger(logger.get());  // raw ptr
		 *
		 * // System default Logger
		 * Logger::getDefault().info("System", "Starting...");
		 * @endcode
		 */
		class Logger {
		public:
			// ========== Factory ==========

			/**
			 * @brief Create new Logger instance
			 * @param name Logger name (for output and identification)
			 * @return unique_ptr<Logger> owned Logger
			 */
			static std::unique_ptr<Logger> create(const std::string& name);

			/**
			 * @brief Get system default Logger
			 * @return Logger& reference to static lifetime default Logger
			 */
			static Logger& getDefault();

			// ========== Non-copyable, Non-movable ==========
			Logger(const Logger&) = delete;
			Logger& operator=(const Logger&) = delete;
			Logger(Logger&&) = delete;
			Logger& operator=(Logger&&) = delete;

			~Logger();

			// ========== Configuration ==========

			void setLogFile(const std::string& filePath);
			void setGlobalLogLevel(LogLevel level);
			void setModuleLogLevel(const std::string& module, LogLevel level);
			void enableRotation(size_t maxFileSize, size_t maxFiles);

			// ========== Logging (fmt style) ==========

			template<typename... Args>
			void trace(const std::string& module, const std::string& fmt, Args&&... args);

			template<typename... Args>
			void debug(const std::string& module, const std::string& fmt, Args&&... args);

			template<typename... Args>
			void info(const std::string& module, const std::string& fmt, Args&&... args);

			template<typename... Args>
			void warn(const std::string& module, const std::string& fmt, Args&&... args);

			template<typename... Args>
			void error(const std::string& module, const std::string& fmt, Args&&... args);

			template<typename... Args>
			void fatal(const std::string& module, const std::string& fmt, Args&&... args);

			// ========== Logging (simple string) ==========

			void trace(const std::string& module, const std::string& message);
			void debug(const std::string& module, const std::string& message);
			void info(const std::string& module, const std::string& message);
			void warn(const std::string& module, const std::string& message);
			void error(const std::string& module, const std::string& message);
			void fatal(const std::string& module, const std::string& message);

			// ========== Query ==========

			std::string getName() const;
			LogLevel getGlobalLogLevel() const;
			bool isValid() const;

			void reset();

		private:
			explicit Logger(const std::string& name);

			void initializeSpdlog();
			void recreateLogger();
			bool shouldLog(const std::string& module, LogLevel level) const;
			void logImpl(LogLevel level, const std::string& module, const std::string& message);

			std::string _name;
			std::shared_ptr<spdlog::logger> _spdlogger;
			LogLevel _globalLogLevel;
			std::unordered_map<std::string, LogLevel> _moduleLogLevels;
			std::string _logFilePath;
			size_t _rotationSize;
			size_t _rotationFiles;
			bool _rotationEnabled;
			mutable std::recursive_mutex _mutex;
		};

		// ========== Template Implementation ==========

		template<typename... Args>
		void Logger::trace(const std::string& module, const std::string& fmt, Args&&... args) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			if (!_spdlogger || !shouldLog(module, LogLevel::TRACE)) return;
			_spdlogger->trace("[{}] " + fmt, module, std::forward<Args>(args)...);
		}

		template<typename... Args>
		void Logger::debug(const std::string& module, const std::string& fmt, Args&&... args) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			if (!_spdlogger || !shouldLog(module, LogLevel::DEBUG)) return;
			_spdlogger->debug("[{}] " + fmt, module, std::forward<Args>(args)...);
		}

		template<typename... Args>
		void Logger::info(const std::string& module, const std::string& fmt, Args&&... args) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			if (!_spdlogger || !shouldLog(module, LogLevel::INFO)) return;
			_spdlogger->info("[{}] " + fmt, module, std::forward<Args>(args)...);
		}

		template<typename... Args>
		void Logger::warn(const std::string& module, const std::string& fmt, Args&&... args) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			if (!_spdlogger || !shouldLog(module, LogLevel::WARN)) return;
			_spdlogger->warn("[{}] " + fmt, module, std::forward<Args>(args)...);
		}

		template<typename... Args>
		void Logger::error(const std::string& module, const std::string& fmt, Args&&... args) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			if (!_spdlogger || !shouldLog(module, LogLevel::ERR)) return;
			_spdlogger->error("[{}] " + fmt, module, std::forward<Args>(args)...);
		}

		template<typename... Args>
		void Logger::fatal(const std::string& module, const std::string& fmt, Args&&... args) {
			std::lock_guard<std::recursive_mutex> lock(_mutex);
			if (!_spdlogger || !shouldLog(module, LogLevel::FATAL)) return;
			_spdlogger->critical("[{}] " + fmt, module, std::forward<Args>(args)...);
		}

	} // namespace Utils
} // namespace WheelDL
