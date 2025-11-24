#pragma once

#include <string>
#include <memory>
#include "../Common/Types.h"

namespace WheelDL {
    namespace Utils {

        /**
         * @class Workspace
         * @brief Manages training workspace directory structure
         *
         * Creates and manages a timestamped workspace directory for training runs.
         * Automatically creates subdirectories for weights, logs, profiler reports, etc.
         *
         * Directory structure:
         *   runs/train_YYYYMMDD_HHMMSS/
         *     ├── weights/          (model checkpoints)
         *     ├── logs/             (training logs)
         *     ├── profiler/         (performance profiler reports)
         *     └── visualizations/   (plots, graphs, etc.)
         */
        class Workspace {
        public:
            /**
             * @brief Constructor - creates a new timestamped workspace
             * @param baseDir Base directory for all workspaces (default: "runs")
             * @param prefix Prefix for workspace directory name (default: "train")
             * @throws WheelLibException if directory creation fails
             */
            explicit Workspace(
                const std::string& baseDir = "runs",
                const std::string& prefix = "train"
            );

            /**
             * @brief Constructor with custom workspace name
             * @param baseDir Base directory for all workspaces
             * @param customName Custom name for workspace (no timestamp)
             * @param useTimestamp Whether to append timestamp to custom name
             * @throws WheelLibException if directory creation fails
             */
            Workspace(
                const std::string& baseDir,
                const std::string& customName,
                bool useTimestamp
            );

            /**
             * @brief Destructor
             */
            ~Workspace() = default;

            // Disable copy
            Workspace(const Workspace&) = delete;
            Workspace& operator=(const Workspace&) = delete;

            // Allow move
            Workspace(Workspace&&) = default;
            Workspace& operator=(Workspace&&) = default;

            /**
             * @brief Get the root directory of this workspace
             * @return Absolute path to workspace root
             */
            std::string getRoot() const { return _workspaceRoot; }

            /**
             * @brief Get the weights directory path
             * @return Absolute path to weights directory
             */
            std::string getWeightsDir() const { return _weightsDir; }

            /**
             * @brief Get the logs directory path
             * @return Absolute path to logs directory
             */
            std::string getLogsDir() const { return _logsDir; }

            /**
             * @brief Get the profiler directory path
             * @return Absolute path to profiler directory
             */
            std::string getProfilerDir() const { return _profilerDir; }

            /**
             * @brief Create a custom subdirectory in the workspace
             * @param subDirName Subdirectory name
             * @return Absolute path to created subdirectory
             * @throws WheelLibException if creation fails
             */
            std::string createSubDirectory(const std::string& subDirName);

            /**
             * @brief Get the timestamp used for this workspace
             * @return Timestamp string (YYYYMMDD_HHMMSS format)
             */
            std::string getTimestamp() const { return _timestamp; }

            /**
             * @brief Check if workspace exists and is valid
             * @return true if workspace directory exists, false otherwise
             */
            bool exists() const;

        private:
            /**
             * @brief Generate timestamp string in YYYYMMDD_HHMMSS format
             * @return Timestamp string
             */
            static std::string generateTimestamp();

            /**
             * @brief Create all standard subdirectories
             */
            void createStandardDirectories();

            std::string _workspaceRoot;
            std::string _weightsDir;
            std::string _logsDir;
            std::string _profilerDir;
            std::string _timestamp;
        };

    } // namespace Utils
} // namespace WheelDL
