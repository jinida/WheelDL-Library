#pragma once

#include "Utils/Common/Types.h"
#include "Utils/Error/WheelLibException.h"
#include <string>
#include <vector>
#include <map>

// Forward declare yaml-cpp types to avoid exposing implementation details
namespace YAML {
	class Node;
}

namespace WheelDL {
	namespace Config {

		/**
		 * @class YamlParser
		 * @brief Static utility class for parsing and validating YAML configuration files
		 *
		 * Provides methods for parsing YAML files and inferring task types from model configurations.
		 * Uses yaml-cpp library internally.
		 */
		class YamlParser {
		public:
			/**
			 * @brief Parse YAML file and return root node
			 * @param yamlPath Path to YAML file
			 * @return YAML::Node Root node of parsed YAML
			 * @throws ConfigurationException if file not found or parse fails
			 */
			static YAML::Node parseFrom(const std::string& yamlPath);

			/**
			 * @brief Validate YAML node structure
			 * @param node YAML node to validate
			 * @return bool True if node is valid (non-null and defined)
			 */
			static bool validate(const YAML::Node& node);

			/**
			 * @brief Infer task type from model configuration
			 * @param modelConfig Model YAML node
			 * @return TaskType Inferred task type
			 *
			 * Inference rules:
			 * - If head contains "Detect": DETECTION
			 * - If head contains "Segment": SEGMENTATION
			 * - If head contains "OBB": OBB
			 * - If head contains "Classify": CLASSIFICATION
			 * - Otherwise: UNKNOWN
			 */
			static TaskType inferTaskTypeFrom(const YAML::Node& modelConfig);

			/**
			 * @brief Check if YAML file exists and is not too large
			 * @param yamlPath Path to YAML file
			 * @param maxSize Maximum file size in bytes (default: 10MB)
			 * @return bool True if file is valid
			 */
			static bool checkFileSize(const std::string& yamlPath, size_t maxSize = 10 * 1024 * 1024);

			/**
			 * @brief Check YAML node depth to prevent stack overflow
			 * @param node YAML node to check
			 * @param maxDepth Maximum allowed depth (default: 100)
			 * @return bool True if depth is within limits
			 */
			static bool checkDepth(const YAML::Node& node, int maxDepth = 100);

			/**
			 * @brief Get string value from YAML node with default
			 * @param node YAML node
			 * @param key Key to lookup
			 * @param defaultValue Default value if key not found
			 * @return std::string Value or default
			 */
			static std::string getString(const YAML::Node& node, const std::string& key, const std::string& defaultValue = "");

			/**
			 * @brief Get integer value from YAML node with default
			 * @param node YAML node
			 * @param key Key to lookup
			 * @param defaultValue Default value if key not found
			 * @return int Value or default
			 */
			static int getInt(const YAML::Node& node, const std::string& key, int defaultValue = 0);

			/**
			 * @brief Get float value from YAML node with default
			 * @param node YAML node
			 * @param key Key to lookup
			 * @param defaultValue Default value if key not found
			 * @return float Value or default
			 */
			static float getFloat(const YAML::Node& node, const std::string& key, float defaultValue = 0.0f);

			/**
			 * @brief Get boolean value from YAML node with default
			 * @param node YAML node
			 * @param key Key to lookup
			 * @param defaultValue Default value if key not found
			 * @return bool Value or default
			 */
			static bool getBool(const YAML::Node& node, const std::string& key, bool defaultValue = false);

			/**
			 * @brief Get class names map from dataset YAML
			 * @param node Dataset YAML node
			 * @return std::map<int, std::string> Class ID to name mapping
			 */
			static std::map<int, std::string> getClassNames(const YAML::Node& node);

		private:
			// Static utility class - no instances allowed
			YamlParser() = delete;
			~YamlParser() = delete;
			YamlParser(const YamlParser&) = delete;
			YamlParser& operator=(const YamlParser&) = delete;

			/**
			 * @brief Internal recursive depth checker
			 * @param node YAML node
			 * @param currentDepth Current recursion depth
			 * @param maxDepth Maximum allowed depth
			 * @return int Actual depth
			 */
			static int checkDepthRecursive(const YAML::Node& node, int currentDepth, int maxDepth);
		};

	} // namespace Config
} // namespace WheelDL
