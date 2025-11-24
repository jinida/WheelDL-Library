#pragma once

#include "Utils/Common/Types.h"
#include "Utils/Error/WheelLibException.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>

namespace WheelDL {
	namespace Config {

		/**
		 * @class JsonParser
		 * @brief Static utility class for parsing and validating JSON configuration files
		 *
		 * Provides methods for parsing JSON files and extracting configuration values.
		 * Uses nlohmann/json library internally.
		 */
		class JsonParser {
		public:
			/**
			 * @brief Parse JSON file and return root object
			 * @param jsonPath Path to JSON file
			 * @return nlohmann::json Root object of parsed JSON
			 * @throws ConfigurationException if file not found or parse fails
			 */
			static nlohmann::json parseFrom(const std::string& jsonPath);

			/**
			 * @brief Validate JSON object structure
			 * @param node JSON object to validate
			 * @return bool True if object is valid (not null)
			 */
			static bool validate(const nlohmann::json& node);

			/**
			 * @brief Check if JSON file exists and is not too large
			 * @param jsonPath Path to JSON file
			 * @param maxSize Maximum file size in bytes (default: 10MB)
			 * @return bool True if file is valid
			 */
			static bool checkFileSize(const std::string& jsonPath, size_t maxSize = 1024 * 1024 * 1024);

			/**
			 * @brief Check JSON object depth to prevent stack overflow
			 * @param node JSON object to check
			 * @param maxDepth Maximum allowed depth (default: 100)
			 * @return bool True if depth is within limits
			 */
			static bool checkDepth(const nlohmann::json& node, int maxDepth = 100);

			/**
			 * @brief Get string value from JSON object with default
			 * @param node JSON object
			 * @param key Key to lookup
			 * @param defaultValue Default value if key not found
			 * @return std::string Value or default
			 */
			static std::string getString(const nlohmann::json& node, const std::string& key, const std::string& defaultValue = "");

			/**
			 * @brief Get integer value from JSON object with default
			 * @param node JSON object
			 * @param key Key to lookup
			 * @param defaultValue Default value if key not found
			 * @return int Value or default
			 */
			static int getInt(const nlohmann::json& node, const std::string& key, int defaultValue = 0);

			/**
			 * @brief Get float value from JSON object with default
			 * @param node JSON object
			 * @param key Key to lookup
			 * @param defaultValue Default value if key not found
			 * @return float Value or default
			 */
			static float getFloat(const nlohmann::json& node, const std::string& key, float defaultValue = 0.0f);

			/**
			 * @brief Get boolean value from JSON object with default
			 * @param node JSON object
			 * @param key Key to lookup
			 * @param defaultValue Default value if key not found
			 * @return bool Value or default
			 */
			static bool getBool(const nlohmann::json& node, const std::string& key, bool defaultValue = false);

			/**
			 * @brief Get class names map from dataset JSON
			 * @param node Dataset JSON object
			 * @return std::map<int, std::string> Class ID to name mapping
			 */
			static std::map<int, std::string> getClassNames(const nlohmann::json& node);

		private:
			// Static utility class - no instances allowed
			JsonParser() = delete;
			~JsonParser() = delete;
			JsonParser(const JsonParser&) = delete;
			JsonParser& operator=(const JsonParser&) = delete;

			/**
			 * @brief Internal recursive depth checker
			 * @param node JSON object
			 * @param currentDepth Current recursion depth
			 * @param maxDepth Maximum allowed depth
			 * @return int Actual depth
			 */
			static int checkDepthRecursive(const nlohmann::json& node, int currentDepth, int maxDepth);
		};

	} // namespace Config
} // namespace WheelDL
