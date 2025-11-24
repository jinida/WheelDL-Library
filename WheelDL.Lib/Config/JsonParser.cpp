#include "pch.h"
#include "JsonParser.h"
#include "Utils/Path/PathValidator.h"
#include "Utils/Error/ErrorCodes.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

namespace WheelDL {
namespace Config {

nlohmann::json JsonParser::parseFrom(const std::string& jsonPath) {
	using namespace Utils;

	// Check if file exists
	if (!PathValidator::fileExists(jsonPath)) {
		throw ConfigurationException(
			ErrorCode::CONFIG_FILE_NOT_FOUND,
			"JSON file not found: " + jsonPath
		);
	}

	// Check file size
	if (!checkFileSize(jsonPath)) {
		throw ConfigurationException(
			ErrorCode::CONFIG_PARSE_FAILED,
			"JSON file too large: " + jsonPath
		);
	}

	try {
		// Parse JSON file
		std::ifstream file(jsonPath);
		if (!file.is_open()) {
			throw ConfigurationException(
				ErrorCode::CONFIG_FILE_NOT_FOUND,
				"Failed to open JSON file: " + jsonPath
			);
		}

		nlohmann::json node = nlohmann::json::parse(file);

		// Validate node
		if (!validate(node)) {
			throw ConfigurationException(
				ErrorCode::CONFIG_PARSE_FAILED,
				"Invalid JSON structure in: " + jsonPath
			);
		}

		// Check depth
		if (!checkDepth(node)) {
			throw ConfigurationException(
				ErrorCode::CONFIG_PARSE_FAILED,
				"JSON depth exceeds maximum limit in: " + jsonPath
			);
		}

		return node;
	}
	catch (const nlohmann::json::parse_error& e) {
		throw ConfigurationException(
			ErrorCode::CONFIG_PARSE_FAILED,
			"JSON parse error in " + jsonPath + ": " + e.what()
		);
	}
	catch (const nlohmann::json::exception& e) {
		throw ConfigurationException(
			ErrorCode::CONFIG_PARSE_FAILED,
			"JSON error in " + jsonPath + ": " + e.what()
		);
	}
}

bool JsonParser::validate(const nlohmann::json& node) {
	return !node.is_null();
}

bool JsonParser::checkFileSize(const std::string& jsonPath, size_t maxSize) {
	try {
		auto fileSize = std::filesystem::file_size(jsonPath);
		return fileSize <= maxSize;
	}
	catch (...) {
		return false;
	}
}

bool JsonParser::checkDepth(const nlohmann::json& node, int maxDepth) {
	int depth = checkDepthRecursive(node, 0, maxDepth);
	return depth <= maxDepth;
}

int JsonParser::checkDepthRecursive(const nlohmann::json& node, int currentDepth, int maxDepth) {
	// Early exit if exceeded maxDepth
	if (currentDepth > maxDepth) {
		return currentDepth;
	}

	// Stop recursion at primitives (null, boolean, number, string)
	if (node.is_null() || node.is_boolean() || node.is_number() || node.is_string()) {
		return currentDepth;
	}

	int maxChildDepth = currentDepth;

	if (node.is_array()) {
		for (const auto& child : node) {
			int childDepth = checkDepthRecursive(child, currentDepth + 1, maxDepth);
			maxChildDepth = (std::max)(maxChildDepth, childDepth);
			// Early exit if already exceeded maxDepth
			if (maxChildDepth > maxDepth) {
				return maxChildDepth;
			}
		}
	}
	else if (node.is_object()) {
		for (const auto& [key, value] : node.items()) {
			int childDepth = checkDepthRecursive(value, currentDepth + 1, maxDepth);
			maxChildDepth = (std::max)(maxChildDepth, childDepth);
			// Early exit if already exceeded maxDepth
			if (maxChildDepth > maxDepth) {
				return maxChildDepth;
			}
		}
	}

	return maxChildDepth;
}

std::string JsonParser::getString(const nlohmann::json& node, const std::string& key, const std::string& defaultValue) {
	if (!validate(node) || !node.contains(key)) {
		return defaultValue;
	}

	try {
		return node[key].get<std::string>();
	}
	catch (const nlohmann::json::exception& e) {
		// JSON parsing error - key exists but conversion failed
		// TODO: Add logging here
		return defaultValue;
	}
	catch (const std::exception& e) {
		// Other errors (bad_cast, etc.)
		// TODO: Add logging here
		return defaultValue;
	}
}

int JsonParser::getInt(const nlohmann::json& node, const std::string& key, int defaultValue) {
	if (!validate(node) || !node.contains(key)) {
		return defaultValue;
	}

	try {
		return node[key].get<int>();
	}
	catch (const nlohmann::json::exception& e) {
		// JSON parsing error - key exists but conversion failed
		// TODO: Add logging here
		return defaultValue;
	}
	catch (const std::exception& e) {
		// Other errors (bad_cast, etc.)
		// TODO: Add logging here
		return defaultValue;
	}
}

float JsonParser::getFloat(const nlohmann::json& node, const std::string& key, float defaultValue) {
	if (!validate(node) || !node.contains(key)) {
		return defaultValue;
	}

	try {
		return node[key].get<float>();
	}
	catch (const nlohmann::json::exception& e) {
		// JSON parsing error - key exists but conversion failed
		// TODO: Add logging here
		return defaultValue;
	}
	catch (const std::exception& e) {
		// Other errors (bad_cast, etc.)
		// TODO: Add logging here
		return defaultValue;
	}
}

bool JsonParser::getBool(const nlohmann::json& node, const std::string& key, bool defaultValue) {
	if (!validate(node) || !node.contains(key)) {
		return defaultValue;
	}

	try {
		return node[key].get<bool>();
	}
	catch (const nlohmann::json::exception& e) {
		// JSON parsing error - key exists but conversion failed
		// TODO: Add logging here
		return defaultValue;
	}
	catch (const std::exception& e) {
		// Other errors (bad_cast, etc.)
		// TODO: Add logging here
		return defaultValue;
	}
}

std::map<int, std::string> JsonParser::getClassNames(const nlohmann::json& node) {
	std::map<int, std::string> classNames;

	if (!validate(node)) {
		return classNames;
	}

	// Support both "names" (YOLO format) and "class_names" (alternative format)
	nlohmann::json names;
	if (node.contains("names")) {
		names = node["names"];
	} else if (node.contains("class_names")) {
		names = node["class_names"];
	} else {
		return classNames;
	}

	try {
		if (names.is_object()) {
			// Format: {"0": "person", "1": "bicycle", ...}
			for (const auto& [key, value] : names.items()) {
				int classId = std::stoi(key);
				std::string className = value.get<std::string>();
				classNames[classId] = className;
			}
		}
		else if (names.is_array()) {
			// Format: ["person", "bicycle", ...]
			int classId = 0;
			for (const auto& name : names) {
				classNames[classId++] = name.get<std::string>();
			}
		}
	}
	catch (const nlohmann::json::exception& e) {
		// JSON parsing error - return empty map
		// TODO: Add logging here
		classNames.clear();
	}
	catch (const std::exception& e) {
		// Other errors - return empty map
		// TODO: Add logging here
		classNames.clear();
	}

	return classNames;
}

} // namespace Config
} // namespace WheelDL
