#include "pch.h"
#include "YamlParser.h"
#include "Utils/Path/PathValidator.h"
#include "Utils/Error/ErrorCodes.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <filesystem>

namespace WheelDL {
namespace Config {

YAML::Node YamlParser::parseFrom(const std::string& yamlPath) {
	using namespace Utils;

	// Check if file exists
	if (!PathValidator::fileExists(yamlPath)) {
		throw ConfigurationException(
			ErrorCode::CONFIG_FILE_NOT_FOUND,
			"YAML file not found: " + yamlPath
		);
	}

	// Check file size
	if (!checkFileSize(yamlPath)) {
		throw ConfigurationException(
			ErrorCode::CONFIG_PARSE_FAILED,
			"YAML file too large: " + yamlPath
		);
	}

	try {
		// Parse YAML file
		YAML::Node node = YAML::LoadFile(yamlPath);

		// Validate node
		if (!validate(node)) {
			throw ConfigurationException(
				ErrorCode::CONFIG_PARSE_FAILED,
				"Invalid YAML structure in: " + yamlPath
			);
		}

		// Check depth
		if (!checkDepth(node)) {
			throw ConfigurationException(
				ErrorCode::CONFIG_PARSE_FAILED,
				"YAML depth exceeds maximum limit in: " + yamlPath
			);
		}

		return node;
	}
	catch (const YAML::Exception& e) {
		throw ConfigurationException(
			ErrorCode::CONFIG_PARSE_FAILED,
			"YAML parse error in " + yamlPath + ": " + e.what()
		);
	}
}

bool YamlParser::validate(const YAML::Node& node) {
	return node.IsDefined() && !node.IsNull();
}

TaskType YamlParser::inferTaskTypeFrom(const YAML::Node& modelConfig) {
	if (!validate(modelConfig)) {
		return TaskType::UNKNOWN;
	}

	// Check if head section exists
	if (!modelConfig["head"]) {
		return TaskType::UNKNOWN;
	}

	const YAML::Node& head = modelConfig["head"];

	// Head can be a sequence of layers
	if (head.IsSequence()) {
		// Iterate through head layers to find task indicator
		for (const auto& layer : head) {
			// Ensure layer is a sequence with at least 3 elements and layer[2] is a scalar
			if (layer.IsSequence() && layer.size() >= 3 && layer[2].IsScalar()) {
				std::string moduleName = layer[2].as<std::string>("");

				if (moduleName.find("Detect") != std::string::npos) {
					return TaskType::DETECTION;
				}
				else if (moduleName.find("Segment") != std::string::npos) {
					return TaskType::SEGMENTATION;
				}
				else if (moduleName.find("OBB") != std::string::npos) {
					return TaskType::OBB;
				}
				else if (moduleName.find("Classify") != std::string::npos ||
					moduleName.find("ClassificationHead") != std::string::npos) {
					return TaskType::CLASSIFICATION;
				}
				else if (moduleName.find("Anomaly") != std::string::npos) {
					return TaskType::ANOMALY;
				}
			}
		}
	}
	// Head can be a map with type field
	else if (head.IsMap()) {
		if (head["type"]) {
			std::string headType = head["type"].as<std::string>("");

			if (headType.find("detect") != std::string::npos ||
				headType.find("detection") != std::string::npos) {
				return TaskType::DETECTION;
			}
			else if (headType.find("segment") != std::string::npos) {
				return TaskType::SEGMENTATION;
			}
			else if (headType.find("obb") != std::string::npos) {
				return TaskType::OBB;
			}
			else if (headType.find("classify") != std::string::npos ||
				headType.find("classification") != std::string::npos) {
				return TaskType::CLASSIFICATION;
			}
			else if (headType.find("anomaly") != std::string::npos) {
				return TaskType::ANOMALY;
			}
		}
	}

	return TaskType::UNKNOWN;
}

bool YamlParser::checkFileSize(const std::string& yamlPath, size_t maxSize) {
	try {
		auto fileSize = std::filesystem::file_size(yamlPath);
		return fileSize <= maxSize;
	}
	catch (...) {
		return false;
	}
}

bool YamlParser::checkDepth(const YAML::Node& node, int maxDepth) {
	int depth = checkDepthRecursive(node, 0, maxDepth);
	return depth <= maxDepth;
}

int YamlParser::checkDepthRecursive(const YAML::Node& node, int currentDepth, int maxDepth) {
	// Early exit if exceeded maxDepth
	if (currentDepth > maxDepth) {
		return currentDepth;
	}

	// Stop recursion at scalars, but continue for undefined/null nodes in collections
	if (node.IsScalar()) {
		return currentDepth;
	}

	// If undefined or null and not in a collection context, return current depth
	if (!node.IsDefined() || node.IsNull()) {
		return currentDepth;
	}

	int maxChildDepth = currentDepth;

	if (node.IsSequence()) {
		for (const auto& child : node) {
			int childDepth = checkDepthRecursive(child, currentDepth + 1, maxDepth);
			maxChildDepth = (std::max)(maxChildDepth, childDepth);
			// Early exit if already exceeded maxDepth
			if (maxChildDepth > maxDepth) {
				return maxChildDepth;
			}
		}
	}
	else if (node.IsMap()) {
		for (const auto& kv : node) {
			int childDepth = checkDepthRecursive(kv.second, currentDepth + 1, maxDepth);
			maxChildDepth = (std::max)(maxChildDepth, childDepth);
			// Early exit if already exceeded maxDepth
			if (maxChildDepth > maxDepth) {
				return maxChildDepth;
			}
		}
	}

	return maxChildDepth;
}

std::string YamlParser::getString(const YAML::Node& node, const std::string& key, const std::string& defaultValue) {
	if (!validate(node) || !node[key]) {
		return defaultValue;
	}

	try {
		return node[key].as<std::string>();
	}
	catch (const YAML::Exception& e) {
		// YAML parsing error - key exists but conversion failed
		// TODO: Add logging here
		return defaultValue;
	}
	catch (const std::exception& e) {
		// Other errors (bad_cast, etc.)
		// TODO: Add logging here
		return defaultValue;
	}
}

int YamlParser::getInt(const YAML::Node& node, const std::string& key, int defaultValue) {
	if (!validate(node) || !node[key]) {
		return defaultValue;
	}

	try {
		return node[key].as<int>();
	}
	catch (const YAML::Exception& e) {
		// YAML parsing error - key exists but conversion failed
		// TODO: Add logging here
		return defaultValue;
	}
	catch (const std::exception& e) {
		// Other errors (bad_cast, etc.)
		// TODO: Add logging here
		return defaultValue;
	}
}

float YamlParser::getFloat(const YAML::Node& node, const std::string& key, float defaultValue) {
	if (!validate(node) || !node[key]) {
		return defaultValue;
	}

	try {
		return node[key].as<float>();
	}
	catch (const YAML::Exception& e) {
		// YAML parsing error - key exists but conversion failed
		// TODO: Add logging here
		return defaultValue;
	}
	catch (const std::exception& e) {
		// Other errors (bad_cast, etc.)
		// TODO: Add logging here
		return defaultValue;
	}
}

bool YamlParser::getBool(const YAML::Node& node, const std::string& key, bool defaultValue) {
	if (!validate(node) || !node[key]) {
		return defaultValue;
	}

	try {
		return node[key].as<bool>();
	}
	catch (const YAML::Exception& e) {
		// YAML parsing error - key exists but conversion failed
		// TODO: Add logging here
		return defaultValue;
	}
	catch (const std::exception& e) {
		// Other errors (bad_cast, etc.)
		// TODO: Add logging here
		return defaultValue;
	}
}

std::map<int, std::string> YamlParser::getClassNames(const YAML::Node& node) {
	std::map<int, std::string> classNames;

	if (!validate(node)) {
		return classNames;
	}

	// Support both "names" (YOLO format) and "class_names" (alternative format)
	YAML::Node names;
	if (node["names"]) {
		names = node["names"];
	} else if (node["class_names"]) {
		names = node["class_names"];
	} else {
		return classNames;
	}

	try {
		if (names.IsMap()) {
			// Format: {0: person, 1: bicycle, ...}
			for (const auto& kv : names) {
				int classId = kv.first.as<int>();
				std::string className = kv.second.as<std::string>();
				classNames[classId] = className;
			}
		}
		else if (names.IsSequence()) {
			// Format: [person, bicycle, ...]
			int classId = 0;
			for (const auto& name : names) {
				classNames[classId++] = name.as<std::string>();
			}
		}
	}
	catch (const YAML::Exception& e) {
		// YAML parsing error - return empty map
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
