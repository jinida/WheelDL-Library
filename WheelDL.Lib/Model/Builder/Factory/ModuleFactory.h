#pragma once

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <functional>

namespace WheelDL {
    namespace Model {
        namespace Builder {

            /**
             * @brief Context information needed for building a module
             *
             * Note: Stores values (not references) to avoid dangling reference issues
             */
            struct ModuleBuildContext {
                int64_t inputChannels;
                int64_t repeats;
                std::string defaultAct;  // Store by value to avoid dangling references
                int64_t numClasses;
                std::vector<int64_t> headChannels;  // Store by value to avoid dangling references

                std::function<std::pair<int64_t, int64_t>(int64_t, int64_t)> applyScale;
            };

            /**
             * @brief Result of building a module
             */
            struct ModuleBuildResult {
                torch::nn::AnyModule module;
                int64_t outputChannels;
            };

            /**
             * @brief Helper class for parsing YAML arguments safely
             */
            class YamlArgsParser {
            public:
                /**
                 * @brief Parse YAML node as integer vector with error handling
                 *
                 * @param argsNode YAML node to parse
                 * @param moduleName Name of module (for error messages)
                 * @return std::vector<int64_t> Parsed integer vector
                 * @throws std::runtime_error if parsing fails
                 */
                static std::vector<int64_t> parseIntVector(
                    const YAML::Node& argsNode,
                    const std::string& moduleName)
                {
                    try {
                        if (!argsNode.IsDefined()) {
                            throw std::runtime_error(moduleName + ": args node is not defined");
                        }
                        if (!argsNode.IsSequence()) {
                            throw std::runtime_error(moduleName + ": args must be a sequence");
                        }
                        return argsNode.as<std::vector<int64_t>>();
                    } catch (const YAML::Exception& e) {
                        throw std::runtime_error(
                            moduleName + ": invalid args - " + std::string(e.what())
                        );
                    }
                }

                /**
                 * @brief Get integer argument from YAML node at index with default value
                 *
                 * @param argsNode YAML sequence node
                 * @param index Index to retrieve
                 * @param defaultValue Default value if index out of bounds
                 * @return int64_t Argument value or default
                 */
                static int64_t getInt(
                    const YAML::Node& argsNode,
                    size_t index,
                    int64_t defaultValue)
                {
                    if (!argsNode.IsSequence() || index >= argsNode.size()) {
                        return defaultValue;
                    }
                    return argsNode[index].as<int64_t>();
                }

                /**
                 * @brief Get boolean argument from YAML node at index with default value
                 *
                 * @param argsNode YAML sequence node
                 * @param index Index to retrieve
                 * @param defaultValue Default value if index out of bounds
                 * @return bool Argument value or default
                 */
                static bool getBool(
                    const YAML::Node& argsNode,
                    size_t index,
                    bool defaultValue)
                {
                    if (!argsNode.IsSequence() || index >= argsNode.size()) {
                        return defaultValue;
                    }
                    auto node = argsNode[index];
                    if (node.IsScalar()) {
                        // Handle both boolean and integer representations
                        try {
                            return node.as<bool>();
                        } catch (...) {
                            return static_cast<bool>(node.as<int64_t>());
                        }
                    }
                    return defaultValue;
                }

                /**
                 * @brief Get double argument from YAML node at index with default value
                 *
                 * @param argsNode YAML sequence node
                 * @param index Index to retrieve
                 * @param defaultValue Default value if index out of bounds
                 * @return double Argument value or default
                 */
                static double getDouble(
                    const YAML::Node& argsNode,
                    size_t index,
                    double defaultValue)
                {
                    if (!argsNode.IsSequence() || index >= argsNode.size()) {
                        return defaultValue;
                    }
                    return argsNode[index].as<double>();
                }

                /**
                 * @brief Get string argument from YAML node at index with default value
                 *
                 * @param argsNode YAML sequence node
                 * @param index Index to retrieve
                 * @param defaultValue Default value if index out of bounds
                 * @return std::string Argument value or default
                 */
                static std::string getString(
                    const YAML::Node& argsNode,
                    size_t index,
                    const std::string& defaultValue)
                {
                    if (!argsNode.IsSequence() || index >= argsNode.size()) {
                        return defaultValue;
                    }
                    return argsNode[index].as<std::string>();
                }

                // Legacy methods for backward compatibility with vector<int64_t>
                static int64_t getIntArg(
                    const std::vector<int64_t>& args,
                    size_t index,
                    int64_t defaultValue)
                {
                    return (index < args.size()) ? args[index] : defaultValue;
                }

                static bool getBoolArg(
                    const std::vector<int64_t>& args,
                    size_t index,
                    bool defaultValue)
                {
                    return (index < args.size()) ? static_cast<bool>(args[index]) : defaultValue;
                }

                static double getDoubleArg(
                    const std::vector<int64_t>& args,
                    size_t index,
                    double defaultValue)
                {
                    return (index < args.size()) ? static_cast<double>(args[index]) : defaultValue;
                }
            };

            /**
             * @brief Abstract factory interface for creating neural network modules
             */
            class IModuleFactory {
            public:
                virtual ~IModuleFactory() = default;

                /**
                 * @brief Create a module from YAML arguments and build context
                 *
                 * @param argsNode YAML node containing module arguments
                 * @param ctx Build context with input channels, scaling function, etc.
                 * @return ModuleBuildResult Module and output channel count
                 */
                virtual ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) = 0;

                /**
                 * @brief Get list of module type names this factory supports
                 *
                 * @return std::vector<std::string> Supported module type names
                 */
                virtual std::vector<std::string> supportedTypes() const = 0;
            };

        } // namespace Builder
    } // namespace Model
} // namespace WheelDL
