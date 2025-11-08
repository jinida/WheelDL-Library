#include "ModelBuilder.h"
#include "../Modules/Conv.h"
#include "../Modules/Block.h"
#include "../Modules/Head.h"
#include "../../Config/YamlParser.h"
#include "Factory/ModuleFactoryRegistry.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace WheelDL {
    namespace Model {
        namespace Builder {

            ModelBuilder::ModelBuilder()
                : _inputChannels(3)
                , _numClasses(80)
                , _imageSize(640)
                , _widthMultiple(1.0)
                , _depthMultiple(1.0)
                , _maxChannels(1024)
                , _defaultAct("ReLU")
                , _currentLayerId(0) {
            }

            ModelBuilder& ModelBuilder::setYamlPath(const std::string& path) {
                _yamlPath = path;
                return *this;
            }

            ModelBuilder& ModelBuilder::setInputChannels(int64_t channels) {
                _inputChannels = channels;
                return *this;
            }

            ModelBuilder& ModelBuilder::setNumClasses(int64_t numClasses) {
                _numClasses = numClasses;
                return *this;
            }

            ModelBuilder& ModelBuilder::setImageSize(int64_t size) {
                _imageSize = size;
                return *this;
            }

            void ModelBuilder::parseYaml() {
                using namespace Config;

                if (_yamlPath.empty()) {
                    throw std::runtime_error("YAML path not set");
                }

                // Use YamlParser to load and validate
                _yamlConfig = YamlParser::parseFrom(_yamlPath);

                // Parse default activation (default to ReLU if not specified)
                if (_yamlConfig["default_act"]) {
                    _defaultAct = _yamlConfig["default_act"].as<std::string>();
                }
                else {
                    _defaultAct = "ReLU";
                }

                // Parse scale parameters using YamlParser helpers
                if (_yamlConfig["scale"]) {
                    auto scale = _yamlConfig["scale"];
                    _widthMultiple = YamlParser::getFloat(scale, "width_multiple", 1.0f);
                    _depthMultiple = YamlParser::getFloat(scale, "depth_multiple", 1.0f);
                    _maxChannels = YamlParser::getInt(scale, "max_channels", 1024);
                }

                // Parse num_classes if specified
                _numClasses = YamlParser::getInt(_yamlConfig, "num_classes", _numClasses);
            }

            std::pair<int64_t, int64_t> ModelBuilder::applyScale(int64_t channels, int64_t repeats)
            {
                // Validate inputs
                if (channels < 0) {
                    throw std::invalid_argument("applyScale: channels must be non-negative, got: " +
                                               std::to_string(channels));
                }
                if (repeats < 0) {
                    throw std::invalid_argument("applyScale: repeats must be non-negative, got: " +
                                               std::to_string(repeats));
                }

                // Check for potential overflow before scaling
                constexpr int64_t max_safe_channels = std::numeric_limits<int64_t>::max() / 8;
                if (_widthMultiple > 0 && channels > max_safe_channels / _widthMultiple) {
                    throw std::overflow_error("applyScale: channel scaling would overflow");
                }

                // Apply width scaling
                double scaledDouble = std::ceil(channels * _widthMultiple / 8.0) * 8.0;

                // Validate result fits in int64_t
                if (scaledDouble > static_cast<double>(std::numeric_limits<int64_t>::max())) {
                    throw std::overflow_error("applyScale: scaled channels exceed int64_t max");
                }

                int64_t scaledChannels = static_cast<int64_t>(scaledDouble);
                scaledChannels = std::min(scaledChannels, _maxChannels);

                // Apply depth scaling with overflow check
                double scaledRepeatsDouble = std::round(repeats * _depthMultiple);
                if (scaledRepeatsDouble > static_cast<double>(std::numeric_limits<int64_t>::max())) {
                    throw std::overflow_error("applyScale: scaled repeats exceed int64_t max");
                }

                int64_t scaledRepeats = std::max(static_cast<int64_t>(scaledRepeatsDouble), int64_t(1));

                return { scaledChannels, scaledRepeats };
            }

            int64_t ModelBuilder::resolveLayerId(int64_t layerId) const
            {
                if (layerId >= 0) {
                    return layerId;
                }
                // -1 means previous layer, -2 means two layers back, etc.
                return _currentLayerId + layerId;
            }

            int64_t ModelBuilder::getOutputChannels(int64_t layerId) const
            {
                auto resolvedId = resolveLayerId(layerId);
                auto it = _layerOutputChannels.find(resolvedId);
                if (it != _layerOutputChannels.end()) {
                    return it->second;
                }

                // If layer 0 (input), return input channels
                if (resolvedId == 0) {
                    return _inputChannels;
                }

                throw std::runtime_error("Cannot find output channels for layer " + std::to_string(layerId));
            }

            torch::nn::AnyModule ModelBuilder::buildModule(
                const std::string& moduleType,
                const YAML::Node& argsNode,
                int64_t inputChannels,
                int64_t repeats,
                int64_t& outputChannels
            ) {
                // Get factory from registry
                auto factory = ModuleFactoryRegistry::instance().getFactory(moduleType);

                if (!factory) {
                    throw std::runtime_error("Unsupported module type: " + moduleType);
                }

                // Create build context
                ModuleBuildContext ctx{
                    inputChannels,
                    repeats,
                    _defaultAct,
                    _numClasses,
                    _headChannels,
                    [this](int64_t ch, int64_t rep) { return this->applyScale(ch, rep); }
                };

                // Create module using factory
                auto result = factory->create(argsNode, ctx);
                outputChannels = result.outputChannels;

                return result.module;
            }

            torch::nn::Sequential ModelBuilder::build() {
                try {
                    parseYaml();

                    // Reset all state to ensure clean build
                    _layerOutputChannels.clear();
                    _headChannels.clear();
                    _saveIndices.clear();
                    _currentLayerId = 0;
                    _layerOutputChannels[0] = _inputChannels;

                    torch::nn::Sequential model;

                // Build backbone
                if (_yamlConfig["backbone"]) {
                    auto backbone = _yamlConfig["backbone"];
                    for (const auto& layer : backbone)
                    {
                        _currentLayerId++;

                        // Parse: [from, repeats, module, args]
                        auto fromNode = layer[0];
                        int64_t repeats = layer[1].as<int64_t>();
                        std::string moduleType = layer[2].as<std::string>();
                        auto argsNode = layer[3];

                        int64_t inputChannels = 0;

                        // Determine input channels based on 'from'
                        // Following Python: save_indices.extend(x for x in ([src] if isinstance(src, int) else src) if x != -1)
                        if (fromNode.IsScalar()) {
                            int64_t fromId = fromNode.as<int64_t>();
                            inputChannels = getOutputChannels(fromId);
                            // Add to save indices if not -1
                            if (fromId != -1) {
                                int64_t resolvedId = resolveLayerId(fromId);
                                _saveIndices.push_back(resolvedId);
                            }
                        }
                        else if (fromNode.IsSequence()) {
                            // Multiple inputs (for Concat)
                            for (const auto& f : fromNode) {
                                int64_t fromId = f.as<int64_t>();
                                inputChannels += getOutputChannels(fromId);
                                // Add to save indices if not -1
                                if (fromId != -1) {
                                    int64_t resolvedId = resolveLayerId(fromId);
                                    _saveIndices.push_back(resolvedId);
                                }
                            }
                        }

                        // Build the module using buildModule helper
                        int64_t outputChannels = 0;
                        auto module = buildModule(moduleType, argsNode, inputChannels, repeats, outputChannels);
                        model->push_back(module);

                        _layerOutputChannels[_currentLayerId] = outputChannels;
                    }
                }

                // Build head
                if (_yamlConfig["head"]) {
                    auto head = _yamlConfig["head"];

                    for (const auto& layer : head) {
                        _currentLayerId++;

                        auto fromNode = layer[0];
                        int64_t repeats = layer[1].as<int64_t>();
                        std::string moduleType = layer[2].as<std::string>();
                        auto argsNode = layer[3];

                        int64_t inputChannels = 0;

                        // Track head input indices
                        if (fromNode.IsScalar()) {
                            int64_t fromId = fromNode.as<int64_t>();
                            inputChannels = getOutputChannels(fromId);

                            // Store head input indices
                            if (fromId != -1) {
                                int64_t resolvedId = resolveLayerId(fromId);
                                _headInputIndices.push_back(resolvedId);
                            }

                            // Add to save indices if not -1
                            if (fromId != -1) {
                                int64_t resolvedId = resolveLayerId(fromId);
                                _saveIndices.push_back(resolvedId);
                            }
                        }
                        else if (fromNode.IsSequence()) {
                            // For Detect head, collect channel sizes
                            if (moduleType == "Detect" || moduleType == "OBB" || moduleType == "Classify") {
                                _headChannels.clear();
                                for (const auto& f : fromNode) {
                                    int64_t fromId = f.as<int64_t>();
                                    int64_t ch = getOutputChannels(fromId);
                                    _headChannels.push_back(ch);
                                    inputChannels += ch;

                                    // Store head input indices - these are the layers head takes from
                                    if (fromId != -1) {
                                        int64_t resolvedId = resolveLayerId(fromId);
                                        _headInputIndices.push_back(resolvedId);
                                        _saveIndices.push_back(resolvedId);
                                    }
                                }
                            }
                            else {
                                // For Concat
                                for (const auto& f : fromNode) {
                                    int64_t fromId = f.as<int64_t>();
                                    inputChannels += getOutputChannels(fromId);
                                    // Add to save indices if not -1
                                    if (fromId != -1) {
                                        int64_t resolvedId = resolveLayerId(fromId);
                                        _saveIndices.push_back(resolvedId);
                                    }
                                }
                            }
                        }

                        // Build head module using buildModule helper
                        int64_t outputChannels = 0;
                        auto module = buildModule(moduleType, argsNode, inputChannels, repeats, outputChannels);
                        model->push_back(module);

                        _layerOutputChannels[_currentLayerId] = outputChannels;
                    }
                }

                    // Sort and remove duplicates from save indices
                    // Following Python: sorted(set(save_indices))
                    std::sort(_saveIndices.begin(), _saveIndices.end());
                    _saveIndices.erase(std::unique(_saveIndices.begin(), _saveIndices.end()), _saveIndices.end());

                    return model;

                } catch (const std::exception& e) {
                    // Clean up partial model state on error
                    _layerOutputChannels.clear();
                    _headChannels.clear();
                    _currentLayerId = 0;
                    throw std::runtime_error(
                        std::string("ModelBuilder::build failed: ") + e.what()
                    );
                }
            }

            std::vector<int64_t> ModelBuilder::getHeadChannels() const {
                return _headChannels;
            }

            std::vector<int64_t> ModelBuilder::getSaveIndices() const {
                return _saveIndices;
            }

            std::vector<int64_t> ModelBuilder::getHeadInputIndices() const {
                return _headInputIndices;
            }
        } // namespace Builder
    } // namespace Model
} // namespace WheelDL
