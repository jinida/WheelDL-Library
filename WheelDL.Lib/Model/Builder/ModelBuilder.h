#pragma once

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace WheelDL {
	namespace Model {
		namespace Builder {

			/**
			 * @brief ModelBuilder class for building YOLO models from YAML configuration
			 *
			 * This class implements the Builder pattern to construct neural network models
			 * from YAML configuration files. It supports various modules and heads for
			 * different tasks (Detection, OBB, Classification).
			 */
			class ModelBuilder {
			public:
				/**
				 * @brief Default constructor
				 */
				ModelBuilder();

				/**
				 * @brief Set YAML configuration path
				 *
				 * @param path Path to the model YAML file
				 * @return Reference to this builder for method chaining
				 */
				ModelBuilder& setYamlPath(const std::string& path);

				/**
				 * @brief Set number of input channels
				 *
				 * @param channels Number of input channels (default: 3 for RGB)
				 * @return Reference to this builder for method chaining
				 */
				ModelBuilder& setInputChannels(int64_t channels);

				/**
				 * @brief Set number of classes
				 *
				 * @param numClasses Number of output classes
				 * @return Reference to this builder for method chaining
				 */
				ModelBuilder& setNumClasses(int64_t numClasses);

				/**
				 * @brief Set image size
				 *
				 * @param size Input image size (width and height)
				 * @return Reference to this builder for method chaining
				 */
				ModelBuilder& setImageSize(int64_t size);

				/**
				 * @brief Build the model as a torch::nn::Sequential
				 *
				 * Parses the YAML configuration and constructs the neural network.
				 *
				 * @return torch::nn::Sequential containing the complete model
				 */
				torch::nn::Sequential build();

				/**
				 * @brief Get output channel sizes for each detection layer
				 *
				 * @return Vector of channel sizes for head layers
				 */
				std::vector<int64_t> getHeadChannels() const;

				/**
				 * @brief Get save indices for skip connections
				 *
				 * @return Vector of layer indices to save for skip connections
				 */
				std::vector<int64_t> getSaveIndices() const;

				/**
				 * @brief Get from indices for each layer
				 *
				 * @return Vector of vectors containing source layer indices for each layer
				 */
				std::vector<std::vector<int64_t>> getFromIndices() const;

			private:
				/**
				 * @brief Validate model configuration after building
				 *
				 * This method checks:
				 * - All referenced layers in _fromIndices exist and are valid
				 * - Negative indices don't cause underflow
				 * - Required layers are included in _saveIndices
				 * - No forward references (layers referencing future layers)
				 *
				 * @throws std::runtime_error if validation fails
				 */
				void validateConfiguration(const torch::nn::Sequential& model);

				/**
				 * @brief Parse YAML configuration file
				 */
				void parseYaml();

				/**
				 * @brief Apply width and depth scaling to channels and repeats
				 *
				 * @param channels Base number of channels
				 * @param repeats Base number of repeats
				 * @return std::pair of (scaled_channels, scaled_repeats)
				 */
				std::pair<int64_t, int64_t> applyScale(int64_t channels, int64_t repeats);

				/**
				 * @brief Build a module from YAML specification
				 *
				 * @param moduleType Module type (Conv, C2f, SPPF, etc.)
				 * @param argsNode YAML node containing module arguments
				 * @param inputChannels Number of input channels
				 * @param repeats Number of times to repeat (for depth scaling)
				 * @param outputChannels [out] Will be set to the output channel count
				 * @return torch::nn::AnyModule containing the built module
				 */
				torch::nn::AnyModule buildModule(
					const std::string& moduleType,
					const YAML::Node& argsNode,
					int64_t inputChannels,
					int64_t repeats,
					int64_t& outputChannels
				);


				/**
				 * @brief Get output channels for a given layer ID
				 *
				 * @param layerId Layer ID
				 * @return Number of output channels
				 */
				int64_t getOutputChannels(int64_t layerId) const;

				/**
				 * @brief Resolve layer ID (handles negative indices)
				 *
				 * @param layerId Layer ID (-1 means previous layer, -2 means two layers back, etc.)
				 * @return Absolute layer ID
				 */
				int64_t resolveLayerId(int64_t layerId) const;

				// Member variables
				std::string _yamlPath;
				int64_t _inputChannels;
				int64_t _numClasses;
				int64_t _imageSize;

				YAML::Node _yamlConfig;
				double _widthMultiple;
				double _depthMultiple;
				int64_t _maxChannels;
				std::string _defaultAct;

				// Layer tracking
				int64_t _currentLayerId;
				std::unordered_map<int64_t, int64_t> _layerOutputChannels;  // layerId -> output channels
				std::unordered_map<int64_t, torch::nn::AnyModule> _layers;  // layerId -> module

				// Head channels (for detection heads)
				std::vector<int64_t> _headChannels;

				// Save indices for skip connections
				std::vector<int64_t> _saveIndices;

				// From indices for each layer (which layers each layer takes input from)
				std::vector<std::vector<int64_t>> _fromIndices;
			};

		} // namespace Builder
	} // namespace Model
} // namespace WheelDL
