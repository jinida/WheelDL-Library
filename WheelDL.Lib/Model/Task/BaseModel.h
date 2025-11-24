#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>
#include "../../Config/Configuration.h"
#include "../../Data/Dataset/BaseDataset.h"
#include "../../Utils/Common/Types.h"

namespace WheelDL {
	namespace Model {

		// Forward declarations
		namespace Loss {
			class BaseLoss;
		}

		/**
		 * @brief Base class for all WheelDL models
		 *
		 * This class provides common functionality for all models including:
		 * - Forward pass handling for training and inference
		 * - Weight loading and saving
		 * - Loss computation
		 *
		 * Derived classes should implement:
		 * - initCriterion(): Initialize loss function
		 * - predictOnce(): Model-specific forward pass
		 */
		class BaseModel : public torch::nn::Module {
		public:
			/**
			 * @brief Default constructor
			 */
			BaseModel();

			/**
			 * @brief Virtual destructor
			 */
			virtual ~BaseModel() = default;

			/**
			 * @brief Main forward pass for training or inference
			 *
			 * @param x Input tensor for inference
			 * @return Output tensor(s) - vector for multi-scale outputs
			 */
			std::vector<torch::Tensor> forward(const torch::Tensor& x);

			/**
			 * @brief Forward pass with DataExample (training mode)
			 *
			 * @param data DataExample containing inputs and targets
			 * @return Loss dictionary with component losses
			 */
			std::unordered_map<std::string, torch::Tensor> forward(const Data::Dataset::DataExample& data);

			/**
			 * @brief Perform prediction (inference mode)
			 *
			 * @param x Input tensor
			 * @return Prediction tensors - vector for multi-scale outputs
			 */
			std::vector<torch::Tensor> predict(const torch::Tensor& x);

			/**
			 * @brief Compute loss for training
			 *
			 * @param batch DataExample containing inputs and targets
			 * @param preds Optional precomputed predictions (vector for multi-scale)
			 * @return Map of loss components
			 */
			virtual std::unordered_map<std::string, torch::Tensor> loss(
				const Data::Dataset::DataExample& batch,
				const std::vector<torch::Tensor>& preds = {});

			/**
			 * @brief Load weights from checkpoint
			 *
			 * @param weights Path to weights file or state_dict
			 */
			void loadWeights(const std::string& weights);
			void loadWeights(const std::unordered_map<std::string, torch::Tensor>& stateDict);

			/**
			 * @brief Save model weights
			 *
			 * @param path Output file path
			 */
			void saveWeights(const std::string& path) const;

			// Getters
			torch::nn::Sequential getModel() const { return _model; }
			std::vector<int64_t> getSaveIndices() const { return _saveIndices; }
			torch::Tensor getStride() const { return _stride; }
			WheelDL::TaskType getTaskType() const { return _taskType; }

			/**
			 * @brief Set model from builder (initial setup only)
			 *
			 * Registers all submodules with PyTorch's module system.
			 * Should only be called once during initialization.
			 *
			 * @param model Sequential model from ModelBuilder
			 */
			void setModel(torch::nn::Sequential model)
			{
				_model = model;
				for (size_t i = 0; i < _model->size(); ++i) {
					register_module(std::to_string(i), _model->ptr(i));
				}
			}

			/**
			 * @brief Swap model sequential (for temporary replacement)
			 *
			 * Temporarily replaces the internal Sequential without re-registering modules.
			 * Used for EMA model swapping during validation/checkpoint.
			 *
			 * @param newSeq New Sequential to swap in
			 * @return torch::nn::Sequential Old Sequential (for restoration)
			 */
			torch::nn::Sequential swapModelSequential(torch::nn::Sequential newSeq)
			{
				auto oldSeq = _model;
				_model = newSeq;
				return oldSeq;
			}

			/**
			 * @brief Set from indices for each layer (which layers each layer takes input from)
			 *
			 * @param indices Vector of vectors containing source layer indices for each layer
			 */
			void setFromIndices(const std::vector<std::vector<int64_t>>& indices)
			{
				_fromIndices = indices;
			}

			/**
			 * @brief Set save indices (which layer outputs need to be saved)
			 *
			 * @param indices Vector of layer indices whose outputs should be saved
			 */
			void setSaveIndices(const std::vector<int64_t>& indices) {
				_saveIndices = indices;
			}

			/**
			 * @brief Set model stride
			 *
			 * @param stride Stride tensor
			 */
			void setStride(const torch::Tensor& stride) {
				_stride = stride;
			}

			// Setters
			void setTaskType(WheelDL::TaskType type) { _taskType = type; }
			void setConfig(std::shared_ptr<Config::Configuration> config) { _config = config; }

			/**
			 * @brief Override to() to also move loss function to target device
			 *
			 * @param device Target device (CPU/CUDA)
			 * @param dtype Optional target dtype
			 */
			void to(torch::Device device, torch::Dtype dtype);

			// Overload for just device (most common case)
			void to(torch::Device device);

		protected:
			/**
			 * @brief Initialize loss criterion (must be implemented by derived classes)
			 *
			 * @return Loss function instance
			 */
			virtual std::unique_ptr<Loss::BaseLoss> initCriterion() = 0;

			/**
			 * @brief Apply function to model tensors
			 *
			 * @param fn Function to apply
			 */
			void applyToTensors(const std::function<torch::Tensor(const torch::Tensor&)>& fn);

			/**
			 * @brief Count total parameters in model
			 *
			 * @return Total parameter count
			 */
			int64_t countParameters() const;

			/**
			 * @brief Count FLOPs for model (approximate)
			 *
			 * @param inputSize Input tensor size [batch, channels, height, width]
			 * @return Estimated FLOPs
			 */
			double countFlops(const torch::IntArrayRef& inputSize) const;

		protected:
			// Model components
			torch::nn::Sequential _model = nullptr;              // Main model sequential
			std::vector<int64_t> _saveIndices;        // Layer indices to save outputs
			std::vector<std::vector<int64_t>> _fromIndices;         // From indices for each layer (supports multi-input like [-1, 6])
			torch::Tensor _stride;                     // Model stride values

			WheelDL::TaskType _taskType;         // Task type (detection, classification, etc.)

			// Loss function
			std::unique_ptr<Loss::BaseLoss> _criterion;

			// Model state
			bool _isInitialized;

			// Configuration
			std::shared_ptr<Config::Configuration> _config;

			// Training state
			bool _inplace;     // Use inplace operations
			bool _training;    // Training mode flag
		};


	} // namespace Model
} // namespace WheelDL