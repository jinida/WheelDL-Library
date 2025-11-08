#pragma once

#include <torch/torch.h>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include "../../Data/Dataset/BaseDataset.h"

namespace WheelDL {
namespace Model {
namespace Loss {

/**
 * @brief Abstract base class for all loss functions
 *
 * BaseLoss provides a common interface for computing loss values.
 * Task-specific loss functions inherit from this class.
 *
 * Loss computation returns a map of loss components:
 * - Simple losses: {"total": loss_value}
 * - Multi-component losses (Detection/OBB): {"box": box_loss, "cls": cls_loss, "dfl": dfl_loss, "total": total_loss}
 */
class BaseLoss {
public:
    virtual ~BaseLoss() = default;

    /**
     * @brief Compute loss value(s) - primary interface for single tensor
     *
     * Most losses only need a single tensor, so this is the primary interface.
     *
     * @param prediction Model predictions
     * @param target Ground truth targets as DataExample
     * @return Map of loss components (always includes "total" key)
     */
    [[nodiscard]] virtual std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const Data::Dataset::DataExample& target) = 0;

    /**
     * @brief Compute loss with multi-scale predictions
     *
     * Default implementation for single-tensor losses: extracts first tensor or concatenates.
     * Multi-scale losses (Detection/OBB) should override this.
     *
     * @param predictions Vector of model predictions
     * @param target Ground truth targets as DataExample
     * @return Map of loss components (always includes "total" key)
     */
    [[nodiscard]] virtual std::unordered_map<std::string, torch::Tensor> compute(
        const std::vector<torch::Tensor>& predictions,
        const Data::Dataset::DataExample& target) {
        if (predictions.empty()) {
            throw std::invalid_argument("Predictions vector cannot be empty");
        }

        // Default behavior for single-tensor losses:
        // If only one tensor, use it directly
        if (predictions.size() == 1)
        {
            return compute(predictions[0], target);
        }

        // For multi-tensor input on single-tensor loss:
        // This shouldn't normally happen, but provide reasonable default
        // Detection/OBB will override this method anyway
        throw std::logic_error(name() + " received multiple predictions but expects single tensor");
    }

    /**
     * @brief Convenience method for single tensor with Tensor target
     *
     * Wraps inputs and calls the main compute method.
     *
     * @param prediction Single model prediction tensor
     * @param target Ground truth targets as Tensor
     * @return Map of loss components (always includes "total" key)
     */
    [[nodiscard]] virtual std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const torch::Tensor& target);

    /**
     * @brief Get total loss from loss map
     *
     * @param lossMap Map of loss components
     * @return Total loss tensor
     */
    [[nodiscard]] static torch::Tensor getTotal(const std::unordered_map<std::string, torch::Tensor>& lossMap) {
        auto it = lossMap.find("total");
        if (it == lossMap.end()) {
            throw std::runtime_error("Loss map must contain 'total' key");
        }
        return it->second;
    }

    /**
     * @brief Get loss name for logging
     *
     * @return Name of the loss function
     */
    [[nodiscard]] virtual std::string name() const = 0;

protected:
    /**
     * @brief Validate loss tensor for NaN/Inf values
     *
     * @param loss Loss tensor to validate
     * @param lossName Name of the loss for error messages
     * @throws std::runtime_error if loss contains NaN or Inf
     */
    static void validateLoss(const torch::Tensor& loss, const std::string& lossName) {
        if (torch::isnan(loss).any().item<bool>()) {
            throw std::runtime_error(lossName + ": Loss computation resulted in NaN");
        }
        if (torch::isinf(loss).any().item<bool>()) {
            throw std::runtime_error(lossName + ": Loss computation resulted in Inf");
        }
    }

    /**
     * @brief Validate tensor dimensions
     *
     * @param tensor Tensor to validate
     * @param expectedDim Expected number of dimensions
     * @param tensorName Name of the tensor for error messages
     * @throws std::invalid_argument if dimension doesn't match
     */
    static void validateDimension(
        const torch::Tensor& tensor,
        int64_t expectedDim,
        const std::string& tensorName
    ) {
        if (tensor.dim() != expectedDim) {
            throw std::invalid_argument(
                tensorName + " must be " + std::to_string(expectedDim) +
                "D tensor, got " + std::to_string(tensor.dim()) + "D"
            );
        }
    }

    /**
     * @brief Validate batch size match between prediction and target
     *
     * @param pred Prediction tensor
     * @param target Target tensor
     * @param context Optional context string for error messages
     * @throws std::invalid_argument if batch sizes don't match
     */
    static void validateBatchSize(
        const torch::Tensor& pred,
        const torch::Tensor& target,
        const std::string& context = ""
    ) {
        if (pred.size(0) != target.size(0)) {
            throw std::invalid_argument(
                context + "Prediction and target batch sizes must match, got " +
                std::to_string(pred.size(0)) + " vs " +
                std::to_string(target.size(0))
            );
        }
    }

    /**
     * @brief Validate shape match between two tensors
     *
     * @param tensor1 First tensor
     * @param tensor2 Second tensor
     * @param name1 Name of first tensor for error messages
     * @param name2 Name of second tensor for error messages
     * @throws std::invalid_argument if shapes don't match
     */
    static void validateShapeMatch(
        const torch::Tensor& tensor1,
        const torch::Tensor& tensor2,
        const std::string& name1,
        const std::string& name2
    ) {
        if (tensor1.sizes() != tensor2.sizes()) {
            std::ostringstream oss;
            oss << name1 << " and " << name2 << " must have same shape, got [";
            for (int64_t i = 0; i < tensor1.dim(); ++i) {
                if (i > 0) oss << ", ";
                oss << tensor1.size(i);
            }
            oss << "] vs [";
            for (int64_t i = 0; i < tensor2.dim(); ++i) {
                if (i > 0) oss << ", ";
                oss << tensor2.size(i);
            }
            oss << "]";
            throw std::invalid_argument(oss.str());
        }
    }

    /**
     * @brief Format tensor shape for error messages
     *
     * @param tensor Tensor to format
     * @return String representation of tensor shape
     */
    static std::string formatTensorShape(const torch::Tensor& tensor) {
        std::ostringstream oss;
        oss << "[";
        for (int64_t i = 0; i < tensor.dim(); ++i) {
            if (i > 0) oss << ", ";
            oss << tensor.size(i);
        }
        oss << "]";
        return oss.str();
    }
};

// ============================================================================
// Common Loss Functions
// ============================================================================

/**
 * @brief Binary Cross-Entropy with Logits Loss
 *
 * Combines sigmoid activation and BCE loss for numerical stability.
 * Commonly used for binary classification and multi-label classification.
 */
class BCEWithLogitsLoss : public BaseLoss {
public:
    explicit BCEWithLogitsLoss(const std::string& reduction = "mean");
    ~BCEWithLogitsLoss() override = default;

    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const Data::Dataset::DataExample& target) override;

    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const torch::Tensor& target) override;

    [[nodiscard]] std::string name() const override { return "BCEWithLogitsLoss"; }

private:
    torch::nn::BCEWithLogitsLoss _criterion;
};

/**
 * @brief Mean Squared Error Loss
 *
 * L2 loss commonly used for regression tasks.
 */
class MSELoss : public BaseLoss {
public:
    explicit MSELoss(const std::string& reduction = "mean");
    ~MSELoss() override = default;

    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const Data::Dataset::DataExample& target) override;

    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const torch::Tensor& target) override;

    [[nodiscard]] std::string name() const override { return "MSELoss"; }

private:
    torch::nn::MSELoss _criterion;
};

/**
 * @brief Mean Absolute Error Loss
 *
 * L1 loss commonly used for robust regression tasks.
 */
class MAELoss : public BaseLoss {
public:
    explicit MAELoss(const std::string& reduction = "mean");
    ~MAELoss() override = default;

    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const Data::Dataset::DataExample& target) override;

    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const torch::Tensor& target) override;

    [[nodiscard]] std::string name() const override { return "MAELoss"; }

private:
    torch::nn::L1Loss _criterion;
};

/**
 * @brief Smooth L1 Loss (Huber Loss)
 *
 * Combination of L1 and L2 loss for robust regression.
 * Less sensitive to outliers than MSE.
 */
class SmoothL1Loss : public BaseLoss {
public:
    explicit SmoothL1Loss(float beta = 1.0f, const std::string& reduction = "mean");
    ~SmoothL1Loss() override = default;

    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const Data::Dataset::DataExample& target) override;

    [[nodiscard]] std::unordered_map<std::string, torch::Tensor> compute(
        const torch::Tensor& prediction,
        const torch::Tensor& target) override;

    [[nodiscard]] std::string name() const override { return "SmoothL1Loss"; }

private:
    torch::nn::SmoothL1Loss _criterion;
};

// ============================================================================
// DFLoss - Distribution Focal Loss
// ============================================================================

/**
 * @brief Distribution Focal Loss (DFL)
 *
 * DFL is used in anchor-free object detection to predict bounding box coordinates
 * as a probability distribution over discrete bins instead of direct regression.
 *
 * Reference: "Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection"
 */
class DFLoss {
public:
    /**
     * @brief Construct a DFLoss
     *
     * @param regMax Maximum regression range (number of bins)
     */
    explicit DFLoss(int64_t regMax) : _regMax(regMax) {
        if (_regMax <= 0) {
            throw std::invalid_argument("regMax must be positive");
        }
    }

    /**
     * @brief Compute DFL loss
     *
     * @param predDist Predicted distribution [N, regMax]
     * @param target Target values [N]
     * @return DFL loss tensor
     */
    [[nodiscard]] torch::Tensor compute(const torch::Tensor& predDist, const torch::Tensor& target) {
        // Constants for DFL computation
        constexpr float CLAMP_EPSILON = 0.01f;

        // Clamp target to valid range [0, regMax - 1 - epsilon]
        auto targetClamped = target.clamp(0.0f, static_cast<float>(_regMax - 1) - CLAMP_EPSILON);

        // Get integer parts (left and right neighbors for interpolation)
        auto tl = targetClamped.floor().to(torch::kLong);
        auto tr = tl + 1;

        // Get interpolation weights
        auto wl = tr.to(torch::kFloat) - targetClamped;
        auto wr = 1.0f - wl;

        // Compute cross-entropy for both neighbors
        auto predDistReshaped = predDist.view({ -1, _regMax });
        auto tlReshaped = tl.view({ -1 });
        auto trReshaped = tr.view({ -1 });

        auto lossLeft = torch::nn::functional::cross_entropy(
            predDistReshaped, tlReshaped,
            torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone)
        ).view(tl.sizes());

        auto lossRight = torch::nn::functional::cross_entropy(
            predDistReshaped, trReshaped,
            torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone)
        ).view(tr.sizes());

        // Weighted combination for smooth interpolation
        auto loss = lossLeft * wl + lossRight * wr;
        return loss.mean(-1, /*keepdim=*/true);
    }

    /**
     * @brief Get regression max value
     *
     * @return regMax value
     */
    [[nodiscard]] int64_t getRegMax() const { return _regMax; }

private:
    int64_t _regMax;  ///< Maximum regression range
};

} // namespace Loss
} // namespace Model
} // namespace WheelDL
