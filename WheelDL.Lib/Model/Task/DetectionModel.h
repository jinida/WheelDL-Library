#pragma once

#include "BaseModel.h"
#include "../Loss/DetectionLoss.h"
#include "../../Config/Configuration.h"
#include <torch/torch.h>
#include <memory>

namespace WheelDL {
namespace Model {

/**
 * @brief Detection Model for object detection tasks
 *
 * This model handles object detection tasks using YOLO-style architectures.
 * It supports anchor-free detection with multi-scale predictions and
 * uses Detection loss with task-aligned assignment.
 *
 * Key features:
 * - Multi-scale detection (P3, P4, P5)
 * - Anchor-free detection with DFL
 * - Task-aligned assignment for training
 * - Configuration-based initialization
 *
 * Example usage:
 * @code
 * auto config = std::make_shared<Configuration>();
 * config->loadFromYaml("yolov8n.yaml", "default.yaml");
 *
 * DetectionModel model(config, "yolov8n.yaml");
 * auto predictions = model.forward(input);
 * @endcode
 */
class DetectionModel : public BaseModel {
public:
    /**
     * @brief Constructor with configuration and model path
     *
     * @param config Configuration object containing all settings
     * @param modelYamlPath Path to model YAML file
     */
    explicit DetectionModel(std::shared_ptr<Config::Configuration> config,
                          const std::string& modelYamlPath);

    /**
     * @brief Destructor
     */
    ~DetectionModel() override = default;

    /**
     * @brief Load pretrained weights
     *
     * @param weightsPath Path to weights file
     * @return true if successful, false otherwise
     */
    bool loadPretrained(const std::string& weightsPath);

protected:
    /**
     * @brief Initialize loss criterion
     *
     * @return Loss function instance
     */
    std::unique_ptr<Loss::BaseLoss> initCriterion() override;

    /**
     * @brief Calculate stride values for detection layers
     *
     * Computes stride by running a forward pass with a dummy input
     * and comparing output sizes to input size
     *
     * @param imageSize Input image size
     * @return Tensor containing stride values for each detection layer
     */
    torch::Tensor calculateStride(int64_t imageSize);

private:
    // Loss parameters
    torch::Tensor _stride;  ///< Stride values for each detection layer
    float _boxGain;         ///< Box loss weight
    float _clsGain;         ///< Classification loss weight
    float _dflGain;         ///< DFL loss weight
};

} // namespace Model
} // namespace WheelDL