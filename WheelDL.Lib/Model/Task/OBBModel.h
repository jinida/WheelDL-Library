#pragma once

#include "BaseModel.h"
#include "../Loss/OBBLoss.h"
#include "../../Config/Configuration.h"
#include <torch/torch.h>
#include <memory>

namespace WheelDL {
namespace Model {

/**
 * @brief Oriented Bounding Box Model for rotated object detection
 *
 * This model handles oriented object detection tasks using YOLO-OBB architectures.
 * It predicts rotated bounding boxes with angle information, suitable for
 * detecting objects that are not axis-aligned.
 *
 * Key features:
 * - Rotated bounding box prediction (x, y, w, h, angle)
 * - Multi-scale detection with OBB heads
 * - Probiou loss for rotated boxes
 * - Configuration-based initialization
 *
 * Example usage:
 * @code
 * auto config = std::make_shared<Configuration>();
 * config->loadFromYaml("yolov8n-obb.yaml", "default.yaml");
 *
 * OBBModel model(config, "yolov8n-obb.yaml");
 * auto predictions = model.forward(input);
 * @endcode
 */
class OBBModel : public BaseModel {
public:
    /**
     * @brief Constructor with configuration and model path
     *
     * @param config Configuration object containing all settings
     * @param modelYamlPath Path to model YAML file
     */
    explicit OBBModel(std::shared_ptr<Config::Configuration> config);

    /**
     * @brief Destructor
     */
    ~OBBModel() override = default;

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
     * @brief Calculate stride values for OBB detection layers
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