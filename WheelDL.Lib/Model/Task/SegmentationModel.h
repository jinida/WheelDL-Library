#pragma once

#include "BaseModel.h"
#include "../Loss/SegmentationLoss.h"
#include "../../Config/Configuration.h"
#include <torch/torch.h>
#include <memory>

namespace WheelDL {
namespace Model {

/**
 * @brief Segmentation Model for semantic segmentation tasks
 *
 * This model handles semantic segmentation tasks, predicting a class
 * label for each pixel in the input image.
 *
 * Key features:
 * - Pixel-wise classification
 * - Multi-class segmentation support
 * - Various loss functions (CrossEntropy, Focal, Dice)
 * - Configuration-based initialization
 *
 * Example usage:
 * @code
 * auto config = std::make_shared<Configuration>();
 * config->loadFromYaml("yolov8n-seg.yaml", "default.yaml");
 *
 * SegmentationModel model(config, "yolov8n-seg.yaml");
 * auto predictions = model.forward(input);
 * @endcode
 */
class SegmentationModel : public BaseModel {
public:
    /**
     * @brief Constructor with configuration and model path
     *
     * @param config Configuration object containing all settings
     * @param modelYamlPath Path to model YAML file
     */
    explicit SegmentationModel(std::shared_ptr<Config::Configuration> config,
                              const std::string& modelYamlPath);

    /**
     * @brief Destructor
     */
    ~SegmentationModel() override = default;

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

private:
    // Loss parameters
    float _diceWeight;
};

} // namespace Model
} // namespace WheelDL