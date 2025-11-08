#pragma once

#include "BaseModel.h"
#include "../Loss/ClassificationLoss.h"
#include "../../Config/Configuration.h"
#include <torch/torch.h>
#include <memory>

namespace WheelDL {
namespace Model {

/**
 * @brief Classification Model for image classification tasks
 *
 * This model handles classification tasks using various network architectures.
 * It supports multiple loss functions (CrossEntropy, Focal, Label Smoothing)
 * and provides top-k accuracy metrics.
 *
 * Key features:
 * - Multi-class and multi-label classification
 * - Support for various backbone architectures
 * - Configurable loss functions
 * - Top-k accuracy computation
 * - Configuration-based initialization
 *
 * Example usage:
 * @code
 * auto config = std::make_shared<Configuration>();
 * config->loadFromYaml("yolov8n-cls.yaml", "default.yaml");
 *
 * ClassificationModel model(config, "yolov8n-cls.yaml");
 * auto predictions = model.forward(input);
 * @endcode
 */
class ClassificationModel : public BaseModel {
public:
    /**
     * @brief Constructor with configuration and model path
     *
     * @param config Configuration object containing all settings
     * @param modelYamlPath Path to model YAML file
     */
    explicit ClassificationModel(std::shared_ptr<Config::Configuration> config,
                                const std::string& modelYamlPath);

    /**
     * @brief Destructor
     */
    ~ClassificationModel() override = default;

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
    Loss::ClassificationLoss::LossType _lossType;
};

} // namespace Model
} // namespace WheelDL