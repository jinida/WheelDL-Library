#pragma once

#include "BaseModel.h"
#include "../Loss/AnomalyLoss.h"
#include "../../Config/Configuration.h"
#include <torch/torch.h>
#include <memory>

namespace WheelDL {
    namespace Model {

        /**
         * @brief Anomaly Detection Model
         *
         * This model handles anomaly detection tasks using reconstruction-based
         * or feature-based approaches. It identifies outliers or anomalous patterns
         * in the input data.
         *
         * Key features:
         * - Reconstruction-based anomaly detection
         * - Feature extraction and comparison
         * - Multiple loss functions (MSE, MAE, Perceptual)
         * - Configuration-based initialization
         *
         * Example usage:
         * @code
         * auto config = std::make_shared<Configuration>();
         * config->loadFromYaml("anomaly-model.yaml", "default.yaml");
         *
         * AnomalyModel model(config, "anomaly-model.yaml");
         * auto scores = model.forward(input);
         * @endcode
         */
        class AnomalyModel : public BaseModel {
        public:
            /**
             * @brief Constructor with configuration and model path
             *
             * @param config Configuration object containing all settings
             * @param modelYamlPath Path to model YAML file
             */
            explicit AnomalyModel(std::shared_ptr<Config::Configuration> config,
                const std::string& modelYamlPath);

            /**
             * @brief Destructor
             */
            ~AnomalyModel() override = default;

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
            Loss::AnomalyLoss::LossType _lossType;
            float _perceptualWeight;
            float _ssimWeight;
        };

    } // namespace Model
} // namespace WheelDL