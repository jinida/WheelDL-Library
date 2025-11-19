#pragma once

#include "BaseModel.h"
#include "../Loss/AnomalyLoss.h"
#include "../Modules/Model.h"
#include "../../Config/Configuration.h"
#include <torch/torch.h>
#include <memory>

namespace WheelDL {
    namespace Model {
        class AnomalyModel : public BaseModel {
        public:
            /**
             * @brief Constructor with configuration and model YAML path
             *
             * @param config Configuration object containing training settings
             * @param modelYamlPath Path to model YAML file (e.g., test_efficientad_model.yaml)
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

            using BaseModel::forward;
            /**
             * @brief Main forward pass for inference
             *
             * @param x Input tensor for inference
             * @return Output tensor(s) - vector for multi-scale outputs
			 */
            std::unordered_map<std::string, torch::Tensor> forward(const Data::Dataset::DataExample& data);

            /**
             * @brief Set teacher feature normalization parameters (EfficientAD only)
             *
             * Computes mean and std of teacher features on normal training data.
             * Must be called before training EfficientAD.
             *
             * @param loader DataLoader with normal training samples
             */
            template<typename DataLoader>
            void prepareTraining(DataLoader& loader)
            {
                auto preparable = std::dynamic_pointer_cast<Modules::IFeaturePreparable>(_anomalyModel);
                if (preparable) 
                {
                    switch (_lossType)
                    {
                        case Loss::AnomalyLoss::LossType::EfficientAD:
                        {
							auto efficientAD = _anomalyModel->as<Modules::EfficientAD>();
							efficientAD->setFeatureParams(loader);
                            break;
                        }
                        case Loss::AnomalyLoss::LossType::PatchCore:
                        {
                            auto patchCore = _anomalyModel->as<Modules::PatchCore>();
                            patchCore->buildMemoryBank(loader);
                            break;
                        }
                        default:
                            // No feature preparation needed for other loss types
                            break;
					}
                }
            }

            /**
             * @brief Set quantile normalization parameters (EfficientAD only)
             *
             * Computes quantiles of anomaly maps on validation data.
             * Must be called after training and before inference.
             *
             * @param loader DataLoader with normal validation samples
             */
            template<typename DataLoader>
            void prepareValidation(DataLoader& loader)
            {
                // Check if the model implements IFeaturePreparable
                auto preparable = std::dynamic_pointer_cast<Modules::IFeaturePreparable>(_anomalyModel);
                if (preparable)
                {
                    switch (_lossType)
                    {
                    case Loss::AnomalyLoss::LossType::EfficientAD:
                    {
                        auto efficientAD = _anomalyModel->as<Modules::EfficientAD>();
                        efficientAD->setQuantiles(loader);
                        break;
                    }
                    case Loss::AnomalyLoss::LossType::PatchCore:
                    {
                        auto patchCore = _anomalyModel->as<Modules::PatchCore>();
						patchCore->subsampleMemoryBank();
                        break;
					}
                    default:
                        // No feature preparation needed for other loss types
                        break;
                    }
                }
            }
        protected:
            /**
             * @brief Initialize loss criterion
             *
             * @return Loss function instance
             */
            std::unique_ptr<Loss::BaseLoss> initCriterion() override;

        private:
            /**
             * @brief Determine loss type from model type string
             * @param modelType Model type string (e.g., "EfficientAD")
             * @return Corresponding loss type
             */
            void determineLossType();

            std::shared_ptr<Modules::IAnomalyModel> _anomalyModel;  ///< Anomaly detection model
            Loss::AnomalyLoss::LossType _lossType;                   ///< Loss type for this model
        };
    } // namespace Model
} // namespace WheelDL