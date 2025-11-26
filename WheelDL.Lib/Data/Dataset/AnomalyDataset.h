#pragma once

#include "BaseDataset.h"

namespace WheelDL
{
    namespace Data
    {
        namespace Dataset
        {
            /**
             * @class AnomalyDataset
             * @brief Dataset for anomaly detection tasks
             *
             * Typically contains only normal images for training.
             * No annotations required (unsupervised learning).
             * For testing, may include anomaly labels.
             */
            class AnomalyDataset : public BaseDataset<AnomalyDataset>
            {
            public:
                /**
                 * @brief Construct anomaly detection dataset
                 * @param config Configuration object containing all settings
                 * @param train Whether this is training dataset (affects transforms)
                 */
                AnomalyDataset(const Config::Configuration& config, bool train = true);

            protected:
                /**
                 * @brief Load anomaly detection data
                 *
                 * For training: Load all images (assumed normal)
                 * For testing: Load images with optional anomaly labels
                 *
                 * Annotation format (optional, for test set):
                 * image_path is_anomaly (0=normal, 1=anomaly)
                 */
                void loadAnnotations() override;

                /**
                 * @brief Build transform pipeline for anomaly detection
                 *
                 * Training transforms:
                 * - Resize to target size
                 * - RandomHorizontalFlip
                 * - RandomVerticalFlip
                 * - ColorJitter
                 * - GaussianBlur
                 * - Normalize
                 *
                 * Validation transforms:
                 * - Resize to target size
                 * - Normalize
                 */
                std::shared_ptr<Transforms::Transform> buildTransforms() override;

                /**
                 * @brief Convert annotation to target tensor
                 * @param index Sample index
                 * @param annotations Annotation (contains anomaly label if available)
                 * @return torch::Tensor Label tensor [1] (0=normal, 1=anomaly), or -1 if unknown
                 */
                torch::Tensor getTargetTensor(size_t index, const Annotation& annotations) override;
            };

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
