#pragma once

#include "BaseDataset.h"
#include <fstream>
#include <sstream>

namespace WheelDL
{
    namespace Data
    {
        namespace Dataset
        {
            /**
             * @class ClassificationDataset
             * @brief Dataset for image classification tasks
             *
             * Annotation format: One class label per image
             * File format: image_path class_id (one per line)
             */
            class ClassificationDataset : public BaseDataset<ClassificationDataset>
            {
            public:
                /**
                 * @brief Construct classification dataset
                 * @param dataPath Path to image directory
                 * @param annotationPath Path to annotation file (txt format)
                 * @param config Configuration object
                 * @param train Whether this is training dataset (affects transforms)
                 */
                ClassificationDataset(const std::string& dataPath,
                                     const std::string& annotationPath,
                                     const Config::Configuration& config,
                                     bool train = true);

            protected:
                /**
                 * @brief Load classification annotations from file
                 *
                 * Expected format (one line per image):
                 * image_path class_id
                 * Example:
                 * images/cat.jpg 0
                 * images/dog.jpg 1
                 */
                void loadAnnotations() override;

                /**
                 * @brief Build transform pipeline for classification
                 *
                 * Training transforms:
                 * - Resize/LetterBox to target size
                 * - RandomHorizontalFlip
                 * - ColorJitter
                 * - Normalize
                 *
                 * Validation transforms:
                 * - Resize/LetterBox to target size
                 * - Normalize
                 */
                std::shared_ptr<Transforms::Transform> buildTransforms() override;

                /**
                 * @brief Convert annotation to target tensor
                 * @param index Sample index
                 * @param annotations Annotation (contains class ID)
                 * @return torch::Tensor Class label tensor [1]
                 */
                torch::Tensor getTargetTensor(size_t index, const Annotation& annotations) override;

            private:
                bool _train;  // Training mode flag
            };

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
