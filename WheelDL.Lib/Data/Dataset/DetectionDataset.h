#pragma once

#include "BaseDataset.h"

namespace WheelDL
{
    namespace Data
    {
        namespace Dataset
        {
            /**
             * @class DetectionDataset
             * @brief Dataset for object detection tasks
             *
             * Annotation format: YOLO format (class_id, x_center, y_center, width, height)
             * All coordinates are normalized to [0, 1]
             */
            class DetectionDataset : public BaseDataset<DetectionDataset>
            {
            public:
                /**
                 * @brief Construct detection dataset
                 * @param dataPath Path to image directory
                 * @param annotationPath Path to annotation directory (YOLO format)
                 * @param config Configuration object
                 * @param train Whether this is training dataset (affects transforms)
                 */
                DetectionDataset(const std::string& dataPath,
                                const std::string& annotationPath,
                                const Config::Configuration& config,
                                bool train = true);

            protected:
                /**
                 * @brief Load detection annotations from files
                 *
                 * Expected format (YOLO format, one line per object):
                 * class_id x_center y_center width height
                 *
                 * Example (labels/image001.txt):
                 * 0 0.5 0.5 0.3 0.4
                 * 1 0.2 0.3 0.1 0.15
                 */
                void loadAnnotations() override;

                /**
                 * @brief Build transform pipeline for detection
                 *
                 * Training transforms:
                 * - LetterBox to target size
                 * - RandomPerspective (rotation, scale, translation)
                 * - RandomHorizontalFlip
                 * - ColorJitter
                 *
                 * Validation transforms:
                 * - LetterBox to target size
                 */
                std::shared_ptr<Transforms::Transform> buildTransforms() override;

                /**
                 * @brief Convert annotation to target tensor
                 * @param index Sample index
                 * @param annotations Annotation (contains bounding boxes)
                 * @return torch::Tensor Target tensor [num_objects, 5] (class, x, y, w, h)
                 */
                torch::Tensor getTargetTensor(size_t index, const Annotation& annotations) override;

            private:
                bool _train;  // Training mode flag
            };

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
