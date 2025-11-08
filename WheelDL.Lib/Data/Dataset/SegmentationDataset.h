#pragma once

#include "BaseDataset.h"

namespace WheelDL
{
    namespace Data
    {
        namespace Dataset
        {
            /**
             * @class SegmentationDataset
             * @brief Dataset for semantic/instance segmentation tasks
             *
             * Annotation format: Polygon coordinates in text files (YOLO segmentation format)
             * Masks are generated dynamically from polygon coordinates
             * Output format: One-hot encoded tensor [C, H, W]
             */
            class SegmentationDataset : public BaseDataset<SegmentationDataset>
            {
            public:
                /**
                 * @brief Construct segmentation dataset
                 * @param dataPath Path to image directory
                 * @param annotationPath Path to annotation directory (text files with polygon coordinates)
                 * @param config Configuration object
                 * @param train Whether this is training dataset (affects transforms)
                 */
                SegmentationDataset(const std::string& dataPath,
                                   const std::string& annotationPath,
                                   const Config::Configuration& config,
                                   bool train = true);

            protected:
                /**
                 * @brief Load segmentation annotations from text files
                 *
                 * Expected format (YOLO segmentation, one line per object):
                 * class_id x1 y1 x2 y2 x3 y3 ... (normalized polygon coordinates)
                 *
                 * Example (labels/image001.txt):
                 * 0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4
                 * 1 0.5 0.5 0.7 0.5 0.7 0.7 0.5 0.7
                 */
                void loadAnnotations() override;

                /**
                 * @brief Build transform pipeline for segmentation
                 *
                 * Training transforms:
                 * - Resize to target size
                 * - RandomHorizontalFlip
                 * - ColorJitter
                 *
                 * Validation transforms:
                 * - Resize to target size
                 */
                std::shared_ptr<Transforms::Transform> buildTransforms() override;

                /**
                 * @brief Convert annotation to target tensor
                 * @param index Sample index
                 * @param annotations Annotation (contains polygon coordinates)
                 * @return torch::Tensor Segmentation mask [C, H, W] one-hot encoded per class
                 */
                torch::Tensor getTargetTensor(size_t index, const Annotation& annotations) override;

            private:
                bool _train;  // Training mode flag
            };

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL

