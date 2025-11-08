#pragma once

#include "BaseDataset.h"

namespace WheelDL
{
    namespace Data
    {
        namespace Dataset
        {
            /**
             * @class OBBDataset
             * @brief Dataset for Oriented Bounding Box (OBB) detection tasks
             *
             * Annotation format: YOLO-OBB format (class_id, x_center, y_center, width, height, angle)
             * All coordinates are normalized to [0, 1], angle in radians
             */
            class OBBDataset : public BaseDataset<OBBDataset>
            {
            public:
                /**
                 * @brief Construct OBB dataset
                 * @param dataPath Path to image directory
                 * @param annotationPath Path to annotation directory (YOLO-OBB format)
                 * @param config Configuration object
                 * @param train Whether this is training dataset (affects transforms)
                 */
                OBBDataset(const std::string& dataPath,
                          const std::string& annotationPath,
                          const Config::Configuration& config,
                          bool train = true);

            protected:
                /**
                 * @brief Load OBB annotations from files
                 *
                 * Expected format (YOLO-OBB format, one line per object):
                 * class_id x_center y_center width height angle
                 *
                 * Example (labels/image001.txt):
                 * 0 0.5 0.5 0.3 0.4 0.785  (45 degrees in radians)
                 */
                void loadAnnotations() override;

                /**
                 * @brief Build transform pipeline for OBB
                 *
                 * Similar to detection but handles rotation correctly
                 */
                std::shared_ptr<Transforms::Transform> buildTransforms() override;

                /**
                 * @brief Convert annotation to target tensor
                 * @param index Sample index
                 * @param annotations Annotation (contains oriented bounding boxes)
                 * @return torch::Tensor Target tensor [num_objects, 6] (class, x, y, w, h, angle)
                 */
                torch::Tensor getTargetTensor(size_t index, const Annotation& annotations) override;

            private:
                bool _train;  // Training mode flag
            };

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
