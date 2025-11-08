#pragma once

#include "Data/Common/Annotation.h"
#include "Utils/Common/Random.h"
#include <memory>
#include <string>
#include <cstdint>

// Forward declaration to avoid OpenCV header dependency
namespace cv {
    class Mat;
}

namespace WheelDL
{
    namespace Data
    {
        namespace Transforms
        {
            /**
             * @class Transform
             * @brief Abstract base class for data augmentation transforms
             *
             * All transforms operate on both images and their annotations.
             * Transforms can be chained together using Compose.
             */
            class Transform
            {
            public:
                virtual ~Transform() = default;

                /**
                 * @brief Apply transformation to image and annotations
                 * @param image Input/output image (modified in-place)
                 * @param annotations Input/output annotations (modified in-place)
                 */
                virtual void apply(cv::Mat& image, Annotation& annotations) = 0;

                /**
                 * @brief Get transform name for debugging/logging
                 * @return Transform name
                 */
                virtual std::string getName() const = 0;

                /**
                 * @brief Check if transform is deterministic or random
                 * @return true if transform has randomness
                 */
                virtual bool isRandom() const { return false; }

                /**
                 * @brief Set random seed (only affects random transforms)
                 * @param seed Random seed value
                 */
                virtual void setSeed(uint64_t seed) {}

                /**
                 * @brief Clone the transform (for multi-threaded data loading)
                 * @return Unique pointer to cloned transform
                 */
                virtual std::unique_ptr<Transform> clone() const = 0;
            };

            /**
             * @class RandomTransform
             * @brief Base class for random transforms with reproducible RNG
             *
             * Provides centralized random number generation using Utils::Random.
             * Each clone gets a fresh RNG instance to avoid state sharing in multi-threading.
             *
             * Important: Seed management is centralized through setSeed() only.
             * Constructors do NOT accept seed parameters to ensure consistent seeding via Compose.
             */
            class RandomTransform : public Transform
            {
            public:
                /**
                 * @brief Construct with random initialization
                 * Note: Use setSeed() to set reproducible seed
                 */
                RandomTransform();

                bool isRandom() const final { return true; }

                void setSeed(uint64_t seed) override;

            protected:
                WheelDL::Utils::Random rng_;  ///< Random number generator
            };

            /**
             * @class Compose
             * @brief Chain multiple transforms together
             *
             * Applies transforms sequentially in the order they were added.
             */
            class Compose : public Transform
            {
            public:
                /**
                 * @brief Construct empty compose
                 */
                Compose() = default;

                /**
                 * @brief Add a transform to the pipeline
                 * @param transform Transform to add (ownership transferred)
                 */
                void addTransform(std::unique_ptr<Transform> transform);

                /**
                 * @brief Apply all transforms in sequence
                 * @param image Input/output image
                 * @param annotations Input/output annotations
                 */
                void apply(cv::Mat& image, Annotation& annotations) override;

                std::string getName() const override { return "Compose"; }

                bool isRandom() const override;

                /**
                 * @brief Set seed for all transforms in the pipeline
                 * @param baseSeed Base seed value (each transform gets baseSeed + index)
                 */
                void setSeed(uint64_t baseSeed) override;

                /**
                 * @brief Check if pipeline is empty
                 */
                bool empty() const { return transforms_.empty(); }

                /**
                 * @brief Get number of transforms in pipeline
                 */
                size_t size() const { return transforms_.size(); }

                std::unique_ptr<Transform> clone() const override;

            private:
                std::vector<std::unique_ptr<Transform>> transforms_;
            };

        } // namespace Transforms
    } // namespace Data
} // namespace WheelDL
