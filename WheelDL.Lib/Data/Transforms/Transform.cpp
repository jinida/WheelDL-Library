#include "pch.h"
#include "Transform.h"
#include <random>

namespace WheelDL
{
    namespace Data
    {
        namespace Transforms
        {
            // ========================================
            // RandomTransform Implementation
            // ========================================

            RandomTransform::RandomTransform()
                : rng_(std::random_device{}())  // Always random initialization
            {
            }

            void RandomTransform::setSeed(uint64_t seed)
            {
                rng_.setSeed(static_cast<unsigned int>(seed));
            }

            // ========================================
            // Compose Implementation
            // ========================================

            void Compose::addTransform(std::unique_ptr<Transform> transform)
            {
                if (transform) {
                    transforms_.push_back(std::move(transform));
                }
            }

            void Compose::apply(cv::Mat& image, Annotation& annotations)
            {
                // Early return for empty pipeline (optimization)
                if (transforms_.empty()) {
                    return;
                }

                // Apply all transforms in sequence
                for (auto& transform : transforms_)
                {
                    // Safety check for nullptr (should never happen, but defensive)
                    if (transform) {
                        transform->apply(image, annotations);
                    }
                }
            }

            bool Compose::isRandom() const
            {
                for (const auto& transform : transforms_)
                {
                    if (transform && transform->isRandom())
                    {
                        return true;
                    }
                }
                return false;
            }

            void Compose::setSeed(uint64_t baseSeed)
            {
                // Set seed for each transform (baseSeed + index for diversity)
                for (size_t i = 0; i < transforms_.size(); ++i)
                {
                    if (transforms_[i]) {
                        transforms_[i]->setSeed(baseSeed + i);
                    }
                }
            }

            std::unique_ptr<Transform> Compose::clone() const
            {
                auto cloned = std::make_unique<Compose>();
                for (const auto& transform : transforms_)
                {
                    if (transform) {
                        cloned->addTransform(transform->clone());
                    }
                }
                return cloned;
            }

        } // namespace Transforms
    } // namespace Data
} // namespace WheelDL
