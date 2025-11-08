#pragma once

#include "Data/Common/Annotation.h"
#include "Utils/Common/Random.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>

namespace WheelDL
{
    namespace Data
    {
        namespace Augmentation
        {
            /**
             * @class Mosaic
             * @brief Mosaic augmentation (combines 4 images in a 2x2 grid)
             *
             * Implements the Mosaic augmentation used in YOLOv5/v8.
             * Places 4 images in a 2x2 grid based on a random center point.
             *
             * NOTE: Does not inherit from Transform class.
             *       Requires multiple images, so it is handled at the Dataset level.
             *
             * Processing flow:
             * 1. Randomly select center point (0.25 ~ 0.75 range)
             * 2. Place 4 images in each quadrant (LetterBox method to maintain aspect ratio)
             * 3. Transform annotation coordinates (offset + scale)
             * 4. Clip annotations to actual image area only
             *
             * Augmentation behavior (follows YOLOv5/v8):
             * - After Mosaic composition, Transform pipeline is applied to entire image
             * - 4 regions are transformed together like one scene (RandomPerspective, ColorJitter, etc.)
             * - This is intentional behavior and provides natural augmentation effects
             */
            class Mosaic
            {
            public:
                /**
                 * @brief Mosaic constructor
                 * @param targetWidth Output image width
                 * @param targetHeight Output image height
                 * @param probability Application probability (0.0~1.0)
                 * @param borderR Border area R value (0-255)
                 * @param borderG Border area G value (0-255)
                 * @param borderB Border area B value (0-255)
                 */
                Mosaic(unsigned int targetWidth = 640,
                       unsigned int targetHeight = 640,
                       float probability = 1.0f,
                       unsigned int borderR = 0,
                       unsigned int borderG = 0,
                       unsigned int borderB = 0);

                /**
                 * @brief Apply Mosaic to 4 images
                 * @param images Input images (vector size must be 4)
                 * @param annotations Annotation for each image (vector size must be 4)
                 * @param outputImage Combined output image
                 * @param outputAnnotation Combined annotation
                 * @return bool true on success, false if skipped by probability
                 */
                bool apply(const std::vector<cv::Mat>& images,
                           const std::vector<Annotation>& annotations,
                           cv::Mat& outputImage,
                           Annotation& outputAnnotation);

                /**
                 * @brief Set seed for reproducibility
                 * @param seed Random seed value
                 */
                void setSeed(unsigned int seed) { rng_.setSeed(seed); }

                float getProbability() const { return probability_; }

            private:
                unsigned int targetWidth_;
                unsigned int targetHeight_;
                float probability_;
                unsigned int borderR_, borderG_, borderB_;
                WheelDL::Utils::Random rng_;  // Own RNG (ensures reproducibility)

                /**
                 * @brief Compute center point for 4 images (random position)
                 * @return std::pair<int, int> Center coordinates (centerX, centerY)
                 */
                std::pair<int, int> computeCenter();

                /**
                 * @brief Place image in specific quadrant
                 * @param image Input image
                 * @param annotations Input annotation
                 * @param output Output image (in/out)
                 * @param quadrant Quadrant index (0: top-left, 1: top-right, 2: bottom-left, 3: bottom-right)
                 * @param centerX Center X coordinate
                 * @param centerY Center Y coordinate
                 * @param outputAnnotations Output annotation (in/out)
                 */
                void placeImageInQuadrant(
                    const cv::Mat& image,
                    const Annotation& annotations,
                    cv::Mat& output,
                    int quadrant,
                    int centerX,
                    int centerY,
                    Annotation& outputAnnotations);
            };

        } // namespace Augmentation
    } // namespace Data
} // namespace WheelDL
