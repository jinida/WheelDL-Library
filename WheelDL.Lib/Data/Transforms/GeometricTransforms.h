#pragma once

#include "Transform.h"
#include <random>

namespace WheelDL
{
    namespace Data
    {
        namespace Transforms
        {
            /**
             * @class Resize
             * @brief Resize image and scale annotations accordingly
             *
             * Simple resize operation that can optionally maintain aspect ratio.
             * If keepAspectRatio is false, stretches the image to target size.
             * If keepAspectRatio is true, resizes maintaining aspect ratio and adds padding.
             */
            class Resize : public Transform
            {
            public:
                /**
                 * @brief Construct resize transform
                 * @param targetHeight Target height (rows)
                 * @param targetWidth Target width (cols)
                 * @param keepAspectRatio If true, maintains aspect ratio with padding
                 */
                Resize(int targetHeight, int targetWidth, bool keepAspectRatio = false);

                void apply(cv::Mat &image, Annotation &annotations) override;
                std::string getName() const override { return "Resize"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                int targetWidth_;
                int targetHeight_;
                bool keepAspectRatio_;
            };

            /**
             * @class LetterBox
             *
             * Resizes image to fit within target size while maintaining aspect ratio,
             * then pads the remaining area with a specified color (typically gray).
             * This is the standard preprocessing used in YOLO models.
             *
             * Difference from Resize:
             * - LetterBox always maintains aspect ratio and adds padding
             * - Uses centered padding with configurable fill color
             * - Annotation are automatically adjusted for the new position/scale
             */
            class LetterBox : public Transform
            {
            public:
                /**
                 * @brief Construct letterbox transform
                 * @param targetHeight Target height (rows)
                 * @param targetWidth Target width (cols)
                 * @param fillR Red channel value for padding (0-255)
                 * @param fillG Green channel value for padding (0-255)
                 * @param fillB Blue channel value for padding (0-255)
                 * @param center If true, center the image; if false, align to top-left
                 * @param scaleUp If false, only scale down (no upscaling); if true, allow upscaling
                 */
                LetterBox(int targetHeight, int targetWidth,
                          int fillR = 0, int fillG = 0, int fillB = 0,
                          bool center = true, bool scaleUp = true);

                void apply(cv::Mat &image, Annotation &annotations) override;
                std::string getName() const override { return "LetterBox"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                int targetWidth_;
                int targetHeight_;
                int fillR_, fillG_, fillB_;
                bool center_;
                bool scaleUp_;
            };

            /**
             * @class RandomHorizontalFlip
             * @brief Randomly flip image horizontally
             */
            class RandomHorizontalFlip : public RandomTransform
            {
            public:
                /**
                 * @brief Construct random horizontal flip
                 * @param probability Probability of applying flip (0.0 to 1.0)
                 */
                explicit RandomHorizontalFlip(float probability = 0.5f);

                void apply(cv::Mat &image, Annotation &annotations) override;
                std::string getName() const override { return "RandomHorizontalFlip"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                float probability_;
            };

            /**
             * @class RandomVerticalFlip
             * @brief Randomly flip image vertically
             */
            class RandomVerticalFlip : public RandomTransform
            {
            public:
                /**
                 * @brief Construct random vertical flip
                 * @param probability Probability of applying flip (0.0 to 1.0)
                 */
                explicit RandomVerticalFlip(float probability = 0.5f);

                void apply(cv::Mat &image, Annotation &annotations) override;
                std::string getName() const override { return "RandomVerticalFlip"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                float probability_;
            };

            class RandomPerspective : public RandomTransform
            {
            public:
                /**
                 * @brief Construct random perspective transform
                 * @param degrees Rotation range in degrees (e.g., 10.0 means [-10, +10])
                 * @param translate Translation range as fraction of image size (e.g., 0.1 means [-10%, +10%])
                 * @param scale Scale range (e.g., 0.1 means [0.9, 1.1])
                 * @param shear Shear range in degrees (e.g., 10.0 means [-10, +10])
                 * @param perspective Perspective range (e.g., 0.0001 for subtle effect)
                 * @param probability Probability of applying transform
                 * @param borderR Red channel value for fill areas (0-255)
                 * @param borderG Green channel value for fill areas (0-255)
                 * @param borderB Blue channel value for fill areas (0-255)
                 * @param minAreaPixels Minimum area in pixels for object to be kept after transform
                 */
                RandomPerspective(float degrees = 0.0f,
                                 float translate = 0.0f,
                                 float scale = 0.0f,
                                 float shear = 0.0f,
                                 float perspective = 0.0f,
                                 unsigned int targetWidth = 640,
                                 unsigned int targetHeight = 640,
                                 float probability = 0.5f,
                                 unsigned int borderR = 0, unsigned int borderG = 0, unsigned int borderB = 0,
                                 float minAreaPixels = 5.0f);

                void apply(cv::Mat &image, Annotation &annotations) override;
                std::string getName() const override { return "RandomPerspective"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                float degrees_;
                float translate_;
                float scale_;
                float shear_;
                float perspective_;
                float probability_;
                unsigned int targetWidth_;
                unsigned int targetHeight_;
                unsigned int borderR_, borderG_, borderB_;
                float minAreaPixels_;

                /**
                 * @brief Generate random 3x3 perspective transformation matrix
                 * @param width Image width
                 * @param height Image height
                 * @return 3x3 transformation matrix
                 */
                cv::Mat generatePerspectiveMatrix(unsigned int width, unsigned int height);
            };

            class EfficientADTransform : public Transform
            {
            public:
                EfficientADTransform(std::unique_ptr<Compose> branchA, std::unique_ptr<Compose> branchB);

                void apply(cv::Mat& image, Annotation& annotations) override;

                std::string getName() const override { return "EfficientADTransform"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                std::unique_ptr<Compose> branchA_;
                std::unique_ptr<Compose> branchB_;

                EfficientADTransform(const EfficientADTransform&) = delete;
                EfficientADTransform& operator=(const EfficientADTransform&) = delete;
            };
        } // namespace Transforms
    } // namespace Data
} // namespace WheelDL
