#include "pch.h"
#include "DatasetAugmentation.h"
#include <stdexcept>
#include <algorithm>

namespace WheelDL
{
    namespace Data
    {
        namespace Augmentation
        {
            Mosaic::Mosaic(unsigned int targetWidth,
                           unsigned int targetHeight,
                           float probability,
                           unsigned int borderR,
                           unsigned int borderG,
                           unsigned int borderB)
                : targetWidth_(targetWidth)
                , targetHeight_(targetHeight)
                , probability_(probability)
                , borderR_(borderR)
                , borderG_(borderG)
                , borderB_(borderB)
                , rng_()
            {
            }

            bool Mosaic::apply(
                const std::vector<cv::Mat>& images,
                const std::vector<Annotation>& annotations,
                cv::Mat& outputImage,
                Annotation& outputAnnotation)
            {
                // Validation
                if (images.size() != 4 || annotations.size() != 4)
                {
                    throw std::invalid_argument("Mosaic requires exactly 4 images and annotations");
                }

                // Probability check
                if (rng_.uniformFloat(0.0f, 1.0f) > probability_)
                {
                    return false;  // Skip application
                }

                // 1. Compute center point
                auto [centerX, centerY] = computeCenter();

                // 2. Create output image (initialize with border color)
                outputImage = cv::Mat(targetHeight_, targetWidth_, CV_8UC3,
                                      cv::Scalar(borderB_, borderG_, borderR_));

                // 3. Determine LabelType (based on first image)
                outputAnnotation = Annotation(annotations[0].getLabelType());

                // 4. Place each image in quadrant
                for (int i = 0; i < 4; ++i)
                {
                    placeImageInQuadrant(images[i], annotations[i], outputImage,
                                         i, centerX, centerY, outputAnnotation);
                }

                return true;
            }

            std::pair<int, int> Mosaic::computeCenter()
            {
                // Select center point randomly in 0.25 ~ 0.75 range
                int centerX = static_cast<int>(rng_.uniformFloat(0.25f, 0.75f) * targetWidth_);
                int centerY = static_cast<int>(rng_.uniformFloat(0.25f, 0.75f) * targetHeight_);
                return { centerX, centerY };
            }

            void Mosaic::placeImageInQuadrant(
                const cv::Mat& image,
                const Annotation& annotations,
                cv::Mat& output,
                int quadrant,
                int centerX,
                int centerY,
                Annotation& outputAnnotations)
            {
                // Calculate region for each quadrant
                int x1_output, y1_output, x2_output, y2_output;

                switch (quadrant)
                {
                case 0: // Top-left
                    x1_output = 0;
                    y1_output = 0;
                    x2_output = centerX;
                    y2_output = centerY;
                    break;
                case 1: // Top-right
                    x1_output = centerX;
                    y1_output = 0;
                    x2_output = targetWidth_;
                    y2_output = centerY;
                    break;
                case 2: // Bottom-left
                    x1_output = 0;
                    y1_output = centerY;
                    x2_output = centerX;
                    y2_output = targetHeight_;
                    break;
                case 3: // Bottom-right
                    x1_output = centerX;
                    y1_output = centerY;
                    x2_output = targetWidth_;
                    y2_output = targetHeight_;
                    break;
                default:
                    throw std::invalid_argument("Invalid quadrant index: " + std::to_string(quadrant));
                }

                int outputWidth = x2_output - x1_output;
                int outputHeight = y2_output - y1_output;

                // LetterBox method: resize while maintaining aspect ratio
                float scaleX = static_cast<float>(outputWidth) / image.cols;
                float scaleY = static_cast<float>(outputHeight) / image.rows;
                float scale = std::min(scaleX, scaleY);  // Choose smaller value to maintain aspect ratio

                int newWidth = static_cast<int>(image.cols * scale);
                int newHeight = static_cast<int>(image.rows * scale);

                // Resize
                cv::Mat resized;
                cv::resize(image, resized, cv::Size(newWidth, newHeight));

                // Calculate padding (center alignment)
                int padX = (outputWidth - newWidth) / 2;
                int padY = (outputHeight - newHeight) / 2;

                // Copy to output region (padding area is already initialized with border color)
                cv::Rect roi(x1_output + padX, y1_output + padY, newWidth, newHeight);
                resized.copyTo(output(roi));

                // Transform annotations (using same scale + offset)
                Annotation transformedAnnotations = annotations.clone();
                transformedAnnotations.scale(scale, scale);  // Same scale for no distortion
                transformedAnnotations.translate(
                    static_cast<float>(x1_output + padX),
                    static_cast<float>(y1_output + padY)
                );

                // Clip to actual image area only (excluding padding area)
                // Important: Clip to actual image placement area, not entire quadrant
                transformedAnnotations.clipToBounds(
                    static_cast<float>(x1_output + padX),              // Actual image start (including padding)
                    static_cast<float>(y1_output + padY),
                    static_cast<float>(x1_output + padX + newWidth),   // Actual image end
                    static_cast<float>(y1_output + padY + newHeight)
                );

                // Merge
                for (size_t i = 0; i < transformedAnnotations.size(); ++i)
                {
                    outputAnnotations.addObject(
                        transformedAnnotations.getClasses()[i],
                        transformedAnnotations.getPoints()[i]
                    );
                }
            }

        } // namespace Augmentation
    } // namespace Data
} // namespace WheelDL
