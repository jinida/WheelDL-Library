#pragma once

#include "Transform.h"
#include <array>
#include <random>

namespace WheelDL
{
    namespace Data
    {
        namespace Transforms
        {
            /**
             * @class Normalize
             * @brief Normalize image using mean and standard deviation
             *
             * Formula: output = (input - mean) / std
             */
            class Normalize : public Transform
            {
            public:
                /**
                 * @brief Construct normalization transform
                 * @param mean Mean values for each channel (RGB)
                 * @param std Standard deviation values for each channel (RGB)
                 */
                Normalize(const std::array<float, 3>& mean, const std::array<float, 3>& std);

                /**
                 * @brief Construct with ImageNet defaults
                 */
                static Normalize imageNet();

                void apply(cv::Mat& image, Annotation& annotations) override;
                std::string getName() const override { return "Normalize"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                std::array<float, 3> mean_;
                std::array<float, 3> std_;
            };

            /**
             * @class ToTensor
             * @brief Convert image from HWC uint8 [0, 255] to CHW float32 [0.0, 1.0]
             *
             * Also converts from BGR to RGB if specified.
             */
            class ToTensor : public Transform
            {
            public:
                /**
                 * @brief Construct ToTensor transform
                 * @param bgrToRgb Convert from BGR to RGB
                 */
                explicit ToTensor(bool bgrToRgb = true);

                void apply(cv::Mat& image, Annotation& annotations) override;
                std::string getName() const override { return "ToTensor"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                bool bgrToRgb_;
            };

            /**
             * @class ColorJitter
             * @brief Randomly change brightness, contrast, saturation, and hue
             */
            class ColorJitter : public RandomTransform
            {
            public:
                /**
                 * @brief Construct color jitter transform
                 * @param brightness Brightness factor range [1-b, 1+b]
                 * @param contrast Contrast factor range [1-c, 1+c]
                 * @param saturation Saturation factor range [1-s, 1+s]
                 * @param hue Hue shift range [-h, h] (in degrees)
                 */
                ColorJitter(float brightness = 0.2f, float contrast = 0.2f,
                           float saturation = 0.2f, float hue = 0.0f);

                void apply(cv::Mat& image, Annotation& annotations) override;
                std::string getName() const override { return "ColorJitter"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                float brightness_;
                float contrast_;
                float saturation_;
                float hue_;
            };

            /**
             * @class GaussianBlur
             * @brief Apply Gaussian blur with random kernel size
             *
             * Note: Kernel sizes are automatically adjusted to odd values as required by OpenCV.
             * Input values are incremented to next odd number if even.
             */
            class GaussianBlur : public RandomTransform
            {
            public:
                /**
                 * @brief Construct Gaussian blur transform
                 * @param minKernelSize Minimum kernel size (will be adjusted to odd if even)
                 * @param maxKernelSize Maximum kernel size (will be adjusted to odd if even)
                 * @param probability Probability of applying blur
                 * @throws std::invalid_argument if kernel sizes are invalid
                 */
                GaussianBlur(int minKernelSize = 3, int maxKernelSize = 7, float probability = 0.5f);

                void apply(cv::Mat& image, Annotation& annotations) override;
                std::string getName() const override { return "GaussianBlur"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                int minKernelSize_;
                int maxKernelSize_;
                float probability_;

                /**
                 * @brief Ensure kernel size is odd (required by OpenCV)
                 * @param size Input size
                 * @return Odd kernel size (size if already odd, size+1 if even)
                 */
                static int makeOddKernelSize(int size);
            };

            /**
             * @class GaussianNoise
             * @brief Add random Gaussian noise to image
             */
            class GaussianNoise : public RandomTransform
            {
            public:
                /**
                 * @brief Construct Gaussian noise transform
                 * @param mean Mean of Gaussian distribution
                 * @param stddev Standard deviation of Gaussian distribution
                 * @param probability Probability of applying noise
                 */
                GaussianNoise(float mean = 0.0f, float stddev = 25.0f);

                void apply(cv::Mat& image, Annotation& annotations) override;
                std::string getName() const override { return "GaussianNoise"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                float mean_;
                float stddev_;
                float probability_ = 0.01;
            };

            /**
             * @class CLAHE
             * @brief Contrast Limited Adaptive Histogram Equalization
             */
            class CLAHE : public RandomTransform
            {
            public:
                /**
                 * @brief Construct CLAHE transform
                 * @param clipLimit Threshold for contrast limiting
                 * @param tileGridSize Size of grid for histogram equalization
                 */
                CLAHE(float clipLimit = 2.0f, int tileGridSize = 8);
                void apply(cv::Mat& image, Annotation& annotations) override;
                std::string getName() const override { return "CLAHE"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                float clipLimit_;
                int tileGridSize_;
                float probability_ = 0.01;
                cv::Ptr<cv::CLAHE> clahe_ = cv::createCLAHE(clipLimit_, cv::Size(tileGridSize_, tileGridSize_));
            };

            /**
             * @class RandomGamma
             * @brief Apply random gamma correction
             */
            class RandomGamma : public RandomTransform
            {
            public:
                /**
                 * @brief Construct random gamma transform
                 * @param gammaRange Gamma range [1-range, 1+range]
                 * @param probability Probability of applying gamma correction
                 */
                RandomGamma(float gammaRange = 0.4f);

                void apply(cv::Mat& image, Annotation& annotations) override;
                std::string getName() const override { return "RandomGamma"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                float gammaRange_;
                float probability_;
            };

            /**
             * @class ToGray
             * @brief Convert image to grayscale
             */
			class ToGray : public RandomTransform
            {
            public:
                /**
                 * @brief Construct ToGray transform
                 * @param keepChannels If true, output 3-channel grayscale (same value in all channels)
                 */
                explicit ToGray(bool keepChannels = false);

                void apply(cv::Mat& image, Annotation& annotations) override;
                std::string getName() const override { return "ToGray"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                bool keepChannels_;
				float probability_ = 0.01;
            };

            /**
             * @class SaltAndPepper
             * @brief Add salt and pepper noise to image
             */
            class SaltAndPepper : public RandomTransform
            {
            public:
                /**
                 * @brief Construct salt and pepper noise transform
                 * @param saltProb Probability of salt noise (white pixels)
                 * @param pepperProb Probability of pepper noise (black pixels)
                 * @param probability Probability of applying noise
                 */
                SaltAndPepper(float saltProb = 0.01f, float pepperProb = 0.01f);

                void apply(cv::Mat& image, Annotation& annotations) override;
                std::string getName() const override { return "SaltAndPepper"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                float saltProb_;
                float pepperProb_;
                float probability_ = 0.01;
            };

            /**
             * @class MedianBlur
             * @brief Apply median blur filter
             *
             * Note: Kernel size must be odd and greater than 1
             */
            class MedianBlur : public RandomTransform
            {
            public:
                /**
                 * @brief Construct median blur transform
                 * @param minKernelSize Minimum kernel size (will be adjusted to odd if even)
                 * @param maxKernelSize Maximum kernel size (will be adjusted to odd if even)
                 * @param probability Probability of applying blur
                 */
                MedianBlur(int minKernelSize = 3, int maxKernelSize = 7, float probability = 0.5f);

                void apply(cv::Mat& image, Annotation& annotations) override;
                std::string getName() const override { return "MedianBlur"; }
                std::unique_ptr<Transform> clone() const override;

            private:
                int minKernelSize_;
                int maxKernelSize_;
                float probability_;

                static int makeOddKernelSize(int size);
            };

        } // namespace Transforms
    } // namespace Data
} // namespace WheelDL
