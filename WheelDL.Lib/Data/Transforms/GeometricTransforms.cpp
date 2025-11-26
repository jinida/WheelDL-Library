#include "pch.h"
#include "GeometricTransforms.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <algorithm>

namespace WheelDL
{
    namespace Data
    {
        namespace Transforms
        {
            // ========================================
            // Resize Implementation
            // ========================================

            Resize::Resize(int targetHeight, int targetWidth, bool keepAspectRatio)
                : targetWidth_(targetWidth)
                , targetHeight_(targetHeight)
                , keepAspectRatio_(keepAspectRatio)
            {
                if (targetWidth <= 0 || targetHeight <= 0) {
                    throw std::invalid_argument("Target dimensions must be positive");
                }
            }

            void Resize::apply(cv::Mat& image, Annotation& annotations)
            {
                // Silently return for empty images to maintain pipeline consistency
                if (image.empty()) {
                    return;
                }

                int originalWidth = image.cols;
                int originalHeight = image.rows;

                if (keepAspectRatio_)
                {
                    // Calculate scale to fit within target size while maintaining aspect ratio
                    float scaleW = static_cast<float>(targetWidth_) / originalWidth;
                    float scaleH = static_cast<float>(targetHeight_) / originalHeight;
                    float scale = std::min(scaleW, scaleH);

                    int newWidth = static_cast<int>(originalWidth * scale);
                    int newHeight = static_cast<int>(originalHeight * scale);

                    // Resize image
                    cv::Mat resized;
                    cv::resize(image, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);

                    // Create padded image with target size (filled with zeros/black)
                    cv::Mat padded = cv::Mat::zeros(targetHeight_, targetWidth_, image.type());

                    // Center the resized image
                    int offsetX = (targetWidth_ - newWidth) / 2;
                    int offsetY = (targetHeight_ - newHeight) / 2;

                    resized.copyTo(padded(cv::Rect(offsetX, offsetY, newWidth, newHeight)));
                    image = padded;

                    // Update annotations: first scale, then translate
                    annotations.scale(scale, scale);
                    annotations.translate(static_cast<float>(offsetX), static_cast<float>(offsetY));
                }
                else
                {
                    // Simple resize (may distort aspect ratio)
                    cv::resize(image, image, cv::Size(targetWidth_, targetHeight_), 0, 0, cv::INTER_LINEAR);

                    // Scale annotations
                    float scaleX = static_cast<float>(targetWidth_) / originalWidth;
                    float scaleY = static_cast<float>(targetHeight_) / originalHeight;
                    annotations.scale(scaleX, scaleY);
                }
            }

            std::unique_ptr<Transform> Resize::clone() const
            {
                return std::make_unique<Resize>(targetHeight_, targetWidth_, keepAspectRatio_);
            }

            // ========================================
            // LetterBox Implementation
            // ========================================

            LetterBox::LetterBox(int targetHeight, int targetWidth,
                                 int fillR, int fillG, int fillB,
                                 bool center, bool scaleUp)
                : targetWidth_(targetWidth)
                , targetHeight_(targetHeight)
                , fillR_(fillR)
                , fillG_(fillG)
                , fillB_(fillB)
                , center_(center)
                , scaleUp_(scaleUp)
            {
                if (targetWidth <= 0 || targetHeight <= 0) {
                    throw std::invalid_argument("Target dimensions must be positive");
                }
            }

            void LetterBox::apply(cv::Mat& image, Annotation& annotations)
            {
                // Silently return for empty images to maintain pipeline consistency
                if (image.empty()) {
                    return;
                }

                int originalWidth = image.cols;
                int originalHeight = image.rows;

                // Calculate scale factor (maintaining aspect ratio)
                float scaleW = static_cast<float>(targetWidth_) / originalWidth;
                float scaleH = static_cast<float>(targetHeight_) / originalHeight;
                float scale = std::min(scaleW, scaleH);

                // Don't scale up if scaleUp_ is false
                if (!scaleUp_)
                {
                    scale = std::min(scale, 1.0f);
                }

                // Prevent degenerate cases with very small images
                // Ensure minimum scale to avoid zero-sized output
                const float MIN_SCALE = 0.1f;  // 10% minimum
                scale = std::max(scale, MIN_SCALE);

                // Calculate new dimensions
                int newWidth = static_cast<int>(originalWidth * scale);
                int newHeight = static_cast<int>(originalHeight * scale);

                // Ensure at least 1 pixel in each dimension
                newWidth = std::max(newWidth, 1);
                newHeight = std::max(newHeight, 1);

                // Resize image
                cv::Mat resized;
				auto flag = scale < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR;
                cv::resize(image, resized, cv::Size(newWidth, newHeight), 0, 0, flag);

                // Create canvas with fill color (OpenCV uses BGR order)
                cv::Mat canvas(targetHeight_, targetWidth_, image.type(),
                              cv::Scalar(fillB_, fillG_, fillR_));

                // Calculate offset
                int offsetX, offsetY;
                if (center_)
                {
                    offsetX = (targetWidth_ - newWidth) / 2;
                    offsetY = (targetHeight_ - newHeight) / 2;
                }
                else
                {
                    offsetX = 0;
                    offsetY = 0;
                }

                // Copy resized image to canvas
                resized.copyTo(canvas(cv::Rect(offsetX, offsetY, newWidth, newHeight)));
                image = canvas;

                // Update annotations: first scale, then translate
                annotations.scale(scale, scale);
                annotations.translate(static_cast<float>(offsetX), static_cast<float>(offsetY));
            }

            std::unique_ptr<Transform> LetterBox::clone() const
            {
                return std::make_unique<LetterBox>(targetWidth_, targetHeight_,
                                                   fillR_, fillG_, fillB_,
                                                   center_, scaleUp_);
            }

            // ========================================
            // RandomHorizontalFlip Implementation
            // ========================================

            RandomHorizontalFlip::RandomHorizontalFlip(float probability)
                : probability_(probability)
            {
                if (probability < 0.0f || probability > 1.0f) {
                    throw std::invalid_argument("Probability must be between 0.0 and 1.0");
                }
            }

            void RandomHorizontalFlip::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty()) {
                    return;
                }

                // Check probability using bernoulli distribution
                if (!rng_.bernoulli(probability_)) {
                    return;  // Don't flip
                }

                // Flip image horizontally
                cv::flip(image, image, 1);  // 1 = horizontal flip

                // Flip annotations
                annotations.flipHorizontal(image.cols);
            }

            std::unique_ptr<Transform> RandomHorizontalFlip::clone() const
            {
                return std::make_unique<RandomHorizontalFlip>(probability_);
            }

            // ========================================
            // RandomVerticalFlip Implementation
            // ========================================

            RandomVerticalFlip::RandomVerticalFlip(float probability)
                : probability_(probability)
            {
                if (probability < 0.0f || probability > 1.0f) {
                    throw std::invalid_argument("Probability must be between 0.0 and 1.0");
                }
            }

            void RandomVerticalFlip::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty()) {
                    return;
                }

                // Check probability using bernoulli distribution
                if (!rng_.bernoulli(probability_)) {
                    return;  // Don't flip
                }

                // Flip image vertically
                cv::flip(image, image, 0);  // 0 = vertical flip

                // Flip annotations
                annotations.flipVertical(image.rows);
            }

            std::unique_ptr<Transform> RandomVerticalFlip::clone() const
            {
                return std::make_unique<RandomVerticalFlip>(probability_);
            }


            RandomPerspective::RandomPerspective(float degrees,
                                                 float translate,
                                                 float scale,
                                                 float shear,
                                                 float perspective,
                                                 unsigned int targetWidth,
                                                 unsigned int targetHeight,
                                                 float probability,
                                                 unsigned int borderR, unsigned int borderG, unsigned int borderB,
                                                 float minAreaPixels)
                : degrees_(degrees)
                , translate_(translate)
                , scale_(scale)
                , shear_(shear)
                , perspective_(perspective)
                , probability_(probability)
                , targetWidth_(targetWidth)
                , targetHeight_(targetHeight)
                , borderR_(borderR)
                , borderG_(borderG)
                , borderB_(borderB)
                , minAreaPixels_(minAreaPixels)
            {
                if (probability < 0.0f || probability > 1.0f) {
                    throw std::invalid_argument("Probability must be between 0.0 and 1.0");
                }
            }

            void RandomPerspective::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty()) {
                    return;
                }

                // Check probability
                if (!rng_.bernoulli(probability_)) {
                    return;  // Don't apply transformation
                }

                unsigned int width = static_cast<unsigned int>(image.cols);
                unsigned int height = static_cast<unsigned int>(image.rows);

                // Validate memory requirements (assume 3 bytes per pixel for BGR)
                const size_t MAX_PIXELS = 256 * 1024 * 1024;  // 256M pixels max (~768MB)
                size_t totalPixels = static_cast<size_t>(targetWidth_) * static_cast<size_t>(targetHeight_);
                if (totalPixels > MAX_PIXELS) {
                    throw std::runtime_error("Target image size exceeds maximum memory limit");
                }

                // Generate random perspective transformation matrix
                cv::Mat transformMat = generatePerspectiveMatrix(width, height);

                annotations.transform(image, transformMat,
                                     cv::Size(static_cast<int>(targetWidth_), static_cast<int>(targetHeight_)),
                                     cv::Scalar(borderB_, borderG_, borderR_),
                                     minAreaPixels_);
            }

            cv::Mat RandomPerspective::generatePerspectiveMatrix(unsigned int width, unsigned int height)
            {
                // Center point
                float cx = width / 2.0f;
                float cy = height / 2.0f;

                // Start with identity matrix
                cv::Mat mat = cv::Mat::eye(3, 3, CV_64F);

                // Only apply non-identity transformations
                bool needsCentering = (degrees_ > 0.0f || scale_ > 0.0f || shear_ > 0.0f);

                if (needsCentering)
                {
                    // Translate to origin
                    cv::Mat T_to_origin = cv::Mat::eye(3, 3, CV_64F);
                    T_to_origin.at<double>(0, 2) = -cx;
                    T_to_origin.at<double>(1, 2) = -cy;
                    mat = T_to_origin * mat;
                }

                // 2. Rotation (around center)
                if (degrees_ > 0.0f)
                {
                    float angle = rng_.uniformFloat(-degrees_, degrees_);
                    float rad = angle * 3.14159265359f / 180.0f;
                    float cosA = std::cos(rad);
                    float sinA = std::sin(rad);

                    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
                    R.at<double>(0, 0) = cosA;
                    R.at<double>(0, 1) = -sinA;
                    R.at<double>(1, 0) = sinA;
                    R.at<double>(1, 1) = cosA;
                    mat = R * mat;
                }

                // 3. Scale
                if (scale_ > 0.0f)
                {
                    float s = rng_.uniformFloat(1.0f - scale_, 1.0f + scale_);
                    // Prevent degenerate transformations (scale too small or negative)
                    s = std::max(s, 0.1f);
                    cv::Mat S = cv::Mat::eye(3, 3, CV_64F);
                    S.at<double>(0, 0) = s;
                    S.at<double>(1, 1) = s;
                    mat = S * mat;
                }

                // 4. Shear
                if (shear_ > 0.0f)
                {
                    float shearX = rng_.uniformFloat(-shear_, shear_) * 3.14159265359f / 180.0f;
                    float shearY = rng_.uniformFloat(-shear_, shear_) * 3.14159265359f / 180.0f;
                    cv::Mat Sh = cv::Mat::eye(3, 3, CV_64F);
                    Sh.at<double>(0, 1) = std::tan(shearX);
                    Sh.at<double>(1, 0) = std::tan(shearY);
                    mat = Sh * mat;
                }

                // 5. Perspective
                if (perspective_ > 0.0f)
                {
                    float px = rng_.uniformFloat(-perspective_, perspective_);
                    float py = rng_.uniformFloat(-perspective_, perspective_);
                    cv::Mat P = cv::Mat::eye(3, 3, CV_64F);
                    P.at<double>(2, 0) = px;
                    P.at<double>(2, 1) = py;
                    mat = P * mat;
                }

                if (needsCentering)
                {
                    // Translate back from origin
                    cv::Mat T_from_origin = cv::Mat::eye(3, 3, CV_64F);
                    T_from_origin.at<double>(0, 2) = cx;
                    T_from_origin.at<double>(1, 2) = cy;
                    mat = T_from_origin * mat;
                }

                // 1. Global translation (applied last in composition)
                if (translate_ > 0.0f)
                {
                    float tx = rng_.uniformFloat(-translate_, translate_) * width;
                    float ty = rng_.uniformFloat(-translate_, translate_) * height;
                    cv::Mat T1 = cv::Mat::eye(3, 3, CV_64F);
                    T1.at<double>(0, 2) = tx;
                    T1.at<double>(1, 2) = ty;
                    mat = T1 * mat;
                }

                return mat;
            }

            std::unique_ptr<Transform> RandomPerspective::clone() const
            {
                return std::make_unique<RandomPerspective>(degrees_, translate_, scale_, shear_, perspective_,
                                                           targetWidth_, targetHeight_, probability_,
                                                           borderR_, borderG_, borderB_, minAreaPixels_);
            }

            EfficientADTransform::EfficientADTransform(std::unique_ptr<Compose> a, std::unique_ptr<Compose> b)
                : branchA_(std::move(a)), branchB_(std::move(b))
            {
                if (!branchA_ || !branchB_) {
                    throw std::invalid_argument("Both branchA and branchB must be valid Compose objects.");
                }
            }

            void EfficientADTransform::apply(cv::Mat& image, Annotation& annotations)
            {
                if (image.empty())
                    throw std::runtime_error("Input image is empty.");

                cv::Mat a = image.clone();
                branchA_->apply(a, annotations);

                cv::Mat b = image.clone();
                branchB_->apply(b, annotations);

                std::vector<cv::Mat> chA, chB;
                cv::split(a, chA);
                cv::split(b, chB);

                chA.insert(chA.end(), chB.begin(), chB.end());

                cv::merge(chA, image);
            }

            std::unique_ptr<Transform> EfficientADTransform::clone() const
            {
                return std::make_unique<EfficientADTransform>(
                    std::unique_ptr<Compose>(static_cast<Compose*>(branchA_->clone().release())),
                    std::unique_ptr<Compose>(static_cast<Compose*>(branchB_->clone().release()))
                );
            }

} // namespace Transforms
    } // namespace Data
} // namespace WheelDL
