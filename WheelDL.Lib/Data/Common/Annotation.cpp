#include "pch.h"
#include "Annotation.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif



namespace WheelDL
{
    namespace Data
    {
        // Epsilon for floating point comparisons
        constexpr float FLOAT_EPSILON = 1e-6f;
		constexpr int MAX_WIDTH = 100000;
		constexpr int MAX_HEIGHT = 100000;

        // ========== Construction ==========

        Annotation::Annotation(LabelType type, bool isNoramlized)
			: labelType_(type), _isNormalized(isNoramlized)
        {
        }

        Annotation Annotation::forClassification(int classId)
        {
            Annotation ann(LabelType::NONE);
            ann.classes_.push_back(classId);
            ann.points_.push_back(std::vector<float>());  // Empty points for classification
            return ann;
        }

        // ========== Object Management ==========

        void Annotation::addObject(int classId, const std::vector<float>& points)
        {
            // Exception-safe: only modify state if both operations succeed
            try
            {
                classes_.push_back(classId);
                try
                {
                    points_.push_back(points);
                }
                catch (...)
                {
                    // If points_ push fails, rollback classes_
                    classes_.pop_back();
                    throw;
                }
            }
            catch (...)
            {
                // Ensure classes_ and points_ remain synchronized
                throw;
            }
        }

        size_t Annotation::size() const
        {
            return classes_.size();
        }

        bool Annotation::empty() const
        {
            return classes_.empty();
        }

        void Annotation::clear()
        {
            classes_.clear();
            points_.clear();
        }

        Annotation Annotation::clone() const
        {
            Annotation copy(labelType_);
            // Reserve space to avoid reallocation during copy
            copy.classes_.reserve(classes_.size());
            copy.points_.reserve(points_.size());
            copy.classes_ = classes_;
            copy.points_ = points_;
            copy._isNormalized = _isNormalized;
            return copy;
        }

        // ========== Coordinate Transformations ==========

        void Annotation::normalize(int imageWidth, int imageHeight)
        {
            // Make normalize idempotent - if already normalized, skip
            if (_isNormalized) {
                return;
            }

            if (imageWidth <= 0 || imageHeight <= 0) {
                throw std::invalid_argument("Image dimensions must be positive");
            }

            float invW = 1.0f / static_cast<float>(imageWidth);
            float invH = 1.0f / static_cast<float>(imageHeight);

            // Move switch outside loop for better branch prediction
            switch (labelType_)
            {
            case LabelType::XYWH:
                // [cx, cy, w, h] - normalize all values
                for (auto& coords : points_)
                {
                    if (coords.size() >= 4)
                    {
                        coords[0] *= invW;  // cx
                        coords[1] *= invH;  // cy
                        coords[2] *= invW;  // w
                        coords[3] *= invH;  // h
                    }
                }
                break;

            case LabelType::XYWHR:
                // [cx, cy, w, h, r] - normalize cx, cy, w, h, but NOT rotation
                for (auto& coords : points_)
                {
                    if (coords.size() >= 4)
                    {
                        coords[0] *= invW;  // cx
                        coords[1] *= invH;  // cy
                        coords[2] *= invW;  // w
                        coords[3] *= invH;  // h
                        // coords[4] is rotation - don't normalize
                    }
                }
                break;

            default:
                // XYXY, POLYGON, XYXYXYXY - normalize all x,y coordinate pairs
                for (auto& coords : points_)
                {
                    for (size_t i = 0; i < coords.size(); i += 2)
                    {
                        if (i < coords.size()) coords[i] *= invW;          // x
                        if (i + 1 < coords.size()) coords[i + 1] *= invH;  // y
                    }
                }
                break;
            }

            _isNormalized = true;
        }

        void Annotation::denormalize(int imageWidth, int imageHeight)
        {
            // Make denormalize idempotent - if already denormalized, skip
            if (!_isNormalized) {
                return;
            }

            if (imageWidth <= 0 || imageHeight <= 0) {
                throw std::invalid_argument("Image dimensions must be positive");
            }

            float w = static_cast<float>(imageWidth);
            float h = static_cast<float>(imageHeight);

            for (auto& coords : points_)
            {
                // Handle different label types differently
                switch (labelType_)
                {
                case LabelType::XYWH:
                    // [cx, cy, w, h] - denormalize all values
                    if (coords.size() >= 4)
                    {
                        coords[0] *= w;  // cx
                        coords[1] *= h;  // cy
                        coords[2] *= w;  // w
                        coords[3] *= h;  // h
                    }
                    break;

                case LabelType::XYWHR:
                    // [cx, cy, w, h, r] - denormalize cx, cy, w, h, but NOT rotation
                    if (coords.size() >= 4)
                    {
                        coords[0] *= w;  // cx
                        coords[1] *= h;  // cy
                        coords[2] *= w;  // w
                        coords[3] *= h;  // h
                        // coords[4] is rotation - don't denormalize
                    }
                    break;

                default:
                    // XYXY, POLYGON, XYXYXYXY - denormalize all x,y coordinate pairs
                    for (size_t i = 0; i < coords.size(); i += 2)
                    {
                        if (i < coords.size()) coords[i] *= w;          // x
                        if (i + 1 < coords.size()) coords[i + 1] *= h;  // y
                    }
                    break;
                }
            }

            _isNormalized = false;
        }

        void Annotation::scale(float scaleX, float scaleY)
        {
            for (auto& coords : points_)
            {
                // Handle different label types differently
                switch (labelType_)
                {
                case LabelType::XYWH:
                    // [cx, cy, w, h] - scale center and size
                    if (coords.size() >= 4)
                    {
                        coords[0] *= scaleX;  // cx
                        coords[1] *= scaleY;  // cy
                        coords[2] *= scaleX;  // w
                        coords[3] *= scaleY;  // h
                    }
                    break;

                case LabelType::XYWHR:
                    // [cx, cy, w, h, r] - convert to corners, scale, convert back
                    // For non-uniform scaling, rotation angle changes
                    if (coords.size() >= 5)
                    {
                        // Convert to 4 corners
                        std::vector<float> polygon = xywhrToXyxyxyxy(coords);

                        // Scale all corner coordinates
                        for (size_t i = 0; i < polygon.size(); i += 2)
                        {
                            polygon[i] *= scaleX;      // x
                            polygon[i + 1] *= scaleY;  // y
                        }

                        // Convert back to XYWHR (recalculates rotation)
                        coords = xyxyxyxyToXywhr(polygon);
                    }
                    break;

                default:
                    // XYXY, POLYGON, XYXYXYXY - scale all coordinate pairs
                    for (size_t i = 0; i < coords.size(); i += 2)
                    {
                        if (i < coords.size()) coords[i] *= scaleX;          // x
                        if (i + 1 < coords.size()) coords[i + 1] *= scaleY;  // y
                    }
                    break;
                }
            }
        }

        void Annotation::translate(float dx, float dy)
        {
            if (_isNormalized)
            {
                throw std::runtime_error("Cannot translate normalized annotations");
			}

            for (auto& coords : points_)
            {
                // Handle different label types differently
                switch (labelType_)
                {
                case LabelType::XYWH:
                    // [cx, cy, w, h] - translate center only, NOT size
                    if (coords.size() >= 2)
                    {
                        coords[0] += dx;  // cx
                        coords[1] += dy;  // cy
                        // coords[2] and coords[3] are w, h - don't translate
                    }
                    break;

                case LabelType::XYWHR:
                    // [cx, cy, w, h, r] - translate center only
                    if (coords.size() >= 2)
                    {
                        coords[0] += dx;  // cx
                        coords[1] += dy;  // cy
                        // coords[2], coords[3], coords[4] are w, h, r - don't translate
                    }
                    break;

                default:
                    // XYXY, POLYGON, XYXYXYXY - translate all coordinate pairs
                    for (size_t i = 0; i < coords.size(); i += 2)
                    {
                        if (i < coords.size()) coords[i] += dx;          // x
                        if (i + 1 < coords.size()) coords[i + 1] += dy;  // y
                    }
                    break;
                }
            }
        }

        void Annotation::flipHorizontal(int imageWidth)
        {
            if (imageWidth <= 0) {
                throw std::invalid_argument("Image width must be positive");
            }

            float w = static_cast<float>(imageWidth);

            if (_isNormalized)
            {
                w = 1;
			}

            for (auto& coords : points_)
            {
                // Handle different label types differently
                switch (labelType_)
                {
                case LabelType::XYWH:
                    // [cx, cy, w, h] - flip center x only
                    if (coords.size() >= 1)
                    {
                        coords[0] = w - coords[0];  // cx
                        // coords[1] is cy - don't flip
                        // coords[2], coords[3] are w, h - don't flip
                    }
                    break;

                case LabelType::XYWHR:
                    // [cx, cy, w, h, r] - flip center x and negate rotation
                    if (coords.size() >= 1)
                    {
                        coords[0] = w - coords[0];  // cx
                        // coords[1] is cy - don't flip
                        // coords[2], coords[3] are w, h - don't flip
                        if (coords.size() >= 5)
                        {
                            coords[4] = -coords[4];  // negate rotation angle
                        }
                    }
                    break;

                default:
                    // XYXY, POLYGON, XYXYXYXY - flip all x coordinates
                    for (size_t i = 0; i < coords.size(); i += 2)
                    {
                        if (i < coords.size()) {
                            coords[i] = w - coords[i];
                        }
                    }
                    break;
                }
            }
        }

        void Annotation::flipVertical(int imageHeight)
        {
            if (imageHeight <= 0) {
                throw std::invalid_argument("Image height must be positive");
            }

            float h = static_cast<float>(imageHeight);
            
            if (_isNormalized)
            {
                h = 1;
            }

            for (auto& coords : points_)
            {
                // Handle different label types differently
                switch (labelType_)
                {
                case LabelType::XYWH:
                    // [cx, cy, w, h] - flip center y only
                    if (coords.size() >= 2)
                    {
                        // coords[0] is cx - don't flip
                        coords[1] = h - coords[1];  // cy
                        // coords[2], coords[3] are w, h - don't flip
                    }
                    break;

                case LabelType::XYWHR:
                    // [cx, cy, w, h, r] - flip center y and negate rotation
                    if (coords.size() >= 2)
                    {
                        // coords[0] is cx - don't flip
                        coords[1] = h - coords[1];  // cy
                        // coords[2], coords[3] are w, h - don't flip
                        if (coords.size() >= 5)
                        {
                            coords[4] = -coords[4];  // negate rotation angle
                        }
                    }
                    break;

                default:
                    // XYXY, POLYGON, XYXYXYXY - flip all y coordinates
                    for (size_t i = 1; i < coords.size(); i += 2)
                    {
                        coords[i] = h - coords[i];
                    }
                    break;
                }
            }
        }

        void Annotation::validateAndClip(int imageWidth, int imageHeight, float minAreaPixels)
        {
            if (_isNormalized)
            {
                imageWidth = 1;
				imageHeight = 1;
            }

            clipAndFilterAnnotations(imageWidth, imageHeight, minAreaPixels);
        }

        void Annotation::clipToBounds(float x1, float y1, float x2, float y2, float minAreaPixels)
        {
            // 1. Translate to move (x1, y1) to origin
            translate(-x1, -y1);

            // 2. Call validateAndClip with the bounds as image size
            int width = static_cast<int>(x2 - x1);
            int height = static_cast<int>(y2 - y1);
            validateAndClip(width, height, minAreaPixels);

            // 3. Translate back to original position
            translate(x1, y1);
        }

        // ========== Accessors ==========

        LabelType Annotation::getLabelType() const
        {
            return labelType_;
        }

        void Annotation::setLabelType(LabelType type)
        {
            labelType_ = type;
        }

        const std::vector<int>& Annotation::getClasses() const
        {
            return classes_;
        }

        std::vector<int>& Annotation::getClasses()
        {
            return classes_;
        }

        const std::vector<std::vector<float>>& Annotation::getPoints() const
        {
            return points_;
        }

        std::vector<std::vector<float>>& Annotation::getPoints()
        {
            return points_;
        }

        // ========== Type Conversion ==========

        void Annotation::convertTo(LabelType targetType)
        {
            if (labelType_ == targetType) {
                return;  // Already the target type
            }

            // Convert all points
            for (auto& point : points_) {
                point = convertPoint(point, labelType_, targetType);
            }

            labelType_ = targetType;
        }

        Annotation Annotation::getAs(LabelType targetType) const
        {
            Annotation converted(targetType);
            converted.classes_ = classes_;

            // Convert all points
            converted.points_.reserve(points_.size());
            for (const auto& point : points_) {
                converted.points_.push_back(convertPoint(point, labelType_, targetType));
            }

            return converted;
        }

        // ========== Type Conversion Helpers ==========

        std::vector<float> Annotation::convertPoint(const std::vector<float>& point, LabelType from, LabelType to)
        {
            if (from == to) {
                return point;  // No conversion needed
            }

            // XYXY conversions
            if (from == LabelType::XYXY && to == LabelType::POLYGON) return xyxyToPolygon(point);
            if (from == LabelType::XYXY && to == LabelType::XYWH) return xyxyToXywh(point);

            // POLYGON conversions
            if (from == LabelType::POLYGON && to == LabelType::XYXY) return polygonToXyxy(point);
            if (from == LabelType::POLYGON && to == LabelType::XYWH) return polygonToXywh(point);
            if (from == LabelType::POLYGON && to == LabelType::XYWHR) return polygonToXywhr(point);
            if (from == LabelType::POLYGON && to == LabelType::XYXYXYXY) return polygonToXyxyxyxy(point);

            // XYWH conversions
            if (from == LabelType::XYWH && to == LabelType::XYXY) return xywhToXyxy(point);
            if (from == LabelType::XYWH && to == LabelType::POLYGON) return xywhToPolygon(point);

            // XYXYXYXY conversions
            if (from == LabelType::XYXYXYXY && to == LabelType::XYWHR) return xyxyxyxyToXywhr(point);
            if (from == LabelType::XYXYXYXY && to == LabelType::POLYGON) return xyxyxyxyToPolygon(point);

            // XYWHR conversions
            if (from == LabelType::XYWHR && to == LabelType::XYXYXYXY) return xywhrToXyxyxyxy(point);
            if (from == LabelType::XYWHR && to == LabelType::POLYGON) return xywhrToPolygon(point);

            throw std::invalid_argument(
                "Conversion from " + std::to_string(static_cast<int>(from)) +
                " to " + std::to_string(static_cast<int>(to)) + " is not supported"
            );
        }

        // ========== Bbox Format Conversions ==========

        std::vector<float> Annotation::xyxyToXywh(const std::vector<float>& xyxy)
        {
            if (xyxy.size() != 4) {
                throw std::invalid_argument("XYXY must have 4 values");
            }
            float x1 = xyxy[0], y1 = xyxy[1], x2 = xyxy[2], y2 = xyxy[3];
            float w = x2 - x1, h = y2 - y1;
            return {x1 + w / 2.0f, y1 + h / 2.0f, w, h};
        }

        std::vector<float> Annotation::xywhToXyxy(const std::vector<float>& xywh)
        {
            if (xywh.size() != 4) {
                throw std::invalid_argument("XYWH must have 4 values");
            }
            float cx = xywh[0], cy = xywh[1], w = xywh[2], h = xywh[3];
            float halfW = w / 2.0f, halfH = h / 2.0f;
            return {cx - halfW, cy - halfH, cx + halfW, cy + halfH};
        }

        // ========== Polygon Conversions ==========

        std::vector<float> Annotation::polygonToXyxy(const std::vector<float>& polygon)
        {
            if (polygon.size() < 2) {
                throw std::invalid_argument("Polygon must have at least 1 point (2 values)");
            }

            float minX = std::numeric_limits<float>::max();
            float minY = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::lowest();
            float maxY = std::numeric_limits<float>::lowest();

            for (size_t i = 0; i < polygon.size(); i += 2) {
                float x = polygon[i];
                float y = polygon[i + 1];
                minX = std::min(minX, x);
                minY = std::min(minY, y);
                maxX = std::max(maxX, x);
                maxY = std::max(maxY, y);
            }

            return {minX, minY, maxX, maxY};
        }

        std::vector<float> Annotation::polygonToXywh(const std::vector<float>& polygon)
        {
            auto xyxy = polygonToXyxy(polygon);
            return xyxyToXywh(xyxy);
        }

        std::vector<float> Annotation::xyxyToPolygon(const std::vector<float>& xyxy)
        {
            if (xyxy.size() != 4) {
                throw std::invalid_argument("XYXY must have 4 values");
            }
            // Convert to 4 corners (clockwise from top-left)
            float x1 = xyxy[0], y1 = xyxy[1], x2 = xyxy[2], y2 = xyxy[3];
            return {x1, y1, x2, y1, x2, y2, x1, y2};
        }

        std::vector<float> Annotation::xywhToPolygon(const std::vector<float>& xywh)
        {
            // Convert to 4 corners
            auto xyxy = xywhToXyxy(xywh);
            float x1 = xyxy[0], y1 = xyxy[1], x2 = xyxy[2], y2 = xyxy[3];
            return {x1, y1, x2, y1, x2, y2, x1, y2};  // 4 corners as polygon
        }

        // ========== Rotated Bbox Conversions ==========

        std::vector<float> Annotation::xyxyxyxyToXywhr(const std::vector<float>& xyxyxyxy)
        {
            if (xyxyxyxy.size() != 8)
            {
                throw std::invalid_argument("XYXYXYXY must have 8 values");
            }

            // Calculate center
            float cx = (xyxyxyxy[0] + xyxyxyxy[2] + xyxyxyxy[4] + xyxyxyxy[6]) / 4.0f;
            float cy = (xyxyxyxy[1] + xyxyxyxy[3] + xyxyxyxy[5] + xyxyxyxy[7]) / 4.0f;

            // Calculate width (distance between first two points)
            float dx1 = xyxyxyxy[2] - xyxyxyxy[0];
            float dy1 = xyxyxyxy[3] - xyxyxyxy[1];
            float w = std::sqrt(dx1 * dx1 + dy1 * dy1);

            // Calculate height (distance between second and third points)
            float dx2 = xyxyxyxy[4] - xyxyxyxy[2];
            float dy2 = xyxyxyxy[5] - xyxyxyxy[3];
            float h = std::sqrt(dx2 * dx2 + dy2 * dy2);

            // Calculate rotation angle (from first edge)
            float angle = std::atan2(dy1, dx1);

            return {cx, cy, w, h, angle};
        }

        std::vector<float> Annotation::xywhrToXyxyxyxy(const std::vector<float>& xywhr)
        {
            if (xywhr.size() != 5) {
                throw std::invalid_argument("XYWHR must have 5 values");
            }

            float cx = xywhr[0], cy = xywhr[1], w = xywhr[2], h = xywhr[3], angle = xywhr[4];
            float halfW = w / 2.0f, halfH = h / 2.0f;
            float cosA = std::cos(angle), sinA = std::sin(angle);

            // Calculate 4 corners of rotated rectangle
            auto rotate = [&](float x, float y) -> std::pair<float, float> {
                return {
                    cx + (x * cosA - y * sinA),
                    cy + (x * sinA + y * cosA)
                };
            };

            auto [x1, y1] = rotate(-halfW, -halfH);
            auto [x2, y2] = rotate(halfW, -halfH);
            auto [x3, y3] = rotate(halfW, halfH);
            auto [x4, y4] = rotate(-halfW, halfH);

            return {x1, y1, x2, y2, x3, y3, x4, y4};
        }

        std::vector<float> Annotation::xyxyxyxyToPolygon(const std::vector<float>& xyxyxyxy)
        {
            if (xyxyxyxy.size() != 8) {
                throw std::invalid_argument("XYXYXYXY must have 8 values");
            }
            // Just type change
            return xyxyxyxy;
        }

        std::vector<float> Annotation::xywhrToPolygon(const std::vector<float>& xywhr)
        {
            return xywhrToXyxyxyxy(xywhr);  // Same as 4 corners
        }

        std::vector<float> Annotation::polygonToXywhr(const std::vector<float>& polygon)
        {
            if (polygon.size() < 6) {
                throw std::invalid_argument("Polygon must have at least 3 points (6 values)");
            }

            // Use OpenCV's minAreaRect for O(n) optimal solution
            std::vector<cv::Point2f> points;
            points.reserve(polygon.size() / 2);

            for (size_t i = 0; i < polygon.size(); i += 2) {
                points.push_back(cv::Point2f(polygon[i], polygon[i + 1]));
            }

            // Get minimum area rotated rectangle
            cv::RotatedRect rect = cv::minAreaRect(points);

            // Convert angle from degrees to radians
            float angle = rect.angle * 3.14159265358979323846f / 180.0f;

            // OpenCV's RotatedRect angle is in [-90, 0), adjust if needed
            // and swap width/height if rotated by 90 degrees
            float w = rect.size.width;
            float h = rect.size.height;

            // Ensure consistent orientation (width >= height)
            if (w < h) {
                std::swap(w, h);
                angle += M_PI_F / 2.0f;  // Add 90 degrees
            }

            return {rect.center.x, rect.center.y, w, h, angle};
        }

        std::vector<float> Annotation::polygonToXyxyxyxy(const std::vector<float>& polygon)
        {
            if (polygon.size() < 6) {
                throw std::invalid_argument("Polygon must have at least 3 points (6 values)");
            }

            // Get minimum rotated bbox parameters
            auto xywhr = polygonToXywhr(polygon);

            // Convert to 4 corners
            return xywhrToXyxyxyxy(xywhr);
        }

        // ========== Advanced Transformations ==========

        void Annotation::transform(cv::Mat& image,
                                   const cv::Mat& transformMat,
                                   const cv::Size& dsize,
                                   const cv::Scalar& borderValue,
                                   float minAreaPixels)
        {
            if (image.empty()) {
                throw std::invalid_argument("Input image is empty");
            }

            if (transformMat.rows != 3 || transformMat.cols != 3) {
                throw std::invalid_argument("Transform matrix must be 3x3");
            }

            if (transformMat.type() != CV_64F && transformMat.type() != CV_32F) {
                throw std::invalid_argument("Transform matrix must be CV_64F or CV_32F");
            }

            // Convert to CV_64F if needed
            cv::Mat mat64;
            if (transformMat.type() == CV_32F) {
                transformMat.convertTo(mat64, CV_64F);
            } else {
                mat64 = transformMat;
            }

            // Determine output size
            cv::Size outputSize = dsize;
            if (outputSize.width == 0 || outputSize.height == 0)
            {
                // Calculate bounding box of transformed image corners
                std::vector<cv::Point2f> corners = {
                    {0, 0},
                    {static_cast<float>(image.cols), 0},
                    {static_cast<float>(image.cols), static_cast<float>(image.rows)},
                    {0, static_cast<float>(image.rows)}
                };

                float minX = std::numeric_limits<float>::max();
                float minY = std::numeric_limits<float>::max();
                float maxX = std::numeric_limits<float>::lowest();
                float maxY = std::numeric_limits<float>::lowest();

                for (const auto& corner : corners) {
                    float x = static_cast<float>(
                        mat64.at<double>(0, 0) * corner.x +
                        mat64.at<double>(0, 1) * corner.y +
                        mat64.at<double>(0, 2)
                    );
                    float y = static_cast<float>(
                        mat64.at<double>(1, 0) * corner.x +
                        mat64.at<double>(1, 1) * corner.y +
                        mat64.at<double>(1, 2)
                    );
                    float w = static_cast<float>(
                        mat64.at<double>(2, 0) * corner.x +
                        mat64.at<double>(2, 1) * corner.y +
                        mat64.at<double>(2, 2)
                    );

                    // Use epsilon-based check to avoid division by near-zero values
                    // Use larger threshold (1e-3) to prevent overflow from very small denominators
                    constexpr float MIN_W_THRESHOLD = 1e-3f;
                    if (std::abs(w) > MIN_W_THRESHOLD) {
                        x /= w;
                        y /= w;
                    }
                    else {
                        // Skip this corner if w is too close to zero (invalid transformation)
                        continue;
                    }

                    minX = std::min(minX, x);
                    minY = std::min(minY, y);
                    maxX = std::max(maxX, x);
                    maxY = std::max(maxY, y);
                }

                outputSize = cv::Size(
                    static_cast<int>(std::ceil(maxX - minX)),
                    static_cast<int>(std::ceil(maxY - minY))
                );

                // Adjust transform matrix to account for translation
                cv::Mat translationMat = (cv::Mat_<double>(3, 3) <<
                    1, 0, -minX,
                    0, 1, -minY,
                    0, 0, 1
                );
                mat64 = translationMat * mat64;
            }

            // Apply perspective transformation to image
            // Use temporary Mat for exception safety (protect original image)
            cv::Mat transformedImage;
            cv::warpPerspective(image, transformedImage, mat64, outputSize,
                               cv::INTER_LINEAR, cv::BORDER_CONSTANT, borderValue);

            // Only swap if transformation succeeded
            std::swap(image, transformedImage);

            // Transform annotation coordinates based on label type
            for (auto& point : points_)
            {
                // For XYWH/XYWHR formats, convert to polygon first, transform, then convert back
                if (labelType_ == LabelType::XYWH || labelType_ == LabelType::XYWHR)
                {
                    // Convert to polygon (4 corners)
                    std::vector<float> polygon;
                    if (labelType_ == LabelType::XYWH) {
                        polygon = xywhToPolygon(point);
                    } else {
                        polygon = xywhrToPolygon(point);
                    }

                    // Transform all corner points using perspective matrix (same as image)
                    for (size_t i = 0; i < polygon.size(); i += 2) {
                        float x = polygon[i];
                        float y = polygon[i + 1];

                        float newX = static_cast<float>(
                            mat64.at<double>(0, 0) * x +
                            mat64.at<double>(0, 1) * y +
                            mat64.at<double>(0, 2)
                        );
                        float newY = static_cast<float>(
                            mat64.at<double>(1, 0) * x +
                            mat64.at<double>(1, 1) * y +
                            mat64.at<double>(1, 2)
                        );
                        float w = static_cast<float>(
                            mat64.at<double>(2, 0) * x +
                            mat64.at<double>(2, 1) * y +
                            mat64.at<double>(2, 2)
                        );

                        // Use epsilon-based check to avoid division by near-zero values
                        if (std::abs(w) > FLOAT_EPSILON) {
                            newX /= w;
                            newY /= w;
                        }

                        polygon[i] = newX;
                        polygon[i + 1] = newY;
                    }

                    // Convert back to original format
                    if (labelType_ == LabelType::XYWH) {
                        point = polygonToXywh(polygon);
                    } else {
                        point = polygonToXywhr(polygon);
                    }
                }
                else
                {
                    // For XYXY, POLYGON, XYXYXYXY - transform coordinate pairs using perspective matrix
                    for (size_t i = 0; i < point.size(); i += 2) {
                        float x = point[i];
                        float y = point[i + 1];

                        float newX = static_cast<float>(
                            mat64.at<double>(0, 0) * x +
                            mat64.at<double>(0, 1) * y +
                            mat64.at<double>(0, 2)
                        );
                        float newY = static_cast<float>(
                            mat64.at<double>(1, 0) * x +
                            mat64.at<double>(1, 1) * y +
                            mat64.at<double>(1, 2)
                        );
                        float w = static_cast<float>(
                            mat64.at<double>(2, 0) * x +
                            mat64.at<double>(2, 1) * y +
                            mat64.at<double>(2, 2)
                        );

                        // Use epsilon-based check to avoid division by near-zero values
                        if (std::abs(w) > FLOAT_EPSILON) {
                            newX /= w;
                            newY /= w;
                        }

                        point[i] = newX;
                        point[i + 1] = newY;
                    }
                }
            }

            // Clip and filter annotations
            validateAndClip(outputSize.width, outputSize.height, minAreaPixels);
        }

        // ========== Private Helper Functions ==========

        void Annotation::clipAndFilterAnnotations(int imageWidth, int imageHeight, float minAreaPixels)
        {
            size_t writeIdx = 0;

            for (size_t readIdx = 0; readIdx < classes_.size(); ++readIdx)
            {
                auto& point = points_[readIdx];
                bool hasValidPoints = false;

                // Handle classification annotations (no spatial coordinates)
                if (labelType_ == LabelType::NONE)
                {
                    // Classification labels have no spatial coordinates - always keep them
                    hasValidPoints = true;
                }
                // Clip coordinates based on label type
                else if (labelType_ == LabelType::XYWH)
                {
                    // [cx, cy, w, h] - clip entire bounding box to image boundaries
                    if (point.size() >= 4)
                    {
                        // Convert to XYXY for clipping
                        float cx = point[0];
                        float cy = point[1];
                        float w = point[2];
                        float h = point[3];

                        float halfW = w / 2.0f;
                        float halfH = h / 2.0f;

                        float x1 = cx - halfW;
                        float y1 = cy - halfH;
                        float x2 = cx + halfW;
                        float y2 = cy + halfH;

                        // Clip to image boundaries using std::clamp for efficiency
                        x1 = std::clamp(x1, 0.0f, static_cast<float>(imageWidth));
                        y1 = std::clamp(y1, 0.0f, static_cast<float>(imageHeight));
                        x2 = std::clamp(x2, 0.0f, static_cast<float>(imageWidth));
                        y2 = std::clamp(y2, 0.0f, static_cast<float>(imageHeight));

                        // Check if clipped box is valid
                        if (x2 > x1 && y2 > y1)
                        {
                            hasValidPoints = true;

                            // Convert back to XYWH
                            float newW = x2 - x1;
                            float newH = y2 - y1;
                            point[0] = x1 + newW / 2.0f;  // new cx
                            point[1] = y1 + newH / 2.0f;  // new cy
                            point[2] = newW;               // new w
                            point[3] = newH;               // new h
                        }
                    }
                }
                else if (labelType_ == LabelType::XYWHR)
                {
                    // [cx, cy, w, h, r] - if center is inside image, don't clip
                    if (point.size() >= 2)
                    {
                        float cx = point[0];
                        float cy = point[1];

                        // Check if center is inside image
                        if (cx > 0 && cx < imageWidth && cy > 0 && cy < imageHeight)
                        {
                            hasValidPoints = true;
                            // Don't clip - keep all values as is
                        }
                        // If center is outside, mark as invalid (will be filtered out)
                    }
                }
                else if (labelType_ == LabelType::XYXYXYXY)
                {
                    // [x1, y1, x2, y2, x3, y3, x4, y4] - if center is inside image, don't clip
                    if (point.size() >= 8)
                    {
                        // Calculate center point
                        float cx = (point[0] + point[2] + point[4] + point[6]) / 4.0f;
                        float cy = (point[1] + point[3] + point[5] + point[7]) / 4.0f;

                        // Check if center is inside image
                        if (cx > 0 && cx < imageWidth && cy > 0 && cy < imageHeight)
                        {
                            hasValidPoints = true;
                            // Don't clip - keep all corner coordinates as is
                        }
                        else
                        {
                            // Center is outside - clip all coordinates
                            for (size_t j = 0; j < point.size(); j += 2)
                            {
                                float& x = point[j];
                                float& y = point[j + 1];

                                x = std::max(0.0f, std::min(x, static_cast<float>(imageWidth)));
                                y = std::max(0.0f, std::min(y, static_cast<float>(imageHeight)));

                                if (x > 0 && x < imageWidth && y > 0 && y < imageHeight)
                                {
                                    hasValidPoints = true;
                                }
                            }
                        }

                        // Normalize polygon order: clockwise, starting from point with smallest y
                        if (hasValidPoints)
                        {
                            normalizePolygonOrder(point);
                        }
                    }
                }
                else
                {
                    // XYXY, POLYGON - clip all coordinate pairs
                    for (size_t j = 0; j < point.size(); j += 2)
                    {
                        float& x = point[j];
                        float& y = point[j + 1];

                        // Clip to [0, imageWidth] x [0, imageHeight]
                        x = std::max(0.0f, std::min(x, static_cast<float>(imageWidth)));
                        y = std::max(0.0f, std::min(y, static_cast<float>(imageHeight)));

                        // Check if at least one point is inside
                        if (x > 0 && x < imageWidth && y > 0 && y < imageHeight)
                        {
                            hasValidPoints = true;
                        }
                    }
                }

                // Calculate area and check validity
                float area = calculateArea(point);
                bool isValid = false;

                if (area >= 0) {
                    // It's a bounding box - check area threshold
                    isValid = (area >= minAreaPixels) && hasValidPoints;
                } else {
                    // Not a bounding box (polygon, etc.) - just check if has valid points
                    isValid = hasValidPoints;
                }

                // Keep valid objects by moving them forward
                if (isValid) {
                    if (writeIdx != readIdx) {
                        classes_[writeIdx] = classes_[readIdx];
                        points_[writeIdx] = std::move(points_[readIdx]);
                    }
                    ++writeIdx;
                }
            }

            // Truncate to valid size
            classes_.resize(writeIdx);
            points_.resize(writeIdx);
        }

        float Annotation::calculateArea(const std::vector<float>& point) const
        {
            switch (labelType_)
            {
            case LabelType::XYWH:
            {
                if (point.size() < 4) return -1.0f;
                // Center-based: x, y, w, h
                float w = point[2];
                float h = point[3];
                return w * h;
            }

            case LabelType::XYXY:
            {
                if (point.size() < 4) return -1.0f;
                // Corner-based: x1, y1, x2, y2
                float x1 = point[0];
                float y1 = point[1];
                float x2 = point[2];
                float y2 = point[3];
                float w = x2 - x1;
                float h = y2 - y1;
                return (w > 0 && h > 0) ? (w * h) : 0.0f;
            }

            case LabelType::XYWHR:
            {
                if (point.size() < 5) return -1.0f;
                // Rotated bounding box: cx, cy, w, h, rotation
                // Area is still w * h regardless of rotation
                float w = point[2];
                float h = point[3];
                return w * h;
            }

            case LabelType::XYXYXYXY:
            {
                if (point.size() < 8) return -1.0f;
                // Quadrilateral - use shoelace formula with Kahan summation for numerical stability
                float x1 = point[0], y1 = point[1];
                float x2 = point[2], y2 = point[3];
                float x3 = point[4], y3 = point[5];
                float x4 = point[6], y4 = point[7];

                // Kahan summation to reduce floating-point errors
                float sum = 0.0f;
                float c = 0.0f;  // Running compensation for lost low-order bits

                auto kahanAdd = [&sum, &c](float value) {
                    float y = value - c;
                    float t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                };

                kahanAdd(x1*y2 - x2*y1);
                kahanAdd(x2*y3 - x3*y2);
                kahanAdd(x3*y4 - x4*y3);
                kahanAdd(x4*y1 - x1*y4);

                float area = 0.5f * std::abs(sum);
                return area;
            }

            case LabelType::POLYGON:
            {
                if (point.size() < 6) return -1.0f;  // Need at least 3 points (triangle)

                // Use Shoelace formula for arbitrary polygon with Kahan summation
                // Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
                // Kahan summation reduces accumulated floating-point errors for large polygons
                float sum = 0.0f;
                float c = 0.0f;  // Running compensation for lost low-order bits
                size_t n = point.size() / 2;  // Number of vertices

                for (size_t i = 0; i < n; ++i)
                {
                    size_t j = (i + 1) % n;  // Next vertex (wraps around)
                    float x_i = point[i * 2];
                    float y_i = point[i * 2 + 1];
                    float x_j = point[j * 2];
                    float y_j = point[j * 2 + 1];

                    float term = (x_i * y_j - x_j * y_i);

                    // Kahan summation
                    float y = term - c;
                    float t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }

                return 0.5f * std::abs(sum);
            }

            default:
                return -1.0f;  // Unknown label type
            }
        }

        // Helper function to normalize XYXYXYXY polygon
        // Ensures clockwise order starting from the point with smallest y
        void Annotation::normalizePolygonOrder(std::vector<float>& point)
        {
            if (point.size() != 8) return;

            // Extract 4 points
            struct Point2D {
                float x, y;
                int originalIndex;
            };

            std::vector<Point2D> points = {
                {point[0], point[1], 0},
                {point[2], point[3], 1},
                {point[4], point[5], 2},
                {point[6], point[7], 3}
            };

            // Find point with smallest y (if tie, smallest x)
            int minIdx = 0;
            for (int i = 1; i < 4; ++i) {
                if (points[i].y < points[minIdx].y ||
                    (std::abs(points[i].y - points[minIdx].y) < FLOAT_EPSILON && points[i].x < points[minIdx].x)) {
                    minIdx = i;
                }
            }

            // Calculate centroid
            float cx = (points[0].x + points[1].x + points[2].x + points[3].x) / 4.0f;
            float cy = (points[0].y + points[1].y + points[2].y + points[3].y) / 4.0f;

            // Sort 4 points by angle from centroid (clockwise)
            // Use manual sorting for 4 elements (faster than std::sort)
            auto compareAngle = [cx, cy](const Point2D& a, const Point2D& b) {
                float angleA = std::atan2(a.y - cy, a.x - cx);
                float angleB = std::atan2(b.y - cy, b.x - cx);
                return angleA > angleB;  // Descending for clockwise
            };

            // Sorting network for 4 elements (5 comparisons)
            if (compareAngle(points[1], points[0])) std::swap(points[0], points[1]);
            if (compareAngle(points[3], points[2])) std::swap(points[2], points[3]);
            if (compareAngle(points[2], points[0])) std::swap(points[0], points[2]);
            if (compareAngle(points[3], points[1])) std::swap(points[1], points[3]);
            if (compareAngle(points[2], points[1])) std::swap(points[1], points[2]);

            // Rotate to start from minIdx
            int startPos = 0;
            for (int i = 0; i < 4; ++i) {
                if (points[i].originalIndex == minIdx) {
                    startPos = i;
                    break;
                }
            }

            // Reconstruct point array in correct order
            std::vector<float> temp(8);
            for (int i = 0; i < 4; ++i) {
                int idx = (startPos + i) % 4;
                temp[i * 2] = points[idx].x;
                temp[i * 2 + 1] = points[idx].y;
            }

            point = temp;
        }

    } // namespace Data
} // namespace WheelDL
