#pragma once
#include <vector>
#include <cstddef>
#include <opencv2/opencv.hpp>

namespace WheelDL
{
    namespace Data
    {
        // Forward declarations
        namespace Cache { class CacheManager; }

        enum class LabelType
        {
            NONE = 0,
            XYWH = 1,       // x, y, width, height (center-based bbox)
            XYXY = 2,       // x1, y1, x2, y2 (corner-based bbox)
            XYWHR = 3,      // x, y, w, h, rotation (oriented bbox)
            XYXYXYXY = 4,   // x1, y1, x2, y2, x3, y3, x4, y4 (rotated bbox as 4 corners)
            POLYGON = 5     // variable length polygon points
        };

        /**
         * @class Annotation
         * @brief Container for image annotations with type-aware operations
         *
         * Stores object annotations (bounding boxes, polygons, etc.) with their class IDs.
         * All objects in one Annotation instance share the same LabelType.
         */
        class Annotation
        {
        public:
            /**
             * @brief Construct annotation with specified label type
             * @param type Label type for all objects (default: NONE)
             */
            explicit Annotation(LabelType type = LabelType::NONE, bool isNormalized=false);

            /**
             * @brief Create annotation for classification task
             * @param classId Classification label
             * @return Annotation with only class label, no spatial information
             */
            static Annotation forClassification(int classId);

            // ========== Object Management ==========

            /**
             * @brief Add an annotation object
             * @param classId Class ID for this object
             * @param points Coordinate points (format depends on labelType)
             */
            void addObject(int classId, const std::vector<float>& points);

            /**
             * @brief Get number of annotation objects
             */
            size_t size() const;

            /**
             * @brief Check if annotations are empty
             */
            bool empty() const;

            /**
             * @brief Clear all annotations
             */
            void clear();

            // ========== Coordinate Transformations ==========

            /**
             * @brief Normalize all coordinates to [0, 1] range
             * @param imageWidth Image width in pixels
             * @param imageHeight Image height in pixels
             */
            void normalize(int imageWidth, int imageHeight);

            /**
             * @brief Denormalize coordinates from [0, 1] to pixel coordinates
             * @param imageWidth Image width in pixels
             * @param imageHeight Image height in pixels
             */
            void denormalize(int imageWidth, int imageHeight);

            /**
             * @brief Scale all coordinates
             * @param scaleX Horizontal scale factor
             * @param scaleY Vertical scale factor
             */
            void scale(float scaleX, float scaleY);

            /**
             * @brief Translate all coordinates
             * @param dx Horizontal offset
             * @param dy Vertical offset
             */
            void translate(float dx, float dy);

            /**
             * @brief Flip coordinates horizontally
             * @param imageWidth Image width for flip calculation
             */
            void flipHorizontal(int imageWidth);

            /**
             * @brief Flip coordinates vertically
             * @param imageHeight Image height for flip calculation
             */
            void flipVertical(int imageHeight);

            /**
             * @brief Validate and clip annotations to image boundaries
             *
             * Clips all annotation coordinates to image boundaries and removes objects that:
             * - Have no valid points inside image boundaries
             * - Have area smaller than minAreaPixels (for bounding boxes)
             *
             * @param imageWidth Image width in pixels
             * @param imageHeight Image height in pixels
             * @param minAreaPixels Minimum area in pixels for object to be kept (default: 5.0)
             */
            void validateAndClip(int imageWidth, int imageHeight, float minAreaPixels = 5.0f);

            /**
             * @brief Clip annotations to specified bounds
             *
             * Clips all annotation coordinates to the specified rectangular bounds and removes objects that:
             * - Have no valid points inside the bounds
             * - Have area smaller than minAreaPixels (for bounding boxes)
             *
             * This is useful for Mosaic augmentation where images are placed in specific regions.
             *
             * @param x1 Left boundary
             * @param y1 Top boundary
             * @param x2 Right boundary
             * @param y2 Bottom boundary
             * @param minAreaPixels Minimum area in pixels for object to be kept (default: 5.0)
             */
            void clipToBounds(float x1, float y1, float x2, float y2, float minAreaPixels = 5.0f);

            // ========== Accessors ==========

            /**
             * @brief Get label type
             */
            LabelType getLabelType() const;

            /**
             * @brief Set label type
             */
            void setLabelType(LabelType type);

            /**
             * @brief Get class IDs (const)
             */
            const std::vector<int>& getClasses() const;

            /**
             * @brief Get class IDs (mutable)
             */
            std::vector<int>& getClasses();

            /**
             * @brief Get coordinate points (const)
             */
            const std::vector<std::vector<float>>& getPoints() const;

            /**
             * @brief Get coordinate points (mutable)
             */
            std::vector<std::vector<float>>& getPoints();

            /**
             * @brief Create a deep copy of this annotation
             * @return A new Annotation object with copied data
             *
             * Creates an explicit deep copy of the annotation. This is useful when
             * transforms need to modify annotations without affecting the original.
             */
            Annotation clone() const;

            // ========== Type Conversion ==========

            /**
             * @brief Convert annotations to different label type (in-place)
             * @param targetType Target label type
             * @throws std::invalid_argument if conversion is not supported
             */
            void convertTo(LabelType targetType);

            /**
             * @brief Get points converted to target type (returns copy)
             * @param targetType Target label type
             * @return Converted copy of annotation
             * @throws std::invalid_argument if conversion is not supported
             */
            Annotation getAs(LabelType targetType) const;

            // ========== Advanced Transformations ==========

            /**
             * @brief Apply 3x3 transformation matrix to image and annotations
             *
             * Applies the transformation matrix to both the image and annotation coordinates,
             * then clips annotations to image boundaries and filters out invalid objects.
             *
             * @param image Input/output image (modified in-place)
             * @param transformMat 3x3 transformation matrix (cv::Mat of type CV_64F or CV_32F)
             * @param dsize Output image size (if empty, calculated automatically)
             * @param borderValue Border fill value for areas outside original image
             * @param minAreaPixels Minimum area in pixels for object to be kept (default: 1)
             * @throws std::invalid_argument if matrix is invalid
             */
            void transform(cv::Mat& image,
                          const cv::Mat& transformMat,
                          const cv::Size& dsize = cv::Size(),
                          const cv::Scalar& borderValue = cv::Scalar(114, 114, 114),
                          float minAreaPixels = 1.0f);

        private:
            LabelType labelType_;
            std::vector<int> classes_;
            std::vector<std::vector<float>> points_;
            bool _isNormalized = false;
            
            // Type conversion helpers
            static std::vector<float> convertPoint(const std::vector<float>& point, LabelType from, LabelType to);

            // Bbox format conversions
            static std::vector<float> xyxyToXywh(const std::vector<float>& xyxy);
            static std::vector<float> xywhToXyxy(const std::vector<float>& xywh);

            // Polygon conversions
            static std::vector<float> polygonToXyxy(const std::vector<float>& polygon);
            static std::vector<float> polygonToXywh(const std::vector<float>& polygon);
            static std::vector<float> xyxyToPolygon(const std::vector<float>& xyxy);
            static std::vector<float> xywhToPolygon(const std::vector<float>& xywh);

            // Rotated bbox conversions
            static std::vector<float> xyxyxyxyToXywhr(const std::vector<float>& xyxyxyxy);
            static std::vector<float> xywhrToXyxyxyxy(const std::vector<float>& xywhr);
            static std::vector<float> xyxyxyxyToPolygon(const std::vector<float>& xyxyxyxy);
            static std::vector<float> xywhrToPolygon(const std::vector<float>& xywhr);
            static std::vector<float> polygonToXywhr(const std::vector<float>& polygon);
            static std::vector<float> polygonToXyxyxyxy(const std::vector<float>& polygon);

            /**
             * @brief Clip annotations to image boundaries and filter invalid objects
             *
             * Removes objects that:
             * - Have no valid points inside image boundaries
             * - Have area smaller than minAreaPixels (for bounding boxes)
             *
             * @param imageWidth Image width
             * @param imageHeight Image height
             * @param minAreaPixels Minimum area in pixels for object to be kept
             */
            void clipAndFilterAnnotations(int imageWidth, int imageHeight, float minAreaPixels);

            /**
             * @brief Calculate area of a bounding box
             * @param point Bounding box coordinates (format depends on labelType)
             * @return Area in pixels, or -1 if not a bounding box
             */
            float calculateArea(const std::vector<float>& point) const;

            /**
             * @brief Normalize XYXYXYXY polygon order
             *
             * Ensures polygon vertices are:
             * 1. In clockwise order
             * 2. Starting from the point with smallest y (tie-break by smallest x)
             *
             * @param point XYXYXYXY coordinates (8 values: x1,y1,x2,y2,x3,y3,x4,y4)
             */
            void normalizePolygonOrder(std::vector<float>& point);

            // CacheManager needs direct access for efficient serialization
            friend class Cache::CacheManager;
        };

    } // namespace Data
} // namespace WheelDL