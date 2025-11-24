#include "pch.h"
#include "SegmentationDataset.h"
#include "Data/Transforms/GeometricTransforms.h"
#include "Data/Transforms/ColorTransforms.h"
#include "Config/JsonParser.h"
#include "Utils/Error/WheelLibException.h"
#include <filesystem>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>

namespace WheelDL
{
    namespace Data
    {
        namespace Dataset
        {
            SegmentationDataset::SegmentationDataset(
                const Config::Configuration& config,
                bool train)
                : BaseDataset<SegmentationDataset>(config, train)
            {
                loadAnnotations();
                _transforms = buildTransforms();
            }

            void SegmentationDataset::loadAnnotations()
            {
                using namespace Config;
                using namespace WheelDL::Utils;

                // Validate paths
                if (!std::filesystem::exists(_annotationPath))
                {
                    throw DataException(
                        ErrorCode::DATA_FILE_NOT_FOUND,
                        "Annotation file does not exist: " + _annotationPath
                    );
                }

                if (!std::filesystem::exists(_dataPath))
                {
                    throw DataException(
                        ErrorCode::DATASET_NOT_FOUND,
                        "Data directory does not exist: " + _dataPath
                    );
                }

                // Parse JSON annotation file
                nlohmann::json annotationJson = JsonParser::parseFrom(_annotationPath);

                // Validate JSON structure
                if (!annotationJson.contains("annotations") || !annotationJson["annotations"].is_array())
                {
                    throw DataException(
                        ErrorCode::DATA_INVALID_FORMAT,
                        "Invalid annotation JSON: missing 'annotations' array"
                    );
                }

                // Get expected role: 0=train, 1=test
                int expectedRole = _train ? 0 : 1;

                // Process each annotation
                const auto& annotations = annotationJson["annotations"];
                for (const auto& annot : annotations)
                {
                    // Check role filter
                    int role = JsonParser::getInt(annot, "role", -1);
                    if (role != expectedRole)
                    {
                        continue;  // Skip annotations not matching current dataset role
                    }

                    // Get filename
                    std::string filename = JsonParser::getString(annot, "filename", "");
                    if (filename.empty())
                    {
                        std::cerr << "Warning: Annotation missing filename, skipping" << std::endl;
                        continue;
                    }

                    // Build full image path
                    std::filesystem::path imagePath = std::filesystem::path(_dataPath) / filename;
                    if (!std::filesystem::exists(imagePath))
                    {
                        std::cerr << "Warning: Image file not found: " << imagePath << std::endl;
                        continue;
                    }

                    int imgWidth, imgHeight;
                    try
                    {
                        auto [w, h] = WheelDL::Data::Utils::ImageIO::getImageDimensions(imagePath.string());
                        imgWidth = w;
                        imgHeight = h;
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << "Warning: Failed to get image dimensions for " << filename << ": " << e.what() << std::endl;
                        continue;
                    }

                    // Parse label array
                    if (!annot.contains("label") || !annot["label"].is_array())
                    {
                        std::cerr << "Warning: Annotation missing or invalid label for " << filename << std::endl;
                        continue;
                    }

                    Annotation annotation(LabelType::POLYGON, false);  // Polygon format, not normalized (pixel coordinates)

                    const auto& labelArray = annot["label"];
                    for (const auto& obj : labelArray)
                    {
                        if (!obj.is_array() || obj.size() < 3)
                        {
                            std::cerr << "Warning: Invalid object format in " << filename << std::endl;
                            continue;
                        }

                        // Parse: [class_id, x1, y1, x2, y2, ..., xn, yn]
                        int classId = obj[0].get<int>();

                        // Extract polygon coordinates
                        std::vector<float> polygon;
                        for (size_t i = 1; i < obj.size(); ++i)
                        {
                            polygon.push_back(obj[i].get<float>());
                        }

                        // Add polygon to annotation (pixel coordinates, not normalized)
                        if (!polygon.empty())
                        {
                            annotation.addObject(classId, polygon);
                        }
                    }

					annotation.normalize(imgWidth, imgHeight);

                    // Add to dataset
                    std::string imagePathStr = imagePath.string();
                    _imagePaths.push_back(imagePathStr);
                    _annotations[imagePathStr] = annotation;
                }

                if (_imagePaths.empty())
                {
                    throw DataException(
                        ErrorCode::DATA_LOAD_FAILED,
                        "No valid images with annotations found for role=" + std::to_string(expectedRole)
                    );
                }
            }

            std::shared_ptr<Transforms::Transform> SegmentationDataset::buildTransforms()
            {
                // Use base class helper to build standard transforms
                // includeColorAugmentation = true for segmentation task
                return buildStandardTransforms(_train, true, _config.getMosaic() > 0.0f);
            }

            torch::Tensor SegmentationDataset::getTargetTensor(size_t index, const Annotation& annotations)
            {
                // Get transformed image size from config
                // Annotations are already transformed to pixel coordinates at this size
                int width = _config.getImageSize();
                int height = _config.getImageSize();

                // Create mutable copy to denormalize
                auto annotations_ = annotations;
                annotations_.denormalize(width, height);  // Ensure annotations are in pixel coordinates

                // Get number of classes
                int numClasses = _config.getNumClasses();

                // Create empty mask with class IDs [H, W]
                cv::Mat mask = cv::Mat::zeros(height, width, CV_32S);

                // Get polygon coordinates and class IDs from annotations
                const auto& classes = annotations_.getClasses();
                const auto& points = annotations_.getPoints();

                // Validate that classes and points sizes match
                if (classes.size() != points.size())
                {
                    throw WheelDL::Utils::DataException(
                        WheelDL::Utils::ErrorCode::DATA_ANNOTATION_INVALID,
                        "Mismatch between classes and points size in SegmentationDataset"
                    );
                }

                // Draw each polygon on the mask
                // Annotations are already in pixel coordinates after transforms
                for (size_t i = 0; i < classes.size(); ++i)
                {
                    int classId = classes[i];
                    const auto& polygon = points[i];

                    // Convert polygon coordinates to cv::Point format
                    // Coordinates are already in pixels (not normalized)
                    std::vector<cv::Point> contour;
                    for (size_t j = 0; j < polygon.size(); j += 2)
                    {
                        if (j + 1 < polygon.size())
                        {
                            int x = static_cast<int>(polygon[j]);
                            int y = static_cast<int>(polygon[j + 1]);
                            contour.push_back(cv::Point(x, y));
                        }
                    }

                    // Fill polygon with class ID
                    if (!contour.empty())
                    {
                        std::vector<std::vector<cv::Point>> contours = {contour};
                        cv::fillPoly(mask, contours, cv::Scalar(classId));
                    }
                }

                // Convert to tensor [H, W]
                torch::Tensor maskTensor = torch::from_blob(
                    mask.data,
                    {mask.rows, mask.cols},
                    torch::kInt32
                ).clone();

                // Convert to one-hot encoding [C, H, W] using scatter for efficiency
                // This is O(H*W) instead of O(C*H*W) for loop-based approach
                torch::Tensor oneHot = torch::zeros({numClasses, height, width}, torch::kFloat32);

                // Reshape maskTensor to [1, H, W] and convert to Long for scatter
                torch::Tensor maskExpanded = maskTensor.unsqueeze(0).to(torch::kLong);

                // Use scatter to set one-hot values in a single pass
                // For each pixel at [h, w], if maskExpanded[0, h, w] = c, then oneHot[c, h, w] = 1.0
                oneHot.scatter_(0, maskExpanded, 1.0f);

                return oneHot;
            }

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
