#include "pch.h"
#include "SegmentationDataset.h"
#include "Data/Transforms/GeometricTransforms.h"
#include "Data/Transforms/ColorTransforms.h"
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
                const std::string& dataPath,
                const std::string& annotationPath,
                const Config::Configuration& config,
                bool train)
                : BaseDataset<SegmentationDataset>(dataPath, annotationPath, config)
                , _train(train)
            {
                loadAnnotations();
                _transforms = buildTransforms();
            }

            void SegmentationDataset::loadAnnotations()
            {
                if (!std::filesystem::exists(_annotationPath))
                {
                    throw std::runtime_error("Annotation directory does not exist: " + _annotationPath);
                }

                if (!std::filesystem::exists(_dataPath))
                {
                    throw std::runtime_error("Data directory does not exist: " + _dataPath);
                }

                // Iterate through all images in data directory
                for (const auto& entry : std::filesystem::directory_iterator(_dataPath))
                {
                    if (!entry.is_regular_file()) continue;

                    std::string imagePath = entry.path().string();
                    std::string extension = entry.path().extension().string();

                    // Check if it's an image file (case-insensitive) using base class helper
                    if (!isImageFile(extension))
                    {
                        continue;
                    }

                    // Find corresponding annotation file using base class helper
                    auto annotationFile = findAnnotationFile(entry.path(), _annotationPath);
                    if (!annotationFile)
                    {
                        continue;
                    }

                    // Load annotation
                    std::ifstream file(*annotationFile);
                    if (!file.is_open())
                    {
                        continue;
                    }

                    Annotation annotation(LabelType::POLYGON, true);  // Polygon format

                    std::string line;
                    int lineNumber = 0;
                    while (std::getline(file, line))
                    {
                        ++lineNumber;
                        if (line.empty()) continue;

                        std::istringstream iss(line);
                        int classId;

                        if (!(iss >> classId))
                        {
                            // Invalid format, log warning and skip this line
                            std::cerr << "Warning: Invalid annotation format in "
                                      << (*annotationFile).string() << " at line "
                                      << lineNumber << ": " << line << std::endl;
                            continue;
                        }

                        // Read polygon coordinates
                        std::vector<float> polygon;
                        float coord;
                        while (iss >> coord)
                        {
                            polygon.push_back(coord);
                        }

                        // Add polygon to annotation (pixel coordinates, not normalized)
                        if (!polygon.empty())
                        {
                            annotation.addObject(classId, polygon);
                        }
                    }
                    // file automatically closed by RAII when going out of scope

                    // NOTE: Normalization will be done after transforms are applied

                    // Add to dataset
                    _imagePaths.push_back(imagePath);
                    _annotations[imagePath] = annotation;
                }

                if (_imagePaths.empty())
                {
                    throw std::runtime_error("No valid images with annotations found");
                }
            }

            std::shared_ptr<Transforms::Transform> SegmentationDataset::buildTransforms()
            {
                // Use base class helper to build standard transforms
                // includeColorAugmentation = true for segmentation task
                return buildStandardTransforms(_train, true);
            }

            torch::Tensor SegmentationDataset::getTargetTensor(size_t index, const Annotation& annotations)
            {
                // Get transformed image size from config
                // Annotations are already transformed to pixel coordinates at this size
                int width = _config.getImageSize();
                int height = _config.getImageSize();

                // Get number of classes
                int numClasses = _config.getNumClasses();

                // Create empty mask with class IDs [H, W]
                cv::Mat mask = cv::Mat::zeros(height, width, CV_32S);

                // Get polygon coordinates and class IDs from annotations
                const auto& classes = annotations.getClasses();
                const auto& points = annotations.getPoints();

                // Validate that classes and points sizes match
                if (classes.size() != points.size())
                {
                    throw std::runtime_error("Mismatch between classes and points size in SegmentationDataset");
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
