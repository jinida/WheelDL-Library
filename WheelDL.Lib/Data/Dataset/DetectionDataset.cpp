#include "pch.h"
#include "DetectionDataset.h"
#include "Data/Transforms/GeometricTransforms.h"
#include "Data/Transforms/ColorTransforms.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace WheelDL
{
    namespace Data
    {
        namespace Dataset
        {
            DetectionDataset::DetectionDataset(
                const std::string& dataPath,
                const std::string& annotationPath,
                const Config::Configuration& config,
                bool train)
                : BaseDataset<DetectionDataset>(dataPath, annotationPath, config)
                , _train(train)
            {
                loadAnnotations();
                _transforms = buildTransforms();
            }

            void DetectionDataset::loadAnnotations()
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
                        // Skip images without annotations
                        continue;
                    }

                    // Load annotation
                    std::ifstream file(*annotationFile);
                    if (!file.is_open())
                    {
                        continue;
                    }

                    Annotation annotation(LabelType::XYWH, true);  // MNIST uses XYWH pixel format (x, y, w, h in pixels)

                    std::string line;
                    int lineNumber = 0;
                    while (std::getline(file, line))
                    {
                        ++lineNumber;
                        if (line.empty()) continue;

                        std::istringstream iss(line);
                        int classId;
                        float x, y, w, h;

                        if (!(iss >> classId >> x >> y >> w >> h))
                        {
                            // Invalid format, log warning and skip this line
                            std::cerr << "Warning: Invalid annotation format in "
                                      << annotationFile->string() << " at line "
                                      << lineNumber << ": " << line << std::endl;
                            continue;
                        }

                        // Add object to annotation (pixel coordinates, not normalized)
                        std::vector<float> bbox = {x, y, w, h};
                        annotation.addObject(classId, bbox);
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

            std::shared_ptr<Transforms::Transform> DetectionDataset::buildTransforms()
            {
                // Use base class helper to build standard transforms
                // includeColorAugmentation = true for detection task
                return buildStandardTransforms(_train, true);
            }

            torch::Tensor DetectionDataset::getTargetTensor(size_t index, const Annotation& annotations)
            {
                const auto& classes = annotations.getClasses();
                const auto& points = annotations.getPoints();

                if (classes.empty())
                {
                    // No objects, return empty tensor
                    return torch::zeros({0, 5}, torch::kFloat32);
                }

                // Validate that classes and points sizes match
                if (classes.size() != points.size())
                {
                    throw std::runtime_error("Mismatch between classes and points size in DetectionDataset");
                }

                // Create tensor [num_objects, 5] (class, x, y, w, h)
                int numObjects = static_cast<int>(classes.size());
                torch::Tensor target = torch::zeros({numObjects, 5}, torch::kFloat32);

                auto accessor = target.accessor<float, 2>();

                for (int i = 0; i < numObjects; ++i)
                {
                    accessor[i][0] = static_cast<float>(classes[i]);  // class

                    if (points[i].size() >= 4)
                    {
                        accessor[i][1] = points[i][0];  // x_center
                        accessor[i][2] = points[i][1];  // y_center
                        accessor[i][3] = points[i][2];  // width
                        accessor[i][4] = points[i][3];  // height
                    }
                }

                return target;
            }

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
