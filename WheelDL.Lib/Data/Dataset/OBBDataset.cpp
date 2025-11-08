#include "pch.h"
#include "OBBDataset.h"
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
            OBBDataset::OBBDataset(
                const std::string& dataPath,
                const std::string& annotationPath,
                const Config::Configuration& config,
                bool train)
                : BaseDataset<OBBDataset>(dataPath, annotationPath, config)
                , _train(train)
            {
                loadAnnotations();
                _transforms = buildTransforms();
            }

            void OBBDataset::loadAnnotations()
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

                    Annotation annotation(LabelType::XYXYXYXY, true);  // MNIST uses XYXYXYXY pixel format (4 corner points)

                    std::string line;
                    int lineNumber = 0;
                    while (std::getline(file, line))
                    {
                        ++lineNumber;
                        if (line.empty()) continue;

                        std::istringstream iss(line);
                        int classId;
                        float x1, y1, x2, y2, x3, y3, x4, y4;

                        if (!(iss >> classId >> x1 >> y1 >> x2 >> y2 >> x3 >> y3 >> x4 >> y4))
                        {
                            // Invalid format, log warning and skip this line
                            std::cerr << "Warning: Invalid annotation format in "
                                      << (*annotationFile).string() << " at line "
                                      << lineNumber << ": " << line << std::endl;
                            continue;
                        }

                        // Add object to annotation (pixel coordinates, not normalized)
                        std::vector<float> obb = {x1, y1, x2, y2, x3, y3, x4, y4};
                        annotation.addObject(classId, obb);
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

            std::shared_ptr<Transforms::Transform> OBBDataset::buildTransforms()
            {
                // Use base class helper to build standard transforms
                // includeColorAugmentation = true for OBB detection task
                return buildStandardTransforms(_train, true);
            }

            torch::Tensor OBBDataset::getTargetTensor(size_t index, const Annotation& annotations)
            {
                const auto& classes = annotations.getClasses();
                const auto& points = annotations.getPoints();

                if (classes.empty())
                {
                    return torch::zeros({0, 9}, torch::kFloat32);
                }

                // Create tensor [num_objects, 9] (class, x1, y1, x2, y2, x3, y3, x4, y4)
                int numObjects = static_cast<int>(classes.size());
                torch::Tensor target = torch::zeros({numObjects, 9}, torch::kFloat32);

                auto accessor = target.accessor<float, 2>();

                for (int i = 0; i < numObjects; ++i)
                {
                    accessor[i][0] = static_cast<float>(classes[i]);  // class

                    if (points[i].size() >= 8)
                    {
                        accessor[i][1] = points[i][0];  // x1
                        accessor[i][2] = points[i][1];  // y1
                        accessor[i][3] = points[i][2];  // x2
                        accessor[i][4] = points[i][3];  // y2
                        accessor[i][5] = points[i][4];  // x3
                        accessor[i][6] = points[i][5];  // y3
                        accessor[i][7] = points[i][6];  // x4
                        accessor[i][8] = points[i][7];  // y4
                    }
                }

                return target;
            }

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
