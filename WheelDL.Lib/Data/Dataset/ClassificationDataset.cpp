#include "pch.h"
#include "ClassificationDataset.h"
#include "Data/Transforms/GeometricTransforms.h"
#include "Data/Transforms/ColorTransforms.h"
#include <filesystem>
#include <stdexcept>

namespace WheelDL
{
    namespace Data
    {
        namespace Dataset
        {
            ClassificationDataset::ClassificationDataset(
                const std::string& dataPath,
                const std::string& annotationPath,
                const Config::Configuration& config,
                bool train)
                : BaseDataset<ClassificationDataset>(dataPath, annotationPath, config)
                , _train(train)
            {
                // Load annotations
                loadAnnotations();

                // Build transforms
                _transforms = buildTransforms();
            }

            void ClassificationDataset::loadAnnotations()
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

                    // Check if it's an image file using base class helper
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

                    // Read class ID from file
                    int classId;
                    if (!(file >> classId))
                    {
                        std::cerr << "Warning: Invalid annotation format in " << annotationFile->string() << std::endl;
                        continue;
                    }
                    // file automatically closed by RAII when going out of scope

                    // Add to image paths
                    _imagePaths.push_back(imagePath);

                    // Create annotation with classification label
                    Annotation annotation = Annotation::forClassification(classId);
                    _annotations[imagePath] = annotation;
                }

                if (_imagePaths.empty())
                {
                    throw std::runtime_error("No valid images with annotations found");
                }
            }

            std::shared_ptr<Transforms::Transform> ClassificationDataset::buildTransforms()
            {
                // Use base class helper to build standard transforms
                // includeColorAugmentation = true for classification task
                return buildStandardTransforms(_train, true);
            }

            torch::Tensor ClassificationDataset::getTargetTensor(size_t index, const Annotation& annotations)
            {
                // Get class ID from annotation
                const auto& classes = annotations.getClasses();

                if (classes.empty())
                {
                    throw std::runtime_error("No class label found for image at index " + std::to_string(index));
                }

                int classId = classes[0];

                // Return as 1D tensor with single element (required for collation)
                return torch::tensor({classId}, torch::kLong);
            }

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
