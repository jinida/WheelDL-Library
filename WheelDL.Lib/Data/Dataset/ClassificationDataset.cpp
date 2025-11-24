#include "pch.h"
#include "ClassificationDataset.h"
#include "Data/Transforms/GeometricTransforms.h"
#include "Data/Transforms/ColorTransforms.h"
#include "Config/JsonParser.h"
#include "Utils/Error/WheelLibException.h"
#include <filesystem>
#include <stdexcept>
#include <iostream>

namespace WheelDL
{
    namespace Data
    {
        namespace Dataset
        {
            ClassificationDataset::ClassificationDataset(
                const Config::Configuration& config,
                bool train)
                : BaseDataset<ClassificationDataset>(config, train)
            {
                // Load annotations
                loadAnnotations();

                // Build transforms
                _transforms = buildTransforms();
            }

            void ClassificationDataset::loadAnnotations()
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

                    // Get class ID
                    int classId = JsonParser::getInt(annot, "label", -1);
                    if (classId < 0)
                    {
                        std::cerr << "Warning: Invalid class ID for " << filename << std::endl;
                        continue;
                    }

                    // Add to dataset
                    std::string imagePathStr = imagePath.string();
                    _imagePaths.push_back(imagePathStr);

                    // Create annotation with classification label
                    Annotation annotation = Annotation::forClassification(classId);
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
                    throw WheelDL::Utils::DataException(
                        WheelDL::Utils::ErrorCode::DATA_ANNOTATION_INVALID,
                        "No class label found for image at index " + std::to_string(index)
                    );
                }

                int classId = classes[0];

                // Return as 1D tensor with single element (required for collation)
                return torch::tensor({classId}, torch::kLong);
            }

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
