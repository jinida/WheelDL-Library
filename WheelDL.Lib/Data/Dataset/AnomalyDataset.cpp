#include "pch.h"
#include "AnomalyDataset.h"
#include "Data/Transforms/GeometricTransforms.h"
#include "Data/Transforms/ColorTransforms.h"
#include "Config/JsonParser.h"
#include "Utils/Error/WheelLibException.h"
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
            AnomalyDataset::AnomalyDataset(
                const Config::Configuration& config,
                bool train)
                : BaseDataset<AnomalyDataset>(config, train)
            {
                loadAnnotations();
                _transforms = buildTransforms();

            }

            void AnomalyDataset::loadAnnotations()
            {
                using namespace Config;
                using namespace WheelDL::Utils;

                // Validate paths
                if (!std::filesystem::exists(_dataPath))
                {
                    throw DataException(
                        ErrorCode::DATASET_NOT_FOUND,
                        "Data directory does not exist: " + _dataPath
                    );
                }

                // Check if annotation file exists
                bool hasLabels = !_annotationPath.empty() && std::filesystem::exists(_annotationPath);

                if (hasLabels)
                {
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

                        // Get anomaly label (0 = normal, 1 = anomaly)
                        int label = JsonParser::getInt(annot, "label", -1);
                        if (label < 0)
                        {
                            std::cerr << "Warning: Invalid label for " << filename << std::endl;
                            continue;
                        }

                        // Add to dataset
                        std::string imagePathStr = imagePath.string();
                        _imagePaths.push_back(imagePathStr);

                        // Create annotation with anomaly label
                        Annotation annotation = Annotation::forClassification(label);
                        _annotations[imagePathStr] = annotation;
                    }
                }
                else
                {
                    // No labels, load all images from directory (assumed normal for training)
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

                        // Add to image paths
                        _imagePaths.push_back(imagePath);

                        // Create annotation with no label (unknown)
                        Annotation annotation(LabelType::NONE);
                        _annotations[imagePath] = annotation;
                    }
                }

                if (_imagePaths.empty())
                {
                    throw DataException(
                        ErrorCode::DATA_LOAD_FAILED,
                        "No valid images found"
                    );
                }
            }

            std::shared_ptr<Transforms::Transform> AnomalyDataset::buildTransforms()
            {
                if (_train)
                {
                    auto aeTransform = std::make_unique<Transforms::Compose>();
                    aeTransform->addTransform(std::make_unique<Transforms::LetterBox>(_config.getImageSize(), _config.getImageSize()));
                    aeTransform->addTransform(std::make_unique<Transforms::ColorJitter>(0.2f, 0.2f, 0.2f, 0.0f));
                    aeTransform->addTransform(std::make_unique<Transforms::ToTensor>());
                    aeTransform->addTransform(std::make_unique<Transforms::Normalize>(Transforms::Normalize::imageNet()));
                    _aeTransforms = std::move(aeTransform);
                }

				auto transform = std::make_shared<Transforms::Compose>();
                transform->addTransform(std::make_unique<Transforms::LetterBox>(_config.getImageSize(), _config.getImageSize()));
                if (_train)
                {
					transform->addTransform(std::make_unique<Transforms::RandomHorizontalFlip>(_config.getFlipLR()));
                }
                transform->addTransform(std::make_unique<Transforms::ToTensor>());
                transform->addTransform(std::make_unique<Transforms::Normalize>(Transforms::Normalize::imageNet()));
				return transform;
            }

            torch::Tensor AnomalyDataset::getTargetTensor(size_t index, const Annotation& annotations)
            {
                // Get anomaly label from annotation
                if (_config.IsEfficientAD() && _train)
                {
                    auto [image, annotations] = loadSample(index);
					_aeTransforms->apply(image, annotations);

                    return imageToTensor(image);
                }
                else
                {
                    const auto& classes = annotations.getClasses();

                    if (classes.empty())
                    {
                        // No label available (training with normal images only)
                        // Return as 1D tensor with single element (required for collation)
                        return torch::tensor({1}, torch::kLong);  // 1 indicates unknown/unlabeled
                    }

                    int label = classes[0];  // 0 = normal, 1 = anomaly

                    // Return as 1D tensor with single element (required for collation)
                    return torch::tensor({label}, torch::kLong);
                }
            }

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
