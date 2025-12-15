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

                    // Check if annotations array is empty - fall back to directory scan
                    const auto& annotations = annotationJson["annotations"];
                    if (annotations.empty())
                    {
                        hasLabels = false;
                    }
                    else
                    {
                        // Get expected role: 0=train, 1=test
                        int expectedRole = _train ? 0 : 1;

                        // Process each annotation
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
                                continue;
                            }

                            // Build full image path
                            std::filesystem::path imagePath = std::filesystem::path(_dataPath) / filename;
                            if (!std::filesystem::exists(imagePath))
                            {
                                continue;
                            }

                            // Get anomaly label (0 = normal, 1 = anomaly)
                            int label = JsonParser::getInt(annot, "label", -1);
                            if (label < 0)
                            {
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
                }

                if (!hasLabels)
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
                if (_config.IsEfficientAD())
                {
                    auto transform1 = std::make_unique<Transforms::Compose>();
                    transform1->addTransform(std::make_unique<Transforms::LetterBox>(_config.getImageSize(), _config.getImageSize()));
                    transform1->addTransform(std::make_unique<Transforms::ToTensor>());
                    transform1->addTransform(std::make_unique<Transforms::Normalize>(Transforms::Normalize::imageNet()));

                    auto transform2 = std::make_unique<Transforms::Compose>();
                    transform2->addTransform(std::make_unique<Transforms::LetterBox>(_config.getImageSize(), _config.getImageSize()));
                    transform2->addTransform(std::make_unique<Transforms::ColorJitter>(0.1f, 0.1f, 0.1f, 0.0f));
                    transform2->addTransform(std::make_unique<Transforms::ToTensor>());

                    return std::make_shared<Transforms::EfficientADTransform>(
                        std::move(transform1),
                        std::move(transform2)
                    );
                }
                else 
                {
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
            }

            torch::Tensor AnomalyDataset::getTargetTensor(size_t index, const Annotation& annotations)
            {
                const auto& classes = annotations.getClasses();

                if (classes.empty())
                {
                    return torch::tensor({0}, torch::kLong);
                }

                int label = classes[0];
                return torch::tensor({label}, torch::kLong);
            }

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
