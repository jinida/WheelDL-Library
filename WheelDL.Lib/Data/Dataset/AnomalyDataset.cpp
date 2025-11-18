#include "pch.h"
#include "AnomalyDataset.h"
#include "Data/Transforms/GeometricTransforms.h"
#include "Data/Transforms/ColorTransforms.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace WheelDL
{
    namespace Data
    {
        namespace Dataset
        {
            AnomalyDataset::AnomalyDataset(
                const std::string& dataPath,
                const std::string& annotationPath,
                const Config::Configuration& config,
                bool train)
                : BaseDataset<AnomalyDataset>(dataPath, annotationPath, config)
                , _train(train)
            {
                loadAnnotations();
                _transforms = buildTransforms();
                
            }

            void AnomalyDataset::loadAnnotations()
            {
                if (!std::filesystem::exists(_dataPath))
                {
                    throw std::runtime_error("Data directory does not exist: " + _dataPath);
                }

                // Check if annotation directory exists
                bool hasLabels = !_annotationPath.empty() && std::filesystem::exists(_annotationPath);

                if (hasLabels)
                {
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

                        // Read anomaly label from file (0 = normal, 1 = anomaly)
                        int isAnomaly;
                        if (!(file >> isAnomaly))
                        {
                            std::cerr << "Warning: Invalid annotation format in " << annotationFile->string() << std::endl;
                            continue;
                        }
                        // file automatically closed by RAII when going out of scope

                        // Add to image paths
                        _imagePaths.push_back(imagePath);

                        // Create annotation with anomaly label
                        Annotation annotation = Annotation::forClassification(isAnomaly);
                        _annotations[imagePath] = annotation;
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
                    throw std::runtime_error("No valid images found");
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
