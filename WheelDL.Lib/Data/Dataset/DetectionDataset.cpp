#include "pch.h"
#include "DetectionDataset.h"
#include "Data/Transforms/GeometricTransforms.h"
#include "Data/Transforms/ColorTransforms.h"
#include "Data/Utils/ImageIO.h"
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
            DetectionDataset::DetectionDataset(
                const Config::Configuration& config,
                bool train)
                : BaseDataset<DetectionDataset>(config, train)
            {
                loadAnnotations();
                _transforms = buildTransforms();
            }

            void DetectionDataset::loadAnnotations()
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

                    // Get image dimensions without loading entire image (fast header parsing)
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

                    // Create annotation with normalized coordinates
                    Annotation annotation(LabelType::XYXY, false);  // XYXY format, normalized coordinates

                    const auto& labelArray = annot["label"];
                    for (const auto& obj : labelArray)
                    {
                        if (!obj.is_array() || obj.size() < 5)
                        {
                            std::cerr << "Warning: Invalid object format in " << filename << std::endl;
                            continue;
                        }

                        // Parse: [class_id, x1, y1, x2, y2] (pixel coordinates)
                        int classId = obj[0].get<int>();
                        float x1 = obj[1].get<float>();
                        float y1 = obj[2].get<float>();
                        float x2 = obj[3].get<float>();
                        float y2 = obj[4].get<float>();

						std::vector<float> bbox = { x1, y1, x2, y2 };
                        annotation.addObject(classId, bbox);
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

            std::shared_ptr<Transforms::Transform> DetectionDataset::buildTransforms()
            {
                // Use base class helper to build standard transforms
                // includeColorAugmentation = true for detection task
                return buildStandardTransforms(_train, true, _config.getMosaic() > 0.0f);
            }

            torch::Tensor DetectionDataset::getTargetTensor(size_t index, const Annotation& annotations)
            {
				auto annotations_ = annotations.getAs(WheelDL::Data::LabelType::XYWH);
                annotations_.denormalize(_config.getImageSize(), _config.getImageSize());

                const auto& points = annotations_.getPoints();

                if (points.empty())
                {
                    return torch::zeros({0, 4}, torch::kFloat32);
                }

                const auto& classes = annotations_.getClasses();

                if (classes.size() != points.size())
                {
                    throw WheelDL::Utils::DataException(
                        WheelDL::Utils::ErrorCode::DATA_ANNOTATION_INVALID,
                        "Mismatch between classes and points size in DetectionDataset"
                    );
                }

                // Create tensor [num_objects, 5] (class, x, y, w, h)
                int numObjects = static_cast<int>(classes.size());
                torch::Tensor target = torch::zeros({numObjects, 4}, torch::kFloat32);

                auto accessor = target.accessor<float, 2>();

                for (int i = 0; i < numObjects; ++i)
                {
                    if (points[i].size() == 4)
                    {
                        accessor[i][0] = points[i][0];  // x_center
                        accessor[i][1] = points[i][1];  // y_center
                        accessor[i][2] = points[i][2];  // width
                        accessor[i][3] = points[i][3];  // height
                    }
                }

                return target;
            }

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
