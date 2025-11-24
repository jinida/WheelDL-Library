#include "pch.h"
#include "OBBDataset.h"
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
            OBBDataset::OBBDataset(
                const Config::Configuration& config,
                bool train)
                : BaseDataset<OBBDataset>(config, train)
            {
                loadAnnotations();
                _transforms = buildTransforms();
            }

            void OBBDataset::loadAnnotations()
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

                    Annotation annotation(LabelType::XYXYXYXY, false);  // XYXYXYXY format, not normalized (pixel coordinates)

                    const auto& labelArray = annot["label"];
                    for (const auto& obj : labelArray)
                    {
                        if (!obj.is_array() || obj.size() < 9)
                        {
                            std::cerr << "Warning: Invalid object format in " << filename << std::endl;
                            continue;
                        }

                        // Parse: [class_id, x1, y1, x2, y2, x3, y3, x4, y4]
                        int classId = obj[0].get<int>();
                        float x1 = obj[1].get<float>();
                        float y1 = obj[2].get<float>();
                        float x2 = obj[3].get<float>();
                        float y2 = obj[4].get<float>();
                        float x3 = obj[5].get<float>();
                        float y3 = obj[6].get<float>();
                        float x4 = obj[7].get<float>();
                        float y4 = obj[8].get<float>();

                        std::vector<float> obb = {x1, y1, x2, y2, x3, y3, x4, y4};
                        annotation.addObject(classId, obb);
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

            std::shared_ptr<Transforms::Transform> OBBDataset::buildTransforms()
            {
                // Use base class helper to build standard transforms
                // includeColorAugmentation = true for OBB detection task
                return buildStandardTransforms(_train, true, _config.getMosaic() > 0.0f);
            }

            torch::Tensor OBBDataset::getTargetTensor(size_t index, const Annotation& annotations)
            {
                auto annotation_ = annotations.getAs(LabelType::XYWHR);
				annotation_.normalize(_config.getImageSize(), _config.getImageSize());
				const auto& points = annotation_.getPoints();
				auto numObjects = static_cast<int>(points.size());

                if (numObjects == 0)
                {
                    return torch::zeros({0, 5}, torch::kFloat32);
				}

                torch::Tensor target = torch::zeros({numObjects, 5}, torch::kFloat32);

                auto accessor = target.accessor<float, 2>();

                for (int i = 0; i < numObjects; ++i)
                {
                    if (points[i].size() == 5)
                    {
                        accessor[i][0] = points[i][0];  // x1
                        accessor[i][1] = points[i][1];  // y1
                        accessor[i][2] = points[i][2];  // x2
                        accessor[i][3] = points[i][3];  // y2
                        accessor[i][4] = points[i][4];  // theta
                    }
                }

                return target;
            }

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
