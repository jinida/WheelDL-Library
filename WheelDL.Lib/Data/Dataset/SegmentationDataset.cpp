#include "pch.h"
#include "SegmentationDataset.h"
#include "Data/Transforms/GeometricTransforms.h"
#include "Data/Transforms/ColorTransforms.h"
#include "Config/JsonParser.h"
#include "Utils/Error/WheelLibException.h"
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
                const Config::Configuration& config,
                bool train)
                : BaseDataset<SegmentationDataset>(config, train)
            {
                loadAnnotations();
                _transforms = buildTransforms();
            }

            void SegmentationDataset::loadAnnotations()
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

                    Annotation annotation(LabelType::POLYGON, false);  // Polygon format, not normalized (pixel coordinates)

                    const auto& labelArray = annot["label"];
                    for (const auto& obj : labelArray)
                    {
                        if (!obj.is_array() || obj.size() < 3)
                        {
                            std::cerr << "Warning: Invalid object format in " << filename << std::endl;
                            continue;
                        }

                        // Parse: [class_id, x1, y1, x2, y2, ..., xn, yn]
                        int classId = obj[0].get<int>();

                        // Extract polygon coordinates
                        std::vector<float> polygon;
                        for (size_t i = 1; i < obj.size(); ++i)
                        {
                            polygon.push_back(obj[i].get<float>());
                        }

                        // Add polygon to annotation (pixel coordinates, not normalized)
                        if (!polygon.empty())
                        {
                            annotation.addObject(classId, polygon);
                        }
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

            std::shared_ptr<Transforms::Transform> SegmentationDataset::buildTransforms()
            {
                // Use base class helper to build standard transforms
                // includeColorAugmentation = true for segmentation task
                return buildStandardTransforms(_train, true, _config.getMosaic() > 0.0f);
            }

            torch::Tensor SegmentationDataset::getTargetTensor(size_t index, const Annotation& annotations)
            {
                int width = _config.getImageSize();
                int height = _config.getImageSize();
                int numClasses = _config.getNumClasses();

                auto annotations_ = annotations;
                annotations_.denormalize(width, height);

                const auto& classes = annotations_.getClasses();
                const auto& points = annotations_.getPoints();

                if (classes.size() != points.size()) {
                    throw WheelDL::Utils::DataException(
                        WheelDL::Utils::ErrorCode::DATA_ANNOTATION_INVALID,
                        "Mismatch between classes and points size"
                    );
                }

                torch::Tensor oneHot = torch::zeros({ numClasses, height, width }, torch::kFloat32);
                float* dataPtr = oneHot.data_ptr<float>();
                int64_t channelSize = static_cast<int64_t>(height) * width;

                for (size_t i = 0; i < classes.size(); ++i)
                {
                    int classId = classes[i];
                    if (classId < 0 || classId >= numClasses) continue;

                    const auto& polygon = points[i];

                    std::vector<cv::Point> contour;
                    contour.reserve(polygon.size() / 2);

                    for (size_t j = 0; j + 1 < polygon.size(); j += 2) {
                        contour.emplace_back(
                            static_cast<int>(polygon[j]),
                            static_cast<int>(polygon[j + 1])
                        );
                    }

                    if (contour.size() >= 3) {
                        cv::Mat channelMat(height, width, CV_32FC1, dataPtr + classId * channelSize);
                        cv::fillPoly(channelMat, std::vector<std::vector<cv::Point>>{contour}, cv::Scalar(1.0f));
                    }
                }

                return oneHot;
            }
        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
