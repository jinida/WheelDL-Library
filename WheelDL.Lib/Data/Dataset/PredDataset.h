#pragma once

#include "../Common/Annotation.h"
#include "../Cache/RAMCache.h"
#include "../Utils/ImageIO.h"
#include "../Transforms/Transform.h"
#include "../Transforms/GeometricTransforms.h"
#include "../Transforms/ColorTransforms.h"
#include "../../Config/JsonParser.h"
#include "../../Config/Configuration.h"
#include "../../Utils/Error/WheelLibException.h"
#include "BaseDataset.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <mutex>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace WheelDL
{
    namespace Data
    {
        namespace Dataset
        {
            /**
             * @struct PredDataExample
             * @brief Single data example returned by prediction dataset
             */
            struct PredDataExample
            {
                torch::Tensor data;                     // Image tensor [C, H, W]
                std::vector<std::string> imagePath;                  // Path to the image
                std::vector<std::tuple<int, int>> originalShape;     // (height, width) of original image
            };

            /**
             * @class PredDataset
             * @brief Dataset class for inference/prediction
             *
             * This is a standalone dataset class designed specifically for inference.
             * Unlike training datasets, it does not require annotations and only loads images.
             *
             * Features:
             * - Image loading with optional RAM caching
             * - Simple transform pipeline: LetterBox -> ToTensor -> (optional) ImageNet Normalization
             * - Returns image tensor, path, and original dimensions for post-processing
             *
             * Usage:
             * @code
             * Config::Configuration config = ...;
             * auto dataset = std::make_shared<PredDataset>(config, true);  // true for ImageNet norm
             * dataset->loadImages("/path/to/images");
             *
             * auto dataLoader = torch::data::make_data_loader(
             *     dataset,
             *     torch::data::DataLoaderOptions().batch_size(16)
             * );
             * @endcode
             */
            class PredDataset : public torch::data::Dataset<PredDataset, PredDataExample>
            {
            public:
                /**
                 * @brief Construct a new PredDataset
                 * @param config Configuration object containing settings (image size, cache, etc.)
                 * @param useImageNetNorm Whether to apply ImageNet normalization (default: false)
                 */
				explicit PredDataset(const Config::Configuration& config, bool train = false)
                    : _config(config)
                    , _cacheType(CacheType::NONE)
                    , _cacheMutex(std::make_shared<std::mutex>())
                {
                    // Enable cache based on configuration
                    std::string cacheType = _config.getCacheType();
                    if (cacheType == "ram")
                    {
                        enableCache(CacheType::RAM);
                    }
                    _useImageNetNorm = _config.getImageNetNorm();
                    _transforms = buildTransforms();
					_train = train;
                    loadAnnotations();
                }


                /**
                 * @brief Get dataset size (required by LibTorch)
                 * @return torch::optional<size_t> Number of images
                 */
                torch::optional<size_t> size() const override
                {
                    return _imagePaths.size();
                }

                /**
                 * @brief Get a single data sample (required by LibTorch)
                 * @param index Sample index
                 * @return PredDataExample The data sample with tensor, path, and original shape
                 * @throws WheelDL::Utils::DataException if index out of range or image loading fails
                 */
                PredDataExample get(size_t index) override
                {
                    if (index >= _imagePaths.size())
                    {
                        throw WheelDL::Utils::DataException(
                            WheelDL::Utils::ErrorCode::OUT_OF_RANGE,
                            "Index out of range: " + std::to_string(index)
                        );
                    }

                    const std::string& imagePath = _imagePaths[index];

                    // Load image (with caching if enabled)
                    cv::Mat image = loadImage(imagePath);

                    if (image.empty())
                    {
                        throw WheelDL::Utils::DataException(
                            WheelDL::Utils::ErrorCode::INVALID_IMAGE_SIZE,
                            "Failed to load image: " + imagePath
                        );
                    }

                    // Store original dimensions
                    int originalHeight = image.rows;
                    int originalWidth = image.cols;

                    Annotation dummyAnnotation;
                    _transforms->apply(image, dummyAnnotation);

                    torch::Tensor imageTensor = imageToTensor(image);

                    PredDataExample example;
                    example.data = imageTensor;
                    example.imagePath = { imagePath };
                    example.originalShape = { std::make_tuple(originalHeight, originalWidth) };

                    return example;
                }

                /**
                 * @brief Enable caching
                 * @param type Cache type (RAM)
                 */
                void enableCache(CacheType type)
                {
                    std::lock_guard<std::mutex> lock(*_cacheMutex);
                    _cacheType = type;

                    size_t maxSize = _config.getCacheSize();

                    if (type == CacheType::RAM)
                    {
                        _imageCache = std::make_shared<Cache::RAMCache>(maxSize);
                    }
                }

                /**
                 * @brief Clear all caches
                 */
                void clearCache()
                {
                    std::lock_guard<std::mutex> lock(*_cacheMutex);
                    if (_imageCache)
                    {
                        _imageCache->clear();
                    }
                }

                /**
                 * @brief Get list of image paths
                 * @return const std::vector<std::string>& Image paths
                 */
                const std::vector<std::string>& getImagePaths() const
                {
                    return _imagePaths;
                }

                void limitSamples(size_t maxSamples)
                {
                    if (maxSamples > 0 && maxSamples < _imagePaths.size())
                    {
                        _imagePaths.resize(maxSamples);
                    }
                }

            private:

                void loadAnnotations()
                {
                    using namespace Config;
                    using namespace WheelDL::Utils;
                    std::filesystem::path annotPath(_config.getDatasetPath());
                    auto dataPath = annotPath.parent_path().parent_path().parent_path();

                    // Validate paths
                    if (!std::filesystem::exists(annotPath))
                    {
                        throw DataException(
                            ErrorCode::DATA_FILE_NOT_FOUND,
                            "Annotation file does not exist: " + annotPath.string()
                        );
                    }

                    if (!std::filesystem::exists(dataPath))
                    {
                        throw DataException(
                            ErrorCode::DATASET_NOT_FOUND,
                            "Data directory does not exist: " + dataPath.string()
                        );
                    }

                    // Parse JSON annotation file
                    nlohmann::json annotationJson = JsonParser::parseFrom(annotPath.string());

                    // Validate JSON structure
                    if (!annotationJson.contains("annotations") || !annotationJson["annotations"].is_array())
                    {
                        throw DataException(
                            ErrorCode::DATA_INVALID_FORMAT,
                            "Invalid annotation JSON: missing 'annotations' array"
                        );
                    }

                    // Process each annotation
                    const auto& annotations = annotationJson["annotations"];
                    for (const auto& annot : annotations)
                    {
                        // Check role filter
                        int role = JsonParser::getInt(annot, "role", -1);
                        if (_train ? (role != 0) : (role == 0))
                        {
                            continue;
                        }
                        // Get filename
                        std::string filename = JsonParser::getString(annot, "filename", "");
                        if (filename.empty())
                        {
                            std::cerr << "Warning: Annotation missing filename, skipping" << std::endl;
                            continue;
                        }

                        // Build full image path
                        std::filesystem::path imagePath = dataPath / filename;
                        if (!std::filesystem::exists(imagePath))
                        {
                            std::cerr << "Warning: Image file not found: " << imagePath << std::endl;
                            continue;
                        }

                        std::string imagePathStr = imagePath.string();
                        _imagePaths.push_back(imagePathStr);
                    }

                    if (_imagePaths.empty())
                    {
                        throw DataException(ErrorCode::DATA_LOAD_FAILED, "No valid images with annotations found for validation or test or none");
                    }
                }
                /**
                 * @brief Load image from disk or cache
                 * @param imagePath Path to image file
                 * @return cv::Mat Loaded image
                 */
                cv::Mat loadImage(const std::string& imagePath)
                {
                    // Check cache first
                    cv::Mat image;
                    {
                        std::lock_guard<std::mutex> lock(*_cacheMutex);
                        if (_cacheType == CacheType::RAM && _imageCache)
                        {
                            auto cachedImage = _imageCache->get(imagePath);
                            if (cachedImage.has_value())
                            {
                                return cachedImage.value();
                            }
                        }
                    }

                    // Cache miss: load from disk
                    image = Utils::ImageIO::loadImage(imagePath);

                    // Add to cache
                    {
                        std::lock_guard<std::mutex> lock(*_cacheMutex);
                        if (_cacheType == CacheType::RAM && _imageCache)
                        {
                            _imageCache->put(imagePath, image);
                        }
                    }

                    return image;
                }

                /**
                 * @brief Build transform pipeline for prediction
                 *
                 * Pipeline:
                 * 1. LetterBox - Resize with aspect ratio preservation and padding
                 * 2. ToTensor - Convert to [C, H, W] float tensor in [0, 1]
                 * 3. Normalize (optional) - Apply ImageNet normalization if enabled
                 *
                 * @return std::shared_ptr<Transforms::Transform> Transform pipeline
                 */
                std::shared_ptr<Transforms::Transform> buildTransforms()
                {
                    auto compose = std::make_unique<Transforms::Compose>();

                    int imageSize = _config.getImageSize();
                    int fillBorder = _config.getFillBorder();

                    // 1. LetterBox - Maintain aspect ratio with padding
                    compose->addTransform(std::make_unique<Transforms::LetterBox>(imageSize, imageSize));

                    // 2. ToTensor - Convert to float [0, 1] and BGR to RGB
                    compose->addTransform(std::make_unique<Transforms::ToTensor>());
                    if (_useImageNetNorm)
                    {
                        compose->addTransform(
                            std::make_unique<Transforms::Normalize>(Transforms::Normalize::imageNet())
                        );
                    }

                    return std::move(compose);
                }

                /**
                 * @brief Convert OpenCV image to LibTorch tensor
                 * @param image OpenCV image
                 * @return torch::Tensor Image tensor [C, H, W] in float32
                 */
                torch::Tensor imageToTensor(const cv::Mat& image)
                {
                    // Validate image
                    if (image.empty())
                    {
                        throw WheelDL::Utils::DataException(
                            WheelDL::Utils::ErrorCode::INVALID_IMAGE_SIZE,
                            "Cannot convert empty image to tensor"
                        );
                    }

                    // Convert to float
                    cv::Mat floatImage;
                    image.convertTo(floatImage, CV_32FC3);

                    // Ensure continuous memory layout
                    if (!floatImage.isContinuous())
                    {
                        floatImage = floatImage.clone();
                    }

                    // Create tensor and copy data
                    torch::Tensor tensor = torch::empty(
                        { floatImage.rows, floatImage.cols, 3 },
                        torch::kFloat32
                    );
                    std::memcpy(
                        tensor.data_ptr<float>(),
                        floatImage.data,
                        floatImage.total() * floatImage.elemSize()
                    );

                    // Permute to [C, H, W]
                    tensor = tensor.permute({ 2, 0, 1 });

                    return tensor;
                }

                /**
                 * @brief Check if file has a valid image extension
                 * @param extension File extension (should include the dot, e.g., ".jpg")
                 * @return bool True if extension is a valid image format
                 */
                static bool isImageFile(const std::string& extension)
                {
                    std::string ext = extension;
                    // Convert to lowercase for case-insensitive comparison
                    std::transform(ext.begin(), ext.end(), ext.begin(),
                        [](unsigned char c) { return std::tolower(c); });

                    return ext == ".jpg" || ext == ".jpeg" ||
                        ext == ".png" || ext == ".bmp" ||
                        ext == ".tiff" || ext == ".tif";
                }

            private:
                Config::Configuration _config;                      // Configuration object
                bool _useImageNetNorm;                              // Whether to use ImageNet normalization
                std::vector<std::string> _imagePaths;               // List of image paths
                CacheType _cacheType;                               // Cache type
                std::shared_ptr<Cache::RAMCache> _imageCache;       // RAM cache
                std::shared_ptr<Transforms::Transform> _transforms; // Transform pipeline
                std::shared_ptr<std::mutex> _cacheMutex;            // Mutex for thread-safe cache access
				bool _train;
            };

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
