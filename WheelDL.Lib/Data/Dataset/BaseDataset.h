#pragma once

#include "../Common/Annotation.h"
#include "../Cache/RAMCache.h"
#include "../Cache/CacheManager.h"
#include "../Utils/ImageIO.h"
#include "../Transforms/Transform.h"
#include "../Transforms/GeometricTransforms.h"
#include "../Transforms/ColorTransforms.h"
#include "../Augmentation/DatasetAugmentation.h"
#include "../../Config/Configuration.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <optional>
#include <opencv2/opencv.hpp>

namespace WheelDL
{
    namespace Data
    {
        namespace Dataset
        {
            /**
             * @enum CacheType
             * @brief Type of caching to use for dataset
             */
            enum class CacheType
            {
                NONE,   // No caching
                RAM,    // Cache in RAM
            };

            /**
             * @struct DataExample
             * @brief Single data example returned by dataset
             */
            struct DataExample
            {
                torch::Tensor data;      // Image tensor [C, H, W]
                torch::Tensor classes;
                torch::Tensor targets;    // Target tensor (labels, boxes, etc.)
                torch::Tensor batchIndices;
            };

            /**
             * @class BaseDataset
             * @brief Base dataset class using CRTP pattern
             *
             * @tparam Derived The derived dataset class (CRTP pattern)
             *
             * This class provides common functionality for all datasets:
             * - Image loading with caching support (RAM)
             * - Annotation management using user-implemented Annotation class
             * - Transform pipeline building (configured per task type)
             * - Integration with LibTorch DataLoader
             *
             * Derived classes must implement:
             * - loadAnnotations(): Load task-specific annotations
             * - buildTransforms(): Build task-specific transform pipeline
             * - getTargetTensor(index): Convert annotations to target tensor
             */

            template<typename Derived>
            class BaseDataset : public torch::data::Dataset<Derived, DataExample>
            {
            public:
                /**
                 * @brief Construct a new BaseDataset
                 * @param dataPath Path to image directory
                 * @param annotationPath Path to annotation file/directory
                 * @param config Configuration object containing all settings
                 */
                explicit BaseDataset(const std::string& dataPath,
                    const std::string& annotationPath,
                    const Config::Configuration& config)
                    : _dataPath(dataPath)
                    , _annotationPath(annotationPath)
                    , _config(config)
                    , _cacheType(CacheType::NONE)
                    , _cacheMutex(std::make_shared<std::mutex>())
                {
                    // Enable cache based on configuration
                    std::string cacheType = _config.getCacheType();
                    if (cacheType == "ram")
                    {
                        enableCache(CacheType::RAM);
                    }
                }


                /**
                 * @brief Get dataset size (required by LibTorch)
                 * @return torch::optional<size_t> Number of samples
                 */
                torch::optional<size_t> size() const override
                {
                    return _imagePaths.size();
                }

                /**
                 * @brief Get a single data sample (required by LibTorch)
                 * @param index Sample index
                 * @return DataExample The data sample
                 */
                DataExample get(size_t index) override
                {
                    if (index >= _imagePaths.size())
                    {
                        throw std::out_of_range("Index out of range: " + std::to_string(index));
                    }

                    // 1. Load base image and annotation
                    auto [image, annotations] = loadSample(index);
                    annotations.denormalize(image.cols, image.rows);

                    // 2. Apply Dataset-level augmentation (if configured and enabled)
                    if (isMosaicEnabled())
                    {
                        // Apply Mosaic (probability check inside apply())
                        auto indices = getRandomIndices(index, 3);  // Additional 3 samples
                        std::vector<cv::Mat> images = { image };
                        std::vector<Annotation> annots = { annotations };

                        for (auto idx : indices)
                        {
                            auto [img, ann] = loadSample(idx);
                            images.push_back(img);
                            annots.push_back(ann);
                        }

                        // apply() returns false if skipped by probability, original kept
                        _mosaicTransform->apply(images, annots, image, annotations);
                    }

                    // 3. Apply general transform pipeline (YOLOv5/v8 method)
                    // NOTE: After Mosaic composition, apply RandomPerspective, ColorJitter, etc. to entire image
                    //       4 regions are transformed together like one scene (intentional behavior)
                    if (_transforms)
                    {
                        _transforms->apply(image, annotations);
                    }

                    // 4. Convert to tensor [C, H, W]
                    torch::Tensor imageTensor = imageToTensor(image);

                    // Get classes from annotations
                    const auto& classIds = annotations.getClasses();
                    torch::Tensor classesTensor;
                    if (!classIds.empty())
                    {
                        classesTensor = torch::tensor(classIds, torch::kLong);
                    }
                    else
                    {
                        classesTensor = torch::zeros({ 0 }, torch::kLong);
                    }

                    if (_config.getTaskType() != TaskType::SEGMENTATION)
                    {
                        annotations.normalize(image.cols, image.rows);
                    }

                    // Get target tensor from derived class
                    torch::Tensor targets = getTargetTensor(index, annotations);

                    DataExample example;
                    example.data = imageTensor;
                    example.classes = classesTensor;
                    example.targets = targets;

                    return example;
                }

                /**
                 * @brief Enable caching
                 * @param type Cache type (RAM or Disk)
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
                 * @brief Get data path
                 */
                const std::string& getDataPath() const { return _dataPath; }

                /**
                 * @brief Get annotation path
                 */
                const std::string& getAnnotationPath() const { return _annotationPath; }

                /**
                 * @brief Get configuration
                 */
                const Config::Configuration& getConfig() const { return _config; }

            protected:
                /**
                 * @brief Load annotations (must be implemented by derived class)
                 *
                 * This function should:
                 * 1. Read annotation files from _annotationPath
                 * 2. Populate _imagePaths vector
                 * 3. Populate _annotations map
                 */
                virtual void loadAnnotations() = 0;

                /**
                 * @brief Build transform pipeline (must be implemented by derived class)
                 *
                 * This function should create and configure transforms based on:
                 * - Task type
                 * - Configuration settings
                 * - Training vs validation mode
                 *
                 * @return std::unique_ptr<Transforms::Transform> Transform pipeline
                 */
                virtual std::shared_ptr<Transforms::Transform> buildTransforms() = 0;

                /**
                 * @brief Get target tensor for a given index (implemented by derived class)
                 *
                 * Convert annotations to task-specific target tensor format.
                 *
                 * @param index Sample index
                 * @param annotations Annotations for this sample (after transforms applied)
                 * @return torch::Tensor Target tensor
                 */
                virtual torch::Tensor getTargetTensor(size_t index, const Annotation& annotations) = 0;

                /**
                 * @brief Disable Mosaic augmentation
                 *
                 * Deletes the Mosaic transform object to disable augmentation.
                 * Used in final epochs for better generalization (YOLOv5/v8 strategy).
                 *
                 * Usage in training loop:
                 * @code
                 * if (epoch >= totalEpochs - closeMosaicEpochs) {
                 *     dataset->disableMosaic();
                 * }
                 * @endcode
                 */
                void disableMosaic()
                {
                    _mosaicTransform.reset();  // Delete Mosaic object
                }

                /**
                 * @brief Check if Mosaic augmentation is currently enabled
                 * @return bool True if enabled (Mosaic object exists)
                 */
                bool isMosaicEnabled() const
                {
                    return _mosaicTransform != nullptr;
                }

                /**
                 * @brief Randomly select other sample indices
                 * @param currentIndex Current index (will be excluded)
                 * @param count Number of additional samples needed
                 * @return std::vector<size_t> Random index list
                 */
                std::vector<size_t> getRandomIndices(size_t currentIndex, size_t count)
                {
                    std::vector<size_t> indices;
                    indices.reserve(count);

                    size_t totalSize = _imagePaths.size();
                    if (totalSize <= 1)
                    {
                        throw std::runtime_error("Dataset size too small for mosaic augmentation");
                    }

                    // Thread-local random generator
                    thread_local WheelDL::Utils::Random rng;

                    for (size_t i = 0; i < count; ++i)
                    {
                        size_t randomIndex;
                        do
                        {
                            randomIndex = rng.uniformInt(0, static_cast<int>(totalSize - 1));
                        } while (randomIndex == currentIndex ||
                            std::find(indices.begin(), indices.end(), randomIndex) != indices.end());

                        indices.push_back(randomIndex);
                    }

                    return indices;
                }

                /**
                 * @brief Load additional sample (for mosaic augmentation)
                 * @param index Sample index
                 * @return std::pair<cv::Mat, Annotation> Image and annotation
                 */
                std::pair<cv::Mat, Annotation> loadSample(size_t index)
                {
                    const std::string& imagePath = _imagePaths[index];

                    // Check cache
                    cv::Mat image;
                    {
                        std::lock_guard<std::mutex> lock(*_cacheMutex);
                        if (_cacheType == CacheType::RAM && _imageCache)
                        {
                            auto cachedImage = _imageCache->get(imagePath);
                            if (cachedImage.has_value())
                            {
                                image = cachedImage.value();
                            }
                        }
                    }

                    // Cache miss: load
                    if (image.empty())
                    {
                        image = Utils::ImageIO::loadImage(imagePath, _config.getImageSize());

                        std::lock_guard<std::mutex> lock(*_cacheMutex);
                        if (_cacheType == CacheType::RAM && _imageCache)
                        {
                            _imageCache->put(imagePath, image);
                        }
                    }

                    // Copy annotation
                    auto it = _annotations.find(imagePath);
                    if (it == _annotations.end())
                    {
                        throw std::runtime_error("Annotation not found for image: " + imagePath);
                    }

                    return { image, it->second.clone() };
                }

                /**
                 * @brief Check if file has a valid image extension (helper for derived classes)
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
                        ext == ".png" || ext == ".bmp";
                }

                /**
                 * @brief Find annotation file for an image (helper for derived classes)
                 * @param imagePath Path to the image file
                 * @param annotationDir Directory containing annotation files
                 * @param annotationExt Annotation file extension (default: ".txt")
                 * @return std::optional<std::filesystem::path> Path to annotation file if found
                 */
                static std::optional<std::filesystem::path> findAnnotationFile(
                    const std::filesystem::path& imagePath,
                    const std::filesystem::path& annotationDir,
                    const std::string& annotationExt = ".txt")
                {
                    std::filesystem::path annotationFile =
                        annotationDir / (imagePath.stem().string() + annotationExt);

                    if (std::filesystem::exists(annotationFile)) {
                        return annotationFile;
                    }
                    return std::nullopt;
                }

                /**
                 * @brief Build standard transform pipeline (helper for derived classes)
                 *
                 * Creates a standard transform pipeline with common augmentations.
                 * Derived classes can call this method to avoid code duplication.
                 *
                 * @param train If true, adds training augmentations; otherwise validation transforms
                 * @param includeColorAugmentation If true, adds ColorJitter and GaussianBlur (default: true)
                 * @param enableMosaicAugmentation If true, enables Mosaic dataset augmentation (default: false)
                 *                                  NOTE: Only use for Detection, OBB, and Segmentation tasks.
                 *                                  Classification and Anomaly tasks should keep this false.
                 * @return std::unique_ptr<Transforms::Transform> Transform pipeline
                 */
                std::shared_ptr<Transforms::Transform> buildStandardTransforms(
                    bool train,
                    bool includeColorAugmentation = true,
                    bool enableMosaicAugmentation = false)
                {
                    auto compose = std::make_unique<Transforms::Compose>();

                    // Get image size from config
                    int imageSize = _config.getImageSize();
                    int fillBorder = _config.getFillBorder();

                    if (train)
                    {
                        // Training transforms with data augmentation
                        // 1. RandomPerspective - MUST be first for training
                        compose->addTransform(std::make_unique<Transforms::RandomPerspective>(
                            _config.getDegrees(),      // degrees
                            _config.getTranslate(),    // translate
                            _config.getScale(),        // scale
                            _config.getShear(),        // shear
                            _config.getPerspective(),  // perspective
                            imageSize, imageSize,
                            1.0f,                      // probability
                            fillBorder, fillBorder, fillBorder
                        ));

                        // RandomHorizontalFlip (skip if probability is 0)
                        float flipLR = _config.getFlipLR();
                        if (flipLR > 0.0f)
                        {
                            compose->addTransform(std::make_unique<Transforms::RandomHorizontalFlip>(flipLR));
                        }

                        // RandomVerticalFlip (skip if probability is 0)
                        float flipUD = _config.getFlipUD();
                        if (flipUD > 0.0f)
                        {
                            compose->addTransform(std::make_unique<Transforms::RandomVerticalFlip>(flipUD));
                        }

                        // Optional color augmentation (for Classification/Detection/OBB/Segmentation)
                        if (includeColorAugmentation)
                        {
                            // ColorJitter (skip if all parameters are 0)
                            float hsvV = _config.getHSVV();
                            float hsvS = _config.getHSVS();
                            float hsvH = _config.getHSVH();
                            if (hsvV > 0.0f || hsvS > 0.0f || hsvH > 0.0f)
                            {
                                compose->addTransform(std::make_unique<Transforms::ColorJitter>(
                                    hsvV,  // brightness
                                    hsvV,  // contrast
                                    hsvS,  // saturation
                                    hsvH   // hue
                                ));
                            }

                            // GaussianBlur (skip if probability is 0)
                            float blurProb = _config.getBlurProbability();
                            if (blurProb > 0.0f)
                            {
                                compose->addTransform(std::make_unique<Transforms::GaussianBlur>(
                                    3, _config.getBlurKernelSize(), blurProb
                                ));
                            }
                        }

                        // ToTensor (convert to float and scale to [0, 1])
                        compose->addTransform(std::make_unique<Transforms::ToTensor>(true));  // BGR to RGB
                    }
                    else
                    {
                        // Validation transforms
                        // 1. LetterBox
                        compose->addTransform(std::make_unique<Transforms::LetterBox>(
                            imageSize, imageSize,
                            fillBorder, fillBorder, fillBorder,
                            true,
                            true
                        ));

                        // 2. ToTensor
                        compose->addTransform(std::make_unique<Transforms::ToTensor>(true));  // BGR to RGB
                    }

                    // Dataset-level augmentation (training mode only)
                    if (train && enableMosaicAugmentation)
                    {
                        float mosaicProb = _config.getMosaic();
                        if (mosaicProb > 0.0f)
                        {
                            _mosaicTransform = std::make_shared<Augmentation::Mosaic>(
                                imageSize, imageSize,
                                mosaicProb,
                                fillBorder, fillBorder, fillBorder
                            );

                            // Set seed for reproducibility
                            _mosaicTransform->setSeed(_config.getSeed() + 1000);
                        }
                    }

                    return std::move(compose);
                }

                /**
                 * @brief Convert OpenCV image to LibTorch tensor
                 * @param image OpenCV image (BGR format)
                 * @return torch::Tensor Image tensor [C, H, W] in float32
                 */
                torch::Tensor imageToTensor(const cv::Mat& image)
                {
                    // Validate image
                    if (image.empty())
                    {
                        throw std::runtime_error("Cannot convert empty image to tensor");
                    }

                    // Convert to float (without normalization - ToTensor will handle that)
                    cv::Mat floatImage;
                    image.convertTo(floatImage, CV_32FC3);

                    // Ensure image data is continuous in memory for safe torch::from_blob
                    if (!floatImage.isContinuous())
                    {
                        floatImage = floatImage.clone();
                    }

                    // Create tensor directly without unnecessary copy
                    // Using torch::empty + memcpy instead of from_blob + clone
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

            protected:
                std::string _dataPath;                                      // Data directory path
                std::string _annotationPath;                                // Annotation path
                Config::Configuration _config;                       // Configuration object (const reference to avoid copy)
                std::vector<std::string> _imagePaths;                       // List of image paths
                std::unordered_map<std::string, Annotation> _annotations;   // Annotations map
                CacheType _cacheType;                                       // Cache type
                std::shared_ptr<Cache::RAMCache> _imageCache;              // RAM cache
                std::shared_ptr<Transforms::Transform> _transforms;         // Transform pipeline
                std::shared_ptr<Augmentation::Mosaic> _mosaicTransform;    // Dataset-level augmentation (optional)
                std::shared_ptr<std::mutex> _cacheMutex;
            };

        } // namespace Dataset
    } // namespace Data
} // namespace WheelDL
