#include "pch.h"
#include "Checkpoint.h"
#include "../../Utils/Error/WheelLibException.h"
#include "../../Utils/Error/ErrorCodes.h"
#include "../../Utils/Common/Constants.h"
#include "../../Utils/Logger/Logger.h"
#include "../../Utils/Path/PathValidator.h"
#include <filesystem>
#include <fstream>
#include <sstream>

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            std::string Checkpoint::getCurrentVersion()
            {
                std::ostringstream oss;
                oss << Constants::MAJOR_VERSION << "."
                    << Constants::MINOR_VERSION << "."
                    << Constants::PATCH_VERSION;
                return oss.str();
            }

            CheckpointMetadata Checkpoint::loadMetadataFromArchive(torch::serialize::InputArchive& archive)
            {
                CheckpointMetadata metadata;

                auto loadString = [&archive](const std::string& key) -> std::string {
                    torch::Tensor tensor;
                    archive.read(key, tensor);
                    // Convert tensor to string (must match save format which uses torch::kChar = int8_t)
                    auto data = tensor.data_ptr<int8_t>();
                    return std::string(reinterpret_cast<const char*>(data), tensor.numel());
                };

                metadata.wheelLibVersion = loadString("meta.wheellib_version");
                metadata.libtorchVersion = loadString("meta.libtorch_version");

                std::string taskTypeStr = loadString("meta.task_type");
                if (taskTypeStr == "Classification") {
                    metadata.taskType = TaskType::CLASSIFICATION;
                } else if (taskTypeStr == "Detection") {
                    metadata.taskType = TaskType::DETECTION;
                } else if (taskTypeStr == "Segmentation") {
                    metadata.taskType = TaskType::SEGMENTATION;
                } else if (taskTypeStr == "Anomaly") {
                    metadata.taskType = TaskType::ANOMALY;
                } else if (taskTypeStr == "OBB") {
                    metadata.taskType = TaskType::OBB;
                } else {
                    metadata.taskType = TaskType::UNKNOWN;
                }

                torch::Tensor epochTensor, fitnessTensor, gpuCountTensor;
                archive.read("meta.epoch", epochTensor);
                archive.read("meta.best_fitness", fitnessTensor);
                archive.read("meta.gpu_count", gpuCountTensor);

                metadata.epoch = epochTensor.item<int>();
                metadata.bestFitness = fitnessTensor.item<float>();
                metadata.timestamp = loadString("meta.timestamp");
                metadata.deviceType = loadString("meta.device_type");
                metadata.gpuCount = gpuCountTensor.item<int>();

                // Load metrics data (optional for backward compatibility)
                torch::Tensor lossTensor, accuracyTensor, precisionTensor, recallTensor;
                torch::Tensor f1ScoreTensor, mAPTensor, thresholdTensor;

                if (archive.try_read("meta.metrics.loss", lossTensor)) {
                    metadata.loss = lossTensor.item<float>();
                } else {
                    metadata.loss = 0.0f;
                }

                if (archive.try_read("meta.metrics.accuracy", accuracyTensor)) {
                    metadata.accuracy = accuracyTensor.item<float>();
                } else {
                    metadata.accuracy = 0.0f;
                }

                if (archive.try_read("meta.metrics.precision", precisionTensor)) {
                    metadata.precision = precisionTensor.item<float>();
                } else {
                    metadata.precision = 0.0f;
                }

                if (archive.try_read("meta.metrics.recall", recallTensor)) {
                    metadata.recall = recallTensor.item<float>();
                } else {
                    metadata.recall = 0.0f;
                }

                if (archive.try_read("meta.metrics.f1Score", f1ScoreTensor)) {
                    metadata.f1Score = f1ScoreTensor.item<float>();
                } else {
                    metadata.f1Score = 0.0f;
                }

                if (archive.try_read("meta.metrics.mAP", mAPTensor)) {
                    metadata.mAP = mAPTensor.item<float>();
                } else {
                    metadata.mAP = 0.0f;
                }

                if (archive.try_read("meta.metrics.threshold", thresholdTensor)) {
                    metadata.threshold = thresholdTensor.item<float>();
                } else {
                    metadata.threshold = 0.0f;
                }

                // Load hyperparameters (optional)
                const std::vector<std::string> hyperParamKeys = {
                    "epochs", "batch_size", "image_size", "optimizer",
                    "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
                    "amp", "cos_lr"
                };

                for (const auto& key : hyperParamKeys) {
                    try {
                        metadata.hyperParams[key] = loadString("meta.hyper." + key);
                    } catch (...) {
                        // Skip if key doesn't exist
                    }
                }

                return metadata;
            }

            void Checkpoint::save(
                const std::string& path,
                const Model::BaseModel& model,
                const torch::optim::Optimizer& optimizer,
                const CheckpointMetadata& metadata)
            {
                auto logger = Utils::Logger::getInstance();
                logger->info("Checkpoint", "Saving checkpoint to: " + path);

                try {
                    // Create parent directory if it doesn't exist
                    std::filesystem::path checkpointPath(path);
                    if (checkpointPath.has_parent_path()) {
                        std::string parentPath = checkpointPath.parent_path().string();
                        if (!Utils::PathValidator::createDirectoryIfNotExists(parentPath)) {
                            throw Utils::ConfigurationException(
                                Utils::ErrorCode::FILE_IO_ERROR,
                                "Failed to create checkpoint directory: " + parentPath
                            );
                        }
                    }

                    // Create a dictionary to hold all checkpoint data
                    torch::serialize::OutputArchive archive;

                    // Save metadata fields as individual tensors
                    auto saveString = [&archive](const std::string& key, const std::string& value) {
                        std::vector<char> chars(value.begin(), value.end());
                        torch::Tensor tensor = torch::from_blob(
                            chars.data(),
                            {static_cast<int64_t>(chars.size())},
                            torch::kChar
                        ).clone();
                        archive.write(key, tensor);
                    };

                    saveString("meta.wheellib_version", metadata.wheelLibVersion);
                    saveString("meta.libtorch_version", metadata.libtorchVersion);
                    saveString("meta.task_type", taskTypeToString(metadata.taskType));
                    archive.write("meta.epoch", torch::tensor(metadata.epoch));
                    archive.write("meta.best_fitness", torch::tensor(metadata.bestFitness));
                    saveString("meta.timestamp", metadata.timestamp);
                    saveString("meta.device_type", metadata.deviceType);
                    archive.write("meta.gpu_count", torch::tensor(metadata.gpuCount));

                    // Save metrics data
                    archive.write("meta.metrics.loss", torch::tensor(metadata.loss));
                    archive.write("meta.metrics.accuracy", torch::tensor(metadata.accuracy));
                    archive.write("meta.metrics.precision", torch::tensor(metadata.precision));
                    archive.write("meta.metrics.recall", torch::tensor(metadata.recall));
                    archive.write("meta.metrics.f1Score", torch::tensor(metadata.f1Score));
                    archive.write("meta.metrics.mAP", torch::tensor(metadata.mAP));
                    archive.write("meta.metrics.threshold", torch::tensor(metadata.threshold));

                    // Save hyperparameters
                    for (const auto& [key, value] : metadata.hyperParams) {
                        saveString("meta.hyper." + key, value);
                    }

                    // Save model state dict
                    auto modelState = model.getModel()->named_parameters();
                    for (const auto& param : modelState) {
                        std::string modelKey = std::string(MODEL_KEY) + "." + param.key();
                        archive.write(modelKey, param.value());
                    }

                    // Save model buffers (e.g., BatchNorm running stats, memory bank)
                    auto modelBuffers = model.getModel()->named_buffers();
                    for (const auto& buffer : modelBuffers) {
                        std::string modelKey = std::string(MODEL_KEY) + ".buffer." + buffer.key();
                        archive.write(modelKey, buffer.value());
                    }

                    // Save model stride
                    archive.write(std::string(MODEL_KEY) + ".stride", model.getStride());

                    // Save optimizer state
                    torch::serialize::OutputArchive optimizerArchive;
                    optimizer.save(optimizerArchive);
                    archive.write(OPTIMIZER_KEY, optimizerArchive);

                    // Write to file
                    archive.save_to(path);

                    logger->info("Checkpoint", "Successfully saved checkpoint (epoch=" +
                                std::to_string(metadata.epoch) + ", fitness=" +
                                std::to_string(metadata.bestFitness) + ")");
                }
                catch (const std::exception& e) {
                    logger->error("Checkpoint", "Failed to save checkpoint: " + std::string(e.what()));
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::FILE_IO_ERROR,
                        "Failed to save checkpoint: " + std::string(e.what())
                    );
                }
            }

            void Checkpoint::saveModelOnly(
                const std::string& path,
                const Model::BaseModel& model,
                const CheckpointMetadata& metadata)
            {
                auto logger = Utils::Logger::getInstance();
                logger->info("Checkpoint", "Saving model-only checkpoint to: " + path);

                try {
                    // Create parent directory if it doesn't exist
                    std::filesystem::path checkpointPath(path);
                    if (checkpointPath.has_parent_path()) {
                        std::string parentPath = checkpointPath.parent_path().string();
                        if (!Utils::PathValidator::createDirectoryIfNotExists(parentPath)) {
                            throw Utils::ConfigurationException(
                                Utils::ErrorCode::FILE_IO_ERROR,
                                "Failed to create checkpoint directory: " + parentPath
                            );
                        }
                    }

                    torch::serialize::OutputArchive archive;

                    // Save metadata fields as individual tensors
                    auto saveString = [&archive](const std::string& key, const std::string& value) {
                        std::vector<char> chars(value.begin(), value.end());
                        torch::Tensor tensor = torch::from_blob(
                            chars.data(),
                            {static_cast<int64_t>(chars.size())},
                            torch::kChar
                        ).clone();
                        archive.write(key, tensor);
                    };

                    saveString("meta.wheellib_version", metadata.wheelLibVersion);
                    saveString("meta.libtorch_version", metadata.libtorchVersion);
                    saveString("meta.task_type", taskTypeToString(metadata.taskType));
                    archive.write("meta.epoch", torch::tensor(metadata.epoch));
                    archive.write("meta.best_fitness", torch::tensor(metadata.bestFitness));
                    saveString("meta.timestamp", metadata.timestamp);
                    saveString("meta.device_type", metadata.deviceType);
                    archive.write("meta.gpu_count", torch::tensor(metadata.gpuCount));

                    // Save metrics data
                    archive.write("meta.metrics.loss", torch::tensor(metadata.loss));
                    archive.write("meta.metrics.accuracy", torch::tensor(metadata.accuracy));
                    archive.write("meta.metrics.precision", torch::tensor(metadata.precision));
                    archive.write("meta.metrics.recall", torch::tensor(metadata.recall));
                    archive.write("meta.metrics.f1Score", torch::tensor(metadata.f1Score));
                    archive.write("meta.metrics.mAP", torch::tensor(metadata.mAP));
                    archive.write("meta.metrics.threshold", torch::tensor(metadata.threshold));

                    // Save hyperparameters
                    for (const auto& [key, value] : metadata.hyperParams) {
                        saveString("meta.hyper." + key, value);
                    }

                    // Save model state dict
                    auto modelState = model.getModel()->named_parameters();
                    for (const auto& param : modelState) {
                        std::string modelKey = std::string(MODEL_KEY) + "." + param.key();
                        archive.write(modelKey, param.value());
                    }

                    // Save model buffers (e.g., BatchNorm running stats, memory bank)
                    auto modelBuffers = model.getModel()->named_buffers();
                    for (const auto& buffer : modelBuffers) {
                        std::string modelKey = std::string(MODEL_KEY) + ".buffer." + buffer.key();
                        archive.write(modelKey, buffer.value());
                    }

                    // Save model stride
                    archive.write(std::string(MODEL_KEY) + ".stride", model.getStride());

                    // Note: Optimizer state is NOT saved for model-only checkpoint

                    // Write to file
                    archive.save_to(path);

                    logger->info("Checkpoint", "Successfully saved model-only checkpoint (epoch=" +
                                std::to_string(metadata.epoch) + ", fitness=" +
                                std::to_string(metadata.bestFitness) + ")");
                }
                catch (const std::exception& e) {
                    logger->error("Checkpoint", "Failed to save model-only checkpoint: " + std::string(e.what()));
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::FILE_IO_ERROR,
                        "Failed to save model-only checkpoint: " + std::string(e.what())
                    );
                }
            }

            CheckpointMetadata Checkpoint::load(
                const std::string& path,
                Model::BaseModel& model,
                torch::optim::Optimizer& optimizer)
            {
                auto logger = Utils::Logger::getInstance();
                logger->info("Checkpoint", "Loading checkpoint from: " + path);

                try {
                    // Check if file exists
                    if (!Utils::PathValidator::fileExists(path)) {
                        logger->error("Checkpoint", "Checkpoint file not found: " + path);
                        throw Utils::ConfigurationException(
                            Utils::ErrorCode::FILE_IO_ERROR,
                            "Checkpoint file not found: " + path
                        );
                    }

                    // Load checkpoint
                    torch::serialize::InputArchive archive;
                    archive.load_from(path);

                    // Load metadata using helper function
                    CheckpointMetadata metadata = loadMetadataFromArchive(archive);

                    // Check compatibility
                    if (!isCompatible(metadata)) {
                        throw Utils::ConfigurationException(
                            Utils::ErrorCode::INVALID_CONFIG,
                            "Checkpoint version incompatible. Checkpoint version: " +
                            metadata.wheelLibVersion + ", Current version: " + getCurrentVersion()
                        );
                    }

                    // Load model state dict (disable gradient for in-place copy)
                    {
                        torch::NoGradGuard noGrad;
                        auto modelParams = model.getModel()->named_parameters();
                        for (auto& param : modelParams) {
                            std::string modelKey = std::string(MODEL_KEY) + "." + param.key();
                            torch::Tensor loadedParam;
                            archive.read(modelKey, loadedParam);
                            param.value().copy_(loadedParam);
                        }
                    }

                    // Load model buffers
                    auto modelBuffers = model.getModel()->named_buffers();
                    for (auto& buffer : modelBuffers) {
                        std::string modelKey = std::string(MODEL_KEY) + ".buffer." + buffer.key();
                        torch::Tensor loadedBuffer;
                        if (archive.try_read(modelKey, loadedBuffer)) {
                            // Move loaded buffer to same device as target buffer before set_()
                            auto targetDevice = buffer.value().device();
                            loadedBuffer = loadedBuffer.to(targetDevice);

                            // Use set_() instead of copy_() to handle dynamic buffer sizes (e.g., memory_bank)
                            buffer.value().set_(loadedBuffer);

                            // Log buffer info for debugging
                            std::string sizeStr = "[";
                            for (int64_t i = 0; i < loadedBuffer.dim(); ++i) {
                                if (i > 0) sizeStr += ", ";
                                sizeStr += std::to_string(loadedBuffer.size(i));
                            }
                            sizeStr += "]";
                            logger->info("Checkpoint", "Loaded buffer: " + buffer.key() + " size=" + sizeStr);
                        }
                    }

                    // Load model stride (if exists)
                    torch::Tensor loadedStride;
                    if (archive.try_read(std::string(MODEL_KEY) + ".stride", loadedStride)) {
                        model.setStride(loadedStride);
                        logger->info("Checkpoint", "Loaded stride from checkpoint");
                    }

                    // Load optimizer state
                    torch::serialize::InputArchive optimizerArchive;
                    archive.read(OPTIMIZER_KEY, optimizerArchive);
                    optimizer.load(optimizerArchive);

                    logger->info("Checkpoint", "Successfully loaded checkpoint (epoch=" +
                                std::to_string(metadata.epoch) + ", version=" +
                                metadata.wheelLibVersion + ")");

                    return metadata;
                }
                catch (const Utils::WheelLibException&) {
                    throw;
                }
                catch (const std::exception& e) {
                    logger->error("Checkpoint", "Failed to load checkpoint: " + std::string(e.what()));
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::FILE_IO_ERROR,
                        "Failed to load checkpoint: " + std::string(e.what())
                    );
                }
            }

            CheckpointMetadata Checkpoint::loadModelOnly(
                const std::string& path,
                Model::BaseModel& model)
            {
                auto logger = Utils::Logger::getInstance();
                logger->info("Checkpoint", "Loading model from checkpoint: " + path);

                try {
                    // Check if file exists
                    if (!Utils::PathValidator::fileExists(path)) {
                        logger->error("Checkpoint", "Checkpoint file not found: " + path);
                        throw Utils::ConfigurationException(
                            Utils::ErrorCode::FILE_IO_ERROR,
                            "Checkpoint file not found: " + path
                        );
                    }

                    // Load checkpoint
                    torch::serialize::InputArchive archive;
                    archive.load_from(path);

                    // Load metadata using helper function
                    CheckpointMetadata metadata = loadMetadataFromArchive(archive);

                    // Check compatibility
                    if (!isCompatible(metadata))
                    {
                        throw Utils::ConfigurationException(
                            Utils::ErrorCode::INVALID_CONFIG,
                            "Checkpoint version incompatible. Checkpoint version: " +
                            metadata.wheelLibVersion + ", Current version: " + getCurrentVersion()
                        );
                    }

                    // Load model state dict (disable gradient for in-place copy)
                    {
                        torch::NoGradGuard noGrad;
                        auto modelParams = model.getModel()->named_parameters();
                        for (auto& param : modelParams) {
                            std::string modelKey = std::string(MODEL_KEY) + "." + param.key();
                            torch::Tensor loadedParam;
                            archive.read(modelKey, loadedParam);
                            try
                            {
                                param.value().copy_(loadedParam);
                            }
                            catch (const std::exception& e)
                            {
                                logger->error("Checkpoint", "Failed to load parameter: " + param.key() + " - " + e.what());
							}
                        }
                    }

                    // Load model buffers
                    auto modelBuffers = model.getModel()->named_buffers();
                    for (auto& buffer : modelBuffers) {
                        std::string modelKey = std::string(MODEL_KEY) + ".buffer." + buffer.key();
                        torch::Tensor loadedBuffer;
                        if (archive.try_read(modelKey, loadedBuffer)) {
                            // Move loaded buffer to same device as target buffer before set_()
                            auto targetDevice = buffer.value().device();
                            loadedBuffer = loadedBuffer.to(targetDevice);

                            // Use set_() instead of copy_() to handle dynamic buffer sizes (e.g., memory_bank)
                            buffer.value().set_(loadedBuffer);
                        }
                    }

                    // Load model stride (if exists)
                    torch::Tensor loadedStride;
                    if (archive.try_read(std::string(MODEL_KEY) + ".stride", loadedStride)) {
                        model.setStride(loadedStride);
                        logger->info("Checkpoint", "Loaded stride from checkpoint");
                    }

                    logger->info("Checkpoint", "Successfully loaded model (epoch=" +
                                std::to_string(metadata.epoch) + ", version=" +
                                metadata.wheelLibVersion + ")");

                    return metadata;
                }
                catch (const Utils::WheelLibException&) {
                    throw;
                }
                catch (const std::exception& e) {
                    logger->error("Checkpoint", "Failed to load model: " + std::string(e.what()));
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::FILE_IO_ERROR,
                        "Failed to load model: " + std::string(e.what())
                    );
                }
            }

            bool Checkpoint::isCompatible(const CheckpointMetadata& metadata)
            {
                // Simple version check - can be extended with semantic versioning
                // For now, accept same major version
                std::string currentVersion = getCurrentVersion();
                std::string checkpointVersion = metadata.wheelLibVersion;

                // Extract major version (first number before '.')
                auto getMajorVersion = [](const std::string& version) -> int {
                    size_t dotPos = version.find('.');
                    if (dotPos != std::string::npos) {
                        return std::stoi(version.substr(0, dotPos));
                    }
                    return std::stoi(version);
                };

                try {
                    int currentMajor = getMajorVersion(currentVersion);
                    int checkpointMajor = getMajorVersion(checkpointVersion);

                    return currentMajor == checkpointMajor;
                }
                catch (...) {
                    // If version parsing fails, assume incompatible
                    return false;
                }
            }

            CheckpointMetadata Checkpoint::loadMetadata(const std::string& path)
            {
                try {
                    if (!Utils::PathValidator::fileExists(path)) {
                        throw Utils::ConfigurationException(
                            Utils::ErrorCode::FILE_IO_ERROR,
                            "Checkpoint file not found: " + path
                        );
                    }

                    torch::serialize::InputArchive archive;
                    archive.load_from(path);

                    // Load metadata using helper function
                    return loadMetadataFromArchive(archive);
                }
                catch (const Utils::WheelLibException&) {
                    throw;
                }
                catch (const std::exception& e) {
                    throw Utils::ConfigurationException(
                        Utils::ErrorCode::FILE_IO_ERROR,
                        "Failed to load metadata: " + std::string(e.what())
                    );
                }
            }

            bool Checkpoint::validate(const std::string& path)
            {
                try {
                    if (!Utils::PathValidator::fileExists(path)) {
                        return false;
                    }

                    // Try to load metadata - if successful, checkpoint is valid
                    loadMetadata(path);
                    return true;
                }
                catch (...) {
                    return false;
                }
            }

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
