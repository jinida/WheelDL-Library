#include "pch.h"
#include "CheckpointMetadata.h"
#include "../../Utils/Common/Constants.h"
#include <sstream>
#include <iomanip>
#include <chrono>
#include <torch/torch.h>

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            CheckpointMetadata CheckpointMetadata::fromConfiguration(
                const Config::Configuration& config,
                int epoch,
                float fitness)
            {
                CheckpointMetadata metadata;

                // Version information
                std::ostringstream versionStream;
                versionStream << Constants::MAJOR_VERSION << "."
                             << Constants::MINOR_VERSION << "."
                             << Constants::PATCH_VERSION;
                metadata.wheelLibVersion = versionStream.str();
                metadata.libtorchVersion = std::to_string(TORCH_VERSION_MAJOR) + "." +
                                          std::to_string(TORCH_VERSION_MINOR) + "." +
                                          std::to_string(TORCH_VERSION_PATCH);

                // Task information
                metadata.taskType = config.getTaskType();
                metadata.epoch = epoch;
                metadata.bestFitness = fitness;

                // Hyperparameters from configuration
                metadata.hyperParams["epochs"] = std::to_string(config.getEpochs());
                metadata.hyperParams["batch_size"] = std::to_string(config.getBatchSize());
                metadata.hyperParams["image_size"] = std::to_string(config.getImageSize());
                metadata.hyperParams["optimizer"] = config.getOptimizer();
                metadata.hyperParams["lr0"] = std::to_string(config.getLearningRate());
                metadata.hyperParams["lrf"] = std::to_string(config.getLRFinalFraction());
                metadata.hyperParams["momentum"] = std::to_string(config.getMomentum());
                metadata.hyperParams["weight_decay"] = std::to_string(config.getWeightDecay());
                metadata.hyperParams["warmup_epochs"] = std::to_string(config.getWarmupEpochs());
                metadata.hyperParams["amp"] = config.useAMP() ? "true" : "false";
                metadata.hyperParams["cos_lr"] = config.useCosineLR() ? "true" : "false";

                // Timestamp (ISO 8601 format)
                auto now = std::chrono::system_clock::now();
                auto time_t_now = std::chrono::system_clock::to_time_t(now);
                std::tm tm_now;
                #ifdef _WIN32
                    localtime_s(&tm_now, &time_t_now);
                #else
                    localtime_r(&time_t_now, &tm_now);
                #endif

                std::ostringstream timestampStream;
                timestampStream << std::put_time(&tm_now, "%Y-%m-%dT%H:%M:%S");
                metadata.timestamp = timestampStream.str();

                // Device information
                std::string deviceStr = config.getDevice();
                if (deviceStr.empty() || deviceStr == "cpu") {
                    metadata.deviceType = "cpu";
                    metadata.gpuCount = 0;
                } else {
                    metadata.deviceType = "cuda";
                    // Count GPUs from device string (e.g., "0,1,2" = 3 GPUs)
                    metadata.gpuCount = 1;
                    for (char c : deviceStr) {
                        if (c == ',') metadata.gpuCount++;
                    }
                }

                return metadata;
            }

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
