#pragma once

#include <string>
#include <unordered_map>
#include "../../Utils/Common/Types.h"
#include "../../Config/Configuration.h"

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            /**
             * @struct CheckpointMetadata
             * @brief Metadata stored with checkpoints for version tracking and reproducibility
             *
             * Contains version information, training state, and hyperparameters
             * to ensure checkpoint compatibility and enable experiment reproduction.
             */
            struct CheckpointMetadata {
                std::string wheelLibVersion;
                std::string libtorchVersion;
                TaskType taskType;
                int epoch;
                float bestFitness;
                std::unordered_map<std::string, std::string> hyperParams;
                std::string timestamp;
                std::string deviceType;
                int gpuCount;

                /**
                 * @brief Create metadata from Configuration and training state
                 * @param config Configuration object
                 * @param epoch Current epoch
                 * @param fitness Best fitness achieved
                 * @return CheckpointMetadata object
                 */
                static CheckpointMetadata fromConfiguration(
                    const Config::Configuration& config,
                    int epoch,
                    float fitness
                );

            };

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
