#pragma once

#include <string>
#include <torch/torch.h>
#include "CheckpointMetadata.h"
#include "../../Model/Task/BaseModel.h"

namespace WheelDL {
    namespace Core {
        namespace Trainer {

            /**
             * @class Checkpoint
             * @brief Static utility class for saving and loading model checkpoints
             *
             * Provides functions to save/load model state, optimizer state,
             * and metadata in a single checkpoint file. Includes version
             * compatibility checking and validation.
             */
            class Checkpoint {
            public:
                /**
                 * @brief Save a complete checkpoint
                 * @param path Checkpoint file path (.pt extension)
                 * @param model Model to save
                 * @param optimizer Optimizer to save
                 * @param metadata Metadata to include
                 * @throws Utils::WheelLibException if save fails
                 */
                static void save(
                    const std::string& path,
                    const Model::BaseModel& model,
                    const torch::optim::Optimizer& optimizer,
                    const CheckpointMetadata& metadata
                );

                /**
                 * @brief Save model only (without optimizer state)
                 * @param path Checkpoint file path (.pt extension)
                 * @param model Model to save
                 * @param metadata Metadata to include
                 * @throws Utils::WheelLibException if save fails
                 */

                static void saveModelOnly(
                    const std::string& path,
                    const Model::BaseModel& model,
                    const CheckpointMetadata& metadata
                );

                /**
                 * @brief Load a complete checkpoint
                 * @param path Checkpoint file path
                 * @param model Model to load state into
                 * @param optimizer Optimizer to load state into
                 * @return CheckpointMetadata from the checkpoint
                 * @throws Utils::WheelLibException if load fails or incompatible
                 */
                static CheckpointMetadata load(
                    const std::string& path,
                    Model::BaseModel& model,
                    torch::optim::Optimizer& optimizer
                );

                /**
                 * @brief Load model only (without optimizer)
                 * @param path Checkpoint file path
                 * @param model Model to load state into
                 * @return CheckpointMetadata from the checkpoint
                 * @throws Utils::WheelLibException if load fails or incompatible
                 */
                static CheckpointMetadata loadModelOnly(
                    const std::string& path,
                    Model::BaseModel& model
                );

                /**
                 * @brief Check if checkpoint metadata is compatible with current version
                 * @param metadata Metadata to check
                 * @return true if compatible, false otherwise
                 */
                static bool isCompatible(const CheckpointMetadata& metadata);

                /**
                 * @brief Load only metadata from checkpoint without loading weights
                 * @param path Checkpoint file path
                 * @return CheckpointMetadata from the checkpoint
                 * @throws Utils::WheelLibException if load fails
                 */
                static CheckpointMetadata loadMetadata(const std::string& path);

                /**
                 * @brief Validate checkpoint file integrity
                 * @param path Checkpoint file path
                 * @return true if valid, false otherwise
                 */
                static bool validate(const std::string& path);

            private:
                static constexpr const char* METADATA_KEY = "metadata";
                static constexpr const char* MODEL_KEY = "model_state_dict";
                static constexpr const char* OPTIMIZER_KEY = "optimizer_state_dict";

                // Helper to get current version string
                static std::string getCurrentVersion();

                // Helper to load metadata from archive (eliminates code duplication)
                static CheckpointMetadata loadMetadataFromArchive(torch::serialize::InputArchive& archive);
            };

        } // namespace Trainer
    } // namespace Core
} // namespace WheelDL
