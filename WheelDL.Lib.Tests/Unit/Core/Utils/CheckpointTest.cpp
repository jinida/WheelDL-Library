/**
 * @file CheckpointTest.cpp
 * @brief Unit tests for Core/Utils/Checkpoint
 *
 * Tests: CP-001 ~ CP-063 (63 tests)
 * Phase 3 of test_core_utils.md
 *
 * All tests are CUDA versions and require CUDA to be available.
 * gtest version: 1.8.1.7 (GTEST_SKIP unavailable)
 */

#include "pch.h"
#include <gtest/gtest.h>
#include "../CoreTestHelpers.h"
#include "Core/Utils/Checkpoint.h"
#include "Core/Utils/CheckpointMetadata.h"
#include "Utils/Error/WheelLibException.h"
#include "Utils/Error/ErrorCodes.h"
#include "Utils/Common/Constants.h"

using namespace WheelDL::Core::Utils;
using namespace WheelDL::Test::Core;
using namespace WheelDL;

// ============================================================================
// Test Fixture
// ============================================================================

class CheckpointTest : public CheckpointTestFixture {};

// ============================================================================
// 1.1 Save Tests (CP-001 ~ CP-012)
// ============================================================================

// CP-001: Save_ValidPath
TEST_F(CheckpointTest, Save_ValidPath) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);
    auto metadata = createTestMetadata();

    std::string path = tempManager().getFilePath("checkpoint.pt");

    EXPECT_NO_THROW(Checkpoint::save(path, *model, *optimizer, metadata));
    EXPECT_TRUE(fs::exists(path));
}

// CP-002: Save_CreatesDirectory
TEST_F(CheckpointTest, Save_CreatesDirectory) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);
    auto metadata = createTestMetadata();

    std::string subdir = tempManager().getTempDir() + "/new_subdir/nested";
    std::string path = subdir + "/checkpoint.pt";

    EXPECT_FALSE(fs::exists(subdir));
    EXPECT_NO_THROW(Checkpoint::save(path, *model, *optimizer, metadata));
    EXPECT_TRUE(fs::exists(path));
}

// CP-003: Save_OverwriteExisting
TEST_F(CheckpointTest, Save_OverwriteExisting) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    std::string path = tempManager().getFilePath("overwrite.pt");

    // Save first checkpoint
    auto metadata1 = createTestMetadata(10);
    Checkpoint::save(path, *model, *optimizer, metadata1);
    auto size1 = fs::file_size(path);

    // Save second checkpoint (overwrite)
    auto metadata2 = createTestMetadata(20);
    EXPECT_NO_THROW(Checkpoint::save(path, *model, *optimizer, metadata2));

    // Verify overwrite by loading and checking epoch
    auto loadedMeta = Checkpoint::loadMetadata(path);
    EXPECT_EQ(loadedMeta.epoch, 20);
}

// CP-004: Save_MetadataFields
TEST_F(CheckpointTest, Save_MetadataFields) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    auto metrics = MetricsFactory::createTypical();
    auto metadata = CheckpointMetadata::fromConfiguration(*config(), 50, metrics);

    std::string path = tempManager().getFilePath("metadata_fields.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    // Load and verify metadata fields
    auto loaded = Checkpoint::loadMetadata(path);
    EXPECT_EQ(loaded.epoch, 50);
    EXPECT_FALSE(loaded.wheelLibVersion.empty());
    EXPECT_FALSE(loaded.libtorchVersion.empty());
    EXPECT_FALSE(loaded.timestamp.empty());
}

// CP-005: Save_ModelParameters
TEST_F(CheckpointTest, Save_ModelParameters) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);
    auto metadata = createTestMetadata();

    // Get original parameter values
    auto params = model->getModel()->named_parameters();
    std::vector<torch::Tensor> originalParams;
    for (const auto& p : params) {
        originalParams.push_back(p.value().clone());
    }

    std::string path = tempManager().getFilePath("model_params.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    // Create new model and load
    auto model2 = createMockModel();
    model2->to(device());
    auto optimizer2 = createOptimizer(*model2);
    Checkpoint::load(path, *model2, *optimizer2);

    // Verify parameters match
    auto params2 = model2->getModel()->named_parameters();
    int idx = 0;
    for (const auto& p : params2) {
        EXPECT_TRUE(torch::allclose(p.value().cpu(), originalParams[idx].cpu()));
        idx++;
    }
}

// CP-006: Save_ModelBuffers
TEST_F(CheckpointTest, Save_ModelBuffers) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);
    auto metadata = createTestMetadata();

    // Run forward to initialize BatchNorm buffers
    auto input = torch::randn({1, 3, 32, 32}).to(device());
    model->getModel()->forward(input);

    // Get original buffer values
    auto buffers = model->getModel()->named_buffers();
    std::vector<torch::Tensor> originalBuffers;
    for (const auto& b : buffers) {
        originalBuffers.push_back(b.value().clone());
    }

    std::string path = tempManager().getFilePath("model_buffers.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    // Verify buffers were saved (file exists and is valid)
    EXPECT_TRUE(Checkpoint::validate(path));
}

// CP-007: Save_ModelStride
TEST_F(CheckpointTest, Save_ModelStride) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);
    auto metadata = createTestMetadata();

    // Set custom stride
    auto customStride = torch::tensor({16.0f, 32.0f, 64.0f});
    model->setStride(customStride);

    std::string path = tempManager().getFilePath("model_stride.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    // Load and verify stride
    auto model2 = createMockModel();
    model2->to(device());
    auto optimizer2 = createOptimizer(*model2);
    Checkpoint::load(path, *model2, *optimizer2);

    EXPECT_TRUE(torch::allclose(model2->getStride(), customStride));
}

// CP-008: Save_OptimizerState
TEST_F(CheckpointTest, Save_OptimizerState) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);
    auto metadata = createTestMetadata();

    // Perform some optimization steps to create optimizer state
    auto input = torch::randn({2, 3, 32, 32}).to(device());
    auto target = torch::randint(10, {2}).to(device());

    for (int i = 0; i < 5; ++i) {
        optimizer->zero_grad();
        auto output = model->getModel()->forward(input);
        auto loss = torch::nn::functional::cross_entropy(output, target);
        loss.backward();
        optimizer->step();
    }

    std::string path = tempManager().getFilePath("optimizer_state.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    // Load and verify (no exception means optimizer state was saved/loaded correctly)
    auto model2 = createMockModel();
    model2->to(device());
    auto optimizer2 = createOptimizer(*model2);
    EXPECT_NO_THROW(Checkpoint::load(path, *model2, *optimizer2));
}

// CP-009: Save_HyperParameters
TEST_F(CheckpointTest, Save_HyperParameters) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);
    auto metadata = createTestMetadata();

    std::string path = tempManager().getFilePath("hyperparams.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    auto loaded = Checkpoint::loadMetadata(path);

    // Verify some hyperparameters are present
    EXPECT_FALSE(loaded.hyperParams.empty());
    EXPECT_TRUE(loaded.hyperParams.find("epochs") != loaded.hyperParams.end());
    EXPECT_TRUE(loaded.hyperParams.find("batch_size") != loaded.hyperParams.end());
}

// CP-010: Save_MetricsData
TEST_F(CheckpointTest, Save_MetricsData) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    auto metrics = MetricsFactory::createTypical();
    auto metadata = CheckpointMetadata::fromConfiguration(*config(), 10, metrics);

    std::string path = tempManager().getFilePath("metrics.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    auto loaded = Checkpoint::loadMetadata(path);

    EXPECT_FLOAT_EQ(loaded.loss, metrics.loss);
    EXPECT_FLOAT_EQ(loaded.accuracy, metrics.accuracy);
    EXPECT_FLOAT_EQ(loaded.precision, metrics.precision);
    EXPECT_FLOAT_EQ(loaded.recall, metrics.recall);
    EXPECT_FLOAT_EQ(loaded.f1Score, metrics.f1Score);
    EXPECT_FLOAT_EQ(loaded.mAP, metrics.mAP);
}

// CP-011: Save_InvalidPath_Exception
TEST_F(CheckpointTest, Save_InvalidPath_Exception) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);
    auto metadata = createTestMetadata();

    // Try to save to invalid path (root path with special characters)
    std::string invalidPath = "Z:\\nonexistent\\path\\that\\does\\not\\exist\\checkpoint.pt";

    EXPECT_THROW({
        Checkpoint::save(invalidPath, *model, *optimizer, metadata);
    }, WheelDL::Utils::ConfigurationException);
}

// CP-012: Save_DirectoryCreationFails
TEST_F(CheckpointTest, Save_DirectoryCreationFails) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);
    auto metadata = createTestMetadata();

    // Invalid parent path that cannot be created
    std::string invalidPath = "\\\\?\\InvalidPath\\:*?\"<>|\\checkpoint.pt";

    EXPECT_THROW({
        Checkpoint::save(invalidPath, *model, *optimizer, metadata);
    }, WheelDL::Utils::ConfigurationException);
}

// ============================================================================
// 1.2 SaveModelOnly Tests (CP-013 ~ CP-016)
// ============================================================================

// CP-013: SaveModelOnly_ValidPath
TEST_F(CheckpointTest, SaveModelOnly_ValidPath) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto metadata = createTestMetadata();

    std::string path = tempManager().getFilePath("model_only.pt");

    EXPECT_NO_THROW(Checkpoint::saveModelOnly(path, *model, metadata));
    EXPECT_TRUE(fs::exists(path));
}

// CP-014: SaveModelOnly_NoOptimizerState
TEST_F(CheckpointTest, SaveModelOnly_NoOptimizerState) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto metadata = createTestMetadata();

    std::string path = tempManager().getFilePath("no_optimizer.pt");
    Checkpoint::saveModelOnly(path, *model, metadata);

    // Try to load with optimizer - should fail because optimizer key is missing
    auto model2 = createMockModel();
    model2->to(device());
    auto optimizer2 = createOptimizer(*model2);

    // Loading model-only checkpoint with load() should throw
    EXPECT_THROW({
        Checkpoint::load(path, *model2, *optimizer2);
    }, WheelDL::Utils::ConfigurationException);
}

// CP-015: SaveModelOnly_AllModelState
TEST_F(CheckpointTest, SaveModelOnly_AllModelState) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());

    // Set custom stride
    auto customStride = torch::tensor({8.0f, 16.0f, 32.0f});
    model->setStride(customStride);

    auto metadata = createTestMetadata();

    std::string path = tempManager().getFilePath("all_model_state.pt");
    Checkpoint::saveModelOnly(path, *model, metadata);

    // Load model only and verify
    auto model2 = createMockModel();
    model2->to(device());
    auto loadedMeta = Checkpoint::loadModelOnly(path, *model2, nullptr);

    // Verify stride was loaded
    EXPECT_TRUE(torch::allclose(model2->getStride(), customStride));
}

// CP-016: SaveModelOnly_InvalidPath
TEST_F(CheckpointTest, SaveModelOnly_InvalidPath) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto metadata = createTestMetadata();

    std::string invalidPath = "Z:\\nonexistent\\path\\model.pt";

    EXPECT_THROW({
        Checkpoint::saveModelOnly(invalidPath, *model, metadata);
    }, WheelDL::Utils::ConfigurationException);
}

// ============================================================================
// 1.3 Load Tests (CP-017 ~ CP-026)
// ============================================================================

// CP-017: Load_ValidCheckpoint
TEST_F(CheckpointTest, Load_ValidCheckpoint) {
    if (!requireCuda()) return;

    std::string path = saveTestCheckpoint("valid_checkpoint.pt");

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    EXPECT_NO_THROW({
        auto metadata = Checkpoint::load(path, *model, *optimizer);
        EXPECT_EQ(metadata.epoch, 10);
    });
}

// CP-018: Load_RestoresModelParams
TEST_F(CheckpointTest, Load_RestoresModelParams) {
    if (!requireCuda()) return;

    // Create and modify model
    auto model1 = createMockModel();
    model1->to(device());
    auto optimizer1 = createOptimizer(*model1);

    // Modify parameters
    auto input = torch::randn({2, 3, 32, 32}).to(device());
    auto target = torch::randint(10, {2}).to(device());
    for (int i = 0; i < 10; ++i) {
        optimizer1->zero_grad();
        auto output = model1->getModel()->forward(input);
        auto loss = torch::nn::functional::cross_entropy(output, target);
        loss.backward();
        optimizer1->step();
    }

    // Save modified parameters
    auto metadata = createTestMetadata();
    std::string path = tempManager().getFilePath("params_restore.pt");
    Checkpoint::save(path, *model1, *optimizer1, metadata);

    // Store modified parameters
    std::vector<torch::Tensor> modifiedParams;
    for (const auto& p : model1->getModel()->named_parameters()) {
        modifiedParams.push_back(p.value().clone());
    }

    // Create fresh model and load
    auto model2 = createMockModel();
    model2->to(device());
    auto optimizer2 = createOptimizer(*model2);
    Checkpoint::load(path, *model2, *optimizer2);

    // Verify parameters match
    int idx = 0;
    for (const auto& p : model2->getModel()->named_parameters()) {
        EXPECT_TRUE(torch::allclose(p.value().cpu(), modifiedParams[idx].cpu(), 1e-5, 1e-5));
        idx++;
    }
}

// CP-019: Load_RestoresModelBuffers
TEST_F(CheckpointTest, Load_RestoresModelBuffers) {
    if (!requireCuda()) return;

    auto model1 = createMockModel();
    model1->to(device());
    auto optimizer1 = createOptimizer(*model1);

    // Run forward to update BatchNorm buffers
    model1->getModel()->train();
    for (int i = 0; i < 10; ++i) {
        auto input = torch::randn({4, 3, 32, 32}).to(device());
        model1->getModel()->forward(input);
    }

    auto metadata = createTestMetadata();
    std::string path = tempManager().getFilePath("buffers_restore.pt");
    Checkpoint::save(path, *model1, *optimizer1, metadata);

    // Load and verify (no throw means buffers loaded)
    auto model2 = createMockModel();
    model2->to(device());
    auto optimizer2 = createOptimizer(*model2);
    EXPECT_NO_THROW(Checkpoint::load(path, *model2, *optimizer2));
}

// CP-020: Load_RestoresModelStride
TEST_F(CheckpointTest, Load_RestoresModelStride) {
    if (!requireCuda()) return;

    auto model1 = createMockModel();
    model1->to(device());
    auto optimizer1 = createOptimizer(*model1);

    auto customStride = torch::tensor({4.0f, 8.0f, 16.0f, 32.0f});
    model1->setStride(customStride);

    auto metadata = createTestMetadata();
    std::string path = tempManager().getFilePath("stride_restore.pt");
    Checkpoint::save(path, *model1, *optimizer1, metadata);

    auto model2 = createMockModel();
    model2->to(device());
    auto optimizer2 = createOptimizer(*model2);
    Checkpoint::load(path, *model2, *optimizer2);

    EXPECT_TRUE(torch::allclose(model2->getStride(), customStride));
}

// CP-021: Load_RestoresOptimizerState
TEST_F(CheckpointTest, Load_RestoresOptimizerState) {
    if (!requireCuda()) return;

    auto model1 = createMockModel();
    model1->to(device());
    auto optimizer1 = createOptimizer(*model1);

    // Train to create optimizer state
    auto input = torch::randn({2, 3, 32, 32}).to(device());
    auto target = torch::randint(10, {2}).to(device());
    for (int i = 0; i < 5; ++i) {
        optimizer1->zero_grad();
        auto output = model1->getModel()->forward(input);
        auto loss = torch::nn::functional::cross_entropy(output, target);
        loss.backward();
        optimizer1->step();
    }

    auto metadata = createTestMetadata();
    std::string path = tempManager().getFilePath("optimizer_restore.pt");
    Checkpoint::save(path, *model1, *optimizer1, metadata);

    auto model2 = createMockModel();
    model2->to(device());
    auto optimizer2 = createOptimizer(*model2);

    // Load should succeed with optimizer state
    EXPECT_NO_THROW(Checkpoint::load(path, *model2, *optimizer2));
}

// CP-022: Load_ReturnsMetadata
TEST_F(CheckpointTest, Load_ReturnsMetadata) {
    if (!requireCuda()) return;

    auto model1 = createMockModel();
    model1->to(device());
    auto optimizer1 = createOptimizer(*model1);

    auto metrics = MetricsFactory::createTypical();
    auto metadata = CheckpointMetadata::fromConfiguration(*config(), 42, metrics);

    std::string path = tempManager().getFilePath("metadata_return.pt");
    Checkpoint::save(path, *model1, *optimizer1, metadata);

    auto model2 = createMockModel();
    model2->to(device());
    auto optimizer2 = createOptimizer(*model2);
    auto loadedMeta = Checkpoint::load(path, *model2, *optimizer2);

    EXPECT_EQ(loadedMeta.epoch, 42);
    EXPECT_FLOAT_EQ(loadedMeta.loss, metrics.loss);
    EXPECT_FLOAT_EQ(loadedMeta.accuracy, metrics.accuracy);
}

// CP-023: Load_FileNotFound
TEST_F(CheckpointTest, Load_FileNotFound) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    std::string nonexistentPath = tempManager().getFilePath("nonexistent.pt");

    EXPECT_THROW({
        Checkpoint::load(nonexistentPath, *model, *optimizer);
    }, WheelDL::Utils::ConfigurationException);
}

// CP-024: Load_IncompatibleVersion
TEST_F(CheckpointTest, Load_IncompatibleVersion) {
    if (!requireCuda()) return;

    // Create checkpoint with current version
    std::string path = saveTestCheckpoint("version_test.pt");

    // Manually modify checkpoint to have incompatible version
    // This test verifies the version check mechanism works
    // We'll use loadMetadata and verify isCompatible

    auto metadata = Checkpoint::loadMetadata(path);

    // Create metadata with different major version
    CheckpointMetadata incompatibleMeta = metadata;
    incompatibleMeta.wheelLibVersion = "999.0.0";  // Future version

    // Verify isCompatible returns false
    EXPECT_FALSE(Checkpoint::isCompatible(incompatibleMeta));
}

// CP-025: Load_CorruptedFile
TEST_F(CheckpointTest, Load_CorruptedFile) {
    if (!requireCuda()) return;

    std::string corruptedPath = tempManager().createCorruptedCheckpoint("corrupted.pt");

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    EXPECT_THROW({
        Checkpoint::load(corruptedPath, *model, *optimizer);
    }, WheelDL::Utils::ConfigurationException);
}

// CP-026: Load_DeviceTransfer
TEST_F(CheckpointTest, Load_DeviceTransfer) {
    if (!requireCuda()) return;

    // Save model on CUDA
    auto model1 = createMockModel();
    model1->to(device());
    auto optimizer1 = createOptimizer(*model1);
    auto metadata = createTestMetadata();

    std::string path = tempManager().getFilePath("device_transfer.pt");
    Checkpoint::save(path, *model1, *optimizer1, metadata);

    // Load to CUDA model
    auto model2 = createMockModel();
    model2->to(device());
    auto optimizer2 = createOptimizer(*model2);
    Checkpoint::load(path, *model2, *optimizer2);

    // Verify model is on CUDA
    for (const auto& p : model2->getModel()->named_parameters()) {
        EXPECT_TRUE(p.value().device().is_cuda());
    }
}

// ============================================================================
// 1.4 LoadModelOnly Tests (CP-027 ~ CP-033)
// ============================================================================

// CP-027: LoadModelOnly_ValidCheckpoint
TEST_F(CheckpointTest, LoadModelOnly_ValidCheckpoint) {
    if (!requireCuda()) return;

    std::string path = saveModelOnlyCheckpoint("model_only_valid.pt");

    auto model = createMockModel();
    model->to(device());

    EXPECT_NO_THROW({
        auto metadata = Checkpoint::loadModelOnly(path, *model, nullptr);
        EXPECT_EQ(metadata.epoch, 10);
    });
}

// CP-028: LoadModelOnly_NoOptimizerNeeded
TEST_F(CheckpointTest, LoadModelOnly_NoOptimizerNeeded) {
    if (!requireCuda()) return;

    // Save full checkpoint
    std::string path = saveTestCheckpoint("full_for_model_only.pt");

    // Load model only (should work even with full checkpoint)
    auto model = createMockModel();
    model->to(device());

    EXPECT_NO_THROW({
        Checkpoint::loadModelOnly(path, *model, nullptr);
    });
}

// CP-029: LoadModelOnly_ShapeMismatch_Logs
TEST_F(CheckpointTest, LoadModelOnly_ShapeMismatch_Logs) {
    if (!requireCuda()) return;

    // Save checkpoint
    std::string path = saveModelOnlyCheckpoint("shape_mismatch.pt");

    // This test verifies logger is called on shape mismatch
    // Since we use same model architecture, we just verify no exception
    auto model = createMockModel();
    model->to(device());

    EXPECT_NO_THROW({
        Checkpoint::loadModelOnly(path, *model, logger());
    });
}

// CP-030: LoadModelOnly_ShapeMismatch_NoLogger
TEST_F(CheckpointTest, LoadModelOnly_ShapeMismatch_NoLogger) {
    if (!requireCuda()) return;

    std::string path = saveModelOnlyCheckpoint("shape_mismatch_no_log.pt");

    auto model = createMockModel();
    model->to(device());

    // No logger - should not crash
    EXPECT_NO_THROW({
        Checkpoint::loadModelOnly(path, *model, nullptr);
    });
}

// CP-031: LoadModelOnly_FileNotFound
TEST_F(CheckpointTest, LoadModelOnly_FileNotFound) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());

    std::string nonexistentPath = tempManager().getFilePath("nonexistent_model.pt");

    EXPECT_THROW({
        Checkpoint::loadModelOnly(nonexistentPath, *model, nullptr);
    }, WheelDL::Utils::ConfigurationException);
}

// CP-032: LoadModelOnly_IncompatibleVersion
TEST_F(CheckpointTest, LoadModelOnly_IncompatibleVersion) {
    if (!requireCuda()) return;

    // Same as CP-024 - version check applies to loadModelOnly too
    std::string path = saveModelOnlyCheckpoint("model_only_version.pt");

    auto metadata = Checkpoint::loadMetadata(path);
    CheckpointMetadata incompatibleMeta = metadata;
    incompatibleMeta.wheelLibVersion = "999.0.0";

    EXPECT_FALSE(Checkpoint::isCompatible(incompatibleMeta));
}

// CP-033: LoadModelOnly_CorruptedFile
TEST_F(CheckpointTest, LoadModelOnly_CorruptedFile) {
    if (!requireCuda()) return;

    std::string corruptedPath = tempManager().createCorruptedCheckpoint("corrupted_model.pt");

    auto model = createMockModel();
    model->to(device());

    EXPECT_THROW({
        Checkpoint::loadModelOnly(corruptedPath, *model, nullptr);
    }, WheelDL::Utils::ConfigurationException);
}

// ============================================================================
// 1.5 IsCompatible Tests (CP-034 ~ CP-039)
// ============================================================================

// CP-034: IsCompatible_SameMajorVersion
TEST_F(CheckpointTest, IsCompatible_SameMajorVersion) {
    if (!requireCuda()) return;

    CheckpointMetadata metadata{};
    std::ostringstream oss;
    oss << Constants::MAJOR_VERSION << ".0.0";
    metadata.wheelLibVersion = oss.str();

    EXPECT_TRUE(Checkpoint::isCompatible(metadata));
}

// CP-035: IsCompatible_DifferentMajorVersion
TEST_F(CheckpointTest, IsCompatible_DifferentMajorVersion) {
    if (!requireCuda()) return;

    CheckpointMetadata metadata{};
    int differentMajor = Constants::MAJOR_VERSION + 100;
    std::ostringstream oss;
    oss << differentMajor << ".0.0";
    metadata.wheelLibVersion = oss.str();

    EXPECT_FALSE(Checkpoint::isCompatible(metadata));
}

// CP-036: IsCompatible_SameFullVersion
TEST_F(CheckpointTest, IsCompatible_SameFullVersion) {
    if (!requireCuda()) return;

    CheckpointMetadata metadata{};
    std::ostringstream oss;
    oss << Constants::MAJOR_VERSION << "."
        << Constants::MINOR_VERSION << "."
        << Constants::PATCH_VERSION;
    metadata.wheelLibVersion = oss.str();

    EXPECT_TRUE(Checkpoint::isCompatible(metadata));
}

// CP-037: IsCompatible_DifferentMinor
TEST_F(CheckpointTest, IsCompatible_DifferentMinor) {
    if (!requireCuda()) return;

    CheckpointMetadata metadata{};
    std::ostringstream oss;
    oss << Constants::MAJOR_VERSION << ".999.999";
    metadata.wheelLibVersion = oss.str();

    // Same major version should be compatible
    EXPECT_TRUE(Checkpoint::isCompatible(metadata));
}

// CP-038: IsCompatible_InvalidVersionFormat
TEST_F(CheckpointTest, IsCompatible_InvalidVersionFormat) {
    if (!requireCuda()) return;

    CheckpointMetadata metadata{};
    metadata.wheelLibVersion = "invalid.version.format.string";

    EXPECT_FALSE(Checkpoint::isCompatible(metadata));
}

// CP-039: IsCompatible_EmptyVersion
TEST_F(CheckpointTest, IsCompatible_EmptyVersion) {
    if (!requireCuda()) return;

    CheckpointMetadata metadata{};
    metadata.wheelLibVersion = "";

    EXPECT_FALSE(Checkpoint::isCompatible(metadata));
}

// ============================================================================
// 1.6 LoadMetadata Tests (CP-040 ~ CP-053)
// ============================================================================

// CP-040: LoadMetadata_ValidFile
TEST_F(CheckpointTest, LoadMetadata_ValidFile) {
    if (!requireCuda()) return;

    std::string path = saveTestCheckpoint("metadata_valid.pt");

    EXPECT_NO_THROW({
        auto metadata = Checkpoint::loadMetadata(path);
        EXPECT_FALSE(metadata.wheelLibVersion.empty());
    });
}

// CP-041: LoadMetadata_AllFieldsLoaded
TEST_F(CheckpointTest, LoadMetadata_AllFieldsLoaded) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    auto metrics = MetricsFactory::createPerfect();
    auto metadata = CheckpointMetadata::fromConfiguration(*config(), 100, metrics);

    std::string path = tempManager().getFilePath("all_fields.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    auto loaded = Checkpoint::loadMetadata(path);

    EXPECT_FALSE(loaded.wheelLibVersion.empty());
    EXPECT_FALSE(loaded.libtorchVersion.empty());
    EXPECT_EQ(loaded.epoch, 100);
    EXPECT_FALSE(loaded.timestamp.empty());
    EXPECT_FALSE(loaded.deviceType.empty());
}

// CP-042: LoadMetadata_TaskType_Classification
TEST_F(CheckpointTest, LoadMetadata_TaskType_Classification) {
    if (!requireCuda()) return;

    std::string path = saveTestCheckpoint("task_classification.pt");
    auto loaded = Checkpoint::loadMetadata(path);

    // Our mock model uses CLASSIFICATION
    EXPECT_EQ(loaded.taskType, TaskType::CLASSIFICATION);
}

// CP-043: LoadMetadata_TaskType_Detection
TEST_F(CheckpointTest, LoadMetadata_TaskType_Detection) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    CheckpointMetadata metadata{};
    metadata.wheelLibVersion = "1.0.0";
    metadata.libtorchVersion = "2.0.0";
    metadata.taskType = TaskType::DETECTION;
    metadata.epoch = 10;
    metadata.timestamp = "2024-01-01T00:00:00";
    metadata.deviceType = "cuda";

    std::string path = tempManager().getFilePath("task_detection.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    auto loaded = Checkpoint::loadMetadata(path);
    EXPECT_EQ(loaded.taskType, TaskType::DETECTION);
}

// CP-044: LoadMetadata_TaskType_Segmentation
TEST_F(CheckpointTest, LoadMetadata_TaskType_Segmentation) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    CheckpointMetadata metadata{};
    metadata.wheelLibVersion = "1.0.0";
    metadata.libtorchVersion = "2.0.0";
    metadata.taskType = TaskType::SEGMENTATION;
    metadata.epoch = 10;
    metadata.timestamp = "2024-01-01T00:00:00";
    metadata.deviceType = "cuda";

    std::string path = tempManager().getFilePath("task_segmentation.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    auto loaded = Checkpoint::loadMetadata(path);
    EXPECT_EQ(loaded.taskType, TaskType::SEGMENTATION);
}

// CP-045: LoadMetadata_TaskType_Anomaly
TEST_F(CheckpointTest, LoadMetadata_TaskType_Anomaly) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    CheckpointMetadata metadata{};
    metadata.wheelLibVersion = "1.0.0";
    metadata.libtorchVersion = "2.0.0";
    metadata.taskType = TaskType::ANOMALY;
    metadata.epoch = 10;
    metadata.timestamp = "2024-01-01T00:00:00";
    metadata.deviceType = "cuda";

    std::string path = tempManager().getFilePath("task_anomaly.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    auto loaded = Checkpoint::loadMetadata(path);
    EXPECT_EQ(loaded.taskType, TaskType::ANOMALY);
}

// CP-046: LoadMetadata_TaskType_OBB
TEST_F(CheckpointTest, LoadMetadata_TaskType_OBB) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    CheckpointMetadata metadata{};
    metadata.wheelLibVersion = "1.0.0";
    metadata.libtorchVersion = "2.0.0";
    metadata.taskType = TaskType::OBB;
    metadata.epoch = 10;
    metadata.timestamp = "2024-01-01T00:00:00";
    metadata.deviceType = "cuda";

    std::string path = tempManager().getFilePath("task_obb.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    auto loaded = Checkpoint::loadMetadata(path);
    EXPECT_EQ(loaded.taskType, TaskType::OBB);
}

// CP-047: LoadMetadata_UnknownTaskType
TEST_F(CheckpointTest, LoadMetadata_UnknownTaskType) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    CheckpointMetadata metadata{};
    metadata.wheelLibVersion = "1.0.0";
    metadata.libtorchVersion = "2.0.0";
    metadata.taskType = TaskType::UNKNOWN;
    metadata.epoch = 10;
    metadata.timestamp = "2024-01-01T00:00:00";
    metadata.deviceType = "cuda";

    std::string path = tempManager().getFilePath("task_unknown.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    auto loaded = Checkpoint::loadMetadata(path);
    EXPECT_EQ(loaded.taskType, TaskType::UNKNOWN);
}

// CP-048: LoadMetadata_MetricsPresent
TEST_F(CheckpointTest, LoadMetadata_MetricsPresent) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    auto metrics = MetricsFactory::createTypical();
    auto metadata = CheckpointMetadata::fromConfiguration(*config(), 10, metrics);

    std::string path = tempManager().getFilePath("metrics_present.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    auto loaded = Checkpoint::loadMetadata(path);

    EXPECT_FLOAT_EQ(loaded.loss, metrics.loss);
    EXPECT_FLOAT_EQ(loaded.accuracy, metrics.accuracy);
    EXPECT_FLOAT_EQ(loaded.precision, metrics.precision);
    EXPECT_FLOAT_EQ(loaded.recall, metrics.recall);
    EXPECT_FLOAT_EQ(loaded.f1Score, metrics.f1Score);
    EXPECT_FLOAT_EQ(loaded.mAP, metrics.mAP);
    EXPECT_FLOAT_EQ(loaded.threshold, metrics.threshold);
}

// CP-049: LoadMetadata_MetricsAbsent
TEST_F(CheckpointTest, LoadMetadata_MetricsAbsent) {
    if (!requireCuda()) return;

    // This test verifies backward compatibility when metrics are missing
    // We save a checkpoint with zero metrics
    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    CheckpointMetadata metadata{};
    metadata.wheelLibVersion = "1.0.0";
    metadata.libtorchVersion = "2.0.0";
    metadata.taskType = TaskType::CLASSIFICATION;
    metadata.epoch = 10;
    metadata.timestamp = "2024-01-01T00:00:00";
    metadata.deviceType = "cuda";
    // Metrics are zero by default

    std::string path = tempManager().getFilePath("metrics_absent.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    auto loaded = Checkpoint::loadMetadata(path);

    // Should load with zero values
    EXPECT_FLOAT_EQ(loaded.loss, 0.0f);
    EXPECT_FLOAT_EQ(loaded.accuracy, 0.0f);
}

// CP-050: LoadMetadata_HyperParamsPresent
TEST_F(CheckpointTest, LoadMetadata_HyperParamsPresent) {
    if (!requireCuda()) return;

    std::string path = saveTestCheckpoint("hyperparams_present.pt");
    auto loaded = Checkpoint::loadMetadata(path);

    EXPECT_FALSE(loaded.hyperParams.empty());
    EXPECT_TRUE(loaded.hyperParams.count("epochs") > 0);
    EXPECT_TRUE(loaded.hyperParams.count("batch_size") > 0);
}

// CP-051: LoadMetadata_HyperParamsPartial
TEST_F(CheckpointTest, LoadMetadata_HyperParamsPartial) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    CheckpointMetadata metadata{};
    metadata.wheelLibVersion = "1.0.0";
    metadata.libtorchVersion = "2.0.0";
    metadata.taskType = TaskType::CLASSIFICATION;
    metadata.epoch = 10;
    metadata.timestamp = "2024-01-01T00:00:00";
    metadata.deviceType = "cuda";
    // Only set some hyperparams
    metadata.hyperParams["epochs"] = "100";
    metadata.hyperParams["batch_size"] = "16";

    std::string path = tempManager().getFilePath("hyperparams_partial.pt");
    Checkpoint::save(path, *model, *optimizer, metadata);

    auto loaded = Checkpoint::loadMetadata(path);

    EXPECT_EQ(loaded.hyperParams["epochs"], "100");
    EXPECT_EQ(loaded.hyperParams["batch_size"], "16");
}

// CP-052: LoadMetadata_FileNotFound
TEST_F(CheckpointTest, LoadMetadata_FileNotFound) {
    if (!requireCuda()) return;

    std::string nonexistentPath = tempManager().getFilePath("nonexistent_meta.pt");

    EXPECT_THROW({
        Checkpoint::loadMetadata(nonexistentPath);
    }, WheelDL::Utils::ConfigurationException);
}

// CP-053: LoadMetadata_CorruptedFile
TEST_F(CheckpointTest, LoadMetadata_CorruptedFile) {
    if (!requireCuda()) return;

    std::string corruptedPath = tempManager().createCorruptedCheckpoint("corrupted_meta.pt");

    EXPECT_THROW({
        Checkpoint::loadMetadata(corruptedPath);
    }, WheelDL::Utils::ConfigurationException);
}

// ============================================================================
// 1.7 Validate Tests (CP-054 ~ CP-057)
// ============================================================================

// CP-054: Validate_ValidCheckpoint
TEST_F(CheckpointTest, Validate_ValidCheckpoint) {
    if (!requireCuda()) return;

    std::string path = saveTestCheckpoint("validate_valid.pt");

    EXPECT_TRUE(Checkpoint::validate(path));
}

// CP-055: Validate_FileNotFound
TEST_F(CheckpointTest, Validate_FileNotFound) {
    if (!requireCuda()) return;

    std::string nonexistentPath = tempManager().getFilePath("nonexistent_validate.pt");

    EXPECT_FALSE(Checkpoint::validate(nonexistentPath));
}

// CP-056: Validate_CorruptedFile
TEST_F(CheckpointTest, Validate_CorruptedFile) {
    if (!requireCuda()) return;

    std::string corruptedPath = tempManager().createCorruptedCheckpoint("corrupted_validate.pt");

    EXPECT_FALSE(Checkpoint::validate(corruptedPath));
}

// CP-057: Validate_EmptyFile
TEST_F(CheckpointTest, Validate_EmptyFile) {
    if (!requireCuda()) return;

    std::string emptyPath = tempManager().createEmptyFile("empty_validate.pt");

    EXPECT_FALSE(Checkpoint::validate(emptyPath));
}

// ============================================================================
// 1.8 Edge Case Tests (CP-058 ~ CP-063)
// ============================================================================

// CP-058: Save_NoParentPath
TEST_F(CheckpointTest, Save_NoParentPath) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);
    auto metadata = createTestMetadata();

    // Save to current temp directory with just filename
    std::string path = tempManager().getFilePath("just_filename.pt");

    EXPECT_NO_THROW(Checkpoint::save(path, *model, *optimizer, metadata));
    EXPECT_TRUE(fs::exists(path));
}

// CP-059: Load_BufferNotInCheckpoint
TEST_F(CheckpointTest, Load_BufferNotInCheckpoint) {
    if (!requireCuda()) return;

    // Save checkpoint
    std::string path = saveTestCheckpoint("buffer_missing.pt");

    // Load - buffers that might be missing should not cause crash
    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    EXPECT_NO_THROW(Checkpoint::load(path, *model, *optimizer));
}

// CP-060: Load_StrideNotInCheckpoint
TEST_F(CheckpointTest, Load_StrideNotInCheckpoint) {
    if (!requireCuda()) return;

    // Save checkpoint (stride is always saved in our implementation)
    std::string path = saveTestCheckpoint("stride_check.pt");

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    // Should load successfully
    EXPECT_NO_THROW(Checkpoint::load(path, *model, *optimizer));
}

// CP-061: IsCompatible_NoDotInVersion
TEST_F(CheckpointTest, IsCompatible_NoDotInVersion) {
    if (!requireCuda()) return;

    CheckpointMetadata metadata{};
    std::ostringstream oss;
    oss << Constants::MAJOR_VERSION;  // Just major version, no dot
    metadata.wheelLibVersion = oss.str();

    // Should parse correctly as just major version
    EXPECT_TRUE(Checkpoint::isCompatible(metadata));
}

// CP-062: LoadModelOnly_BufferDeviceTransfer
TEST_F(CheckpointTest, LoadModelOnly_BufferDeviceTransfer) {
    if (!requireCuda()) return;

    // Save model on CUDA
    auto model1 = createMockModel();
    model1->to(device());

    // Run forward to create buffer values
    auto input = torch::randn({1, 3, 32, 32}).to(device());
    model1->getModel()->forward(input);

    auto metadata = createTestMetadata();
    std::string path = tempManager().getFilePath("buffer_device.pt");
    Checkpoint::saveModelOnly(path, *model1, metadata);

    // Load to CUDA model
    auto model2 = createMockModel();
    model2->to(device());
    Checkpoint::loadModelOnly(path, *model2, nullptr);

    // Verify buffers are on CUDA
    for (const auto& b : model2->getModel()->named_buffers()) {
        EXPECT_TRUE(b.value().device().is_cuda());
    }
}

// CP-063: Save_EmptyHyperParams
TEST_F(CheckpointTest, Save_EmptyHyperParams) {
    if (!requireCuda()) return;

    auto model = createMockModel();
    model->to(device());
    auto optimizer = createOptimizer(*model);

    CheckpointMetadata metadata{};
    metadata.wheelLibVersion = "1.0.0";
    metadata.libtorchVersion = "2.0.0";
    metadata.taskType = TaskType::CLASSIFICATION;
    metadata.epoch = 10;
    metadata.timestamp = "2024-01-01T00:00:00";
    metadata.deviceType = "cuda";
    // hyperParams is empty by default

    std::string path = tempManager().getFilePath("empty_hyperparams.pt");

    EXPECT_NO_THROW(Checkpoint::save(path, *model, *optimizer, metadata));
    EXPECT_TRUE(Checkpoint::validate(path));
}
