#pragma once

/**
 * @file TestFixtures.h
 * @brief Test fixtures and helper utilities for Model/Task tests
 *
 * This file provides the base test fixture class and utility functions
 * for testing Task model classes (BaseModel, ClassificationModel, etc.)
 *
 * Phase 1 of Model_Task_Test_Plan.md
 */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

// Forward declarations
namespace WheelDL {
    namespace Config {
        class Configuration;
    }
    namespace Data {
        namespace Dataset {
            struct DataExample;
        }
    }
}

namespace WheelDL {
namespace Test {

/**
 * @class TaskModelTestFixture
 * @brief Base test fixture for Task model tests
 *
 * Provides:
 * - Consistent random seed for reproducibility
 * - Helper methods for creating mock objects
 * - Device management utilities
 */
class TaskModelTestFixture : public ::testing::Test {
protected:
    /**
     * @brief Set up test environment
     *
     * Initializes random seed for reproducibility
     */
    void SetUp() override {
        torch::manual_seed(42);
        torch::set_num_threads(1);
    }

    /**
     * @brief Clean up test environment
     */
    void TearDown() override {
        // Clean up any CUDA memory if needed
        if (torch::cuda::is_available()) {
            torch::cuda::synchronize();
        }
    }

    // ========== Assertion Helpers ==========

    /**
     * @brief Compare tensor value with expected float value
     * @param actual Tensor to check
     * @param expected Expected float value
     * @param tol Tolerance (default 1e-4)
     */
    void expectNear(const torch::Tensor& actual, float expected, float tol = 1e-4f) {
        EXPECT_NEAR(actual.item<float>(), expected, tol);
    }

    /**
     * @brief Assert tensor is finite (no NaN or Inf)
     * @param tensor Tensor to check
     * @param name Name for error message
     */
    void assertFinite(const torch::Tensor& tensor, const std::string& name = "tensor") {
        ASSERT_FALSE(tensor.isnan().any().item<bool>())
            << name << " contains NaN values";
        ASSERT_FALSE(tensor.isinf().any().item<bool>())
            << name << " contains Inf values";
    }

    /**
     * @brief Assert two tensors have the same shape
     * @param a First tensor
     * @param b Second tensor
     */
    void assertSameShape(const torch::Tensor& a, const torch::Tensor& b) {
        ASSERT_EQ(a.sizes(), b.sizes())
            << "Shape mismatch: " << a.sizes() << " vs " << b.sizes();
    }

    /**
     * @brief Assert tensor has expected shape
     * @param tensor Tensor to check
     * @param expectedShape Expected shape
     */
    void assertShape(const torch::Tensor& tensor, const std::vector<int64_t>& expectedShape) {
        ASSERT_EQ(tensor.sizes(), torch::IntArrayRef(expectedShape))
            << "Shape mismatch: got " << tensor.sizes()
            << ", expected {" << expectedShape[0];
        for (size_t i = 1; i < expectedShape.size(); ++i) {
            // Continue message is handled by ASSERT_EQ
        }
    }

    /**
     * @brief Assert loss map contains required keys
     * @param lossMap Loss map to check
     * @param requiredKeys Keys that must be present
     */
    void assertLossMapKeys(
        const std::unordered_map<std::string, torch::Tensor>& lossMap,
        const std::vector<std::string>& requiredKeys)
    {
        for (const auto& key : requiredKeys) {
            ASSERT_TRUE(lossMap.find(key) != lossMap.end())
                << "Missing required key: " << key;
        }
    }

    // ========== Device Utilities ==========

    /**
     * @brief Get available test device
     * @return CUDA device if available, CPU otherwise
     */
    torch::Device getTestDevice() const {
        return torch::cuda::is_available()
            ? torch::Device(torch::kCUDA, 0)
            : torch::Device(torch::kCPU);
    }

    /**
     * @brief Check if CUDA is available
     * @return true if CUDA is available
     */
    bool hasCuda() const {
        return torch::cuda::is_available();
    }

    // ========== Path Utilities ==========

    /**
     * @brief Get the test config directory path
     * @return Path to test configuration files
     */
    std::string getTestConfigPath() const {
        return "Config/Valid/";
    }

    /**
     * @brief Get path to test model YAML file
     * @param filename YAML filename
     * @return Full path to YAML file
     */
    std::string getModelYamlPath(const std::string& filename) const {
        return getTestConfigPath() + "model/" + filename;
    }

    /**
     * @brief Get path to test hyperparameter YAML file
     * @param filename YAML filename
     * @return Full path to YAML file
     */
    std::string getHypYamlPath(const std::string& filename) const {
        return getTestConfigPath() + "hyp/" + filename;
    }
};

} // namespace Test
} // namespace WheelDL
