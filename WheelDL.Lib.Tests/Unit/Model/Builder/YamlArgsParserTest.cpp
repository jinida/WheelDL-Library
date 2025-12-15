/**
 * @file YamlArgsParserTest.cpp
 * @brief Unit tests for YamlArgsParser helper class
 *
 * Tests cover:
 * - parseIntVector() with valid/invalid inputs
 * - getInt(), getBool(), getDouble(), getString() with various scenarios
 * - Legacy methods (getIntArg, getBoolArg, getDoubleArg)
 * - Exception handling for malformed YAML
 *
 * @author WheelDL Team
 */

#include "pch.h"
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include "Model/Builder/Factory/ModuleFactory.h"

using namespace WheelDL::Model::Builder;

// ============================================================================
// YamlArgsParser Test Fixture
// ============================================================================

class YamlArgsParserTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// parseIntVector Tests
// ============================================================================

TEST_F(YamlArgsParserTest, parseIntVector_ValidSequence) {
    YAML::Node args = YAML::Load("[1, 2, 3]");

    auto result = YamlArgsParser::parseIntVector(args, "TestModule");

    ASSERT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 2);
    EXPECT_EQ(result[2], 3);
}

TEST_F(YamlArgsParserTest, parseIntVector_EmptySequence) {
    YAML::Node args = YAML::Load("[]");

    auto result = YamlArgsParser::parseIntVector(args, "TestModule");

    EXPECT_TRUE(result.empty());
}

TEST_F(YamlArgsParserTest, parseIntVector_SingleElement) {
    YAML::Node args = YAML::Load("[42]");

    auto result = YamlArgsParser::parseIntVector(args, "TestModule");

    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 42);
}

TEST_F(YamlArgsParserTest, parseIntVector_NegativeValues) {
    YAML::Node args = YAML::Load("[-1, -2, -3]");

    auto result = YamlArgsParser::parseIntVector(args, "TestModule");

    ASSERT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], -1);
    EXPECT_EQ(result[1], -2);
    EXPECT_EQ(result[2], -3);
}

TEST_F(YamlArgsParserTest, parseIntVector_LargeValues) {
    YAML::Node args = YAML::Load("[1000000, 2147483647]");

    auto result = YamlArgsParser::parseIntVector(args, "TestModule");

    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 1000000);
    EXPECT_EQ(result[1], 2147483647);
}

TEST_F(YamlArgsParserTest, parseIntVector_UndefinedNode_ThrowsException) {
    YAML::Node args;  // Undefined node

    EXPECT_THROW(
        YamlArgsParser::parseIntVector(args, "TestModule"),
        std::runtime_error
    );
}

TEST_F(YamlArgsParserTest, parseIntVector_NotSequence_ThrowsException) {
    YAML::Node args = YAML::Load("123");  // Scalar, not sequence

    EXPECT_THROW(
        YamlArgsParser::parseIntVector(args, "TestModule"),
        std::runtime_error
    );
}

TEST_F(YamlArgsParserTest, parseIntVector_MapNode_ThrowsException) {
    YAML::Node args = YAML::Load("{key: value}");  // Map, not sequence

    EXPECT_THROW(
        YamlArgsParser::parseIntVector(args, "TestModule"),
        std::runtime_error
    );
}

TEST_F(YamlArgsParserTest, parseIntVector_InvalidType_ThrowsException) {
    YAML::Node args = YAML::Load("[a, b, c]");  // Strings, not ints

    EXPECT_THROW(
        YamlArgsParser::parseIntVector(args, "TestModule"),
        std::runtime_error
    );
}

TEST_F(YamlArgsParserTest, parseIntVector_ExceptionContainsModuleName) {
    YAML::Node args;  // Undefined node

    try {
        YamlArgsParser::parseIntVector(args, "MyTestModule");
        FAIL() << "Expected exception not thrown";
    } catch (const std::runtime_error& e) {
        std::string errorMsg = e.what();
        EXPECT_TRUE(errorMsg.find("MyTestModule") != std::string::npos)
            << "Error message should contain module name. Got: " << errorMsg;
    }
}

// ============================================================================
// getInt Tests
// ============================================================================

TEST_F(YamlArgsParserTest, getInt_ValidIndex) {
    YAML::Node args = YAML::Load("[64, 128, 256]");

    EXPECT_EQ(YamlArgsParser::getInt(args, 0, 0), 64);
    EXPECT_EQ(YamlArgsParser::getInt(args, 1, 0), 128);
    EXPECT_EQ(YamlArgsParser::getInt(args, 2, 0), 256);
}

TEST_F(YamlArgsParserTest, getInt_OutOfBounds_ReturnsDefault) {
    YAML::Node args = YAML::Load("[64, 128]");

    EXPECT_EQ(YamlArgsParser::getInt(args, 5, 999), 999);
    EXPECT_EQ(YamlArgsParser::getInt(args, 100, -1), -1);
}

TEST_F(YamlArgsParserTest, getInt_EmptySequence_ReturnsDefault) {
    YAML::Node args = YAML::Load("[]");

    EXPECT_EQ(YamlArgsParser::getInt(args, 0, 42), 42);
}

TEST_F(YamlArgsParserTest, getInt_NotSequence_ReturnsDefault) {
    YAML::Node args = YAML::Load("123");  // Scalar

    EXPECT_EQ(YamlArgsParser::getInt(args, 0, 99), 99);
}

TEST_F(YamlArgsParserTest, getInt_NegativeValue) {
    YAML::Node args = YAML::Load("[-100]");

    EXPECT_EQ(YamlArgsParser::getInt(args, 0, 0), -100);
}

// ============================================================================
// getBool Tests
// ============================================================================

TEST_F(YamlArgsParserTest, getBool_True) {
    YAML::Node args = YAML::Load("[true]");

    EXPECT_TRUE(YamlArgsParser::getBool(args, 0, false));
}

TEST_F(YamlArgsParserTest, getBool_False) {
    YAML::Node args = YAML::Load("[false]");

    EXPECT_FALSE(YamlArgsParser::getBool(args, 0, true));
}

TEST_F(YamlArgsParserTest, getBool_IntAsTrue) {
    YAML::Node args = YAML::Load("[1]");

    EXPECT_TRUE(YamlArgsParser::getBool(args, 0, false));
}

TEST_F(YamlArgsParserTest, getBool_IntAsFalse) {
    YAML::Node args = YAML::Load("[0]");

    EXPECT_FALSE(YamlArgsParser::getBool(args, 0, true));
}

TEST_F(YamlArgsParserTest, getBool_OutOfBounds_ReturnsDefault) {
    YAML::Node args = YAML::Load("[true]");

    EXPECT_FALSE(YamlArgsParser::getBool(args, 5, false));
    EXPECT_TRUE(YamlArgsParser::getBool(args, 5, true));
}

TEST_F(YamlArgsParserTest, getBool_NotSequence_ReturnsDefault) {
    YAML::Node args = YAML::Load("true");  // Scalar, not sequence

    EXPECT_FALSE(YamlArgsParser::getBool(args, 0, false));
}

TEST_F(YamlArgsParserTest, getBool_MixedSequence) {
    YAML::Node args = YAML::Load("[true, false, 1, 0]");

    EXPECT_TRUE(YamlArgsParser::getBool(args, 0, false));
    EXPECT_FALSE(YamlArgsParser::getBool(args, 1, true));
    EXPECT_TRUE(YamlArgsParser::getBool(args, 2, false));
    EXPECT_FALSE(YamlArgsParser::getBool(args, 3, true));
}

// ============================================================================
// getDouble Tests
// ============================================================================

TEST_F(YamlArgsParserTest, getDouble_ValidIndex) {
    YAML::Node args = YAML::Load("[0.5, 1.5, 2.25]");

    EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(args, 0, 0.0), 0.5);
    EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(args, 1, 0.0), 1.5);
    EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(args, 2, 0.0), 2.25);
}

TEST_F(YamlArgsParserTest, getDouble_OutOfBounds_ReturnsDefault) {
    YAML::Node args = YAML::Load("[0.5]");

    EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(args, 5, 3.14), 3.14);
}

TEST_F(YamlArgsParserTest, getDouble_NotSequence_ReturnsDefault) {
    YAML::Node args = YAML::Load("0.5");  // Scalar

    EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(args, 0, 1.0), 1.0);
}

TEST_F(YamlArgsParserTest, getDouble_IntAsDouble) {
    YAML::Node args = YAML::Load("[3]");

    EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(args, 0, 0.0), 3.0);
}

TEST_F(YamlArgsParserTest, getDouble_NegativeValue) {
    YAML::Node args = YAML::Load("[-0.5]");

    EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(args, 0, 0.0), -0.5);
}

TEST_F(YamlArgsParserTest, getDouble_VerySmallValue) {
    YAML::Node args = YAML::Load("[0.0001]");

    EXPECT_NEAR(YamlArgsParser::getDouble(args, 0, 0.0), 0.0001, 1e-6);
}

// ============================================================================
// getString Tests
// ============================================================================

TEST_F(YamlArgsParserTest, getString_ValidIndex) {
    YAML::Node args = YAML::Load("[SiLU, ReLU, GELU]");

    EXPECT_EQ(YamlArgsParser::getString(args, 0, ""), "SiLU");
    EXPECT_EQ(YamlArgsParser::getString(args, 1, ""), "ReLU");
    EXPECT_EQ(YamlArgsParser::getString(args, 2, ""), "GELU");
}

TEST_F(YamlArgsParserTest, getString_OutOfBounds_ReturnsDefault) {
    YAML::Node args = YAML::Load("[hello]");

    EXPECT_EQ(YamlArgsParser::getString(args, 5, "default"), "default");
}

TEST_F(YamlArgsParserTest, getString_NotSequence_ReturnsDefault) {
    YAML::Node args = YAML::Load("hello");  // Scalar

    EXPECT_EQ(YamlArgsParser::getString(args, 0, "default"), "default");
}

TEST_F(YamlArgsParserTest, getString_EmptyString) {
    YAML::Node args = YAML::Load("['']");

    EXPECT_EQ(YamlArgsParser::getString(args, 0, "default"), "");
}

TEST_F(YamlArgsParserTest, getString_WithQuotes) {
    YAML::Node args = YAML::Load("[\"quoted string\"]");

    EXPECT_EQ(YamlArgsParser::getString(args, 0, ""), "quoted string");
}

// ============================================================================
// Legacy Methods Tests (getIntArg, getBoolArg, getDoubleArg)
// ============================================================================

TEST_F(YamlArgsParserTest, getIntArg_ValidIndex) {
    std::vector<int64_t> args = {10, 20, 30};

    EXPECT_EQ(YamlArgsParser::getIntArg(args, 0, 0), 10);
    EXPECT_EQ(YamlArgsParser::getIntArg(args, 1, 0), 20);
    EXPECT_EQ(YamlArgsParser::getIntArg(args, 2, 0), 30);
}

TEST_F(YamlArgsParserTest, getIntArg_OutOfBounds_ReturnsDefault) {
    std::vector<int64_t> args = {10, 20};

    EXPECT_EQ(YamlArgsParser::getIntArg(args, 5, 999), 999);
}

TEST_F(YamlArgsParserTest, getIntArg_EmptyVector) {
    std::vector<int64_t> args;

    EXPECT_EQ(YamlArgsParser::getIntArg(args, 0, 42), 42);
}

TEST_F(YamlArgsParserTest, getBoolArg_ValidIndex) {
    std::vector<int64_t> args = {1, 0, 1};

    EXPECT_TRUE(YamlArgsParser::getBoolArg(args, 0, false));
    EXPECT_FALSE(YamlArgsParser::getBoolArg(args, 1, true));
    EXPECT_TRUE(YamlArgsParser::getBoolArg(args, 2, false));
}

TEST_F(YamlArgsParserTest, getBoolArg_OutOfBounds_ReturnsDefault) {
    std::vector<int64_t> args = {1};

    EXPECT_FALSE(YamlArgsParser::getBoolArg(args, 5, false));
    EXPECT_TRUE(YamlArgsParser::getBoolArg(args, 5, true));
}

TEST_F(YamlArgsParserTest, getBoolArg_NonZeroAsTrue) {
    std::vector<int64_t> args = {5, -1, 100};  // Non-zero values

    EXPECT_TRUE(YamlArgsParser::getBoolArg(args, 0, false));
    EXPECT_TRUE(YamlArgsParser::getBoolArg(args, 1, false));
    EXPECT_TRUE(YamlArgsParser::getBoolArg(args, 2, false));
}

TEST_F(YamlArgsParserTest, getDoubleArg_ValidIndex) {
    std::vector<int64_t> args = {3, 5, 7};

    EXPECT_DOUBLE_EQ(YamlArgsParser::getDoubleArg(args, 0, 0.0), 3.0);
    EXPECT_DOUBLE_EQ(YamlArgsParser::getDoubleArg(args, 1, 0.0), 5.0);
    EXPECT_DOUBLE_EQ(YamlArgsParser::getDoubleArg(args, 2, 0.0), 7.0);
}

TEST_F(YamlArgsParserTest, getDoubleArg_OutOfBounds_ReturnsDefault) {
    std::vector<int64_t> args = {3};

    EXPECT_DOUBLE_EQ(YamlArgsParser::getDoubleArg(args, 5, 3.14), 3.14);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(YamlArgsParserTest, NullYamlNode) {
    YAML::Node args = YAML::Load("null");

    // null is not a sequence, so should return default
    EXPECT_EQ(YamlArgsParser::getInt(args, 0, 42), 42);
    EXPECT_FALSE(YamlArgsParser::getBool(args, 0, false));
    EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(args, 0, 1.5), 1.5);
    EXPECT_EQ(YamlArgsParser::getString(args, 0, "default"), "default");
}

TEST_F(YamlArgsParserTest, MixedTypeSequence) {
    YAML::Node args = YAML::Load("[64, true, 0.5, hello]");

    // Each getter should handle its own type at the correct index
    EXPECT_EQ(YamlArgsParser::getInt(args, 0, 0), 64);
    EXPECT_TRUE(YamlArgsParser::getBool(args, 1, false));
    EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(args, 2, 0.0), 0.5);
    EXPECT_EQ(YamlArgsParser::getString(args, 3, ""), "hello");
}
