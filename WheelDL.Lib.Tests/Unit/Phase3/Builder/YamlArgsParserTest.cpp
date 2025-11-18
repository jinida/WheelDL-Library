#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Builder/Factory/ModuleFactory.h"
#include <yaml-cpp/yaml.h>

using namespace WheelDL;
using namespace WheelDL::Model::Builder;

class YamlArgsParserTest : public ::testing::Test {
protected:
	YAML::Node createArgsNode(const std::vector<int64_t>& values) {
		YAML::Node node;
		for (const auto& val : values) {
			node.push_back(val);
		}
		return node;
	}

	YAML::Node createMixedArgsNode() {
		YAML::Node node;
		node.push_back(64);           // int
		node.push_back(true);         // bool
		node.push_back(3.14);         // double
		node.push_back("SiLU");       // string
		return node;
	}
};

// ========== parseIntVector Tests ==========

TEST_F(YamlArgsParserTest, ParseIntVector_ValidSequence_ReturnsVector) {
	auto argsNode = createArgsNode({64, 3, 2, 1});

	auto result = YamlArgsParser::parseIntVector(argsNode, "TestModule");

	ASSERT_EQ(result.size(), 4);
	EXPECT_EQ(result[0], 64);
	EXPECT_EQ(result[1], 3);
	EXPECT_EQ(result[2], 2);
	EXPECT_EQ(result[3], 1);
}

TEST_F(YamlArgsParserTest, ParseIntVector_EmptySequence_ReturnsEmpty) {
	// Create an explicitly empty YAML sequence node
	YAML::Node emptyNode(YAML::NodeType::Sequence);

	auto result = YamlArgsParser::parseIntVector(emptyNode, "TestModule");

	EXPECT_TRUE(result.empty());
}

TEST_F(YamlArgsParserTest, ParseIntVector_SingleValue_ReturnsVector) {
	auto argsNode = createArgsNode({64});

	auto result = YamlArgsParser::parseIntVector(argsNode, "TestModule");

	ASSERT_EQ(result.size(), 1);
	EXPECT_EQ(result[0], 64);
}

TEST_F(YamlArgsParserTest, ParseIntVector_UndefinedNode_ThrowsException) {
	YAML::Node undefinedNode;
	undefinedNode = YAML::Node(YAML::NodeType::Undefined);

	EXPECT_THROW({
		YamlArgsParser::parseIntVector(undefinedNode, "TestModule");
	}, std::runtime_error);
}

TEST_F(YamlArgsParserTest, ParseIntVector_NonSequenceNode_ThrowsException) {
	YAML::Node scalarNode = YAML::Load("42");

	EXPECT_THROW({
		YamlArgsParser::parseIntVector(scalarNode, "TestModule");
	}, std::runtime_error);
}

// ========== getInt Tests ==========

TEST_F(YamlArgsParserTest, GetInt_ValidIndex_ReturnsValue) {
	auto argsNode = createArgsNode({64, 128, 256});

	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 0, 0), 64);
	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 1, 0), 128);
	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 2, 0), 256);
}

TEST_F(YamlArgsParserTest, GetInt_OutOfBounds_ReturnsDefault) {
	auto argsNode = createArgsNode({64});

	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 5, 99), 99);
	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 10, 123), 123);
}

TEST_F(YamlArgsParserTest, GetInt_EmptyNode_ReturnsDefault) {
	YAML::Node emptyNode;

	EXPECT_EQ(YamlArgsParser::getInt(emptyNode, 0, 42), 42);
}

TEST_F(YamlArgsParserTest, GetInt_NonSequenceNode_ReturnsDefault) {
	YAML::Node scalarNode = YAML::Load("42");

	EXPECT_EQ(YamlArgsParser::getInt(scalarNode, 0, 99), 99);
}

TEST_F(YamlArgsParserTest, GetInt_NegativeValues_Success) {
	auto argsNode = createArgsNode({-1, -10, -100});

	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 0, 0), -1);
	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 1, 0), -10);
	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 2, 0), -100);
}

TEST_F(YamlArgsParserTest, GetInt_ZeroIndex_ReturnsFirst) {
	auto argsNode = createArgsNode({123, 456});

	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 0, 0), 123);
}

// ========== getBool Tests ==========

TEST_F(YamlArgsParserTest, GetBool_TrueValue_ReturnsTrue) {
	YAML::Node node;
	node.push_back(true);
	node.push_back(false);

	EXPECT_TRUE(YamlArgsParser::getBool(node, 0, false));
	EXPECT_FALSE(YamlArgsParser::getBool(node, 1, true));
}

TEST_F(YamlArgsParserTest, GetBool_IntegerOne_ReturnsTrue) {
	auto argsNode = createArgsNode({1, 0});

	EXPECT_TRUE(YamlArgsParser::getBool(argsNode, 0, false));
	EXPECT_FALSE(YamlArgsParser::getBool(argsNode, 1, true));
}

TEST_F(YamlArgsParserTest, GetBool_OutOfBounds_ReturnsDefault) {
	YAML::Node node;
	node.push_back(true);

	EXPECT_FALSE(YamlArgsParser::getBool(node, 5, false));
	EXPECT_TRUE(YamlArgsParser::getBool(node, 10, true));
}

TEST_F(YamlArgsParserTest, GetBool_EmptyNode_ReturnsDefault) {
	YAML::Node emptyNode;

	EXPECT_TRUE(YamlArgsParser::getBool(emptyNode, 0, true));
	EXPECT_FALSE(YamlArgsParser::getBool(emptyNode, 0, false));
}

TEST_F(YamlArgsParserTest, GetBool_NonSequenceNode_ReturnsDefault) {
	YAML::Node scalarNode = YAML::Load("true");

	EXPECT_TRUE(YamlArgsParser::getBool(scalarNode, 0, true));
}

// ========== getDouble Tests ==========

TEST_F(YamlArgsParserTest, GetDouble_ValidIndex_ReturnsValue) {
	YAML::Node node;
	node.push_back(3.14);
	node.push_back(2.71);
	node.push_back(1.41);

	EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(node, 0, 0.0), 3.14);
	EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(node, 1, 0.0), 2.71);
	EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(node, 2, 0.0), 1.41);
}

TEST_F(YamlArgsParserTest, GetDouble_OutOfBounds_ReturnsDefault) {
	YAML::Node node;
	node.push_back(3.14);

	EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(node, 5, 9.99), 9.99);
}

TEST_F(YamlArgsParserTest, GetDouble_EmptyNode_ReturnsDefault) {
	YAML::Node emptyNode;

	EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(emptyNode, 0, 1.23), 1.23);
}

TEST_F(YamlArgsParserTest, GetDouble_IntegerValue_ConvertsToDouble) {
	auto argsNode = createArgsNode({42});

	EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(argsNode, 0, 0.0), 42.0);
}

TEST_F(YamlArgsParserTest, GetDouble_NegativeValues_Success) {
	YAML::Node node;
	node.push_back(-3.14);
	node.push_back(-2.71);

	EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(node, 0, 0.0), -3.14);
	EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(node, 1, 0.0), -2.71);
}

TEST_F(YamlArgsParserTest, GetDouble_ZeroValue_Success) {
	YAML::Node node;
	node.push_back(0.0);

	EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(node, 0, 1.0), 0.0);
}

// ========== getString Tests ==========

TEST_F(YamlArgsParserTest, getString_ValidIndex_ReturnsValue) {
	YAML::Node node;
	node.push_back("SiLU");
	node.push_back("ReLU");
	node.push_back("GELU");

	EXPECT_EQ(YamlArgsParser::getString(node, 0, ""), "SiLU");
	EXPECT_EQ(YamlArgsParser::getString(node, 1, ""), "ReLU");
	EXPECT_EQ(YamlArgsParser::getString(node, 2, ""), "GELU");
}

TEST_F(YamlArgsParserTest, GetString_OutOfBounds_ReturnsDefault) {
	YAML::Node node;
	node.push_back("SiLU");

	EXPECT_EQ(YamlArgsParser::getString(node, 5, "default"), "default");
}

TEST_F(YamlArgsParserTest, GetString_EmptyNode_ReturnsDefault) {
	YAML::Node emptyNode;

	EXPECT_EQ(YamlArgsParser::getString(emptyNode, 0, "default"), "default");
}

TEST_F(YamlArgsParserTest, GetString_EmptyString_Success) {
	YAML::Node node;
	node.push_back("");

	EXPECT_EQ(YamlArgsParser::getString(node, 0, "default"), "");
}

TEST_F(YamlArgsParserTest, GetString_NumericString_Success) {
	YAML::Node node;
	node.push_back("123");
	node.push_back("3.14");

	EXPECT_EQ(YamlArgsParser::getString(node, 0, ""), "123");
	EXPECT_EQ(YamlArgsParser::getString(node, 1, ""), "3.14");
}

// ========== Legacy Vector Methods Tests ==========

TEST_F(YamlArgsParserTest, GetIntArg_ValidIndex_ReturnsValue) {
	std::vector<int64_t> args = {64, 128, 256};

	EXPECT_EQ(YamlArgsParser::getIntArg(args, 0, 0), 64);
	EXPECT_EQ(YamlArgsParser::getIntArg(args, 1, 0), 128);
	EXPECT_EQ(YamlArgsParser::getIntArg(args, 2, 0), 256);
}

TEST_F(YamlArgsParserTest, GetIntArg_OutOfBounds_ReturnsDefault) {
	std::vector<int64_t> args = {64};

	EXPECT_EQ(YamlArgsParser::getIntArg(args, 5, 99), 99);
}

TEST_F(YamlArgsParserTest, GetBoolArg_ValidIndex_ReturnsValue) {
	std::vector<int64_t> args = {1, 0, 1};

	EXPECT_TRUE(YamlArgsParser::getBoolArg(args, 0, false));
	EXPECT_FALSE(YamlArgsParser::getBoolArg(args, 1, true));
	EXPECT_TRUE(YamlArgsParser::getBoolArg(args, 2, false));
}

TEST_F(YamlArgsParserTest, GetBoolArg_OutOfBounds_ReturnsDefault) {
	std::vector<int64_t> args = {1};

	EXPECT_FALSE(YamlArgsParser::getBoolArg(args, 5, false));
	EXPECT_TRUE(YamlArgsParser::getBoolArg(args, 10, true));
}

TEST_F(YamlArgsParserTest, GetDoubleArg_ValidIndex_ReturnsValue) {
	std::vector<int64_t> args = {3, 7, 10};

	EXPECT_DOUBLE_EQ(YamlArgsParser::getDoubleArg(args, 0, 0.0), 3.0);
	EXPECT_DOUBLE_EQ(YamlArgsParser::getDoubleArg(args, 1, 0.0), 7.0);
	EXPECT_DOUBLE_EQ(YamlArgsParser::getDoubleArg(args, 2, 0.0), 10.0);
}

TEST_F(YamlArgsParserTest, GetDoubleArg_OutOfBounds_ReturnsDefault) {
	std::vector<int64_t> args = {42};

	EXPECT_DOUBLE_EQ(YamlArgsParser::getDoubleArg(args, 5, 9.99), 9.99);
}

// ========== Mixed Type Tests ==========

TEST_F(YamlArgsParserTest, MixedTypes_AllGetters_WorkCorrectly) {
	auto node = createMixedArgsNode();

	EXPECT_EQ(YamlArgsParser::getInt(node, 0, 0), 64);
	EXPECT_TRUE(YamlArgsParser::getBool(node, 1, false));
	EXPECT_DOUBLE_EQ(YamlArgsParser::getDouble(node, 2, 0.0), 3.14);
	EXPECT_EQ(YamlArgsParser::getString(node, 3, ""), "SiLU");
}

// ========== Edge Cases Tests ==========

TEST_F(YamlArgsParserTest, GetInt_LargeValues_Success) {
	auto argsNode = createArgsNode({1024, 2048, 4096});

	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 0, 0), 1024);
	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 1, 0), 2048);
	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 2, 0), 4096);
}

TEST_F(YamlArgsParserTest, GetDouble_VerySmallValues_Success) {
	YAML::Node node;
	node.push_back(0.0001);
	node.push_back(0.00001);

	EXPECT_NEAR(YamlArgsParser::getDouble(node, 0, 0.0), 0.0001, 1e-10);
	EXPECT_NEAR(YamlArgsParser::getDouble(node, 1, 0.0), 0.00001, 1e-10);
}

TEST_F(YamlArgsParserTest, GetString_SpecialCharacters_Success) {
	YAML::Node node;
	node.push_back("Module_With-Special.Chars");
	node.push_back("nn.BatchNorm2d");

	EXPECT_EQ(YamlArgsParser::getString(node, 0, ""), "Module_With-Special.Chars");
	EXPECT_EQ(YamlArgsParser::getString(node, 1, ""), "nn.BatchNorm2d");
}

TEST_F(YamlArgsParserTest, GetInt_ZeroValue_Success) {
	auto argsNode = createArgsNode({0, 1, 0});

	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 0, 99), 0);
	EXPECT_EQ(YamlArgsParser::getInt(argsNode, 2, 99), 0);
}

TEST_F(YamlArgsParserTest, GetBool_ZeroValue_ReturnsFalse) {
	auto argsNode = createArgsNode({0});

	EXPECT_FALSE(YamlArgsParser::getBool(argsNode, 0, true));
}
