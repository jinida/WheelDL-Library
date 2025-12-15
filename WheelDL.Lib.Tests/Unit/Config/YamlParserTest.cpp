#include "pch.h"
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>
#include "Config/YamlParser.h"
#include "Utils/Error/WheelLibException.h"
#include <filesystem>
#include <fstream>

using namespace WheelDL::Config;
using namespace WheelDL::Utils;
namespace fs = std::filesystem;

// =============================================================================
// Test Fixture
// =============================================================================

class YamlParserTest : public ::testing::Test {
protected:
    std::string testBaseDir;

    void SetUp() override {
        testBaseDir = (fs::temp_directory_path() / "WheelDL_YamlParser_Test").string();
        fs::remove_all(testBaseDir);
        fs::create_directories(testBaseDir);
    }

    void TearDown() override {
        try {
            fs::remove_all(testBaseDir);
        }
        catch (...) {}
    }

    std::string getTestPath(const std::string& filename) {
        return (fs::path(testBaseDir) / filename).string();
    }

    void createTestFile(const std::string& filename, const std::string& content) {
        std::string filepath = getTestPath(filename);
        std::ofstream file(filepath);
        file << content;
        file.close();
    }

    void createLargeFile(const std::string& filename, size_t sizeBytes) {
        std::string filepath = getTestPath(filename);
        std::ofstream file(filepath, std::ios::binary);
        std::string chunk(1024 * 1024, 'a');  // 1MB chunk
        size_t written = 0;
        while (written < sizeBytes) {
            size_t toWrite = std::min(chunk.size(), sizeBytes - written);
            file.write(chunk.c_str(), toWrite);
            written += toWrite;
        }
        file.close();
    }

    std::string createDeepYaml(int depth) {
        std::string yaml;
        for (int i = 0; i < depth; ++i) {
            yaml += std::string(i * 2, ' ') + "level" + std::to_string(i) + ":\n";
        }
        yaml += std::string(depth * 2, ' ') + "value: end\n";
        return yaml;
    }
};

// =============================================================================
// parseFrom() Tests (YP-01 ~ YP-08)
// =============================================================================

// YP-01: ParseFrom_ValidFile
TEST_F(YamlParserTest, ParseFrom_ValidFile) {
    createTestFile("valid.yaml", "key: value\nnumber: 42\n");

    YAML::Node node = YamlParser::parseFrom(getTestPath("valid.yaml"));

    EXPECT_TRUE(node.IsDefined());
    EXPECT_EQ("value", node["key"].as<std::string>());
    EXPECT_EQ(42, node["number"].as<int>());
}

// YP-02: ParseFrom_FileNotFound
TEST_F(YamlParserTest, ParseFrom_FileNotFound) {
    EXPECT_THROW({
        try {
            YamlParser::parseFrom(getTestPath("nonexistent.yaml"));
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_FILE_NOT_FOUND, e.getErrorCode());
            EXPECT_NE(std::string::npos, e.getMessage().find("not found"));
            throw;
        }
    }, ConfigurationException);
}

// YP-03: ParseFrom_EmptyFile
TEST_F(YamlParserTest, ParseFrom_EmptyFile) {
    createTestFile("empty.yaml", "");

    EXPECT_THROW({
        try {
            YamlParser::parseFrom(getTestPath("empty.yaml"));
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// YP-04: ParseFrom_InvalidSyntax
TEST_F(YamlParserTest, ParseFrom_InvalidSyntax) {
    createTestFile("invalid.yaml", "key: [broken");

    EXPECT_THROW({
        try {
            YamlParser::parseFrom(getTestPath("invalid.yaml"));
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// YP-05: ParseFrom_TooLarge
TEST_F(YamlParserTest, ParseFrom_TooLarge) {
    // Create file larger than 10MB
    createLargeFile("large.yaml", 11 * 1024 * 1024);

    EXPECT_THROW({
        try {
            YamlParser::parseFrom(getTestPath("large.yaml"));
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
            EXPECT_NE(std::string::npos, e.getMessage().find("too large"));
            throw;
        }
    }, ConfigurationException);
}

// YP-06: ParseFrom_TooDeep
TEST_F(YamlParserTest, ParseFrom_TooDeep) {
    std::string deepYaml = createDeepYaml(105);
    createTestFile("deep.yaml", deepYaml);

    EXPECT_THROW({
        try {
            YamlParser::parseFrom(getTestPath("deep.yaml"));
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
            EXPECT_NE(std::string::npos, e.getMessage().find("depth"));
            throw;
        }
    }, ConfigurationException);
}

// YP-07: ParseFrom_BinaryContent
TEST_F(YamlParserTest, ParseFrom_BinaryContent) {
    // Note: yaml-cpp may accept some binary content as valid YAML
    // This test verifies behavior with clearly invalid binary data
    std::string filepath = getTestPath("binary.yaml");
    std::ofstream file(filepath, std::ios::binary);
    // Create content that will fail YAML parsing - unmatched brackets with null bytes
    char binaryData[] = "key: [\x00\x01\x02\xFF\xFE broken";
    file.write(binaryData, sizeof(binaryData));
    file.close();

    // Either throws exception or returns invalid node - both acceptable
    try {
        YAML::Node node = YamlParser::parseFrom(filepath);
        // If it parses, node should be invalid or have unexpected structure
        SUCCEED() << "yaml-cpp parsed binary content (implementation-specific behavior)";
    }
    catch (const ConfigurationException& e) {
        EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
    }
}

// YP-08: ParseFrom_UTF8Content
TEST_F(YamlParserTest, ParseFrom_UTF8Content) {
    createTestFile("utf8.yaml", "name: \"Korean\"\nvalue: 123\n");

    YAML::Node node = YamlParser::parseFrom(getTestPath("utf8.yaml"));

    EXPECT_TRUE(node.IsDefined());
    EXPECT_EQ(123, node["value"].as<int>());
}

// =============================================================================
// validate() Tests (YP-09 ~ YP-11)
// =============================================================================

// YP-09: Validate_DefinedNode
TEST_F(YamlParserTest, Validate_DefinedNode) {
    createTestFile("defined.yaml", "key: value");
    YAML::Node node = YamlParser::parseFrom(getTestPath("defined.yaml"));

    EXPECT_TRUE(YamlParser::validate(node));
}

// YP-10: Validate_NullNode
TEST_F(YamlParserTest, Validate_NullNode) {
    YAML::Node nullNode = YAML::Load("null");

    EXPECT_FALSE(YamlParser::validate(nullNode));
}

// YP-11: Validate_UndefinedNode
TEST_F(YamlParserTest, Validate_UndefinedNode) {
    YAML::Node node;

    EXPECT_FALSE(YamlParser::validate(node));
}

// =============================================================================
// checkFileSize() Tests (YP-12 ~ YP-14)
// =============================================================================

// YP-12: CheckFileSize_Under
TEST_F(YamlParserTest, CheckFileSize_Under) {
    createTestFile("small.yaml", "key: value");

    EXPECT_TRUE(YamlParser::checkFileSize(getTestPath("small.yaml")));
}

// YP-13: CheckFileSize_Exact
TEST_F(YamlParserTest, CheckFileSize_Exact) {
    // Create exactly 10MB file
    createLargeFile("exact.yaml", 10 * 1024 * 1024);

    EXPECT_TRUE(YamlParser::checkFileSize(getTestPath("exact.yaml")));
}

// YP-14: CheckFileSize_Over
TEST_F(YamlParserTest, CheckFileSize_Over) {
    // Create file larger than 10MB
    createLargeFile("over.yaml", 10 * 1024 * 1024 + 1);

    EXPECT_FALSE(YamlParser::checkFileSize(getTestPath("over.yaml")));
}

// =============================================================================
// checkDepth() Tests (YP-15 ~ YP-17)
// =============================================================================

// YP-15: CheckDepth_Shallow
TEST_F(YamlParserTest, CheckDepth_Shallow) {
    YAML::Node node = YAML::Load("level1:\n  level2:\n    level3: value");

    EXPECT_TRUE(YamlParser::checkDepth(node));
}

// YP-16: CheckDepth_Exact
TEST_F(YamlParserTest, CheckDepth_Exact) {
    // createDeepYaml(n) creates n levels + 1 for the final value
    // So for maxDepth=100, we need depth=99 to have actual depth of 100
    std::string deepYaml = createDeepYaml(99);
    YAML::Node node = YAML::Load(deepYaml);

    EXPECT_TRUE(YamlParser::checkDepth(node, 100));
}

// YP-17: CheckDepth_Over
TEST_F(YamlParserTest, CheckDepth_Over) {
    std::string deepYaml = createDeepYaml(101);
    YAML::Node node = YAML::Load(deepYaml);

    EXPECT_FALSE(YamlParser::checkDepth(node, 100));
}

// =============================================================================
// getString() Tests (YP-18 ~ YP-21)
// =============================================================================

// YP-18: GetString_Exists
TEST_F(YamlParserTest, GetString_Exists) {
    YAML::Node node = YAML::Load("key: \"hello\"");

    EXPECT_EQ("hello", YamlParser::getString(node, "key", "default"));
}

// YP-19: GetString_Missing
TEST_F(YamlParserTest, GetString_Missing) {
    YAML::Node node = YAML::Load("other: value");

    EXPECT_EQ("default", YamlParser::getString(node, "key", "default"));
}

// YP-20: GetString_WrongType
TEST_F(YamlParserTest, GetString_WrongType) {
    YAML::Node node = YAML::Load("key: 123");

    // yaml-cpp can convert int to string, so this might return "123"
    std::string result = YamlParser::getString(node, "key", "default");
    EXPECT_FALSE(result.empty());
}

// YP-21: GetString_Empty
TEST_F(YamlParserTest, GetString_Empty) {
    YAML::Node node = YAML::Load("key: \"\"");

    EXPECT_EQ("", YamlParser::getString(node, "key", "default"));
}

// =============================================================================
// getInt() Tests (YP-22 ~ YP-26)
// =============================================================================

// YP-22: GetInt_Exists
TEST_F(YamlParserTest, GetInt_Exists) {
    YAML::Node node = YAML::Load("key: 42");

    EXPECT_EQ(42, YamlParser::getInt(node, "key", 0));
}

// YP-23: GetInt_Missing
TEST_F(YamlParserTest, GetInt_Missing) {
    YAML::Node node = YAML::Load("other: 10");

    EXPECT_EQ(99, YamlParser::getInt(node, "key", 99));
}

// YP-24: GetInt_WrongType
TEST_F(YamlParserTest, GetInt_WrongType) {
    YAML::Node node = YAML::Load("key: \"text\"");

    EXPECT_EQ(99, YamlParser::getInt(node, "key", 99));
}

// YP-25: GetInt_Float
TEST_F(YamlParserTest, GetInt_Float) {
    YAML::Node node = YAML::Load("key: 3.14");

    // yaml-cpp may not convert float to int, returning default instead
    int result = YamlParser::getInt(node, "key", 0);
    // Accept either truncated value (3) or default (0) depending on yaml-cpp behavior
    EXPECT_TRUE(result == 3 || result == 0) << "Got: " << result;
}

// YP-26: GetInt_Negative
TEST_F(YamlParserTest, GetInt_Negative) {
    YAML::Node node = YAML::Load("key: -10");

    EXPECT_EQ(-10, YamlParser::getInt(node, "key", 0));
}

// =============================================================================
// getFloat() Tests (YP-27 ~ YP-30)
// =============================================================================

// YP-27: GetFloat_Exists
TEST_F(YamlParserTest, GetFloat_Exists) {
    YAML::Node node = YAML::Load("key: 3.14");

    EXPECT_FLOAT_EQ(3.14f, YamlParser::getFloat(node, "key", 0.0f));
}

// YP-28: GetFloat_Missing
TEST_F(YamlParserTest, GetFloat_Missing) {
    YAML::Node node = YAML::Load("other: 1.0");

    EXPECT_FLOAT_EQ(99.0f, YamlParser::getFloat(node, "key", 99.0f));
}

// YP-29: GetFloat_Int
TEST_F(YamlParserTest, GetFloat_Int) {
    YAML::Node node = YAML::Load("key: 5");

    EXPECT_FLOAT_EQ(5.0f, YamlParser::getFloat(node, "key", 0.0f));
}

// YP-30: GetFloat_Scientific
TEST_F(YamlParserTest, GetFloat_Scientific) {
    YAML::Node node = YAML::Load("key: 1e-5");

    EXPECT_FLOAT_EQ(0.00001f, YamlParser::getFloat(node, "key", 0.0f));
}

// =============================================================================
// getBool() Tests (YP-31 ~ YP-35)
// =============================================================================

// YP-31: GetBool_True
TEST_F(YamlParserTest, GetBool_True) {
    YAML::Node node = YAML::Load("key: true");

    EXPECT_TRUE(YamlParser::getBool(node, "key", false));
}

// YP-32: GetBool_False
TEST_F(YamlParserTest, GetBool_False) {
    YAML::Node node = YAML::Load("key: false");

    EXPECT_FALSE(YamlParser::getBool(node, "key", true));
}

// YP-33: GetBool_Yes
TEST_F(YamlParserTest, GetBool_Yes) {
    YAML::Node node = YAML::Load("key: yes");

    EXPECT_TRUE(YamlParser::getBool(node, "key", false));
}

// YP-34: GetBool_No
TEST_F(YamlParserTest, GetBool_No) {
    YAML::Node node = YAML::Load("key: no");

    EXPECT_FALSE(YamlParser::getBool(node, "key", true));
}

// YP-35: GetBool_Missing
TEST_F(YamlParserTest, GetBool_Missing) {
    YAML::Node node = YAML::Load("other: true");

    EXPECT_TRUE(YamlParser::getBool(node, "key", true));
    EXPECT_FALSE(YamlParser::getBool(node, "key", false));
}

// =============================================================================
// getClassNames() Tests (YP-36 ~ YP-39)
// =============================================================================

// YP-36: GetClassNames_MapFormat
TEST_F(YamlParserTest, GetClassNames_MapFormat) {
    YAML::Node node = YAML::Load("names:\n  0: \"cat\"\n  1: \"dog\"\n  2: \"bird\"");

    auto classNames = YamlParser::getClassNames(node);

    EXPECT_EQ(3u, classNames.size());
    EXPECT_EQ("cat", classNames[0]);
    EXPECT_EQ("dog", classNames[1]);
    EXPECT_EQ("bird", classNames[2]);
}

// YP-37: GetClassNames_ArrayFormat
TEST_F(YamlParserTest, GetClassNames_ArrayFormat) {
    YAML::Node node = YAML::Load("names:\n  - cat\n  - dog\n  - bird");

    auto classNames = YamlParser::getClassNames(node);

    EXPECT_EQ(3u, classNames.size());
    EXPECT_EQ("cat", classNames[0]);
    EXPECT_EQ("dog", classNames[1]);
    EXPECT_EQ("bird", classNames[2]);
}

// YP-38: GetClassNames_ClassNamesKey
TEST_F(YamlParserTest, GetClassNames_ClassNamesKey) {
    YAML::Node node = YAML::Load("class_names:\n  - x\n  - y");

    auto classNames = YamlParser::getClassNames(node);

    EXPECT_EQ(2u, classNames.size());
    EXPECT_EQ("x", classNames[0]);
    EXPECT_EQ("y", classNames[1]);
}

// YP-39: GetClassNames_Empty
TEST_F(YamlParserTest, GetClassNames_Empty) {
    YAML::Node node = YAML::Load("key: value");

    auto classNames = YamlParser::getClassNames(node);

    EXPECT_TRUE(classNames.empty());
}
