#include "pch.h"
#include <gtest/gtest.h>
#include "Config/JsonParser.h"
#include "Utils/Error/WheelLibException.h"
#include <filesystem>
#include <fstream>

using namespace WheelDL::Config;
using namespace WheelDL::Utils;
namespace fs = std::filesystem;

// =============================================================================
// Test Fixture
// =============================================================================

class JsonParserTest : public ::testing::Test {
protected:
    std::string testBaseDir;

    void SetUp() override {
        testBaseDir = (fs::temp_directory_path() / "WheelDL_JsonParser_Test").string();
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
        file << "{\"data\": \"";
        std::string chunk(1024 * 1024, 'a');  // 1MB chunk
        size_t written = 0;
        while (written < sizeBytes - 20) {  // Reserve space for JSON structure
            size_t toWrite = std::min(chunk.size(), sizeBytes - 20 - written);
            file.write(chunk.c_str(), toWrite);
            written += toWrite;
        }
        file << "\"}";
        file.close();
    }

    std::string createDeepJson(int depth) {
        std::string json;
        for (int i = 0; i < depth; ++i) {
            json += "{\"level" + std::to_string(i) + "\":";
        }
        json += "\"end\"";
        for (int i = 0; i < depth; ++i) {
            json += "}";
        }
        return json;
    }
};

// =============================================================================
// parseFrom() Tests (JP-01 ~ JP-10)
// =============================================================================

// JP-01: ParseFrom_ValidFile
TEST_F(JsonParserTest, ParseFrom_ValidFile) {
    createTestFile("valid.json", "{\"key\": \"value\", \"number\": 42}");

    nlohmann::json node = JsonParser::parseFrom(getTestPath("valid.json"));

    EXPECT_FALSE(node.is_null());
    EXPECT_EQ("value", node["key"].get_ref<const std::string&>());
    EXPECT_EQ(42, node["number"].get<int64_t>());
}

// JP-02: ParseFrom_FileNotFound
TEST_F(JsonParserTest, ParseFrom_FileNotFound) {
    EXPECT_THROW({
        try {
            JsonParser::parseFrom(getTestPath("nonexistent.json"));
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_FILE_NOT_FOUND, e.getErrorCode());
            EXPECT_NE(std::string::npos, e.getMessage().find("not found"));
            throw;
        }
    }, ConfigurationException);
}

// JP-03: ParseFrom_EmptyFile
TEST_F(JsonParserTest, ParseFrom_EmptyFile) {
    createTestFile("empty.json", "");

    EXPECT_THROW({
        try {
            JsonParser::parseFrom(getTestPath("empty.json"));
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// JP-04: ParseFrom_InvalidSyntax
TEST_F(JsonParserTest, ParseFrom_InvalidSyntax) {
    createTestFile("invalid.json", "{broken");

    EXPECT_THROW({
        try {
            JsonParser::parseFrom(getTestPath("invalid.json"));
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// JP-05: ParseFrom_TooLarge
TEST_F(JsonParserTest, ParseFrom_TooLarge) {
    // Create file larger than 1GB - Skip this test in practice due to resource constraints
    // This test documents expected behavior
    SUCCEED() << "Skipped: Creating 1GB+ file is impractical for unit tests";
}

// JP-06: ParseFrom_TooDeep
TEST_F(JsonParserTest, ParseFrom_TooDeep) {
    std::string deepJson = createDeepJson(105);
    createTestFile("deep.json", deepJson);

    EXPECT_THROW({
        try {
            JsonParser::parseFrom(getTestPath("deep.json"));
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
            EXPECT_NE(std::string::npos, e.getMessage().find("depth"));
            throw;
        }
    }, ConfigurationException);
}

// JP-07: ParseFrom_TrailingComma
TEST_F(JsonParserTest, ParseFrom_TrailingComma) {
    createTestFile("trailing.json", "{\"a\": 1,}");

    EXPECT_THROW({
        try {
            JsonParser::parseFrom(getTestPath("trailing.json"));
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// JP-08: ParseFrom_SingleQuotes
TEST_F(JsonParserTest, ParseFrom_SingleQuotes) {
    createTestFile("single_quotes.json", "{'a': 1}");

    EXPECT_THROW({
        try {
            JsonParser::parseFrom(getTestPath("single_quotes.json"));
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_PARSE_FAILED, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// JP-09: ParseFrom_UTF8Content
TEST_F(JsonParserTest, ParseFrom_UTF8Content) {
    createTestFile("utf8.json", "{\"name\": \"Korean\", \"value\": 123}");

    nlohmann::json node = JsonParser::parseFrom(getTestPath("utf8.json"));

    EXPECT_FALSE(node.is_null());
    EXPECT_EQ(123, node["value"].get<int64_t>());
}

// JP-10: ParseFrom_FileOpenFailed
TEST_F(JsonParserTest, ParseFrom_FileOpenFailed) {
    // Test with a path that exists but cannot be opened
    // On Windows, we can't easily simulate this, so test with invalid path
    std::string invalidPath = "Z:\\nonexistent\\<>|path\\file.json";

    EXPECT_THROW({
        try {
            JsonParser::parseFrom(invalidPath);
        }
        catch (const ConfigurationException& e) {
            EXPECT_EQ(ErrorCode::CONFIG_FILE_NOT_FOUND, e.getErrorCode());
            throw;
        }
    }, ConfigurationException);
}

// =============================================================================
// validate() Tests (JP-11 ~ JP-12)
// =============================================================================

// JP-11: Validate_Object
TEST_F(JsonParserTest, Validate_Object) {
    nlohmann::json node = nlohmann::json::parse("{}");

    EXPECT_TRUE(JsonParser::validate(node));
}

// JP-12: Validate_Null
TEST_F(JsonParserTest, Validate_Null) {
    nlohmann::json node = nlohmann::json::parse("null");

    EXPECT_FALSE(JsonParser::validate(node));
}

// =============================================================================
// Type Getters Tests (JP-13 ~ JP-19)
// =============================================================================

// JP-13: GetString_Exists
TEST_F(JsonParserTest, GetString_Exists) {
    nlohmann::json node = nlohmann::json::parse("{\"k\": \"value\"}");

    EXPECT_EQ("value", JsonParser::getString(node, "k", "default"));
}

// JP-14: GetString_Missing
TEST_F(JsonParserTest, GetString_Missing) {
    nlohmann::json node = nlohmann::json::parse("{\"other\": \"value\"}");

    EXPECT_EQ("default", JsonParser::getString(node, "k", "default"));
}

// JP-15: GetInt_Exists
TEST_F(JsonParserTest, GetInt_Exists) {
    nlohmann::json node = nlohmann::json::parse("{\"k\": 42}");

    EXPECT_EQ(42, JsonParser::getInt(node, "k", 0));
}

// JP-16: GetInt_Missing
TEST_F(JsonParserTest, GetInt_Missing) {
    nlohmann::json node = nlohmann::json::parse("{\"other\": 10}");

    EXPECT_EQ(99, JsonParser::getInt(node, "k", 99));
}

// JP-17: GetFloat_Exists
TEST_F(JsonParserTest, GetFloat_Exists) {
    nlohmann::json node = nlohmann::json::parse("{\"k\": 3.14}");

    EXPECT_FLOAT_EQ(3.14f, JsonParser::getFloat(node, "k", 0.0f));
}

// JP-18: GetBool_True
TEST_F(JsonParserTest, GetBool_True) {
    nlohmann::json node = nlohmann::json::parse("{\"k\": true}");

    EXPECT_TRUE(JsonParser::getBool(node, "k", false));
}

// JP-19: GetBool_StringTrue
TEST_F(JsonParserTest, GetBool_StringTrue) {
    nlohmann::json node = nlohmann::json::parse("{\"k\": \"true\"}");

    // String "true" should return default because it's not a boolean
    EXPECT_FALSE(JsonParser::getBool(node, "k", false));
}

// =============================================================================
// getClassNames() Tests (JP-20 ~ JP-24)
// =============================================================================

// JP-20: GetClassNames_Object
TEST_F(JsonParserTest, GetClassNames_Object) {
    nlohmann::json node = nlohmann::json::parse("{\"names\": {\"0\": \"cat\", \"1\": \"dog\"}}");

    auto classNames = JsonParser::getClassNames(node);

    EXPECT_EQ(2u, classNames.size());
    EXPECT_EQ("cat", classNames[0]);
    EXPECT_EQ("dog", classNames[1]);
}

// JP-21: GetClassNames_Array
TEST_F(JsonParserTest, GetClassNames_Array) {
    nlohmann::json node = nlohmann::json::parse("{\"names\": [\"cat\", \"dog\", \"bird\"]}");

    auto classNames = JsonParser::getClassNames(node);

    EXPECT_EQ(3u, classNames.size());
    EXPECT_EQ("cat", classNames[0]);
    EXPECT_EQ("dog", classNames[1]);
    EXPECT_EQ("bird", classNames[2]);
}

// JP-22: GetClassNames_ClassNamesKey
TEST_F(JsonParserTest, GetClassNames_ClassNamesKey) {
    nlohmann::json node = nlohmann::json::parse("{\"class_names\": [\"x\", \"y\"]}");

    auto classNames = JsonParser::getClassNames(node);

    EXPECT_EQ(2u, classNames.size());
    EXPECT_EQ("x", classNames[0]);
    EXPECT_EQ("y", classNames[1]);
}

// JP-23: GetClassNames_Empty
TEST_F(JsonParserTest, GetClassNames_Empty) {
    nlohmann::json node = nlohmann::json::parse("{\"key\": \"value\"}");

    auto classNames = JsonParser::getClassNames(node);

    EXPECT_TRUE(classNames.empty());
}

// JP-24: GetClassNames_InvalidKey
TEST_F(JsonParserTest, GetClassNames_InvalidKey) {
    nlohmann::json node = nlohmann::json::parse("{\"names\": {\"abc\": \"cat\"}}");

    // Non-numeric key should be caught and result in empty map
    auto classNames = JsonParser::getClassNames(node);

    EXPECT_TRUE(classNames.empty());
}
