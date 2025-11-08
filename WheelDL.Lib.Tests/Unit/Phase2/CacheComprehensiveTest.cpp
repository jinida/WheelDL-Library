#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Cache/CacheManager.h"
#include "WheelDL.Lib/Data/Common/Annotation.h"
#include "WheelDL.Lib/Utils/Common/Constants.h"
#include <filesystem>
#include <fstream>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>

using namespace WheelDL::Data::Cache;
using namespace WheelDL::Data;
namespace fs = std::filesystem;

namespace {
	std::filesystem::path getTestDataPath()
	{
		std::filesystem::path sourceDir = __FILE__;
		sourceDir = sourceDir.parent_path();
		while (sourceDir.filename() != "WheelDL.Lib.Tests" && sourceDir.has_parent_path()) {
			sourceDir = sourceDir.parent_path();
		}
		return sourceDir / "Data";
	}
}

/**
 * @class CacheComprehensiveTest
 * @brief Comprehensive test suite for CacheManager and cache functionality
 */
class CacheComprehensiveTest : public ::testing::Test
{
protected:
	fs::path testDir;
	fs::path testCachePath;
	std::string mnistDataPath;
	std::string mnistLabelsPath;

	void SetUp() override
	{
		// Create temporary test directory
		testDir = fs::temp_directory_path() / "CacheComprehensiveTest";
		testCachePath = testDir / "test_cache.bin";

		// Clean up if exists
		if (fs::exists(testDir)) {
			fs::remove_all(testDir);
		}
		fs::create_directories(testDir);

		// Set up MNIST paths
		auto testDataPath = getTestDataPath();
		mnistDataPath = (testDataPath / "mnist_sample" / "images").string();
		mnistLabelsPath = (testDataPath / "mnist_sample" / "labels" / "objectdetection").string();
	}

	void TearDown() override
	{
		// Clean up test directory
		if (fs::exists(testDir)) {
			fs::remove_all(testDir);
		}
	}

	// Helper function to create annotations of various types
	Annotation createAnnotation(LabelType type, int numObjects = 3)
	{
		Annotation ann(type);

		switch (type) {
		case LabelType::XYWH:
			for (int i = 0; i < numObjects; ++i) {
				ann.addObject(i % 10, {
					10.0f + i * 20.0f,  // x center
					20.0f + i * 15.0f,  // y center
					30.0f,              // width
					40.0f               // height
					});
			}
			break;

		case LabelType::XYXY:
			for (int i = 0; i < numObjects; ++i) {
				ann.addObject(i % 10, {
					10.0f + i * 20.0f,  // x1
					20.0f + i * 15.0f,  // y1
					40.0f + i * 20.0f,  // x2
					60.0f + i * 15.0f   // y2
					});
			}
			break;

		case LabelType::POLYGON:
			for (int i = 0; i < numObjects; ++i) {
				std::vector<float> polygon;
				int numPoints = 6 + (i % 4) * 2;  // 6, 8, 10, 12 points
				for (int j = 0; j < numPoints; ++j) {
					polygon.push_back(10.0f + i * 20.0f + j * 2.0f);  // x
					polygon.push_back(20.0f + i * 15.0f + j * 1.5f);  // y
				}
				ann.addObject(i % 10, polygon);
			}
			break;

		case LabelType::XYXYXYXY:
			for (int i = 0; i < numObjects; ++i) {
				ann.addObject(i % 10, {
					10.0f + i * 20.0f,   // x1
					20.0f + i * 15.0f,   // y1
					40.0f + i * 20.0f,   // x2
					20.0f + i * 15.0f,   // y2
					40.0f + i * 20.0f,   // x3
					60.0f + i * 15.0f,   // y3
					10.0f + i * 20.0f,   // x4
					60.0f + i * 15.0f    // y4
					});
			}
			break;

		case LabelType::XYWHR:
			for (int i = 0; i < numObjects; ++i) {
				ann.addObject(i % 10, {
					10.0f + i * 20.0f,  // x center
					20.0f + i * 15.0f,  // y center
					30.0f,              // width
					40.0f,              // height
					i * 0.1f            // rotation
					});
			}
			break;

		case LabelType::NONE:
			// Classification - no spatial data
			ann.addObject(5, {});
			break;
		}

		return ann;
	}

	void expectAnnotationEqual(const Annotation& a, const Annotation& b)
	{
		EXPECT_EQ(a.getLabelType(), b.getLabelType());
		EXPECT_EQ(a.getClasses(), b.getClasses());
		EXPECT_EQ(a.getPoints(), b.getPoints());
	}
};

// ========================================
// SECTION 1: Binary Serialization Tests
// ========================================

TEST_F(CacheComprehensiveTest, Serialization_XYWH_Annotations)
{
	Annotation original = createAnnotation(LabelType::XYWH, 5);

	std::string serialized = CacheManager::serializeAnnotation(original);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	expectAnnotationEqual(deserialized, original);
	EXPECT_EQ(deserialized.size(), 5);
}

TEST_F(CacheComprehensiveTest, Serialization_XYXY_Annotations)
{
	Annotation original = createAnnotation(LabelType::XYXY, 5);

	std::string serialized = CacheManager::serializeAnnotation(original);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	expectAnnotationEqual(deserialized, original);
	EXPECT_EQ(deserialized.size(), 5);
}

TEST_F(CacheComprehensiveTest, Serialization_POLYGON_Annotations)
{
	Annotation original = createAnnotation(LabelType::POLYGON, 5);

	std::string serialized = CacheManager::serializeAnnotation(original);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	expectAnnotationEqual(deserialized, original);
	EXPECT_EQ(deserialized.size(), 5);
}

TEST_F(CacheComprehensiveTest, Serialization_XYXYXYXY_Annotations)
{
	Annotation original = createAnnotation(LabelType::XYXYXYXY, 5);

	std::string serialized = CacheManager::serializeAnnotation(original);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	expectAnnotationEqual(deserialized, original);
	EXPECT_EQ(deserialized.size(), 5);
}

TEST_F(CacheComprehensiveTest, Serialization_XYWHR_Annotations)
{
	Annotation original = createAnnotation(LabelType::XYWHR, 5);

	std::string serialized = CacheManager::serializeAnnotation(original);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	expectAnnotationEqual(deserialized, original);
	EXPECT_EQ(deserialized.size(), 5);
}

TEST_F(CacheComprehensiveTest, Serialization_NONE_ClassificationAnnotations)
{
	Annotation original = createAnnotation(LabelType::NONE, 1);

	std::string serialized = CacheManager::serializeAnnotation(original);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	expectAnnotationEqual(deserialized, original);
	EXPECT_EQ(deserialized.size(), 1);
}

TEST_F(CacheComprehensiveTest, Serialization_EmptyAnnotation)
{
	Annotation original(LabelType::XYWH);
	// No objects added

	std::string serialized = CacheManager::serializeAnnotation(original);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	expectAnnotationEqual(deserialized, original);
	EXPECT_EQ(deserialized.size(), 0);
}

TEST_F(CacheComprehensiveTest, Serialization_MixedLabelTypes)
{
	// Test that different label types can be serialized/deserialized correctly
	std::vector<LabelType> types = {
		LabelType::XYWH,
		LabelType::XYXY,
		LabelType::POLYGON,
		LabelType::XYXYXYXY,
		LabelType::XYWHR,
		LabelType::NONE
	};

	for (auto type : types) {
		Annotation original = createAnnotation(type, 3);
		std::string serialized = CacheManager::serializeAnnotation(original);
		Annotation deserialized = CacheManager::deserializeAnnotation(serialized);
		expectAnnotationEqual(deserialized, original);
	}
}

// ========================================
// SECTION 2: Deserialization Integrity Tests
// ========================================

TEST_F(CacheComprehensiveTest, Deserialization_IntegrityCheck_MultipleRounds)
{
	Annotation original = createAnnotation(LabelType::POLYGON, 10);

	// Serialize and deserialize multiple times
	Annotation current = original;
	for (int i = 0; i < 5; ++i) {
		std::string serialized = CacheManager::serializeAnnotation(current);
		current = CacheManager::deserializeAnnotation(serialized);
	}

	expectAnnotationEqual(current, original);
}

TEST_F(CacheComprehensiveTest, Deserialization_CorruptedData_ThrowsException)
{
	std::string corruptedData = "this is not valid binary data";

	EXPECT_THROW(
		CacheManager::deserializeAnnotation(corruptedData),
		std::runtime_error
	);
}

TEST_F(CacheComprehensiveTest, Deserialization_TruncatedData_ThrowsException)
{
	Annotation original = createAnnotation(LabelType::XYWH, 5);
	std::string serialized = CacheManager::serializeAnnotation(original);

	// Truncate the data
	std::string truncated = serialized.substr(0, serialized.size() / 2);

	EXPECT_THROW(
		CacheManager::deserializeAnnotation(truncated),
		std::runtime_error
	);
}

TEST_F(CacheComprehensiveTest, Deserialization_EmptyData_ThrowsException)
{
	std::string emptyData = "";

	EXPECT_THROW(
		CacheManager::deserializeAnnotation(emptyData),
		std::runtime_error
	);
}

// ========================================
// SECTION 3: Cache File Creation and Loading Tests
// ========================================

TEST_F(CacheComprehensiveTest, CacheFile_CreateAndLoad_SingleEntry)
{
	std::unordered_map<std::string, Annotation> cache;
	cache["image1.jpg"] = createAnnotation(LabelType::XYWH, 5);

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "test_hash");
	ASSERT_TRUE(fs::exists(testCachePath));

	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), 1);
	ASSERT_TRUE(loaded.count("image1.jpg"));
	expectAnnotationEqual(loaded["image1.jpg"], cache["image1.jpg"]);
}

TEST_F(CacheComprehensiveTest, CacheFile_CreateAndLoad_MultipleEntries)
{
	std::unordered_map<std::string, Annotation> cache;

	cache["image1.jpg"] = createAnnotation(LabelType::XYWH, 5);
	cache["image2.jpg"] = createAnnotation(LabelType::XYXY, 3);
	cache["image3.jpg"] = createAnnotation(LabelType::POLYGON, 7);
	cache["subfolder/image4.jpg"] = createAnnotation(LabelType::XYWHR, 4);

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "test_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), 4);
	expectAnnotationEqual(loaded["image1.jpg"], cache["image1.jpg"]);
	expectAnnotationEqual(loaded["image2.jpg"], cache["image2.jpg"]);
	expectAnnotationEqual(loaded["image3.jpg"], cache["image3.jpg"]);
	expectAnnotationEqual(loaded["subfolder/image4.jpg"], cache["subfolder/image4.jpg"]);
}

TEST_F(CacheComprehensiveTest, CacheFile_CreateAndLoad_EmptyCache)
{
	std::unordered_map<std::string, Annotation> cache;

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "empty_hash");
	ASSERT_TRUE(fs::exists(testCachePath));

	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	EXPECT_EQ(loaded.size(), 0);
}

TEST_F(CacheComprehensiveTest, CacheFile_AtomicWrite_TemporaryFileCleanup)
{
	std::unordered_map<std::string, Annotation> cache;
	cache["test.jpg"] = createAnnotation(LabelType::XYWH, 3);

	std::string tempPath = (testCachePath.parent_path() / (testCachePath.filename().string() + ".tmp")).string();

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "test_hash");

	// Temporary file should be cleaned up
	EXPECT_FALSE(fs::exists(tempPath));
	EXPECT_TRUE(fs::exists(testCachePath));
}

// ========================================
// SECTION 4: Hash Computation Tests
// ========================================

TEST_F(CacheComprehensiveTest, HashComputation_SameDirectory_SameHash)
{
	// Create test files
	fs::path file1 = testDir / "file1.txt";
	fs::path file2 = testDir / "file2.txt";

	std::ofstream(file1.string()) << "content1";
	std::ofstream(file2.string()) << "content2";

	std::string hash1 = CacheManager::computeHash(testDir.string());
	std::string hash2 = CacheManager::computeHash(testDir.string());

	EXPECT_EQ(hash1, hash2);
	EXPECT_FALSE(hash1.empty());
}

TEST_F(CacheComprehensiveTest, HashComputation_DifferentContent_DifferentHash)
{
	// First scenario
	fs::path file1 = testDir / "file1.txt";
	std::ofstream(file1.string()) << "content1";
	std::string hash1 = CacheManager::computeHash(testDir.string());

	// Modify content
	std::ofstream(file1.string(), std::ios::trunc) << "different_content_that_is_much_longer";
	std::string hash2 = CacheManager::computeHash(testDir.string());

	EXPECT_NE(hash1, hash2);
}

TEST_F(CacheComprehensiveTest, HashComputation_AdditionalFile_DifferentHash)
{
	fs::path file1 = testDir / "file1.txt";
	std::ofstream(file1.string()) << "content1";
	std::string hash1 = CacheManager::computeHash(testDir.string());

	// Add new file
	fs::path file2 = testDir / "file2.txt";
	std::ofstream(file2.string()) << "content2";
	std::string hash2 = CacheManager::computeHash(testDir.string());

	EXPECT_NE(hash1, hash2);
}

TEST_F(CacheComprehensiveTest, HashComputation_NestedDirectories)
{
	// Create nested structure
	fs::path subdir1 = testDir / "subdir1";
	fs::path subdir2 = testDir / "subdir1" / "subdir2";
	fs::create_directories(subdir2);

	{
		std::ofstream file1((testDir / "file1.txt").string());
		file1 << "content1";
		file1.close();
	}
	{
		std::ofstream file2((subdir1 / "file2.txt").string());
		file2 << "content2";
		file2.close();
	}
	{
		std::ofstream file3((subdir2 / "file3.txt").string());
		file3 << "content3";
		file3.close();
	}

	std::string hash = CacheManager::computeHash(testDir.string());
	EXPECT_FALSE(hash.empty());

	// Verify hash changes when nested file changes
	{
		std::ofstream file((subdir2 / "file3.txt").string(), std::ios::trunc);
		file << "modified content that is different";
		file.flush();
		file.close();  // Ensure file is flushed to disk
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Give filesystem time to update
	std::string hash2 = CacheManager::computeHash(testDir.string());

	EXPECT_NE(hash, hash2);
}

TEST_F(CacheComprehensiveTest, HashComputation_EmptyDirectory)
{
	std::string hash = CacheManager::computeHash(testDir.string());
	EXPECT_FALSE(hash.empty());
}

TEST_F(CacheComprehensiveTest, HashComputation_NonExistentDirectory)
{
	fs::path nonExistentDir = testDir / "nonexistent";

	EXPECT_THROW(
		CacheManager::computeHash(nonExistentDir.string()),
		std::runtime_error
	);
}

TEST_F(CacheComprehensiveTest, HashComputation_RealMnistDataset)
{
	// Skip if MNIST data doesn't exist
	if (!fs::exists(mnistDataPath)) {
		std::cout << "SKIPPED: MNIST dataset not found at " << mnistDataPath << std::endl;
		return;
	}

	std::string hash = CacheManager::computeHash(mnistDataPath);
	EXPECT_FALSE(hash.empty());

	// Hash should be consistent
	std::string hash2 = CacheManager::computeHash(mnistDataPath);
	EXPECT_EQ(hash, hash2);
}

// ========================================
// SECTION 5: Cache Verification Tests
// ========================================

TEST_F(CacheComprehensiveTest, CacheVerification_ValidCache)
{
	// Create test dataset
	fs::path file1 = testDir / "file1.txt";
	std::ofstream(file1.string()) << "content";

	std::string datasetHash = CacheManager::computeHash(testDir.string());

	// Create cache with matching hash
	std::unordered_map<std::string, Annotation> cache;
	cache["test.jpg"] = createAnnotation(LabelType::XYWH, 3);
	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, datasetHash);

	// Verify with matching hash
	bool isValid = CacheManager::verifyCacheFrom(testCachePath.string(), datasetHash);

	EXPECT_TRUE(isValid);
}

TEST_F(CacheComprehensiveTest, CacheVerification_InvalidCache_WrongHash)
{
	std::unordered_map<std::string, Annotation> cache;
	cache["test.jpg"] = createAnnotation(LabelType::XYWH, 3);
	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "original_hash");

	bool isValid = CacheManager::verifyCacheFrom(testCachePath.string(), "wrong_hash");

	EXPECT_FALSE(isValid);
}

TEST_F(CacheComprehensiveTest, CacheVerification_NonExistentCache)
{
	fs::path nonExistentPath = testDir / "nonexistent.bin";
	bool isValid = CacheManager::verifyCacheFrom(nonExistentPath.string(), "any_hash");

	EXPECT_FALSE(isValid);
}

TEST_F(CacheComprehensiveTest, CacheVerification_CorruptedCacheFile)
{
	// Create corrupted cache file
	std::ofstream corrupt(testCachePath.string(), std::ios::binary);
	corrupt << "corrupted data that is not a valid cache file";
	corrupt.close();

	bool isValid = CacheManager::verifyCacheFrom(testCachePath.string(), "any_hash");

	EXPECT_FALSE(isValid);
}

// ========================================
// SECTION 6: Cache Invalidation Tests
// ========================================

TEST_F(CacheComprehensiveTest, CacheInvalidation_DatasetFileAdded)
{
	// Create initial dataset
	fs::path file1 = testDir / "file1.txt";
	std::ofstream(file1.string()) << "content1";
	std::string hash1 = CacheManager::computeHash(testDir.string());

	// Create cache
	std::unordered_map<std::string, Annotation> cache;
	cache["test.jpg"] = createAnnotation(LabelType::XYWH, 3);
	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, hash1);

	// Verify cache is valid
	EXPECT_TRUE(CacheManager::verifyCacheFrom(testCachePath.string(), hash1));

	// Add new file to dataset
	fs::path file2 = testDir / "file2.txt";
	std::ofstream(file2.string()) << "content2";
	std::string hash2 = CacheManager::computeHash(testDir.string());

	// Cache should now be invalid
	EXPECT_FALSE(CacheManager::verifyCacheFrom(testCachePath.string(), hash2));
}

TEST_F(CacheComprehensiveTest, CacheInvalidation_DatasetFileModified)
{
	// Create initial dataset
	fs::path file1 = testDir / "file1.txt";
	std::ofstream(file1.string()) << "content1";
	std::string hash1 = CacheManager::computeHash(testDir.string());

	// Create cache
	std::unordered_map<std::string, Annotation> cache;
	cache["test.jpg"] = createAnnotation(LabelType::XYWH, 3);
	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, hash1);

	// Verify cache is valid
	EXPECT_TRUE(CacheManager::verifyCacheFrom(testCachePath.string(), hash1));

	// Modify file content
	std::ofstream(file1.string(), std::ios::trunc) << "modified_content_different";
	std::string hash2 = CacheManager::computeHash(testDir.string());

	// Cache should now be invalid
	EXPECT_FALSE(CacheManager::verifyCacheFrom(testCachePath.string(), hash2));
}

TEST_F(CacheComprehensiveTest, CacheInvalidation_DatasetFileRemoved)
{
	// Create initial dataset
	fs::path file1 = testDir / "file1.txt";
	fs::path file2 = testDir / "file2.txt";
	std::ofstream(file1.string()) << "content1";
	std::ofstream(file2.string()) << "content2";
	std::string hash1 = CacheManager::computeHash(testDir.string());

	// Create cache
	std::unordered_map<std::string, Annotation> cache;
	cache["test.jpg"] = createAnnotation(LabelType::XYWH, 3);
	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, hash1);

	// Verify cache is valid
	EXPECT_TRUE(CacheManager::verifyCacheFrom(testCachePath.string(), hash1));

	// Remove file
	fs::remove(file2);
	std::string hash2 = CacheManager::computeHash(testDir.string());

	// Cache should now be invalid
	EXPECT_FALSE(CacheManager::verifyCacheFrom(testCachePath.string(), hash2));
}

// ========================================
// SECTION 7: Corrupted Cache File Handling
// ========================================

TEST_F(CacheComprehensiveTest, CorruptedCache_WrongVersion)
{
	// Create a cache file with wrong version
	std::ofstream file(testCachePath.string(), std::ios::binary);
	uint32_t wrongVersion = 999999;
	file.write(reinterpret_cast<const char*>(&wrongVersion), sizeof(wrongVersion));
	file.close();

	EXPECT_THROW(
		CacheManager::loadLabelCacheFrom(testCachePath.string()),
		std::runtime_error
	);
}

TEST_F(CacheComprehensiveTest, CorruptedCache_InvalidBinaryData)
{
	std::ofstream file(testCachePath.string(), std::ios::binary);
	file << "This is not valid binary cache data at all!";
	file.close();

	EXPECT_THROW(
		CacheManager::loadLabelCacheFrom(testCachePath.string()),
		std::runtime_error
	);
}

TEST_F(CacheComprehensiveTest, CorruptedCache_TruncatedFile)
{
	// Create a valid cache
	std::unordered_map<std::string, Annotation> cache;
	cache["test.jpg"] = createAnnotation(LabelType::XYWH, 3);
	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "test_hash");

	// Read and truncate
	std::ifstream inFile(testCachePath.string(), std::ios::binary);
	std::string content((std::istreambuf_iterator<char>(inFile)),
		std::istreambuf_iterator<char>());
	inFile.close();

	// Write truncated version
	std::ofstream outFile(testCachePath.string(), std::ios::binary | std::ios::trunc);
	outFile.write(content.data(), content.size() / 2);
	outFile.close();

	EXPECT_THROW(
		CacheManager::loadLabelCacheFrom(testCachePath.string()),
		std::runtime_error
	);
}

// ========================================
// SECTION 8: Label Cache Tests - Different Annotation Types
// ========================================

TEST_F(CacheComprehensiveTest, LabelCache_XYWH_SaveAndLoad)
{
	std::unordered_map<std::string, Annotation> cache;

	for (int i = 0; i < 10; ++i) {
		std::string imagePath = "image_" + std::to_string(i) + ".jpg";
		cache[imagePath] = createAnnotation(LabelType::XYWH, 5);
	}

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "xywh_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), cache.size());
	for (const auto& [key, value] : cache) {
		ASSERT_TRUE(loaded.count(key));
		expectAnnotationEqual(loaded[key], value);
	}
}

TEST_F(CacheComprehensiveTest, LabelCache_XYXY_SaveAndLoad)
{
	std::unordered_map<std::string, Annotation> cache;

	for (int i = 0; i < 10; ++i) {
		std::string imagePath = "image_" + std::to_string(i) + ".jpg";
		cache[imagePath] = createAnnotation(LabelType::XYXY, 5);
	}

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "xyxy_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), cache.size());
	for (const auto& [key, value] : cache) {
		ASSERT_TRUE(loaded.count(key));
		expectAnnotationEqual(loaded[key], value);
	}
}

TEST_F(CacheComprehensiveTest, LabelCache_POLYGON_SaveAndLoad)
{
	std::unordered_map<std::string, Annotation> cache;

	for (int i = 0; i < 10; ++i) {
		std::string imagePath = "image_" + std::to_string(i) + ".jpg";
		cache[imagePath] = createAnnotation(LabelType::POLYGON, 3);
	}

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "polygon_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), cache.size());
	for (const auto& [key, value] : cache) {
		ASSERT_TRUE(loaded.count(key));
		expectAnnotationEqual(loaded[key], value);
	}
}

TEST_F(CacheComprehensiveTest, LabelCache_XYXYXYXY_SaveAndLoad)
{
	std::unordered_map<std::string, Annotation> cache;

	for (int i = 0; i < 10; ++i) {
		std::string imagePath = "image_" + std::to_string(i) + ".jpg";
		cache[imagePath] = createAnnotation(LabelType::XYXYXYXY, 4);
	}

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "xyxyxyxy_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), cache.size());
	for (const auto& [key, value] : cache) {
		ASSERT_TRUE(loaded.count(key));
		expectAnnotationEqual(loaded[key], value);
	}
}

TEST_F(CacheComprehensiveTest, LabelCache_XYWHR_SaveAndLoad)
{
	std::unordered_map<std::string, Annotation> cache;

	for (int i = 0; i < 10; ++i) {
		std::string imagePath = "image_" + std::to_string(i) + ".jpg";
		cache[imagePath] = createAnnotation(LabelType::XYWHR, 4);
	}

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "xywhr_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), cache.size());
	for (const auto& [key, value] : cache) {
		ASSERT_TRUE(loaded.count(key));
		expectAnnotationEqual(loaded[key], value);
	}
}

TEST_F(CacheComprehensiveTest, LabelCache_NONE_Classification_SaveAndLoad)
{
	std::unordered_map<std::string, Annotation> cache;

	for (int i = 0; i < 10; ++i) {
		std::string imagePath = "image_" + std::to_string(i) + ".jpg";
		cache[imagePath] = createAnnotation(LabelType::NONE, 1);
	}

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "none_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), cache.size());
	for (const auto& [key, value] : cache) {
		ASSERT_TRUE(loaded.count(key));
		expectAnnotationEqual(loaded[key], value);
	}
}

TEST_F(CacheComprehensiveTest, LabelCache_MixedTypes_SaveAndLoad)
{
	std::unordered_map<std::string, Annotation> cache;

	cache["xywh_1.jpg"] = createAnnotation(LabelType::XYWH, 5);
	cache["xyxy_1.jpg"] = createAnnotation(LabelType::XYXY, 3);
	cache["polygon_1.jpg"] = createAnnotation(LabelType::POLYGON, 7);
	cache["xyxyxyxy_1.jpg"] = createAnnotation(LabelType::XYXYXYXY, 4);
	cache["xywhr_1.jpg"] = createAnnotation(LabelType::XYWHR, 6);
	cache["none_1.jpg"] = createAnnotation(LabelType::NONE, 1);

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "mixed_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), 6);
	for (const auto& [key, value] : cache) {
		ASSERT_TRUE(loaded.count(key));
		expectAnnotationEqual(loaded[key], value);
	}
}

// ========================================
// SECTION 9: Large Dataset Caching Tests
// ========================================

TEST_F(CacheComprehensiveTest, LargeDataset_1000Annotations)
{
	std::unordered_map<std::string, Annotation> cache;

	// Create 1000 annotations with varying complexity
	for (int i = 0; i < 1000; ++i) {
		std::string imagePath = "large_dataset/image_" + std::to_string(i) + ".jpg";
		LabelType type = static_cast<LabelType>((i % 5) + 1);  // Cycle through types
		int numObjects = 1 + (i % 10);  // 1-10 objects per image
		cache[imagePath] = createAnnotation(type, numObjects);
	}

	auto startSave = std::chrono::high_resolution_clock::now();
	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "large_hash");
	auto endSave = std::chrono::high_resolution_clock::now();

	auto startLoad = std::chrono::high_resolution_clock::now();
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());
	auto endLoad = std::chrono::high_resolution_clock::now();

	auto saveDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSave - startSave);
	auto loadDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endLoad - startLoad);

	ASSERT_EQ(loaded.size(), 1000);

	// Verify a sampling of the data
	for (int i = 0; i < 1000; i += 100) {
		std::string imagePath = "large_dataset/image_" + std::to_string(i) + ".jpg";
		ASSERT_TRUE(loaded.count(imagePath));
		expectAnnotationEqual(loaded[imagePath], cache[imagePath]);
	}

	// Performance check - should be reasonably fast
	EXPECT_LT(saveDuration.count(), 5000) << "Saving 1000 annotations took " << saveDuration.count() << "ms";
	EXPECT_LT(loadDuration.count(), 5000) << "Loading 1000 annotations took " << loadDuration.count() << "ms";
}

TEST_F(CacheComprehensiveTest, LargeDataset_EmptyDataset)
{
	std::unordered_map<std::string, Annotation> cache;
	// Empty cache

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "empty_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	EXPECT_EQ(loaded.size(), 0);
}

// ========================================
// SECTION 10: Thread Safety Tests
// ========================================

TEST_F(CacheComprehensiveTest, ThreadSafety_ConcurrentReads)
{
	// Create and save cache
	std::unordered_map<std::string, Annotation> cache;
	for (int i = 0; i < 100; ++i) {
		cache["image_" + std::to_string(i) + ".jpg"] = createAnnotation(LabelType::XYWH, 5);
	}
	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "thread_hash");

	// Multiple threads reading concurrently
	const int numThreads = 8;
	std::vector<std::thread> threads;
	std::atomic<int> successCount{ 0 };
	std::atomic<int> errorCount{ 0 };

	for (int i = 0; i < numThreads; ++i) {
		threads.emplace_back([this, &successCount, &errorCount]()
			{
				try {
					auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());
					if (loaded.size() == 100) {
						successCount++;
					}
				}
				catch (...) {
					errorCount++;
				}
			});
	}

	for (auto& t : threads) {
		t.join();
	}

	EXPECT_EQ(successCount, numThreads);
	EXPECT_EQ(errorCount, 0);
}

TEST_F(CacheComprehensiveTest, ThreadSafety_MultipleThreadsWritingDifferentCaches)
{
	const int numThreads = 8;
	std::vector<std::thread> threads;
	std::atomic<int> successCount{ 0 };

	for (int i = 0; i < numThreads; ++i) {
		threads.emplace_back([this, i, &successCount]()
			{
				try {
					fs::path cachePath = testDir / ("cache_" + std::to_string(i) + ".bin");
					std::unordered_map<std::string, Annotation> cache;

					for (int j = 0; j < 10; ++j) {
						cache["image_" + std::to_string(j) + ".jpg"] = createAnnotation(LabelType::XYWH, 3);
					}

					CacheManager::saveLabelCacheTo(cachePath.string(), cache, "thread_hash_" + std::to_string(i));
					auto loaded = CacheManager::loadLabelCacheFrom(cachePath.string());

					if (loaded.size() == 10) {
						successCount++;
					}
				}
				catch (...) {
					// Error
				}
			});
	}

	for (auto& t : threads) {
		t.join();
	}

	EXPECT_EQ(successCount, numThreads);
}

TEST_F(CacheComprehensiveTest, ThreadSafety_SerializationFromMultipleThreads)
{
	const int numThreads = 8;
	std::vector<std::thread> threads;
	std::atomic<int> successCount{ 0 };

	for (int i = 0; i < numThreads; ++i) {
		threads.emplace_back([i, &successCount]()
			{
				try {
					Annotation original = Annotation(LabelType::POLYGON);
					for (int j = 0; j < 10; ++j) {
						original.addObject(j, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });
					}

					std::string serialized = CacheManager::serializeAnnotation(original);
					Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

					if (deserialized.size() == 10) {
						successCount++;
					}
				}
				catch (...) {
					// Error
				}
			});
	}

	for (auto& t : threads) {
		t.join();
	}

	EXPECT_EQ(successCount, numThreads);
}

// ========================================
// SECTION 11: Edge Cases
// ========================================

TEST_F(CacheComprehensiveTest, EdgeCase_VeryLargePolygon_10000Points)
{
	Annotation ann(LabelType::POLYGON);

	// Create a very large polygon with 10000 points (20000 floats for x,y pairs)
	std::vector<float> largePolygon(20000);
	for (size_t i = 0; i < largePolygon.size(); ++i) {
		largePolygon[i] = static_cast<float>(i * 0.1f);
	}
	ann.addObject(5, largePolygon);

	std::string serialized = CacheManager::serializeAnnotation(ann);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	expectAnnotationEqual(deserialized, ann);
	EXPECT_EQ(deserialized.getPoints()[0].size(), 20000);
}

TEST_F(CacheComprehensiveTest, EdgeCase_NonExistentCacheFile)
{
	fs::path nonExistentPath = testDir / "does_not_exist.bin";

	EXPECT_THROW(
		CacheManager::loadLabelCacheFrom(nonExistentPath.string()),
		std::runtime_error
	);
}

TEST_F(CacheComprehensiveTest, EdgeCase_ReadOnlyCacheFile)
{
	// Create a cache file
	std::unordered_map<std::string, Annotation> cache;
	cache["test.jpg"] = createAnnotation(LabelType::XYWH, 3);
	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "test_hash");

	// Make it read-only
	fs::permissions(testCachePath,
		fs::perms::owner_read | fs::perms::group_read | fs::perms::others_read,
		fs::perm_options::replace);

	// Should still be able to read
	EXPECT_NO_THROW({
		auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());
		EXPECT_EQ(loaded.size(), 1);
		});

	// Restore permissions for cleanup
	fs::permissions(testCachePath, fs::perms::all, fs::perm_options::replace);
}

TEST_F(CacheComprehensiveTest, EdgeCase_UnicodeImagePaths)
{
	std::unordered_map<std::string, Annotation> cache;

	// UTF-8 encoded paths
	cache["images/?�진1.jpg"] = createAnnotation(LabelType::XYWH, 3);
	cache["images/?�片2.jpg"] = createAnnotation(LabelType::XYXY, 4);
	cache["images/?о?о3.jpg"] = createAnnotation(LabelType::POLYGON, 5);

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "unicode_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), 3);
	EXPECT_TRUE(loaded.count("images/?�진1.jpg"));
	EXPECT_TRUE(loaded.count("images/?�片2.jpg"));
	EXPECT_TRUE(loaded.count("images/?о?о3.jpg"));
}

TEST_F(CacheComprehensiveTest, EdgeCase_VeryLongImagePath)
{
	std::unordered_map<std::string, Annotation> cache;

	// Create a very long path
	std::string longPath = "images/";
	for (int i = 0; i < 50; ++i) {
		longPath += "very_long_subdirectory_name_" + std::to_string(i) + "/";
	}
	longPath += "image.jpg";

	cache[longPath] = createAnnotation(LabelType::XYWH, 3);

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "long_path_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), 1);
	EXPECT_TRUE(loaded.count(longPath));
}

TEST_F(CacheComprehensiveTest, EdgeCase_SpecialCharactersInPath)
{
	std::unordered_map<std::string, Annotation> cache;

	cache["images/test[1].jpg"] = createAnnotation(LabelType::XYWH, 3);
	cache["images/test(2).jpg"] = createAnnotation(LabelType::XYXY, 4);
	cache["images/test#3.jpg"] = createAnnotation(LabelType::POLYGON, 5);
	cache["images/test 4.jpg"] = createAnnotation(LabelType::XYWHR, 2);

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "special_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), 4);
	EXPECT_TRUE(loaded.count("images/test[1].jpg"));
	EXPECT_TRUE(loaded.count("images/test(2).jpg"));
	EXPECT_TRUE(loaded.count("images/test#3.jpg"));
	EXPECT_TRUE(loaded.count("images/test 4.jpg"));
}

TEST_F(CacheComprehensiveTest, EdgeCase_ManyObjectsInSingleAnnotation)
{
	Annotation ann(LabelType::XYWH);

	// Add 1000 objects to a single annotation
	for (int i = 0; i < 1000; ++i) {
		ann.addObject(i % 10, {
			static_cast<float>(i * 10),
			static_cast<float>(i * 10),
			50.0f,
			50.0f
			});
	}

	std::string serialized = CacheManager::serializeAnnotation(ann);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	expectAnnotationEqual(deserialized, ann);
	EXPECT_EQ(deserialized.size(), 1000);
}

TEST_F(CacheComprehensiveTest, EdgeCase_FloatingPointPrecision)
{
	Annotation original(LabelType::XYWH);

	// Add objects with very precise floating point values
	original.addObject(0, { 1.123456789f, 2.987654321f, 3.141592653f, 4.271828182f });
	original.addObject(1, { 0.000001f, 0.999999f, 123.456789f, 987.654321f });

	std::string serialized = CacheManager::serializeAnnotation(original);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	// Check that floating point values are preserved
	const auto& origPoints = original.getPoints();
	const auto& deserPoints = deserialized.getPoints();

	ASSERT_EQ(origPoints.size(), deserPoints.size());
	for (size_t i = 0; i < origPoints.size(); ++i) {
		ASSERT_EQ(origPoints[i].size(), deserPoints[i].size());
		for (size_t j = 0; j < origPoints[i].size(); ++j) {
			EXPECT_FLOAT_EQ(origPoints[i][j], deserPoints[i][j]);
		}
	}
}

// ========================================
// SECTION 12: Stress Tests
// ========================================

TEST_F(CacheComprehensiveTest, StressTest_RapidSaveLoadCycles)
{
	std::unordered_map<std::string, Annotation> cache;
	for (int i = 0; i < 50; ++i) {
		cache["image_" + std::to_string(i) + ".jpg"] = createAnnotation(LabelType::POLYGON, 5);
	}

	// Perform 100 rapid save/load cycles
	for (int cycle = 0; cycle < 100; ++cycle) {
		fs::path cyclePath = testDir / ("cycle_" + std::to_string(cycle) + ".bin");

		CacheManager::saveLabelCacheTo(cyclePath.string(), cache, "cycle_hash_" + std::to_string(cycle));
		auto loaded = CacheManager::loadLabelCacheFrom(cyclePath.string());

		ASSERT_EQ(loaded.size(), 50);

		// Clean up to prevent disk space issues
		fs::remove(cyclePath);
	}
}

TEST_F(CacheComprehensiveTest, StressTest_VeryLargeCache_FileSize)
{
	std::unordered_map<std::string, Annotation> cache;

	// Create a cache that will result in a large file
	for (int i = 0; i < 1000; ++i) {
		Annotation ann(LabelType::POLYGON);
		// Each object has 100 points (200 floats)
		for (int j = 0; j < 10; ++j) {
			std::vector<float> polygon(200);
			for (size_t k = 0; k < polygon.size(); ++k) {
				polygon[k] = static_cast<float>(i * 1000 + j * 10 + k);
			}
			ann.addObject(j % 10, polygon);
		}
		cache["image_" + std::to_string(i) + ".jpg"] = ann;
	}

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "large_file_hash");

	// Verify file exists and has reasonable size
	ASSERT_TRUE(fs::exists(testCachePath));
	auto fileSize = fs::file_size(testCachePath);
	EXPECT_GT(fileSize, 0);

	// Load and verify
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());
	ASSERT_EQ(loaded.size(), 1000);
}

TEST_F(CacheComprehensiveTest, StressTest_HashComputationConsistency)
{
	// Create test dataset
	fs::path subdir1 = testDir / "data1";
	fs::path subdir2 = testDir / "data2";
	fs::create_directories(subdir1);
	fs::create_directories(subdir2);

	for (int i = 0; i < 20; ++i) {
		std::ofstream((subdir1 / ("file_" + std::to_string(i) + ".txt")).string()) << "content_" << i;
		std::ofstream((subdir2 / ("file_" + std::to_string(i) + ".txt")).string()) << "content_" << i;
	}

	// Compute hash multiple times
	std::vector<std::string> hashes;
	for (int i = 0; i < 10; ++i) {
		hashes.push_back(CacheManager::computeHash(testDir.string()));
	}

	// All hashes should be identical
	for (size_t i = 1; i < hashes.size(); ++i) {
		EXPECT_EQ(hashes[0], hashes[i]);
	}
}
