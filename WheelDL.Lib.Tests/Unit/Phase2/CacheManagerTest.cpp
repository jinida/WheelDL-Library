#include "pch.h"
#include "gtest/gtest.h"
#include "WheelDL.Lib/Data/Cache/CacheManager.h"
#include <filesystem>
#include <fstream>

using namespace WheelDL::Data::Cache;
using WheelDL::Data::Annotation;
namespace fs = std::filesystem;

/**
 * @class CacheManagerTest
 * @brief Test suite for CacheManager utility class
 */
class CacheManagerTest : public ::testing::Test
{
protected:
	fs::path testDir;
	fs::path testCachePath;

	void SetUp() override {
		// Create temporary test directory
		testDir = fs::temp_directory_path() / "CacheManagerTest";
		testCachePath = testDir / "test_cache.bin";

		// Clean up if exists
		if (fs::exists(testDir)) {
			fs::remove_all(testDir);
		}
		fs::create_directories(testDir);
	}

	void TearDown() override {
		// Clean up test directory
		if (fs::exists(testDir)) {
			fs::remove_all(testDir);
		}
	}

	Annotation createSampleAnnotation() {
		Annotation ann(WheelDL::Data::LabelType::XYWH);
		ann.addObject(0, { 10.0f, 20.0f, 30.0f, 40.0f });              // Object 1: bbox
		ann.addObject(1, { 50.0f, 60.0f, 70.0f, 80.0f });              // Object 2: bbox
		ann.addObject(2, { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f });      // Object 3: polygon
		return ann;
	}

	void expectAnnotationEqual(const Annotation& a, const Annotation& b) {
		EXPECT_EQ(a.getLabelType(), b.getLabelType());
		EXPECT_EQ(a.getClasses(), b.getClasses());
		EXPECT_EQ(a.getPoints(), b.getPoints());
	}
};

// ========== Serialization Tests ==========

TEST_F(CacheManagerTest, SerializeDeserialize_BasicAnnotation) {
	Annotation original = createSampleAnnotation();

	std::string serialized = CacheManager::serializeAnnotation(original);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	expectAnnotationEqual(deserialized, original);
}

TEST_F(CacheManagerTest, SerializeDeserialize_EmptyAnnotation) {
	Annotation original(WheelDL::Data::LabelType::NONE);
	// All fields are empty

	std::string serialized = CacheManager::serializeAnnotation(original);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	expectAnnotationEqual(deserialized, original);
}

TEST_F(CacheManagerTest, SerializeDeserialize_LargeAnnotation) {
	Annotation original(WheelDL::Data::LabelType::POLYGON);

	// 1000 objects with varying point counts
	for (size_t i = 0; i < 1000; ++i) {
		// Some objects have 4 points (bbox), others have more (polygons)
		size_t numCoords = (i % 3 == 0) ? ((i % 50) + 10) * 2 : 4;  // Varying sizes
		std::vector<float> points(numCoords);
		for (size_t j = 0; j < numCoords; ++j) {
			points[j] = static_cast<float>(i * 10 + j);
		}
		original.addObject(i % 10, points);
	}

	std::string serialized = CacheManager::serializeAnnotation(original);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	expectAnnotationEqual(deserialized, original);
}

// ========== Cache Save/Load Tests ==========

TEST_F(CacheManagerTest, SaveLoadCache_SingleEntry) {
	std::unordered_map<std::string, Annotation> cache;
	cache["image1.jpg"] = createSampleAnnotation();

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "test_hash");
	ASSERT_TRUE(fs::exists(testCachePath));

	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), 1);
	ASSERT_TRUE(loaded.count("image1.jpg"));
	expectAnnotationEqual(loaded["image1.jpg"], cache["image1.jpg"]);
}

TEST_F(CacheManagerTest, SaveLoadCache_MultipleEntries) {
	std::unordered_map<std::string, Annotation> cache;

	Annotation ann1 = createSampleAnnotation();
	Annotation ann2(WheelDL::Data::LabelType::POLYGON);
	ann2.addObject(5, { 100.0f, 200.0f, 300.0f, 400.0f });                  // Bbox-like
	ann2.addObject(6, { 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f });         // Polygon

	cache["image1.jpg"] = ann1;
	cache["image2.jpg"] = ann2;
	cache["subfolder/image3.jpg"] = ann1;

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "test_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), 3);
	expectAnnotationEqual(loaded["image1.jpg"], ann1);
	expectAnnotationEqual(loaded["image2.jpg"], ann2);
	expectAnnotationEqual(loaded["subfolder/image3.jpg"], ann1);
}

TEST_F(CacheManagerTest, SaveLoadCache_EmptyCache) {
	std::unordered_map<std::string, Annotation> cache;

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "empty_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	EXPECT_EQ(loaded.size(), 0);
}

TEST_F(CacheManagerTest, LoadCache_FileNotFound) {
	fs::path nonExistentPath = testDir / "nonexistent.bin";

	EXPECT_THROW(
		CacheManager::loadLabelCacheFrom(nonExistentPath.string()),
		std::runtime_error
	);
}

TEST_F(CacheManagerTest, SaveCache_InvalidPath) {
	std::unordered_map<std::string, Annotation> cache;
	cache["test.jpg"] = createSampleAnnotation();

	// Invalid path (directory doesn't exist)
	std::string invalidPath = "Z:\\nonexistent_drive\\cache.bin";

	EXPECT_THROW(
		CacheManager::saveLabelCacheTo(invalidPath, cache, "test_hash"),
		std::runtime_error
	);
}

// ========== Hash Computation Tests ==========

TEST_F(CacheManagerTest, ComputeHash_SameDirectory_SameHash) {
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

TEST_F(CacheManagerTest, ComputeHash_DifferentContent_DifferentHash) {
	// First scenario
	fs::path file1 = testDir / "file1.txt";
	std::ofstream(file1.string()) << "content1";
	std::string hash1 = CacheManager::computeHash(testDir.string());

	// Modify content
	std::ofstream(file1.string(), std::ios::trunc) << "different_content";
	std::string hash2 = CacheManager::computeHash(testDir.string());

	EXPECT_NE(hash1, hash2);
}

TEST_F(CacheManagerTest, ComputeHash_AdditionalFile_DifferentHash) {
	fs::path file1 = testDir / "file1.txt";
	std::ofstream(file1.string()) << "content1";
	std::string hash1 = CacheManager::computeHash(testDir.string());

	// Add new file
	fs::path file2 = testDir / "file2.txt";
	std::ofstream(file2.string()) << "content2";
	std::string hash2 = CacheManager::computeHash(testDir.string());

	EXPECT_NE(hash1, hash2);
}

TEST_F(CacheManagerTest, ComputeHash_EmptyDirectory) {
	std::string hash = CacheManager::computeHash(testDir.string());
	EXPECT_FALSE(hash.empty());
}

TEST_F(CacheManagerTest, ComputeHash_NonExistentDirectory) {
	fs::path nonExistentDir = testDir / "nonexistent";

	EXPECT_THROW(
		CacheManager::computeHash(nonExistentDir.string()),
		std::runtime_error
	);
}

TEST_F(CacheManagerTest, ComputeHash_NestedDirectories) {
	// Create nested structure
	fs::path subdir = testDir / "subdir";
	fs::create_directories(subdir);

	std::ofstream((testDir / "file1.txt").string()) << "content1";
	std::ofstream((subdir / "file2.txt").string()) << "content2";

	std::string hash = CacheManager::computeHash(testDir.string());
	EXPECT_FALSE(hash.empty());

	// Verify hash changes when nested file changes
	{
		std::ofstream file((subdir / "file2.txt").string(), std::ios::trunc);
		file << "modified_content_that_is_much_longer_to_ensure_different_hash";
		file.close();
	}
	std::string hash2 = CacheManager::computeHash(testDir.string());

	// Hash should change when file content changes
	EXPECT_NE(hash, hash2);
}

// ========== Cache Verification Tests ==========

TEST_F(CacheManagerTest, VerifyCache_ValidCache) {
	// Create test dataset
	fs::path file1 = testDir / "file1.txt";
	std::ofstream(file1.string()) << "content";

	std::string datasetHash = CacheManager::computeHash(testDir.string());

	// Create cache with matching hash
	std::unordered_map<std::string, Annotation> cache;
	cache["test.jpg"] = createSampleAnnotation();
	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, datasetHash);

	// Verify with matching hash
	bool isValid = CacheManager::verifyCacheFrom(testCachePath.string(), datasetHash);

	// Should be valid because we saved with the same hash
	EXPECT_TRUE(isValid);
}

TEST_F(CacheManagerTest, VerifyCache_NonExistentCache) {
	fs::path nonExistentPath = testDir / "nonexistent.bin";
	bool isValid = CacheManager::verifyCacheFrom(nonExistentPath.string(), "any_hash");

	EXPECT_FALSE(isValid);
}

TEST_F(CacheManagerTest, VerifyCache_DifferentHash) {
	std::unordered_map<std::string, Annotation> cache;
	cache["test.jpg"] = createSampleAnnotation();
	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "original_hash");

	bool isValid = CacheManager::verifyCacheFrom(testCachePath.string(), "wrong_hash");

	EXPECT_FALSE(isValid);
}

// ========== Edge Cases ==========

TEST_F(CacheManagerTest, Annotation_LargePolygon) {
	Annotation ann = createSampleAnnotation();

	// Add a very large polygon (10000 points = 20000 floats for x,y pairs)
	std::vector<float> largePolygon(20000);
	for (size_t i = 0; i < largePolygon.size(); ++i) {
		largePolygon[i] = static_cast<float>(i * 0.1f);
	}
	ann.addObject(99, largePolygon);

	std::string serialized = CacheManager::serializeAnnotation(ann);
	Annotation deserialized = CacheManager::deserializeAnnotation(serialized);

	expectAnnotationEqual(deserialized, ann);
}

TEST_F(CacheManagerTest, Cache_UnicodeImagePaths) {
	std::unordered_map<std::string, Annotation> cache;

	// UTF-8 encoded paths
	cache["images/사진1.jpg"] = createSampleAnnotation();
	cache["images/图片2.jpg"] = createSampleAnnotation();

	CacheManager::saveLabelCacheTo(testCachePath.string(), cache, "unicode_hash");
	auto loaded = CacheManager::loadLabelCacheFrom(testCachePath.string());

	ASSERT_EQ(loaded.size(), 2);
	EXPECT_TRUE(loaded.count("images/사진1.jpg"));
	EXPECT_TRUE(loaded.count("images/图片2.jpg"));
}
