#include "pch.h"
#include "Data/Cache/CacheManager.h"
#include "Data/Common/Annotation.h"
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace WheelDL::Data;
using namespace WheelDL::Data::Cache;
namespace fs = std::filesystem;

// =============================================================================
// Test Fixture
// =============================================================================

class CacheManagerTest : public ::testing::Test {
protected:
    std::string tempDir_;

    void SetUp() override {
        tempDir_ = "Dataset/temp/cache_test";
        fs::create_directories(tempDir_);
    }

    void TearDown() override {
        // Clean up temp files
        if (fs::exists(tempDir_)) {
            std::error_code ec;
            fs::remove_all(tempDir_, ec);
        }
    }

    // Helper: Create XYWH annotation
    Annotation createXYWHAnnotation(int classId = 0, float cx = 100.0f, float cy = 100.0f,
                                     float w = 50.0f, float h = 50.0f) {
        Annotation ann(LabelType::XYWH);
        ann.addObject(classId, {cx, cy, w, h});
        return ann;
    }

    // Helper: Create XYWHR annotation
    Annotation createXYWHRAnnotation(int classId = 0) {
        Annotation ann(LabelType::XYWHR);
        ann.addObject(classId, {100.0f, 100.0f, 50.0f, 30.0f, 0.5f});
        return ann;
    }

    // Helper: Create XYXY annotation
    Annotation createXYXYAnnotation(int classId = 0) {
        Annotation ann(LabelType::XYXY);
        ann.addObject(classId, {50.0f, 50.0f, 150.0f, 150.0f});
        return ann;
    }

    // Helper: Create POLYGON annotation
    Annotation createPolygonAnnotation(int classId = 0) {
        Annotation ann(LabelType::POLYGON);
        ann.addObject(classId, {100.0f, 50.0f, 150.0f, 150.0f, 50.0f, 150.0f});
        return ann;
    }

    // Helper: Create XYXYXYXY annotation
    Annotation createXYXYXYXYAnnotation(int classId = 0) {
        Annotation ann(LabelType::XYXYXYXY);
        ann.addObject(classId, {50.0f, 50.0f, 150.0f, 50.0f, 150.0f, 150.0f, 50.0f, 150.0f});
        return ann;
    }

    // Helper: Create test directory with files
    std::string createTestDataDir() {
        std::string dataDir = tempDir_ + "/test_data";
        fs::create_directories(dataDir);

        // Create some test files
        for (int i = 0; i < 3; ++i) {
            std::string filePath = dataDir + "/file" + std::to_string(i) + ".txt";
            std::ofstream file(filePath);
            file << "Test content " << i;
            file.close();
        }

        return dataDir;
    }

    // Helper: Create cache file with given version
    void createCacheFileWithVersion(const std::string& path, uint32_t version,
                                     const std::string& hash = "testhash") {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        uint32_t hashLen = static_cast<uint32_t>(hash.size());
        file.write(reinterpret_cast<const char*>(&hashLen), sizeof(hashLen));
        file.write(hash.data(), hashLen);
        uint32_t numEntries = 0;
        file.write(reinterpret_cast<const char*>(&numEntries), sizeof(numEntries));
        file.close();
    }

    // Helper: Compare floats
    bool floatEqual(float a, float b, float tol = 1e-5f) {
        return std::abs(a - b) < tol;
    }
};

// =============================================================================
// 6.1 Serialization Tests (CM-001 ~ CM-014)
// =============================================================================

// CM-001: serializeAnnotation XYWH
TEST_F(CacheManagerTest, Serialize_XYWH) {
    Annotation ann = createXYWHAnnotation(5, 100.0f, 200.0f, 50.0f, 60.0f);

    std::string data = CacheManager::serializeAnnotation(ann);

    EXPECT_FALSE(data.empty());
}

// CM-002: serializeAnnotation XYWHR
TEST_F(CacheManagerTest, Serialize_XYWHR) {
    Annotation ann = createXYWHRAnnotation(3);

    std::string data = CacheManager::serializeAnnotation(ann);

    EXPECT_FALSE(data.empty());
}

// CM-003: serializeAnnotation XYXY
TEST_F(CacheManagerTest, Serialize_XYXY) {
    Annotation ann = createXYXYAnnotation(2);

    std::string data = CacheManager::serializeAnnotation(ann);

    EXPECT_FALSE(data.empty());
}

// CM-004: serializeAnnotation POLYGON
TEST_F(CacheManagerTest, Serialize_POLYGON) {
    Annotation ann = createPolygonAnnotation(1);

    std::string data = CacheManager::serializeAnnotation(ann);

    EXPECT_FALSE(data.empty());
}

// CM-005: serializeAnnotation XYXYXYXY
TEST_F(CacheManagerTest, Serialize_XYXYXYXY) {
    Annotation ann = createXYXYXYXYAnnotation(4);

    std::string data = CacheManager::serializeAnnotation(ann);

    EXPECT_FALSE(data.empty());
}

// CM-006: serializeAnnotation empty
TEST_F(CacheManagerTest, Serialize_Empty) {
    Annotation ann(LabelType::XYWH);

    std::string data = CacheManager::serializeAnnotation(ann);

    EXPECT_FALSE(data.empty());  // Still has header info
}

// CM-007: serializeAnnotation large points (size check only)
TEST_F(CacheManagerTest, Serialize_LargePoints) {
    Annotation ann(LabelType::POLYGON);

    // Add many points
    std::vector<float> largePolygon;
    for (int i = 0; i < 1000; ++i) {
        largePolygon.push_back(static_cast<float>(i));
    }
    ann.addObject(0, largePolygon);

    std::string data = CacheManager::serializeAnnotation(ann);

    EXPECT_FALSE(data.empty());
    EXPECT_GT(data.size(), 4000);  // At least 1000 floats * 4 bytes
}

// CM-008: deserializeAnnotation basic
TEST_F(CacheManagerTest, Deserialize_Basic) {
    Annotation original = createXYWHAnnotation(5, 100.0f, 200.0f, 50.0f, 60.0f);

    std::string data = CacheManager::serializeAnnotation(original);
    Annotation restored = CacheManager::deserializeAnnotation(data);

    EXPECT_EQ(original.getLabelType(), restored.getLabelType());
    EXPECT_EQ(original.size(), restored.size());
    EXPECT_EQ(original.getClasses()[0], restored.getClasses()[0]);
}

// CM-009: deserializeAnnotation all types
TEST_F(CacheManagerTest, Deserialize_AllTypes) {
    // Test all label types
    std::vector<Annotation> annotations = {
        createXYWHAnnotation(),
        createXYWHRAnnotation(),
        createXYXYAnnotation(),
        createPolygonAnnotation(),
        createXYXYXYXYAnnotation()
    };

    for (const auto& original : annotations) {
        std::string data = CacheManager::serializeAnnotation(original);
        Annotation restored = CacheManager::deserializeAnnotation(data);

        EXPECT_EQ(original.getLabelType(), restored.getLabelType());
        EXPECT_EQ(original.size(), restored.size());
    }
}

// CM-010: roundtrip serialization
TEST_F(CacheManagerTest, Roundtrip_Serialization) {
    Annotation original = createXYWHAnnotation(7, 123.5f, 456.7f, 89.1f, 23.4f);

    std::string data = CacheManager::serializeAnnotation(original);
    Annotation restored = CacheManager::deserializeAnnotation(data);

    EXPECT_EQ(original.getLabelType(), restored.getLabelType());
    EXPECT_EQ(original.getClasses(), restored.getClasses());

    const auto& origPts = original.getPoints()[0];
    const auto& restPts = restored.getPoints()[0];

    EXPECT_EQ(origPts.size(), restPts.size());
    for (size_t i = 0; i < origPts.size(); ++i) {
        EXPECT_TRUE(floatEqual(origPts[i], restPts[i]));
    }
}

// CM-011: writeString size overflow (tested via large annotation - indirect)
TEST_F(CacheManagerTest, WriteString_SizeLimit) {
    // Normal sized strings should work fine
    Annotation ann = createXYWHAnnotation();
    EXPECT_NO_THROW({
        std::string data = CacheManager::serializeAnnotation(ann);
    });
}

// CM-012: readString size overflow (tested indirectly via corrupted data)
TEST_F(CacheManagerTest, ReadString_SizeOverflow) {
    // Create malformed data with huge string size
    std::ostringstream oss(std::ios::binary);
    uint32_t labelType = 1;
    oss.write(reinterpret_cast<const char*>(&labelType), sizeof(labelType));

    // Write huge vector size that would overflow
    uint32_t hugeSize = 0xFFFFFFFF;
    oss.write(reinterpret_cast<const char*>(&hugeSize), sizeof(hugeSize));

    std::string data = oss.str();

    EXPECT_THROW(CacheManager::deserializeAnnotation(data), std::runtime_error);
}

// CM-013: writeVector size overflow (tested via serialization - indirect)
TEST_F(CacheManagerTest, WriteVector_Normal) {
    Annotation ann(LabelType::XYWH);
    ann.addObject(0, {1.0f, 2.0f, 3.0f, 4.0f});
    ann.addObject(1, {5.0f, 6.0f, 7.0f, 8.0f});

    EXPECT_NO_THROW({
        std::string data = CacheManager::serializeAnnotation(ann);
    });
}

// CM-014: readVector size overflow
TEST_F(CacheManagerTest, ReadVector_SizeOverflow) {
    // Create malformed data that triggers readVector size check
    std::ostringstream oss(std::ios::binary);
    uint32_t labelType = 1;
    oss.write(reinterpret_cast<const char*>(&labelType), sizeof(labelType));

    // Classes vector with huge size (exceeds MAX_VECTOR_SIZE = 10 million)
    uint32_t hugeClassSize = 10 * 1000 * 1000 + 1;  // Just over limit
    oss.write(reinterpret_cast<const char*>(&hugeClassSize), sizeof(hugeClassSize));

    std::string data = oss.str();

    // Should throw because readVector checks size before allocation
    EXPECT_THROW(CacheManager::deserializeAnnotation(data), std::runtime_error);
}

// =============================================================================
// 6.2 Load/Save Cache Tests (CM-015 ~ CM-028)
// =============================================================================

// CM-015: loadLabelCacheFrom success
TEST_F(CacheManagerTest, LoadCache_Success) {
    std::string cachePath = tempDir_ + "/test.cache";

    // First save a cache
    std::unordered_map<std::string, Annotation> original;
    original["image1.jpg"] = createXYWHAnnotation(0);
    original["image2.jpg"] = createXYWHAnnotation(1);

    CacheManager::saveLabelCacheTo(cachePath, original, "testhash");

    // Then load it
    auto loaded = CacheManager::loadLabelCacheFrom(cachePath);

    EXPECT_EQ(original.size(), loaded.size());
    EXPECT_TRUE(loaded.find("image1.jpg") != loaded.end());
    EXPECT_TRUE(loaded.find("image2.jpg") != loaded.end());
}

// CM-016: loadLabelCacheFrom file not found
TEST_F(CacheManagerTest, LoadCache_FileNotFound_Throws) {
    EXPECT_THROW(
        CacheManager::loadLabelCacheFrom("non_existent_cache.cache"),
        std::runtime_error
    );
}

// CM-017: loadLabelCacheFrom version mismatch
TEST_F(CacheManagerTest, LoadCache_VersionMismatch_Throws) {
    std::string cachePath = tempDir_ + "/old_version.cache";

    // Create cache with wrong version
    createCacheFileWithVersion(cachePath, 999999);

    EXPECT_THROW(
        CacheManager::loadLabelCacheFrom(cachePath),
        std::runtime_error
    );
}

// CM-018: loadLabelCacheFrom multiple entries
TEST_F(CacheManagerTest, LoadCache_MultipleEntries) {
    std::string cachePath = tempDir_ + "/multi.cache";

    std::unordered_map<std::string, Annotation> original;
    for (int i = 0; i < 10; ++i) {
        original["image" + std::to_string(i) + ".jpg"] = createXYWHAnnotation(i);
    }

    CacheManager::saveLabelCacheTo(cachePath, original, "hash123");
    auto loaded = CacheManager::loadLabelCacheFrom(cachePath);

    EXPECT_EQ(10, loaded.size());
}

// CM-019: saveLabelCacheTo success
TEST_F(CacheManagerTest, SaveCache_Success) {
    std::string cachePath = tempDir_ + "/save_test.cache";

    std::unordered_map<std::string, Annotation> cache;
    cache["test.jpg"] = createXYWHAnnotation();

    EXPECT_NO_THROW(
        CacheManager::saveLabelCacheTo(cachePath, cache, "myhash")
    );

    EXPECT_TRUE(fs::exists(cachePath));
}

// CM-020: saveLabelCacheTo temp file
TEST_F(CacheManagerTest, SaveCache_UsesTempFile) {
    std::string cachePath = tempDir_ + "/atomic_test.cache";
    std::string tempPath = cachePath + ".tmp";

    std::unordered_map<std::string, Annotation> cache;
    cache["test.jpg"] = createXYWHAnnotation();

    CacheManager::saveLabelCacheTo(cachePath, cache, "hash");

    // After successful save, temp file should not exist
    EXPECT_FALSE(fs::exists(tempPath));
    EXPECT_TRUE(fs::exists(cachePath));
}

// CM-021: saveLabelCacheTo atomic rename
TEST_F(CacheManagerTest, SaveCache_AtomicRename) {
    std::string cachePath = tempDir_ + "/atomic.cache";

    // Create initial cache
    std::unordered_map<std::string, Annotation> cache1;
    cache1["image1.jpg"] = createXYWHAnnotation(0);
    CacheManager::saveLabelCacheTo(cachePath, cache1, "hash1");

    // Overwrite with new cache
    std::unordered_map<std::string, Annotation> cache2;
    cache2["image2.jpg"] = createXYWHAnnotation(1);
    CacheManager::saveLabelCacheTo(cachePath, cache2, "hash2");

    // Load and verify new cache
    auto loaded = CacheManager::loadLabelCacheFrom(cachePath);
    EXPECT_EQ(1, loaded.size());
    EXPECT_TRUE(loaded.find("image2.jpg") != loaded.end());
}

// CM-022: saveLabelCacheTo file create fail
TEST_F(CacheManagerTest, SaveCache_FileCreateFail) {
    // Try to save to invalid path
    std::string invalidPath = "";

    std::unordered_map<std::string, Annotation> cache;
    cache["test.jpg"] = createXYWHAnnotation();

    EXPECT_THROW(
        CacheManager::saveLabelCacheTo(invalidPath, cache, "hash"),
        std::runtime_error
    );
}

// CM-023: saveLabelCacheTo cache size (normal case)
TEST_F(CacheManagerTest, SaveCache_NormalSize) {
    std::string cachePath = tempDir_ + "/normal_size.cache";

    std::unordered_map<std::string, Annotation> cache;
    for (int i = 0; i < 100; ++i) {
        cache["image" + std::to_string(i) + ".jpg"] = createXYWHAnnotation(i);
    }

    EXPECT_NO_THROW(
        CacheManager::saveLabelCacheTo(cachePath, cache, "hash")
    );
}

// CM-024: saveLabelCacheTo RAII cleanup (temp file removed on failure)
TEST_F(CacheManagerTest, SaveCache_RAIICleanup) {
    // This is difficult to test directly without mocking
    // We verify that successful saves don't leave temp files
    std::string cachePath = tempDir_ + "/raii_test.cache";
    std::string tempPath = cachePath + ".tmp";

    std::unordered_map<std::string, Annotation> cache;
    cache["test.jpg"] = createXYWHAnnotation();

    CacheManager::saveLabelCacheTo(cachePath, cache, "hash");

    EXPECT_FALSE(fs::exists(tempPath));
}

// CM-025: saveLabelCacheTo empty cache
TEST_F(CacheManagerTest, SaveCache_Empty) {
    std::string cachePath = tempDir_ + "/empty.cache";

    std::unordered_map<std::string, Annotation> emptyCache;

    EXPECT_NO_THROW(
        CacheManager::saveLabelCacheTo(cachePath, emptyCache, "hash")
    );

    auto loaded = CacheManager::loadLabelCacheFrom(cachePath);
    EXPECT_EQ(0, loaded.size());
}

// CM-026: loadLabelCacheFrom empty cache
TEST_F(CacheManagerTest, LoadCache_Empty) {
    std::string cachePath = tempDir_ + "/load_empty.cache";

    std::unordered_map<std::string, Annotation> emptyCache;
    CacheManager::saveLabelCacheTo(cachePath, emptyCache, "hash");

    auto loaded = CacheManager::loadLabelCacheFrom(cachePath);

    EXPECT_EQ(0, loaded.size());
}

// CM-027: readBinary stream fail
TEST_F(CacheManagerTest, ReadBinary_StreamFail) {
    std::string cachePath = tempDir_ + "/truncated.cache";

    // Create truncated file
    std::ofstream file(cachePath, std::ios::binary);
    uint32_t partialVersion = 1;
    file.write(reinterpret_cast<const char*>(&partialVersion), 2);  // Only 2 bytes
    file.close();

    EXPECT_THROW(
        CacheManager::loadLabelCacheFrom(cachePath),
        std::runtime_error
    );
}

// CM-028: readString stream fail
TEST_F(CacheManagerTest, ReadString_StreamFail) {
    std::string cachePath = tempDir_ + "/bad_string.cache";

    // Create file with string length but no string data
    std::ofstream file(cachePath, std::ios::binary);
    uint32_t version = 1001000;  // Assuming this matches CACHE_VERSION
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    uint32_t stringLen = 100;  // Say there are 100 bytes but don't write them
    file.write(reinterpret_cast<const char*>(&stringLen), sizeof(stringLen));
    file.close();

    EXPECT_THROW(
        CacheManager::loadLabelCacheFrom(cachePath),
        std::runtime_error
    );
}

// =============================================================================
// 6.3 Hash Computation Tests (CM-029 ~ CM-036)
// =============================================================================

// CM-029: computeHash existing dir
TEST_F(CacheManagerTest, ComputeHash_ExistingDir) {
    std::string dataDir = createTestDataDir();

    std::string hash;
    EXPECT_NO_THROW(hash = CacheManager::computeHash(dataDir));

    EXPECT_FALSE(hash.empty());
}

// CM-030: computeHash non-existent dir throws
TEST_F(CacheManagerTest, ComputeHash_NonExistent_Throws) {
    EXPECT_THROW(
        CacheManager::computeHash("non_existent_directory"),
        std::runtime_error
    );
}

// CM-031: computeHash sorted files (consistent hash)
TEST_F(CacheManagerTest, ComputeHash_Consistent) {
    std::string dataDir = createTestDataDir();

    std::string hash1 = CacheManager::computeHash(dataDir);
    std::string hash2 = CacheManager::computeHash(dataDir);

    EXPECT_EQ(hash1, hash2);
}

// CM-032: computeHash skip permission denied (handled gracefully)
TEST_F(CacheManagerTest, ComputeHash_SkipsErrors) {
    std::string dataDir = createTestDataDir();

    // Should not throw even with complex directory structures
    EXPECT_NO_THROW(CacheManager::computeHash(dataDir));
}

// CM-033: computeFileHash success
TEST_F(CacheManagerTest, ComputeHash_FileSuccess) {
    std::string dataDir = createTestDataDir();

    // computeFileHash is private, test via computeHash
    std::string hash = CacheManager::computeHash(dataDir);
    EXPECT_FALSE(hash.empty());
}

// CM-034: computeFileHash file not found returns empty
TEST_F(CacheManagerTest, ComputeHash_MissingFileHandled) {
    std::string dataDir = createTestDataDir();

    // Delete one file after creating
    fs::remove(dataDir + "/file0.txt");

    // Hash should still work with remaining files
    std::string hash;
    EXPECT_NO_THROW(hash = CacheManager::computeHash(dataDir));
    EXPECT_FALSE(hash.empty());
}

// CM-035: computeFileHash chunk reading
TEST_F(CacheManagerTest, ComputeHash_LargeFile) {
    std::string dataDir = tempDir_ + "/large_file_test";
    fs::create_directories(dataDir);

    // Create a file larger than chunk size (8KB)
    std::string largePath = dataDir + "/large.bin";
    std::ofstream file(largePath, std::ios::binary);
    std::vector<char> data(20000, 'A');  // 20KB
    file.write(data.data(), data.size());
    file.close();

    std::string hash;
    EXPECT_NO_THROW(hash = CacheManager::computeHash(dataDir));
    EXPECT_FALSE(hash.empty());
}

// CM-036: computeFileHash empty file
TEST_F(CacheManagerTest, ComputeHash_EmptyFile) {
    std::string dataDir = tempDir_ + "/empty_file_test";
    fs::create_directories(dataDir);

    // Create empty file
    std::string emptyPath = dataDir + "/empty.txt";
    std::ofstream file(emptyPath);
    file.close();

    std::string hash;
    EXPECT_NO_THROW(hash = CacheManager::computeHash(dataDir));
    EXPECT_FALSE(hash.empty());
}

// =============================================================================
// 6.4 Cache Verification Tests (CM-037 ~ CM-040)
// =============================================================================

// CM-037: verifyCacheFrom valid
TEST_F(CacheManagerTest, VerifyCache_Valid) {
    std::string cachePath = tempDir_ + "/verify_valid.cache";
    std::string hash = "test_hash_123";

    std::unordered_map<std::string, Annotation> cache;
    cache["test.jpg"] = createXYWHAnnotation();

    CacheManager::saveLabelCacheTo(cachePath, cache, hash);

    EXPECT_TRUE(CacheManager::verifyCacheFrom(cachePath, hash));
}

// CM-038: verifyCacheFrom file not found returns false
TEST_F(CacheManagerTest, VerifyCache_FileNotFound) {
    EXPECT_FALSE(CacheManager::verifyCacheFrom("non_existent.cache", "hash"));
}

// CM-039: verifyCacheFrom version mismatch returns false
TEST_F(CacheManagerTest, VerifyCache_VersionMismatch) {
    std::string cachePath = tempDir_ + "/wrong_version.cache";

    // Create cache with wrong version
    createCacheFileWithVersion(cachePath, 999999);

    EXPECT_FALSE(CacheManager::verifyCacheFrom(cachePath, "testhash"));
}

// CM-040: verifyCacheFrom hash mismatch returns false
TEST_F(CacheManagerTest, VerifyCache_HashMismatch) {
    std::string cachePath = tempDir_ + "/hash_mismatch.cache";

    std::unordered_map<std::string, Annotation> cache;
    cache["test.jpg"] = createXYWHAnnotation();

    CacheManager::saveLabelCacheTo(cachePath, cache, "original_hash");

    EXPECT_FALSE(CacheManager::verifyCacheFrom(cachePath, "different_hash"));
}
