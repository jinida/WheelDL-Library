#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Builder/Factory/HeadFactory.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <yaml-cpp/yaml.h>

using namespace WheelDL;
using namespace WheelDL::Model::Builder;
using namespace WheelDL::Utils;

class HeadFactoryTest : public ::testing::Test {
protected:
	ModuleBuildContext createContext(
		int64_t inputChannels = 3,
		const std::string& defaultAct = "SiLU",
		int64_t numClasses = 80)
	{
		ModuleBuildContext ctx;
		ctx.inputChannels = inputChannels;
		ctx.repeats = 1;
		ctx.defaultAct = defaultAct;
		ctx.numClasses = numClasses;
		ctx.headChannels = {64, 128, 256};  // Typical head channels

		// Default scale function (no scaling)
		ctx.applyScale = [](int64_t channels, int64_t repeats) {
			return std::make_pair(channels, repeats);
		};

		return ctx;
	}

	YAML::Node createArgsNode(const std::vector<int64_t>& values) {
		YAML::Node node;
		for (const auto& val : values) {
			node.push_back(val);
		}
		return node;
	}

	YAML::Node createArgsNode(const std::vector<std::string>& values) {
		YAML::Node node;
		for (const auto& val : values) {
			node.push_back(val);
		}
		return node;
	}

	YAML::Node createMixedArgsNode(const std::string& str, int64_t val1, bool val2) {
		YAML::Node node;
		node.push_back(str);
		node.push_back(val1);
		node.push_back(val2);
		return node;
	}
};

// ========== Detect Factory Tests ==========

TEST_F(HeadFactoryTest, DetectFactory_SupportedTypes_ReturnsDetect) {
	DetectFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "Detect");
}

TEST_F(HeadFactoryTest, DetectFactory_Create_ValidParams_Success) {
	DetectFactory factory;
	auto ctx = createContext(256, "SiLU", 80);
	YAML::Node emptyArgs;

	EXPECT_NO_THROW({
		auto result = factory.create(emptyArgs, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_GT(result.outputChannels, 0);
	});
}

TEST_F(HeadFactoryTest, DetectFactory_Create_DifferentClassCounts_Success) {
	DetectFactory factory;
	YAML::Node emptyArgs;

	// Test 10 classes (COCO-subset)
	auto ctx10 = createContext(256, "SiLU", 10);
	auto result10 = factory.create(emptyArgs, ctx10);
	EXPECT_FALSE(result10.module.is_empty());

	// Test 80 classes (COCO)
	auto ctx80 = createContext(256, "SiLU", 80);
	auto result80 = factory.create(emptyArgs, ctx80);
	EXPECT_FALSE(result80.module.is_empty());

	// Test 1000 classes (ImageNet)
	auto ctx1000 = createContext(256, "SiLU", 1000);
	auto result1000 = factory.create(emptyArgs, ctx1000);
	EXPECT_FALSE(result1000.module.is_empty());
}

TEST_F(HeadFactoryTest, DetectFactory_Create_WithHeadChannels_Success) {
	DetectFactory factory;
	auto ctx = createContext(256, "SiLU", 80);
	ctx.headChannels = {128, 256, 512};  // Multi-scale head channels
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_GT(result.outputChannels, 0);
}

TEST_F(HeadFactoryTest, DetectFactory_Create_SingleClass_Success) {
	DetectFactory factory;
	auto ctx = createContext(256, "SiLU", 1);
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_GT(result.outputChannels, 0);
}

// ========== OBB Factory Tests ==========

TEST_F(HeadFactoryTest, OBBFactory_SupportedTypes_ReturnsOBB) {
	OBBFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "OBB");
}

TEST_F(HeadFactoryTest, OBBFactory_Create_ValidParams_Success) {
	OBBFactory factory;
	auto ctx = createContext(256, "SiLU", 80);
	YAML::Node emptyArgs;

	EXPECT_NO_THROW({
		auto result = factory.create(emptyArgs, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_GT(result.outputChannels, 0);
	});
}

TEST_F(HeadFactoryTest, OBBFactory_Create_DifferentClassCounts_Success) {
	OBBFactory factory;
	YAML::Node emptyArgs;

	// Test 15 classes (DOTA-subset)
	auto ctx15 = createContext(256, "SiLU", 15);
	auto result15 = factory.create(emptyArgs, ctx15);
	EXPECT_FALSE(result15.module.is_empty());

	// Test 80 classes
	auto ctx80 = createContext(256, "SiLU", 80);
	auto result80 = factory.create(emptyArgs, ctx80);
	EXPECT_FALSE(result80.module.is_empty());
}

TEST_F(HeadFactoryTest, OBBFactory_Create_WithHeadChannels_Success) {
	OBBFactory factory;
	auto ctx = createContext(256, "SiLU", 15);
	ctx.headChannels = {128, 256, 512};
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_GT(result.outputChannels, 0);
}

// ========== Classify Factory Tests ==========

TEST_F(HeadFactoryTest, ClassifyFactory_SupportedTypes_ReturnsClassify) {
	ClassifyFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "Classify");
}

TEST_F(HeadFactoryTest, ClassifyFactory_Create_ValidParams_Success) {
	ClassifyFactory factory;
	auto ctx = createContext(512, "SiLU", 1000);
	YAML::Node emptyArgs;

	EXPECT_NO_THROW({
		auto result = factory.create(emptyArgs, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_EQ(result.outputChannels, 1000);
	});
}

TEST_F(HeadFactoryTest, ClassifyFactory_Create_DifferentClassCounts_Success) {
	ClassifyFactory factory;
	YAML::Node emptyArgs;

	// Test 10 classes
	auto ctx10 = createContext(512, "SiLU", 10);
	auto result10 = factory.create(emptyArgs, ctx10);
	EXPECT_FALSE(result10.module.is_empty());
	EXPECT_EQ(result10.outputChannels, 10);

	// Test 100 classes
	auto ctx100 = createContext(512, "SiLU", 100);
	auto result100 = factory.create(emptyArgs, ctx100);
	EXPECT_FALSE(result100.module.is_empty());
	EXPECT_EQ(result100.outputChannels, 100);

	// Test 1000 classes
	auto ctx1000 = createContext(512, "SiLU", 1000);
	auto result1000 = factory.create(emptyArgs, ctx1000);
	EXPECT_FALSE(result1000.module.is_empty());
	EXPECT_EQ(result1000.outputChannels, 1000);
}

TEST_F(HeadFactoryTest, ClassifyFactory_Create_DifferentChannels_Success) {
	ClassifyFactory factory;
	YAML::Node emptyArgs;

	// Test 256 input channels
	auto ctx256 = createContext(256, "SiLU", 1000);
	auto result256 = factory.create(emptyArgs, ctx256);
	EXPECT_FALSE(result256.module.is_empty());

	// Test 512 input channels
	auto ctx512 = createContext(512, "SiLU", 1000);
	auto result512 = factory.create(emptyArgs, ctx512);
	EXPECT_FALSE(result512.module.is_empty());

	// Test 1024 input channels
	auto ctx1024 = createContext(1024, "SiLU", 1000);
	auto result1024 = factory.create(emptyArgs, ctx1024);
	EXPECT_FALSE(result1024.module.is_empty());
}

TEST_F(HeadFactoryTest, ClassifyFactory_Create_BinaryClassification_Success) {
	ClassifyFactory factory;
	auto ctx = createContext(512, "SiLU", 2);
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 2);
}

// ========== Segment Factory Tests ==========

TEST_F(HeadFactoryTest, SegmentFactory_SupportedTypes_ReturnsSegment) {
	SegmentFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "Segment");
}

TEST_F(HeadFactoryTest, SegmentFactory_Create_ValidParams_Success) {
	SegmentFactory factory;
	auto ctx = createContext(256, "ReLU", 21);  // PASCAL VOC classes
	YAML::Node emptyArgs;

	EXPECT_NO_THROW({
		auto result = factory.create(emptyArgs, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_EQ(result.outputChannels, 21);
	});
}

TEST_F(HeadFactoryTest, SegmentFactory_Create_DifferentClassCounts_Success) {
	SegmentFactory factory;
	YAML::Node emptyArgs;

	// Test 21 classes (PASCAL VOC)
	auto ctx21 = createContext(256, "ReLU", 21);
	auto result21 = factory.create(emptyArgs, ctx21);
	EXPECT_FALSE(result21.module.is_empty());
	EXPECT_EQ(result21.outputChannels, 21);

	// Test 80 classes (COCO)
	auto ctx80 = createContext(256, "ReLU", 80);
	auto result80 = factory.create(emptyArgs, ctx80);
	EXPECT_FALSE(result80.module.is_empty());
	EXPECT_EQ(result80.outputChannels, 80);

	// Test 150 classes (ADE20K)
	auto ctx150 = createContext(256, "ReLU", 150);
	auto result150 = factory.create(emptyArgs, ctx150);
	EXPECT_FALSE(result150.module.is_empty());
	EXPECT_EQ(result150.outputChannels, 150);
}

TEST_F(HeadFactoryTest, SegmentFactory_Create_WithSiLU_Success) {
	SegmentFactory factory;
	auto ctx = createContext(256, "SiLU", 21);
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 21);
}

TEST_F(HeadFactoryTest, SegmentFactory_Create_BinarySegmentation_Success) {
	SegmentFactory factory;
	auto ctx = createContext(256, "ReLU", 1);  // Binary segmentation
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 1);
}

// ========== Anomaly Factory Tests ==========

TEST_F(HeadFactoryTest, AnomalyFactory_SupportedTypes_ReturnsAnomaly) {
	AnomalyFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "Anomaly");
}

TEST_F(HeadFactoryTest, AnomalyFactory_Create_EfficientAD_Success) {
	AnomalyFactory factory;
	auto ctx = createContext(256, "SiLU", 1);

	YAML::Node argsNode;
	argsNode.push_back("EfficientAD");
	argsNode.push_back(384);   // outChannels
	argsNode.push_back(true);  // small

	EXPECT_NO_THROW({
		auto result = factory.create(argsNode, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_GT(result.outputChannels, 0);
	});
}

TEST_F(HeadFactoryTest, AnomalyFactory_Create_EfficientAD_LargeModel_Success) {
	AnomalyFactory factory;
	auto ctx = createContext(256, "SiLU", 1);

	YAML::Node argsNode;
	argsNode.push_back("EfficientAD");
	argsNode.push_back(384);    // outChannels
	argsNode.push_back(false);  // large model

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_GT(result.outputChannels, 0);
}

TEST_F(HeadFactoryTest, AnomalyFactory_Create_PatchCore_Success) {
	AnomalyFactory factory;
	auto ctx = createContext(256, "SiLU", 1);

	YAML::Node argsNode;
	argsNode.push_back("PatchCore");
	argsNode.push_back(9);      // numNeighbors
	argsNode.push_back(25600);  // maxMemoryBankPatches

	EXPECT_NO_THROW({
		auto result = factory.create(argsNode, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_GT(result.outputChannels, 0);
	});
}

TEST_F(HeadFactoryTest, AnomalyFactory_Create_PatchCore_DifferentParams_Success) {
	AnomalyFactory factory;
	auto ctx = createContext(256, "SiLU", 1);

	YAML::Node argsNode;
	argsNode.push_back("PatchCore");
	argsNode.push_back(5);      // fewer neighbors
	argsNode.push_back(10000);  // smaller memory bank

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_GT(result.outputChannels, 0);
}

TEST_F(HeadFactoryTest, AnomalyFactory_Create_DefaultEfficientAD_Success) {
	AnomalyFactory factory;
	auto ctx = createContext(256, "SiLU", 1);

	YAML::Node argsNode;
	argsNode.push_back("EfficientAD");  // Use defaults for other params

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_GT(result.outputChannels, 0);
}

TEST_F(HeadFactoryTest, AnomalyFactory_Create_InvalidModelType_ThrowsException) {
	AnomalyFactory factory;
	auto ctx = createContext(256, "SiLU", 1);

	YAML::Node argsNode;
	argsNode.push_back("InvalidModel");

	EXPECT_THROW({
		auto result = factory.create(argsNode, ctx);
	}, std::runtime_error);
}

// ========== Edge Cases Tests ==========

TEST_F(HeadFactoryTest, DetectFactory_Create_EmptyHeadChannels_ThrowsException) {
	DetectFactory factory;
	auto ctx = createContext(256, "SiLU", 80);
	ctx.headChannels = {};  // Empty head channels - should be rejected
	YAML::Node emptyArgs;

	// Library correctly rejects empty head channels
	EXPECT_THROW({
		auto result = factory.create(emptyArgs, ctx);
	}, std::exception);
}

TEST_F(HeadFactoryTest, ClassifyFactory_Create_LargeClassCount_Success) {
	ClassifyFactory factory;
	auto ctx = createContext(2048, "SiLU", 10000);
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 10000);
}

TEST_F(HeadFactoryTest, SegmentFactory_Create_SmallChannels_Success) {
	SegmentFactory factory;
	auto ctx = createContext(32, "ReLU", 21);
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 21);
}

TEST_F(HeadFactoryTest, OBBFactory_Create_LargeChannels_Success) {
	OBBFactory factory;
	auto ctx = createContext(1024, "SiLU", 80);
	ctx.headChannels = {256, 512, 1024};
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_GT(result.outputChannels, 0);
}

TEST_F(HeadFactoryTest, AnomalyFactory_Create_EfficientAD_DifferentChannels_Success) {
	AnomalyFactory factory;
	auto ctx = createContext(256, "SiLU", 1);

	// Test 256 channels
	YAML::Node args256;
	args256.push_back("EfficientAD");
	args256.push_back(256);
	args256.push_back(true);
	auto result256 = factory.create(args256, ctx);
	EXPECT_FALSE(result256.module.is_empty());

	// Test 512 channels
	YAML::Node args512;
	args512.push_back("EfficientAD");
	args512.push_back(512);
	args512.push_back(true);
	auto result512 = factory.create(args512, ctx);
	EXPECT_FALSE(result512.module.is_empty());
}
