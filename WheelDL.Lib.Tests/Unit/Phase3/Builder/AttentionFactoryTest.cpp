#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Builder/Factory/AttentionFactory.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <yaml-cpp/yaml.h>

using namespace WheelDL;
using namespace WheelDL::Model::Builder;
using namespace WheelDL::Utils;

class AttentionFactoryTest : public ::testing::Test {
protected:
	ModuleBuildContext createContext(
		int64_t inputChannels = 3,
		const std::string& defaultAct = "SiLU")
	{
		ModuleBuildContext ctx;
		ctx.inputChannels = inputChannels;
		ctx.repeats = 1;
		ctx.defaultAct = defaultAct;
		ctx.numClasses = 80;

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
};

// ========== PSA Factory Tests ==========

TEST_F(AttentionFactoryTest, PSAFactory_SupportedTypes_ReturnsPSA) {
	PSAFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "PSA");
}

TEST_F(AttentionFactoryTest, PSAFactory_Create_ValidParams_Success) {
	PSAFactory factory;
	auto ctx = createContext(256, "SiLU");
	auto argsNode = createArgsNode({256});  // out_channels

	EXPECT_NO_THROW({
		auto result = factory.create(argsNode, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_EQ(result.outputChannels, 256);
	});
}

TEST_F(AttentionFactoryTest, PSAFactory_Create_DifferentChannels_Success) {
	PSAFactory factory;
	auto ctx = createContext(128, "SiLU");

	// Test 128 channels
	auto args128 = createArgsNode({128});
	auto result128 = factory.create(args128, ctx);
	EXPECT_FALSE(result128.module.is_empty());
	EXPECT_EQ(result128.outputChannels, 128);

	// Test 256 channels
	ctx.inputChannels = 256;
	auto args256 = createArgsNode({256});
	auto result256 = factory.create(args256, ctx);
	EXPECT_FALSE(result256.module.is_empty());
	EXPECT_EQ(result256.outputChannels, 256);

	// Test 512 channels
	ctx.inputChannels = 512;
	auto args512 = createArgsNode({512});
	auto result512 = factory.create(args512, ctx);
	EXPECT_FALSE(result512.module.is_empty());
	EXPECT_EQ(result512.outputChannels, 512);
}

TEST_F(AttentionFactoryTest, PSAFactory_Create_WithScaling_AppliesCorrectly) {
	PSAFactory factory;
	auto ctx = createContext(256, "SiLU");

	// Apply 0.5x width scaling
	ctx.applyScale = [](int64_t channels, int64_t repeats) {
		return std::make_pair(channels / 2, repeats);
	};

	auto argsNode = createArgsNode({512});
	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 256);  // 512 * 0.5 = 256
}

TEST_F(AttentionFactoryTest, PSAFactory_Create_SmallChannels_Success) {
	PSAFactory factory;
	auto ctx = createContext(32, "SiLU");
	auto argsNode = createArgsNode({32});  // PSA requires c1 == c2

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 32);  // Same as input
}

TEST_F(AttentionFactoryTest, PSAFactory_Create_LargeChannels_Success) {
	PSAFactory factory;
	auto ctx = createContext(1024, "SiLU");
	auto argsNode = createArgsNode({1024});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 1024);
}

// ========== CBAM Factory Tests ==========

TEST_F(AttentionFactoryTest, CBAMFactory_SupportedTypes_ReturnsCBAM) {
	CBAMFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "CBAM");
}

TEST_F(AttentionFactoryTest, CBAMFactory_Create_NoArgs_Success) {
	CBAMFactory factory;
	auto ctx = createContext(256, "SiLU");
	YAML::Node emptyArgs;

	EXPECT_NO_THROW({
		auto result = factory.create(emptyArgs, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_EQ(result.outputChannels, 256);  // Output = input for CBAM
	});
}

TEST_F(AttentionFactoryTest, CBAMFactory_Create_DifferentChannels_Success) {
	CBAMFactory factory;
	YAML::Node emptyArgs;

	// Test 64 channels
	auto ctx64 = createContext(64, "SiLU");
	auto result64 = factory.create(emptyArgs, ctx64);
	EXPECT_FALSE(result64.module.is_empty());
	EXPECT_EQ(result64.outputChannels, 64);

	// Test 128 channels
	auto ctx128 = createContext(128, "SiLU");
	auto result128 = factory.create(emptyArgs, ctx128);
	EXPECT_FALSE(result128.module.is_empty());
	EXPECT_EQ(result128.outputChannels, 128);

	// Test 512 channels
	auto ctx512 = createContext(512, "SiLU");
	auto result512 = factory.create(emptyArgs, ctx512);
	EXPECT_FALSE(result512.module.is_empty());
	EXPECT_EQ(result512.outputChannels, 512);
}

TEST_F(AttentionFactoryTest, CBAMFactory_Create_PreservesChannels_Success) {
	CBAMFactory factory;
	auto ctx = createContext(256, "SiLU");
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, ctx.inputChannels);  // CBAM preserves channels
}

TEST_F(AttentionFactoryTest, CBAMFactory_Create_SmallChannels_Success) {
	CBAMFactory factory;
	auto ctx = createContext(16, "SiLU");
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 16);
}

TEST_F(AttentionFactoryTest, CBAMFactory_Create_LargeChannels_Success) {
	CBAMFactory factory;
	auto ctx = createContext(2048, "SiLU");
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 2048);
}

// ========== Attention Factory Tests ==========

TEST_F(AttentionFactoryTest, AttentionFactory_SupportedTypes_ReturnsAttention) {
	AttentionFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "Attention");
}

TEST_F(AttentionFactoryTest, AttentionFactory_Create_ValidParams_Success) {
	AttentionFactory factory;
	auto ctx = createContext(256, "SiLU");
	auto argsNode = createArgsNode({8});  // num_heads

	EXPECT_NO_THROW({
		auto result = factory.create(argsNode, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_EQ(result.outputChannels, 256);  // Output = input for self-attention
	});
}

TEST_F(AttentionFactoryTest, AttentionFactory_Create_DifferentHeadCounts_Success) {
	AttentionFactory factory;
	auto ctx = createContext(256, "SiLU");

	// Test 4 heads
	auto args4 = createArgsNode({4});
	auto result4 = factory.create(args4, ctx);
	EXPECT_FALSE(result4.module.is_empty());
	EXPECT_EQ(result4.outputChannels, 256);

	// Test 8 heads
	auto args8 = createArgsNode({8});
	auto result8 = factory.create(args8, ctx);
	EXPECT_FALSE(result8.module.is_empty());
	EXPECT_EQ(result8.outputChannels, 256);

	// Test 16 heads
	auto args16 = createArgsNode({16});
	auto result16 = factory.create(args16, ctx);
	EXPECT_FALSE(result16.module.is_empty());
	EXPECT_EQ(result16.outputChannels, 256);
}

TEST_F(AttentionFactoryTest, AttentionFactory_Create_DefaultHeads_Success) {
	AttentionFactory factory;
	auto ctx = createContext(256, "SiLU");
	YAML::Node emptyArgs;  // Uses default num_heads=8

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 256);
}

TEST_F(AttentionFactoryTest, AttentionFactory_Create_PreservesChannels_Success) {
	AttentionFactory factory;
	auto ctx = createContext(512, "SiLU");
	auto argsNode = createArgsNode({8});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, ctx.inputChannels);  // Self-attention preserves channels
}

TEST_F(AttentionFactoryTest, AttentionFactory_Create_DifferentChannelSizes_Success) {
	AttentionFactory factory;
	auto argsNode = createArgsNode({8});

	// Test 128 channels
	auto ctx128 = createContext(128, "SiLU");
	auto result128 = factory.create(argsNode, ctx128);
	EXPECT_FALSE(result128.module.is_empty());
	EXPECT_EQ(result128.outputChannels, 128);

	// Test 256 channels
	auto ctx256 = createContext(256, "SiLU");
	auto result256 = factory.create(argsNode, ctx256);
	EXPECT_FALSE(result256.module.is_empty());
	EXPECT_EQ(result256.outputChannels, 256);

	// Test 512 channels
	auto ctx512 = createContext(512, "SiLU");
	auto result512 = factory.create(argsNode, ctx512);
	EXPECT_FALSE(result512.module.is_empty());
	EXPECT_EQ(result512.outputChannels, 512);
}

// ========== Edge Cases Tests ==========

TEST_F(AttentionFactoryTest, PSAFactory_Create_WithDoubleScaling_Success) {
	PSAFactory factory;
	auto ctx = createContext(256, "SiLU");  // Set inputChannels to 256

	// Apply 2.0x width scaling
	ctx.applyScale = [](int64_t channels, int64_t repeats) {
		return std::make_pair(channels * 2, repeats);
	};

	// PSA requires c1 == c2
	// PSA(c1=ctx.inputChannels, c2=scaled output)
	// ctx.inputChannels = 256, args = 128 * 2 = 256 after scaling
	// So c1 (256) == c2 (256) ✓
	auto argsNode = createArgsNode({128});  // Will be scaled to 256
	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 256);  // 128 * 2.0 = 256
}

TEST_F(AttentionFactoryTest, AttentionFactory_Create_SingleChannel_Success) {
	AttentionFactory factory;
	auto ctx = createContext(1, "SiLU");
	auto argsNode = createArgsNode({1});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 1);
}

TEST_F(AttentionFactoryTest, CBAMFactory_Create_WithReLU_Success) {
	CBAMFactory factory;
	auto ctx = createContext(256, "ReLU");
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 256);
}

TEST_F(AttentionFactoryTest, PSAFactory_Create_WithMish_Success) {
	PSAFactory factory;
	auto ctx = createContext(256, "Mish");
	auto argsNode = createArgsNode({256});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 256);
}

TEST_F(AttentionFactoryTest, AttentionFactory_Create_LargeHeadCount_Success) {
	AttentionFactory factory;
	auto ctx = createContext(512, "SiLU");
	auto argsNode = createArgsNode({32});  // 32 heads

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 512);
}

TEST_F(AttentionFactoryTest, CBAMFactory_Create_OddChannels_Success) {
	CBAMFactory factory;
	auto ctx = createContext(127, "SiLU");
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 127);
}
