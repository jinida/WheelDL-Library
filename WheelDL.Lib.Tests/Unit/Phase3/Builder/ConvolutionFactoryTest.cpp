#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Builder/Factory/ConvolutionFactory.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <yaml-cpp/yaml.h>

using namespace WheelDL;
using namespace WheelDL::Model::Builder;
using namespace WheelDL::Utils;

class ConvolutionFactoryTest : public ::testing::Test {
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

// ========== Conv Factory Tests ==========

TEST_F(ConvolutionFactoryTest, ConvFactory_SupportedTypes_ReturnsConv) {
	ConvFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "Conv");
}

TEST_F(ConvolutionFactoryTest, ConvFactory_Create_ValidParams_Success) {
	ConvFactory factory;
	auto ctx = createContext(3, "SiLU");
	auto argsNode = createArgsNode({64, 3, 2});  // out_channels=64, kernel=3, stride=2

	EXPECT_NO_THROW({
		auto result = factory.create(argsNode, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_EQ(result.outputChannels, 64);
	});
}

TEST_F(ConvolutionFactoryTest, ConvFactory_Create_OnlyOutChannels_UsesDefaults) {
	ConvFactory factory;
	auto ctx = createContext(3, "SiLU");
	auto argsNode = createArgsNode({64});  // Only out_channels

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

TEST_F(ConvolutionFactoryTest, ConvFactory_Create_DifferentKernels_Success) {
	ConvFactory factory;
	auto ctx = createContext(3, "SiLU");

	// Test kernel=1
	auto args1 = createArgsNode({64, 1, 1});
	auto result1 = factory.create(args1, ctx);
	EXPECT_FALSE(result1.module.is_empty());

	// Test kernel=3
	auto args3 = createArgsNode({64, 3, 1});
	auto result3 = factory.create(args3, ctx);
	EXPECT_FALSE(result3.module.is_empty());

	// Test kernel=5
	auto args5 = createArgsNode({64, 5, 1});
	auto result5 = factory.create(args5, ctx);
	EXPECT_FALSE(result5.module.is_empty());
}

TEST_F(ConvolutionFactoryTest, ConvFactory_Create_WithReLU_Success) {
	ConvFactory factory;
	auto ctx = createContext(3, "ReLU");
	auto argsNode = createArgsNode({64, 3, 1});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

// ========== DWConv Factory Tests ==========

TEST_F(ConvolutionFactoryTest, DWConvFactory_SupportedTypes_ReturnsDWConv) {
	DWConvFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "DWConv");
}

TEST_F(ConvolutionFactoryTest, DWConvFactory_Create_ValidParams_Success) {
	DWConvFactory factory;
	auto ctx = createContext(64, "SiLU");
	auto argsNode = createArgsNode({64, 3, 1});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

TEST_F(ConvolutionFactoryTest, DWConvFactory_Create_IdenticalChannels_Success) {
	DWConvFactory factory;
	auto ctx = createContext(64, "SiLU");
	auto argsNode = createArgsNode({64, 3, 1});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

TEST_F(ConvolutionFactoryTest, DWConvFactory_Create_OnlyOutChannels_UsesDefaults) {
	DWConvFactory factory;
	auto ctx = createContext(64, "SiLU");
	auto argsNode = createArgsNode({64});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

// ========== RepConv Factory Tests ==========

TEST_F(ConvolutionFactoryTest, RepConvFactory_SupportedTypes_ReturnsRepConv) {
	RepConvFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "RepConv");
}

TEST_F(ConvolutionFactoryTest, RepConvFactory_Create_ValidParams_Success) {
	RepConvFactory factory;
	auto ctx = createContext(3, "SiLU");
	auto argsNode = createArgsNode({64, 3, 1});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

TEST_F(ConvolutionFactoryTest, RepConvFactory_Create_DefaultKernel3_Success) {
	RepConvFactory factory;
	auto ctx = createContext(3, "SiLU");
	auto argsNode = createArgsNode({64});  // Defaults to kernel=3

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

// ========== GhostConv Factory Tests ==========

TEST_F(ConvolutionFactoryTest, GhostConvFactory_SupportedTypes_ReturnsGhostConv) {
	GhostConvFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "GhostConv");
}

TEST_F(ConvolutionFactoryTest, GhostConvFactory_Create_ValidParams_Success) {
	GhostConvFactory factory;
	auto ctx = createContext(3, "SiLU");
	auto argsNode = createArgsNode({64, 3, 1});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

// ========== Focus Factory Tests ==========

TEST_F(ConvolutionFactoryTest, FocusFactory_SupportedTypes_ReturnsFocus) {
	FocusFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "Focus");
}

TEST_F(ConvolutionFactoryTest, FocusFactory_Create_ValidParams_Success) {
	FocusFactory factory;
	auto ctx = createContext(12, "SiLU");  // Focus expands channels by 4x
	auto argsNode = createArgsNode({64, 3, 1});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

// ========== ConvTranspose Factory Tests ==========

TEST_F(ConvolutionFactoryTest, ConvTransposeFactory_SupportedTypes_ReturnsMultiple) {
	ConvTransposeFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 2);
	EXPECT_EQ(types[0], "ConvTranspose");
	EXPECT_EQ(types[1], "nn.ConvTranspose2d");
}

TEST_F(ConvolutionFactoryTest, ConvTransposeFactory_Create_ValidParams_Success) {
	ConvTransposeFactory factory;
	auto ctx = createContext(3, "SiLU");
	auto argsNode = createArgsNode({64, 2, 2});  // Upsampling by 2x

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

TEST_F(ConvolutionFactoryTest, ConvTransposeFactory_Create_DefaultParams_Success) {
	ConvTransposeFactory factory;
	auto ctx = createContext(3, "SiLU");
	auto argsNode = createArgsNode({64});  // Defaults to kernel=2, stride=2

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

// ========== NaiveConv Factory Tests ==========

TEST_F(ConvolutionFactoryTest, NaiveConvFactory_SupportedTypes_ReturnsNaiveConv) {
	NaiveConvFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "NaiveConv");
}

TEST_F(ConvolutionFactoryTest, NaiveConvFactory_Create_ValidParams_Success) {
	NaiveConvFactory factory;
	auto ctx = createContext(3, "SiLU");
	auto argsNode = createArgsNode({64, 3, 1});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

TEST_F(ConvolutionFactoryTest, NaiveConvFactory_Create_WithAllParams_Success) {
	NaiveConvFactory factory;
	auto ctx = createContext(3, "SiLU");
	auto argsNode = createArgsNode({64, 3, 1, 1, 1, 1});  // out, k, s, p, g, d

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

// ========== Scale Application Tests ==========

TEST_F(ConvolutionFactoryTest, ConvFactory_Create_WithScaling_AppliesCorrectly) {
	ConvFactory factory;
	auto ctx = createContext(3, "SiLU");

	// Apply 0.5x width scaling
	ctx.applyScale = [](int64_t channels, int64_t repeats) {
		return std::make_pair(channels / 2, repeats);
	};

	auto argsNode = createArgsNode({64, 3, 1});
	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 32);  // 64 * 0.5 = 32
}

TEST_F(ConvolutionFactoryTest, DWConvFactory_Create_WithScaling_AppliesCorrectly) {
	DWConvFactory factory;
	auto ctx = createContext(64, "SiLU");

	// Apply 2.0x width scaling
	ctx.applyScale = [](int64_t channels, int64_t repeats) {
		return std::make_pair(channels * 2, repeats);
	};

	auto argsNode = createArgsNode({64, 3, 1});
	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);  // 64 * 2.0 = 128
}

// ========== Edge Cases Tests ==========

TEST_F(ConvolutionFactoryTest, ConvFactory_Create_EmptyArgs_UsesDefaults) {
	ConvFactory factory;
	auto ctx = createContext(3, "SiLU");
	YAML::Node emptyArgs;

	auto result = factory.create(emptyArgs, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_GT(result.outputChannels, 0);  // Should have some default value
}

TEST_F(ConvolutionFactoryTest, ConvFactory_Create_LargeChannels_Success) {
	ConvFactory factory;
	auto ctx = createContext(3, "SiLU");
	auto argsNode = createArgsNode({1024, 3, 1});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 1024);
}

TEST_F(ConvolutionFactoryTest, ConvFactory_Create_LargeKernel_Success) {
	ConvFactory factory;
	auto ctx = createContext(3, "SiLU");
	auto argsNode = createArgsNode({64, 7, 1});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}
