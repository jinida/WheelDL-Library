#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Builder/Factory/BlockFactory.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <yaml-cpp/yaml.h>

using namespace WheelDL;
using namespace WheelDL::Model::Builder;
using namespace WheelDL::Utils;

class BlockFactoryTest : public ::testing::Test {
protected:
	ModuleBuildContext createContext(
		int64_t inputChannels = 3,
		const std::string& defaultAct = "SiLU",
		int64_t repeats = 1)
	{
		ModuleBuildContext ctx;
		ctx.inputChannels = inputChannels;
		ctx.repeats = repeats;
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

// ========== Bottleneck Factory Tests ==========

TEST_F(BlockFactoryTest, BottleneckFactory_SupportedTypes_ReturnsBottleneck) {
	BottleneckFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "Bottleneck");
}

TEST_F(BlockFactoryTest, BottleneckFactory_Create_ValidParams_Success) {
	BottleneckFactory factory;
	auto ctx = createContext(64, "SiLU");
	auto argsNode = createArgsNode({128});  // out_channels

	EXPECT_NO_THROW({
		auto result = factory.create(argsNode, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_EQ(result.outputChannels, 128);
	});
}

TEST_F(BlockFactoryTest, BottleneckFactory_Create_WithAllParams_Success) {
	BottleneckFactory factory;
	auto ctx = createContext(64, "SiLU");

	YAML::Node argsNode;
	argsNode.push_back(128);   // out_channels
	argsNode.push_back(true);  // shortcut
	argsNode.push_back(1);     // g
	argsNode.push_back(0.5);   // e

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, BottleneckFactory_Create_WithScaling_AppliesCorrectly) {
	BottleneckFactory factory;
	auto ctx = createContext(64, "SiLU");

	// Apply 0.5x width scaling
	ctx.applyScale = [](int64_t channels, int64_t repeats) {
		return std::make_pair(channels / 2, repeats);
	};

	auto argsNode = createArgsNode({128});
	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);  // 128 * 0.5 = 64
}

// ========== C2f Factory Tests ==========

TEST_F(BlockFactoryTest, C2fFactory_SupportedTypes_ReturnsC2f) {
	C2fFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "C2f");
}

TEST_F(BlockFactoryTest, C2fFactory_Create_ValidParams_Success) {
	C2fFactory factory;
	auto ctx = createContext(64, "SiLU", 3);
	auto argsNode = createArgsNode({128});

	EXPECT_NO_THROW({
		auto result = factory.create(argsNode, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_EQ(result.outputChannels, 128);
	});
}

TEST_F(BlockFactoryTest, C2fFactory_Create_WithAllParams_Success) {
	C2fFactory factory;
	auto ctx = createContext(64, "SiLU", 3);

	YAML::Node argsNode;
	argsNode.push_back(128);    // out_channels
	argsNode.push_back(false);  // shortcut
	argsNode.push_back(1);      // g
	argsNode.push_back(0.5);    // e

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, C2fFactory_Create_WithRepeats_Success) {
	C2fFactory factory;
	auto ctx = createContext(64, "SiLU", 5);
	auto argsNode = createArgsNode({128});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, C2fFactory_Create_WithScaling_AppliesCorrectly) {
	C2fFactory factory;
	auto ctx = createContext(64, "SiLU", 3);

	// Apply 2.0x width scaling and 0.5x depth scaling
	ctx.applyScale = [](int64_t channels, int64_t repeats) {
		return std::make_pair(channels * 2, repeats / 2);
	};

	auto argsNode = createArgsNode({64});
	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);  // 64 * 2.0 = 128
}

// ========== C2 Factory Tests ==========

TEST_F(BlockFactoryTest, C2Factory_SupportedTypes_ReturnsC2) {
	C2Factory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "C2");
}

TEST_F(BlockFactoryTest, C2Factory_Create_ValidParams_Success) {
	C2Factory factory;
	auto ctx = createContext(64, "SiLU", 3);
	auto argsNode = createArgsNode({128});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, C2Factory_Create_WithShortcut_Success) {
	C2Factory factory;
	auto ctx = createContext(64, "SiLU", 3);

	YAML::Node argsNode;
	argsNode.push_back(128);   // out_channels
	argsNode.push_back(true);  // shortcut (default is true)

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

// ========== C3 Factory Tests ==========

TEST_F(BlockFactoryTest, C3Factory_SupportedTypes_ReturnsC3) {
	C3Factory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "C3");
}

TEST_F(BlockFactoryTest, C3Factory_Create_ValidParams_Success) {
	C3Factory factory;
	auto ctx = createContext(64, "SiLU", 3);
	auto argsNode = createArgsNode({128});

	EXPECT_NO_THROW({
		auto result = factory.create(argsNode, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_EQ(result.outputChannels, 128);
	});
}

TEST_F(BlockFactoryTest, C3Factory_Create_WithAllParams_Success) {
	C3Factory factory;
	auto ctx = createContext(64, "SiLU", 3);

	YAML::Node argsNode;
	argsNode.push_back(128);   // out_channels
	argsNode.push_back(true);  // shortcut
	argsNode.push_back(1);     // g
	argsNode.push_back(0.5);   // e

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, C3Factory_Create_WithScaling_AppliesCorrectly) {
	C3Factory factory;
	auto ctx = createContext(64, "SiLU", 3);

	// Apply 0.5x width scaling
	ctx.applyScale = [](int64_t channels, int64_t repeats) {
		return std::make_pair(channels / 2, repeats);
	};

	auto argsNode = createArgsNode({256});
	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);  // 256 * 0.5 = 128
}

// ========== C3x Factory Tests ==========

TEST_F(BlockFactoryTest, C3xFactory_SupportedTypes_ReturnsC3x) {
	C3xFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "C3x");
}

TEST_F(BlockFactoryTest, C3xFactory_Create_ValidParams_Success) {
	C3xFactory factory;
	auto ctx = createContext(64, "SiLU", 3);
	auto argsNode = createArgsNode({128});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

// ========== C3k2 Factory Tests ==========

TEST_F(BlockFactoryTest, C3k2Factory_SupportedTypes_ReturnsC3k2) {
	C3k2Factory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "C3k2");
}

TEST_F(BlockFactoryTest, C3k2Factory_Create_ValidParams_Success) {
	C3k2Factory factory;
	auto ctx = createContext(64, "SiLU", 3);
	auto argsNode = createArgsNode({128});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, C3k2Factory_Create_WithC3kFlag_Success) {
	C3k2Factory factory;
	auto ctx = createContext(64, "SiLU", 3);
	auto argsNode = createArgsNode({128, 1});  // c3k=true

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

// ========== C2fPSA Factory Tests ==========

TEST_F(BlockFactoryTest, C2fPSAFactory_SupportedTypes_ReturnsC2fPSA) {
	C2fPSAFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "C2fPSA");
}

TEST_F(BlockFactoryTest, C2fPSAFactory_Create_ValidParams_Success) {
	C2fPSAFactory factory;
	auto ctx = createContext(64, "SiLU", 3);
	auto argsNode = createArgsNode({128});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, C2fPSAFactory_Create_WithScaling_AppliesCorrectly) {
	C2fPSAFactory factory;
	auto ctx = createContext(64, "SiLU", 3);

	// Apply 2.0x width scaling
	ctx.applyScale = [](int64_t channels, int64_t repeats) {
		return std::make_pair(channels * 2, repeats);
	};

	auto argsNode = createArgsNode({64});
	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);  // 64 * 2.0 = 128
}

// ========== C2PSA Factory Tests ==========

TEST_F(BlockFactoryTest, C2PSAFactory_SupportedTypes_ReturnsC2PSA) {
	C2PSAFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "C2PSA");
}

TEST_F(BlockFactoryTest, C2PSAFactory_Create_ValidParams_Success) {
	C2PSAFactory factory;
	auto ctx = createContext(64, "SiLU", 3);
	auto argsNode = createArgsNode({128});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

// ========== RepC3 Factory Tests ==========

TEST_F(BlockFactoryTest, RepC3Factory_SupportedTypes_ReturnsRepC3) {
	RepC3Factory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "RepC3");
}

TEST_F(BlockFactoryTest, RepC3Factory_Create_ValidParams_Success) {
	RepC3Factory factory;
	auto ctx = createContext(64, "SiLU", 3);
	auto argsNode = createArgsNode({128});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

// ========== C3Ghost Factory Tests ==========

TEST_F(BlockFactoryTest, C3GhostFactory_SupportedTypes_ReturnsC3Ghost) {
	C3GhostFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "C3Ghost");
}

TEST_F(BlockFactoryTest, C3GhostFactory_Create_ValidParams_Success) {
	C3GhostFactory factory;
	auto ctx = createContext(64, "SiLU", 3);
	auto argsNode = createArgsNode({128});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, C3GhostFactory_Create_WithAllParams_Success) {
	C3GhostFactory factory;
	auto ctx = createContext(64, "SiLU", 3);

	YAML::Node argsNode;
	argsNode.push_back(128);   // out_channels
	argsNode.push_back(true);  // shortcut
	argsNode.push_back(1);     // g
	argsNode.push_back(0.5);   // e

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

// ========== BottleneckCSP Factory Tests ==========

TEST_F(BlockFactoryTest, BottleneckCSPFactory_SupportedTypes_ReturnsBottleneckCSP) {
	BottleneckCSPFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "BottleneckCSP");
}

TEST_F(BlockFactoryTest, BottleneckCSPFactory_Create_ValidParams_Success) {
	BottleneckCSPFactory factory;
	auto ctx = createContext(64, "SiLU", 3);
	auto argsNode = createArgsNode({128});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

// ========== SPP Factory Tests ==========

TEST_F(BlockFactoryTest, SPPFactory_SupportedTypes_ReturnsSPP) {
	SPPFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "SPP");
}

TEST_F(BlockFactoryTest, SPPFactory_Create_ValidParams_Success) {
	SPPFactory factory;
	auto ctx = createContext(512, "SiLU");
	auto argsNode = createArgsNode({512, 5});  // out_channels, kernel_size

	EXPECT_NO_THROW({
		auto result = factory.create(argsNode, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_EQ(result.outputChannels, 512);
	});
}

TEST_F(BlockFactoryTest, SPPFactory_Create_DifferentKernels_Success) {
	SPPFactory factory;
	auto ctx = createContext(512, "SiLU");

	// Test kernel=3
	auto args3 = createArgsNode({512, 3});
	auto result3 = factory.create(args3, ctx);
	EXPECT_FALSE(result3.module.is_empty());

	// Test kernel=5
	auto args5 = createArgsNode({512, 5});
	auto result5 = factory.create(args5, ctx);
	EXPECT_FALSE(result5.module.is_empty());

	// Test kernel=7
	auto args7 = createArgsNode({512, 7});
	auto result7 = factory.create(args7, ctx);
	EXPECT_FALSE(result7.module.is_empty());
}

TEST_F(BlockFactoryTest, SPPFactory_Create_WithScaling_AppliesCorrectly) {
	SPPFactory factory;
	auto ctx = createContext(512, "SiLU");

	// Apply 0.5x width scaling
	ctx.applyScale = [](int64_t channels, int64_t repeats) {
		return std::make_pair(channels / 2, repeats);
	};

	auto argsNode = createArgsNode({512, 5});
	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 256);  // 512 * 0.5 = 256
}

// ========== SPPF Factory Tests ==========

TEST_F(BlockFactoryTest, SPPFFactory_SupportedTypes_ReturnsSPPF) {
	SPPFFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "SPPF");
}

TEST_F(BlockFactoryTest, SPPFFactory_Create_ValidParams_Success) {
	SPPFFactory factory;
	auto ctx = createContext(512, "SiLU");
	auto argsNode = createArgsNode({512, 5});  // out_channels, kernel_size

	EXPECT_NO_THROW({
		auto result = factory.create(argsNode, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_EQ(result.outputChannels, 512);
	});
}

TEST_F(BlockFactoryTest, SPPFFactory_Create_DifferentKernels_Success) {
	SPPFFactory factory;
	auto ctx = createContext(512, "SiLU");

	// Test kernel=3
	auto args3 = createArgsNode({512, 3});
	auto result3 = factory.create(args3, ctx);
	EXPECT_FALSE(result3.module.is_empty());

	// Test kernel=5
	auto args5 = createArgsNode({512, 5});
	auto result5 = factory.create(args5, ctx);
	EXPECT_FALSE(result5.module.is_empty());
}

TEST_F(BlockFactoryTest, SPPFFactory_Create_WithScaling_AppliesCorrectly) {
	SPPFFactory factory;
	auto ctx = createContext(512, "SiLU");

	// Apply 2.0x width scaling
	ctx.applyScale = [](int64_t channels, int64_t repeats) {
		return std::make_pair(channels * 2, repeats);
	};

	auto argsNode = createArgsNode({256, 5});
	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 512);  // 256 * 2.0 = 512
}

// ========== ResNetBlock Factory Tests ==========

TEST_F(BlockFactoryTest, ResNetBlockFactory_SupportedTypes_ReturnsResNetBlock) {
	ResNetBlockFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "ResNetBlock");
}

TEST_F(BlockFactoryTest, ResNetBlockFactory_Create_ValidParams_Success) {
	ResNetBlockFactory factory;
	auto ctx = createContext(64, "SiLU");
	auto argsNode = createArgsNode({64});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

TEST_F(BlockFactoryTest, ResNetBlockFactory_Create_WithStride_Success) {
	ResNetBlockFactory factory;
	auto ctx = createContext(64, "SiLU");
	auto argsNode = createArgsNode({128, 2});  // out_channels, stride

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, ResNetBlockFactory_Create_WithExpansion_Success) {
	ResNetBlockFactory factory;
	auto ctx = createContext(64, "SiLU");

	YAML::Node argsNode;
	argsNode.push_back(64);   // out_channels
	argsNode.push_back(1);    // stride
	argsNode.push_back(4.0);  // expansion

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

// ========== ResNetLayer Factory Tests ==========

TEST_F(BlockFactoryTest, ResNetLayerFactory_SupportedTypes_ReturnsResNetLayer) {
	ResNetLayerFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "ResNetLayer");
}

TEST_F(BlockFactoryTest, ResNetLayerFactory_Create_ValidParams_Success) {
	ResNetLayerFactory factory;
	auto ctx = createContext(64, "SiLU", 3);
	auto argsNode = createArgsNode({64});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_GT(result.outputChannels, 0);
}

TEST_F(BlockFactoryTest, ResNetLayerFactory_Create_AsFirstLayer_Success) {
	ResNetLayerFactory factory;
	auto ctx = createContext(64, "SiLU", 3);

	YAML::Node argsNode;
	argsNode.push_back(64);     // out_channels
	argsNode.push_back(1);      // stride
	argsNode.push_back(true);   // is_first

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

// ========== DBlock Factory Tests ==========

TEST_F(BlockFactoryTest, DBlockFactory_SupportedTypes_ReturnsBoth) {
	DBlockFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 2);
	EXPECT_EQ(types[0], "DBlock");
	EXPECT_EQ(types[1], "DenseBlock");
}

TEST_F(BlockFactoryTest, DBlockFactory_Create_ValidParams_Success) {
	DBlockFactory factory;
	auto ctx = createContext(64, "SiLU");
	auto argsNode = createArgsNode({128, 32});  // out_channels, growth_rate

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, DBlockFactory_Create_WithBnSize_Success) {
	DBlockFactory factory;
	auto ctx = createContext(64, "SiLU");
	auto argsNode = createArgsNode({128, 32, 4});  // out_channels, growth_rate, bn_size

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, DBlockFactory_Create_WithScaling_AppliesCorrectly) {
	DBlockFactory factory;
	auto ctx = createContext(64, "SiLU");

	// Apply 0.5x width scaling
	ctx.applyScale = [](int64_t channels, int64_t repeats) {
		return std::make_pair(channels / 2, repeats);
	};

	auto argsNode = createArgsNode({256, 64});
	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);  // 256 * 0.5 = 128
}

// ========== CNXBlock Factory Tests ==========

TEST_F(BlockFactoryTest, CNXBlockFactory_SupportedTypes_ReturnsBoth) {
	CNXBlockFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 2);
	EXPECT_EQ(types[0], "CNXBlock");
	EXPECT_EQ(types[1], "ConvNeXtBlock");
}

TEST_F(BlockFactoryTest, CNXBlockFactory_Create_ValidParams_Success) {
	CNXBlockFactory factory;
	auto ctx = createContext(64, "SiLU");
	auto argsNode = createArgsNode({64});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);  // Output = input for residual block
}

TEST_F(BlockFactoryTest, CNXBlockFactory_Create_WithLayerScale_Success) {
	CNXBlockFactory factory;
	auto ctx = createContext(128, "SiLU");

	YAML::Node argsNode;
	argsNode.push_back(128);   // channels
	argsNode.push_back(1e-6);  // layer_scale_init

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

// ========== TransformerBlock Factory Tests ==========

TEST_F(BlockFactoryTest, TransformerBlockFactory_SupportedTypes_ReturnsTransformerBlock) {
	TransformerBlockFactory factory;
	auto types = factory.supportedTypes();

	ASSERT_EQ(types.size(), 1);
	EXPECT_EQ(types[0], "TransformerBlock");
}

TEST_F(BlockFactoryTest, TransformerBlockFactory_Create_ValidParams_Success) {
	TransformerBlockFactory factory;
	auto ctx = createContext(256, "SiLU");
	auto argsNode = createArgsNode({256, 8, 6});  // out_channels, num_heads, num_layers

	EXPECT_NO_THROW({
		auto result = factory.create(argsNode, ctx);

		EXPECT_FALSE(result.module.is_empty());
		EXPECT_EQ(result.outputChannels, 256);
	});
}

TEST_F(BlockFactoryTest, TransformerBlockFactory_Create_DifferentHeadCounts_Success) {
	TransformerBlockFactory factory;
	auto ctx = createContext(256, "SiLU");

	// Test 4 heads
	auto args4 = createArgsNode({256, 4, 6});
	auto result4 = factory.create(args4, ctx);
	EXPECT_FALSE(result4.module.is_empty());

	// Test 8 heads
	auto args8 = createArgsNode({256, 8, 6});
	auto result8 = factory.create(args8, ctx);
	EXPECT_FALSE(result8.module.is_empty());

	// Test 16 heads
	auto args16 = createArgsNode({256, 16, 6});
	auto result16 = factory.create(args16, ctx);
	EXPECT_FALSE(result16.module.is_empty());
}

TEST_F(BlockFactoryTest, TransformerBlockFactory_Create_WithScaling_AppliesCorrectly) {
	TransformerBlockFactory factory;
	auto ctx = createContext(256, "SiLU");

	// Apply 2.0x width scaling
	ctx.applyScale = [](int64_t channels, int64_t repeats) {
		return std::make_pair(channels * 2, repeats);
	};

	auto argsNode = createArgsNode({128, 8, 6});
	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 256);  // 128 * 2.0 = 256
}

// ========== Edge Cases Tests ==========

TEST_F(BlockFactoryTest, C2fFactory_Create_LargeChannels_Success) {
	C2fFactory factory;
	auto ctx = createContext(512, "SiLU", 3);
	auto argsNode = createArgsNode({1024});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 1024);
}

TEST_F(BlockFactoryTest, SPPFactory_Create_SmallChannels_Success) {
	SPPFactory factory;
	auto ctx = createContext(32, "SiLU");
	auto argsNode = createArgsNode({64, 5});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 64);
}

TEST_F(BlockFactoryTest, C3Factory_Create_WithReLU_Success) {
	C3Factory factory;
	auto ctx = createContext(64, "ReLU", 3);
	auto argsNode = createArgsNode({128});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 128);
}

TEST_F(BlockFactoryTest, BottleneckFactory_Create_SingleChannel_Success) {
	BottleneckFactory factory;
	auto ctx = createContext(1, "SiLU");
	auto argsNode = createArgsNode({16});

	auto result = factory.create(argsNode, ctx);

	EXPECT_FALSE(result.module.is_empty());
	EXPECT_EQ(result.outputChannels, 16);
}
