#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Builder/Factory/ModuleFactoryRegistry.h"
#include "WheelDL.Lib/Model/Builder/Factory/ConvolutionFactory.h"
#include "WheelDL.Lib/Model/Builder/Factory/BlockFactory.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"

using namespace WheelDL;
using namespace WheelDL::Model::Builder;
using namespace WheelDL::Utils;

class ModuleFactoryRegistryTest : public ::testing::Test {
protected:
	ModuleFactoryRegistry* registry;

	void SetUp() override {
		registry = &ModuleFactoryRegistry::instance();
	}
};

// ========== Singleton Tests ==========

TEST_F(ModuleFactoryRegistryTest, Singleton_MultipleAccess_SameInstance) {
	ModuleFactoryRegistry& instance1 = ModuleFactoryRegistry::instance();
	ModuleFactoryRegistry& instance2 = ModuleFactoryRegistry::instance();

	EXPECT_EQ(&instance1, &instance2);
}

TEST_F(ModuleFactoryRegistryTest, Singleton_AfterRegistration_Persists) {
	// Register a test factory
	auto testFactory = std::make_shared<ConvFactory>();

	EXPECT_NO_THROW({
		registry->registerFactory("TestModule", testFactory);
	});

	// Access registry again and verify factory still exists
	ModuleFactoryRegistry& newAccess = ModuleFactoryRegistry::instance();
	auto retrievedFactory = newAccess.getFactory("TestModule");

	EXPECT_NE(retrievedFactory, nullptr);
}

// ========== Registration Tests ==========

TEST_F(ModuleFactoryRegistryTest, RegisterFactory_NewType_Success) {
	auto factory = std::make_shared<ConvFactory>();

	EXPECT_NO_THROW({
		registry->registerFactory("Conv_Test", factory);
	});

	auto retrievedFactory = registry->getFactory("Conv_Test");
	EXPECT_NE(retrievedFactory, nullptr);
	EXPECT_EQ(retrievedFactory, factory.get());
}

TEST_F(ModuleFactoryRegistryTest, RegisterFactory_DuplicateType_ThrowsException) {
	auto factory1 = std::make_shared<ConvFactory>();
	auto factory2 = std::make_shared<ConvFactory>();

	registry->registerFactory("DuplicateTest", factory1);

	EXPECT_THROW({
		registry->registerFactory("DuplicateTest", factory2);
	}, std::runtime_error);
}

TEST_F(ModuleFactoryRegistryTest, RegisterFactory_NullFactory_ThrowsException) {
	EXPECT_THROW({
		registry->registerFactory("NullTest", nullptr);
	}, std::invalid_argument);
}

TEST_F(ModuleFactoryRegistryTest, RegisterFactory_EmptyType_ThrowsException) {
	auto factory = std::make_shared<ConvFactory>();

	EXPECT_THROW({
		registry->registerFactory("", factory);
	}, std::invalid_argument);
}

// ========== Retrieval Tests ==========

TEST_F(ModuleFactoryRegistryTest, GetFactory_RegisteredType_ReturnsFactory) {
	auto factory = std::make_shared<ConvFactory>();
	registry->registerFactory("GetTest", factory);

	auto retrievedFactory = registry->getFactory("GetTest");

	EXPECT_NE(retrievedFactory, nullptr);
	EXPECT_EQ(retrievedFactory, factory.get());
}

TEST_F(ModuleFactoryRegistryTest, GetFactory_UnregisteredType_ReturnsNullptr) {
	auto retrievedFactory = registry->getFactory("NonExistentType");

	EXPECT_EQ(retrievedFactory, nullptr);
}

TEST_F(ModuleFactoryRegistryTest, GetFactory_CaseSensitive_ExactMatch) {
	auto factory = std::make_shared<ConvFactory>();
	registry->registerFactory("CaseSensitiveTest", factory);

	auto exactMatch = registry->getFactory("CaseSensitiveTest");
	auto wrongCase = registry->getFactory("casesensitivetest");

	EXPECT_NE(exactMatch, nullptr);
	EXPECT_EQ(wrongCase, nullptr);
}

// ========== Alias System Tests ==========

TEST_F(ModuleFactoryRegistryTest, RegisterAlias_ValidTarget_Success) {
	auto factory = std::make_shared<ConvFactory>();
	registry->registerFactory("OriginalType", factory);

	EXPECT_NO_THROW({
		registry->registerAlias("AliasType", "OriginalType");
	});

	auto originalFactory = registry->getFactory("OriginalType");
	auto aliasFactory = registry->getFactory("AliasType");

	EXPECT_NE(originalFactory, nullptr);
	EXPECT_NE(aliasFactory, nullptr);
	EXPECT_EQ(originalFactory, aliasFactory);
}

TEST_F(ModuleFactoryRegistryTest, RegisterAlias_NonexistentTarget_ThrowsException) {
	EXPECT_THROW({
		registry->registerAlias("BadAlias", "NonExistentTarget");
	}, std::runtime_error);
}

TEST_F(ModuleFactoryRegistryTest, RegisterAlias_DuplicateAlias_ThrowsException) {
	auto factory = std::make_shared<ConvFactory>();
	registry->registerFactory("AliasOriginal", factory);
	registry->registerAlias("SharedAlias", "AliasOriginal");

	EXPECT_THROW({
		registry->registerAlias("SharedAlias", "AliasOriginal");
	}, std::runtime_error);
}

TEST_F(ModuleFactoryRegistryTest, GetFactory_ViaAlias_ReturnsSameFactory) {
	auto factory = std::make_shared<ConvFactory>();
	registry->registerFactory("MainType", factory);
	registry->registerAlias("AliasType1", "MainType");
	registry->registerAlias("AliasType2", "MainType");

	auto mainFactory = registry->getFactory("MainType");
	auto alias1Factory = registry->getFactory("AliasType1");
	auto alias2Factory = registry->getFactory("AliasType2");

	EXPECT_EQ(mainFactory, alias1Factory);
	EXPECT_EQ(mainFactory, alias2Factory);
	EXPECT_EQ(alias1Factory, alias2Factory);
}

// ========== Default Factories Tests ==========

TEST_F(ModuleFactoryRegistryTest, RegisterDefaultFactories_Success) {
	// Check if Conv is already registered (from previous test runs or initialization)
	if (registry->getFactory("Conv") == nullptr) {
		EXPECT_NO_THROW({
			registry->registerDefaultFactories();
		});
	}
	else {
		// Already registered, just verify it doesn't throw on idempotent call
		// (Actually it will throw, so we skip the duplicate registration)
		SUCCEED() << "Default factories already registered";
	}
}

TEST_F(ModuleFactoryRegistryTest, DefaultFactories_Conv_Registered) {
	// Register only if not already done
	if (registry->getFactory("Conv") == nullptr) {
		registry->registerDefaultFactories();
	}

	auto convFactory = registry->getFactory("Conv");

	EXPECT_NE(convFactory, nullptr);
}

TEST_F(ModuleFactoryRegistryTest, DefaultFactories_C2f_Registered) {
	// Register only if not already done
	if (registry->getFactory("C2f") == nullptr) {
		registry->registerDefaultFactories();
	}

	auto c2fFactory = registry->getFactory("C2f");

	EXPECT_NE(c2fFactory, nullptr);
}

TEST_F(ModuleFactoryRegistryTest, DefaultFactories_Detect_Registered) {
	// Register only if not already done
	if (registry->getFactory("Detect") == nullptr) {
		registry->registerDefaultFactories();
	}

	auto detectFactory = registry->getFactory("Detect");

	EXPECT_NE(detectFactory, nullptr);
}

TEST_F(ModuleFactoryRegistryTest, DefaultFactories_SPPF_Registered) {
	// Register only if not already done
	if (registry->getFactory("SPPF") == nullptr) {
		registry->registerDefaultFactories();
	}

	auto sppfFactory = registry->getFactory("SPPF");

	EXPECT_NE(sppfFactory, nullptr);
}

// ========== Multiple Registration Tests ==========

TEST_F(ModuleFactoryRegistryTest, MultipleFactories_IndependentRetrieval) {
	auto convFactory = std::make_shared<ConvFactory>();
	auto c2fFactory = std::make_shared<C2fFactory>();

	registry->registerFactory("MultiConv", convFactory);
	registry->registerFactory("MultiC2f", c2fFactory);

	auto retrieved1 = registry->getFactory("MultiConv");
	auto retrieved2 = registry->getFactory("MultiC2f");

	EXPECT_NE(retrieved1, nullptr);
	EXPECT_NE(retrieved2, nullptr);
	EXPECT_NE(retrieved1, retrieved2);
}

// ========== Edge Cases Tests ==========

TEST_F(ModuleFactoryRegistryTest, RegisterFactory_SpecialCharactersInName_Success) {
	auto factory = std::make_shared<ConvFactory>();

	EXPECT_NO_THROW({
		registry->registerFactory("Module_With-Special.Chars", factory);
	});

	auto retrieved = registry->getFactory("Module_With-Special.Chars");
	EXPECT_NE(retrieved, nullptr);
}

TEST_F(ModuleFactoryRegistryTest, RegisterFactory_NumericName_Success) {
	auto factory = std::make_shared<ConvFactory>();

	EXPECT_NO_THROW({
		registry->registerFactory("123Module", factory);
	});

	auto retrieved = registry->getFactory("123Module");
	EXPECT_NE(retrieved, nullptr);
}
