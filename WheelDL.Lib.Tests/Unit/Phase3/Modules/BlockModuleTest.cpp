#include "pch.h"
#include <gtest/gtest.h>
#include "WheelDL.Lib/Model/Modules/Block.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include <torch/torch.h>

using namespace WheelDL;
using namespace WheelDL::Model::Modules;
using namespace WheelDL::Utils;

// ========== Bottleneck Module Tests ==========

class BottleneckModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<Bottleneck> bottleneckModule;

    static void SetUpTestSuite() {
        try {
            bottleneckModule = std::make_unique<Bottleneck>(64, 64, true, 1);
        } catch (const std::exception& e) {
            std::cerr << "Bottleneck setup failed: " << e.what() << std::endl;
            bottleneckModule.reset();
        }
    }

    static void TearDownTestSuite() {
        bottleneckModule.reset();
    }
};

std::unique_ptr<Bottleneck> BottleneckModuleTest::bottleneckModule;

TEST_F(BottleneckModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        Bottleneck bottleneck(64, 64, true, 1);
    });
}

TEST_F(BottleneckModuleTest, Forward_WithShortcut_Success) {
    if (!bottleneckModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});
    
    EXPECT_NO_THROW({
        auto output = bottleneckModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(BottleneckModuleTest, Forward_WithoutShortcut_Success) {
    Bottleneck bottleneck(64, 64, false, 1);
    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = bottleneck->forward(input);
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(BottleneckModuleTest, Forward_DifferentExpansionRatios_CorrectOutputShape) {
    torch::Tensor input = torch::randn({2, 64, 32, 32});

    // Expansion ratio 0.5
    Bottleneck bottleneck1(64, 64, true, 1);
    auto out1 = bottleneck1->forward(input);
    EXPECT_EQ(out1.size(1), 64);

    // Expansion ratio 1.0
    Bottleneck bottleneck2(64, 64, true, 1, std::vector<int64_t>{3, 3}, 1.0f);
    auto out2 = bottleneck2->forward(input);
    EXPECT_EQ(out2.size(1), 64);
}

// ========== C2f Module Tests ==========

class C2fModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<C2f> c2fModule;

    static void SetUpTestSuite() {
        try {
            c2fModule = std::make_unique<C2f>(64, 128, 2, false);
        } catch (const std::exception& e) {
            std::cerr << "C2f setup failed: " << e.what() << std::endl;
            c2fModule.reset();
        }
    }

    static void TearDownTestSuite() {
        c2fModule.reset();
    }
};

std::unique_ptr<C2f> C2fModuleTest::c2fModule;

TEST_F(C2fModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        C2f c2f(64, 128, 2, false);
    });
}

TEST_F(C2fModuleTest, Forward_ValidInput_Success) {
    if (!c2fModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = c2fModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 128);
    });
}

TEST_F(C2fModuleTest, Forward_VaryingRepeats_CorrectOutputShape) {
    torch::Tensor input = torch::randn({2, 64, 32, 32});

    // 1 repeat
    C2f c2f1(64, 128, 1, false);
    auto out1 = c2f1->forward(input);
    EXPECT_EQ(out1.size(1), 128);

    // 3 repeats
    C2f c2f3(64, 128, 3, false);
    auto out3 = c2f3->forward(input);
    EXPECT_EQ(out3.size(1), 128);
}

TEST_F(C2fModuleTest, Forward_ConcatenationPath_CorrectBehavior) {
    C2f c2f(64, 128, 2, false);
    torch::Tensor input = torch::randn({1, 64, 32, 32});

    auto output = c2f->forward(input);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

// ========== C3 Module Tests ==========

class C3ModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<C3> c3Module;

    static void SetUpTestSuite() {
        try {
            c3Module = std::make_unique<C3>(64, 128, 2, true);
        } catch (const std::exception& e) {
            std::cerr << "C3 setup failed: " << e.what() << std::endl;
            c3Module.reset();
        }
    }

    static void TearDownTestSuite() {
        c3Module.reset();
    }
};

std::unique_ptr<C3> C3ModuleTest::c3Module;

TEST_F(C3ModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        C3 c3(64, 128, 2, true);
    });
}

TEST_F(C3ModuleTest, Forward_ValidInput_Success) {
    if (!c3Module) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = c3Module->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 128);
    });
}

TEST_F(C3ModuleTest, Forward_CSPVariant_CorrectOutputShape) {
    C3 c3(64, 128, 2, true);
    torch::Tensor input = torch::randn({2, 64, 32, 32});

    auto output = c3->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

// ========== C3x Module Tests ==========

class C3xModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<C3x> c3xModule;

    static void SetUpTestSuite() {
        try {
            c3xModule = std::make_unique<C3x>(64, 128, 2, true);
        } catch (const std::exception& e) {
            std::cerr << "C3x setup failed: " << e.what() << std::endl;
            c3xModule.reset();
        }
    }

    static void TearDownTestSuite() {
        c3xModule.reset();
    }
};

std::unique_ptr<C3x> C3xModuleTest::c3xModule;

TEST_F(C3xModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        C3x c3x(64, 128, 2, true);
    });
}

TEST_F(C3xModuleTest, Forward_ValidInput_Success) {
    if (!c3xModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = c3xModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 128);
    });
}

// ========== C3k Module Tests ==========

class C3kModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<C3k> c3kModule;

    static void SetUpTestSuite() {
        try {
            c3kModule = std::make_unique<C3k>(64, 128, 2, true, 1, 0.5, 3);
        } catch (const std::exception& e) {
            std::cerr << "C3k setup failed: " << e.what() << std::endl;
            c3kModule.reset();
        }
    }

    static void TearDownTestSuite() {
        c3kModule.reset();
    }
};

std::unique_ptr<C3k> C3kModuleTest::c3kModule;

TEST_F(C3kModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        C3k c3k(64, 128, 2, true, 1, 0.5, 3);
    });
}

TEST_F(C3kModuleTest, Forward_ValidInput_Success) {
    if (!c3kModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = c3kModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 128);
    });
}

// ========== C3k2 Module Tests ==========

class C3k2ModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<C3k2> c3k2Module;

    static void SetUpTestSuite() {
        try {
            c3k2Module = std::make_unique<C3k2>(64, 128, 2, false);
        } catch (const std::exception& e) {
            std::cerr << "C3k2 setup failed: " << e.what() << std::endl;
            c3k2Module.reset();
        }
    }

    static void TearDownTestSuite() {
        c3k2Module.reset();
    }
};

std::unique_ptr<C3k2> C3k2ModuleTest::c3k2Module;

TEST_F(C3k2ModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        C3k2 c3k2(64, 128, 2, false);
    });
}

TEST_F(C3k2ModuleTest, Forward_ValidInput_Success) {
    if (!c3k2Module) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = c3k2Module->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 128);
    });
}

// ========== SPP Module Tests ==========

class SPPModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<SPP> sppModule;

    static void SetUpTestSuite() {
        try {
            sppModule = std::make_unique<SPP>(256, 256, std::vector<int64_t>{5, 9, 13});
        } catch (const std::exception& e) {
            std::cerr << "SPP setup failed: " << e.what() << std::endl;
            sppModule.reset();
        }
    }

    static void TearDownTestSuite() {
        sppModule.reset();
    }
};

std::unique_ptr<SPP> SPPModuleTest::sppModule;

TEST_F(SPPModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        SPP spp(256, 256, std::vector<int64_t>{5, 9, 13});
    });
}

TEST_F(SPPModuleTest, Forward_ValidInput_Success) {
    if (!sppModule) return;

    torch::Tensor input = torch::randn({1, 256, 32, 32});

    EXPECT_NO_THROW({
        auto output = sppModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 256);
    });
}

TEST_F(SPPModuleTest, Forward_PoolingBlocks_CorrectOutputShape) {
    SPP spp(256, 256, std::vector<int64_t>{5, 9, 13});
    torch::Tensor input = torch::randn({2, 256, 32, 32});

    auto output = spp->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 256);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(SPPModuleTest, Forward_DifferentKernelSizes_Success) {
    SPP spp(128, 128, std::vector<int64_t>{3, 5, 7});
    torch::Tensor input = torch::randn({1, 128, 32, 32});

    EXPECT_NO_THROW({
        auto output = spp->forward(input);
        EXPECT_EQ(output.size(1), 128);
    });
}

// ========== SPPF Module Tests ==========

class SPPFModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<SPPF> sppfModule;

    static void SetUpTestSuite() {
        try {
            sppfModule = std::make_unique<SPPF>(256, 256, 5);
        } catch (const std::exception& e) {
            std::cerr << "SPPF setup failed: " << e.what() << std::endl;
            sppfModule.reset();
        }
    }

    static void TearDownTestSuite() {
        sppfModule.reset();
    }
};

std::unique_ptr<SPPF> SPPFModuleTest::sppfModule;

TEST_F(SPPFModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        SPPF sppf(256, 256, 5);
    });
}

TEST_F(SPPFModuleTest, Forward_ValidInput_Success) {
    if (!sppfModule) return;

    torch::Tensor input = torch::randn({1, 256, 32, 32});

    EXPECT_NO_THROW({
        auto output = sppfModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 256);
    });
}

TEST_F(SPPFModuleTest, Forward_FasterThanSPP_CorrectOutputShape) {
    SPPF sppf(256, 256, 5);
    torch::Tensor input = torch::randn({2, 256, 32, 32});

    auto output = sppf->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 256);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

// ========== PSABlock Module Tests ==========

class PSABlockModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<PSABlock> psablockModule;

    static void SetUpTestSuite() {
        try {
            psablockModule = std::make_unique<PSABlock>(128, 0.5, 4, true);
        } catch (const std::exception& e) {
            std::cerr << "PSABlock setup failed: " << e.what() << std::endl;
            psablockModule.reset();
        }
    }

    static void TearDownTestSuite() {
        psablockModule.reset();
    }
};

std::unique_ptr<PSABlock> PSABlockModuleTest::psablockModule;

TEST_F(PSABlockModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        PSABlock psablock(128, 0.5, 4, true);
    });
}

TEST_F(PSABlockModuleTest, Forward_ValidInput_Success) {
    if (!psablockModule) return;

    torch::Tensor input = torch::randn({1, 128, 32, 32});

    EXPECT_NO_THROW({
        auto output = psablockModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 128);
    });
}

TEST_F(PSABlockModuleTest, Forward_AttentionBlock_CorrectOutputShape) {
    PSABlock psablock(128, 0.5, 4, true);
    torch::Tensor input = torch::randn({2, 128, 32, 32});

    auto output = psablock->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(PSABlockModuleTest, Forward_DifferentNumHeads_Success) {
    PSABlock psablock8(128, 0.5, 8, true);
    torch::Tensor input = torch::randn({1, 128, 32, 32});

    EXPECT_NO_THROW({
        auto output = psablock8->forward(input);
        EXPECT_EQ(output.size(1), 128);
    });
}

// ========== C2PSA Module Tests ==========

class C2PSAModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<C2PSA> c2psaModule;

    static void SetUpTestSuite() {
        try {
            c2psaModule = std::make_unique<C2PSA>(64, 128, 2, 0.5);
        } catch (const std::exception& e) {
            std::cerr << "C2PSA setup failed: " << e.what() << std::endl;
            c2psaModule.reset();
        }
    }

    static void TearDownTestSuite() {
        c2psaModule.reset();
    }
};

std::unique_ptr<C2PSA> C2PSAModuleTest::c2psaModule;

TEST_F(C2PSAModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        C2PSA c2psa(64, 128, 2, 0.5);
    });
}

TEST_F(C2PSAModuleTest, Forward_ValidInput_Success) {
    if (!c2psaModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = c2psaModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 128);
    });
}

// ========== HGBlock Module Tests ==========

class HGBlockModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<HGBlock> hgblockModule;

    static void SetUpTestSuite() {
        try {
            hgblockModule = std::make_unique<HGBlock>(64, 128, 128, 3, 4, false, false);
        } catch (const std::exception& e) {
            std::cerr << "HGBlock setup failed: " << e.what() << std::endl;
            hgblockModule.reset();
        }
    }

    static void TearDownTestSuite() {
        hgblockModule.reset();
    }
};

std::unique_ptr<HGBlock> HGBlockModuleTest::hgblockModule;

TEST_F(HGBlockModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        HGBlock hgblock(64, 128, 128, 3, 4, false, false);
    });
}

TEST_F(HGBlockModuleTest, Forward_ValidInput_Success) {
    if (!hgblockModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = hgblockModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 128);
    });
}

TEST_F(HGBlockModuleTest, Forward_HigherHRNetBlocks_CorrectOutputShape) {
    HGBlock hgblock(64, 128, 128, 3, 6, false, false);
    torch::Tensor input = torch::randn({2, 64, 32, 32});

    auto output = hgblock->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

// ========== RepBottleneck Module Tests ==========

class RepBottleneckModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<RepBottleneck> repbottleneckModule;

    static void SetUpTestSuite() {
        try {
            repbottleneckModule = std::make_unique<RepBottleneck>(64, 64, true, 1);
        } catch (const std::exception& e) {
            std::cerr << "RepBottleneck setup failed: " << e.what() << std::endl;
            repbottleneckModule.reset();
        }
    }

    static void TearDownTestSuite() {
        repbottleneckModule.reset();
    }
};

std::unique_ptr<RepBottleneck> RepBottleneckModuleTest::repbottleneckModule;

TEST_F(RepBottleneckModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        RepBottleneck repbottleneck(64, 64, true, 1);
    });
}

TEST_F(RepBottleneckModuleTest, Forward_ValidInput_Success) {
    if (!repbottleneckModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = repbottleneckModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 64);
    });
}

TEST_F(RepBottleneckModuleTest, Forward_RepVGGVariant_CorrectOutputShape) {
    RepBottleneck repbottleneck(64, 64, true, 1);
    torch::Tensor input = torch::randn({2, 64, 32, 32});

    auto output = repbottleneck->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 64);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

// ========== RepC3 Module Tests ==========

class RepC3ModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<RepC3> repc3Module;

    static void SetUpTestSuite() {
        try {
            repc3Module = std::make_unique<RepC3>(64, 128, 3, 1.0);
        } catch (const std::exception& e) {
            std::cerr << "RepC3 setup failed: " << e.what() << std::endl;
            repc3Module.reset();
        }
    }

    static void TearDownTestSuite() {
        repc3Module.reset();
    }
};

std::unique_ptr<RepC3> RepC3ModuleTest::repc3Module;

TEST_F(RepC3ModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        RepC3 repc3(64, 128, 3, 1.0);
    });
}

TEST_F(RepC3ModuleTest, Forward_ValidInput_Success) {
    if (!repc3Module) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = repc3Module->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 128);
    });
}

// ========== DBlock Module Tests ==========

class DBlockModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<DBlock> dblockModule;

    static void SetUpTestSuite() {
        try {
            dblockModule = std::make_unique<DBlock>(64, 128, 16, 4, "ReLU");
        } catch (const std::exception& e) {
            std::cerr << "DBlock setup failed: " << e.what() << std::endl;
            dblockModule.reset();
        }
    }

    static void TearDownTestSuite() {
        dblockModule.reset();
    }
};

std::unique_ptr<DBlock> DBlockModuleTest::dblockModule;

TEST_F(DBlockModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        DBlock dblock(64, 128, 16, 4, "ReLU");
    });
}

TEST_F(DBlockModuleTest, Forward_ValidInput_Success) {
    if (!dblockModule) return;

    torch::Tensor input = torch::randn({1, 64, 32, 32});

    EXPECT_NO_THROW({
        auto output = dblockModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 128);
    });
}

TEST_F(DBlockModuleTest, Forward_DenseNetBlocks_CorrectOutputShape) {
    DBlock dblock(64, 128, 32, 4, "ReLU");
    torch::Tensor input = torch::randn({2, 64, 32, 32});

    auto output = dblock->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

// ========== CNXBlock Module Tests ==========

class CNXBlockModuleTest : public ::testing::Test {
protected:
    static std::unique_ptr<CNXBlock> cnxblockModule;

    static void SetUpTestSuite() {
        try {
            cnxblockModule = std::make_unique<CNXBlock>(128, 1e-6);
        } catch (const std::exception& e) {
            std::cerr << "CNXBlock setup failed: " << e.what() << std::endl;
            cnxblockModule.reset();
        }
    }

    static void TearDownTestSuite() {
        cnxblockModule.reset();
    }
};

std::unique_ptr<CNXBlock> CNXBlockModuleTest::cnxblockModule;

TEST_F(CNXBlockModuleTest, Construction_ValidParams_Success) {
    EXPECT_NO_THROW({
        CNXBlock cnxblock(128, 1e-6);
    });
}

TEST_F(CNXBlockModuleTest, Forward_ValidInput_Success) {
    if (!cnxblockModule) return;

    torch::Tensor input = torch::randn({1, 128, 32, 32});

    EXPECT_NO_THROW({
        auto output = cnxblockModule->ptr()->forward(input);
        EXPECT_FALSE(output.sizes().empty());
        EXPECT_EQ(output.size(0), 1);
        EXPECT_EQ(output.size(1), 128);
    });
}

TEST_F(CNXBlockModuleTest, Forward_ConvNeXtBlocks_CorrectOutputShape) {
    CNXBlock cnxblock(128, 1e-6);
    torch::Tensor input = torch::randn({2, 128, 32, 32});

    auto output = cnxblock->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST_F(CNXBlockModuleTest, Forward_ResidualConnection_OutputShapeMatchesInput) {
    CNXBlock cnxblock(256, 1e-6);
    torch::Tensor input = torch::randn({1, 256, 16, 16});

    auto output = cnxblock->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}
