#pragma once

#include "ModuleFactory.h"
#include "../../Modules/Conv.h"

namespace WheelDL {
namespace Model {
namespace Builder {

/**
 * @brief Factory for Conv module
 * YAML args: [out_channels, kernel?, stride?]
 * Constructor: ConvImpl(c1, c2, k, s, p, g, d, act)
 */
class ConvFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"Conv"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [out_channels, kernel?, stride?]
        int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 64);
        int64_t k = YamlArgsParser::getInt(argsNode, 1, 1);
        int64_t s = YamlArgsParser::getInt(argsNode, 2, 1);

        auto scaled = ctx.applyScale(outCh, 1);

        ModuleBuildResult result;
        // Constructor: ConvImpl(c1, c2, k, s, p, g, d, act)
        result.module = torch::nn::AnyModule(
            Modules::Conv(ctx.inputChannels, scaled.first, k, s, std::nullopt, 1, 1, ctx.defaultAct)
        );
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for DWConv module
 * YAML args: [out_channels, kernel?, stride?]
 * Constructor: DWConvImpl(c1, c2, k, s, d, act)
 */
class DWConvFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"DWConv"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [out_channels, kernel?, stride?]
        int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 64);
        int64_t k = YamlArgsParser::getInt(argsNode, 1, 1);
        int64_t s = YamlArgsParser::getInt(argsNode, 2, 1);

        auto scaled = ctx.applyScale(outCh, 1);

        ModuleBuildResult result;
        // Constructor: DWConvImpl(c1, c2, k, s, d, act)
        result.module = torch::nn::AnyModule(
            Modules::DWConv(ctx.inputChannels, scaled.first, k, s, 1, ctx.defaultAct)
        );
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for RepConv module
 * YAML args: [out_channels, kernel?, stride?]
 * Constructor: RepConvImpl(c1, c2, k, s, p, g, d, act)
 */
class RepConvFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"RepConv"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [out_channels, kernel?, stride?]
        int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 64);
        int64_t k = YamlArgsParser::getInt(argsNode, 1, 3);  // RepConv defaults to kernel=3
        int64_t s = YamlArgsParser::getInt(argsNode, 2, 1);

        auto scaled = ctx.applyScale(outCh, 1);

        ModuleBuildResult result;
        // Constructor: RepConvImpl(c1, c2, k, s, p, g, d, act)
        result.module = torch::nn::AnyModule(
            Modules::RepConv(ctx.inputChannels, scaled.first, k, s, 1, 1, 1, ctx.defaultAct)
        );
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for GhostConv module
 * YAML args: [out_channels, kernel?, stride?]
 * Constructor: GhostConvImpl(c1, c2, k, s, g, act)
 */
class GhostConvFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"GhostConv"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [out_channels, kernel?, stride?]
        int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 64);
        int64_t k = YamlArgsParser::getInt(argsNode, 1, 1);
        int64_t s = YamlArgsParser::getInt(argsNode, 2, 1);

        auto scaled = ctx.applyScale(outCh, 1);

        ModuleBuildResult result;
        // Constructor: GhostConvImpl(c1, c2, k, s, g, act)
        result.module = torch::nn::AnyModule(
            Modules::GhostConv(ctx.inputChannels, scaled.first, k, s, 1, ctx.defaultAct)
        );
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for Focus module
 * YAML args: [out_channels, kernel?, stride?]
 * Constructor: FocusImpl(c1, c2, k, s, p, g, act)
 */
class FocusFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"Focus"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [out_channels, kernel?, stride?]
        int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 64);
        int64_t k = YamlArgsParser::getInt(argsNode, 1, 1);
        int64_t s = YamlArgsParser::getInt(argsNode, 2, 1);

        auto scaled = ctx.applyScale(outCh, 1);

        ModuleBuildResult result;
        // Constructor: FocusImpl(c1, c2, k, s, p, g, act)
        result.module = torch::nn::AnyModule(
            Modules::Focus(ctx.inputChannels, scaled.first, k, s, std::nullopt, 1, ctx.defaultAct)
        );
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for ConvTranspose module
 * YAML args: [out_channels, kernel?, stride?]
 * Constructor: ConvTransposeImpl(c1, c2, k, s, p, op)
 */
class ConvTransposeFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"ConvTranspose", "nn.ConvTranspose2d"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [out_channels, kernel?, stride?]
        int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 64);
        int64_t k = YamlArgsParser::getInt(argsNode, 1, 2);
        int64_t s = YamlArgsParser::getInt(argsNode, 2, 2);

        auto scaled = ctx.applyScale(outCh, 1);

        ModuleBuildResult result;
        // Constructor: ConvTransposeImpl(c1, c2, k, s)
        result.module = torch::nn::AnyModule(
            Modules::ConvTranspose(ctx.inputChannels, scaled.first, k, s)
        );
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for NaiveConv module (Conv2d without BatchNorm/Activation)
 * YAML args: [out_channels, kernel?, stride?, padding?, groups?, dilation?]
 * Constructor: NaiveConvImpl(c1, c2, k, s, p, g, d)
 */
class NaiveConvFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"NaiveConv"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [out_channels, kernel?, stride?, padding?, groups?, dilation?]
        int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 64);
        int64_t k = YamlArgsParser::getInt(argsNode, 1, 1);
        int64_t s = YamlArgsParser::getInt(argsNode, 2, 1);
        std::optional<int64_t> p = std::nullopt;
        if (argsNode.IsSequence() && argsNode.size() > 3) {
            p = YamlArgsParser::getInt(argsNode, 3, 0);
        }
        
        int64_t g = YamlArgsParser::getInt(argsNode, 4, 1);
        int64_t d = YamlArgsParser::getInt(argsNode, 5, 1);

        auto scaled = ctx.applyScale(outCh, 1);

        ModuleBuildResult result;
        // Constructor: NaiveConvImpl(c1, c2, k, s, p, g, d)
        result.module = torch::nn::AnyModule(
            Modules::NaiveConv(ctx.inputChannels, scaled.first, k, s, p, g, d)
        );
        result.outputChannels = scaled.first;
        return result;
    }
};

} // namespace Builder
} // namespace Model
} // namespace WheelDL
