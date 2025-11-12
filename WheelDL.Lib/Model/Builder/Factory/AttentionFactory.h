#pragma once

#include "ModuleFactory.h"
#include "../../Modules/Attention.h"

namespace WheelDL {
namespace Model {
namespace Builder {

/**
 * @brief Factory for PSA module
 * YAML args: [out_channels]
 * Constructor: PSAImpl(c1, c2)
 */
class PSAFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"PSA"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [out_channels]
        int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 256);

        auto scaled = ctx.applyScale(outCh, 1);

        ModuleBuildResult result;
        result.module = torch::nn::AnyModule(Modules::PSA(ctx.inputChannels, scaled.first));
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for CBAM module
 */
class CBAMFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"CBAM"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        ModuleBuildResult result;
        result.module = torch::nn::AnyModule(Modules::CBAM(ctx.inputChannels));
        result.outputChannels = ctx.inputChannels;
        return result;
    }
};

/**
 * @brief Factory for Attention module
 * YAML args: [num_heads]
 * Constructor: AttentionImpl(channels, num_heads)
 */
class AttentionFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"Attention"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [num_heads]
        int64_t numHeads = YamlArgsParser::getInt(argsNode, 0, 8);

        ModuleBuildResult result;
        result.module = torch::nn::AnyModule(Modules::Attention(ctx.inputChannels, numHeads));
        result.outputChannels = ctx.inputChannels;
        return result;
    }
};

} // namespace Builder
} // namespace Model
} // namespace WheelDL
