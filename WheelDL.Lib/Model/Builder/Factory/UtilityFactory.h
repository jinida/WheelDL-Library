#pragma once

#include "ModuleFactory.h"
#include "../../Modules/Block.h"
#include "../../Modules/Conv.h"

namespace WheelDL {
namespace Model {
namespace Builder {

/**
 * @brief Factory for Upsample module
 * YAML args: [scale_factor?]
 * Constructor: torch::nn::Upsample(options)
 */
class UpsampleFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"nn.Upsample", "Upsample"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override
    {
        double scaleFactor = YamlArgsParser::getDouble(argsNode, 0, 2.0);
        std::string mode = YamlArgsParser::getString(argsNode, 1, "nearest");
        ModuleBuildResult result;
        result.module = torch::nn::AnyModule(Modules::Upsample(scaleFactor, mode));
        result.outputChannels = ctx.inputChannels;
        return result;
    }
};

/**
 * @brief Factory for Concat module
 * YAML args: [dimension]
 * Constructor: ConcatImpl(dim)
 */
class ConcatFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"Concat"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [dimension]
        // Concat typically uses dimension 1 for channel concatenation
        int64_t dimension = YamlArgsParser::getInt(argsNode, 0, 1);

        // Allow negative indices and defer strict validation to runtime
        // PyTorch will validate the dimension against actual tensor shape
        // Only validate extreme values that are definitely invalid
        if (dimension < -4 || dimension > 10) {
            throw std::runtime_error("Concat dimension out of reasonable range: " +
                                    std::to_string(dimension) +
                                    " (expected -4 to 10 for typical tensors)");
        }

        ModuleBuildResult result;
        result.module = torch::nn::AnyModule(Modules::Concat(dimension));
        result.outputChannels = ctx.inputChannels;
        return result;
    }
};

/**
 * @brief Factory for MaxPool2d module
 * YAML args: [kernel_size, stride?, padding?]
 * Constructor: MaxPool2dImpl(k, s, p)
 */
class MaxPool2dFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"MaxPool2d", "nn.MaxPool2d"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        int64_t k = YamlArgsParser::getInt(argsNode, 0, 2);
        int64_t s = YamlArgsParser::getInt(argsNode, 1, -1);  // -1 means use k as stride

        std::optional<int64_t> p = std::nullopt;
        if (argsNode.IsSequence() && argsNode.size() > 2) {
            p = YamlArgsParser::getInt(argsNode, 2, 0);
        }

        ModuleBuildResult result;
        result.module = torch::nn::AnyModule(Modules::MaxPool2d(k, s, p));
        result.outputChannels = ctx.inputChannels;
        return result;
    }
};

/**
 * @brief Factory for AvgPool2d module
 * YAML args: [kernel_size, stride?, padding?]
 * Constructor: AvgPool2dImpl(k, s, p)
 */
class AvgPool2dFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"AvgPool2d", "nn.AvgPool2d"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        int64_t k = YamlArgsParser::getInt(argsNode, 0, 2);
        int64_t s = YamlArgsParser::getInt(argsNode, 1, -1);  // -1 means use k as stride
        std::optional<int64_t> p = std::nullopt;
        if (argsNode.IsSequence() && argsNode.size() > 2) 
        {
            p = YamlArgsParser::getInt(argsNode, 2, 0);
        }

        ModuleBuildResult result;
        result.module = torch::nn::AnyModule(Modules::AvgPool2d(k, s, p));
        result.outputChannels = ctx.inputChannels;
        return result;
    }
};

/**
 * @brief Factory for GlobalAvgPool module
 * YAML args: [] (no arguments)
 * Constructor: GlobalAvgPoolImpl()
 */
class GlobalAvgPoolFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"GlobalAvgPool", "nn.AdaptiveAvgPool2d"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        ModuleBuildResult result;
        result.module = torch::nn::AnyModule(Modules::GlobalAvgPool());
        result.outputChannels = ctx.inputChannels;
        return result;
    }
};

} // namespace Builder
} // namespace Model
} // namespace WheelDL
