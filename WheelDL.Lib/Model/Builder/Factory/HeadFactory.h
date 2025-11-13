#pragma once

#include "ModuleFactory.h"
#include "../../Modules/Head.h"

namespace WheelDL {
namespace Model {
namespace Builder {

/**
 * @brief Factory for Detect module
 */
class DetectFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"Detect"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        ModuleBuildResult result;
        result.module = torch::nn::AnyModule(Modules::Detect(ctx.numClasses, ctx.headChannels));
        result.outputChannels = ctx.numClasses + 64;
        return result;
    }
};

/**
 * @brief Factory for OBB module
 */
class OBBFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"OBB"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        ModuleBuildResult result;
        result.module = torch::nn::AnyModule(Modules::OBB(ctx.numClasses, 1, ctx.headChannels));
        result.outputChannels = ctx.numClasses + 64 + 1;
        return result;
    }
};

/**
 * @brief Factory for Classify module
 */
class ClassifyFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"Classify"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode, const ModuleBuildContext& ctx) override
    {
        ModuleBuildResult result;
        result.module = torch::nn::AnyModule(Modules::Classify(ctx.inputChannels, ctx.numClasses));
        result.outputChannels = ctx.numClasses;
        return result;
    }
};

/**
 * @brief Factory for Segment module
 */
class SegmentFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"Segment"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode, const ModuleBuildContext& ctx) override
    {
        ModuleBuildResult result;
        result.module = torch::nn::AnyModule(Modules::Segment(ctx.inputChannels, ctx.numClasses, ctx.defaultAct));
        result.outputChannels = ctx.numClasses;
        return result;
    }
};

} // namespace Builder
} // namespace Model
} // namespace WheelDL
