#pragma once

#include "ModuleFactory.h"
#include "../../Modules/Head.h"
#include "../../Modules/Model.h"

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

/**
 * @brief Factory for Anomaly module
 * YAML args: [model_type, ...model-specific args]
 * Example: ["EfficientAD", 384, True] or ["PatchCore", ...] or ["SimpleNet", ...]
 */
class AnomalyFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"Anomaly"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode, const ModuleBuildContext& ctx) override
    {
        // Parse model type from args[0] (default: "EfficientAD")
        std::string modelType = YamlArgsParser::getString(argsNode, 0, "EfficientAD");

        // Create the anomaly model based on type
        std::shared_ptr<Modules::IAnomalyModel> anomalyModel = createAnomalyModel(modelType, argsNode);

        ModuleBuildResult result;
        result.module = torch::nn::AnyModule(Modules::Anomaly(anomalyModel));
        result.outputChannels = anomalyModel->getOutputChannels();
        return result;
    }

private:
    std::shared_ptr<Modules::IAnomalyModel> createAnomalyModel(
        const std::string& modelType,
        const YAML::Node& argsNode)
    {
        if (modelType == "EfficientAD") {
            // args: [modelType, outChannels, small]
            int64_t outChannels = YamlArgsParser::getInt(argsNode, 1, 384);
            bool small = YamlArgsParser::getBool(argsNode, 2, true);
            return std::make_shared<Modules::EfficientADImpl>(outChannels, small);
        }
        else if (modelType == "PatchCore") {
            // args: [modelType, numNeighbors, maxMemoryBankPatches, projectedDim]
            int64_t numNeighbors = YamlArgsParser::getInt(argsNode, 1, 9);
            int64_t maxMemoryBankPatches = YamlArgsParser::getInt(argsNode, 2, 25600);
            int64_t projectedDim = YamlArgsParser::getInt(argsNode, 3, 128);  // 0 = no projection
            return std::make_shared<Modules::PatchCoreImpl>(numNeighbors, maxMemoryBankPatches, projectedDim);
        }
        else if (modelType == "SimpleNet") {
            // TODO: Implement SimpleNet creation
            throw std::runtime_error("SimpleNet not implemented yet");
        }
        else {
            throw std::runtime_error("Unknown anomaly model type: " + modelType);
        }
    }
};

} // namespace Builder
} // namespace Model
} // namespace WheelDL
