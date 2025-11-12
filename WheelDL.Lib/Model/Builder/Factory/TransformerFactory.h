#pragma once

#include "ModuleFactory.h"
#include "../../Modules/Transformer.h"

namespace WheelDL {
namespace Model {
namespace Builder {

/**
 * @brief Factory for LayerNorm2d module
 * YAML args: [num_channels, eps?]
 * Constructor: LayerNorm2dImpl(num_channels, eps)
 */
class LayerNorm2dFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"LayerNorm2d", "nn.LayerNorm"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [num_channels, eps?]
        int64_t numChannels = YamlArgsParser::getInt(argsNode, 0, ctx.inputChannels);
        double eps = YamlArgsParser::getDouble(argsNode, 1, 1e-6);

        auto scaled = ctx.applyScale(numChannels, 1);

        ModuleBuildResult result;
        // Constructor: LayerNorm2dImpl(num_channels, eps)
        result.module = torch::nn::AnyModule(Modules::LayerNorm2d(scaled.first, eps));
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for MLPBlock module
 * YAML args: [embedding_dim, mlp_dim]
 * Constructor: MLPBlockImpl(embedding_dim, mlp_dim)
 */
class MLPBlockFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"MLPBlock"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [embedding_dim, mlp_dim]
        int64_t embeddingDim = YamlArgsParser::getInt(argsNode, 0, 256);
        int64_t mlpDim = YamlArgsParser::getInt(argsNode, 1, 1024);

        auto scaled = ctx.applyScale(embeddingDim, 1);

        ModuleBuildResult result;
        // Constructor: MLPBlockImpl(embedding_dim, mlp_dim)
        result.module = torch::nn::AnyModule(Modules::MLPBlock(scaled.first, mlpDim));
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for MLP module
 * YAML args: [input_dim, hidden_dim, output_dim, num_layers, sigmoid?]
 * Constructor: MLPImpl(input_dim, hidden_dim, output_dim, num_layers, sigmoid)
 */
class MLPFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"MLP"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [input_dim, hidden_dim, output_dim, num_layers, sigmoid?]
        int64_t inputDim = YamlArgsParser::getInt(argsNode, 0, ctx.inputChannels);
        int64_t hiddenDim = YamlArgsParser::getInt(argsNode, 1, 256);
        int64_t outputDim = YamlArgsParser::getInt(argsNode, 2, 256);
        int64_t numLayers = YamlArgsParser::getInt(argsNode, 3, 3);
        bool sigmoid = YamlArgsParser::getBool(argsNode, 4, false);

        auto scaled = ctx.applyScale(outputDim, 1);

        ModuleBuildResult result;
        // Constructor: MLPImpl(input_dim, hidden_dim, output_dim, num_layers, sigmoid)
        result.module = torch::nn::AnyModule(
            Modules::MLP(inputDim, hiddenDim, scaled.first, numLayers, sigmoid)
        );
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for TransformerLayer module
 * YAML args: [channels, num_heads]
 * Constructor: TransformerLayerImpl(c, num_heads)
 */
class TransformerLayerFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"TransformerLayer"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [channels, num_heads]
        int64_t c = YamlArgsParser::getInt(argsNode, 0, ctx.inputChannels);
        int64_t numHeads = YamlArgsParser::getInt(argsNode, 1, 8);

        auto scaled = ctx.applyScale(c, 1);

        ModuleBuildResult result;
        // Constructor: TransformerLayerImpl(c, num_heads)
        result.module = torch::nn::AnyModule(Modules::TransformerLayer(scaled.first, numHeads));
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for TransformerEncoderLayer module
 * YAML args: [c1, cm?, num_heads?, dropout?, normalize_before?]
 * Constructor: TransformerEncoderLayerImpl(c1, cm, num_heads, dropout, normalize_before)
 */
class TransformerEncoderLayerFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"TransformerEncoderLayer"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [c1, cm?, num_heads?, dropout?, normalize_before?]
        int64_t c1 = YamlArgsParser::getInt(argsNode, 0, ctx.inputChannels);
        int64_t cm = YamlArgsParser::getInt(argsNode, 1, 2048);
        int64_t numHeads = YamlArgsParser::getInt(argsNode, 2, 8);
        double dropout = YamlArgsParser::getDouble(argsNode, 3, 0.0);
        bool normalizeBefore = YamlArgsParser::getBool(argsNode, 4, false);

        auto scaled = ctx.applyScale(c1, 1);

        ModuleBuildResult result;
        // Constructor: TransformerEncoderLayerImpl(c1, cm, num_heads, dropout, normalize_before)
        result.module = torch::nn::AnyModule(
            Modules::TransformerEncoderLayer(scaled.first, cm, numHeads, dropout, normalizeBefore)
        );
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for AIFI module
 * YAML args: [c1, cm?, num_heads?, dropout?, normalize_before?]
 * Constructor: AIFIImpl(c1, cm, num_heads, dropout, normalize_before)
 */
class AIFIFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"AIFI"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [c1, cm?, num_heads?, dropout?, normalize_before?]
        int64_t c1 = YamlArgsParser::getInt(argsNode, 0, ctx.inputChannels);
        int64_t cm = YamlArgsParser::getInt(argsNode, 1, 2048);
        int64_t numHeads = YamlArgsParser::getInt(argsNode, 2, 8);
        double dropout = YamlArgsParser::getDouble(argsNode, 3, 0.0);
        bool normalizeBefore = YamlArgsParser::getBool(argsNode, 4, false);

        auto scaled = ctx.applyScale(c1, 1);

        ModuleBuildResult result;
        // Constructor: AIFIImpl(c1, cm, num_heads, dropout, normalize_before)
        result.module = torch::nn::AnyModule(
            Modules::AIFI(scaled.first, cm, numHeads, dropout, normalizeBefore)
        );
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for MSDeformAttn module
 * YAML args: [d_model?, n_levels?, n_heads?, n_points?]
 * Constructor: MSDeformAttnImpl(d_model, n_levels, n_heads, n_points)
 */
class MSDeformAttnFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"MSDeformAttn"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [d_model?, n_levels?, n_heads?, n_points?]
        int64_t dModel = YamlArgsParser::getInt(argsNode, 0, 256);
        int64_t nLevels = YamlArgsParser::getInt(argsNode, 1, 4);
        int64_t nHeads = YamlArgsParser::getInt(argsNode, 2, 8);
        int64_t nPoints = YamlArgsParser::getInt(argsNode, 3, 4);

        auto scaled = ctx.applyScale(dModel, 1);

        ModuleBuildResult result;
        // Constructor: MSDeformAttnImpl(d_model, n_levels, n_heads, n_points)
        result.module = torch::nn::AnyModule(
            Modules::MSDeformAttn(scaled.first, nLevels, nHeads, nPoints)
        );
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for DeformableTransformerDecoderLayer module
 * YAML args: [d_model?, n_heads?, d_ffn?, dropout?, n_levels?, n_points?]
 * Constructor: DeformableTransformerDecoderLayerImpl(d_model, n_heads, d_ffn, dropout, n_levels, n_points)
 */
class DeformableTransformerDecoderLayerFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"DeformableTransformerDecoderLayer"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [d_model?, n_heads?, d_ffn?, dropout?, n_levels?, n_points?]
        int64_t dModel = YamlArgsParser::getInt(argsNode, 0, 256);
        int64_t nHeads = YamlArgsParser::getInt(argsNode, 1, 8);
        int64_t dFfn = YamlArgsParser::getInt(argsNode, 2, 1024);
        double dropout = YamlArgsParser::getDouble(argsNode, 3, 0.0);
        int64_t nLevels = YamlArgsParser::getInt(argsNode, 4, 4);
        int64_t nPoints = YamlArgsParser::getInt(argsNode, 5, 4);

        auto scaled = ctx.applyScale(dModel, 1);

        ModuleBuildResult result;
        // Constructor: DeformableTransformerDecoderLayerImpl(d_model, n_heads, d_ffn, dropout, n_levels, n_points)
        result.module = torch::nn::AnyModule(
            Modules::DeformableTransformerDecoderLayer(
                scaled.first, nHeads, dFfn, dropout, nLevels, nPoints
            )
        );
        result.outputChannels = scaled.first;
        return result;
    }
};

/**
 * @brief Factory for DeformableTransformerDecoder module
 * YAML args: [hidden_dim, num_layers, eval_idx?]
 * Note: This factory requires a decoder_layer to be passed, which is complex.
 * For now, this is a simplified version that creates a default decoder layer internally.
 */
class DeformableTransformerDecoderFactory : public IModuleFactory {
public:
    std::vector<std::string> supportedTypes() const override {
        return {"DeformableTransformerDecoder"};
    }

    ModuleBuildResult create(const YAML::Node& argsNode,
                              const ModuleBuildContext& ctx) override {
        // Parse YAML args: [hidden_dim, num_layers, eval_idx?]
        int64_t hiddenDim = YamlArgsParser::getInt(argsNode, 0, 256);
        int64_t numLayers = YamlArgsParser::getInt(argsNode, 1, 6);
        int64_t evalIdx = YamlArgsParser::getInt(argsNode, 2, -1);

        auto scaled = ctx.applyScale(hiddenDim, 1);

        // Create a default decoder layer
        auto decoderLayer = Modules::DeformableTransformerDecoderLayer(
            scaled.first, 8, 1024, 0.0, 4, 4
        );

        ModuleBuildResult result;
        // Constructor: DeformableTransformerDecoderImpl(hidden_dim, decoder_layer, num_layers, eval_idx)
        result.module = torch::nn::AnyModule(
            Modules::DeformableTransformerDecoder(scaled.first, decoderLayer, numLayers, evalIdx)
        );
        result.outputChannels = scaled.first;
        return result;
    }
};

} // namespace Builder
} // namespace Model
} // namespace WheelDL
