#pragma once

#include "ModuleFactory.h"
#include "../../Modules/Block.h"
#include "../../Modules/Transformer.h"

namespace WheelDL {
    namespace Model {
        namespace Builder {

            /**
             * @brief Factory for Bottleneck module
             * YAML args: [out_channels, shortcut?, g?, k?, e?]
             * Constructor: BottleneckImpl(c1, c2, shortcut, g, k, e)
             */
            class BottleneckFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "Bottleneck" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    // Parse YAML args: [out_channels, shortcut?, g?, k?, e?]
                    int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 256);
                    bool shortcut = YamlArgsParser::getBool(argsNode, 1, true);
                    int64_t g = YamlArgsParser::getInt(argsNode, 2, 1);
                    // k is vector, use default {3, 3}
                    double e = YamlArgsParser::getDouble(argsNode, 3, 0.5);

                    auto scaled = ctx.applyScale(outCh, 1);

                    ModuleBuildResult result;
                    // Constructor: BottleneckImpl(c1, c2, shortcut, g, k, e)
                    result.module = torch::nn::AnyModule(
                        Modules::Bottleneck(ctx.inputChannels, scaled.first, shortcut, g, std::vector<int64_t>{3, 3}, e)
                    );
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for C2f module
             * YAML args: [out_channels, shortcut?, g?, e?]
             * Constructor: C2fImpl(c1, c2, n, shortcut, g, e)
             */
            class C2fFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "C2f" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    // Parse YAML args: [out_channels, shortcut?, g?, e?]
                    int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 256);
                    bool shortcut = YamlArgsParser::getBool(argsNode, 1, false);
                    int64_t g = YamlArgsParser::getInt(argsNode, 2, 1);
                    double e = YamlArgsParser::getDouble(argsNode, 3, 0.5);

                    auto scaled = ctx.applyScale(outCh, ctx.repeats);

                    ModuleBuildResult result;
                    // Constructor: C2fImpl(c1, c2, n, shortcut, g, e)
                    result.module = torch::nn::AnyModule(
                        Modules::C2f(ctx.inputChannels, scaled.first, scaled.second, shortcut, g, e)
                    );
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for C2 module
             * YAML args: [out_channels, shortcut?, g?, e?]
             * Constructor: C2Impl(c1, c2, n, shortcut, g, e)
             */
            class C2Factory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "C2" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    // Parse YAML args: [out_channels, shortcut?, g?, e?]
                    int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 256);
                    bool shortcut = YamlArgsParser::getBool(argsNode, 1, true);
                    int64_t g = YamlArgsParser::getInt(argsNode, 2, 1);
                    double e = YamlArgsParser::getDouble(argsNode, 3, 0.5);

                    auto scaled = ctx.applyScale(outCh, ctx.repeats);

                    ModuleBuildResult result;
                    // Constructor: C2Impl(c1, c2, n, shortcut, g, e)
                    result.module = torch::nn::AnyModule(
                        Modules::C2(ctx.inputChannels, scaled.first, scaled.second, shortcut, g, e)
                    );
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for C3 module
             * YAML args: [out_channels, shortcut?, g?, e?]
             * Constructor: C3Impl(c1, c2, n, shortcut, g, e)
             */
            class C3Factory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "C3" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    // Parse YAML args: [out_channels, shortcut?, g?, e?]
                    int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 256);
                    bool shortcut = YamlArgsParser::getBool(argsNode, 1, true);
                    int64_t g = YamlArgsParser::getInt(argsNode, 2, 1);
                    double e = YamlArgsParser::getDouble(argsNode, 3, 0.5);

                    auto scaled = ctx.applyScale(outCh, ctx.repeats);

                    ModuleBuildResult result;
                    // Constructor: C3Impl(c1, c2, n, shortcut, g, e)
                    result.module = torch::nn::AnyModule(
                        Modules::C3(ctx.inputChannels, scaled.first, scaled.second, shortcut, g, e)
                    );
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for C3x module
             * YAML args: [out_channels, shortcut?, g?, e?]
             * Constructor: C3xImpl(c1, c2, n, shortcut, g, e)
             */
            class C3xFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "C3x" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    // Parse YAML args: [out_channels, shortcut?, g?, e?]
                    int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 256);
                    bool shortcut = YamlArgsParser::getBool(argsNode, 1, true);
                    int64_t g = YamlArgsParser::getInt(argsNode, 2, 1);
                    double e = YamlArgsParser::getDouble(argsNode, 3, 0.5);

                    auto scaled = ctx.applyScale(outCh, ctx.repeats);

                    ModuleBuildResult result;
                    // Constructor: C3xImpl(c1, c2, n, shortcut, g, e)
                    result.module = torch::nn::AnyModule(
                        Modules::C3x(ctx.inputChannels, scaled.first, scaled.second, shortcut, g, e)
                    );
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for C3k2 module
             */
            class C3k2Factory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "C3k2" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    auto args = argsNode.as<std::vector<int64_t>>();
                    if (args.empty()) {
                        throw std::runtime_error("C3k2 requires at least one argument (output channels)");
                    }
                    int64_t outCh = args[0];
                    bool c3k = args.size() > 1 ? static_cast<bool>(args[1]) : true;

                    auto scaled = ctx.applyScale(outCh, ctx.repeats);

                    ModuleBuildResult result;
                    result.module = torch::nn::AnyModule(Modules::C3k2(ctx.inputChannels, scaled.first, scaled.second, c3k));
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for C2fPSA module
             */
            class C2fPSAFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "C2fPSA" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    auto args = argsNode.as<std::vector<int64_t>>();
                    if (args.empty()) {
                        throw std::runtime_error("C2fPSA requires at least one argument (output channels)");
                    }
                    int64_t outCh = args[0];

                    auto scaled = ctx.applyScale(outCh, ctx.repeats);

                    ModuleBuildResult result;
                    result.module = torch::nn::AnyModule(Modules::C2fPSA(ctx.inputChannels, scaled.first, scaled.second));
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for C2PSA module
             */
            class C2PSAFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "C2PSA" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    auto args = argsNode.as<std::vector<int64_t>>();
                    if (args.empty()) {
                        throw std::runtime_error("C2PSA requires at least one argument (output channels)");
                    }
                    int64_t outCh = args[0];

                    auto scaled = ctx.applyScale(outCh, ctx.repeats);

                    ModuleBuildResult result;
                    result.module = torch::nn::AnyModule(Modules::C2PSA(ctx.inputChannels, scaled.first, scaled.second));
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for RepC3 module
             */
            class RepC3Factory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "RepC3" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    auto args = argsNode.as<std::vector<int64_t>>();
                    if (args.empty()) {
                        throw std::runtime_error("RepC3 requires at least one argument (output channels)");
                    }
                    int64_t outCh = args[0];

                    auto scaled = ctx.applyScale(outCh, ctx.repeats);

                    ModuleBuildResult result;
                    result.module = torch::nn::AnyModule(Modules::RepC3(ctx.inputChannels, scaled.first, scaled.second));
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for C3Ghost module
             * YAML args: [out_channels, shortcut?, g?, e?]
             * Constructor: C3GhostImpl(c1, c2, n, shortcut, g, e)
             */
            class C3GhostFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "C3Ghost" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    // Parse YAML args: [out_channels, shortcut?, g?, e?]
                    int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 256);
                    bool shortcut = YamlArgsParser::getBool(argsNode, 1, true);
                    int64_t g = YamlArgsParser::getInt(argsNode, 2, 1);
                    double e = YamlArgsParser::getDouble(argsNode, 3, 0.5);

                    auto scaled = ctx.applyScale(outCh, ctx.repeats);

                    ModuleBuildResult result;
                    // Constructor: C3GhostImpl(c1, c2, n, shortcut, g, e)
                    result.module = torch::nn::AnyModule(
                        Modules::C3Ghost(ctx.inputChannels, scaled.first, scaled.second, shortcut, g, e)
                    );
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for BottleneckCSP module
             */
            class BottleneckCSPFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "BottleneckCSP" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    auto args = argsNode.as<std::vector<int64_t>>();
                    if (args.empty()) {
                        throw std::runtime_error("BottleneckCSP requires at least one argument (output channels)");
                    }
                    int64_t outCh = args[0];

                    auto scaled = ctx.applyScale(outCh, ctx.repeats);

                    ModuleBuildResult result;
                    result.module = torch::nn::AnyModule(Modules::BottleneckCSP(ctx.inputChannels, scaled.first, scaled.second));
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for SPP module
             */
            class SPPFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "SPP" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    auto args = argsNode.as<std::vector<int64_t>>();
                    if (args.size() < 2) {
                        throw std::runtime_error("SPP requires 2 arguments: [out_channels, kernel_size]");
                    }
                    int64_t outCh = args[0];
                    int64_t k = args[1];

                    auto scaled = ctx.applyScale(outCh, 1);

                    ModuleBuildResult result;
                    result.module = torch::nn::AnyModule(Modules::SPP(ctx.inputChannels, scaled.first, std::vector<int64_t>{static_cast<int64_t>(k)}));
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for SPPF module
             */
            class SPPFFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "SPPF" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    auto args = argsNode.as<std::vector<int64_t>>();
                    if (args.size() < 2) {
                        throw std::runtime_error("SPPF requires 2 arguments: [out_channels, kernel_size]");
                    }
                    int64_t outCh = args[0];
                    int64_t k = args[1];

                    auto scaled = ctx.applyScale(outCh, 1);

                    ModuleBuildResult result;
                    result.module = torch::nn::AnyModule(Modules::SPPF(ctx.inputChannels, scaled.first, k));
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for ResNetBlock module
             * YAML args: [out_channels, stride?, e?]
             * Constructor: ResNetBlockImpl(c1, c2, s, e)
             */
            class ResNetBlockFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "ResNetBlock" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    // Parse YAML args: [out_channels, stride?, e?]
                    int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 256);
                    int64_t s = YamlArgsParser::getInt(argsNode, 1, 1);
                    double e = YamlArgsParser::getDouble(argsNode, 2, 4.0);

                    auto scaled = ctx.applyScale(outCh, 1);

                    ModuleBuildResult result;
                    // Constructor: ResNetBlockImpl(c1, c2, s, e)
                    result.module = torch::nn::AnyModule(
                        Modules::ResNetBlock(ctx.inputChannels, scaled.first, s, e)
                    );
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for ResNetLayer module
             * YAML args: [out_channels, stride?, is_first?, e?]
             * Constructor: ResNetLayerImpl(c1, c2, s, is_first, n, e)
             */
            class ResNetLayerFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "ResNetLayer" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 256);
                    int64_t s = YamlArgsParser::getInt(argsNode, 1, 1);
                    bool isFirst = YamlArgsParser::getBool(argsNode, 2, false);
                    double e = YamlArgsParser::getDouble(argsNode, 3, 4.0);

                    auto scaled = ctx.applyScale(outCh, ctx.repeats);

                    ModuleBuildResult result;
                    result.module = torch::nn::AnyModule(
                        Modules::ResNetLayer(ctx.inputChannels, scaled.first, s, isFirst, scaled.second, e)
                    );
                    result.outputChannels = isFirst ? scaled.first : scaled.first * e;
                    return result;
                }
            };

            /**
             * @brief Factory for DBlock (DenseBlock) module
             * YAML args: [out_channels, growth_rate, bn_size?]
             * Constructor: DBlockImpl(c1, c2, growth_rate, bn_size, act)
             */
            class DBlockFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "DBlock", "DenseBlock" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    // Parse YAML args: [out_channels, growth_rate, bn_size?]
                    int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 256);
                    int64_t growthRate = YamlArgsParser::getInt(argsNode, 1, 32);
                    int64_t bnSize = YamlArgsParser::getInt(argsNode, 2, 4);

                    // Apply scaling to output channels
                    auto scaled = ctx.applyScale(outCh, 1);
                    ModuleBuildResult result;
                    // Constructor: DBlockImpl(c1, c2, growth_rate, bn_size, act)
                    result.module = torch::nn::AnyModule(Modules::DBlock(ctx.inputChannels, scaled.first, growthRate, bnSize, ctx.defaultAct));
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

            /**
             * @brief Factory for CNXBlock (ConvNeXt Block)
             * YAML args: [channels, layer_scale_init?]
             * Constructor: CNXBlockImpl(dim, layer_scale_init)
             *
             * Note: channels input/output are the same (residual block)
             */
            class CNXBlockFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "CNXBlock", "ConvNeXtBlock" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    // Parse YAML args: [channels, layer_scale_init?]
                    int64_t channels = YamlArgsParser::getInt(argsNode, 0, 256);
                    double layerScaleInit = YamlArgsParser::getDouble(argsNode, 1, 1e-6);

                    // Apply scaling to channels
                    auto scaled = ctx.applyScale(channels, 1);

                    ModuleBuildResult result;
                    // Constructor: CNXBlockImpl(dim, layer_scale_init)
                    result.module = torch::nn::AnyModule(Modules::CNXBlock(scaled.first, layerScaleInit));
                    result.outputChannels = scaled.first;  // Output channels = input channels
                    return result;
                }
            };

            /**
             * @brief Factory for TransformerBlock
             * YAML args: [out_channels, num_heads, num_layers]
             * Constructor: TransformerBlockImpl(c1, c2, num_heads, num_layers)
             */
            class TransformerBlockFactory : public IModuleFactory {
            public:
                std::vector<std::string> supportedTypes() const override {
                    return { "TransformerBlock" };
                }

                ModuleBuildResult create(const YAML::Node& argsNode,
                    const ModuleBuildContext& ctx) override {
                    // Parse YAML args: [out_channels, num_heads, num_layers]
                    int64_t outCh = YamlArgsParser::getInt(argsNode, 0, 256);
                    int64_t numHeads = YamlArgsParser::getInt(argsNode, 1, 8);
                    int64_t numLayers = YamlArgsParser::getInt(argsNode, 2, 1);

                    auto scaled = ctx.applyScale(outCh, 1);

                    ModuleBuildResult result;
                    // Constructor: TransformerBlockImpl(c1, c2, num_heads, num_layers)
                    result.module = torch::nn::AnyModule(
                        Modules::TransformerBlock(ctx.inputChannels, scaled.first, numHeads, numLayers)
                    );
                    result.outputChannels = scaled.first;
                    return result;
                }
            };

        } // namespace Builder
    } // namespace Model
} // namespace WheelDL
