#include "pch.h"
#include "ModuleFactoryRegistry.h"
#include "ConvolutionFactory.h"
#include "BlockFactory.h"
#include "UtilityFactory.h"
#include "AttentionFactory.h"
#include "HeadFactory.h"
#include "TransformerFactory.h"

namespace WheelDL {
    namespace Model {
        namespace Builder {

            ModuleFactoryRegistry& ModuleFactoryRegistry::instance() {
                static ModuleFactoryRegistry registry;
                return registry;
            }

            void ModuleFactoryRegistry::registerFactory(const std::string& type,
                std::shared_ptr<IModuleFactory> factory) {
                // Validate input
                if (!factory) {
                    throw std::invalid_argument(
                        "Cannot register null factory for type: " + type
                    );
                }

                if (type.empty()) {
                    throw std::invalid_argument(
                        "Cannot register factory with empty type name"
                    );
                }

                // Check for duplicate registration
                auto [it, inserted] = _factories.insert({type, factory});
                if (!inserted) {
                    throw std::runtime_error(
                        "Factory already registered for type: " + type
                    );
                }
            }

            void ModuleFactoryRegistry::registerAlias(const std::string& alias,
                const std::string& existingType) {
                // Check if alias is already registered
                if (_factories.find(alias) != _factories.end()) {
                    throw std::runtime_error(
                        "Alias already registered: " + alias
                    );
                }

                // Find the existing factory
                auto it = _factories.find(existingType);
                if (it == _factories.end()) {
                    throw std::runtime_error(
                        "Cannot create alias '" + alias +
                        "': existing type '" + existingType + "' not found"
                    );
                }

                // Register alias with same factory instance
                _factories[alias] = it->second;
            }

            IModuleFactory* ModuleFactoryRegistry::getFactory(const std::string& type) {
                auto it = _factories.find(type);
                if (it != _factories.end()) {
                    return it->second.get();
                }
                return nullptr;
            }

            ModuleFactoryRegistry::ModuleFactoryRegistry() {
                registerDefaultFactories();
            }

            void ModuleFactoryRegistry::registerDefaultFactories() {
                // Convolution factories
                registerFactory("Conv", std::make_shared<ConvFactory>());
                registerFactory("NaiveConv", std::make_shared<NaiveConvFactory>());
                registerFactory("DWConv", std::make_shared<DWConvFactory>());
                registerFactory("RepConv", std::make_shared<RepConvFactory>());
                registerFactory("GhostConv", std::make_shared<GhostConvFactory>());
                registerFactory("Focus", std::make_shared<FocusFactory>());
                registerFactory("ConvTranspose", std::make_shared<ConvTransposeFactory>());
                registerAlias("nn.ConvTranspose2d", "ConvTranspose");

                // Block factories
                registerFactory("Bottleneck", std::make_shared<BottleneckFactory>());
                registerFactory("C2f", std::make_shared<C2fFactory>());
                registerFactory("C2", std::make_shared<C2Factory>());
                registerFactory("C3", std::make_shared<C3Factory>());
                registerFactory("C3x", std::make_shared<C3xFactory>());
                registerFactory("C3k2", std::make_shared<C3k2Factory>());
                registerFactory("C2fPSA", std::make_shared<C2fPSAFactory>());
                registerFactory("C2PSA", std::make_shared<C2PSAFactory>());
                registerFactory("RepC3", std::make_shared<RepC3Factory>());
                registerFactory("C3Ghost", std::make_shared<C3GhostFactory>());
                registerFactory("BottleneckCSP", std::make_shared<BottleneckCSPFactory>());
                registerFactory("SPP", std::make_shared<SPPFactory>());
                registerFactory("SPPF", std::make_shared<SPPFFactory>());
                registerFactory("ResNetLayer", std::make_shared<ResNetLayerFactory>());
                registerFactory("ResNetBlock", std::make_shared<ResNetBlockFactory>());
                registerFactory("DBlock", std::make_shared<DBlockFactory>());
                registerAlias("DenseBlock", "DBlock");
                registerFactory("CNXBlock", std::make_shared<CNXBlockFactory>());
                registerAlias("ConvNeXtBlock", "CNXBlock");
                registerFactory("TransformerBlock", std::make_shared<TransformerBlockFactory>());

                // Utility factories (ConvTranspose, Upsample, Concat, Pooling)
                registerFactory("Upsample", std::make_shared<UpsampleFactory>());
                registerAlias("nn.Upsample", "Upsample");
                registerFactory("Concat", std::make_shared<ConcatFactory>());
                registerFactory("MaxPool2d", std::make_shared<MaxPool2dFactory>());
                registerAlias("nn.MaxPool2d", "MaxPool2d");
                registerFactory("AvgPool2d", std::make_shared<AvgPool2dFactory>());
                registerAlias("nn.AvgPool2d", "AvgPool2d");
                registerFactory("GlobalAvgPool", std::make_shared<GlobalAvgPoolFactory>());
                registerAlias("nn.AdaptiveAvgPool2d", "GlobalAvgPool");

                // Attention factories
                registerFactory("PSA", std::make_shared<PSAFactory>());
                registerFactory("CBAM", std::make_shared<CBAMFactory>());
                registerFactory("Attention", std::make_shared<AttentionFactory>());

                // Transformer factories
                registerFactory("LayerNorm2d", std::make_shared<LayerNorm2dFactory>());
                registerAlias("nn.LayerNorm", "LayerNorm2d");
                registerFactory("MLPBlock", std::make_shared<MLPBlockFactory>());
                registerFactory("MLP", std::make_shared<MLPFactory>());
                registerFactory("TransformerLayer", std::make_shared<TransformerLayerFactory>());
                registerFactory("TransformerEncoderLayer", std::make_shared<TransformerEncoderLayerFactory>());
                registerFactory("AIFI", std::make_shared<AIFIFactory>());
                registerFactory("MSDeformAttn", std::make_shared<MSDeformAttnFactory>());
                registerFactory("DeformableTransformerDecoderLayer", std::make_shared<DeformableTransformerDecoderLayerFactory>());
                registerFactory("DeformableTransformerDecoder", std::make_shared<DeformableTransformerDecoderFactory>());

                // Head factories
                registerFactory("Detect", std::make_shared<DetectFactory>());
                registerFactory("OBB", std::make_shared<OBBFactory>());
                registerFactory("Classify", std::make_shared<ClassifyFactory>());
                registerFactory("Segment", std::make_shared<SegmentFactory>());
                registerFactory("Anomaly", std::make_shared<AnomalyFactory>());
            }

        } // namespace Builder
    } // namespace Model
} // namespace WheelDL
