#pragma once

#include "ModuleFactory.h"
#include <memory>
#include <unordered_map>
#include <string>

namespace WheelDL {
    namespace Model {
        namespace Builder {

            /**
             * @brief Singleton registry for module factories
             *
             * This registry maintains a mapping from module type names to their
             * corresponding factory instances. It provides a centralized place to
             * register and retrieve module factories.
             */
            class ModuleFactoryRegistry {
            public:
                /**
                 * @brief Get the singleton instance
                 *
                 * @return ModuleFactoryRegistry& Singleton instance
                 */
                static ModuleFactoryRegistry& instance();

                /**
                 * @brief Register a factory for one or more module types
                 *
                 * @param type Module type name
                 * @param factory Shared pointer to factory instance
                 * @throws std::invalid_argument if factory is null or type is empty
                 * @throws std::runtime_error if type is already registered
                 */
                void registerFactory(const std::string& type, std::shared_ptr<IModuleFactory> factory);

                /**
                 * @brief Register an alias for an existing factory
                 *
                 * This allows multiple type names to map to the same factory instance.
                 * For example, "nn.Upsample" can be an alias for "Upsample".
                 *
                 * @param alias Alias type name
                 * @param existingType Existing registered type name
                 * @throws std::runtime_error if existingType is not found or alias already exists
                 */
                void registerAlias(const std::string& alias, const std::string& existingType);

                /**
                 * @brief Get factory for a specific module type
                 *
                 * @param type Module type name
                 * @return IModuleFactory* Pointer to factory, or nullptr if not found
                 */
                IModuleFactory* getFactory(const std::string& type);

                /**
                 * @brief Register all default factories
                 *
                 * This method is called during initialization to register all
                 * built-in module factories.
                 */
                void registerDefaultFactories();

            private:
                ModuleFactoryRegistry();
                ~ModuleFactoryRegistry() = default;

                // Delete copy constructor and assignment operator
                ModuleFactoryRegistry(const ModuleFactoryRegistry&) = delete;
                ModuleFactoryRegistry& operator=(const ModuleFactoryRegistry&) = delete;

                std::unordered_map<std::string, std::shared_ptr<IModuleFactory>> _factories;
            };

        } // namespace Builder
    } // namespace Model
} // namespace WheelDL
