#pragma once

#include "ExportTypes.h"
#include "../Utils/Error/ErrorCodes.h"
#include "../Utils/Error/WheelLibException.h"
#include <unordered_map>
#include <mutex>
#include <memory>
#include <atomic>

namespace WheelDL {
namespace Export {

/**
 * @class HandleRegistry
 * @brief Thread-safe registry for managing opaque handles to C++ objects
 *
 * Provides a type-safe mapping between void* handles and actual C++ objects.
 * All operations are thread-safe.
 *
 * @tparam T The type of object to manage
 */
template<typename T>
class HandleRegistry {
public:
    /**
     * @brief Get singleton instance
     * @return HandleRegistry& Singleton reference
     */
    static HandleRegistry& getInstance() {
        static HandleRegistry instance;
        return instance;
    }

    /**
     * @brief Register a shared_ptr and return an opaque handle
     * @param obj Shared pointer to object
     * @return WheelHandle Opaque handle
     * @throws WheelLibException if registration fails
     */
    WheelHandle registerObject(std::shared_ptr<T> obj) {
        if (!obj) {
            throw Utils::WheelLibException(
                Utils::ErrorCode::INVALID_ARGUMENT,
                "Cannot register null object");
        }

        std::lock_guard<std::mutex> lock(_mutex);

        // Generate unique handle ID
        uint64_t id = _nextId++;
        WheelHandle handle = reinterpret_cast<WheelHandle>(id);

        _registry[id] = std::move(obj);
        return handle;
    }

    /**
     * @brief Get object from handle
     * @param handle Opaque handle
     * @return std::shared_ptr<T> Object pointer
     * @throws WheelLibException if handle is invalid
     */
    std::shared_ptr<T> getObject(WheelHandle handle) const {
        if (!handle) {
            throw Utils::WheelLibException(
                Utils::ErrorCode::INVALID_ARGUMENT,
                "Invalid null handle");
        }

        std::lock_guard<std::mutex> lock(_mutex);

        uint64_t id = reinterpret_cast<uint64_t>(handle);
        auto it = _registry.find(id);
        if (it == _registry.end()) {
            throw Utils::WheelLibException(
                Utils::ErrorCode::INVALID_ARGUMENT,
                "Handle not found in registry");
        }

        return it->second;
    }

    /**
     * @brief Check if handle is valid
     * @param handle Opaque handle
     * @return bool True if handle exists in registry
     */
    bool isValid(WheelHandle handle) const {
        if (!handle) return false;

        std::lock_guard<std::mutex> lock(_mutex);
        uint64_t id = reinterpret_cast<uint64_t>(handle);
        return _registry.find(id) != _registry.end();
    }

    /**
     * @brief Unregister handle and release object
     * @param handle Opaque handle
     * @return bool True if handle was found and removed
     */
    bool unregisterObject(WheelHandle handle) {
        if (!handle) return false;

        std::lock_guard<std::mutex> lock(_mutex);
        uint64_t id = reinterpret_cast<uint64_t>(handle);
        return _registry.erase(id) > 0;
    }

    /**
     * @brief Get number of registered handles
     * @return size_t Number of handles
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(_mutex);
        return _registry.size();
    }

    /**
     * @brief Clear all registered handles
     */
    void clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        _registry.clear();
    }

private:
    HandleRegistry() : _nextId(1) {}
    ~HandleRegistry() = default;

    HandleRegistry(const HandleRegistry&) = delete;
    HandleRegistry& operator=(const HandleRegistry&) = delete;

    mutable std::mutex _mutex;
    std::unordered_map<uint64_t, std::shared_ptr<T>> _registry;
    std::atomic<uint64_t> _nextId;
};

} // namespace Export
} // namespace WheelDL
