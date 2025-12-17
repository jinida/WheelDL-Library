#pragma once

#include "ExportTypes.h"
#include "../Utils/Common/Types.h"
#include <functional>

namespace WheelDL {
namespace Export {

/**
 * @class CallbackBridge
 * @brief Bridge between C++ ProgressCallback and C-style WheelProgressCallback
 *
 * Converts WheelDL::ProgressData to WheelProgressData and invokes the C callback.
 * Handles thread safety and exception safety at the callback boundary.
 */
class CallbackBridge {
public:
    /**
     * @brief Create a ProgressCallback that bridges to a C-style callback
     * @param callback C-style callback function pointer (may be null)
     * @param userData User data pointer to pass to callback
     * @return ProgressCallback C++ callback that invokes the C callback
     */
    static ProgressCallback createBridge(WheelProgressCallback callback, void* userData);

    /**
     * @brief Convert ProgressData to WheelProgressData
     * @param src C++ ProgressData
     * @param dst C-style WheelProgressData
     */
    static void convertProgressData(const ProgressData& src, WheelProgressData& dst);

    /**
     * @brief Convert MetricsData to WheelMetricsData
     * @param src C++ MetricsData
     * @param dst C-style WheelMetricsData
     */
    static void convertMetricsData(const MetricsData& src, WheelMetricsData& dst);

    /**
     * @brief Convert WheelProgressStage from C++ ProgressStage
     * @param stage C++ ProgressStage
     * @return WheelProgressStage C-style stage enum
     */
    static WheelProgressStage convertProgressStage(ProgressStage stage);

private:
    CallbackBridge() = delete;
    ~CallbackBridge() = delete;
};

} // namespace Export
} // namespace WheelDL
