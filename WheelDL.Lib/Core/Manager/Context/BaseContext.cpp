#include "pch.h"
#include "BaseContext.h"
#include <sstream>
#include <iomanip>

namespace WheelDL {
namespace Core {
namespace Manager {

std::atomic<uint64_t> BaseContext::_taskIdCounter{0};

BaseContext::BaseContext(
    std::shared_ptr<Config::Configuration> config,
    OperationType operationType)
    : _operationType(operationType)
    , _config(config)
{
    _taskId = generateTaskId(operationType);
    initializeResources();
}

void BaseContext::initializeResources() {
    // Logger: Factory pattern
    _logger = WheelDL::Utils::Logger::create(_taskId);

    // Workspace: Direct constructor (baseDir="runs", prefix=taskId)
    _workspace = std::make_unique<WheelDL::Utils::Workspace>("runs", _taskId, false);

    // Profiler: Factory pattern
    _profiler = WheelDL::Utils::PerformanceProfiler::create();

    // Logger setup: Use Workspace log directory
    _logger->setLogFile(_workspace->getLogsDir() + "/" + _taskId + ".log");

    _logger->info("Context", "Resources initialized for: " + _taskId);
}

std::string BaseContext::generateTaskId(OperationType type) {
    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << operationTypeToString(type) << "_";
    ss << std::put_time(std::localtime(&timeT), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
    ss << "_" << _taskIdCounter.fetch_add(1, std::memory_order_relaxed);

    return ss.str();
}

void BaseContext::requestStop() {
    _stopRequested.store(true, std::memory_order_release);
    _logger->info("Context", "Stop requested");
}

bool BaseContext::isStopRequested() const {
    return _stopRequested.load(std::memory_order_acquire);
}

} // namespace Manager
} // namespace Core
} // namespace WheelDL
