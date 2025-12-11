#pragma once

#include "Types.h"
#include "../../Config/Configuration.h"
#include <memory>
#include <string>

namespace WheelDL {
namespace Core {
namespace Manager {

/**
 * @struct BaseRequest
 * @brief Base structure for all task requests
 *
 * Contains common fields for all operation types.
 * Uses ProgressCallback from Utils/Common/Types.h
 */
struct BaseRequest {
    std::shared_ptr<Config::Configuration> config;  ///< Configuration for the task
    OperationType operationType;                     ///< Type of operation
    TaskPriority priority = TaskPriority::NORMAL;    ///< Task priority

    virtual ~BaseRequest() = default;

protected:
    BaseRequest(std::shared_ptr<Config::Configuration> cfg, OperationType type)
        : config(std::move(cfg)), operationType(type) {}
};

/**
 * @struct TrainRequest
 * @brief Request for training operation
 *
 * Uses ProgressCallback from Utils/Common/Types.h
 */
struct TrainRequest : public BaseRequest {
    ProgressCallback progressCallback;              ///< Optional progress callback (from Utils/Common/Types.h)

    explicit TrainRequest(std::shared_ptr<Config::Configuration> cfg)
        : BaseRequest(std::move(cfg), OperationType::TRAIN) {}
};

/**
 * @struct ValidateRequest
 * @brief Request for validation operation
 */
struct ValidateRequest : public BaseRequest {
    std::string checkpointPath;                     ///< Path to model checkpoint

    explicit ValidateRequest(std::shared_ptr<Config::Configuration> cfg)
        : BaseRequest(std::move(cfg), OperationType::VALIDATE) {}
};

/**
 * @struct PredictRequest
 * @brief Request for prediction/inference operation
 */
struct PredictRequest : public BaseRequest {
    std::string checkpointPath;                     ///< Path to model checkpoint

    explicit PredictRequest(std::shared_ptr<Config::Configuration> cfg)
        : BaseRequest(std::move(cfg), OperationType::PREDICT) {}
};

} // namespace Manager
} // namespace Core
} // namespace WheelDL
