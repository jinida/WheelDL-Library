#pragma once

#include "BaseContext.h"
#include "../Request.h"
#include "../../Engine/BasePredictor.h"

namespace WheelDL {
namespace Core {
namespace Manager {

/**
 * @class PredictContext
 * @brief Context for prediction operations
 */
class PredictContext : public BaseContext {
public:
    explicit PredictContext(const PredictRequest& request);
    ~PredictContext() override = default;

    TaskResult run() override;

private:
    std::unique_ptr<Predictor::BasePredictor> createPredictor();

private:
    std::string _checkpointPath;
};

} // namespace Manager
} // namespace Core
} // namespace WheelDL
