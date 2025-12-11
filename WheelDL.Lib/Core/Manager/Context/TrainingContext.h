#pragma once

#include "BaseContext.h"
#include "../Request.h"
#include "../../Engine/BaseTrainer.h"

namespace WheelDL {
namespace Core {
namespace Manager {

/**
 * @class TrainingContext
 * @brief Context for training operations
 */
class TrainingContext : public BaseContext {
public:
    explicit TrainingContext(const TrainRequest& request);
    ~TrainingContext() override = default;

    TaskResult run() override;

private:
    std::unique_ptr<Trainer::BaseTrainer> createTrainer();

private:
    ProgressCallback _progressCallback;
};

} // namespace Manager
} // namespace Core
} // namespace WheelDL
