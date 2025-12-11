#pragma once

#include "BaseContext.h"
#include "../Request.h"
#include "../../Engine/BaseValidator.h"

namespace WheelDL {
namespace Core {
namespace Manager {

/**
 * @class ValidateContext
 * @brief Context for validation operations
 */
class ValidateContext : public BaseContext {
public:
    explicit ValidateContext(const ValidateRequest& request);
    ~ValidateContext() override = default;

    TaskResult run() override;

private:
    std::unique_ptr<Validator::BaseValidator> createValidator();

private:
    std::string _checkpointPath;
};

} // namespace Manager
} // namespace Core
} // namespace WheelDL
