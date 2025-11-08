#include "pch.h"
#include "WheelLibException.h"
#include <sstream>

namespace WheelDL {
	namespace Utils {

		WheelLibException::WheelLibException(ErrorCode code, const std::string& message)
			: _errorCode(code)
			, _message(message)
			, _innerException(nullptr)
		{
		}

		WheelLibException::WheelLibException(ErrorCode code, const std::string& message,
			std::exception_ptr innerException)
			: _errorCode(code)
			, _message(message)
			, _innerException(innerException)
		{
		}

		const char* WheelLibException::what() const noexcept {
			return _message.c_str();
		}

		std::string WheelLibException::getErrorCodeString() const noexcept {
			// Use ostringstream to avoid intermediate string copies
			std::ostringstream oss;
			oss << errorCodeToString(_errorCode) << " (" << static_cast<int>(_errorCode) << ")";
			return oss.str();
		}

	} // namespace Utils
} // namespace WheelDL
