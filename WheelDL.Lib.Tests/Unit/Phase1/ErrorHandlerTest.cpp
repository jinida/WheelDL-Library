#include "pch.h"
#include "WheelDL.Lib/Utils/Error/ErrorHandler.h"
#include "WheelDL.Lib/Utils/Error/WheelLibException.h"
#include "WheelDL.Lib/Utils/Error/ErrorCodes.h"
#include <cstring>

using namespace WheelDL::Utils;

/**
 * @brief ErrorHandler utility tests
 */
class ErrorHandlerTest : public ::testing::Test {
protected:
	void SetUp() override {
		// Clear error buffer
		std::memset(errorBuffer, 0, sizeof(errorBuffer));
	}

	static constexpr int BUFFER_SIZE = 256;
	char errorBuffer[BUFFER_SIZE];
};

TEST_F(ErrorHandlerTest, HandleWheelLibException) {
	// Create a WheelLibException
	WheelLibException ex(ErrorCode::INVALID_CONFIG, "Test configuration error");

	// Handle the exception
	int result = ErrorHandler::handleException(ex, errorBuffer, BUFFER_SIZE);

	// Should return the error code
	EXPECT_EQ(result, static_cast<int>(ErrorCode::INVALID_CONFIG));

	// Buffer should contain error message
	std::string errorMsg(errorBuffer);
	EXPECT_FALSE(errorMsg.empty());
	EXPECT_NE(errorMsg.find("INVALID_CONFIG"), std::string::npos);
	EXPECT_NE(errorMsg.find("Test configuration error"), std::string::npos);
}

TEST_F(ErrorHandlerTest, HandleStandardException) {
	// Create a standard exception
	std::runtime_error ex("Standard error message");

	// Handle the exception
	int result = ErrorHandler::handleException(ex, errorBuffer, BUFFER_SIZE);

	// Should return UNKNOWN_ERROR
	EXPECT_EQ(result, static_cast<int>(ErrorCode::UNKNOWN_ERROR));

	// Buffer should contain error message
	std::string errorMsg(errorBuffer);
	EXPECT_FALSE(errorMsg.empty());
	EXPECT_NE(errorMsg.find("UNKNOWN_ERROR"), std::string::npos);
	EXPECT_NE(errorMsg.find("Standard error message"), std::string::npos);
}

TEST_F(ErrorHandlerTest, HandleUnknownException) {
	// Handle unknown exception
	int result = ErrorHandler::handleUnknownException(errorBuffer, BUFFER_SIZE);

	// Should return UNKNOWN_ERROR
	EXPECT_EQ(result, static_cast<int>(ErrorCode::UNKNOWN_ERROR));

	// Buffer should contain error message
	std::string errorMsg(errorBuffer);
	EXPECT_FALSE(errorMsg.empty());
	EXPECT_NE(errorMsg.find("UNKNOWN_ERROR"), std::string::npos);
	EXPECT_NE(errorMsg.find("unknown exception"), std::string::npos);
}

TEST_F(ErrorHandlerTest, GetErrorMessageForAllCodes) {
	// Test that all error codes have messages
	std::vector<ErrorCode> codes = {
		ErrorCode::SUCCESS,
		ErrorCode::INVALID_CONFIG,
		ErrorCode::MODEL_LOAD_FAILED,
		ErrorCode::DATA_LOAD_FAILED,
		ErrorCode::GPU_OUT_OF_MEMORY,
		ErrorCode::TRAINING_FAILED,
		ErrorCode::DISTRIBUTED_INIT_FAILED,
		ErrorCode::UNKNOWN_ERROR,
		ErrorCode::NOT_IMPLEMENTED
	};

	for (ErrorCode code : codes) {
		std::string msg = ErrorHandler::getErrorMessage(code);
		EXPECT_FALSE(msg.empty());
		EXPECT_GT(msg.length(), 0);
	}
}

TEST_F(ErrorHandlerTest, SafeCopyToBufferNormal) {
	std::string message = "Test error message";

	ErrorHandler::safeCopyToBuffer(message, errorBuffer, BUFFER_SIZE);

	// Should match the message
	EXPECT_STREQ(errorBuffer, message.c_str());

	// Should be null-terminated
	EXPECT_EQ(errorBuffer[message.length()], '\0');
}

TEST_F(ErrorHandlerTest, SafeCopyToBufferTruncation) {
	// Create a long message
	std::string longMessage(500, 'X');

	// Small buffer
	char smallBuffer[50];
	ErrorHandler::safeCopyToBuffer(longMessage, smallBuffer, 50);

	// Should be truncated to 49 characters + null terminator
	EXPECT_EQ(std::strlen(smallBuffer), 49);
	EXPECT_EQ(smallBuffer[49], '\0');

	// Should be all 'X's
	for (int i = 0; i < 49; ++i) {
		EXPECT_EQ(smallBuffer[i], 'X');
	}
}

TEST_F(ErrorHandlerTest, SafeCopyToBufferNullBuffer) {
	std::string message = "Test";

	// Should not crash with null buffer
	EXPECT_NO_THROW(ErrorHandler::safeCopyToBuffer(message, nullptr, 100));
}

TEST_F(ErrorHandlerTest, SafeCopyToBufferZeroSize) {
	std::string message = "Test";

	// Should not crash with zero size
	EXPECT_NO_THROW(ErrorHandler::safeCopyToBuffer(message, errorBuffer, 0));
}

TEST_F(ErrorHandlerTest, SafeCopyToBufferEmptyMessage) {
	std::string message = "";

	ErrorHandler::safeCopyToBuffer(message, errorBuffer, BUFFER_SIZE);

	// Should be empty and null-terminated
	EXPECT_STREQ(errorBuffer, "");
	EXPECT_EQ(errorBuffer[0], '\0');
}

TEST_F(ErrorHandlerTest, FormatErrorMessage) {
	std::string formatted = ErrorHandler::formatErrorMessage(
		ErrorCode::MODEL_LOAD_FAILED,
		"File not found: model.pt"
	);

	// Should contain error code name
	EXPECT_NE(formatted.find("MODEL_LOAD_FAILED"), std::string::npos);

	// Should contain error description
	EXPECT_NE(formatted.find("Failed to load model"), std::string::npos);

	// Should contain custom message
	EXPECT_NE(formatted.find("File not found: model.pt"), std::string::npos);
}

TEST_F(ErrorHandlerTest, FormatErrorMessageEmptyContext) {
	std::string formatted = ErrorHandler::formatErrorMessage(
		ErrorCode::GPU_OUT_OF_MEMORY,
		""
	);

	// Should contain error code and description
	EXPECT_NE(formatted.find("GPU_OUT_OF_MEMORY"), std::string::npos);
	EXPECT_NE(formatted.find("GPU out of memory"), std::string::npos);
}

TEST_F(ErrorHandlerTest, HandleConfigurationException) {
	ConfigurationException ex(ErrorCode::CONFIG_PARSE_FAILED, "Invalid YAML syntax");

	int result = ErrorHandler::handleException(ex, errorBuffer, BUFFER_SIZE);

	EXPECT_EQ(result, static_cast<int>(ErrorCode::CONFIG_PARSE_FAILED));

	std::string errorMsg(errorBuffer);
	EXPECT_NE(errorMsg.find("CONFIG_PARSE_FAILED"), std::string::npos);
	EXPECT_NE(errorMsg.find("Invalid YAML syntax"), std::string::npos);
}

TEST_F(ErrorHandlerTest, HandleModelException) {
	ModelException ex(ErrorCode::MODEL_BUILD_FAILED, "Invalid layer configuration");

	int result = ErrorHandler::handleException(ex, errorBuffer, BUFFER_SIZE);

	EXPECT_EQ(result, static_cast<int>(ErrorCode::MODEL_BUILD_FAILED));

	std::string errorMsg(errorBuffer);
	EXPECT_NE(errorMsg.find("MODEL_BUILD_FAILED"), std::string::npos);
	EXPECT_NE(errorMsg.find("Invalid layer configuration"), std::string::npos);
}

TEST_F(ErrorHandlerTest, HandleDataException) {
	DataException ex(ErrorCode::DATA_LOAD_FAILED, "Image file corrupted");

	int result = ErrorHandler::handleException(ex, errorBuffer, BUFFER_SIZE);

	EXPECT_EQ(result, static_cast<int>(ErrorCode::DATA_LOAD_FAILED));

	std::string errorMsg(errorBuffer);
	EXPECT_NE(errorMsg.find("DATA_LOAD_FAILED"), std::string::npos);
	EXPECT_NE(errorMsg.find("Image file corrupted"), std::string::npos);
}

TEST_F(ErrorHandlerTest, HandleGPUException) {
	GPUException ex(ErrorCode::GPU_CUDA_ERROR, "CUDA kernel launch failed");

	int result = ErrorHandler::handleException(ex, errorBuffer, BUFFER_SIZE);

	EXPECT_EQ(result, static_cast<int>(ErrorCode::GPU_CUDA_ERROR));

	std::string errorMsg(errorBuffer);
	EXPECT_NE(errorMsg.find("GPU_CUDA_ERROR"), std::string::npos);
	EXPECT_NE(errorMsg.find("CUDA kernel launch failed"), std::string::npos);
}

TEST_F(ErrorHandlerTest, HandleTrainingException) {
	TrainingException ex(ErrorCode::LOSS_IS_NAN, "Loss became NaN at epoch 5");

	int result = ErrorHandler::handleException(ex, errorBuffer, BUFFER_SIZE);

	EXPECT_EQ(result, static_cast<int>(ErrorCode::LOSS_IS_NAN));

	std::string errorMsg(errorBuffer);
	EXPECT_NE(errorMsg.find("LOSS_IS_NAN"), std::string::npos);
	EXPECT_NE(errorMsg.find("Loss became NaN at epoch 5"), std::string::npos);
}

TEST_F(ErrorHandlerTest, HandleDistributedException) {
	DistributedException ex(ErrorCode::DISTRIBUTED_SYNC_FAILED, "Rank 2 timeout");

	int result = ErrorHandler::handleException(ex, errorBuffer, BUFFER_SIZE);

	EXPECT_EQ(result, static_cast<int>(ErrorCode::DISTRIBUTED_SYNC_FAILED));

	std::string errorMsg(errorBuffer);
	EXPECT_NE(errorMsg.find("DISTRIBUTED_SYNC_FAILED"), std::string::npos);
	EXPECT_NE(errorMsg.find("Rank 2 timeout"), std::string::npos);
}
