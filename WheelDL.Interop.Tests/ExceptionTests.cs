using WheelDL.Interop.Native;

namespace WheelDL.Interop.Tests;

/// <summary>
/// Tests for exception mapping from native errors
/// </summary>
public class ExceptionTests
{
    [Theory]
    [InlineData("CONFIG", typeof(ConfigurationException))]
    [InlineData("MODEL", typeof(ModelException))]
    [InlineData("DATA", typeof(DataException))]
    [InlineData("GPU", typeof(GpuException))]
    [InlineData("TRAINING", typeof(TrainingException))]
    [InlineData("DISTRIBUTED", typeof(DistributedException))]
    [InlineData("TASK", typeof(TaskException))]
    [InlineData("UNKNOWN", typeof(WheelException))]
    [InlineData("OTHER", typeof(WheelException))]
    public void FromNative_ShouldMapCategoryToCorrectExceptionType(string category, Type expectedType)
    {
        var err = new WheelError
        {
            Code = 1000,
            Message = "Test error message",
            Category = category,
            TaskId = "TEST_TASK_001"
        };

        var exception = WheelException.FromNative(err);

        Assert.IsType(expectedType, exception);
        Assert.Equal(1000, exception.ErrorCode);
        Assert.Equal("Test error message", exception.Message);
        Assert.Equal(category, exception.Category);
    }

    [Fact]
    public void TaskException_ShouldIncludeTaskId()
    {
        var err = new WheelError
        {
            Code = 7000,
            Message = "Task not found",
            Category = "TASK",
            TaskId = "TRAIN_20251216_001"
        };

        var exception = WheelException.FromNative(err);

        Assert.IsType<TaskException>(exception);
        var taskException = (TaskException)exception;
        Assert.Equal("TRAIN_20251216_001", taskException.RelatedTaskId);
        Assert.Equal("TRAIN_20251216_001", taskException.TaskId);
    }

    [Fact]
    public void ThrowIfError_ShouldNotThrowOnOk()
    {
        var err = new WheelError();

        var ex = Record.Exception(() => WheelException.ThrowIfError(WheelResult.Ok, ref err));

        Assert.Null(ex);
    }

    [Fact]
    public void ThrowIfError_ShouldThrowOnError()
    {
        var err = new WheelError
        {
            Code = 1000,
            Message = "Configuration error",
            Category = "CONFIG"
        };

        Assert.Throws<ConfigurationException>(() =>
            WheelException.ThrowIfError(WheelResult.Error, ref err));
    }

    [Fact]
    public void TaskTimeoutException_ShouldHaveCorrectErrorCode()
    {
        var exception = new TaskTimeoutException("TASK_001", "Timeout after 30s");

        Assert.Equal(7004, exception.ErrorCode);
        Assert.Equal("TASK", exception.Category);
        Assert.Equal("TASK_001", exception.TaskId);
        Assert.Contains("Timeout", exception.Message);
    }

    [Fact]
    public void TaskStoppedException_ShouldHaveCorrectErrorCode()
    {
        var exception = new TaskStoppedException("TASK_001");

        Assert.Equal(7003, exception.ErrorCode);
        Assert.Equal("TASK", exception.Category);
        Assert.Equal("TASK_001", exception.TaskId);
        Assert.Contains("stopped", exception.Message.ToLower());
    }

    [Fact]
    public void WheelException_DefaultConstructor_ShouldSetDefaults()
    {
        var exception = new WheelException("Test message");

        Assert.Equal("Test message", exception.Message);
        Assert.Equal(0, exception.ErrorCode);
        Assert.Equal("UNKNOWN", exception.Category);
        Assert.Null(exception.TaskId);
    }

    [Fact]
    public void WheelException_WithInnerException_ShouldPreserveIt()
    {
        var inner = new InvalidOperationException("Inner error");
        var exception = new WheelException("Outer error", inner);

        Assert.Equal("Outer error", exception.Message);
        Assert.Same(inner, exception.InnerException);
    }

    [Theory]
    [InlineData(1000, "INVALID_CONFIG")]
    [InlineData(2000, "MODEL_LOAD_FAILED")]
    [InlineData(3000, "DATA_LOAD_FAILED")]
    [InlineData(4000, "GPU_OUT_OF_MEMORY")]
    [InlineData(5000, "TRAINING_FAILED")]
    [InlineData(6000, "DISTRIBUTED_INIT_FAILED")]
    [InlineData(7000, "TASK_NOT_FOUND")]
    [InlineData(9000, "UNKNOWN_ERROR")]
    public void ErrorCodes_ShouldBePreserved(int errorCode, string expectedDescription)
    {
        var err = new WheelError
        {
            Code = errorCode,
            Message = expectedDescription,
            Category = "TEST"
        };

        var exception = WheelException.FromNative(err);

        Assert.Equal(errorCode, exception.ErrorCode);
    }
}
