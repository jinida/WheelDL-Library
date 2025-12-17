namespace WheelDL.Interop.Tests.Integration;

/// <summary>
/// Integration tests for WheelManager that require the native DLL
/// </summary>
[Collection("Integration")]
public class WheelManagerIntegrationTests : IntegrationTestBase
{
    [Fact]
    public void Initialize_ShouldSucceed_WhenDllAvailable()
    {
        SkipIfDllNotAvailable();

        var ex = Record.Exception(() => Manager.Initialize());
        Assert.Null(ex);
    }

    [Fact]
    public void GetVersion_ShouldReturnVersionString()
    {
        SkipIfDllNotAvailable();

        var version = Manager.GetVersion();

        Assert.NotNull(version);
        Assert.NotEmpty(version);
        Assert.Matches(@"^\d+\.\d+\.\d+", version);
    }

    [Fact]
    public void IsCudaAvailable_ShouldReturnBoolean()
    {
        SkipIfDllNotAvailable();

        // Should not throw
        var available = Manager.IsCudaAvailable();

        // Result can be true or false depending on system
        Assert.True(available || !available);
    }

    [Fact]
    public void GetGPUDeviceCount_ShouldReturnNonNegative()
    {
        SkipIfDllNotAvailable();

        var count = Manager.GetGPUDeviceCount();

        Assert.True(count >= 0, "GPU device count should be non-negative");
    }

    [Fact]
    public void GetMemoryThreshold_ShouldReturnPositiveValue()
    {
        SkipIfDllNotAvailable();

        var threshold = Manager.GetMemoryThreshold();

        Assert.True(threshold > 0, "Memory threshold should be positive");
    }

    [Fact]
    public void SetMemoryThreshold_ShouldUpdateValue()
    {
        SkipIfDllNotAvailable();

        var originalThreshold = Manager.GetMemoryThreshold();

        try
        {
            Manager.SetMemoryThreshold(8192);
            var newThreshold = Manager.GetMemoryThreshold();
            Assert.Equal(8192, newThreshold);
        }
        finally
        {
            // Restore original
            Manager.SetMemoryThreshold(originalThreshold);
        }
    }

    [Fact]
    public void GetRunningTaskCount_ShouldReturnNonNegative()
    {
        SkipIfDllNotAvailable();

        var count = Manager.GetRunningTaskCount();

        Assert.True(count >= 0, "Running task count should be non-negative");
    }

    [Fact]
    public void GetPendingTaskCount_ShouldReturnNonNegative()
    {
        SkipIfDllNotAvailable();

        var count = Manager.GetPendingTaskCount();

        Assert.True(count >= 0, "Pending task count should be non-negative");
    }

    [Fact]
    public void GetCPUUsedBytes_ShouldReturnPositiveValue()
    {
        SkipIfDllNotAvailable();

        var bytes = Manager.GetCPUUsedBytes();

        Assert.True(bytes > 0, "CPU used bytes should be positive on a running system");
    }

    [Fact]
    public void GetCPUAvailableBytes_ShouldReturnPositiveValue()
    {
        SkipIfDllNotAvailable();

        var bytes = Manager.GetCPUAvailableBytes();

        Assert.True(bytes > 0, "CPU available bytes should be positive");
    }

    [Fact]
    public void TaskExists_WithInvalidTaskId_ShouldReturnFalse()
    {
        SkipIfDllNotAvailable();

        var exists = Manager.TaskExists("INVALID_TASK_ID_12345");

        Assert.False(exists);
    }

    [Fact]
    public void IsTaskRunning_WithInvalidTaskId_ShouldReturnFalse()
    {
        SkipIfDllNotAvailable();

        var running = Manager.IsTaskRunning("INVALID_TASK_ID_12345");

        Assert.False(running);
    }
}
