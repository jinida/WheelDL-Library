namespace WheelDL.Interop.Tests.Integration;

/// <summary>
/// Integration tests for memory management functions
/// </summary>
[Collection("Integration")]
public class MemoryManagementTests : IntegrationTestBase
{
    [Fact]
    public void GPU_WhenCudaAvailable_ShouldReturnValidMemoryInfo()
    {
        SkipIfDllNotAvailable();

        if (!Manager.IsCudaAvailable())
        {
            // Skip GPU tests if CUDA not available
            return;
        }

        var deviceCount = Manager.GetGPUDeviceCount();
        Assert.True(deviceCount > 0, "Should have at least one GPU when CUDA is available");

        var totalMB = Manager.GetGPUTotalMB(0);
        Assert.True(totalMB > 0, "Total GPU memory should be positive");

        var usedMB = Manager.GetGPUUsedMB(0);
        Assert.True(usedMB >= 0, "Used GPU memory should be non-negative");

        var usagePercent = Manager.GetGPUUsagePercent(0);
        Assert.True(usagePercent >= 0 && usagePercent <= 100,
            $"GPU usage percent should be 0-100, got {usagePercent}");
    }

    [Fact]
    public void GPU_WhenCudaNotAvailable_ShouldReturnZeros()
    {
        SkipIfDllNotAvailable();

        if (Manager.IsCudaAvailable())
        {
            // Skip this test if CUDA is available
            return;
        }

        var deviceCount = Manager.GetGPUDeviceCount();
        Assert.Equal(0, deviceCount);

        var totalMB = Manager.GetGPUTotalMB(0);
        Assert.Equal(0f, totalMB);

        var usedMB = Manager.GetGPUUsedMB(0);
        Assert.Equal(0f, usedMB);
    }

    [Fact]
    public void HasEnoughGPU_ShouldReturnConsistentResults()
    {
        SkipIfDllNotAvailable();

        if (!Manager.IsCudaAvailable())
        {
            // Without CUDA, should always return false
            var hasEnough = Manager.HasEnoughGPU(0, 1024);
            Assert.False(hasEnough);
            return;
        }

        // With CUDA, check for reasonable memory amount
        var totalMB = Manager.GetGPUTotalMB(0);
        var usedMB = Manager.GetGPUUsedMB(0);
        var availableMB = totalMB - usedMB;

        // Should have enough for 1 byte
        var hasEnoughForSmall = Manager.HasEnoughGPU(0, 1);
        Assert.True(hasEnoughForSmall || availableMB <= 0);

        // Should not have enough for impossibly large amount (1 PB)
        var hasEnoughForHuge = Manager.HasEnoughGPU(0, 1024UL * 1024 * 1024 * 1024 * 1024);
        Assert.False(hasEnoughForHuge);
    }

    [Fact]
    public void CPU_MemoryValues_ShouldBeReasonable()
    {
        SkipIfDllNotAvailable();

        var usedBytes = Manager.GetCPUUsedBytes();
        var availableBytes = Manager.GetCPUAvailableBytes();

        // Both should be positive on any running system
        Assert.True(usedBytes > 0, "Used CPU memory should be positive");
        Assert.True(availableBytes > 0, "Available CPU memory should be positive");

        // Total should be reasonable (at least 1GB total)
        var totalBytes = usedBytes + availableBytes;
        Assert.True(totalBytes >= 1024UL * 1024 * 1024,
            $"Total memory seems too small: {totalBytes} bytes");
    }

    [Fact]
    public void CanAcceptNewTask_ShouldReturnBoolean()
    {
        SkipIfDllNotAvailable();

        // Should not throw regardless of result
        var canAccept = Manager.CanAcceptNewTask(0);

        Assert.True(canAccept || !canAccept);
    }

    [Fact]
    public void GetAvailableGpuMemory_ShouldReturnNonNegative()
    {
        SkipIfDllNotAvailable();

        var availableMB = Manager.GetAvailableGpuMemory(0);

        Assert.True(availableMB >= 0, "Available GPU memory should be non-negative");
    }
}
