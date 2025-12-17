using System.Runtime.InteropServices;
using WheelDL.Interop.Native;

namespace WheelDL.Interop.Tests.Native;

/// <summary>
/// Tests for native type structures and marshalling
/// </summary>
public class NativeTypesTests
{
    [Fact]
    public void WheelError_ShouldHaveCorrectLayout()
    {
        // Verify struct size is reasonable for marshalling
        var size = Marshal.SizeOf<WheelError>();

        // Code (4) + Message (512) + Category (32) + TaskId (64) = 612 bytes
        Assert.True(size >= 612, $"WheelError size should be at least 612 bytes, got {size}");
    }

    [Fact]
    public void WheelMetricsData_ShouldHaveCorrectLayout()
    {
        var size = Marshal.SizeOf<WheelMetricsData>();

        // 9 floats * 4 bytes = 36 bytes
        Assert.Equal(36, size);
    }

    [Fact]
    public void WheelProfilingData_ShouldHaveCorrectLayout()
    {
        var size = Marshal.SizeOf<WheelProfilingData>();

        // 6 doubles (48) + 2 ulongs (16) = 64 bytes
        Assert.Equal(64, size);
    }

    [Fact]
    public void WheelProgressData_ShouldHaveCorrectLayout()
    {
        var size = Marshal.SizeOf<WheelProgressData>();

        // Should be large enough for all fields
        Assert.True(size > 0, "WheelProgressData should have positive size");
    }

    [Fact]
    public void WheelTaskResult_ShouldHaveCorrectLayout()
    {
        var size = Marshal.SizeOf<WheelTaskResult>();

        // Should be large enough for all fields
        Assert.True(size > 0, "WheelTaskResult should have positive size");
    }

    [Theory]
    [InlineData(WheelResult.Ok, 0)]
    [InlineData(WheelResult.Error, 1)]
    [InlineData(WheelResult.Timeout, 2)]
    [InlineData(WheelResult.Stopped, 3)]
    public void WheelResult_ShouldHaveCorrectValues(WheelResult result, int expected)
    {
        Assert.Equal(expected, (int)result);
    }

    [Theory]
    [InlineData(WheelTaskType.Classification, 0)]
    [InlineData(WheelTaskType.Detection, 1)]
    [InlineData(WheelTaskType.Segmentation, 2)]
    [InlineData(WheelTaskType.Anomaly, 3)]
    [InlineData(WheelTaskType.Obb, 4)]
    [InlineData(WheelTaskType.Unknown, 99)]
    public void WheelTaskType_ShouldHaveCorrectValues(WheelTaskType type, int expected)
    {
        Assert.Equal(expected, (int)type);
    }

    [Theory]
    [InlineData(WheelOperationType.Train, 0)]
    [InlineData(WheelOperationType.Validate, 1)]
    [InlineData(WheelOperationType.Predict, 2)]
    public void WheelOperationType_ShouldHaveCorrectValues(WheelOperationType type, int expected)
    {
        Assert.Equal(expected, (int)type);
    }

    [Theory]
    [InlineData(WheelTrainingState.Idle, 0)]
    [InlineData(WheelTrainingState.Initializing, 1)]
    [InlineData(WheelTrainingState.Training, 2)]
    [InlineData(WheelTrainingState.Validating, 3)]
    [InlineData(WheelTrainingState.Paused, 4)]
    [InlineData(WheelTrainingState.Completed, 5)]
    [InlineData(WheelTrainingState.Failed, 6)]
    [InlineData(WheelTrainingState.Stopped, 7)]
    public void WheelTrainingState_ShouldHaveCorrectValues(WheelTrainingState state, int expected)
    {
        Assert.Equal(expected, (int)state);
    }

    [Theory]
    [InlineData(WheelProgressStage.TrainBatch, 0)]
    [InlineData(WheelProgressStage.TrainEpoch, 1)]
    [InlineData(WheelProgressStage.ValBatch, 2)]
    [InlineData(WheelProgressStage.ValEpoch, 3)]
    [InlineData(WheelProgressStage.CheckpointSaved, 4)]
    public void WheelProgressStage_ShouldHaveCorrectValues(WheelProgressStage stage, int expected)
    {
        Assert.Equal(expected, (int)stage);
    }
}
