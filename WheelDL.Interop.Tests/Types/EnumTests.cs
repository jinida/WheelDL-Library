namespace WheelDL.Interop.Tests.Types;

/// <summary>
/// Tests for managed enum values matching native enum values
/// </summary>
public class EnumTests
{
    [Theory]
    [InlineData(TaskType.Classification, 0)]
    [InlineData(TaskType.Detection, 1)]
    [InlineData(TaskType.Segmentation, 2)]
    [InlineData(TaskType.Anomaly, 3)]
    [InlineData(TaskType.Obb, 4)]
    [InlineData(TaskType.Unknown, 99)]
    public void TaskType_ShouldMatchNativeValues(TaskType type, int expected)
    {
        Assert.Equal(expected, (int)type);
    }

    [Theory]
    [InlineData(OperationType.Train, 0)]
    [InlineData(OperationType.Validate, 1)]
    [InlineData(OperationType.Predict, 2)]
    public void OperationType_ShouldMatchNativeValues(OperationType type, int expected)
    {
        Assert.Equal(expected, (int)type);
    }

    [Theory]
    [InlineData(TrainingState.Idle, 0)]
    [InlineData(TrainingState.Initializing, 1)]
    [InlineData(TrainingState.Training, 2)]
    [InlineData(TrainingState.Validating, 3)]
    [InlineData(TrainingState.Paused, 4)]
    [InlineData(TrainingState.Completed, 5)]
    [InlineData(TrainingState.Failed, 6)]
    [InlineData(TrainingState.Stopped, 7)]
    public void TrainingState_ShouldMatchNativeValues(TrainingState state, int expected)
    {
        Assert.Equal(expected, (int)state);
    }

    [Theory]
    [InlineData(ProgressStage.TrainBatch, 0)]
    [InlineData(ProgressStage.TrainEpoch, 1)]
    [InlineData(ProgressStage.ValBatch, 2)]
    [InlineData(ProgressStage.ValEpoch, 3)]
    [InlineData(ProgressStage.CheckpointSaved, 4)]
    public void ProgressStage_ShouldMatchNativeValues(ProgressStage stage, int expected)
    {
        Assert.Equal(expected, (int)stage);
    }

    [Fact]
    public void TaskType_ShouldHaveAllExpectedValues()
    {
        var values = Enum.GetValues<TaskType>();
        Assert.Equal(6, values.Length);
    }

    [Fact]
    public void OperationType_ShouldHaveAllExpectedValues()
    {
        var values = Enum.GetValues<OperationType>();
        Assert.Equal(3, values.Length);
    }

    [Fact]
    public void TrainingState_ShouldHaveAllExpectedValues()
    {
        var values = Enum.GetValues<TrainingState>();
        Assert.Equal(8, values.Length);
    }

    [Fact]
    public void ProgressStage_ShouldHaveAllExpectedValues()
    {
        var values = Enum.GetValues<ProgressStage>();
        Assert.Equal(5, values.Length);
    }
}
