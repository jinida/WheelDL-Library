using WheelDL.Interop.Native;

namespace WheelDL.Interop.Tests.Types;

/// <summary>
/// Tests for managed type conversions from native types
/// </summary>
public class TypeConversionTests
{
    [Fact]
    public void MetricsInfo_FromNative_ShouldConvertCorrectly()
    {
        var native = new WheelMetricsData
        {
            Loss = 0.5f,
            Accuracy = 0.95f,
            Precision = 0.92f,
            Recall = 0.88f,
            F1Score = 0.90f,
            MAP = 0.85f,
            Fitness = 0.87f,
            Threshold = 0.5f,
            AucROC = 0.93f
        };

        var managed = MetricsInfo.FromNative(native);

        Assert.Equal(0.5f, managed.Loss);
        Assert.Equal(0.95f, managed.Accuracy);
        Assert.Equal(0.92f, managed.Precision);
        Assert.Equal(0.88f, managed.Recall);
        Assert.Equal(0.90f, managed.F1Score);
        Assert.Equal(0.85f, managed.MAP);
        Assert.Equal(0.87f, managed.Fitness);
        Assert.Equal(0.5f, managed.Threshold);
        Assert.Equal(0.93f, managed.AucROC);
    }

    [Fact]
    public void ProgressInfo_FromNative_ShouldConvertCorrectly()
    {
        var native = new WheelProgressData
        {
            Stage = WheelProgressStage.TrainEpoch,
            CurrentEpoch = 5,
            TotalEpochs = 100,
            CurrentBatch = 32,
            TotalBatches = 64,
            Loss = 0.123f,
            LearningRate = 0.001f,
            GpuMemoryUsage = 4096.0f,
            ElapsedTime = 3600.0,
            Eta = 7200.0,
            Message = "Training in progress"
        };

        var managed = ProgressInfo.FromNative(native);

        Assert.Equal(ProgressStage.TrainEpoch, managed.Stage);
        Assert.Equal(5, managed.CurrentEpoch);
        Assert.Equal(100, managed.TotalEpochs);
        Assert.Equal(32, managed.CurrentBatch);
        Assert.Equal(64, managed.TotalBatches);
        Assert.Equal(0.123f, managed.Loss);
        Assert.Equal(0.001f, managed.LearningRate);
        Assert.Equal(4096.0f, managed.GpuMemoryUsage);
        Assert.Equal(TimeSpan.FromSeconds(3600), managed.ElapsedTime);
        Assert.Equal(TimeSpan.FromSeconds(7200), managed.Eta);
        Assert.Equal("Training in progress", managed.Message);
    }

    [Fact]
    public void ProgressInfo_ProgressPercent_ShouldCalculateCorrectly()
    {
        var native = new WheelProgressData
        {
            CurrentEpoch = 50,
            TotalEpochs = 100,
            CurrentBatch = 0,
            TotalBatches = 0
        };

        var managed = ProgressInfo.FromNative(native);

        Assert.Equal(50.0f, managed.ProgressPercent);
    }

    [Fact]
    public void ProgressInfo_ProgressPercent_WithBatches_ShouldCalculateCorrectly()
    {
        var native = new WheelProgressData
        {
            CurrentEpoch = 0,
            TotalEpochs = 10,
            CurrentBatch = 50,
            TotalBatches = 100
        };

        var managed = ProgressInfo.FromNative(native);

        // Should include batch progress: 0/10 + (50/100)/10 = 0.05 = 5%
        Assert.Equal(5.0f, managed.ProgressPercent, 0.1f);
    }

    [Fact]
    public void ProfilingInfo_FromNative_ShouldConvertCorrectly()
    {
        var native = new WheelProfilingData
        {
            TotalTimeMs = 60000.0,
            GpuTimeMs = 45000.0,
            DataLoadTimeMs = 5000.0,
            PreprocessTimeMs = 3000.0,
            InferenceTimeMs = 40000.0,
            PostprocessTimeMs = 2000.0,
            PeakGpuMemoryMB = 8192,
            PeakCpuMemoryMB = 16384
        };

        var managed = ProfilingInfo.FromNative(native);

        Assert.Equal(TimeSpan.FromMilliseconds(60000), managed.TotalTime);
        Assert.Equal(TimeSpan.FromMilliseconds(45000), managed.GpuTime);
        Assert.Equal(TimeSpan.FromMilliseconds(5000), managed.DataLoadTime);
        Assert.Equal(TimeSpan.FromMilliseconds(3000), managed.PreprocessTime);
        Assert.Equal(TimeSpan.FromMilliseconds(40000), managed.InferenceTime);
        Assert.Equal(TimeSpan.FromMilliseconds(2000), managed.PostprocessTime);
        Assert.Equal(8192UL, managed.PeakGpuMemoryMB);
        Assert.Equal(16384UL, managed.PeakCpuMemoryMB);
    }

    [Fact]
    public void TaskResultInfo_FromNative_ShouldConvertCorrectly()
    {
        var native = new WheelTaskResult
        {
            TaskId = "TRAIN_20251216_120000_001_0",
            FinalState = WheelTrainingState.Completed,
            OperationType = WheelOperationType.Train,
            HasMetrics = 1,
            Metrics = new WheelMetricsData { Loss = 0.1f, Accuracy = 0.99f },
            OutputPath = "/output/model",
            CheckpointPath = "/checkpoint/best.pt",
            TotalTimeMs = 3600000.0,
            ErrorMessage = "",
            ErrorCode = 0
        };

        var managed = TaskResultInfo.FromNative(native);

        Assert.Equal("TRAIN_20251216_120000_001_0", managed.TaskId);
        Assert.Equal(TrainingState.Completed, managed.FinalState);
        Assert.Equal(OperationType.Train, managed.OperationType);
        Assert.NotNull(managed.Metrics);
        Assert.Equal(0.1f, managed.Metrics!.Loss);
        Assert.Equal("/output/model", managed.OutputPath);
        Assert.Equal("/checkpoint/best.pt", managed.CheckpointPath);
        Assert.Equal(TimeSpan.FromHours(1), managed.TotalTime);
        Assert.True(managed.IsSuccess);
        Assert.False(managed.WasStopped);
        Assert.False(managed.IsFailed);
    }

    [Fact]
    public void TaskResultInfo_IsCompleted_ShouldReturnCorrectly()
    {
        var completedStates = new[]
        {
            WheelTrainingState.Completed,
            WheelTrainingState.Failed,
            WheelTrainingState.Stopped
        };

        foreach (var state in completedStates)
        {
            var native = new WheelTaskResult { FinalState = state };
            var managed = TaskResultInfo.FromNative(native);
            Assert.True(managed.IsCompleted, $"State {state} should be considered completed");
        }

        var notCompletedStates = new[]
        {
            WheelTrainingState.Idle,
            WheelTrainingState.Initializing,
            WheelTrainingState.Training,
            WheelTrainingState.Validating,
            WheelTrainingState.Paused
        };

        foreach (var state in notCompletedStates)
        {
            var native = new WheelTaskResult { FinalState = state };
            var managed = TaskResultInfo.FromNative(native);
            Assert.False(managed.IsCompleted, $"State {state} should not be considered completed");
        }
    }
}
