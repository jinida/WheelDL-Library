using WheelDL.Interop.Native;

namespace WheelDL.Interop;

/// <summary>
/// Managed task result information
/// </summary>
public sealed class TaskResultInfo
{
    public string TaskId { get; init; } = string.Empty;
    public TrainingState FinalState { get; init; }
    public OperationType OperationType { get; init; }
    public MetricsInfo? Metrics { get; init; }
    public string OutputPath { get; init; } = string.Empty;
    public string CheckpointPath { get; init; } = string.Empty;
    public TimeSpan TotalTime { get; init; }
    public ProfilingInfo Profiling { get; init; } = null!;
    public string ErrorMessage { get; init; } = string.Empty;
    public int ErrorCode { get; init; }

    /// <summary>
    /// Whether the task completed successfully
    /// </summary>
    public bool IsSuccess => FinalState == TrainingState.Completed;

    /// <summary>
    /// Whether the task was stopped by user
    /// </summary>
    public bool WasStopped => FinalState == TrainingState.Stopped;

    /// <summary>
    /// Whether the task failed
    /// </summary>
    public bool IsFailed => FinalState == TrainingState.Failed;

    /// <summary>
    /// Whether the task has completed (success, failed, or stopped)
    /// </summary>
    public bool IsCompleted => FinalState is TrainingState.Completed
        or TrainingState.Failed or TrainingState.Stopped;

    internal static TaskResultInfo FromNative(WheelTaskResult native) => new()
    {
        TaskId = native.TaskId ?? string.Empty,
        FinalState = (TrainingState)native.FinalState,
        OperationType = (OperationType)native.OperationType,
        Metrics = native.HasMetrics != 0 ? MetricsInfo.FromNative(native.Metrics) : null,
        OutputPath = native.OutputPath ?? string.Empty,
        CheckpointPath = native.CheckpointPath ?? string.Empty,
        TotalTime = TimeSpan.FromMilliseconds(native.TotalTimeMs),
        Profiling = ProfilingInfo.FromNative(native.Profiling),
        ErrorMessage = native.ErrorMessage ?? string.Empty,
        ErrorCode = native.ErrorCode
    };

    public override string ToString() =>
        $"Task {TaskId}: {FinalState}, Time={TotalTime.TotalSeconds:F2}s" +
        (Metrics != null ? $", {Metrics}" : "") +
        (!string.IsNullOrEmpty(ErrorMessage) ? $", Error: {ErrorMessage}" : "");
}
