using WheelDL.Interop.Native;

namespace WheelDL.Interop;

/// <summary>
/// Managed progress information
/// </summary>
public sealed class ProgressInfo
{
    public ProgressStage Stage { get; init; }
    public int CurrentEpoch { get; init; }
    public int TotalEpochs { get; init; }
    public int CurrentBatch { get; init; }
    public int TotalBatches { get; init; }
    public float Loss { get; init; }
    public float LearningRate { get; init; }
    public float GpuMemoryUsage { get; init; }
    public TimeSpan ElapsedTime { get; init; }
    public TimeSpan Eta { get; init; }
    public MetricsInfo Metrics { get; init; } = null!;
    public string Message { get; init; } = string.Empty;

    /// <summary>
    /// Overall progress percentage (0-100)
    /// </summary>
    public float ProgressPercent
    {
        get
        {
            if (TotalEpochs <= 0) return 0;
            float epochProgress = (float)CurrentEpoch / TotalEpochs;
            if (TotalBatches > 0 && CurrentEpoch < TotalEpochs)
            {
                float batchProgress = (float)CurrentBatch / TotalBatches / TotalEpochs;
                return (epochProgress + batchProgress) * 100f;
            }
            return epochProgress * 100f;
        }
    }

    internal static ProgressInfo FromNative(WheelProgressData native) => new()
    {
        Stage = (ProgressStage)native.Stage,
        CurrentEpoch = native.CurrentEpoch,
        TotalEpochs = native.TotalEpochs,
        CurrentBatch = native.CurrentBatch,
        TotalBatches = native.TotalBatches,
        Loss = native.Loss,
        LearningRate = native.LearningRate,
        GpuMemoryUsage = native.GpuMemoryUsage,
        ElapsedTime = TimeSpan.FromSeconds(native.ElapsedTime),
        Eta = TimeSpan.FromSeconds(native.Eta),
        Metrics = MetricsInfo.FromNative(native.Metrics),
        Message = native.Message ?? string.Empty
    };

    public override string ToString() =>
        $"Epoch {CurrentEpoch}/{TotalEpochs}, Batch {CurrentBatch}/{TotalBatches}, Loss={Loss:F4}, {ProgressPercent:F1}%";
}
