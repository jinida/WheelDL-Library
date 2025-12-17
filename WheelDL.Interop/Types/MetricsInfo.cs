using WheelDL.Interop.Native;

namespace WheelDL.Interop;

/// <summary>
/// Managed metrics information
/// </summary>
public sealed class MetricsInfo
{
    public float Loss { get; init; }
    public float Accuracy { get; init; }
    public float Precision { get; init; }
    public float Recall { get; init; }
    public float F1Score { get; init; }
    public float MAP { get; init; }
    public float Fitness { get; init; }
    public float Threshold { get; init; }
    public float AucROC { get; init; }

    internal static MetricsInfo FromNative(WheelMetricsData native) => new()
    {
        Loss = native.Loss,
        Accuracy = native.Accuracy,
        Precision = native.Precision,
        Recall = native.Recall,
        F1Score = native.F1Score,
        MAP = native.MAP,
        Fitness = native.Fitness,
        Threshold = native.Threshold,
        AucROC = native.AucROC
    };

    public override string ToString() =>
        $"Loss={Loss:F4}, Accuracy={Accuracy:P2}, mAP={MAP:F4}, F1={F1Score:F4}";
}
