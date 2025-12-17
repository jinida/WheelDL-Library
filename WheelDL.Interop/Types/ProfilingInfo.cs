using WheelDL.Interop.Native;

namespace WheelDL.Interop;

/// <summary>
/// Managed profiling information
/// </summary>
public sealed class ProfilingInfo
{
    public TimeSpan TotalTime { get; init; }
    public TimeSpan GpuTime { get; init; }
    public TimeSpan DataLoadTime { get; init; }
    public TimeSpan PreprocessTime { get; init; }
    public TimeSpan InferenceTime { get; init; }
    public TimeSpan PostprocessTime { get; init; }
    public ulong PeakGpuMemoryMB { get; init; }
    public ulong PeakCpuMemoryMB { get; init; }

    internal static ProfilingInfo FromNative(WheelProfilingData native) => new()
    {
        TotalTime = TimeSpan.FromMilliseconds(native.TotalTimeMs),
        GpuTime = TimeSpan.FromMilliseconds(native.GpuTimeMs),
        DataLoadTime = TimeSpan.FromMilliseconds(native.DataLoadTimeMs),
        PreprocessTime = TimeSpan.FromMilliseconds(native.PreprocessTimeMs),
        InferenceTime = TimeSpan.FromMilliseconds(native.InferenceTimeMs),
        PostprocessTime = TimeSpan.FromMilliseconds(native.PostprocessTimeMs),
        PeakGpuMemoryMB = native.PeakGpuMemoryMB,
        PeakCpuMemoryMB = native.PeakCpuMemoryMB
    };

    public override string ToString() =>
        $"Total={TotalTime.TotalSeconds:F2}s, GPU={GpuTime.TotalSeconds:F2}s, " +
        $"PeakGPU={PeakGpuMemoryMB}MB, PeakCPU={PeakCpuMemoryMB}MB";
}
