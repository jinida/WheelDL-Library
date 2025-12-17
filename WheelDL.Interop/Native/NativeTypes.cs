using System.Runtime.InteropServices;

namespace WheelDL.Interop.Native;

/// <summary>
/// Native result codes from WheelDL library
/// </summary>
public enum WheelResult
{
    Ok = 0,
    Error = 1,
    Timeout = 2,
    Stopped = 3
}

/// <summary>
/// Native task type enumeration
/// </summary>
public enum WheelTaskType
{
    Classification = 0,
    Detection = 1,
    Segmentation = 2,
    Anomaly = 3,
    Obb = 4,
    Unknown = 99
}

/// <summary>
/// Native operation type enumeration
/// </summary>
public enum WheelOperationType
{
    Train = 0,
    Validate = 1,
    Predict = 2
}

/// <summary>
/// Native training state enumeration
/// </summary>
public enum WheelTrainingState
{
    Idle = 0,
    Initializing = 1,
    Training = 2,
    Validating = 3,
    Paused = 4,
    Completed = 5,
    Failed = 6,
    Stopped = 7
}

/// <summary>
/// Native progress stage enumeration
/// </summary>
public enum WheelProgressStage
{
    TrainBatch = 0,
    TrainEpoch = 1,
    ValBatch = 2,
    ValEpoch = 3,
    CheckpointSaved = 4
}

/// <summary>
/// Native error structure
/// </summary>
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
public struct WheelError
{
    public int Code;

    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
    public string Message;

    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
    public string Category;

    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
    public string TaskId;
}

/// <summary>
/// Native metrics data structure
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct WheelMetricsData
{
    public float Loss;
    public float Accuracy;
    public float Precision;
    public float Recall;
    public float F1Score;
    public float MAP;
    public float Fitness;
    public float Threshold;
    public float AucROC;
}

/// <summary>
/// Native progress data structure
/// </summary>
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
public struct WheelProgressData
{
    public WheelProgressStage Stage;
    public int CurrentEpoch;
    public int TotalEpochs;
    public int CurrentBatch;
    public int TotalBatches;
    public float Loss;
    public float LearningRate;
    public float GpuMemoryUsage;
    public double ElapsedTime;
    public double Eta;
    public WheelMetricsData Metrics;

    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
    public string Message;
}

/// <summary>
/// Native profiling data structure
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct WheelProfilingData
{
    public double TotalTimeMs;
    public double GpuTimeMs;
    public double DataLoadTimeMs;
    public double PreprocessTimeMs;
    public double InferenceTimeMs;
    public double PostprocessTimeMs;
    public ulong PeakGpuMemoryMB;
    public ulong PeakCpuMemoryMB;
}

/// <summary>
/// Native task result structure
/// </summary>
[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
public struct WheelTaskResult
{
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
    public string TaskId;

    public WheelTrainingState FinalState;
    public WheelOperationType OperationType;
    public WheelMetricsData Metrics;
    public int HasMetrics;

    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
    public string OutputPath;

    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
    public string CheckpointPath;

    public double TotalTimeMs;
    public WheelProfilingData Profiling;

    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
    public string ErrorMessage;

    public int ErrorCode;
}

/// <summary>
/// Native progress callback delegate
/// </summary>
[UnmanagedFunctionPointer(CallingConvention.StdCall)]
public delegate void WheelProgressCallback(ref WheelProgressData progress, IntPtr userData);
