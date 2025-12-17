using System.Runtime.InteropServices;
using System.Text;

namespace WheelDL.Interop.Native;

/// <summary>
/// P/Invoke declarations for WheelDL native library
/// </summary>
internal static partial class NativeMethods
{
    private const string DllName = "WheelDL.Lib.dll";
    private const CallingConvention CallConv = CallingConvention.StdCall;

    // ==================== Library Lifecycle ====================

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_Initialize(ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_Shutdown(ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv, CharSet = CharSet.Ansi)]
    public static extern WheelResult Wheel_GetVersion(
        StringBuilder buffer,
        int bufferSize,
        ref WheelError err);

    // ==================== Configuration ====================

    [DllImport(DllName, CallingConvention = CallConv, CharSet = CharSet.Ansi)]
    public static extern WheelResult Wheel_Config_Create(
        string modelPath,
        string hyperParamPath,
        string datasetPath,
        out IntPtr outHandle,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_Config_Destroy(
        IntPtr handle,
        ref WheelError err);

    // ==================== Task Submission ====================

    [DllImport(DllName, CallingConvention = CallConv, CharSet = CharSet.Ansi)]
    public static extern WheelResult Wheel_Task_SubmitTraining(
        IntPtr configHandle,
        WheelProgressCallback? callback,
        IntPtr callbackUserData,
        StringBuilder outTaskId,
        int taskIdBufferSize,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv, CharSet = CharSet.Ansi)]
    public static extern WheelResult Wheel_Task_SubmitValidation(
        IntPtr configHandle,
        string checkpointPath,
        StringBuilder outTaskId,
        int taskIdBufferSize,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv, CharSet = CharSet.Ansi)]
    public static extern WheelResult Wheel_Task_SubmitPrediction(
        IntPtr configHandle,
        string checkpointPath,
        StringBuilder outTaskId,
        int taskIdBufferSize,
        ref WheelError err);

    // ==================== Task Query ====================

    [DllImport(DllName, CallingConvention = CallConv, CharSet = CharSet.Ansi)]
    public static extern WheelResult Wheel_Task_GetStatus(
        string taskId,
        out WheelProgressData progressData,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv, CharSet = CharSet.Ansi)]
    public static extern WheelResult Wheel_Task_GetResult(
        string taskId,
        out WheelTaskResult result,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv, CharSet = CharSet.Ansi)]
    public static extern WheelResult Wheel_Task_Exists(
        string taskId,
        out int exists,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv, CharSet = CharSet.Ansi)]
    public static extern WheelResult Wheel_Task_IsRunning(
        string taskId,
        out int isRunning,
        ref WheelError err);

    // ==================== Task Control ====================

    [DllImport(DllName, CallingConvention = CallConv, CharSet = CharSet.Ansi)]
    public static extern WheelResult Wheel_Task_Stop(
        string taskId,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv, CharSet = CharSet.Ansi)]
    public static extern WheelResult Wheel_Task_Wait(
        string taskId,
        ulong timeoutMs,
        out WheelTaskResult result,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv, CharSet = CharSet.Ansi)]
    public static extern WheelResult Wheel_Task_Remove(
        string taskId,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_Task_ClearCompleted(ref WheelError err);

    // ==================== Resource Query ====================

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_GetAvailableGpuMemory(
        int deviceIndex,
        out float outMB,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_CanAcceptNewTask(
        int deviceIndex,
        out int canAccept,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_GetRunningTaskCount(
        out int count,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_GetPendingTaskCount(
        out int count,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_SetMemoryThreshold(
        int thresholdMB,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_GetMemoryThreshold(
        out int thresholdMB,
        ref WheelError err);

    // ==================== Memory Management ====================

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_Memory_GetGPUDeviceCount(
        out int count,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_Memory_IsCudaAvailable(
        out int available,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_Memory_GetGPUUsedMB(
        int deviceIndex,
        out float outMB,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_Memory_GetGPUTotalMB(
        int deviceIndex,
        out float outMB,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_Memory_GetGPUUsagePercent(
        int deviceIndex,
        out float percent,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_Memory_HasEnoughGPU(
        int deviceIndex,
        ulong requiredBytes,
        out int hasEnough,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_Memory_GetCPUUsedBytes(
        out ulong outBytes,
        ref WheelError err);

    [DllImport(DllName, CallingConvention = CallConv)]
    public static extern WheelResult Wheel_Memory_GetCPUAvailableBytes(
        out ulong outBytes,
        ref WheelError err);
}
