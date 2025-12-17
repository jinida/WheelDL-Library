using System.Runtime.InteropServices;
using WheelDL.Interop.Native;

namespace WheelDL.Interop;

/// <summary>
/// Handle to a submitted task
/// </summary>
public sealed class TaskHandle : IDisposable
{
    private readonly WheelProgressCallback? _nativeCallback;
    private readonly GCHandle _callbackHandle;
    private readonly Action<ProgressInfo>? _userCallback;
    private bool _disposed;

    /// <summary>
    /// Task ID
    /// </summary>
    public string TaskId { get; }

    /// <summary>
    /// Operation type of this task
    /// </summary>
    public OperationType OperationType { get; }

    internal TaskHandle(string taskId, OperationType operationType,
        Action<ProgressInfo>? userCallback = null)
    {
        TaskId = taskId;
        OperationType = operationType;
        _userCallback = userCallback;

        // If we have a user callback, create and pin the native callback
        if (userCallback != null)
        {
            _nativeCallback = OnNativeProgress;
            _callbackHandle = GCHandle.Alloc(_nativeCallback);
        }
    }

    private void OnNativeProgress(ref WheelProgressData progress, IntPtr userData)
    {
        try
        {
            var info = ProgressInfo.FromNative(progress);
            _userCallback?.Invoke(info);
        }
        catch
        {
            // Never let exceptions escape to native code
        }
    }

    internal WheelProgressCallback? NativeCallback => _nativeCallback;

    /// <summary>
    /// Get current task status
    /// </summary>
    public ProgressInfo GetStatus()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_GetStatus(TaskId, out var progress, ref err);
        WheelException.ThrowIfError(result, ref err);
        return ProgressInfo.FromNative(progress);
    }

    /// <summary>
    /// Get task result (only valid after task completes)
    /// </summary>
    public TaskResultInfo GetResult()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_GetResult(TaskId, out var taskResult, ref err);
        WheelException.ThrowIfError(result, ref err);
        return TaskResultInfo.FromNative(taskResult);
    }

    /// <summary>
    /// Check if task exists in the registry
    /// </summary>
    public bool Exists()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_Exists(TaskId, out var exists, ref err);
        WheelException.ThrowIfError(result, ref err);
        return exists != 0;
    }

    /// <summary>
    /// Check if task is currently running
    /// </summary>
    public bool IsRunning()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_IsRunning(TaskId, out var isRunning, ref err);
        WheelException.ThrowIfError(result, ref err);
        return isRunning != 0;
    }

    /// <summary>
    /// Stop the running task
    /// </summary>
    public void Stop()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_Stop(TaskId, ref err);
        WheelException.ThrowIfError(result, ref err);
    }

    /// <summary>
    /// Wait for task to complete
    /// </summary>
    /// <param name="timeout">Timeout duration (null for infinite)</param>
    /// <returns>Task result</returns>
    /// <exception cref="TaskTimeoutException">Thrown if timeout exceeded</exception>
    /// <exception cref="TaskStoppedException">Thrown if task was stopped</exception>
    public TaskResultInfo Wait(TimeSpan? timeout = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var timeoutMs = timeout.HasValue ? (ulong)timeout.Value.TotalMilliseconds : 0UL;

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_Wait(TaskId, timeoutMs, out var taskResult, ref err);

        if (result == WheelResult.Timeout)
            throw new TaskTimeoutException(TaskId, $"Task {TaskId} timed out after {timeout}");

        if (result == WheelResult.Stopped)
            throw new TaskStoppedException(TaskId);

        WheelException.ThrowIfError(result, ref err);
        return TaskResultInfo.FromNative(taskResult);
    }

    /// <summary>
    /// Wait for task to complete asynchronously
    /// </summary>
    public Task<TaskResultInfo> WaitAsync(TimeSpan? timeout = null,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Wait(timeout), cancellationToken);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // Free the callback handle if we allocated one
        if (_callbackHandle.IsAllocated)
        {
            _callbackHandle.Free();
        }
    }

    public override string ToString() => $"TaskHandle({TaskId}, {OperationType})";
}
