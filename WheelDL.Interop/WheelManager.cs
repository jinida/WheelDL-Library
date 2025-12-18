using System.Text;
using WheelDL.Interop.Native;

namespace WheelDL.Interop;

/// <summary>
/// Main entry point for WheelDL library
/// </summary>
public sealed class WheelManager : IDisposable
{
    private static readonly Lazy<WheelManager> _instance = new(() => new WheelManager());
    private bool _initialized;
    private bool _disposed;

    /// <summary>
    /// Singleton instance
    /// </summary>
    public static WheelManager Instance => _instance.Value;

    private WheelManager() { }

    /// <summary>
    /// Initialize the WheelDL library
    /// </summary>
    public void Initialize()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_initialized) return;

        var err = new WheelError();
        var result = NativeMethods.Wheel_Initialize(ref err);
        WheelException.ThrowIfError(result, ref err);
        _initialized = true;
    }

    /// <summary>
    /// Shutdown the WheelDL library
    /// </summary>
    public void Shutdown()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_initialized) return;

        var err = new WheelError();
        var result = NativeMethods.Wheel_Shutdown(ref err);
        WheelException.ThrowIfError(result, ref err);
        _initialized = false;
    }

    /// <summary>
    /// Get library version string
    /// </summary>
    public string GetVersion()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var buffer = new StringBuilder(64);
        var err = new WheelError();
        var result = NativeMethods.Wheel_GetVersion(buffer, buffer.Capacity, ref err);
        WheelException.ThrowIfError(result, ref err);
        return buffer.ToString();
    }

    // ==================== Configuration ====================

    /// <summary>
    /// Create configuration from file paths
    /// </summary>
    /// <param name="modelPath">Path to model YAML file</param>
    /// <param name="hyperParamPath">Path to hyperparameters YAML file</param>
    /// <param name="datasetPath">Path to dataset JSON file</param>
    /// <returns>Configuration object</returns>
    public Configuration CreateConfiguration(string modelPath, string hyperParamPath, string datasetPath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EnsureInitialized();

        var err = new WheelError();
        var result = NativeMethods.Wheel_Config_Create(
            modelPath, hyperParamPath, datasetPath, out var handle, ref err);
        WheelException.ThrowIfError(result, ref err);

        return new Configuration(handle);
    }

    // ==================== Task Submission ====================

    /// <summary>
    /// Submit a training task
    /// </summary>
    /// <param name="config">Configuration to use</param>
    /// <param name="progressCallback">Optional progress callback</param>
    /// <returns>Task handle</returns>
    public TaskHandle SubmitTraining(Configuration config, Action<ProgressInfo>? progressCallback = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EnsureInitialized();

        // Create TaskHandle first - this pins the callback delegate
        var taskHandle = new TaskHandle(string.Empty, OperationType.Train, progressCallback);
        var taskIdBuffer = new StringBuilder(64);
        var err = new WheelError();

        try
        {
            var result = NativeMethods.Wheel_Task_SubmitTraining(
                config.Handle,
                taskHandle.NativeCallback,
                IntPtr.Zero,
                taskIdBuffer,
                taskIdBuffer.Capacity,
                ref err);

            WheelException.ThrowIfError(result, ref err);

            // Update task ID on the same TaskHandle (keeps callback pinned)
            taskHandle.SetTaskId(taskIdBuffer.ToString());
            return taskHandle;
        }
        catch
        {
            // Only dispose on failure - callback won't be called
            taskHandle.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Submit a validation task
    /// </summary>
    /// <param name="config">Configuration to use</param>
    /// <param name="checkpointPath">Path to checkpoint file</param>
    /// <returns>Task handle</returns>
    public TaskHandle SubmitValidation(Configuration config, string checkpointPath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EnsureInitialized();

        var taskIdBuffer = new StringBuilder(64);
        var err = new WheelError();

        var result = NativeMethods.Wheel_Task_SubmitValidation(
            config.Handle, checkpointPath, taskIdBuffer, taskIdBuffer.Capacity, ref err);
        WheelException.ThrowIfError(result, ref err);

        return new TaskHandle(taskIdBuffer.ToString(), OperationType.Validate);
    }

    /// <summary>
    /// Submit a prediction task
    /// </summary>
    /// <param name="config">Configuration to use</param>
    /// <param name="checkpointPath">Path to checkpoint file</param>
    /// <returns>Task handle</returns>
    public TaskHandle SubmitPrediction(Configuration config, string checkpointPath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        EnsureInitialized();

        var taskIdBuffer = new StringBuilder(64);
        var err = new WheelError();

        var result = NativeMethods.Wheel_Task_SubmitPrediction(
            config.Handle, checkpointPath, taskIdBuffer, taskIdBuffer.Capacity, ref err);
        WheelException.ThrowIfError(result, ref err);

        return new TaskHandle(taskIdBuffer.ToString(), OperationType.Predict);
    }

    // ==================== Task Query ====================

    /// <summary>
    /// Get task status by ID
    /// </summary>
    public ProgressInfo GetTaskStatus(string taskId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_GetStatus(taskId, out var progress, ref err);
        WheelException.ThrowIfError(result, ref err);
        return ProgressInfo.FromNative(progress);
    }

    /// <summary>
    /// Get task result by ID
    /// </summary>
    public TaskResultInfo GetTaskResult(string taskId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_GetResult(taskId, out var taskResult, ref err);
        WheelException.ThrowIfError(result, ref err);
        return TaskResultInfo.FromNative(taskResult);
    }

    /// <summary>
    /// Check if task exists
    /// </summary>
    public bool TaskExists(string taskId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_Exists(taskId, out var exists, ref err);
        WheelException.ThrowIfError(result, ref err);
        return exists != 0;
    }

    /// <summary>
    /// Check if task is running
    /// </summary>
    public bool IsTaskRunning(string taskId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_IsRunning(taskId, out var isRunning, ref err);
        WheelException.ThrowIfError(result, ref err);
        return isRunning != 0;
    }

    // ==================== Task Control ====================

    /// <summary>
    /// Stop a running task
    /// </summary>
    public void StopTask(string taskId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_Stop(taskId, ref err);
        WheelException.ThrowIfError(result, ref err);
    }

    /// <summary>
    /// Wait for task to complete
    /// </summary>
    public TaskResultInfo WaitForTask(string taskId, TimeSpan? timeout = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var timeoutMs = timeout.HasValue ? (ulong)timeout.Value.TotalMilliseconds : 0UL;

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_Wait(taskId, timeoutMs, out var taskResult, ref err);

        if (result == WheelResult.Timeout)
            throw new TaskTimeoutException(taskId, $"Task {taskId} timed out after {timeout}");

        if (result == WheelResult.Stopped)
            throw new TaskStoppedException(taskId);

        WheelException.ThrowIfError(result, ref err);
        return TaskResultInfo.FromNative(taskResult);
    }

    /// <summary>
    /// Remove task from registry
    /// </summary>
    public void RemoveTask(string taskId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_Remove(taskId, ref err);
        WheelException.ThrowIfError(result, ref err);
    }

    /// <summary>
    /// Clear all completed tasks
    /// </summary>
    public void ClearCompletedTasks()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Task_ClearCompleted(ref err);
        WheelException.ThrowIfError(result, ref err);
    }

    // ==================== Resource Query ====================

    /// <summary>
    /// Get available GPU memory in MB
    /// </summary>
    public float GetAvailableGpuMemory(int deviceIndex = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_GetAvailableGpuMemory(deviceIndex, out var mb, ref err);
        WheelException.ThrowIfError(result, ref err);
        return mb;
    }

    /// <summary>
    /// Check if library can accept a new task
    /// </summary>
    public bool CanAcceptNewTask(int deviceIndex = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_CanAcceptNewTask(deviceIndex, out var canAccept, ref err);
        WheelException.ThrowIfError(result, ref err);
        return canAccept != 0;
    }

    /// <summary>
    /// Get number of running tasks
    /// </summary>
    public int GetRunningTaskCount()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_GetRunningTaskCount(out var count, ref err);
        WheelException.ThrowIfError(result, ref err);
        return count;
    }

    /// <summary>
    /// Get number of pending tasks
    /// </summary>
    public int GetPendingTaskCount()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_GetPendingTaskCount(out var count, ref err);
        WheelException.ThrowIfError(result, ref err);
        return count;
    }

    /// <summary>
    /// Set memory threshold for task acceptance (in MB)
    /// </summary>
    public void SetMemoryThreshold(int thresholdMB)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_SetMemoryThreshold(thresholdMB, ref err);
        WheelException.ThrowIfError(result, ref err);
    }

    /// <summary>
    /// Get current memory threshold (in MB)
    /// </summary>
    public int GetMemoryThreshold()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_GetMemoryThreshold(out var threshold, ref err);
        WheelException.ThrowIfError(result, ref err);
        return threshold;
    }

    // ==================== Memory Management ====================

    /// <summary>
    /// Get number of available CUDA GPU devices
    /// </summary>
    public int GetGPUDeviceCount()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Memory_GetGPUDeviceCount(out var count, ref err);
        WheelException.ThrowIfError(result, ref err);
        return count;
    }

    /// <summary>
    /// Check if CUDA is available
    /// </summary>
    public bool IsCudaAvailable()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Memory_IsCudaAvailable(out var available, ref err);
        WheelException.ThrowIfError(result, ref err);
        return available != 0;
    }

    /// <summary>
    /// Get GPU memory currently used in MB
    /// </summary>
    public float GetGPUUsedMB(int deviceIndex = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Memory_GetGPUUsedMB(deviceIndex, out var mb, ref err);
        WheelException.ThrowIfError(result, ref err);
        return mb;
    }

    /// <summary>
    /// Get total GPU memory in MB
    /// </summary>
    public float GetGPUTotalMB(int deviceIndex = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Memory_GetGPUTotalMB(deviceIndex, out var mb, ref err);
        WheelException.ThrowIfError(result, ref err);
        return mb;
    }

    /// <summary>
    /// Get GPU memory usage percentage
    /// </summary>
    public float GetGPUUsagePercent(int deviceIndex = 0)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Memory_GetGPUUsagePercent(deviceIndex, out var percent, ref err);
        WheelException.ThrowIfError(result, ref err);
        return percent;
    }

    /// <summary>
    /// Check if GPU has enough memory available
    /// </summary>
    public bool HasEnoughGPU(int deviceIndex, ulong requiredBytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Memory_HasEnoughGPU(deviceIndex, requiredBytes, out var hasEnough, ref err);
        WheelException.ThrowIfError(result, ref err);
        return hasEnough != 0;
    }

    /// <summary>
    /// Get CPU memory currently used in bytes
    /// </summary>
    public ulong GetCPUUsedBytes()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Memory_GetCPUUsedBytes(out var bytes, ref err);
        WheelException.ThrowIfError(result, ref err);
        return bytes;
    }

    /// <summary>
    /// Get available CPU memory in bytes
    /// </summary>
    public ulong GetCPUAvailableBytes()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var err = new WheelError();
        var result = NativeMethods.Wheel_Memory_GetCPUAvailableBytes(out var bytes, ref err);
        WheelException.ThrowIfError(result, ref err);
        return bytes;
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            Initialize();
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_initialized)
        {
            Shutdown();
        }
    }
}
