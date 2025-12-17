using WheelDL.Interop.Native;

namespace WheelDL.Interop;

/// <summary>
/// Base exception for all WheelDL errors
/// </summary>
public class WheelException : Exception
{
    /// <summary>
    /// Native error code
    /// </summary>
    public int ErrorCode { get; }

    /// <summary>
    /// Error category (CONFIG, MODEL, DATA, GPU, TRAINING, TASK, etc.)
    /// </summary>
    public string Category { get; }

    /// <summary>
    /// Associated task ID (only for task-related errors)
    /// </summary>
    public string? TaskId { get; }

    public WheelException(string message) : base(message)
    {
        ErrorCode = 0;
        Category = "UNKNOWN";
    }

    public WheelException(string message, Exception innerException)
        : base(message, innerException)
    {
        ErrorCode = 0;
        Category = "UNKNOWN";
    }

    protected WheelException(int errorCode, string category, string? taskId, string message)
        : base(message)
    {
        ErrorCode = errorCode;
        Category = category;
        TaskId = taskId;
    }

    internal static WheelException FromNative(WheelError err)
    {
        var message = err.Message ?? "Unknown error";
        var category = err.Category ?? "UNKNOWN";
        var taskId = string.IsNullOrEmpty(err.TaskId) ? null : err.TaskId;

        return category switch
        {
            "CONFIG" => new ConfigurationException(err.Code, category, message),
            "MODEL" => new ModelException(err.Code, category, message),
            "DATA" => new DataException(err.Code, category, message),
            "GPU" => new GpuException(err.Code, category, message),
            "TRAINING" => new TrainingException(err.Code, category, message),
            "DISTRIBUTED" => new DistributedException(err.Code, category, message),
            "TASK" => new TaskException(err.Code, category, taskId, message),
            _ => new WheelException(err.Code, category, taskId, message)
        };
    }

    internal static void ThrowIfError(WheelResult result, ref WheelError err)
    {
        if (result == WheelResult.Ok) return;

        throw FromNative(err);
    }
}

/// <summary>
/// Exception for configuration-related errors
/// </summary>
public class ConfigurationException : WheelException
{
    internal ConfigurationException(int errorCode, string category, string message)
        : base(errorCode, category, null, message) { }
}

/// <summary>
/// Exception for model-related errors
/// </summary>
public class ModelException : WheelException
{
    internal ModelException(int errorCode, string category, string message)
        : base(errorCode, category, null, message) { }
}

/// <summary>
/// Exception for data-related errors
/// </summary>
public class DataException : WheelException
{
    internal DataException(int errorCode, string category, string message)
        : base(errorCode, category, null, message) { }
}

/// <summary>
/// Exception for GPU-related errors
/// </summary>
public class GpuException : WheelException
{
    internal GpuException(int errorCode, string category, string message)
        : base(errorCode, category, null, message) { }
}

/// <summary>
/// Exception for training-related errors
/// </summary>
public class TrainingException : WheelException
{
    internal TrainingException(int errorCode, string category, string message)
        : base(errorCode, category, null, message) { }
}

/// <summary>
/// Exception for distributed training errors
/// </summary>
public class DistributedException : WheelException
{
    internal DistributedException(int errorCode, string category, string message)
        : base(errorCode, category, null, message) { }
}

/// <summary>
/// Exception for task-related errors
/// </summary>
public class TaskException : WheelException
{
    /// <summary>
    /// The task ID associated with this error
    /// </summary>
    public string? RelatedTaskId => TaskId;

    internal TaskException(int errorCode, string category, string? taskId, string message)
        : base(errorCode, category, taskId, message) { }
}

/// <summary>
/// Exception thrown when a task times out
/// </summary>
public class TaskTimeoutException : TaskException
{
    internal TaskTimeoutException(string? taskId, string message)
        : base(7004, "TASK", taskId, message) { }
}

/// <summary>
/// Exception thrown when a task is stopped by user
/// </summary>
public class TaskStoppedException : TaskException
{
    internal TaskStoppedException(string? taskId)
        : base(7003, "TASK", taskId, "Task was stopped by user request") { }
}
