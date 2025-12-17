namespace WheelDL.Interop;

/// <summary>
/// Task type enumeration
/// </summary>
public enum TaskType
{
    Classification = 0,
    Detection = 1,
    Segmentation = 2,
    Anomaly = 3,
    Obb = 4,
    Unknown = 99
}

/// <summary>
/// Operation type enumeration
/// </summary>
public enum OperationType
{
    Train = 0,
    Validate = 1,
    Predict = 2
}

/// <summary>
/// Training state enumeration
/// </summary>
public enum TrainingState
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
/// Progress stage enumeration
/// </summary>
public enum ProgressStage
{
    TrainBatch = 0,
    TrainEpoch = 1,
    ValBatch = 2,
    ValEpoch = 3,
    CheckpointSaved = 4
}
