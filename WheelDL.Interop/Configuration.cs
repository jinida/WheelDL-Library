using WheelDL.Interop.Native;

namespace WheelDL.Interop;

/// <summary>
/// Managed wrapper for WheelDL configuration.
/// Configuration is loaded from YAML/JSON files on the C++ side.
/// </summary>
public sealed class Configuration : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;

    /// <summary>
    /// Native handle (internal use only)
    /// </summary>
    internal IntPtr Handle
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _handle;
        }
    }

    internal Configuration(IntPtr handle)
    {
        _handle = handle;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_handle != IntPtr.Zero)
        {
            var err = new WheelError();
            NativeMethods.Wheel_Config_Destroy(_handle, ref err);
            _handle = IntPtr.Zero;
        }
    }
}
