namespace WheelDL.Interop.Tests.Integration;

/// <summary>
/// Base class for integration tests that require the native DLL
/// </summary>
public abstract class IntegrationTestBase : IDisposable
{
    protected static bool IsDllAvailable { get; }
    protected WheelManager Manager { get; }

    static IntegrationTestBase()
    {
        // Check if DLL is available by trying to load it
        try
        {
            var manager = WheelManager.Instance;
            manager.Initialize();
            IsDllAvailable = true;
        }
        catch (DllNotFoundException)
        {
            IsDllAvailable = false;
        }
        catch (Exception)
        {
            // Other errors might occur even if DLL exists
            IsDllAvailable = false;
        }
    }

    protected IntegrationTestBase()
    {
        Manager = WheelManager.Instance;
    }

    protected void SkipIfDllNotAvailable()
    {
        Skip.If(!IsDllAvailable, "Native DLL not available");
    }

    public virtual void Dispose()
    {
        // Don't dispose the singleton manager
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Custom Skip attribute for conditional test execution
/// </summary>
public static class Skip
{
    public static void If(bool condition, string reason)
    {
        if (condition)
        {
            throw new SkipException(reason);
        }
    }
}

/// <summary>
/// Exception thrown when a test should be skipped
/// </summary>
public class SkipException : Exception
{
    public SkipException(string reason) : base(reason) { }
}
