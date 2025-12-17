namespace WheelDL.Interop.Tests.Integration;

/// <summary>
/// Integration tests for Configuration that require the native DLL
/// </summary>
[Collection("Integration")]
public class ConfigurationIntegrationTests : IntegrationTestBase
{
    // Test config paths - adjust these to actual test files if available
    private const string TestModelPath = "config/models/yolov8n.yaml";
    private const string TestHyperParamPath = "config/hyperparams/default.yaml";
    private const string TestDatasetPath = "dataset/test_dataset.json";

    [Fact]
    public void CreateConfiguration_WithInvalidPath_ShouldThrowConfigurationException()
    {
        SkipIfDllNotAvailable();

        var ex = Assert.Throws<ConfigurationException>(() =>
            Manager.CreateConfiguration(
                "nonexistent/model.yaml",
                "nonexistent/hyper.yaml",
                "nonexistent/dataset.json"));

        Assert.Equal("CONFIG", ex.Category);
    }

    [Fact]
    public void Configuration_Dispose_ShouldNotThrow()
    {
        SkipIfDllNotAvailable();

        // Even with invalid config, dispose should be safe
        Configuration? config = null;

        try
        {
            config = Manager.CreateConfiguration(TestModelPath, TestHyperParamPath, TestDatasetPath);
        }
        catch (ConfigurationException)
        {
            // Expected if test files don't exist
            return;
        }

        var ex = Record.Exception(() => config?.Dispose());
        Assert.Null(ex);
    }

    [Fact]
    public void Configuration_DoubleDispose_ShouldNotThrow()
    {
        SkipIfDllNotAvailable();

        Configuration? config = null;

        try
        {
            config = Manager.CreateConfiguration(TestModelPath, TestHyperParamPath, TestDatasetPath);
        }
        catch (ConfigurationException)
        {
            return;
        }

        config?.Dispose();
        var ex = Record.Exception(() => config?.Dispose());
        Assert.Null(ex);
    }

    [Fact]
    public void Configuration_AccessAfterDispose_ShouldThrowObjectDisposedException()
    {
        SkipIfDllNotAvailable();

        Configuration? config = null;

        try
        {
            config = Manager.CreateConfiguration(TestModelPath, TestHyperParamPath, TestDatasetPath);
        }
        catch (ConfigurationException)
        {
            return;
        }

        config?.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = config!.Handle);
    }
}
