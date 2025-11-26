using System.Collections;
using UnityEngine;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UnityUtils;
using Meta.XR;

/// <summary>
/// Frame provider that feeds camera frames from Meta Quest PassthroughCameraAccess to OpenCVImageProcessor.
/// Similar to ARCameraFrameProvider for ARFoundation, but for Meta Quest passthrough cameras.
///
/// IMPORTANT: Meta's PassthroughCameraAccess provides historical camera poses at the exact timestamp
/// when each frame was captured. This is different from ARFoundation which uses the current transform.
/// </summary>
[RequireComponent(typeof(OpenCVImageProcessor))]
public class PassthroughCameraFrameProvider : MonoBehaviour
{
    [Header("Passthrough Camera")]
    public PassthroughCameraAccess passthroughCamera;

    [Tooltip("Automatically request the highest available resolution for better tracking")]
    public bool UseHighestResolution = true;

    [Header("Processing")]
    [Tooltip("Process every Nth frame (1 = every frame, 2 = every other frame, etc.)")]
    [Range(1, 10)]
    public int frameSkip = 1;

    private OpenCVImageProcessor processor;
    private Mat frameRGBA;
    private int frameCounter = 0;
    private bool isInitialized = false;

    // Virtual camera to represent the historical camera pose
    private GameObject virtualCameraObject;
    private Transform virtualCameraTransform;

    void Start()
    {
        processor = GetComponent<OpenCVImageProcessor>();

        if (processor == null)
        {
            Debug.LogError("OpenCVImageProcessor component not found!");
            enabled = false;
            return;
        }

        if (passthroughCamera == null)
        {
            Debug.LogError("Passthrough camera not assigned!");
            enabled = false;
            return;
        }

        // Request highest resolution before camera starts playing
        if (UseHighestResolution)
        {
            SetHighestResolution(passthroughCamera);
        }

        // Create a virtual camera GameObject to represent the historical camera pose
        // This is necessary because Meta's PassthroughCameraAccess provides camera poses
        // at the exact timestamp when frames were captured (in the past), not the current pose
        virtualCameraObject = new GameObject("VirtualPassthroughCamera");
        virtualCameraTransform = virtualCameraObject.transform;

        // Set the virtual camera as the reference for the processor
        processor.SetCameraTransform(virtualCameraTransform);

        StartCoroutine(WaitForCameraAndInitialize());
    }

    private void SetHighestResolution(PassthroughCameraAccess camera)
    {
        var supportedResolutions = PassthroughCameraAccess.GetSupportedResolutions(camera.CameraPosition);
        if (supportedResolutions == null || supportedResolutions.Length == 0)
        {
            Debug.LogWarning("Could not get supported resolutions, using default");
            return;
        }

        // Find the highest resolution (by total pixel count)
        var highest = supportedResolutions[0];
        var highestPixels = highest.x * highest.y;

        for (var i = 1; i < supportedResolutions.Length; i++)
        {
            var pixels = supportedResolutions[i].x * supportedResolutions[i].y;
            if (pixels > highestPixels)
            {
                highest = supportedResolutions[i];
                highestPixels = pixels;
            }
        }

        camera.RequestedResolution = highest;
        Debug.Log($"[{camera.CameraPosition}] Requesting highest resolution: {highest.x}x{highest.y} (from {supportedResolutions.Length} supported resolutions)");
    }

    IEnumerator WaitForCameraAndInitialize()
    {
        // Wait for camera to start playing
        while (!passthroughCamera.IsPlaying)
        {
            yield return null;
        }

        // Get camera resolution and intrinsics
        Vector2Int currentResolution = passthroughCamera.CurrentResolution;
        PassthroughCameraAccess.CameraIntrinsics intrinsics = passthroughCamera.Intrinsics;
        Vector2Int sensorResolution = intrinsics.SensorResolution;

        Debug.Log($"Passthrough camera initialized: {currentResolution.x}x{currentResolution.y}");
        Debug.Log($"Sensor Resolution: {sensorResolution.x}x{sensorResolution.y}");
        Debug.Log($"Focal Length (sensor): {intrinsics.FocalLength}");
        Debug.Log($"Principal Point (sensor): {intrinsics.PrincipalPoint}");

        // Scale intrinsics from sensor resolution to current resolution
        // The intrinsics are defined for the full sensor, so we need to adjust them
        // when using a cropped/scaled resolution
        float scaleX = (float)currentResolution.x / sensorResolution.x;
        float scaleY = (float)currentResolution.y / sensorResolution.y;
        float scale = Mathf.Max(scaleX, scaleY);

        // Calculate the crop offset (center crop)
        float cropOffsetX = (sensorResolution.x * scale - currentResolution.x) * 0.5f;
        float cropOffsetY = (sensorResolution.y * scale - currentResolution.y) * 0.5f;

        // Scale focal length and adjust principal point for the current resolution
        float fx = intrinsics.FocalLength.x * scale;
        float fy = intrinsics.FocalLength.y * scale;
        float cx = intrinsics.PrincipalPoint.x * scale - cropOffsetX;
        float cy = intrinsics.PrincipalPoint.y * scale - cropOffsetY;

        Debug.Log($"Scaled Focal Length: ({fx}, {fy})");
        Debug.Log($"Scaled Principal Point: ({cx}, {cy})");

        // Update camera intrinsics in the processor
        processor.UpdateCameraIntrinsics(fx, fy, cx, cy);

        // Pre-allocate frame Mat
        frameRGBA = new Mat(currentResolution.y, currentResolution.x, CvType.CV_8UC4);

        isInitialized = true;
        Debug.Log($"PassthroughCameraFrameProvider initialized successfully");
    }

    void Update()
    {
        if (!isInitialized)
            return;

        // Check if camera is playing
        if (!passthroughCamera.IsPlaying)
            return;

        // Skip frames if configured
        frameCounter++;
        if (frameCounter % frameSkip != 0)
            return;

        // Only process if texture was updated this frame
        if (!passthroughCamera.IsUpdatedThisFrame)
            return;

        // CRITICAL: Update the virtual camera pose to match the historical pose
        Pose cameraPose = passthroughCamera.GetCameraPose();
        virtualCameraTransform.SetPositionAndRotation(cameraPose.position, cameraPose.rotation);

        // Get and convert camera texture
        Texture2D cameraTexture = passthroughCamera.GetTexture() as Texture2D;
        if (cameraTexture == null)
            return;

        Utils.texture2DToMat(cameraTexture, frameRGBA);

        // Process frame
        processor.ProcessFrame(frameRGBA);
    }

    void OnDestroy()
    {
        if (frameRGBA != null)
        {
            frameRGBA.Dispose();
            frameRGBA = null;
        }

        // Clean up virtual camera object
        if (virtualCameraObject != null)
        {
            Destroy(virtualCameraObject);
        }
    }
}
