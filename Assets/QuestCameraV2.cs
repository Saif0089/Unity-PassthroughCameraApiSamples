using TMPro;
// using Trace;
using System;
using Meta.XR;
using UnityEngine;
// using Trace.Anchors;
using System.Collections;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UnityUtils;
using System.Collections.Generic;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.Features2dModule;

public class QuestCameraV2 : MonoBehaviour
{
    [SerializeField] private float _imageWidth;
    [SerializeField] private Texture2D _imageToBeScanned;
    [SerializeField] private TMP_Text _progressText;
    [SerializeField] private TMP_Text _inliersText;
    [SerializeField] private Vector3 _rotationOffset = Vector3.zero;

    [Header("References")]
    [SerializeField] private Camera _centerEyeCam;
    // [SerializeField] private ImageAnchorProviderMetaQuest _imageAnchorProvider;
    [SerializeField] private PassthroughCameraAccess _passthroughCamera; // NEW: PCA component
    [SerializeField] private EnvironmentRaycastManager _raycastManager;
    [SerializeField] private GameObject _ring;
    public GameObject ImageAnchor;

    [Header("Detection Settings")]
    [Tooltip("Minimum inlier count for valid tracking")]
    [SerializeField] private int _minInliers = 12;

    [Tooltip("Cooldown between frame processing (sec)")]
    [SerializeField] private float _processInterval = 0.5f;

    [Tooltip("Downscale factor for input frames (0.1 - 1)")]
    [Range(0.1f, 1f)]
    [SerializeField] private float _downscaleFactor = 0.5f;

    [Tooltip("Ratio threshold for Lowe's test")]
    [Range(0.5f, 0.85f)]
    [SerializeField] private float _ratioThreshold = 0.75f;

    [Header("Lighting Adaptation")]
    [Tooltip("Enable CLAHE for better low-light performance")]
    [SerializeField] private bool _useCLAHE = true;

    [Tooltip("CLAHE clip limit (higher = more contrast)")]
    [Range(1f, 10f)]
    [SerializeField] private float _claheClipLimit = 2.0f;

    private Texture2D _referenceImage;
    private Mat _refMat, _refDescriptors;
    private MatOfKeyPoint _refKeypoints;
    private MatOfPoint3f _objectPoints3D; // NEW: 3D points of reference image
    private ORB _orb;
    private BFMatcher _matcher;
    private Mat _grayMat;
    private CLAHE _clahe;
    private bool _isCameraReady;
    private float _lastProcessTime;
    private float _currentSize;
    private Point[] _corners;

    // Camera intrinsics from PCA
    private Mat _cameraMatrix;
    private MatOfDouble _distCoeffs;
    private bool _scan = false;

    private void Awake()
    {
        _orb = ORB.create(2000);
        _matcher = BFMatcher.create(Core.NORM_HAMMING, false);

        if (_useCLAHE)
        {
            _clahe = Imgproc.createCLAHE(_claheClipLimit, new Size(8, 8));
        }
    }

    private void Start()
    {
        _ = StartCoroutine(WaitForCamera());
        ProcessImg();
    }

    private async void ProcessImg()
    {
        ScanImage();
        await ProcessReferenceImageAsync(_imageToBeScanned, _imageWidth);
    }

    private IEnumerator WaitForCamera()
    {
        Debug.Log("[Tracker] Waiting for PCA camera...");

        // Wait for PassthroughCameraAccess to be ready
        while (!_passthroughCamera.IsPlaying)
        {
            yield return null;
        }

        // Get camera intrinsics from PCA
        var intrinsics = _passthroughCamera.Intrinsics;

        // Build OpenCV camera matrix
        _cameraMatrix = new Mat(3, 3, CvType.CV_64FC1);
        _ = _cameraMatrix.put(0, 0,
            intrinsics.FocalLength.x, 0, intrinsics.PrincipalPoint.x,
            0, intrinsics.FocalLength.y, intrinsics.PrincipalPoint.y,
            0, 0, 1);

        // For now, assume zero distortion (PCA provides undistorted images)
        _distCoeffs = new MatOfDouble(0, 0, 0, 0, 0);

        Debug.Log($"[Tracker] Camera Matrix: fx={intrinsics.FocalLength.x:F2}, fy={intrinsics.FocalLength.y:F2}, " +
                  $"cx={intrinsics.PrincipalPoint.x:F2}, cy={intrinsics.PrincipalPoint.y:F2}");

        _isCameraReady = true;
        enabled = true;
        Debug.Log("[Tracker] PCA Camera ready.");
    }

    public void ScanImage()
    {
        if (_grayMat != null) _grayMat.Dispose();
        _grayMat = new Mat();
    }

    private void Update()
    {
        if (!_isCameraReady) return;
        if (!_passthroughCamera.IsPlaying) return;

        // Check if frame was updated this frame
        if (!_passthroughCamera.IsUpdatedThisFrame) return;

        if (Time.time - _lastProcessTime < _processInterval) return;
        _lastProcessTime = Time.time;

        if (_scan)
        {
            ProcessFrame();
        }
    }

    private void ProcessFrame()
    {
        // Get texture from PCA
        var pcaTexture = _passthroughCamera.GetTexture() as Texture2D;
        if (pcaTexture == null) return;

        var w = pcaTexture.width;
        var h = pcaTexture.height;
        if (w < 16 || h < 16) return;

        // Convert to Mat
        var rgba = new Mat(h, w, CvType.CV_8UC4);
        Utils.texture2DToMat(pcaTexture, rgba);

        if (_grayMat == null) _grayMat = new Mat();
        Imgproc.cvtColor(rgba, _grayMat, Imgproc.COLOR_RGBA2GRAY);

        // Apply CLAHE for better lighting adaptation
        if (_useCLAHE && _clahe != null)
        {
            _clahe.apply(_grayMat, _grayMat);
        }
        else
        {
            Imgproc.equalizeHist(_grayMat, _grayMat);
        }

        // Downscale if needed
        float actualScale = 1f;
        if (_downscaleFactor < 1f)
        {
            var size = new Size(_grayMat.cols() * _downscaleFactor, _grayMat.rows() * _downscaleFactor);
            Imgproc.resize(_grayMat, _grayMat, size);
            actualScale = _downscaleFactor;
        }

        // Detect + compute
        var keypoints = new MatOfKeyPoint();
        var descriptors = new Mat();
        _orb.detectAndCompute(_grayMat, new Mat(), keypoints, descriptors);

        if (descriptors.empty() || _refDescriptors.empty())
        {
            rgba.Dispose();
            return;
        }

        // KNN matching with Lowe's ratio test
        var knnMatches = new List<MatOfDMatch>();
        _matcher.knnMatch(_refDescriptors, descriptors, knnMatches, 2);

        var goodMatches = new List<DMatch>();
        foreach (var m in knnMatches)
        {
            var arr = m.toArray();
            if (arr.Length >= 2 && arr[0].distance < _ratioThreshold * arr[1].distance)
                goodMatches.Add(arr[0]);
        }

        if (goodMatches.Count < 80)
        {
            var matchProgress = Mathf.Clamp01(goodMatches.Count / 20f);
            _progressText.text = $"Scanning... {(int)(matchProgress * 100)}%";
            rgba.Dispose();
            return;
        }

        // Prepare point lists
        List<Point> objPts2D = new List<Point>();
        List<Point3> objPts3D = new List<Point3>();
        List<Point> scenePts = new List<Point>();

        var refKP = _refKeypoints.toArray();
        var curKP = keypoints.toArray();
        var obj3D = _objectPoints3D.toArray();

        foreach (var m in goodMatches)
        {
            objPts2D.Add(refKP[m.queryIdx].pt);
            objPts3D.Add(obj3D[m.queryIdx]);
            scenePts.Add(curKP[m.trainIdx].pt);
        }

        if (objPts3D.Count < 4)
        {
            rgba.Dispose();
            return;
        }

        // Scale scene points back to original resolution
        for (int i = 0; i < scenePts.Count; i++)
        {
            scenePts[i] = new Point(scenePts[i].x / actualScale, scenePts[i].y / actualScale);
        }

        // Use solvePnPRansac for robust ROTATION estimation
        var rvec = new Mat();
        var tvec = new Mat();
        var inliers = new Mat();

        bool success = Calib3d.solvePnPRansac(
            new MatOfPoint3f(objPts3D.ToArray()),
            new MatOfPoint2f(scenePts.ToArray()),
            _cameraMatrix,
            _distCoeffs,
            rvec,
            tvec,
            false,
            100,
            8.0f,
            0.99,
            inliers
        );

        var inlierCount = inliers.rows();
        // _inliersText.text = $"Inliers: {inlierCount}";

        if (!success || inlierCount < _minInliers)
        {
            rgba.Dispose();
            return;
        }

        // ========== NEW: Use rotation from solvePnP, but position from raycast ==========

        // 1. Get ROTATION from solvePnP (this is accurate!)
        var rotMat = new Mat();
        Calib3d.Rodrigues(rvec, rotMat);
        Quaternion targetRotationInCameraSpace = OpenCVRotationToUnity(rotMat);

        // Get camera pose in world space from PCA
        var cameraPose = _passthroughCamera.GetCameraPose();
        Quaternion targetRotationInWorldSpace = cameraPose.rotation * targetRotationInCameraSpace;

        // 2. Get POSITION from raycast (this grounds it to real geometry!)
        // Calculate center of detected image in pixel coordinates
        var inlierIndices = new List<int>();
        for (int i = 0; i < inliers.rows(); i++)
        {
            if (inliers.get(i, 0)[0] != 0)
                inlierIndices.Add(i);
        }

        double centerX = 0, centerY = 0;
        foreach (var idx in inlierIndices)
        {
            centerX += scenePts[idx].x;
            centerY += scenePts[idx].y;
        }
        centerX /= inlierIndices.Count;
        centerY /= inlierIndices.Count;

        // Convert to viewport coordinates (0-1 range)
        // Note: OpenCV has origin at top-left, we need bottom-left for Unity
        var viewportPoint = new Vector2(
            (float)(centerX / w),
            (float)(1.0 - centerY / h)  // Flip Y axis
        );

        Debug.Log($"[Tracker] Center pixel: ({centerX:F1}, {centerY:F1}), Viewport: {viewportPoint}");

        // Cast ray from camera through image center using PCA intrinsics
        var ray = _passthroughCamera.ViewportPointToRay(viewportPoint);

        Debug.Log($"[Tracker] Ray origin: {ray.origin}, direction: {ray.direction}");

        // Raycast to find position on real environment
        if (_raycastManager.Raycast(ray, out var hit))
        {
            // Apply rotation offset
            var offsetRotation = Quaternion.Euler(_rotationOffset);
            targetRotationInWorldSpace *= offsetRotation;

            Debug.Log($"[Tracker] Hit at: {hit.point}, Rotation: {targetRotationInWorldSpace.eulerAngles} (Inliers: {inlierCount})");

            // Snap rotation
            var snappedRotation = QuaternionSnapper.SnapToMainAngles(targetRotationInWorldSpace);
            Debug.Log($"[Tracker] Snapped Rotation: {snappedRotation.eulerAngles}");

            // Show ring and anchor
            _ring.SetActive(true);
            _ring.transform.SetPositionAndRotation(hit.point, snappedRotation);
            _ring.transform.localScale = new Vector3(_currentSize, _currentSize, _currentSize);

            _ = StartCoroutine(ShowImageAnchor(hit.point, snappedRotation));
        }
        else
        {
            Debug.LogWarning("[Tracker] Raycast missed! Ray may not hit environment geometry.");
            _ring.SetActive(false);
        }

        rgba.Dispose();
    }

    private Quaternion OpenCVRotationToUnity(Mat rotMat)
    {
        // Extract 3x3 rotation matrix values
        float m00 = (float)rotMat.get(0, 0)[0];
        float m01 = (float)rotMat.get(0, 1)[0];
        float m02 = (float)rotMat.get(0, 2)[0];
        float m10 = (float)rotMat.get(1, 0)[0];
        float m11 = (float)rotMat.get(1, 1)[0];
        float m12 = (float)rotMat.get(1, 2)[0];
        float m20 = (float)rotMat.get(2, 0)[0];
        float m21 = (float)rotMat.get(2, 1)[0];
        float m22 = (float)rotMat.get(2, 2)[0];

        // Convert OpenCV coordinate system to Unity
        // OpenCV: +X right, +Y down, +Z forward
        // Unity: +X right, +Y up, +Z forward
        // We need to flip Y axis
        var cvMatrix = new Matrix4x4
        {
            m00 = m00,
            m01 = -m01,
            m02 = m02,
            m03 = 0,
            m10 = -m10,
            m11 = m11,
            m12 = -m12,
            m13 = 0,
            m20 = m20,
            m21 = -m21,
            m22 = m22,
            m23 = 0,
            m30 = 0,
            m31 = 0,
            m32 = 0,
            m33 = 1
        };

        return cvMatrix.rotation;
    }

    private IEnumerator ShowImageAnchor(Vector3 position, Quaternion rotation)
    {
        yield return new WaitForSeconds(0.8f);
        _ring.SetActive(false);

        ImageAnchor.SetActive(true);
        ImageAnchor.transform.SetPositionAndRotation(position, rotation);
        ImageAnchor.transform.localScale = new Vector3(_imageWidth, _imageWidth, _imageWidth);

        Debug.Log($"[Tracker] Final Position: {position}, Rotation: {rotation.eulerAngles}");

        // Build a temporary Transform to carry the pose
        var dummy = new GameObject("CVAnchor");
        dummy.transform.SetPositionAndRotation(position, rotation);

        // Create MarkerInfo
        // var model = _imageAnchorProvider.GetMarkerTarget();
        // Debug.Log($"[Tracker] Model Info: {model}");
        // var info = new MarkerInfo(model, dummy.transform);
        // Debug.Log($"[Tracker] Marker Info: {info}");

        // _imageAnchorProvider.HandleFound(info);
    }

    public async Task ProcessReferenceImageAsync(Texture2D tex, float realWorldSizeInMeters)
    {
        Debug.Log("[Tracker] Processing reference image...");

        _currentSize = realWorldSizeInMeters;

        // Clean up old resources
        if (_refMat != null) _refMat.Dispose();
        if (_refKeypoints != null) _refKeypoints.Dispose();
        if (_refDescriptors != null) _refDescriptors.Dispose();
        if (_objectPoints3D != null) _objectPoints3D.Dispose();

        // Convert texture to grayscale Mat
        var grayMat = new Mat(tex.height, tex.width, CvType.CV_8UC4);
        Utils.texture2DToMat(tex, grayMat);
        Imgproc.cvtColor(grayMat, grayMat, Imgproc.COLOR_RGBA2GRAY);

        // Apply CLAHE to reference image as well
        if (_useCLAHE && _clahe != null)
        {
            _clahe.apply(grayMat, grayMat);
        }

        // Get dynamic scaling multiplier
        var scaleMultiplier = GetScaleMultiplierForImageSize(realWorldSizeInMeters);
        var scaledMeters = realWorldSizeInMeters * scaleMultiplier;

        var assumedPixelsPerMeter = 1600f;
        var targetPixels = scaledMeters * assumedPixelsPerMeter;
        float maxDimension = Mathf.Max(tex.width, tex.height);
        var scale = targetPixels / maxDimension;

        // Clamp to safe scale range
        scale = Mathf.Clamp(scale, 0.8f, 4f);

        Debug.Log($"[Tracker] ScaleMultiplier: {scaleMultiplier}, Final scale: {scale:F2}");

        var scaledMat = new Mat();
        if (Mathf.Abs(scale - 1f) > 0.01f)
        {
            var newSize = new Size(grayMat.width() * scale, grayMat.height() * scale);
            Imgproc.resize(grayMat, scaledMat, newSize);
        }
        else
        {
            scaledMat = grayMat;
        }

        _refMat = scaledMat;

        // Use tuned ORB parameters
        _orb = ORB.create(
            2000, 1.2f, 8, 16, 0, 2, ORB.HARRIS_SCORE, 31, 10
        );

        _refKeypoints = new MatOfKeyPoint();
        _refDescriptors = new Mat();
        _orb.detectAndCompute(_refMat, new Mat(), _refKeypoints, _refDescriptors);

        // Fallback to original if nothing was found
        if (_refDescriptors.empty() || _refKeypoints.empty())
        {
            Debug.LogWarning("[Tracker] No keypoints after scaling. Reverting to original image.");
            _refMat = grayMat;
            _refKeypoints = new MatOfKeyPoint();
            _refDescriptors = new Mat();
            _orb.detectAndCompute(_refMat, new Mat(), _refKeypoints, _refDescriptors);
        }

        // Create 3D object points for solvePnP
        // Assume the image lies flat on XZ plane (Y=0)
        // Center at origin, size = realWorldSizeInMeters
        var halfSize = realWorldSizeInMeters / 2f;
        var aspectRatio = (float)tex.width / tex.height;

        var refKP = _refKeypoints.toArray();
        List<Point3> objectPoints = new List<Point3>();

        foreach (var kp in refKP)
        {
            // Normalize keypoint coordinates to [-0.5, 0.5] range
            float normX = (float)(kp.pt.x / _refMat.width() - 0.5);
            float normY = (float)(kp.pt.y / _refMat.height() - 0.5);

            // Scale to real-world size
            float worldX = normX * realWorldSizeInMeters * aspectRatio;
            float worldZ = normY * realWorldSizeInMeters;

            // Create 3D point (lying flat on XZ plane, Y=0)
            objectPoints.Add(new Point3(worldX, 0, worldZ));
        }

        _objectPoints3D = new MatOfPoint3f(objectPoints.ToArray());
        _referenceImage = tex;
        _scan = true;

        Debug.Log($"[Tracker] Reference descriptors: {_refDescriptors.rows()}x{_refDescriptors.cols()} | " +
                  $"Keypoints: {_refKeypoints.size().height} | 3D Points: {_objectPoints3D.rows()}");
    }

    private float GetScaleMultiplierForImageSize(float sizeInMeters)
    {
        if (sizeInMeters <= 0.03f) return 6f;
        if (sizeInMeters <= 0.05f) return 5f;
        if (sizeInMeters <= 0.07f) return 4f;
        if (sizeInMeters <= 0.10f) return 3f;
        if (sizeInMeters <= 0.15f) return 2.5f;
        if (sizeInMeters <= 0.20f) return 2f;
        if (sizeInMeters <= 0.30f) return 1.5f;
        return 1f;
    }

    private void OnDisable()
    {
        _ring.SetActive(false);

        // Clean up OpenCV resources
        if (_refMat != null) _refMat.Dispose();
        if (_refKeypoints != null) _refKeypoints.Dispose();
        if (_refDescriptors != null) _refDescriptors.Dispose();
        if (_objectPoints3D != null) _objectPoints3D.Dispose();
        if (_grayMat != null) _grayMat.Dispose();
        if (_cameraMatrix != null) _cameraMatrix.Dispose();
        if (_distCoeffs != null) _distCoeffs.Dispose();
        if (_clahe != null) _clahe.Dispose();
    }
}

public static class QuaternionSnapper
{
    public static Quaternion SnapToMainAngles(Quaternion rotation, float snapThreshold = 20f)
    {
        var eulerAngles = rotation.eulerAngles;

        var snappedX = SnapAngle(eulerAngles.x, snapThreshold);
        var snappedY = eulerAngles.y;
        var snappedZ = SnapAngle(eulerAngles.z, snapThreshold);

        return Quaternion.Euler(snappedX, snappedY, snappedZ);
    }

    private static float SnapAngle(float angle, float threshold)
    {
        float[] mainAngles = { 0f, 90f, 180f, 270f };

        angle %= 360f;
        if (angle < 0) angle += 360f;

        foreach (var mainAngle in mainAngles)
        {
            var distance = Mathf.Abs(Mathf.DeltaAngle(angle, mainAngle));
            if (distance <= threshold)
            {
                return mainAngle;
            }
        }

        return angle;
    }
}