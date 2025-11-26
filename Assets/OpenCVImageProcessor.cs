using System.Collections.Generic;
using System.Collections;
using UnityEngine;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.Features2dModule;
using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.UnityUtils;
using System;
using System.Threading;
using System.Collections.Concurrent;

/// <summary>
/// Generic OpenCV-based image processor for marker tracking.
/// Accepts frames from any source (AR camera, webcam, video file, etc.)
/// and performs feature matching and pose estimation.
/// </summary>
public class OpenCVImageProcessor : MonoBehaviour
{
    [Header("Marker")]
    public Texture2D markerTexture;
    [Tooltip("Physical width of the marker in centimeters")]
    public float markerWidthCm = 10f;

    // Internal: marker width in meters (auto-converted)
    private float markerWidthMeters => markerWidthCm / 100f;

    [Header("Output")]
    public Transform targetTransform;

    [Header("Anchor Lock")]
    public bool enableAnchorLock = true;
    public int requiredConsecutiveDetections = 15;
    public float stabilityThreshold = 0.02f;

    [Header("Tracking Stabilization")]
    [Tooltip("Interpolation speed for smooth movement (higher = faster response)")]
    [Range(1f, 30f)]
    public float interpolationSpeed = 15f;
    [Tooltip("Use time-based smoothing (smoother at varying framerates)")]
    public bool useTimeBasedSmoothing = true;
    [Tooltip("Maximum position jump (meters) - larger jumps reset tracking")]
    public float maxPositionJump = 0.5f;
    [Tooltip("Maximum rotation jump (degrees) - larger jumps reset tracking")]
    public float maxRotationJump = 45f;

    [Header("Temporal Filtering")]
    [Tooltip("Enable temporal filtering to reduce jitter with fewer features")]
    public bool useTemporalFiltering = true;
    [Tooltip("Number of recent poses to average (higher = smoother but more lag)")]
    [Range(2, 10)]
    public int temporalFilterSize = 5;


    [Header("Tracking Quality")]
    [Tooltip("Minimum good matches required (higher = more stable but harder to track)")]
    [Range(15, 100)]
    public int minGoodMatches = 30;
    [Tooltip("Ratio test for feature matching (lower = stricter, more accurate)")]
    [Range(0.6f, 0.85f)]
    public float ratioTestThreshold = 0.7f;
    [Tooltip("Maximum reprojection error in pixels (lower = stricter pose validation)")]
    [Range(1f, 10f)]
    public float maxReprojectionError = 3f;

    // Private settings with default values
    private bool useORB = true;  // ORB is 10-100x faster than SIFT
    private int nFeatures = 2000;  // Fast detection with fewer features
    private bool useHomography = true;
    private float processingScale = 1.0f;  // Full resolution with ORB
    private bool flipX = false;
    private bool flipY = true;
    private bool flipZ = false;
    private bool swapYandZ = false;

    // Multi-scale ORB settings for better accuracy with fewer features
    private float orbScaleFactor = 1.2f;  // Scale factor between pyramid levels
    private int orbLevels = 8;  // Number of pyramid levels (more = better scale invariance)
    private int orbEdgeThreshold = 15;  // Edge threshold (lower = more features near edges)
    private int orbPatchSize = 31;  // Patch size for descriptor
    private int orbFastThreshold = 7;  // FAST threshold (lower = more keypoints)

    // SIFT parameters (unused when useORB = true)
    private int nOctaveLayers = 3;
    private float contrastThreshold = 0.04f;
    private float edgeThreshold = 10f;
    private float sigma = 1.6f;

    // Public read-only status
    [HideInInspector]
    public bool isLocked = true;
    [HideInInspector]
    public bool markerDetected = true;

    // Private tracking variables
    private int consecutiveDetectionCount = 0;
    private float lastPositionChange = 0f;

    // Smoothing state
    private Vector3 targetPosition;
    private Quaternion targetRotation;
    private bool isTrackingInitialized = false;

    // Temporal filtering for jitter reduction
    private Queue<Vector3> recentPositions = new Queue<Vector3>();
    private Queue<Quaternion> recentRotations = new Queue<Quaternion>();

    // OpenCV objects - reused across frames
    Mat markerGray;
    Mat markerDescriptors;
    MatOfKeyPoint markerKeypoints;
    Feature2D detector;
    DescriptorMatcher matcher;

    Mat grayMat;
    Mat scaledGrayMat;

    // Reusable Mats for ProcessFrame - avoid per-frame allocations dq
    MatOfKeyPoint frameKeypoints;
    Mat frameDescriptors;
    Mat homography;
    Mat rvec;
    Mat tvec;
    Mat rotationMatrix;
    MatOfPoint2f markerMatPts;
    MatOfPoint2f sceneMatPts;
    MatOfPoint3f objPts;
    MatOfPoint2f imgPts;
    MatOfPoint2f markerCorners;
    MatOfPoint2f sceneCorners;
    List<MatOfDMatch> knnMatches;

    // Empty mask mat (reused)
    Mat emptyMask;

    // Reusable point arrays to avoid per-frame allocation
    Point[] reusableMarkerPts;
    Point[] reusableScenePts;
    Point[] reusableCorners;
    Point[] reusableImgPts;

    // Cached marker keypoints (never change after Start)
    KeyPoint[] cachedMarkerKeypoints;

    // Temp arrays for fromArray (need exact size match)
    Point[] tempMarkerPts;
    Point[] tempScenePts;

    // Camera intrinsics (must be set by frame provider)
    Mat camMatrix;
    MatOfDouble distCoeffs;

    // Anchor locking
    private Vector3 lockedPosition;
    private Quaternion lockedRotation;
    private Vector3 previousDetectedPosition;
    private int consecutiveStableDetections;

    // Threading for background processing
    private Thread processingThread;
    private ConcurrentQueue<ProcessingJob> jobQueue = new ConcurrentQueue<ProcessingJob>();
    private volatile bool isProcessingThreadRunning = false;
    private volatile bool isProcessing = false;
    private object resultLock = new object();
    private ProcessingResult latestResult = null;

    // Reference to camera transform (set by frame provider)
    private Transform cameraTransform;

    // Structure to pass frame data to background thread
    private class ProcessingJob
    {
        public Mat frameGray;
        public float scale;
        public int width;
        public int height;
    }

    // Structure to return results from background thread
    private class ProcessingResult
    {
        public bool markerDetected;
        public Vector3 position;
        public Quaternion rotation;
        public Vector3 centerObj;
    }

    void Start()
    {
        if (markerTexture == null)
        {
            Debug.LogError("Marker texture not assigned.");
            enabled = false;
            return;
        }

        Core.setNumThreads(SystemInfo.processorCount);

        // Prepare marker
        markerGray = new Mat();
        Mat markerMat = new Mat(markerTexture.height, markerTexture.width, CvType.CV_8UC4);
        Utils.texture2DToMat(markerTexture, markerMat);
        Imgproc.cvtColor(markerMat, markerGray, Imgproc.COLOR_RGBA2GRAY);
        markerMat.Dispose();

        markerKeypoints = new MatOfKeyPoint();
        markerDescriptors = new Mat();

        // Create detector (ORB or SIFT)
        try
        {
            if (useORB)
            {
                // ORB with optimized parameters for quality with fewer features
                // Key improvements:
                // - More pyramid levels (8) for better scale invariance
                // - Lower edge threshold (15) to detect features closer to edges
                // - Lower FAST threshold (7) to get more candidate keypoints
                // - HARRIS_SCORE for better feature quality ranking
                detector = ORB.create(
                    nFeatures,              // nfeatures: 500 for speed
                    orbScaleFactor,         // scaleFactor: 1.2 for fine scale steps
                    orbLevels,              // nlevels: 8 pyramid levels for scale invariance
                    orbEdgeThreshold,       // edgeThreshold: 15 (lower = more features near edges)
                    0,                      // firstLevel
                    2,                      // WTA_K: 2 for WTA_K=2 (more discriminative)
                    ORB.HARRIS_SCORE,       // scoreType: Harris corner score (better quality)
                    orbPatchSize,           // patchSize: 31
                    orbFastThreshold        // fastThreshold: 7 (lower = more keypoints)
                );
                // Use BruteForce-Hamming for ORB (binary descriptors)
                matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
                Debug.Log($"ORB initialized with {nFeatures} features (optimized for quality, {orbLevels} pyramid levels)");
            }
            else
            {
                detector = SIFT.create(
                    nFeatures,           // nfeatures
                    nOctaveLayers,       // nOctaveLayers
                    contrastThreshold,   // contrastThreshold
                    edgeThreshold,       // edgeThreshold
                    sigma                // sigma
                );
                matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
                Debug.Log($"SIFT initialized with {nFeatures} features, {nOctaveLayers} octave layers");
            }
        }
        catch (System.SystemException e)
        {
            Debug.LogError($"Feature detector initialization failed: {e.Message}");
            enabled = false;
            return;
        }

        // Detect features in marker
        detector.detectAndCompute(markerGray, new Mat(), markerKeypoints, markerDescriptors);

        Debug.Log($"Detected {markerKeypoints.total()} keypoints in marker image");

        // Camera intrinsics (will be set by frame provider)
        camMatrix = new Mat(3, 3, CvType.CV_64F);
        distCoeffs = new MatOfDouble(0.0, 0.0, 0.0, 0.0);

        // Pre-allocate reusable Mats
        frameKeypoints = new MatOfKeyPoint();
        frameDescriptors = new Mat();
        homography = new Mat();
        rvec = new Mat();
        tvec = new Mat();
        rotationMatrix = new Mat();
        markerMatPts = new MatOfPoint2f();
        sceneMatPts = new MatOfPoint2f();
        markerCorners = new MatOfPoint2f();
        sceneCorners = new MatOfPoint2f();
        knnMatches = new List<MatOfDMatch>();

        // Pre-allocate Mats for pose estimation (will be resized as needed)
        objPts = new MatOfPoint3f();
        imgPts = new MatOfPoint2f();

        // Pre-allocate empty mask (reused for detectAndCompute)
        emptyMask = new Mat();

        // Pre-allocate corner arrays (fixed size of 4)
        reusableCorners = new Point[4];
        reusableImgPts = new Point[4];
        for (int i = 0; i < 4; i++)
        {
            reusableCorners[i] = new Point();
            reusableImgPts[i] = new Point();
        }

        // Start background processing thread
        isProcessingThreadRunning = true;
        processingThread = new Thread(ProcessingThreadLoop);
        processingThread.IsBackground = true;
        processingThread.Start();
    }

    void OnDestroy()
    {
        // Stop background thread
        isProcessingThreadRunning = false;
        if (processingThread != null && processingThread.IsAlive)
        {
            processingThread.Join(1000); // Wait up to 1 second
        }

        // Clear job queue and dispose any remaining jobs
        while (jobQueue.TryDequeue(out ProcessingJob job))
        {
            job.frameGray?.Dispose();
        }

        // Clean up all Mats
        markerGray?.Dispose();
        markerDescriptors?.Dispose();
        markerKeypoints?.Dispose();
        grayMat?.Dispose();
        scaledGrayMat?.Dispose();
        camMatrix?.Dispose();
        distCoeffs?.Dispose();
        frameKeypoints?.Dispose();
        frameDescriptors?.Dispose();
        homography?.Dispose();
        rvec?.Dispose();
        tvec?.Dispose();
        rotationMatrix?.Dispose();
        markerMatPts?.Dispose();
        sceneMatPts?.Dispose();
        objPts?.Dispose();
        imgPts?.Dispose();
        markerCorners?.Dispose();
        sceneCorners?.Dispose();
        emptyMask?.Dispose();

        // Dispose knnMatches
        if (knnMatches != null)
        {
            foreach (var m in knnMatches)
                m?.Dispose();
        }
    }

    /// <summary>
    /// Set the camera transform reference (called by frame provider)
    /// </summary>
    public void SetCameraTransform(Transform camTransform)
    {
        cameraTransform = camTransform;
    }

    /// <summary>
    /// Update camera intrinsics matrix (called by frame provider)
    /// </summary>
    public void UpdateCameraIntrinsics(float fx, float fy, float cx, float cy)
    {
        camMatrix.put(0, 0, fx);
        camMatrix.put(0, 1, 0);
        camMatrix.put(0, 2, cx);
        camMatrix.put(1, 0, 0);
        camMatrix.put(1, 1, fy);
        camMatrix.put(1, 2, cy);
        camMatrix.put(2, 0, 0);
        camMatrix.put(2, 1, 0);
        camMatrix.put(2, 2, 1);
    }

    /// <summary>
    /// Process a frame (Mat). This is the main entry point for frame providers.
    /// </summary>
    public void ProcessFrame(Mat frameRGBA)
    {
        // Skip if already processing (prevents queue buildup)
        if (isProcessing)
            return;

        if (frameRGBA == null || frameRGBA.empty())
            return;

        int width = frameRGBA.cols();
        int height = frameRGBA.rows();

        // Reuse grayMat
        if (grayMat == null || grayMat.cols() != width || grayMat.rows() != height)
        {
            grayMat?.Dispose();
            grayMat = new Mat(height, width, CvType.CV_8UC1);
        }

        Imgproc.cvtColor(frameRGBA, grayMat, Imgproc.COLOR_RGBA2GRAY);

        ProcessGrayscaleFrame(grayMat, width, height);
    }

    /// <summary>
    /// Internal method to process a grayscale frame
    /// </summary>
    private void ProcessGrayscaleFrame(Mat grayFrame, int originalWidth, int originalHeight)
    {
        // Downscale for processing if scale < 1
        Mat processedGray;
        float actualScale = 1f;

        if (processingScale < 1f)
        {
            int scaledWidth = (int)(originalWidth * processingScale);
            int scaledHeight = (int)(originalHeight * processingScale);

            if (scaledGrayMat == null || scaledGrayMat.cols() != scaledWidth || scaledGrayMat.rows() != scaledHeight)
            {
                scaledGrayMat?.Dispose();
                scaledGrayMat = new Mat(scaledHeight, scaledWidth, CvType.CV_8UC1);
            }

            Imgproc.resize(grayFrame, scaledGrayMat, new Size(scaledWidth, scaledHeight));
            processedGray = scaledGrayMat;
            actualScale = processingScale;
        }
        else
        {
            processedGray = grayFrame;
        }

        // Clone the gray mat for background thread (thread-safe)
        Mat clonedGray = processedGray.clone();

        // Queue job for background processing
        ProcessingJob job = new ProcessingJob
        {
            frameGray = clonedGray,
            scale = actualScale,
            width = originalWidth,
            height = originalHeight
        };

        // Clear old jobs from queue (keep only latest)
        while (jobQueue.Count > 0)
        {
            if (jobQueue.TryDequeue(out ProcessingJob oldJob))
            {
                oldJob.frameGray?.Dispose();
            }
        }

        jobQueue.Enqueue(job);
    }


    ProcessingResult ProcessFrameOnBackgroundThread(ProcessingJob job)
    {
        Mat frameGray = job.frameGray;
        float scale = job.scale;

        if (isLocked)
        {
            return new ProcessingResult { markerDetected = false };
        }

        // Reuse frameKeypoints and frameDescriptors
        frameKeypoints.release();
        frameDescriptors.release();

        // Detect features in frame
        detector.detectAndCompute(frameGray, emptyMask, frameKeypoints, frameDescriptors);

        if (frameDescriptors.empty() || markerDescriptors.empty())
        {
            return new ProcessingResult { markerDetected = false };
        }

        // Clear and reuse knnMatches list
        for (int i = 0; i < knnMatches.Count; i++)
            knnMatches[i]?.Dispose();
        knnMatches.Clear();

        try
        {
            matcher.knnMatch(markerDescriptors, frameDescriptors, knnMatches, 2);
        }
        catch (System.Exception e)
        {
            Debug.LogError("Matcher failed: " + e.Message);
            return new ProcessingResult { markerDetected = false };
        }

        // Count good matches first without allocating (use configurable ratio test)
        int goodMatchCount = 0;
        for (int i = 0; i < knnMatches.Count; i++)
        {
            MatOfDMatch m = knnMatches[i];
            if (m.rows() >= 2)
            {
                float dist0 = (float)m.get(0, 0)[3]; // distance is 4th element
                float dist1 = (float)m.get(1, 0)[3];
                if (dist0 < ratioTestThreshold * dist1)
                    goodMatchCount++;
            }
        }

        if (goodMatchCount < minGoodMatches)
        {
            return new ProcessingResult { markerDetected = false };
        }

        // Ensure arrays are large enough
        EnsurePointArrayCapacity(goodMatchCount);

        // Get keypoint data - cache marker keypoints since they don't change
        if (cachedMarkerKeypoints == null)
        {
            cachedMarkerKeypoints = markerKeypoints.toArray();
        }

        KeyPoint[] fkpts = frameKeypoints.toArray();

        float invScale = 1f / scale;
        int idx = 0;

        for (int i = 0; i < knnMatches.Count; i++)
        {
            MatOfDMatch m = knnMatches[i];
            if (m.rows() >= 2)
            {
                // DMatch structure: queryIdx(0), trainIdx(1), imgIdx(2), distance(3)
                double[] match0 = m.get(0, 0);
                double[] match1 = m.get(1, 0);

                float dist0 = (float)match0[3];
                float dist1 = (float)match1[3];

                if (dist0 < ratioTestThreshold * dist1)
                {
                    int queryIdx = (int)match0[0];
                    int trainIdx = (int)match0[1];

                    // Reuse Point objects
                    reusableMarkerPts[idx].x = cachedMarkerKeypoints[queryIdx].pt.x;
                    reusableMarkerPts[idx].y = cachedMarkerKeypoints[queryIdx].pt.y;

                    reusableScenePts[idx].x = fkpts[trainIdx].pt.x * invScale;
                    reusableScenePts[idx].y = fkpts[trainIdx].pt.y * invScale;

                    idx++;
                }
            }
        }

        // Use pre-allocated arrays - need to create correctly sized array for fromArray
        if (tempMarkerPts == null || tempMarkerPts.Length != idx)
        {
            tempMarkerPts = new Point[idx];
            tempScenePts = new Point[idx];
        }
        Array.Copy(reusableMarkerPts, tempMarkerPts, idx);
        Array.Copy(reusableScenePts, tempScenePts, idx);

        markerMatPts.fromArray(tempMarkerPts);
        sceneMatPts.fromArray(tempScenePts);

        double markerWidth = markerGray.cols();
        double markerHeight = markerGray.rows();
        double scaleX = markerWidthMeters / markerWidth;
        double scaleY = markerWidthMeters / markerHeight;

        rvec.release();
        tvec.release();

        if (useHomography)
        {
            // HOMOGRAPHY METHOD: Find homography and compute corners
            // Maximum accuracy RANSAC: tighter threshold (2.0), more iterations (5000), higher confidence (0.999)
            homography.release();
            Mat mask = new Mat();
            homography = Calib3d.findHomography(markerMatPts, sceneMatPts, Calib3d.RANSAC, 2.0, mask, 5000, 0.999);
            mask.Dispose();

            if (homography.empty())
            {
                return new ProcessingResult { markerDetected = false };
            }

            // Define marker corners
            reusableCorners[0].x = 0;
            reusableCorners[0].y = 0;
            reusableCorners[1].x = markerWidth;
            reusableCorners[1].y = 0;
            reusableCorners[2].x = markerWidth;
            reusableCorners[2].y = markerHeight;
            reusableCorners[3].x = 0;
            reusableCorners[3].y = markerHeight;

            markerCorners.fromArray(reusableCorners);
            sceneCorners.release();

            Core.perspectiveTransform(markerCorners, sceneCorners, homography);
            Point[] sceneCornersPts = sceneCorners.toArray();

            // Build 3D object points for marker corners
            Point3[] objCorners3D = new Point3[4];
            objCorners3D[0] = new Point3(0, 0, 0);
            objCorners3D[1] = new Point3(markerWidthMeters, 0, 0);
            objCorners3D[2] = new Point3(markerWidthMeters, markerWidthMeters, 0);
            objCorners3D[3] = new Point3(0, markerWidthMeters, 0);

            objPts.fromArray(objCorners3D);

            // Copy scene corners to reusableImgPts
            for (int i = 0; i < 4; i++)
            {
                reusableImgPts[i].x = sceneCornersPts[i].x;
                reusableImgPts[i].y = sceneCornersPts[i].y;
            }
            imgPts.fromArray(reusableImgPts);

            bool solved = Calib3d.solvePnP(objPts, imgPts, camMatrix, distCoeffs, rvec, tvec);
            if (!solved)
            {
                return new ProcessingResult { markerDetected = false };
            }
        }
        else
        {
            // DIRECT SOLVEPNP METHOD: Use all matched keypoints directly
            // Create 3D object points from 2D marker keypoints (assuming planar marker at Z=0)
            Point3[] objPoints3D = new Point3[idx];
            Point[] imgPoints2D = new Point[idx];

            for (int i = 0; i < idx; i++)
            {
                // Convert marker 2D points to 3D object space
                objPoints3D[i] = new Point3(
                    tempMarkerPts[i].x * scaleX,
                    tempMarkerPts[i].y * scaleY,
                    0
                );
                imgPoints2D[i] = tempScenePts[i];
            }

            objPts.fromArray(objPoints3D);
            imgPts.fromArray(imgPoints2D);

            // Use RANSAC with configurable reprojection error threshold
            Mat inliersMask = new Mat();
            bool solved = Calib3d.solvePnPRansac(objPts, imgPts, camMatrix, distCoeffs, rvec, tvec, false, 500, maxReprojectionError, 0.999, inliersMask);

            if (!solved)
            {
                inliersMask.Dispose();
                return new ProcessingResult { markerDetected = false };
            }

            // Count inliers and reject if too few
            int inlierCount = Core.countNonZero(inliersMask);
            float inlierRatio = (float)inlierCount / idx;
            inliersMask.Dispose();

            if (inlierRatio < 0.5f) // At least 50% inliers required
            {
                return new ProcessingResult { markerDetected = false };
            }

            // Refine pose with iterative optimization
            Calib3d.solvePnPRefineLM(objPts, imgPts, camMatrix, distCoeffs, rvec, tvec);
        }

        rotationMatrix.release();
        Calib3d.Rodrigues(rvec, rotationMatrix);

        // Read rotation matrix directly
        double r00 = rotationMatrix.get(0, 0)[0];
        double r01 = rotationMatrix.get(0, 1)[0];
        double r02 = rotationMatrix.get(0, 2)[0];
        double r10 = rotationMatrix.get(1, 0)[0];
        double r11 = rotationMatrix.get(1, 1)[0];
        double r12 = rotationMatrix.get(1, 2)[0];
        double r20 = rotationMatrix.get(2, 0)[0];
        double r21 = rotationMatrix.get(2, 1)[0];
        double r22 = rotationMatrix.get(2, 2)[0];

        double tx = tvec.get(0, 0)[0];
        double ty = tvec.get(1, 0)[0];
        double tz = tvec.get(2, 0)[0];

        // Validate pose: check for reasonable translation values
        double distanceFromCamera = Math.Sqrt(tx * tx + ty * ty + tz * tz);
        if (distanceFromCamera < 0.1 || distanceFromCamera > 10.0) // 10cm to 10m range
        {
            return new ProcessingResult { markerDetected = false };
        }

        float sx = flipX ? -1f : 1f;
        float sy = flipY ? -1f : 1f;
        float sz = flipZ ? -1f : 1f;

        Vector3 position = new Vector3((float)tx * sx, (float)ty * sy, (float)tz * sz);

        Vector3 col0 = new Vector3((float)r00, (float)r10, (float)r20);
        Vector3 col1 = new Vector3((float)r01, (float)r11, (float)r21);
        Vector3 col2 = new Vector3((float)r02, (float)r12, (float)r22);

        col0 = Vector3.Scale(col0, new Vector3(sx, sy, sz));
        col1 = Vector3.Scale(col1, new Vector3(sx, sy, sz));
        col2 = Vector3.Scale(col2, new Vector3(sx, sy, sz));

        if (swapYandZ)
        {
            col0 = new Vector3(col0.x, col0.z, col0.y);
            col1 = new Vector3(col1.x, col1.z, col1.y);
            col2 = new Vector3(col2.x, col2.z, col2.y);
            position = new Vector3(position.x, position.z, position.y);
        }

        Matrix4x4 unityM = new Matrix4x4();
        unityM.SetColumn(0, new Vector4(col0.x, col0.y, col0.z, 0));
        unityM.SetColumn(1, new Vector4(col1.x, col1.y, col1.z, 0));
        unityM.SetColumn(2, new Vector4(col2.x, col2.y, col2.z, 0));
        unityM.SetColumn(3, new Vector4(position.x, position.y, position.z, 1));

        Quaternion rot = QuaternionFromMatrix(unityM);

        Vector3 centerObj = new Vector3(markerWidthMeters * 0.5f, markerWidthMeters * 0.5f, 0f);
        Vector3 camLocalPos = unityM.MultiplyPoint(centerObj);

        return new ProcessingResult
        {
            markerDetected = true,
            position = camLocalPos,
            rotation = rot,
            centerObj = centerObj
        };
    }

    void EnsurePointArrayCapacity(int needed)
    {
        if (reusableMarkerPts == null || reusableMarkerPts.Length < needed)
        {
            reusableMarkerPts = new Point[needed + 50];
            for (int i = 0; i < reusableMarkerPts.Length; i++)
                reusableMarkerPts[i] = new Point();
        }
        if (reusableScenePts == null || reusableScenePts.Length < needed)
        {
            reusableScenePts = new Point[needed + 50];
            for (int i = 0; i < reusableScenePts.Length; i++)
                reusableScenePts[i] = new Point();
        }
    }

    void LockAnchor()
    {
        isLocked = true;
        lockedPosition = targetTransform.position;
        lockedRotation = targetTransform.rotation;
        Debug.Log($"<color=green>Anchor LOCKED at: {lockedPosition}</color>");
    }

    public void UnlockAnchor()
    {
        isLocked = false;
        consecutiveStableDetections = 0;
        consecutiveDetectionCount = 0;
        ResetTracking();
        Debug.Log("<color=yellow>Anchor UNLOCKED</color>");
    }

    void ResetTracking()
    {
        isTrackingInitialized = false;
        targetPosition = Vector3.zero;
        targetRotation = Quaternion.identity;

        // Clear temporal filter buffers
        recentPositions.Clear();
        recentRotations.Clear();
    }

    void Update()
    {
        if (isLocked && targetTransform != null)
        {
            targetTransform.position = lockedPosition;
            targetTransform.rotation = lockedRotation;
            return;
        }

        // Apply results from background thread on main thread
        ProcessingResult result = null;
        lock (resultLock)
        {
            if (latestResult != null)
            {
                result = latestResult;
                latestResult = null;
            }
        }

        // Update target position/rotation from detection results
        if (result != null)
        {
            markerDetected = result.markerDetected;

            if (result.markerDetected && cameraTransform != null)
            {
                Vector3 camLocalPos = result.position;
                Vector3 worldPos = cameraTransform.TransformPoint(camLocalPos);
                Quaternion worldRot = cameraTransform.rotation * result.rotation;
                Quaternion finalRotation = worldRot * Quaternion.Euler(90f, 180f, 0f);

                // Check for tracking loss (large jumps indicate marker moved or tracking error)
                if (isTrackingInitialized)
                {
                    float positionDelta = Vector3.Distance(worldPos, targetPosition);
                    float rotationDelta = Quaternion.Angle(finalRotation, targetRotation);

                    if (positionDelta > maxPositionJump || rotationDelta > maxRotationJump)
                    {
                        // Large jump detected - reset tracking
                        Debug.LogWarning($"Tracking lost: position delta={positionDelta:F3}m, rotation delta={rotationDelta:F1}°");
                        ResetTracking();
                        isTrackingInitialized = false;
                        consecutiveDetectionCount = 0;
                        consecutiveStableDetections = 0;
                    }
                }

                // Update target (what we lerp towards)
                if (!isTrackingInitialized)
                {
                    // First detection - initialize directly
                    targetPosition = worldPos;
                    targetRotation = finalRotation;
                    isTrackingInitialized = true;

                    // Initialize transform immediately
                    if (targetTransform != null)
                    {
                        targetTransform.position = targetPosition;
                        targetTransform.rotation = targetRotation;
                    }
                }
                else
                {
                    // Apply temporal filtering to reduce jitter with fewer features
                    if (useTemporalFiltering)
                    {
                        // Add new pose to history
                        recentPositions.Enqueue(worldPos);
                        recentRotations.Enqueue(finalRotation);

                        // Keep only recent N poses
                        while (recentPositions.Count > temporalFilterSize)
                        {
                            recentPositions.Dequeue();
                            recentRotations.Dequeue();
                        }

                        // Average recent positions
                        Vector3 avgPosition = Vector3.zero;
                        foreach (var pos in recentPositions)
                            avgPosition += pos;
                        avgPosition /= recentPositions.Count;

                        // Average recent rotations using quaternion averaging
                        Quaternion avgRotation = AverageQuaternions(recentRotations);

                        targetPosition = avgPosition;
                        targetRotation = avgRotation;
                    }
                    else
                    {
                        // No filtering, use raw detection
                        targetPosition = worldPos;
                        targetRotation = finalRotation;
                    }
                }

                // Handle anchor locking with simple consecutive detection counting
                if (enableAnchorLock && !isLocked)
                {
                    consecutiveDetectionCount++;

                    // Check stability by comparing with previous position
                    if (consecutiveDetectionCount > 1)
                    {
                        lastPositionChange = Vector3.Distance(targetPosition, previousDetectedPosition);

                        if (lastPositionChange < stabilityThreshold)
                        {
                            consecutiveStableDetections++;

                            if (consecutiveStableDetections >= requiredConsecutiveDetections)
                            {
                                LockAnchor();
                            }
                        }
                        else
                        {
                            // Position changed too much, reset counter
                            consecutiveStableDetections = 0;
                        }
                    }

                    previousDetectedPosition = targetPosition;
                }
            }
            else
            {
                // Detection lost, reset counters and tracking
                if (!isLocked)
                {
                    consecutiveDetectionCount = 0;
                    consecutiveStableDetections = 0;
                    ResetTracking();
                }
            }
        }

        // Continuously smooth towards target every frame
        if (isTrackingInitialized && targetTransform != null)
        {
            // Smooth using interpolationSpeed (time-based)
            float t = 1f - Mathf.Exp(-interpolationSpeed * Time.deltaTime);
            targetTransform.position = Vector3.Lerp(targetTransform.position, targetPosition, t);
            targetTransform.rotation = Quaternion.Slerp(targetTransform.rotation, targetRotation, t);

            targetTransform.localScale = new Vector3(markerWidthMeters, targetTransform.localScale.y, markerWidthMeters);
        }
    }

    /// <summary>
    /// Average multiple quaternions with equal weights
    /// </summary>
    Quaternion AverageQuaternions(Queue<Quaternion> quaternions)
    {
        if (quaternions.Count == 0)
            return Quaternion.identity;

        if (quaternions.Count == 1)
            return quaternions.Peek();

        // Use the first quaternion as reference
        Quaternion first = quaternions.Peek();
        float weight = 1.0f / quaternions.Count;

        Vector4 sum = new Vector4(0, 0, 0, 0);
        foreach (Quaternion q in quaternions)
        {
            // Ensure quaternions are on the same hemisphere
            Quaternion aligned = q;
            if (Quaternion.Dot(first, q) < 0)
            {
                aligned.x = -q.x;
                aligned.y = -q.y;
                aligned.z = -q.z;
                aligned.w = -q.w;
            }

            sum.x += aligned.x * weight;
            sum.y += aligned.y * weight;
            sum.z += aligned.z * weight;
            sum.w += aligned.w * weight;
        }

        // Normalize
        float magnitude = Mathf.Sqrt(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z + sum.w * sum.w);
        if (magnitude > 0.0001f)
            sum /= magnitude;

        return new Quaternion(sum.x, sum.y, sum.z, sum.w);
    }

    void ProcessingThreadLoop()
    {
        while (isProcessingThreadRunning)
        {
            if (jobQueue.TryDequeue(out ProcessingJob job))
            {
                isProcessing = true;
                ProcessingResult result = ProcessFrameOnBackgroundThread(job);
                job.frameGray?.Dispose();

                lock (resultLock)
                {
                    latestResult = result;
                }

                isProcessing = false;
            }
            else
            {
                Thread.Sleep(1); // Small sleep to prevent busy-waiting
            }
        }
    }

    static Quaternion QuaternionFromMatrix(Matrix4x4 m)
    {
        Vector3 forward = m.GetColumn(2);
        Vector3 up = m.GetColumn(1);
        if (forward.sqrMagnitude < 1e-8)
            return Quaternion.identity;
        return Quaternion.LookRotation(forward, up);
    }
}
