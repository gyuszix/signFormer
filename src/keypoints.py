"""
src/keypoints.py

Shared MediaPipe hand keypoint extraction utilities.
Used by scripts/preprocess.py (offline, per-video) and src/demo.py (live, per-frame).
"""

import numpy as np
from pathlib import Path


def get_hand_detector():
    """Create and return a MediaPipe HandLandmarker detector (Tasks API)."""
    import urllib.request
    import mediapipe as mp

    model_path = Path("data/hand_landmarker.task")
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print("Downloading MediaPipe hand model...")
        import ssl
        try:
            import certifi
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ssl_ctx) as r, \
             open(str(model_path), "wb") as f:
            f.write(r.read())
        print("Done.")

    BaseOptions           = mp.tasks.BaseOptions
    HandLandmarker        = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode     = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def extract_keypoints_from_frame(rgb: np.ndarray, detector) -> tuple[np.ndarray, object]:
    """
    Extract 126-dim keypoint vector from a single RGB frame.

    Args:
        rgb:      (H, W, 3) uint8 RGB array
        detector: MediaPipe HandLandmarker instance (from get_hand_detector())

    Returns:
        kpts:   (126,) float32 array — [left_hand(63), right_hand(63)]
        result: raw MediaPipe detection result (for drawing skeleton in demo)
    """
    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    lh = np.zeros(63, dtype=np.float32)
    rh = np.zeros(63, dtype=np.float32)
    for i, hand_landmarks in enumerate(result.hand_landmarks):
        handedness = result.handedness[i][0].category_name
        coords = np.array([[lm.x, lm.y, lm.z]
                            for lm in hand_landmarks],
                           dtype=np.float32).flatten()
        if handedness == "Left":
            lh = coords
        else:
            rh = coords

    return np.concatenate([lh, rh]), result


# ── Holistic (hands + full body pose) ─────────────────────────────────────────

def get_holistic_detector():
    """
    Create a MediaPipe Holistic detector.
    Returns pose + hand landmarks in a single call — used for holistic preprocessing.
    Requires mediapipe >= 0.10.0 (legacy solutions API, still supported).
    """
    import mediapipe as mp
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_holistic_from_frame(rgb: np.ndarray, detector) -> np.ndarray:
    """
    Extract 225-dim holistic keypoint vector from a single RGB frame.

    Layout: [left_hand(63), right_hand(63), pose(99)]
      - Hands: 21 landmarks × 3 (x, y, z), zeros if not detected
      - Pose:  33 landmarks × 3 (x, y, z), zeros if not detected

    Args:
        rgb:      (H, W, 3) uint8 RGB array
        detector: MediaPipe Holistic instance (from get_holistic_detector())

    Returns:
        kpts: (225,) float32 array
    """
    from src.config import HOLISTIC_DIM
    result = detector.process(rgb)

    lh   = np.zeros(63,  dtype=np.float32)
    rh   = np.zeros(63,  dtype=np.float32)
    pose = np.zeros(99,  dtype=np.float32)

    if result.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                        for lm in result.left_hand_landmarks.landmark],
                       dtype=np.float32).flatten()

    if result.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                        for lm in result.right_hand_landmarks.landmark],
                       dtype=np.float32).flatten()

    if result.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                          for lm in result.pose_landmarks.landmark],
                         dtype=np.float32).flatten()

    return np.concatenate([lh, rh, pose])
