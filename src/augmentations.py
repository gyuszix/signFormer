"""
src/augmentations.py

Keypoint augmentation transforms for training.
Called by WLASLDataset — not a runtime concern of the dataloader itself.

Supports two input formats:
  - Hand-only (126-dim): [lh(63), rh(63)]
  - Holistic  (225-dim): [lh(63), rh(63), pose(99)]
"""

import numpy as np
from src.config import KEYPOINT_DIM, HOLISTIC_DIM, POSE_FLIP_PAIRS


def augment_keypoints(kpts: np.ndarray, training: bool = True) -> np.ndarray:
    """
    Apply training augmentations to a (T, 126) keypoint array.

    Augmentations (training only):
        1. Horizontal flip    — swap left/right hands (doubles effective dataset)
        2. Speed perturbation — resample sequence at 0.8x–1.2x speed
        3. Temporal jitter    — randomly drop or repeat individual frames
        4. Scale augmentation — randomly scale keypoints 0.85x–1.15x (simulates hand size variation)
        5. Gaussian noise     — simulate MediaPipe detection noise

    Always applied (train + val/test):
        5. Wrist-relative normalization — translation invariance
        6. Scale normalization          — hand-span normalization for scale invariance
    """
    if not training:
        return normalize_keypoints(kpts)

    if np.random.rand() < 0.5:
        kpts = flip_keypoints(kpts)

    kpts = speed_perturb(kpts, min_rate=0.8, max_rate=1.2)
    kpts = temporal_jitter(kpts, jitter_prob=0.1)
    kpts = scale_augment(kpts, min_scale=0.85, max_scale=1.15)
    kpts = kpts + np.random.randn(*kpts.shape).astype(np.float32) * 0.01
    kpts = normalize_keypoints(kpts)

    return kpts


def flip_keypoints(kpts: np.ndarray) -> np.ndarray:
    """
    Mirror left/right: swap hands and flip x coordinates.
    Handles both 126-dim (hands only) and 225-dim (hands + pose) input.
    """
    flipped = kpts.copy()
    lh = kpts[:, :63].copy()
    rh = kpts[:, 63:126].copy()

    # Flip x for both hands, then swap
    rh[:, 0::3] = 1.0 - rh[:, 0::3]
    lh[:, 0::3] = 1.0 - lh[:, 0::3]
    flipped[:, :63]  = rh
    flipped[:, 63:126] = lh

    # Holistic: also flip pose landmarks
    if kpts.shape[1] == HOLISTIC_DIM:
        pose = flipped[:, 126:].copy()   # (T, 99)
        pose[:, 0::3] = 1.0 - pose[:, 0::3]   # flip x
        # Swap left-right landmark pairs
        for l_idx, r_idx in POSE_FLIP_PAIRS:
            l_off, r_off = l_idx * 3, r_idx * 3
            tmp = pose[:, l_off:l_off+3].copy()
            pose[:, l_off:l_off+3] = pose[:, r_off:r_off+3]
            pose[:, r_off:r_off+3] = tmp
        flipped[:, 126:] = pose

    return flipped


def speed_perturb(kpts: np.ndarray, min_rate: float = 0.8, max_rate: float = 1.2) -> np.ndarray:
    """
    Resample the sequence at a random speed between min_rate and max_rate.

    rate < 1.0 → slower signing (more frames, sequence stretched)
    rate > 1.0 → faster signing (fewer frames, sequence compressed)

    Uses linear interpolation between frames so keypoints stay smooth.
    """
    T = kpts.shape[0]
    if T < 2:
        return kpts

    rate        = np.random.uniform(min_rate, max_rate)
    new_T       = max(2, int(round(T / rate)))
    old_indices = np.linspace(0, T - 1, new_T)
    new_kpts    = np.zeros((new_T, kpts.shape[1]), dtype=np.float32)

    for i, idx in enumerate(old_indices):
        lo  = int(idx)
        hi  = min(lo + 1, T - 1)
        frac = idx - lo
        new_kpts[i] = (1 - frac) * kpts[lo] + frac * kpts[hi]

    return new_kpts


def temporal_jitter(kpts: np.ndarray, jitter_prob: float = 0.1) -> np.ndarray:
    """Randomly drop or repeat individual frames."""
    T = kpts.shape[0]
    result = []
    for t in range(T):
        r = np.random.rand()
        if r < jitter_prob / 2 and len(result) > 0:
            continue
        elif r < jitter_prob and t > 0:
            result.append(kpts[t - 1])
        result.append(kpts[t])
    return np.stack(result) if result else kpts


def scale_augment(kpts: np.ndarray, min_scale: float = 0.85, max_scale: float = 1.15) -> np.ndarray:
    """
    Randomly scale all keypoint coordinates by a uniform factor.
    Simulates signers with different hand sizes or at different distances from the camera.
    Applied before normalize_keypoints so the scale is absorbed by normalization.
    """
    scale = np.random.uniform(min_scale, max_scale)
    return (kpts * scale).astype(np.float32)


def normalize_keypoints(kpts: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints for translation and scale invariance.
    Handles both 126-dim (hands only) and 225-dim (hands + pose) input.

    Hands (indices 0–125):
      Step 1 — Translation: subtract dominant wrist per frame.
               Uses right wrist if detected, falls back to left wrist.
      Step 2 — Scale: divide by mean wrist-to-middle-fingertip span.

    Pose (indices 126–224, holistic only):
      Normalized relative to the midpoint of the two hip landmarks (23, 24)
      and scaled by the shoulder width (distance between landmarks 11, 12).
    """
    normalized = kpts.copy()
    dim        = kpts.shape[1]

    # ── Hand normalization ────────────────────────────────────────────────────
    rh_wrist    = kpts[:, 63:66]   # (T, 3)
    lh_wrist    = kpts[:, 0:3]     # (T, 3)
    rh_detected = (rh_wrist != 0).any(axis=1)
    lh_detected = (lh_wrist != 0).any(axis=1)

    wrist = np.zeros((kpts.shape[0], 3), dtype=np.float32)
    wrist[rh_detected]                 = rh_wrist[rh_detected]
    wrist[~rh_detected & lh_detected]  = lh_wrist[~rh_detected & lh_detected]

    # Step 1: translate hands
    for i in range(0, 126, 3):
        normalized[:, i]     -= wrist[:, 0]
        normalized[:, i + 1] -= wrist[:, 1]
        normalized[:, i + 2] -= wrist[:, 2]

    # Step 2: scale by hand span
    rh_midtip = normalized[:, 99:102]
    lh_midtip = normalized[:, 36:39]
    rh_span   = np.linalg.norm(rh_midtip - normalized[:, 63:66], axis=1)
    lh_span   = np.linalg.norm(lh_midtip - normalized[:, 0:3],   axis=1)
    span      = np.where(rh_detected, rh_span, lh_span)
    mean_span = span[span > 1e-4].mean() if (span > 1e-4).any() else 1.0

    normalized[:, :126] = normalized[:, :126] / mean_span

    # ── Pose normalization (holistic only) ────────────────────────────────────
    if dim == HOLISTIC_DIM:
        pose = normalized[:, 126:].reshape(-1, 33, 3)  # (T, 33, 3)

        # Hip midpoint as translation reference (landmarks 23 + 24)
        hip_mid    = (pose[:, 23, :] + pose[:, 24, :]) / 2.0  # (T, 3)
        pose_detected = (pose[:, 23, :] != 0).any(axis=1) | (pose[:, 24, :] != 0).any(axis=1)

        pose[pose_detected] -= hip_mid[pose_detected, np.newaxis, :]

        # Shoulder width as scale reference (landmarks 11 + 12)
        shoulder_w = np.linalg.norm(pose[:, 11, :] - pose[:, 12, :], axis=1)
        mean_sw    = shoulder_w[shoulder_w > 1e-4].mean() if (shoulder_w > 1e-4).any() else 1.0

        pose = pose / mean_sw
        normalized[:, 126:] = pose.reshape(-1, 99)

    return normalized
