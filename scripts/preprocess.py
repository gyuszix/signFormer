"""
scripts/preprocess.py

One-time offline preprocessing: extract MediaPipe hand keypoints from raw
WLASL videos and save as .npy files with a split manifest. Run from project root:

    python scripts/preprocess.py --split train --vocab 2000
    python scripts/preprocess.py --split val   --vocab 2000
    python scripts/preprocess.py --split test  --vocab 2000
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Allow running as `python scripts/preprocess.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.config import DATA_RAW_DIR, DATA_PROC_DIR, KEYPOINT_DIM, HOLISTIC_DIM
from src.keypoints import (get_hand_detector, extract_keypoints_from_frame,
                            get_holistic_detector, extract_holistic_from_frame)


def extract_keypoints_from_video(video_path: str, detector, holistic: bool = False) -> np.ndarray:
    """Run keypoint extraction on every frame of a video.

    Returns:
        (T, 126) array  — hand-only mode (holistic=False)
        (T, 225) array  — holistic mode  (holistic=True)
    """
    dim         = HOLISTIC_DIM if holistic else KEYPOINT_DIM
    cap         = cv2.VideoCapture(video_path)
    frames_kpts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if holistic:
            kpts = extract_holistic_from_frame(rgb, detector)
        else:
            kpts, _ = extract_keypoints_from_frame(rgb, detector)
        frames_kpts.append(kpts)

    cap.release()
    return np.stack(frames_kpts) if frames_kpts else np.zeros((1, dim), dtype=np.float32)


def preprocess_dataset(split: str = "train", vocab_size: int = 100, holistic: bool = False):
    """Extract keypoints for all videos in a split and save as .npy files.

    If holistic=True, saves 225-dim keypoints to data/processed/{split}_holistic/
    and writes {split}_holistic_manifest.json. Existing hand-only files are untouched.
    """
    suffix   = "_holistic" if holistic else ""
    proc_dir = DATA_PROC_DIR / f"{split}{suffix}"
    proc_dir.mkdir(parents=True, exist_ok=True)

    anno_path = DATA_RAW_DIR / "WLASL_v0.3.json"
    if not anno_path.exists():
        raise FileNotFoundError(f"WLASL annotation not found at {anno_path}")

    with open(anno_path) as f:
        data = json.load(f)

    # Build vocabulary — top vocab_size most frequent signs across all splits
    sign_counts = {}
    for entry in data:
        for inst in entry["instances"]:
            sign_counts[entry["gloss"]] = sign_counts.get(entry["gloss"], 0) + 1

    top_signs = sorted(sign_counts, key=sign_counts.get, reverse=True)[:vocab_size]
    vocab     = {sign: idx for idx, sign in enumerate(top_signs)}

    with open(DATA_PROC_DIR / "vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary: {len(vocab)} signs")

    detector = get_holistic_detector() if holistic else get_hand_detector()
    manifest = []

    for entry in tqdm(data, desc=f"Extracting {split}{' (holistic)' if holistic else ''}"):
        gloss = entry["gloss"]
        if gloss not in vocab:
            continue
        label_idx = vocab[gloss]

        for inst in entry["instances"]:
            if inst["split"] != split:
                continue
            video_id   = inst["video_id"]
            video_path = str(DATA_RAW_DIR / "videos" / f"{video_id}.mp4")
            if not os.path.exists(video_path):
                continue

            save_path = proc_dir / f"{video_id}.npy"
            if not save_path.exists():
                kpts = extract_keypoints_from_video(video_path, detector, holistic=holistic)
                np.save(str(save_path), kpts)

            manifest.append({
                "path":      str(save_path),
                "label":     gloss,
                "label_idx": label_idx,
            })

    if holistic:
        detector.close()
    else:
        detector.close()

    manifest_name = f"{split}{suffix}_manifest.json"
    with open(DATA_PROC_DIR / manifest_name, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"{split}{suffix}: {len(manifest)} samples saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    type=str,  default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--vocab",    type=int,  default=300)
    parser.add_argument("--holistic", action="store_true",
                        help="Extract 225-dim holistic keypoints (hands + full body pose)")
    args = parser.parse_args()
    preprocess_dataset(split=args.split, vocab_size=args.vocab, holistic=args.holistic)
