"""
scripts/download_aslcitizen.py

Download ASL Citizen pre-extracted .pose files from HuggingFace and convert
to (T, 126) hand keypoint numpy arrays compatible with our training pipeline.

Only downloads clips whose labels overlap with the WLASL vocabulary.

Usage:
    python scripts/download_aslcitizen.py --vocab 300
    python scripts/download_aslcitizen.py --vocab 100

Output:
    data/processed/aslcitizen_train_manifest.json
    data/processed/aslcitizen/  (*.npy keypoint files)
"""

import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm

HF_REPO      = "SorensenAI/asl-citizen-poses"
HF_BASE_URL  = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
NUM_BATCHES  = 17
DATA_PROC    = Path("data/processed")
SAVE_DIR     = DATA_PROC / "aslcitizen"


def get_wlasl_vocab(vocab_size: int) -> dict:
    """Load or build WLASL vocab. Requires vocab.json to exist."""
    vocab_path = DATA_PROC / "vocab.json"
    if vocab_path.exists():
        with open(vocab_path) as f:
            vocab = json.load(f)
        if len(vocab) >= vocab_size:
            # trim to vocab_size
            vocab = {w: i for w, i in vocab.items() if i < vocab_size}
            return vocab

    # Build from annotation
    anno = Path("data/raw/wlasl/WLASL_v0.3.json")
    with open(anno) as f:
        data = json.load(f)
    train_counts = {}
    for entry in data:
        for inst in entry["instances"]:
            if inst["split"] == "train":
                g = entry["gloss"]
                train_counts[g] = train_counts.get(g, 0) + 1
    top = sorted(train_counts, key=train_counts.get, reverse=True)[:vocab_size]
    return {sign: idx for idx, sign in enumerate(top)}


def pose_to_keypoints(pose_path: str) -> np.ndarray:
    """
    Read a .pose file and extract (T, 126) normalized hand keypoints.
    LEFT_HAND_LANDMARKS:  indices 501:522  -> 21 pts × 3 = 63 dims
    RIGHT_HAND_LANDMARKS: indices 522:543  -> 21 pts × 3 = 63 dims

    Normalizes pixel coordinates to [0, 1] to match WLASL MediaPipe output:
        x / width, y / height, z / width
    """
    from pose_format import Pose
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())

    w = pose.header.dimensions.width  or 640
    h = pose.header.dimensions.height or 480

    data = pose.body.data.data  # (T, 1, 576, 3)
    lh   = data[:, 0, 501:522, :].copy()  # (T, 21, 3)
    rh   = data[:, 0, 522:543, :].copy()  # (T, 21, 3)

    # Normalize: x→/w, y→/h, z→/w
    for arr in (lh, rh):
        arr[:, :, 0] /= w
        arr[:, :, 1] /= h
        arr[:, :, 2] /= w

    kpts = np.concatenate([lh.reshape(-1, 63), rh.reshape(-1, 63)], axis=1)
    return kpts.astype(np.float32)  # (T, 126)


def download_file(url: str, dest: Path) -> bool:
    """Download a file using curl. Returns True on success."""
    result = subprocess.run(
        ["curl", "-L", "-s", "-f", "-o", str(dest), url],
        capture_output=True,
    )
    return result.returncode == 0 and dest.exists() and dest.stat().st_size > 100


def list_batch_files(batch_num: int) -> list[str]:
    """Fetch the file list for a batch from HuggingFace API."""
    import urllib.request, ssl, json
    ctx = ssl._create_unverified_context()
    api_url = (f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main"
               f"/poses_batches/batch_{batch_num:03d}")
    try:
        with urllib.request.urlopen(api_url, context=ctx) as r:
            return [item["path"] for item in json.loads(r.read())]
    except Exception:
        return []


def main(vocab_size: int):
    vocab = get_wlasl_vocab(vocab_size)
    print(f"Vocab: {len(vocab)} words (top-{vocab_size})")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path("/tmp/aslcitizen_poses")
    tmp_dir.mkdir(exist_ok=True)

    manifest      = []
    skipped_vocab = 0
    skipped_dl    = 0
    skipped_conv  = 0

    for batch_num in range(1, NUM_BATCHES + 1):
        print(f"\n=== Batch {batch_num:03d}/{NUM_BATCHES} ===")
        files = list_batch_files(batch_num)
        if not files:
            print(f"  Could not list batch {batch_num}, skipping")
            continue

        pose_files = [f for f in files if f.endswith(".pose")]
        print(f"  {len(pose_files)} .pose files found")

        for path in tqdm(pose_files, desc=f"Batch {batch_num:03d}"):
            filename = Path(path).name                    # e.g. 000...-LIBRARY.pose
            word     = filename.rsplit("-", 1)[-1].replace(".pose", "").lower()

            if word not in vocab:
                skipped_vocab += 1
                continue

            npy_path = SAVE_DIR / filename.replace(".pose", ".npy")
            if npy_path.exists():
                manifest.append({
                    "path":      str(npy_path),
                    "label":     word,
                    "label_idx": vocab[word],
                    "source":    "aslcitizen",
                })
                continue

            # Download
            tmp_path = tmp_dir / filename
            url      = f"{HF_BASE_URL}/{path}"
            if not download_file(url, tmp_path):
                skipped_dl += 1
                continue

            # Convert
            try:
                kpts = pose_to_keypoints(str(tmp_path))
                np.save(str(npy_path), kpts)
                tmp_path.unlink(missing_ok=True)
                manifest.append({
                    "path":      str(npy_path),
                    "label":     word,
                    "label_idx": vocab[word],
                    "source":    "aslcitizen",
                })
            except Exception as e:
                skipped_conv += 1
                tmp_path.unlink(missing_ok=True)

    # Save manifest
    manifest_path = DATA_PROC / "aslcitizen_train_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== Done ===")
    print(f"Saved {len(manifest)} samples → {manifest_path}")
    print(f"Skipped (not in vocab): {skipped_vocab}")
    print(f"Skipped (download failed): {skipped_dl}")
    print(f"Skipped (conversion failed): {skipped_conv}")

    # Merge with WLASL manifest
    wlasl_path = DATA_PROC / "train_manifest.json"
    if wlasl_path.exists():
        with open(wlasl_path) as f:
            wlasl = json.load(f)
        for e in wlasl:
            e.setdefault("source", "wlasl")
        combined = wlasl + manifest
        combined_path = DATA_PROC / "combined_train_manifest.json"
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"Combined train manifest: {len(combined)} samples → {combined_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=300)
    args = parser.parse_args()
    main(args.vocab)
