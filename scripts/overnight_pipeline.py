"""
scripts/overnight_pipeline.py

Full 108k pipeline — downloads ALL Aslense videos, builds union vocab,
extracts keypoints, deletes videos immediately after extraction.

FAILURE MODE PROTECTIONS:
  1. HuggingFace rate limiting  → auto-retry with exponential backoff
  2. MediaPipe crash on video   → try/except skips bad video, continues
  3. Disk fills up              → stops at 15GB free, prints resume command

DISK SPACE REQUIREMENTS:
  - Peak usage:  ~1.5GB (one batch of 300 videos in /tmp, deleted after)
  - Final .npy:  ~5.4GB (108k keypoint files at ~50KB each)
  - Minimum free: 15GB recommended, 10GB absolute minimum
  - Your current free: check with `df -h ~` before running

SAFE TO INTERRUPT AND RESUME:
  - Already-extracted .npy files are skipped automatically
  - Manifest checkpoint saved after every batch
  - Re-run the script to pick up where you left off

Run from project root:
    python3 scripts/overnight_pipeline.py
"""

import json, shutil, random, os, time, traceback
import numpy as np
import cv2
import mediapipe as mp
import urllib.request
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID         = "akasheroor/American-Sign-Language-Dataset"
BATCH_DIR       = Path("/tmp/aslense_batch")
ASLENSE_NPY_DIR = Path("data/processed/aslense")
PROC_DIR        = Path("data/processed")
BATCH_SIZE      = 300
MIN_SAMPLES     = 10    # minimum total samples across all datasets to include word
DISK_STOP_GB    = 15    # stop if free disk drops below this
MAX_RETRIES     = 3     # HuggingFace download retries
RETRY_DELAY     = 5     # seconds between retries (doubles each attempt)
random.seed(42)

BATCH_DIR.mkdir(parents=True, exist_ok=True)
ASLENSE_NPY_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_free_gb():
    """Return free disk space in GB at home directory."""
    statvfs = os.statvfs(str(Path.home()))
    return statvfs.f_frsize * statvfs.f_bavail / 1e9


def check_disk(label=""):
    """Print disk status and return False if below threshold."""
    free = get_free_gb()
    status = f"  Disk free: {free:.1f}GB"
    if label:
        status += f" [{label}]"
    print(status)
    if free < DISK_STOP_GB:
        print(f"\n  ⚠ DISK WARNING: Only {free:.1f}GB free (threshold: {DISK_STOP_GB}GB)")
        print(f"  Stopping to prevent disk full. Free up space and re-run to resume.")
        print(f"  Re-run command: python3 scripts/overnight_pipeline.py")
        return False
    return True


def download_with_retry(repo_id, filepath, local_dir, max_retries=MAX_RETRIES):
    """
    Download a file from HuggingFace with exponential backoff retry.
    Handles rate limiting (429) and transient network errors.
    Returns local path on success, None on failure.
    """
    delay = RETRY_DELAY
    for attempt in range(max_retries):
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=filepath,
                repo_type="dataset",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False)
            return Path(local)
        except Exception as e:
            err_str = str(e).lower()
            if attempt < max_retries - 1:
                if "429" in err_str or "rate" in err_str:
                    wait = delay * (2 ** attempt)
                    tqdm.write(f"  Rate limited — waiting {wait}s before retry "
                               f"({attempt+1}/{max_retries})")
                    time.sleep(wait)
                elif "connection" in err_str or "timeout" in err_str:
                    tqdm.write(f"  Network error — retrying in {delay}s "
                               f"({attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    return None  # non-retriable error
            else:
                return None
    return None


def safe_extract_keypoints(video_path, detector):
    """
    Extract keypoints from a video, returning None if MediaPipe crashes.
    Protects against: corrupt videos, zero-frame videos, mediapipe errors.
    """
    try:
        cap    = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = detector.detect(mp_image)
                lh = np.zeros(63, dtype=np.float32)
                rh = np.zeros(63, dtype=np.float32)
                for i, hand_landmarks in enumerate(result.hand_landmarks):
                    handedness = result.handedness[i][0].category_name
                    coords     = np.array([[lm.x, lm.y, lm.z]
                                            for lm in hand_landmarks],
                                           dtype=np.float32).flatten()
                    if handedness == "Left":
                        lh = coords
                    else:
                        rh = coords
                frames.append(np.concatenate([lh, rh]))
            except Exception:
                # Skip bad frame, continue with rest of video
                continue
        cap.release()
        if not frames:
            return None
        return np.stack(frames)
    except Exception as e:
        tqdm.write(f"  MediaPipe error on {video_path.name}: {e}")
        return None


# ── Pre-flight disk check ─────────────────────────────────────────────────────
print("=" * 60)
print("PRE-FLIGHT CHECK")
print("=" * 60)
free = get_free_gb()
print(f"  Free disk space: {free:.1f}GB")
print(f"  Required minimum: {DISK_STOP_GB}GB")
print(f"  Peak usage per batch: ~1.5GB (deleted immediately after)")
print(f"  Final .npy storage: ~5.4GB")
if free < DISK_STOP_GB:
    print(f"\n  ✗ Not enough disk space. Free up at least {DISK_STOP_GB}GB and re-run.")
    exit(1)
else:
    print(f"  ✓ Disk space OK")


# ── Step 0: Back up everything ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 0: Backing up existing work")
print("=" * 60)

backup_dir = PROC_DIR / "backup_v1_300class"
backup_dir.mkdir(exist_ok=True)

for fname in [
    "vocab.json", "train_manifest.json", "val_manifest.json",
    "test_manifest.json", "combined_train_manifest.json",
    "train_manifest_stratified.json", "val_manifest_stratified.json",
    "aslcitizen_train_manifest.json",
]:
    src, dst = PROC_DIR / fname, backup_dir / fname
    if src.exists() and not dst.exists():
        shutil.copy(src, dst)
        print(f"  Backed up: {fname}")
    elif dst.exists():
        print(f"  Already backed up: {fname}")
    else:
        print(f"  Not found (skipping): {fname}")

ckpt_backup = Path("models/checkpoints/backup_v1_300class")
ckpt_backup.mkdir(parents=True, exist_ok=True)
for ckpt in Path("models/checkpoints").glob("*.pt"):
    dst = ckpt_backup / ckpt.name
    if not dst.exists():
        shutil.copy(ckpt, dst)
        print(f"  Backed up checkpoint: {ckpt.name}")

print(f"\n  ✓ Backups at: {backup_dir}")
print(f"  ✓ Checkpoint backups at: {ckpt_backup}")
print(f"\n  To restore old baseline at any time:")
print(f"    cp {backup_dir}/vocab.json data/processed/vocab.json")
print(f"    cp {backup_dir}/combined_train_manifest.json "
      f"data/processed/combined_train_manifest.json")
print(f"    cp {backup_dir}/val_manifest.json data/processed/val_manifest.json")


# ── Step 1: Get full Aslense file list ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Fetching full Aslense file list")
print("=" * 60)

print("Fetching from HuggingFace (~2 min)...")
try:
    all_files = list(list_repo_files(REPO_ID, repo_type="dataset"))
except Exception as e:
    print(f"  ✗ Failed to fetch file list: {e}")
    print(f"  Check your internet connection and re-run.")
    exit(1)

mp4_files = [f for f in all_files if f.endswith('.mp4')]
print(f"  ✓ Total videos in repo: {len(mp4_files):,}")

aslense_word_counts = Counter()
for f in mp4_files:
    word = f.split('-', 1)[-1].replace('.mp4', '').lower().strip()
    aslense_word_counts[word] += 1
print(f"  ✓ Unique words: {len(aslense_word_counts):,}")


# ── Step 2: Build union vocab ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Building union vocab")
print("=" * 60)

all_counts = defaultdict(int)

with open('data/raw/wlasl/WLASL_v0.3.json') as f:
    wlasl_data = json.load(f)
for entry in wlasl_data:
    for inst in entry['instances']:
        if inst['split'] == 'train':
            all_counts[entry['gloss'].lower()] += 1
print(f"  WLASL: {len(all_counts):,} words")

aslc_path = PROC_DIR / 'aslcitizen_train_manifest.json'
if aslc_path.exists():
    with open(aslc_path) as f:
        for s in json.load(f):
            all_counts[s['label'].lower()] += 1
    print(f"  ASL Citizen: added")

for word, count in aslense_word_counts.items():
    all_counts[word] += count
print(f"  Aslense: {len(aslense_word_counts):,} words added")

vocab = {
    word: idx
    for idx, (word, count) in enumerate(
        sorted(all_counts.items(), key=lambda x: -x[1])
    )
    if count >= MIN_SAMPLES
}

print(f"\n  Union vocab: {len(vocab):,} words")
print(f"  Total samples: {sum(all_counts[w] for w in vocab):,}")
print(f"  Avg samples/class: {sum(all_counts[w] for w in vocab)/len(vocab):.1f}")

# Save new vocab
if not (backup_dir / 'vocab.json').exists():
    shutil.copy(PROC_DIR / 'vocab.json', backup_dir / 'vocab.json')
with open(PROC_DIR / 'vocab.json', 'w') as f:
    json.dump(vocab, f, indent=2)
print(f"  ✓ New vocab saved")


# ── Step 3: MediaPipe setup ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Setting up MediaPipe")
print("=" * 60)

model_path = Path("data/hand_landmarker.task")
if not model_path.exists():
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
    print("  Downloading MediaPipe hand model...")
    urllib.request.urlretrieve(url, str(model_path))
else:
    print("  MediaPipe model already exists")

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode
detector = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
print("  ✓ Detector ready")


# ── Step 4: Download + extract ALL Aslense videos ────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Downloading + extracting all Aslense videos")
print("=" * 60)

to_process   = []
already_done = 0
not_in_vocab = 0

for f in mp4_files:
    word = f.split('-', 1)[-1].replace('.mp4', '').lower().strip()
    if word not in vocab:
        not_in_vocab += 1
        continue
    npy_path = ASLENSE_NPY_DIR / (Path(f).stem + '.npy')
    if npy_path.exists():
        already_done += 1
    else:
        to_process.append((f, word, npy_path))

print(f"  Already extracted (skipping): {already_done:,}")
print(f"  Not in vocab (skipping):      {not_in_vocab:,}")
print(f"  To download + extract:        {len(to_process):,}")
print(f"  Estimated time: {len(to_process) * 8 / 3600:.1f}h "
      f"(~8s per video download+extract)")

# Pre-load already-done into manifest
aslense_manifest = []
for npy in ASLENSE_NPY_DIR.glob('*.npy'):
    stem = npy.stem
    word = stem.split('-', 1)[1].lower().strip() if '-' in stem else stem.lower()
    if word in vocab:
        aslense_manifest.append({
            "path":      str(npy),
            "label":     word,
            "label_idx": vocab[word],
            "source":    "aslense",
        })
print(f"  Pre-loaded {len(aslense_manifest):,} existing samples")

checkpoint_path = PROC_DIR / 'aslense_manifest.json'
if checkpoint_path.exists() and len(aslense_manifest) == 0:
    with open(checkpoint_path) as f:
        aslense_manifest = json.load(f)
    print(f"  Resumed from checkpoint: {len(aslense_manifest):,} samples")

failed_download  = 0
failed_mediapipe = 0
failed_disk      = 0

for batch_start in range(0, len(to_process), BATCH_SIZE):
    batch         = to_process[batch_start:batch_start + BATCH_SIZE]
    batch_num     = batch_start // BATCH_SIZE + 1
    total_batches = (len(to_process) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\n  Batch {batch_num}/{total_batches} — {len(batch)} videos")

    # ── Failure mode 3: Check disk before each batch ──────────────────────
    if not check_disk("pre-batch"):
        break

    # ── Failure mode 1: Download with retry ───────────────────────────────
    downloaded = []
    for filepath, word, npy_path in tqdm(batch, desc="    Download"):
        tmp_path = BATCH_DIR / Path(filepath).name
        result   = download_with_retry(REPO_ID, filepath, BATCH_DIR)
        if result is not None:
            try:
                result.rename(tmp_path)
                downloaded.append((tmp_path, word, npy_path))
            except Exception:
                if tmp_path.exists():
                    downloaded.append((tmp_path, word, npy_path))
        else:
            failed_download += 1

    # ── Failure mode 2: Extract with MediaPipe crash protection ───────────
    for tmp_path, word, npy_path in tqdm(downloaded, desc="    Extract"):
        try:
            kpts = safe_extract_keypoints(tmp_path, detector)
            if kpts is not None:
                np.save(str(npy_path), kpts)
                aslense_manifest.append({
                    "path":      str(npy_path),
                    "label":     word,
                    "label_idx": vocab[word],
                    "source":    "aslense",
                })
            else:
                failed_mediapipe += 1
                tqdm.write(f"    Skipped (bad video): {tmp_path.name}")
        except Exception as e:
            failed_mediapipe += 1
            tqdm.write(f"    Extraction error {tmp_path.name}: {e}")
        finally:
            # Always delete video regardless of success or failure
            if tmp_path.exists():
                tmp_path.unlink()

    # Save manifest checkpoint after every batch — safe to interrupt
    with open(checkpoint_path, 'w') as f:
        json.dump(aslense_manifest, f, indent=2)

    free = get_free_gb()
    print(f"    Extracted: {len(aslense_manifest):,} | "
          f"DL fails: {failed_download} | "
          f"MP fails: {failed_mediapipe} | "
          f"Disk: {free:.1f}GB free")

    # ── Failure mode 3: Stop if disk getting full ─────────────────────────
    if free < DISK_STOP_GB:
        print(f"\n  ⚠ Disk low — stopping at batch {batch_num}/{total_batches}")
        print(f"  Free up space then re-run: python3 scripts/overnight_pipeline.py")
        break

detector.close()
print(f"\n  ✓ Aslense done: {len(aslense_manifest):,} extracted")
print(f"  Download failures:  {failed_download}")
print(f"  MediaPipe failures: {failed_mediapipe}")


# ── Step 5: Rebuild WLASL manifests ──────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Rebuilding WLASL manifests")
print("=" * 60)

for split in ['train', 'val', 'test']:
    split_manifest   = []
    skipped_no_vocab = 0
    skipped_no_npy   = 0
    for entry in wlasl_data:
        gloss = entry['gloss'].lower()
        if gloss not in vocab:
            skipped_no_vocab += 1
            continue
        for inst in entry['instances']:
            if inst['split'] != split:
                continue
            vid_id   = inst['video_id']
            npy_path = Path(f"data/processed/{split}/{vid_id}.npy")
            if not npy_path.exists():
                skipped_no_npy += 1
                continue
            split_manifest.append({
                "path":      str(npy_path),
                "label":     gloss,
                "label_idx": vocab[gloss],
                "source":    "wlasl",
            })
    with open(PROC_DIR / f'{split}_manifest.json', 'w') as f:
        json.dump(split_manifest, f, indent=2)
    print(f"  WLASL {split}: {len(split_manifest):,} samples "
          f"({skipped_no_vocab} not in vocab, {skipped_no_npy} missing .npy)")


# ── Step 6: Update ASL Citizen labels ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Updating ASL Citizen manifest")
print("=" * 60)

aslc_updated = []
skipped      = 0
if aslc_path.exists():
    with open(aslc_path) as f:
        aslc = json.load(f)
    for s in aslc:
        word = s['label'].lower()
        if word in vocab:
            s['label_idx'] = vocab[word]
            aslc_updated.append(s)
        else:
            skipped += 1
    with open(aslc_path, 'w') as f:
        json.dump(aslc_updated, f, indent=2)
    print(f"  ✓ ASL Citizen: {len(aslc_updated):,} kept, {skipped} skipped")


# ── Step 7: Combined manifest ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Building combined manifest")
print("=" * 60)

with open(PROC_DIR / 'train_manifest.json') as f:
    wlasl_train = json.load(f)

combined = wlasl_train + aslc_updated + aslense_manifest
with open(PROC_DIR / 'combined_train_manifest.json', 'w') as f:
    json.dump(combined, f, indent=2)

src_counts = Counter(s['source'] for s in combined)
print(f"  WLASL:       {src_counts.get('wlasl', 0):,}")
print(f"  ASL Citizen: {src_counts.get('aslcitizen', 0):,}")
print(f"  Aslense:     {src_counts.get('aslense', 0):,}")
print(f"  Total:       {len(combined):,}")


# ── Step 8: Stratified val split ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Stratified val split")
print("=" * 60)

by_class = defaultdict(list)
for s in combined:
    by_class[s['label']].append(s)

new_train, new_val = [], []
for label, samples in by_class.items():
    random.shuffle(samples)
    n_val = max(1, min(3, len(samples) // 5))
    new_val   += samples[:n_val]
    new_train += samples[n_val:]

with open(PROC_DIR / 'train_manifest_stratified.json', 'w') as f:
    json.dump(new_train, f, indent=2)
with open(PROC_DIR / 'val_manifest_stratified.json', 'w') as f:
    json.dump(new_val, f, indent=2)

shutil.copy(PROC_DIR / 'train_manifest_stratified.json',
            PROC_DIR / 'combined_train_manifest.json')
shutil.copy(PROC_DIR / 'val_manifest_stratified.json',
            PROC_DIR / 'val_manifest.json')

val_counts = Counter(s['label'] for s in new_val)
print(f"  ✓ Train: {len(new_train):,} samples")
print(f"  ✓ Val:   {len(new_val):,} samples")
print(f"  ✓ Val classes with 2+ samples: "
      f"{sum(1 for c in val_counts.values() if c >= 2)}")


# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL DONE ✓")
print("=" * 60)
print(f"\n  Failure summary:")
print(f"    Download failures:  {failed_download}")
print(f"    MediaPipe failures: {failed_mediapipe}")
print(f"\n  Old baseline preserved at:")
print(f"    {backup_dir}/")
print(f"    {ckpt_backup}/")
print(f"\n  To restore old baseline:")
print(f"    cp {backup_dir}/vocab.json data/processed/vocab.json")
print(f"    cp {backup_dir}/combined_train_manifest.json "
      f"data/processed/combined_train_manifest.json")
print(f"    cp {backup_dir}/val_manifest.json data/processed/val_manifest.json")
print(f"\n  Train command:")
print(f"    python3 -m src.train --vocab {len(vocab)} "
      f"--epochs 150 --combined --workers 0")