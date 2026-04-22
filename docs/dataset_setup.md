# Dataset Setup Guide

This project uses two ASL datasets: **WLASL** (primary — training + evaluation) and **ASL Citizen** (supplemental training data). This guide covers how to download and process both.

---

## Prerequisites

Install all Python dependencies and system tools before starting:

```bash
pip install -r requirements.txt      # includes requests, tqdm, yt-dlp, mediapipe, opencv-python, pose-format
brew install ffmpeg                  # macOS — required for keypoint extraction
```

All commands below must be run from the **project root** (`asl-realtime/`).

---

## Dataset 1 — WLASL (Primary)

| Attribute | Value |
|-----------|-------|
| Classes | 2,000 signs |
| Instances | 21,083 clips |
| Signers | 119 |
| Access | Free — [github.com/dxli94/WLASL](https://github.com/dxli94/WLASL) |
| Role in project | Training + validation + test evaluation |

The annotation file `data/raw/wlasl/WLASL_v0.3.json` is already committed to this repo. Only the video files need to be downloaded.

### Step 1 — Download WLASL videos

```bash
python3 scripts/download_wlasl.py
```

This attempts to download all 21,083 video instances via:
- **Direct HTTP links** (tried first for non-YouTube URLs)
- **YouTube via yt-dlp** (for YouTube-hosted clips)

Already-downloaded files are automatically skipped on re-runs. A failure log is written to `data/raw/wlasl/failed_downloads.jsonl`.

**Expected yield (as of March 2026):** ~6,400 usable clips (~30%). Many direct-link hosts are dead since the dataset was compiled in 2020. This yield is sufficient for training.

**To skip YouTube downloads** (faster, but fewer samples):
```bash
python3 scripts/download_wlasl.py --skip-youtube
```

**YouTube bot detection workaround:**
If yt-dlp is blocked, export your browser cookies and save them as `data/raw/wlasl/cookies.txt` (gitignored). The script picks up the file automatically on the next run.

```bash
# Export cookies from Chrome using a browser extension like "Get cookies.txt LOCALLY"
# Save the exported file to:
#   data/raw/wlasl/cookies.txt
```

Videos are saved to `data/raw/wlasl/videos/` as `<video_id>.mp4`.

### Step 2 — Extract WLASL keypoints

```bash
python3 scripts/preprocess.py --split train --vocab 2000
python3 scripts/preprocess.py --split val   --vocab 2000
python3 scripts/preprocess.py --split test  --vocab 2000
```

This uses MediaPipe HandLandmarker to extract per-frame hand keypoints (shape `[T, 126]`) from each video. A ~25MB MediaPipe model is downloaded automatically on first run.

Output:
```
data/processed/
├── vocab.json              # sign → class index mapping (2000 entries)
├── train_manifest.json     # [{path, label, label_idx}, ...]
├── val_manifest.json
└── test_manifest.json
```

---

## Dataset 2 — ASL Citizen (Supplemental Training Data)

| Attribute | Value |
|-----------|-------|
| Classes | ~2,700 signs |
| Instances | ~83,000 clips |
| Signers | 52 |
| Access | Free — [HuggingFace: SorensenAI/asl-citizen-poses](https://huggingface.co/datasets/SorensenAI/asl-citizen-poses) |
| Role in project | Supplemental training data merged with WLASL |
| Format | Pre-extracted `.pose` files — **no video downloading needed** |

ASL Citizen is unique in that it provides pre-extracted pose keypoints rather than raw videos. The downloader fetches `.pose` files directly from HuggingFace, converts them to our `(T, 126)` format, and merges them with the WLASL training manifest.

Only clips whose labels overlap with the WLASL vocabulary are downloaded.

### Step 1 — Download and convert ASL Citizen

Run **after** WLASL keypoint extraction (requires `data/processed/vocab.json` to exist):

```bash
python3 scripts/download_aslcitizen.py --vocab 2000
```

What it does:
1. Reads `data/processed/vocab.json` to determine which sign labels to keep
2. Fetches `.pose` files from HuggingFace across 17 batches
3. Converts each `.pose` file to a `(T, 126)` numpy array (same format as WLASL)
4. Saves `.npy` files to `data/processed/aslcitizen/`
5. Writes `data/processed/aslcitizen_train_manifest.json`
6. Merges with `data/processed/train_manifest.json` (WLASL) and writes `data/processed/combined_train_manifest.json`

```
data/processed/
├── aslcitizen/                        # converted .npy keypoint files
├── aslcitizen_train_manifest.json     # ASL Citizen samples only
└── combined_train_manifest.json       # WLASL + ASL Citizen merged
```

**To use a smaller vocabulary** (e.g. for quick testing):
```bash
python3 scripts/download_aslcitizen.py --vocab 100
```

---

## Step 3 — Train

Use the combined manifest to train with both datasets:

```bash
python3 src/train.py --vocab 2000 --epochs 50 --d_model 128 --n_layers 3
```

The dataloader will pick up `combined_train_manifest.json` automatically when it exists.

---

## Disk Space Estimates

| Content | Approximate Size |
|---------|-----------------|
| WLASL raw videos (~6,400 clips) | ~15 GB |
| WLASL processed keypoints (all splits) | ~500 MB |
| ASL Citizen `.npy` keypoints (vocab-filtered) | ~200 MB |

Raw video directories (`data/raw/*/videos/`) are gitignored. Do not commit them.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `yt-dlp: command not found` | `pip3 install yt-dlp` |
| `ffmpeg: command not found` | `brew install ffmpeg` (Mac) or `apt install ffmpeg` (Linux) |
| YouTube bot detection (HTTP 429) | Export browser cookies → save to `data/raw/wlasl/cookies.txt` |
| Very low WLASL yield (<5,000 clips) | Expected — most direct links are dead. Run without `--skip-youtube`. |
| `pose_format` import error | `pip install pose-format` |
| HuggingFace batch returns empty list | Transient network error — re-run the script (already-converted files are skipped) |
| `vocab.json` not found when running ASL Citizen downloader | Run WLASL keypoint extraction first (`scripts/preprocess.py`) |
