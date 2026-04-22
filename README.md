# Real-Time ASL Translation
### Applied Deep Learning — Final Project
**Northeastern University · Spring 2026**

> Keypoint-based temporal transformer for real-time American Sign Language word recognition on CPU.

---

## Group Members
| Name | GitHub | Role |
|------|--------|------|
| Radhika Khurana | [@khuranaradhika](https://github.com/khuranaradhika) | Data pipeline, model architecture + training, web server |
| Jian Gao | [@iamjaygao](https://github.com/iamjaygao) | Dataset sourcing, keypoint extraction + augmentation |
| Hrishikesh Pradhan | [@hspgit](https://github.com/hspgit) | Baseline experiments + knowledge distillation |
| Gyula Planky | [@gyuszix](https://github.com/gyuszix) | ONNX export, evaluation + demo |

---

## Project Overview

Given a continuous video stream of a person signing, our system identifies which ASL word is being signed and produces a text transcript in real time. We target the top 1,896 most frequent signs across our combined training corpus.

Existing models achieving strong benchmark accuracy (I3D, SlowFast, VideoMAE) rely on GPU inference over raw video frames and cannot run on a standard laptop. Our approach avoids raw video at inference time entirely — MediaPipe reduces each frame to a 126-dimensional hand keypoint vector, which a compact transformer classifies. The result is a system fast enough to run on CPU with no GPU required.

**Pipeline:**
```
Webcam → MediaPipe HandLandmarker → keypoints → Transformer → Text
          126 floats/frame, ~8ms                ~5ms inference
```

**Success metrics:**
- ≥ 60% Top-1 accuracy on held-out val set (vocab=1896)
- ≤ 50ms end-to-end inference latency on CPU
- Runs on standard laptop with no GPU required

---

## Architecture

We compare three temporal modeling approaches on the same data and training setup:

| Model | How it sees time | Params | Loss |
|-------|-----------------|--------|------|
| **1D CNN** | Local 3-frame windows | ~450K | Cross-entropy |
| **BiLSTM** | Sequential hidden state | ~735K | Cross-entropy |
| **Transformer** | Full self-attention across all frames | ~452K | Cross-entropy |

All three use **cross-entropy loss with label smoothing (0.1)** and global average pooling for isolated word classification.

### Transformer Architecture (main model)

```
Input: (B, T, 126) keypoint sequences
  → Linear projection: 126 → 128   (d_model)
  → Sinusoidal positional encoding
  → 3 × TransformerEncoderLayer
      - 4 attention heads
      - FFN dim: 256
      - Pre-norm (norm_first=True) — more stable training
      - Dropout: 0.1
  → Global mean pool over non-padded frames
  → Linear classifier: 128 → 1,896
  → Cross-entropy loss (label_smoothing=0.1)
```

**Key training details:**
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
- Scheduler: Cosine annealing (T_max=epochs, eta_min=1e-6)
- Gradient clipping: max_norm=1.0
- WeightedRandomSampler: ensures equal class representation per batch
- MPS acceleration on Apple Silicon (M4 Pro)

**Why CTC is not the primary loss:** CTC is designed for continuous sequence labeling where the alignment between input and output is unknown (e.g. streaming speech recognition). For isolated word classification — where each clip contains exactly one sign — cross-entropy with mean pooling is the correct and simpler approach. CTC is retained in the codebase for future continuous signing work.

---

## Repository Structure

```
asl-realtime/
│
├── data/
│   ├── raw/
│   │   ├── wlasl/
│   │   │   ├── WLASL_v0.3.json         # Annotations (2000 signs, 21k instances)
│   │   │   └── videos/                 # Downloaded .mp4 files (gitignored)
│   │   └── msasl/
│   │       ├── MSASL_{train,val,test}.json
│   │       └── MSASL_classes.json
│   └── processed/
│       ├── train/                      # WLASL extracted .npy keypoints (gitignored)
│       ├── val/
│       ├── test/
│       ├── aslcitizen/                 # ASL Citizen .npy keypoints (gitignored)
│       ├── aslense/                    # Aslense .npy keypoints (gitignored)
│       ├── vocab.json                  # 1896-sign vocab (sign → index)
│       ├── train_manifest.json         # WLASL train samples
│       ├── val_manifest.json           # Stratified val — 5,567 samples
│       ├── combined_train_manifest.json    # All sources — 52,998 samples
│       ├── aslcitizen_train_manifest.json  # ASL Citizen — 1,542 samples
│       └── aslense_manifest.json           # Aslense — 53,933 samples
│
├── src/
│   ├── config.py             # Shared constants and paths
│   ├── augmentations.py      # Flip, jitter, noise, normalization
│   ├── keypoints.py          # MediaPipe HandLandmarker extraction
│   ├── dataloader.py         # WLASLDataset + WeightedRandomSampler
│   ├── model.py              # SignTransformer, CNN, BiLSTM
│   ├── decode.py             # Greedy CTC decode
│   ├── evaluate.py           # Top-1/Top-5 accuracy, per-class breakdown
│   ├── train.py              # Training loop, checkpointing
│   ├── export.py             # ONNX export + CPU latency benchmark
│   └── demo.py               # Real-time webcam demo
│
├── scripts/
│   ├── download_wlasl.py         # WLASL downloader (direct links + yt-dlp)
│   ├── download_msasl.py         # MS-ASL downloader (YouTube + ffmpeg trim)
│   ├── download_aslcitizen.py    # ASL Citizen downloader (HuggingFace)
│   ├── download_aslense.py       # Aslense downloader (HuggingFace, 108k videos)
│   └── overnight_pipeline.py     # Full data pipeline (download + extract + manifest)
│
├── models/
│   ├── checkpoints/          # .pt checkpoint files (gitignored)
│   └── sign_model.onnx       # Exported deployment model
│
├── docs/
│   ├── experiments.md        # Running experiment log
│   └── dataset_setup.md      # Dataset setup guide
│
├── results/
│   ├── figures/              # Loss curves, confusion matrices
│   └── metrics/              # JSON per-run results
│
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/khuranaradhika/asl-realtime.git
cd asl-realtime
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Datasets

We train on a combined corpus of three datasets filtered to vocab=1896.
Processed keypoints are hosted on HuggingFace and can be downloaded with one command (see below).

| Dataset | Role | Clips | Classes | Access |
|---------|------|-------|---------|--------|
| WLASL | Train + in-distribution eval | ~4,143 train / 1,208 val / 206 test | 2,000 | [Free](https://github.com/dxli94/WLASL) |
| ASL Citizen | Train only | 1,542 | ~300 | HuggingFace |
| Aslense | Train only | 53,933 | 2,208 | HuggingFace |
| **Combined** | **Training corpus** | **52,998 train / 5,567 val** | **1,896** | — |

**Total training data: 52,998 samples across 1,896 sign classes (~28 samples/class average)**

### Option A: Download pre-extracted keypoints (fastest — recommended for teammates)

```bash
pip install huggingface_hub
python3 - << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="khuranaradhika/asl-realtime-keypoints",
    repo_type="dataset",
    local_dir="data/processed")
print("Done — ready to train")
EOF
```

### Option B: Download raw videos and extract keypoints yourself

```bash
# WLASL
python3 scripts/download_wlasl.py

# ASL Citizen + Aslense (downloads and extracts automatically)
python3 scripts/overnight_pipeline.py
```

The overnight pipeline downloads videos in batches of 300, extracts keypoints immediately,
and deletes the raw video — peak disk usage is ~2GB at any time. Final keypoints are ~2.7GB total.

**WLASL download yield (March 2026):** 6,417 clips out of 21,083 (~30%).
Most losses are dead hosting links from 2020. Export browser cookies via
"Get cookies.txt LOCALLY" (Chrome), save as `data/raw/wlasl/cookies.txt` (gitignored).

---

## Keypoint Extraction

If downloading raw videos, run extraction after download. MediaPipe HandLandmarker
model (~25MB) downloads automatically on first run.

```bash
python3 -m src.dataloader --extract --split train --vocab 1896
python3 -m src.dataloader --extract --split val   --vocab 1896
python3 -m src.dataloader --extract --split test  --vocab 1896
```

**Output format:** Each `.npy` file is `(T, 126)` — T frames × 126 features
(21 left-hand landmarks × 3 + 21 right-hand landmarks × 3, xyz normalized).

---

## Augmentations

Applied during training only:

| Augmentation | Description | Effect |
|---|---|---|
| Horizontal flip | Swap left/right hands, mirror x | Doubles effective dataset size |
| Temporal jitter | Randomly drop or repeat frames | Robustness to dropped frames |
| Gaussian noise | σ=0.01 on all coordinates | Simulates MediaPipe detection noise |
| Wrist normalization | Subtract dominant wrist position | Translation invariance |

---

## Training

```bash
# Main model — Transformer with augmentation + WeightedRandomSampler
python3 -m src.train --vocab 1896 --epochs 150 --combined --workers 0

# Ablation — no augmentation
python3 -m src.train --vocab 1896 --epochs 150 --combined --workers 0 --no-augment

# Baseline — BiLSTM
python3 -m src.train --model lstm --vocab 1896 --epochs 150 --combined --workers 0

# Baseline — 1D CNN
python3 -m src.train --model cnn --vocab 1896 --epochs 150 --combined --workers 0
```

**Additional flags:**
```bash
--dropout 0.3        # stronger regularization (default 0.1)
--loss ctc           # CTC loss for future continuous signing work
--teacher            # larger teacher model (d=512, 6 layers, ~18M params)
--d_model 256        # wider model
--n_layers 4         # deeper model
```

Each run saves:
- `models/checkpoints/{run_name}_best.pt` — best val checkpoint (only overwrites if beaten)
- `results/metrics/{run_name}_history.json` — loss/Top-1/Top-5 per epoch
- `results/metrics/{run_name}_per_class.json` — per-class accuracy breakdown

---

## Evaluation

Top-1 and Top-5 accuracy are computed on the stratified val set after each epoch.
The final epoch also produces a full per-class breakdown saved to `results/metrics/`.

```bash
# Standalone evaluation on a saved checkpoint
python3 -m src.evaluate \
  --checkpoint models/checkpoints/transformer_d128_l3_v1896_combined_best.pt \
  --vocab 1896 --combined
```

**Evaluation method:** Mean log-probability over non-padded frames (more robust than
greedy CTC argmax for isolated word classification). For each sample, the model outputs
log-probabilities for every frame — we average over the valid (non-padded) frames and
take the argmax as the Top-1 prediction.

**Per-class breakdown** is printed automatically at end of training:
- Top-10 and Bottom-10 classes by accuracy
- Count of classes with 0% accuracy (data-starved classes)
- Saved to `results/metrics/{run_name}_per_class.json`

---

## Export + Demo

```bash
# Export to ONNX and benchmark CPU latency
python3 -m src.export \
  --checkpoint models/checkpoints/transformer_d128_l3_v1896_combined_best.pt \
  --vocab 1896

# Real-time webcam demo
python3 -m src.demo --model models/sign_model.onnx --vocab 1896
```

---

## Results

### Experiment Log

| Exp | Vocab | Train Samples | Val Samples | Top-1 | Top-5 | Notes |
|-----|-------|--------------|-------------|-------|-------|-------|
| EXP-001 | 100 | ~570 | ~100 | 7.8% | — | Sanity check, no aug |
| EXP-002 | 100 | ~570 | ~100 | 3.9% | — | Sanity check, with aug |
| EXP-003 | 300 | 2,288 | 229 | 6.7% | — | Broken val set (150/188 classes had 1 sample) |
| EXP-003 CE | 300 | 2,288 | 456 | **40.8%** | **56.4%** | Stratified val fix, CE loss |
| EXP-004 | 1,896 | 52,998 | 5,567 | — | — | Full dataset, WRS — **running** |

---

### Ablation Study (vocab=1896, 52,998 train samples)

| Model | Augmentation | WeightedSampler | Top-1 | Top-5 | Params | Command |
|-------|-------------|----------------|-------|-------|--------|---------|
| 1D CNN | ✅ | ✅ | — | — | ~450K | `--model cnn` |
| BiLSTM | ✅ | ✅ | — | — | ~735K | `--model lstm` |
| Transformer | ❌ | ❌ | — | — | ~452K | `--no-augment` |
| Transformer | ✅ | ❌ | 40.8%* | 56.4%* | ~452K | default |
| **Transformer** | **✅** | **✅** | **—** | **—** | **~452K** | **EXP-004 (running)** |

*\*Result from vocab=300 run (EXP-003 CE) — comparable configuration.*

Ablation results will be filled in as experiments complete.
Run baselines with:
```bash
python3 -m src.train --model lstm --vocab 1896 --epochs 50 --combined --workers 0
python3 -m src.train --model cnn  --vocab 1896 --epochs 50 --combined --workers 0
python3 -m src.train --model transformer --vocab 1896 --epochs 50 --combined --workers 0 --no-augment
```

---

## Key Design Decisions

**Why not 3D-CNN or ViT?** GPU-only at inference time — cannot meet our CPU latency target.

**Why keypoints instead of raw video?** 126 floats/frame vs 1920×1080×3 pixels. 
100,000x smaller input, 8ms extraction vs 50ms+ video processing, privacy-preserving.

**Why vocab=1896?** Best balance of class density (~28 samples/class) and coverage.
Using all 6,414 WLASL videos across 1,861 classes gives only 3.4 samples/class — too sparse to learn from.

**Why WeightedRandomSampler?** Without it, frequent classes dominate each batch.
Weighted sampling guarantees every class appears equally — critical when class sizes range from 10 to 100+ samples.

---

## Git Workflow

- `main` — stable, working code only
- `dev` — integration branch
- `train` — active training branch (Radhika)
- Feature branches: `feature/your-name-description`
- Open a PR → at least **one review** → merge to `dev`, then `main`
- **Never commit** `cookies.txt`, `failed_downloads.jsonl`, or anything in `data/raw/*/videos/` or `data/processed/*/` (keypoints on HuggingFace instead)

---

## Key References

- [WLASL Dataset](https://github.com/dxli94/WLASL) — Li et al., WACV 2020
- [ASL Citizen](https://huggingface.co/datasets/google/asl-citizen) — Desai et al., 2023
- [Aslense / American Sign Language Dataset](https://huggingface.co/datasets/akasheroor/American-Sign-Language-Dataset) — Mittha, 2025
- [SPOTER](https://github.com/matyasbohacek/spoter) — Bohácek & Hrúz, WACV 2022
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) — Google, 2023
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube downloader used for video acquisition