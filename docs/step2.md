# Step 2 — Model Architecture, Training Pipeline, and Web Infrastructure
**Owner: Radhika Khurana**
**Date: April 2026**
**Branch: dev**

This document describes all changes made in step 2. It is written so that any team member can understand what changed, why, and how to use it — even if you weren't part of this session.

Changes are split by ownership. If something touched your file, there's a section for you at the bottom explaining exactly what changed and why.

---

## Background: Why Step 2 Was Needed

After step 1 (data pipeline + basic training), the first full training run completed with **6.7% Top-1 accuracy on the WLASL validation set** (vocab=300). That's roughly 20x better than random chance (0.33%), so the model is learning, but it's far below the 60% target.

The root cause was the loss function. We were using **CTC loss**, which is designed for streaming sequence-to-sequence tasks — for example, mapping a continuous audio clip to a sequence of characters in speech recognition. Our task is different: each video clip corresponds to exactly one word label. Using CTC for this is like using a sledgehammer to hang a picture frame — it technically works but is wrong for the job. **Cross-entropy with mean pooling** is the correct approach for isolated word classification, and the switch is expected to bring accuracy to 20–40%.

Step 2 also adds the full web infrastructure: a FastAPI server and browser frontend so the model can run as a real-time web app — which is the direction this project is headed.

---

## Change 1: Cross-Entropy Loss (Most Important)

**Files:** `src/model.py`, `src/train.py`, `src/evaluate.py`

### What changed

A new model class `SignClassifier` was added to `src/model.py`. It has the **same transformer encoder** as the existing `SignTransformer` (same number of layers, same attention heads, same parameters) but with a different output head:

| Model | Output shape | Loss | Use case |
|-------|-------------|------|---------|
| `SignTransformer` | `(T, B, C+1)` | CTC | Future: continuous/streaming signing |
| `SignClassifier` | `(B, C)` | CrossEntropy | Now: isolated word classification |

The difference is at the end of the forward pass:

**SignTransformer (old):**
```python
x = self.classifier(x)    # (B, T, C+1)
x = x.permute(1, 0, 2)   # (T, B, C+1) for CTCLoss
return F.log_softmax(x, dim=-1)
```

**SignClassifier (new):**
```python
x = self.encoder(x, ...)   # (B, T, d_model)

# Global average pooling — averages all frame embeddings into one vector
# Padded frames are excluded using the mask
mask = (~src_key_padding_mask).float().unsqueeze(-1)  # (B, T, 1)
x    = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, d_model)

return self.classifier(x)  # (B, C) — one score per class
```

Instead of asking "what is the most likely sequence of labels over T frames?", it asks "given the average representation of all frames, what is the most likely word?" — which is exactly the right question for isolated word classification.

### Why mean pooling

Mean pooling is a standard way to convert a variable-length sequence into a fixed-size representation. It's equivalent to asking: "across all the frames in this clip, what sign was most consistently represented?" The transformer's self-attention already lets each frame incorporate information from all other frames, so by the time we pool, each frame's representation already has global context. We just average them.

Alternative approaches (CLS token, max pooling, attention pooling) all exist but mean pooling is simple and works well empirically for this type of task.

### Label smoothing

Cross-entropy training also uses `label_smoothing=0.1`. This means instead of training the model to output probability 1.0 for the correct class and 0.0 for everything else, it trains toward 0.9 for the correct class and 0.1 / (C-1) for the others. This prevents overconfident predictions and improves generalization on small datasets like ours.

### Where to find it

- New class: `src/model.py` — `SignClassifier` (around line 120)
- New builder: `build_student_classifier(n_classes=300, input_dim=126)`
- Old CTC model is unchanged: `SignTransformer`, `build_student_model()`

---

## Change 2: `--loss` Flag in train.py

**File:** `src/train.py`

A `--loss` flag was added to `train.py` with two options:

```bash
--loss ce     # cross-entropy (default) — use SignClassifier
--loss ctc    # CTC — use SignTransformer (kept for future continuous signing work)
```

Cross-entropy is now the default. You don't need to specify it explicitly.

### Checkpoint naming

Every experiment produces a uniquely named checkpoint so runs never overwrite each other:

| Command | Checkpoint saved |
|---------|-----------------|
| `--model transformer` (CE, default) | `transformer_d128_l3_v300_combined_best.pt` |
| `--model transformer --loss ctc` | `transformer_d128_l3_v300_ctc_combined_best.pt` |
| `--model lstm` | `lstm_h128_l2_v300_combined_best.pt` |
| `--model cnn` | `cnn_d128_l4_v300_combined_best.pt` |
| `--no-augment` | adds `_noaug` to the name |

Each run also saves:
- `results/metrics/{run_name}_history.json` — loss, Top-1, Top-5 per epoch
- `results/metrics/{run_name}_per_class.json` — per-class accuracy breakdown at end of training

### Per-class breakdown

At the end of every training run, `train.py` now prints which signs the model is best and worst at, and saves a full per-class JSON. This tells us which signs to focus on and which have too little data. Example output:

```
Top-10 classes:
  book                 14/16  (88%)
  help                 12/15  (80%)
  ...
Bottom-10 classes:
  grandmother          0/2    (0%)
  ...
Classes with 0% accuracy: 47/300
```

---

## Change 3: Augmentation Fixes and Holistic Support

**Files:** `src/augmentations.py`, `src/config.py`

### What was fixed

`normalize_keypoints()` previously only did translation normalization (subtract wrist position). Scale normalization (divide by hand span) was added. Without scale normalization, the model sees different-sized hands as different features — a signer with larger hands would look different from one with smaller hands even signing the same word. After normalization, both look the same.

`flip_keypoints()` was fixed to correctly mirror pose landmarks when holistic input is used (see below).

### Holistic input support (225-dim)

The existing pipeline produces **126-dim** keypoints (left hand + right hand). A planned upgrade adds full body pose from MediaPipe Holistic to produce **225-dim** keypoints:

```
126-dim (current):  [left_hand × 63,  right_hand × 63]
225-dim (holistic): [left_hand × 63,  right_hand × 63,  pose × 99]
```

The pose component gives the model access to arm position, body lean, and head location — all meaningful in ASL. For example, signs produced near the forehead vs. the chest vs. the hip are different words.

Both `normalize_keypoints()` and `flip_keypoints()` now handle both 126 and 225 automatically — they detect which input format is being used based on `kpts.shape[1]`.

Two new constants were added to `src/config.py`:
```python
HOLISTIC_DIM  = 225     # hands (126) + pose (99)
POSE_FLIP_PAIRS = [(11,12),(13,14),...]  # landmark pairs swapped on horizontal flip
```

The 126-dim pipeline is completely unchanged — this is opt-in.

---

## Change 4: evaluate.py Fixes

**File:** `src/evaluate.py`

Two fixes:

1. **`per_class=True` support** — train.py calls `evaluate(..., per_class=True)` at the end of training to get the class-by-class breakdown. This parameter was missing from evaluate.py, causing a crash at the end of every training run. Fixed.

2. **CE evaluation path** — evaluate.py now handles both model output formats:
   - CE model outputs `(B, C)` → `argmax(dim=1)` for Top-1, `topk(5, dim=1)` for Top-5
   - CTC model outputs `(T, B, C+1)` → greedy decode then check

The `evaluate()` function signature is:
```python
evaluate(model, loader, device, vocab_size, per_class=False, loss_type="ce")
```

---

## Change 5: FastAPI WebSocket Server

**File:** `src/server.py` *(new file)*

This is the backend for the web demo. It's a FastAPI application with a WebSocket endpoint that:

1. Receives one frame at a time from the browser as a JSON message: `{"type": "frame", "keypoints": [126 floats]}`
2. Accumulates frames into a sliding window buffer (up to `MAX_SEQ_LEN = 150` frames)
3. Every 6 frames (~200ms at 30fps), runs ONNX inference on the current window
4. Applies `normalize_keypoints()` before inference (same normalization as training)
5. Applies majority-vote smoothing over the last `SMOOTH_WINDOW = 5` predictions
6. Sends back: `{"prediction": "hello", "confidence": 0.8, "buffer_fill": 1.0}`

The server also serves the frontend HTML at `GET /`.

**Privacy:** The browser only sends 126 floats per frame — the hand keypoint coordinates. No video, no images, no audio ever leaves the browser. This is what makes this viable as a privacy-respecting web tool.

### Architecture

```
Browser
  ├── Webcam → MediaPipe.js → 126 floats/frame
  └── WebSocket → FastAPI server
                    ├── Buffer accumulation
                    ├── normalize_keypoints()
                    ├── ONNX inference (~5ms)
                    └── Prediction → WebSocket → Browser UI
```

### How to run

```bash
# Install server dependencies (once)
pip install fastapi "uvicorn[standard]"

# Start the server (from project root)
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in any browser.

The server loads `models/sign_model.onnx` and `data/processed/vocab.json` at startup. If the ONNX model doesn't exist yet, it logs a warning and returns an error to any connecting browser. Export it first with `python3 -m src.export`.

---

## Change 6: Web Frontend

**File:** `frontend/index.html` *(new file)*

A single HTML file that runs the entire browser side of the demo. No build step, no npm, no dependencies to install — it loads MediaPipe.js from CDN.

**What it does:**
- Requests webcam access from the browser
- Loads MediaPipe HandLandmarker (Tasks Vision JS) — the same model used in `src/keypoints.py`
- Each video frame: runs hand detection, extracts 126 keypoint floats, sends to server via WebSocket
- Draws hand skeleton overlay on the video feed
- Displays the predicted word prominently, with a confidence percentage bar
- Appends new predictions to a running transcript (only when confidence ≥ 60% and the word changes)
- Purple bar at the bottom of the video shows buffer fill status — translation starts when the bar is full

**Keypoint format:** The JS frontend builds the same `[lh(63), rh(63)]` vector as the Python pipeline. "Left"/"Right" is labeled from the image perspective (matching the Python Tasks API), not mirrored.

---

## Change 7: Checkpoint Overwrite Protection

**File:** `src/train.py`

A bug caused the best checkpoint to be overwritten whenever a new training run started. The issue: `best_top1` initialized to `0.0` each run, so epoch 1 (e.g. 0.87%) would immediately beat it and write over a previously saved 11.4% checkpoint.

**Fix:** At the start of training, if a checkpoint already exists for the current `run_name`, read its `top1` and initialize `best_top1` from it. A new run only saves if it genuinely beats the prior best.

```python
ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
if ckpt_path.exists():
    existing  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_top1 = existing.get("top1", 0.0)
    print(f"Existing checkpoint found: Top-1 = {best_top1:.3f} — will only overwrite if beaten.")
```

---

## Change 8: `--dropout` and `--d_model`/`--n_layers` Flags

**File:** `src/train.py`, `src/model.py`

New CLI flags for hyperparameter tuning without editing code:

```bash
--dropout 0.3      # default 0.1 — try 0.3 for stronger regularization on small datasets
--d_model 256      # default 128 — larger model dimension
--n_layers 4       # default 3 — more transformer layers
```

Different `d_model`/`n_layers` values produce unique checkpoint names (e.g. `transformer_d256_l4_v300_combined_best.pt`) so they never collide with the default run.

`build_student_classifier()` in `src/model.py` updated to accept these params and auto-set `nhead` (4 for d≤128, 8 for d≥256) and `dim_feedforward = d_model * 2`.

To try the teacher model (18M params, much larger):
```bash
python3 -m src.train --model transformer --teacher --vocab 300 --epochs 150 --combined
```

---

## Change 9: WeightedRandomSampler

**File:** `src/dataloader.py`

Previously, training batches were random — frequent classes appeared more often than rare ones. With a long-tail class distribution this means the model sees common classes many times per epoch and rare classes almost never.

`WeightedRandomSampler` fixes this by giving each sample a weight inversely proportional to its class frequency. Every class gets approximately equal representation in each batch regardless of how many samples it has.

```python
labels       = [s["label_idx"] for s in dataset.samples]
class_counts = Counter(labels)
weights      = [1.0 / class_counts[l] for l in labels]
sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
```

This replaces `shuffle=True` in the training DataLoader — `shuffle` and `sampler` are mutually exclusive in PyTorch. Val/test loaders are unchanged.

No CLI changes needed. Takes effect automatically on the next training run.

---

## Change 10: Scale Augmentation

**File:** `src/augmentations.py`

A new augmentation — `scale_augment()` — randomly multiplies all keypoint coordinates by a uniform factor between 0.85 and 1.15. This simulates signers with different hand sizes and at different distances from the camera.

It runs *before* `normalize_keypoints()`, so the scale factor is absorbed by the hand-span normalization step and doesn't cause the model to see unnormalized inputs.

```python
def scale_augment(kpts, min_scale=0.85, max_scale=1.15):
    scale = np.random.uniform(min_scale, max_scale)
    return (kpts * scale).astype(np.float32)
```

Added to the augmentation pipeline between temporal jitter and Gaussian noise:
```
flip → speed_perturb → temporal_jitter → scale_augment → gaussian_noise → normalize
```

---

## Minor Fixes (Gyula's Files — Bug Only, No Logic Changes)

These are small fixes in files outside Radhika's scope. They fix bugs rather than add features.

### `src/demo.py`
The inference call was missing `normalize_keypoints()`. The model was trained with normalization applied to every sample, but demo.py was sending raw (unnormalized) keypoints to the ONNX model. This meant the model was seeing a different data distribution at inference than at training, which degraded predictions significantly.

**Fix:** Added one line before inference:
```python
seq_arr = normalize_keypoints(np.stack(list(frame_buffer), axis=0).astype(np.float32))
```

### `src/export.py`
The `--vocab` default was 100. Since all our training uses `--vocab 300`, exporting without explicitly passing `--vocab 300` would produce the wrong output dimension. Fixed default to 300.

---

## How to Use Everything

### Run the main training experiments

Always run from the project root with `python3 -m`:

```bash
# EXP-006 — Transformer + CE + augmentation (main model)
python3 -m src.train --model transformer --vocab 300 --epochs 50 --combined

# EXP-005 — Transformer, no augmentation (ablation)
python3 -m src.train --model transformer --vocab 300 --epochs 50 --no-augment --combined

# EXP-004 — BiLSTM baseline
python3 -m src.train --model lstm --vocab 300 --epochs 50 --combined

# EXP-003 — 1D CNN baseline
python3 -m src.train --model cnn --vocab 300 --epochs 50 --combined
```

Each run is independent and saves to a unique checkpoint name. You can run them in parallel in separate terminals.

Watch for the `✓ New best` lines — that's when the checkpoint file is updated.

### Fill the results table

After each run finishes, the per-class JSON is at `results/metrics/{run_name}_per_class.json` and the history JSON is at `results/metrics/{run_name}_history.json`. The final Top-1 is also printed in the terminal.

Update `docs/experiments.md` with each result.

### Export to ONNX and start the web demo

```bash
# Step 1: Export the best transformer checkpoint
python3 -m src.export \
  --checkpoint models/checkpoints/transformer_d128_l3_v300_combined_best.pt \
  --vocab 300

# Step 2: Start the web server
uvicorn src.server:app --host 0.0.0.0 --port 8000

# Step 3: Open in browser
open http://localhost:8000
```

### Run holistic preprocessing (future — after current experiments are done)

This extracts 225-dim keypoints that include full body pose. Saves to separate directories so existing files are untouched.

```bash
python3 scripts/preprocess.py --split train --vocab 300 --holistic
python3 scripts/preprocess.py --split val   --vocab 300 --holistic
python3 scripts/preprocess.py --split test  --vocab 300 --holistic
```

Before retraining with holistic input, update `build_student_classifier` call in train.py:
```python
model = build_student_classifier(n_classes=args.vocab, input_dim=225)
```

---

## Expected Accuracy

| Experiment | Model | Loss | Expected Top-1 | Notes |
|---|---|---|---|---|
| EXP-003 | 1D CNN | CE | 12–25% | Local temporal patterns only |
| EXP-004 | BiLSTM | CE | 15–30% | Sequential baseline |
| EXP-005 | Transformer | CE, no aug | 10–25% | Ablation: effect of augmentation |
| EXP-006 | Transformer | CE + aug | 20–40% | Main model |
| — | Transformer | CTC (step 1) | ~6–8% | Already run — baseline for comparison |
| Future | Transformer | CE + holistic | 25–45% | After holistic preprocessing |

Estimates are based on dataset size (~9 samples/class average) and comparable published results on WLASL-300 with keypoint-based models.

---

## Files Changed Summary

| File | Owner | Status | What changed |
|------|-------|--------|-------------|
| `src/model.py` | Radhika | Modified | Added `SignClassifier` + `build_student_classifier()` with `d_model`/`n_layers`/`dropout` params |
| `src/train.py` | Radhika | Modified | CE loss, label smoothing, per-class breakdown, checkpoint overwrite protection, `--dropout`/`--d_model`/`--n_layers` flags, MPS device support |
| `src/evaluate.py` | Radhika | Modified | `loss_type` param, CE eval path, `per_class` support |
| `src/augmentations.py` | Radhika | Modified | Holistic flip/normalize, scale normalization, `scale_augment()` |
| `src/dataloader.py` | Radhika | Modified | `WeightedRandomSampler` for balanced class sampling |
| `src/config.py` | Radhika | Modified | `HOLISTIC_DIM`, `POSE_FLIP_PAIRS` |
| `src/server.py` | Radhika | **New** | FastAPI WebSocket inference server |
| `frontend/index.html` | Radhika | **New** | Browser demo with MediaPipe.js |
| `src/keypoints.py` | Jian | Modified | Added holistic functions at bottom (additive only) |
| `scripts/preprocess.py` | Jian | Modified | Added `--holistic` flag (additive only) |
| `src/demo.py` | Gyula | Modified | Bug fix: added `normalize_keypoints()` before inference |
| `src/export.py` | Gyula | Modified | Bug fix: default `--vocab` 100 → 300 |
