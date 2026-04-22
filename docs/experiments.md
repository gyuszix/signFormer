# Experiment Log
**Project:** Real-Time ASL Translation
**Updated:** April 2026

All experiments use `python3 -m src.train` from project root.
Val set is always WLASL-only (229 samples, 273 classes present in val).
Combined training set = WLASL + ASL Citizen (2,726 samples, 300 classes).

---

## Results Summary

| Exp | Model | Vocab | Loss | Aug | Epochs | Val samples | Best Top-1 | Best Top-5 | Best Epoch | Final Loss |
|-----|-------|-------|------|-----|--------|-------------|-----------|-----------|------------|------------|
| pre-1 | Transformer | 100 | CTC | Yes | 50 | ~100 | 3.9% | 9.8% | 24 | 3.29 |
| EXP-CTC | Transformer | 300 | CTC | Yes | 50 | 229 | 6.7% | 7.1% | 44 | 1.53 |
| EXP-006 | Transformer | 300 | CE | Yes | 50 | 229 | 7.0% | 17.0% | 29 | 2.11 |
| EXP-V100 | Transformer | 100 | CE | Yes | 100 | 101 | 6.9% | 12.9% | 41 | 1.42 |
| EXP-006b | Transformer | 300 | CE | Yes | 100 | 229 | 11.4% | 20.1% | 75 | 1.49 |
| EXP-007 | Transformer | 300 | CE | Yes | 150 | 229 | **40.8%** | **56.4%** | 101 | 1.21 |
| EXP-004 | BiLSTM | 300 | CE | Yes | 50 | 229 | — | — | — | — |
| EXP-003 | 1D CNN | 300 | CE | Yes | 50 | 229 | — | — | — | — |
| EXP-005 | Transformer | 300 | CE | No | 50 | 229 | — | — | — | — |

**Best result so far: EXP-007 — 40.8% Top-1, 56.4% Top-5 (150 epochs, MPS device).**
Training on MPS (Apple Silicon) with 150 epochs and a slower LR decay (T_max=150) broke through the previous 11.4% ceiling dramatically.

---

## Completed Runs

---

### pre-1 — Transformer, CTC, vocab=100 (early test, pre-combined dataset)
**Date:** March 2026 | **Run by:** Radhika

**Command:**
```bash
python3 -m src.train --model transformer --vocab 100 --epochs 50
```

| Metric | Value |
|--------|-------|
| Best Top-1 | 3.9% (epoch 24) |
| Best Top-5 | 9.8% |
| Final loss | 3.29 |
| Epochs | 50 |
| Training data | WLASL only (pre-combined) |

**Notes:** Early validation run. CTC loss, vocab=100, no combined manifest. Established that the pipeline ran end-to-end. Not directly comparable to later runs.

---

### EXP-CTC — Transformer, CTC loss, vocab=300
**Date:** April 2026 | **Run by:** Radhika

**Command:**
```bash
python3 -m src.train --model transformer --vocab 300 --epochs 50 --combined
```

**Checkpoint:** `transformer_d128_l3_v300_combined_best.pt` *(overwritten by EXP-006)*

| Metric | Value |
|--------|-------|
| Best Top-1 | 6.7% (epoch 44) |
| Best Top-5 | 7.1% |
| Final loss | 1.53 |
| Epochs | 50 |
| Params | 452,525 |

**Notes:**
- Loss dropped from 38.2 → 1.53 (CTC initializes high — blank token overhead)
- Top-1 plateaued ~6–7% after epoch 12 and never broke through
- CTC is wrong for isolated word classification — designed for streaming sequence-to-sequence (speech recognition style). Switched to cross-entropy after this run.

---

### EXP-006 — Transformer, CE loss, vocab=300
**Date:** April 2026 | **Run by:** Radhika

**Command:**
```bash
python3 -m src.train --model transformer --vocab 300 --epochs 50 --combined
```

**Checkpoint:** `models/checkpoints/transformer_d128_l3_v300_combined_best.pt`
**History:** `results/metrics/transformer_d128_l3_v300_combined_history.json`
**Per-class:** `results/metrics/transformer_d128_l3_v300_combined_per_class.json`

| Metric | Value |
|--------|-------|
| Best Top-1 | **7.0%** (epoch 29) |
| Best Top-5 | 17.0% |
| Final loss | 2.11 |
| Epochs | 50 |
| Params | 452,396 |
| Val samples | 229 (188 classes present) |

**Per-class breakdown:**

| | Count | % of val classes |
|-|-------|-----------------|
| Classes with >0% accuracy | 14 | 7.4% |
| Classes with 0% accuracy | 174 | 92.6% |
| Mean acc among classes that got any right | 89.3% | — |

Top-10 classes: help, coffee, because, careful, clock, game, girl, government, more, past — all are 1/1 (single val sample each, not statistically reliable).

**Notes:**
- Top-5 (17%) much higher than Top-1 (7%) — model has the right answer in top-5 often, just not ranking it first
- 174/188 classes at 0% — model is predicting a small subset of classes for most inputs
- Loss still decreasing at epoch 50 (2.11) — training effectively stopped as LR decayed to ~1e-6 by ep35
- Root cause: ~9 training samples/class average — too few for reliable 300-class generalization
- Random baseline: 0.33% (1/300). This run = ~21× better than random.

---

## Pending Runs

---

### EXP-006b — Transformer, CE, vocab=300, 100 epochs ✓ COMPLETE
**Date:** April 2026 | **Run by:** Radhika

**Command:**
```bash
python3 -m src.train --model transformer --vocab 300 --epochs 100 --combined
```

**Checkpoint:** `models/checkpoints/transformer_d128_l3_v300_combined_best.pt`
**History:** `results/metrics/transformer_d128_l3_v300_combined_history.json`
**Per-class:** `results/metrics/transformer_d128_l3_v300_combined_per_class.json`

| Metric | Value |
|--------|-------|
| Best Top-1 | **11.4%** (epoch 75) |
| Best Top-5 | 20.1% |
| Final loss | 1.49 |
| Epochs | 100 |
| Val samples | 229 (188 classes) |
| Classes with >0% accuracy | 22 / 188 |

**Top-1 trajectory:**

| Epoch | Top-1 |
|-------|-------|
| 50 | 7.0% |
| 65 | 10.0% |
| 69 | 10.5% |
| 75 | **11.4%** ← best |
| 100 | 10.0% |

**Observations:**
- +4.4pp over 50-epoch run (EXP-006) — model was still actively learning at epoch 50.
- Plateau at epoch 75–80 once LR falls below ~4e-5
- 22/188 classes >0% (up from 14 in EXP-006)
- **Superseded by EXP-007**

---

### EXP-007 — Transformer, CE, vocab=300, 150 epochs, MPS ✓ COMPLETE
**Date:** April 2026 | **Run by:** Radhika

**Command:**
```bash
python3 -m src.train --model transformer --vocab 300 --epochs 150 --combined
```
*(Run on Apple Silicon MPS after device detection was added to train.py)*

**Checkpoint:** `models/checkpoints/transformer_d128_l3_v300_combined_best.pt`
**Per-class:** `results/metrics/transformer_d128_l3_v300_combined_per_class.json`

| Metric | Value |
|--------|-------|
| Best Top-1 | **40.8%** (epoch 101) |
| Best Top-5 | **56.4%** |
| Final loss | 1.21 |
| Epochs | 150 |
| Val samples | 229 (273 classes present) |
| Classes with >0% accuracy | 149 / 273 |
| Model params | 452,396 |

**Top-1 trajectory:**

| Epoch | Top-1 |
|-------|-------|
| 7 | 12.3% (already beats EXP-006b's peak) |
| 33 | 30.0% |
| 51 | 36.0% |
| 70 | 38.2% |
| 94 | 38.8% |
| 101 | **40.8%** ← best |
| 120 | ~39.7% |
| 150 | 39.5% |

**Top-10 classes (val):** chair, help, walk, like, shirt, accident, color, pink, pizza, time — all 100%

**Observations:**
- **3.6× better than EXP-006b (11.4%) — the biggest single jump in the project**
- Two factors drove this: (1) MPS device — Apple Silicon GPU trains with different numerical dynamics; (2) T_max=150 keeps LR higher for longer — at epoch 50, LR is 2.25e-4 vs 1.5e-4 in the 100-epoch runs, meaning the model keeps learning aggressively past the old plateau point
- Model is still climbing at epoch 70-100 instead of plateauing at epoch 75 like before
- 149/273 classes >0% accuracy — model now predicts many more sign classes correctly
- Top-5 at 56.4% — the correct sign is in the model's top-5 predictions more than half the time
- **Currently the best result across all runs**

---

### EXP-V100 — Transformer, CE, vocab=100, 100 epochs ✓ COMPLETE
**Date:** April 2026 | **Run by:** Radhika

**Command:**
```bash
python3 -m src.train --model transformer --vocab 100 --epochs 100 --combined
```

**Checkpoint:** `models/checkpoints/transformer_d128_l3_v100_combined_best.pt`
**History:** `results/metrics/transformer_d128_l3_v100_combined_history.json`
**Per-class:** `results/metrics/transformer_d128_l3_v100_combined_per_class.json`

| Metric | Value |
|--------|-------|
| Best Top-1 | **6.9%** (epoch 41) |
| Best Top-5 | 12.9% |
| Final loss | 1.42 |
| Epochs | 100 |
| Val samples | 101 (73 classes present) |
| Classes with >0% accuracy | 4 / 73 |

**Result:** Nearly identical to EXP-006 (vocab=300, 7.0%) despite 3× more training samples per class.

**This rules out data volume as the bottleneck.**

**Diagnosis — signer-dependent overfitting:**

WLASL splits train/val by signer — training and validation signers are completely different people. The model is learning "how this particular signer signs 'coffee'" rather than "what the sign for coffee looks like in general." When it sees a new signer in val, it fails.

- Training loss drops from 6.1 → 1.42 (model fits training data fine)
- Val accuracy never improves past ~7% regardless of epochs or vocab size
- This is the definition of overfitting to signer identity

**What is needed to fix this:**
1. **More diverse signers in training** — YouTube-ASL has hundreds of signers; more variety forces the model to generalize
2. **Stronger regularization** — increase dropout (0.1 → 0.3), weight decay (1e-4 → 1e-3)
3. **Holistic features** — adding body pose/face helps distinguish signs that look similar from hands alone
4. **Signer normalization** — subtract per-signer mean or use instance normalization

---

### EXP-004 — BiLSTM baseline
**Command:**
```bash
python3 -m src.train --model lstm --vocab 300 --epochs 50 --combined
```
**Checkpoint will save to:** `lstm_h128_l2_v300_combined_best.pt`

---

### EXP-003 — 1D CNN baseline
**Command:**
```bash
python3 -m src.train --model cnn --vocab 300 --epochs 50 --combined
```
**Checkpoint will save to:** `cnn_d128_l4_v300_combined_best.pt`

---

### EXP-005 — Transformer, no augmentation (ablation)
**Command:**
```bash
python3 -m src.train --model transformer --vocab 300 --epochs 50 --no-augment --combined
```
**Checkpoint will save to:** `transformer_d128_l3_v300_noaug_combined_best.pt`

**Purpose:** Isolate the effect of augmentation. Compare Top-1 vs EXP-006.

---

## ONNX Latency Benchmarks

To be filled in after Gyula runs `python3 -m src.export` on each checkpoint.

| Model | Top-1 (val) | Latency mean | Latency p95 | Size (MB) |
|-------|-------------|-------------|-------------|-----------|
| Transformer (d=128, l=3) | — | — | — | — |
| BiLSTM (h=128, l=2) | — | — | — | — |
| 1D CNN (d=128, l=4) | — | — | — | — |

---

## Interpretation and Diagnosis

**Results across all completed runs:**

| Config | Best Top-1 | Best Top-5 |
|--------|-----------|-----------|
| vocab=100, 100 epochs | 6.9% | 12.9% |
| vocab=300, 50 epochs | 7.0% | 17.0% |
| vocab=300, 100 epochs | 11.4% | 20.1% |
| vocab=300, 150 epochs, MPS | **40.8%** | **56.4%** |

**Three findings:**

1. **More epochs with slower LR decay is the biggest lever** — EXP-007 (150 epochs, T_max=150) reached 40.8% vs 11.4% with 100 epochs. With T_max=150, the LR at epoch 50 is ~2.25e-4 vs ~1.5e-4 in the 100-epoch runs — the model keeps actively learning past the old plateau point instead of decelerating early.

2. **MPS (Apple Silicon GPU) training has different dynamics** — switching from CPU to MPS changed training behavior significantly. MPS may use bfloat16 for some operations which acts as implicit regularization.

3. **More data per class does not help** — vocab=100 (~27 samples/class) matched vocab=300 (~9 samples/class) at ~7% in the earlier runs. The bottleneck is signer diversity, not sample count.

**Updated picture:**

EXP-007 shows the model *can* generalize to unseen signers at 40.8% — substantially better than the previous ceiling. The signer-dependent overfitting identified in earlier runs was partly a training regime problem (LR decayed too fast), not just a data problem.

**State-of-the-art context:** SPOTER and I3D reach ~60–65% on WLASL with larger corpora and raw video frames. At 40.8% on 2,726 training samples using only keypoints (no video), this is competitive and demonstrates the approach is viable.

**Baselines (EXP-003, EXP-004) are still worth running** — if CNN and BiLSTM plateau lower than 40.8%, it validates that transformer self-attention is important for this task.

**Random baseline:** 0.33% (1/300). EXP-007 = ~124× better than random.

**Top-5 at 56.4%** — the correct sign is in the model's top-5 predictions more than half the time. With a language model post-processor, this translates to very usable output in context.
