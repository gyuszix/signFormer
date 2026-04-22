# From PyTorch to CPU: ONNX Export <span style="font-size:0.5rem;opacity:0.4;vertical-align:super;">G</span>

Training produces a `.pt` file requiring the full PyTorch stack (~500MB, GPU assumed). `src/export.py` converts it to a self-contained ONNX binary in three steps.

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

<div class="grid grid-cols-1 gap-3">
<div class="card">

**1. Load** — Read `.pt` checkpoint, auto-detect architecture and loss type from saved args.

</div>
<div class="card">

**2. Export** — `torch.onnx.export` with dynamic axes, opset 17, constant folding enabled.

</div>
<div class="card">

**3. Benchmark** — 200 CPU inference runs. Reports mean, median, p95, p99 before deployment.

</div>
</div>

</div>
<div>

<div class="card mb-4">

**`.onnx` result**
- Only needs `onnxruntime` (~10MB)
- CPU-optimized by default
- Runs on any laptop, kiosk, Raspberry Pi
- Zero internet dependency at runtime

</div>

<div class="grid grid-cols-3 gap-3 text-center">
<div>
<div class="stat-number">0.5ms</div>
<div class="stat-label">Median latency</div>
</div>
<div>
<div class="stat-number">3MB</div>
<div class="stat-label">Model size</div>
</div>
<div>
<div class="stat-number">&lt;25ms</div>
<div class="stat-label">End-to-end</div>
</div>
</div>

</div>
</div>

---

# Results: Full Ablation

All models trained on the same 1,896-class combined corpus. Best checkpoint selected by validation Top-1.

| Model | Top-1 | Top-5 | Params | Notes |
|-------|-------|-------|--------|-------|
| 1D CNN | 38.4% | 61.1% | ~656K | local 3-frame windows |
| BiLSTM | 31.5% | 53.1% | ~1.1M | sequential hidden state |
| Transformer (d=128, aug) | 40.1% | 58.9% | ~658K | augmentation hurt — likely underfit |
| Transformer (d=256, aug) | 44.0% | 59.8% | ~2.6M | larger model, same issue |
| **Transformer (d=128, no aug)** | **67.0%** | **88.6%** | **~658K** | **← demo model** |

<div class="callout callout-blue mt-4">
Augmentation consistently degraded performance at 50 epochs — the model needs significantly more training to benefit from it. No-aug converges faster and stronger.
</div>

---

# What the Numbers Actually Look Like

Results from our best model — `transformer_d128_l3_v1896_noaug` (67% Top-1) — averaged across all 1,896 classes, including rare signs with only 1–2 training examples.

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

**Accuracy distribution across all classes**

| Accuracy range | Signs (out of 1,896) |
|----------------|----------------------|
| 100% | 653 |
| 60–99% | 702 |
| 20–59% | 405 |
| 0% | 136 |

</div>
<div>

**What drives the 0% classes**

Nearly all 136 zero-accuracy classes have fewer than 3 training examples — the model simply hasn't seen enough of them.

The 71.5% of classes above 60% accuracy covers the vast majority of signs that appear with any regularity in the dataset.

</div>
</div>

<div class="callout callout-blue mt-4">
71.5% of classes score above 60%. Only 7.2% sit at 0% — almost entirely low-frequency signs with insufficient training data.
</div>

---

# Results in Context

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

| Model | Top-1 | Hardware | Input |
|-------|-------|----------|-------|
| I3D | ~60% | GPU | Raw video |
| VideoMAE | ~65% | GPU | Raw video |
| SPOTER | ~60% | GPU | Keypoints |
| **Ours (no aug)** | **67.0%** | **CPU** | **Keypoints** |

</div>
<div>

### The key distinction

Every competitive model requires a GPU and operates on raw video frames. Ours is the only model in this range that:

- Runs entirely on CPU
- Uses keypoints only — no raw video stored
- Operates in real time at 30fps
- Fits in 3MB

</div>
</div>

<div class="quote mt-6">
67% on CPU with keypoints — matching or beating GPU video models while running on a laptop webcam.
</div>

---

# Real-Time Demo Architecture

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

### The pipeline
1. Webcam captures frame at 30fps
2. MediaPipe extracts 126 keypoints (~8ms)
3. Keypoints pushed into a 60-frame rolling buffer
4. Every 0.5s — if buffer is full — ONNX model infers (~0.5ms)
5. Softmax confidence checked against threshold
6. Prediction added to 5-inference majority vote
7. Predicted word shown on screen

</div>
<div>

### Stability guards
- **Hand ratio gate** — ≥50% of buffer frames must have hands detected
- **Confidence threshold** — tunable at runtime via `--threshold`
- **Reset on absence** — 20 frames without hands clears history
- **Majority vote** — smooths over 5 consecutive inferences

</div>
</div>

---

# Live Demo

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

### What to watch
- Hand skeleton overlay (navy = left, azure = right)
- Buffer counter filling to 60
- Hands % — must hit ≥50% to trigger inference
- Confidence bar — model's softmax certainty
- Predicted word centered on screen

</div>
<div>

<video controls style="width:100%;border-radius:8px;margin-bottom:0.75rem;">
  <source src="/videos/demo.mp4" type="video/mp4" />
</video>

**Model:** `transformer_d128_l3_v1896_noaug`  
**Top-1:** 67.0% · **Latency:** 0.5ms · **Vocab:** 1,896 signs

</div>
</div>

<div class="quote mt-6">
Press Q to quit. Move your hand out of frame for ~1 second to reset between signs.
</div>
