# From PyTorch to CPU: ONNX Export <span style="font-size:0.5rem;opacity:0.4;vertical-align:super;">G</span>

Training produces a `.pt` checkpoint requiring the full PyTorch stack (~500MB, GPU assumed). `src/export.py` converts it to a self-contained ONNX binary in three steps.

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

<div class="grid grid-cols-1 gap-3">
<div class="card">

**1. Load** — Read `.pt` checkpoint, auto-detect architecture and loss type from saved args.

</div>
<div class="card">

**2. Export** — `torch.onnx.export` with dynamic axes on batch and sequence dimensions, opset 17, constant folding enabled.

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
<div class="stat-number">0.6ms</div>
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

# Failure Mode Analysis

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

**Systematic confusions**

| Confused pair | Reason |
|--------------|--------|
| MOTHER / FATHER | Same handshape, different position |
| WEEK / NEXT-WEEK | Same motion, one delayed |
| HELP / ASSIST | Near-identical hand configuration |
| APPLE / ONION | Both twist at cheek |

</div>
<div>

**Environmental failures**

The hand detector is the weakest link — not the classifier.

- **Poor lighting** — keypoints become noisy
- **Partial occlusion** — one hand exits frame mid-sign
- **Fast signers** — sign completes before buffer fills
- **Background clutter** — detection confidence degrades

<div class="callout callout-yellow mt-3">
72.8% Top-1 is conditioned on clean keypoint extraction — when MediaPipe fails, the model never sees valid input.
</div>

</div>
</div>

---

# Results in Context

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

| Model | Top-1 | Hardware | Input |
|-------|-------|----------|-------|
| I3D | ~60% | GPU | Raw video |
| VideoMAE | ~65% | GPU | Raw video |
| SPOTER | ~60% | GPU | Keypoints |
| **d=128 + MAE, no aug** | **71.5%** | **CPU** | **Keypoints** |
| **d=256, no aug** | **72.8%** | **CPU** | **Keypoints** |

</div>
<div>

**The key distinction**

Every competitive model requires a GPU and raw video. Ours is the only model in this range that:

- Runs entirely on CPU
- No raw video stored or transmitted
- Real time at 30fps · fits in 3MB
- Fully offline — zero network requests

</div>
</div>

<div class="quote mt-3">
72.8% on CPU with keypoints — matching or beating GPU video models on a laptop webcam.
</div>

---

# Real-Time Demo Architecture

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

### The pipeline
1. Webcam captures frame at 30fps
2. MediaPipe extracts 126 keypoints (~8ms)
3. Keypoints pushed into a **60-frame rolling buffer**
4. Every 0.5s — if buffer full — ONNX model infers (~0.6ms)
5. Softmax confidence checked against threshold
6. Prediction added to 5-inference majority vote
7. Predicted word shown on screen

</div>
<div>

### Stability guards

- **Hand ratio gate** — ≥50% of buffer frames must have hands detected
- **Confidence threshold** — tunable at runtime via `--threshold`
- **Reset on absence** — 20 frames without hands clears history
- **Majority vote** — smooths over 5 consecutive inferences to suppress flicker

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

**Model:** `transformer_d256_l3_v1896_noaug`  
**Top-1:** 72.8% · **Latency:** 0.6ms · **Vocab:** 1,896 signs

</div>
</div>

<div class="quote mt-6">
Press Q to quit. Move your hand out of frame for ~1 second to reset between signs.
</div>

---

# Future Work

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

**Model improvements**

- Continuous signing (not just isolated words)
- Sentence-level language model for context
- Cross-lingual extension (BSL, LSF, ASL regional variants)
- Larger vocabulary (5,000+ signs)

</div>
<div>

**Deployment improvements**

- Mobile app (iOS/Android via ONNX Mobile)
- WebAssembly for browser deployment
- Edge device optimization (Raspberry Pi, Jetson Nano)
- Signed language → text → speech pipeline

</div>
</div>

<div class="quote mt-8">
The bottleneck is no longer computation — it's data. Real-time ASL recognition at scale is now achievable on commodity hardware. The next challenge is getting there for 1,896 → 10,000 signs with signer diversity.
</div>

---

<div class="h-full flex flex-col justify-center items-center text-center">

<div class="mono-label !text-teal-600 mb-4">Northeastern University · Spring 2026</div>

<h1 class="text-5xl font-bold mb-6 tracking-tighter gradient-text">Thank You</h1>

<p class="text-slate-500 text-lg font-light mb-8 max-w-lg leading-relaxed">
Real-time ASL recognition on CPU — no gloves, no GPU, no cloud.
</p>

<div class="grid grid-cols-4 gap-4 w-full max-w-2xl">
<div class="card text-center py-4">
<div class="text-3xl font-bold text-teal-600 tracking-tight">72.8%</div>
<div class="text-xs font-semibold text-slate-700 mt-1">Top-1 Accuracy</div>
<div class="text-xs text-slate-400 mt-1">1,896 classes · d=256</div>
</div>
<div class="card text-center py-4">
<div class="text-3xl font-bold text-slate-800 tracking-tight">100%</div>
<div class="text-xs font-semibold text-slate-700 mt-1">Top-500 Signs</div>
<div class="text-xs text-slate-400 mt-1">Everyday ASL mastered</div>
</div>
<div class="card text-center py-4">
<div class="text-3xl font-bold text-teal-600 tracking-tight">&lt;25ms</div>
<div class="text-xs font-semibold text-slate-700 mt-1">End-to-End</div>
<div class="text-xs text-slate-400 mt-1">CPU only · no GPU</div>
</div>
<div class="card text-center py-4">
<div class="text-3xl font-bold text-slate-800 tracking-tight">125</div>
<div class="text-xs font-semibold text-slate-700 mt-1">Zero-acc Classes</div>
<div class="text-xs text-slate-400 mt-1">Down from 749 · −83%</div>
</div>
</div>

<div class="flex flex-col items-center mt-6">
  <img src="/images/qrcode.png" alt="QR code" style="width:100px;height:100px;object-fit:contain;border-radius:6px;" />
  <div class="mono-label !text-slate-400 mt-2">Read the full article on Medium</div>
</div>

</div>
