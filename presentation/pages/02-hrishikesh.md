# Pipeline: Webcam to Prediction in <25ms

<div class="grid grid-cols-5 gap-3 mt-8">
<div class="card text-center">
<div class="text-3xl font-bold text-blue-600">01</div>
<div class="font-semibold mt-2">Webcam</div>
<div class="text-sm text-gray-600 mt-2">30fps RGB video</div>
<div class="text-xs text-gray-500">6.2M pixels/frame</div>
</div>

<div class="card text-center">
<div class="text-3xl font-bold text-teal-600">02</div>
<div class="font-semibold mt-2">MediaPipe</div>
<div class="text-sm text-gray-600 mt-2">Keypoint extraction</div>
<div class="text-xs text-gray-500">126 floats · ~8ms · CPU</div>
</div>

<div class="card text-center">
<div class="text-3xl font-bold text-purple-600">03</div>
<div class="font-semibold mt-2">60-Frame Buffer</div>
<div class="text-sm text-gray-600 mt-2">2-second window</div>
<div class="text-xs text-gray-500">Padded to model shape</div>
</div>

<div class="card text-center">
<div class="text-3xl font-bold text-orange-600">04</div>
<div class="font-semibold mt-2">Transformer</div>
<div class="text-sm text-gray-600 mt-2">ONNX inference</div>
<div class="text-xs text-gray-500">~5ms · 1,896 classes</div>
</div>

<div class="card text-center">
<div class="text-3xl font-bold text-red-600">05</div>
<div class="font-semibold mt-2">Prediction</div>
<div class="text-sm text-gray-600 mt-2">Top-3 results</div>
<div class="text-xs text-gray-500">Confidence scoring</div>
</div>
</div>

<div class="grid grid-cols-4 gap-4 mt-8">
<div class="card border-l-4 border-blue-500">
<div class="text-2xl font-bold text-blue-600">&lt;25ms</div>
<div class="text-sm text-gray-600">End-to-end latency</div>
</div>

<div class="card border-l-4 border-teal-500">
<div class="text-2xl font-bold text-teal-600">3MB</div>
<div class="text-sm text-gray-600">ONNX model size</div>
</div>

<div class="card border-l-4 border-purple-500">
<div class="text-2xl font-bold text-purple-600">0</div>
<div class="text-sm text-gray-600">Network requests</div>
</div>

<div class="card border-l-4 border-orange-500">
<div class="text-2xl font-bold text-orange-600">No GPU</div>
<div class="text-sm text-gray-600">CPU only</div>
</div>
</div>


---

# Baselines: CNN vs. LSTM

Two efficient baselines establish a performance spectrum from localized (CNN) to global-but-bottlenecked (LSTM).

<div class="grid grid-cols-3 gap-4 mt-6">
<div>

**CNN (Conv1d)**

```
4 × residual blocks
kernel=3
receptive field: ~21 frames
```

- **Speed**: ~2–3ms (fastest)
- **Parameters**: ~560K
- **Accuracy**: **~31%**
- **Shortcoming**: No long-range temporal context

</div>
<div>

**LSTM (BiLSTM)**

```
2 × stacked layers
hidden=128 → 256 out
bidirectional pass
```

- **Speed**: ~3–5ms
- **Parameters**: ~560K
- **Accuracy**: **~38%**
- **Shortcoming**: Hidden state bottleneck limits long-range reasoning

</div>
<div>

**Transformer**

```
self-attention encoder
d_model=128, 3 layers
4 attention heads
```

- **Speed**: ~5ms
- **Parameters**: ~450K
- **Accuracy**: **~72%**
- **Advantage**: Multi-head attention, no bottlenecks

</div>
</div>
<div class="callout callout-blue mt-1">
Key insight: Sign language needs temporal reasoning across the full sequence*. CNN fails (no context). LSTM succeeds partially (weak long-range). Transformer excels (parallel, multi-headed attention).
</div>


---

# Transformer: Multi-Head Attention

<div class="grid grid-cols-2 gap-4 mt-1">
<div>

**Architecture**

```
Input (B, T, 126)
    ↓
Positional encoding
    ↓
3 × Transformer blocks
  - 4 attention heads
  - d_model=128
  - FFN d_ff=256
    ↓
Mean pooling
    ↓
Linear classifier → C
```
<div class="callout callout-blue mt-6">
Self-attention allows the model to compare any frame to any other frame in parallel — discovering which frames matter.
</div>
</div>
<div>

**Why It Works**

- **Parallel computation**: All frame-to-frame comparisons happen at once (unlike LSTM's sequential gates)
- **Multi-head diversity**: 4 heads learn different temporal patterns
  - Head 1: handshape transitions
  - Head 2: global arm trajectory
  - Head 3: finger motion onset
  - Head 4: release patterns
- **Learned importance**: Attention weights reveal which frames matter *per sign*

</div>
</div>


---

# Attention Weights: Learned Temporal Landmarks

The model discovers *which frames are discriminative* without hand-coded rules.

<div class="grid grid-cols-2 gap-6 mt-2">
<div>
  Example 1: "FRIEND"

| Frame Range | Activity | Attention |
|---|---|---|
| 1–7 | Setup (low relevance) | ▁ Low |
| 8–12 | Hand approach | ▃ High |
| 13–21 | Hand traveling | ▁ Low |
| 22–26 | **Interlocked fingers** | ▇ **Highest** |
| 27–30 | Release motion | ▄ Medium |

</div>
<div>

![Attention visualization for FRIEND and AIRPLANE signs](../images/friend-asl.png)

</div>
</div>


