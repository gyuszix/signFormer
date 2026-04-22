# The Problem: 70 Million People. No Accessible AI.

Over 70 million deaf and hard-of-hearing people use sign language as their primary language. Only ~2% of hearing people know any sign language.

<div class="grid grid-cols-2 gap-6 mt-6">
<div class="grid grid-cols-1 gap-4">
<div class="card">

**70M+ Signers**

Only ~2% of hearing people know any sign language. Communication barriers are constant and pervasive.

</div>
<div class="card">

**GPU-Only Models**

State-of-the-art video models (I3D, SlowFast, VideoMAE) require dedicated NVIDIA hardware — unusable on a laptop, phone, or kiosk.

</div>
</div>
<div class="grid grid-cols-1 gap-4">
<div class="card">

**Latency Bottleneck**

Processing 6.2M pixels per frame in real time is computationally prohibitive on commodity CPUs.

</div>
<div class="card">

**Our Goal**

Webcam-only, real-time ASL word recognition: no gloves, no special hardware. Raw RGB video → predicted ASL word, live, on any modern CPU.

</div>
</div>
</div>

<!--
Sign language is the primary language for over 70 million people worldwide — not a secondary mode of communication, their first language. The problem isn't that tools don't exist, it's that every tool that works requires hardware most people don't have.

The state-of-the-art models — I3D, SlowFast, VideoMAE — all operate on raw video and require a dedicated NVIDIA GPU. That rules out every laptop, every phone, every hospital kiosk, every school computer. The models are impressive technically but completely inaccessible in practice.

Our goal was to close that gap: build something that works on a standard webcam with no special hardware, in real time, on any modern CPU.
-->

---

# Our Insight: The Pixels Don't Matter. The Hands Do.

A human signs with their hands — not with the background, lighting, or camera angle. MediaPipe extracts **21 landmarks per hand × 3 coordinates (x, y, z) = 126 numbers per frame** — a **49,371× reduction** in data volume. No raw video stored at any point.

<div class="grid grid-cols-2 gap-8 mt-6">
<div>

<div class="grid grid-cols-1 gap-4">
<div class="card">

**Raw Video**

<span class="text-red-500 font-semibold text-lg">6,220,800 pixels/frame</span>

GPU required · Privacy risk

</div>
<div class="card">

**Keypoints**

<span class="text-teal-600 font-semibold text-lg">126 floats/frame</span>

CPU-ready · Private by design

</div>
</div>

<div class="quote mt-4">
Same semantic information at a fraction of the computational cost. No background, no lighting, no camera angle — just hand geometry.
</div>

</div>
<div class="flex items-start justify-center">

<img src="/images/mediapipe.png" alt="MediaPipe hand landmarks" style="max-height: 280px; object-fit: contain; margin-top: 1.5rem;" />

</div>
</div>

<!--
The key insight is that a sign is defined entirely by hand shape and motion — not by your background, your lighting, or the camera angle. MediaPipe's hand landmark detector finds 21 points on each hand and gives us their x, y, z coordinates. That's 126 numbers per frame.

Compare that to a raw video frame at 1080p — 6.2 million pixels. We're throwing away 49,371 pixels for every number we keep. And we lose nothing semantically meaningful. The geometry of the hands is the sign. Everything else is noise.

There's also a privacy benefit that matters for deployment. We never store raw video — only keypoints. A sequence of floating point numbers is meaningless without the extraction model, so there's no identifiable video footage retained at any point.
-->

---

# Data Engineering: Building the Dataset from Scratch

No single clean dataset existed. We aggregated three sources, handled dead links, batch-processed 108k videos overnight, and kept peak disk usage under 2GB through rolling deletion of raw video after keypoint extraction.

| Dataset | Raw Videos | Extracted Clips | Classes | Status |
|---------|-----------|-----------------|---------|--------|
| WLASL | 21,083 available | 2,659 clips | 2,000 | 70% of links dead since 2020 |
| ASL Citizen | Pre-extracted | 1,542 poses | ~300 | Google/HuggingFace — clean |
| ASLense | 108,000 | 48,797 clips | 2,208 | 108k processed, raw deleted |
| **Combined** | | **52,998 train** | **1,896** | **This project** |

<div class="grid grid-cols-2 gap-4 mt-4">
<div class="callout callout-yellow">

⚠️ WLASL: 70% of hosting links were removed after 2020. Only 6,845 of 21,083 videos were recoverable.

</div>
<div class="callout callout-green">

✓ Final keypoints: 2.7GB. Stratified val split: **5,376 samples** across all 1,896 classes.

</div>
</div>

<!--
There's no single clean ASL dataset — we had to build one. We pulled from three sources with very different characteristics.

WLASL was the most well-known academic dataset but turned out to be nearly unusable. 70% of the original hosting links were dead — YouTube videos removed, Vimeo accounts deleted. We recovered 6,845 clips out of 21,083. That's a real problem for reproducibility in this space.

ASL Citizen was clean — pre-extracted poses from Google and HuggingFace, no download issues. ASLense was the largest by far: 108,000 raw videos that we processed overnight. To keep disk usage under 2GB, we deleted each raw video immediately after extracting its keypoints.

The final combined dataset is 52,998 training clips across 1,896 classes, with 5,376 held out for validation — stratified so every class has validation coverage.
-->

---

# Why 1,896 Classes? The Vocabulary Density Tradeoff

The vocabulary size directly controls the difficulty of the learning problem. Too few classes and the model is useless. Too many and there are not enough samples per class to learn anything.

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

| Vocab | Samples/Class | Learnable? | Coverage |
|-------|--------------|------------|---------|
| 300 | 9.6 | Barely | Limited |
| **1,896** | **28.0** | **✓ Yes** | **Good** |
| 2,591 | 20.7 | Marginal | Better |
| Full WLASL | 3.4 | ✗ No | — |

<div class="callout callout-yellow mt-3">
vocab=2,591 dropped accuracy ~4pts in early epochs — samples/class fell below the learning threshold.
</div>

</div>
<div>

**Sweet Spot: 1,896 Classes**

- All 1,896 classes have training data
- 96% of classes have 10+ samples
- 449 classes have 30+ samples
- 28 samples/class = minimum viable density

<div class="quote mt-3">
Class imbalance addressed by **WeightedRandomSampler** — each class seen equally per batch. Weight = 1/class_count.
</div>

</div>
</div>

<!--
Vocabulary size is a fundamental tradeoff. More classes means better coverage but fewer training examples per class, which makes the learning problem harder.

We tested several cutoffs. At 300 classes, we only have 9.6 samples per class on average — barely enough to learn from. At the full WLASL vocabulary, that drops to 3.4 — essentially impossible. At 2,591 classes, accuracy dropped by about 4 points in early training epochs because samples per class fell below the threshold where the model could generalize.

1,896 sits at the sweet spot: 28 samples per class on average, 96% of classes have at least 10 examples, and the vocabulary is large enough to cover everyday ASL with good breadth.

Even at 1,896 classes, the distribution is uneven — common signs have hundreds of clips, rare signs have a handful. We handle this with WeightedRandomSampler, which gives each class equal representation in every training batch regardless of how many clips it has.
-->
