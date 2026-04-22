# Model Architecture

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

| Step | What |
|------|------|
| Input | `(B, T, 126)` — batch × 150 frames × 126 keypoints |
| Linear projection | 126 → d_model · lifts raw coords into feature space |
| Positional encoding | Sinusoidal · transformers are order-agnostic without it |
| Transformer × 3 | 4 attention heads · FFN 2×d_model · pre-norm |
| Mean pool | Masked — averages only non-padded frames |
| Classifier | d_model → 1,896 classes |

</div>
<div>

**Two design choices worth noting**

<div class="card mb-3">

**Sinusoidal positional encoding**

A transformer sees all frames simultaneously — frame 1 and frame 60 look identical without position information. Positional encoding injects frame order so the model knows *when* each keypoint occurs in the sign.

</div>
<div class="card">

**Masked mean pool (not CLS token)**

Signs don't end at a consistent frame — padding varies per clip. Averaging over only the valid frames gives a stable whole-sign representation. A CLS token or last-frame pool would include padding noise.

</div>

</div>
</div>

<!--
The input is a sequence of keypoint frames — 126 numbers per frame, 150 frames per clip. The first thing we do is project those 126 raw coordinates into a higher-dimensional space the transformer can work in — raw x, y, z values aren't a useful feature representation on their own.

The two choices I want to call out are positional encoding and the mean pool. Transformers process all frames in parallel, which is fast, but it means the model has no inherent sense of time — without positional encoding, it literally can't tell if a finger moved up before or after it moved sideways. We add sinusoidal position embeddings to fix that.

The mean pool is the other non-obvious choice. We average over only the non-padded frames to get a single vector representing the full sign. We tried a CLS token — the BERT approach — but signs don't have a clean ending frame, so last-frame and CLS pooling both picked up padding noise. Masked mean pooling was more stable.
-->

---

# Training Setup

| Setting | Value | Why |
|---------|-------|-----|
| Optimizer | AdamW | Decoupled weight decay; better on sparse class gradients |
| Learning rate | 3e-4 | Standard transformer LR |
| Schedule | Cosine annealing | Smooth decay avoids late-stage plateaus |
| Gradient clip | 1.0 | Stabilizes attention layer training |
| Label smoothing | 0.1 | 1,896 classes — prevents overconfidence on ambiguous signs |
| Sequence length | T = 150 | Covers the longest sign; mask drops padding from pooling |
| Sampler | 1/class_count | Rare signs get equal batch exposure as common ones |

<!--
Most of these are standard choices, but a few are worth explaining in the context of this specific problem.

Label smoothing matters a lot at 1,896 classes. Without it, the model becomes very confident on the easy, common signs and essentially gives up on the ambiguous ones. Smoothing keeps it honest.

The sampler is the other key one. Our dataset is heavily imbalanced — some signs have 100+ clips, others have 3. Without WeightedRandomSampler, the model would see HELLO a hundred times for every time it sees a rare sign like ANNIVERSARY. The 1/class_count weighting ensures every class gets equal representation in each batch.

T=150 was chosen to cover the longest sign in the dataset. Shorter clips get zero-padded to 150 frames, and the padding mask in the mean pool ensures those empty frames don't influence the final representation.
-->

---

# Five Models. Five Lessons.

| Experiment | Change | Top-1 | Lesson |
|------------|--------|-------|--------|
| EXP-003 | CTC loss | **6.7%** | Wrong loss for single-label classification |
| EXP-003CE | Cross-entropy | **40.8%** | Correct loss is foundational |
| EXP-004 | d=256 + aug | **44.0%** | Scale helps marginally; aug is the bottleneck |
| EXP-004B | Distillation | **49.5%** | Teacher soft labels improve calibration |
| EXP-005 | MAE + no aug | **71.5%** | Pre-training + clean geometry: +31pt jump |
| EXP-006 | d=256, no aug | **72.8%** | Best: capacity + no geometry corruption |

<div class="callout callout-blue mt-3">
Largest single jump: removing augmentation (+27pt). Augmentation actively harmed this task — more on why next.
</div>

<!--
Each of these experiments was run to isolate one variable. Let me walk through what we learned.

EXP-003 used CTC loss — which is designed for sequence-to-sequence tasks like speech recognition where the output is a sequence of tokens. We're doing single-label classification, so CTC is completely wrong here. 6.7% is basically noise. Switching to cross-entropy immediately got us to 40.8% — nothing else changed.

From there we scaled up the model and added augmentation, expecting the usual improvements. The model got a bit better — 44% — but not by much. The distilled student pushed to 49.5%. These are reasonable improvements.

Then we removed augmentation. And the model jumped to 71.5%. That's a 27-point gain from a single change. That was the moment we realized augmentation wasn't helping — it was actively hurting. The next slide explains why.
-->

---

# Augmentation Hurts

The models trained *without* augmentation dramatically outperform those trained with it.

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

| | Aug | No Aug |
|---|---|---|
| Top-1 (d=256) | 44.0% | **72.8%** |
| Top-5 | 60.2% | **90.0%** |
| Classes at 100% | 266 | **844** |
| Classes at 0% | 465 | **125** |

</div>
<div>

**Why augmentation fails here**

Standard augmentations (random flip, jitter, scale, noise) destroy the precision that makes sign language meaningful.

- Flipping a hand makes "A" look like a mirrored non-sign
- Jittering finger positions blurs the exact geometry that separates similar signs
- ASL encodes meaning in millimeter-level finger configuration — not in style or viewpoint

The model needs to learn the *exact* geometry, not invariance to it.

</div>
</div>

<div class="quote mt-4">
This is the opposite of image classification. For skeletal keypoints, precision is the signal — corruption is noise.
</div>

<!--
This was the most surprising result in the project. In image classification, augmentation almost always helps — you flip, crop, rotate, and the model becomes more robust. We applied the same intuition here and it made things significantly worse.

The reason is fundamental to the nature of sign language. ASL encodes meaning in exact hand configuration. Flipping a hand laterally can turn a valid sign into a mirrored version that doesn't exist in the language. Adding jitter to finger positions blurs the millimeter-level geometry that distinguishes signs like MOTHER and FATHER, which use the same handshape but different locations. Noise on keypoints corrupts exactly the signal the model is trying to learn.

Look at the zero-accuracy column — with augmentation, 465 classes score 0%. Without it, that drops to 125. And classes scoring 100% jump from 266 to 844. The model isn't more robust with augmentation — it's more confused.

For raw video, augmentation teaches invariance to lighting and viewpoint, which are irrelevant to meaning. For keypoints, there's no irrelevant variation to become invariant to. Precision is the entire point.
-->

---

# MAE Pre-Training

<div class="grid grid-cols-2 gap-6 mt-4">
<div>

**Masked Autoencoder (MAE)**

Self-supervised pre-training: mask a random subset of input frames, train the encoder to reconstruct the missing keypoints. No labels required — the model learns hand geometry structure from data alone.

```
30% frames masked → reconstruct keypoints (MSE)
50 epochs · loss 0.496 → 0.329
Encoder weights → transferred to SignClassifier
```

Gives the encoder a structured prior on hand motion before fine-tuning — critical when a class has only 1–2 training examples.

</div>
<div>

**Class recovery vs. overall accuracy**

| | d=128 + MAE | d=256 no-aug |
|---|---|---|
| Zero-acc classes | **113** | 125 |
| Classes recovered | **+23** | — |
| Recovered class avg acc | ~10–15% | 0% |
| Overall Top-1 | 71.5% | **72.8%** |

<div class="quote mt-2">
Recovered classes contribute ~10–15% each — not enough to offset d=256's capacity advantage on well-represented signs.
</div>

</div>
</div>

<!--
MAE stands for Masked Autoencoder — a self-supervised technique originally from computer vision. The idea is simple: randomly mask some of the input and train the model to reconstruct what's missing. No class labels needed. The model has to learn the underlying structure of the data to reconstruct it accurately.

In our case, we mask 30% of the keypoint frames and train the encoder to predict those missing frames from the surrounding ones. After 50 epochs, we transfer those encoder weights into the classifier and fine-tune on labeled data. The pre-trained encoder already understands hand motion patterns before it ever sees a class label.

The interesting tension is in this table. MAE recovers 23 classes that the larger d=256 model never gets to — signs with only 1 or 2 training examples. But those recovered classes only reach 10–15% accuracy, because the data is so sparse. The d=256 model scores higher overall because it has 4x more parameters to fit the well-represented classes better. MAE wins on coverage, d=256 wins on aggregate accuracy.
-->

---

# Ablation: Every Model, Every Variable

<div class="mt-4">

| Model | Top-1 | Top-5 | Params | Zero-Acc |
|-------|-------|-------|--------|---------|
| BiLSTM | 31.5% | 53.1% | 1.1M | 749 |
| 1D CNN | 38.4% | 61.1% | 656K | 612 |
| d=128 + aug | 40.1% | 58.9% | 658K | ~500 |
| d=256 + aug | 44.0% | 60.2% | 2.6M | 465 |
| Distilled (d=128) | 49.5% | 71.3% | 658K | 374 |
| **d=128 + MAE, no aug** | **71.5%** | **89.5%** | **658K** | **113** |
| **d=256, no aug** | **72.8%** | **90.0%** | **2.6M** | **125** |

</div>

<div class="callout callout-blue mt-3">
Random baseline: 0.05% (1/1,896). Best result = 1,456× above random — matching SOTA GPU video models on CPU with keypoints only.
</div>

<!--
This table tells the full story of the training process in one view. A few things worth pointing out.

BiLSTM scores the worst at 31.5%, despite being specifically designed for sequential data. 749 zero-accuracy classes — nearly 40% of the vocabulary the model simply never learned. This is the hidden state bottleneck: the LSTM compresses the entire sequence into a single vector, which isn't enough for 1,896 classes.

The CNN does better than BiLSTM at 38.4%, even though it only sees 3-frame windows. Local motion patterns turn out to be more informative than a bottlenecked global state.

The interesting jump is from 44% with augmentation to 71.5% without — same architecture, same data, same everything else. That one change accounts for most of the performance.

And to put these numbers in context: state-of-the-art models like I3D and VideoMAE reach about 60–65% on this task using raw video on a GPU. Our best model reaches 72.8% on CPU with keypoints only.
-->

---

# What the Numbers Actually Look Like

Results from `transformer_d256_l3_v1896_noaug` (72.8% Top-1) — averaged across all 1,896 classes, including rare signs with only 1–2 training examples.

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

**Accuracy distribution across all 1,896 classes**

| Accuracy range | Signs |
|----------------|-------|
| 100% | **844** |
| 60–99% | ~700 |
| 20–59% | ~227 |
| 0% | **125** |

</div>
<div>

**What drives the 0% classes**

Nearly all 125 zero-accuracy classes have fewer than 3 training examples. The model simply hasn't seen enough of them to generalize.

The 84%+ of classes above 60% accuracy covers the vast majority of signs that appear with any regularity in the dataset.

</div>
</div>

<div class="callout callout-blue mt-4">
MAE pre-training recovered 23 of these zero-accuracy classes — from 136 → 113 — by giving the encoder a better prior for rare hand configurations.
</div>

<!--
72.8% is a single number that hides a very bimodal distribution. I want to show what's actually underneath it.

844 classes — nearly half the vocabulary — score 100%. The model has fully learned those signs. Another 700 are above 60%. So for roughly 80% of the vocabulary, the model is performing well.

The 125 zero-accuracy classes are almost entirely a data problem, not a model problem. We went back and looked at each one — nearly all have fewer than 3 training examples. You can't learn a sign from 2 clips. The model has nothing to generalize from.

This is an important distinction: 72.8% isn't the ceiling because the model architecture is weak. It's the ceiling because a meaningful fraction of the vocabulary is data-limited. With more training examples for those rare signs, the number would go up significantly. That's a data collection problem, not a modeling problem.
-->
