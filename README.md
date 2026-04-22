# signFormer — Real-Time ASL Recognition

> Keypoint-based temporal transformer for real-time American Sign Language word recognition on CPU.

Live demo (hand tracking): **[signformer.onrender.com](https://signformer.onrender.com)**  
Offline recognition demo: **[Watch on YouTube](https://www.youtube.com/watch?v=jfe_zdpMaG8)**

---

## Demo

The live page runs MediaPipe hand tracking in your browser — no data leaves your device. Full sign recognition (1,896 words) requires running the project locally; see [How to Run](#how-to-run) below.

The offline demo video shows the complete pipeline: webcam → hand keypoints → transformer → real-time word prediction at ~72% Top-1 accuracy across 1,896 ASL signs, running on CPU with no GPU required.

This project is a fork of [asl-realtime](https://github.com/khuranaradhika/asl-realtime).

---

## Architecture

The core insight: instead of feeding raw video frames into a heavy GPU model, we reduce each frame to a 126-dimensional keypoint vector using MediaPipe's hand landmarker, then classify the resulting sequence with a compact transformer.

```
Webcam → MediaPipe HandLandmarker → Transformer → Predicted Sign
          21 landmarks × 2 hands               ~1.5ms on CPU
          = 126 floats/frame
```

### Model

Three architectures were trained and compared on the same 52,998-sample corpus across 1,896 sign classes:

| Model | Top-1 | Top-5 | Params |
|-------|-------|-------|--------|
| BiLSTM | 31.5% | 53.1% | ~1.1M |
| 1D CNN | 38.4% | 61.1% | ~656K |
| Transformer (d=128, 3 layers) | 67.0% | 88.6% | ~658K |
| Transformer + MAE pre-training | 71.5% | 89.5% | ~658K |
| **Transformer (d=256, 4 layers)** | **72.8%** | **—** | **2.6M** |

The best model is a standard transformer encoder with sinusoidal positional encoding, pre-norm layers, and global mean pooling — no tricks, no GPU at inference time.

```
Input (B, T, 126)
  → Linear: 126 → 256
  → Sinusoidal positional encoding
  → 4 × TransformerEncoderLayer (8 heads, FFN 512, dropout 0.1)
  → Global mean pool
  → Linear: 256 → 1,896
  → Cross-entropy (label_smoothing=0.1)
```

### What went into it

- **Data**: Combined corpus from WLASL, ASL Citizen, and Aslense — 52,998 training samples, 5,567 validation samples across 1,896 classes
- **Augmentation ablation**: Augmentation (flip, temporal jitter, Gaussian noise) consistently hurt performance at 50 epochs — the model benefits from augmentation only with longer training. No-aug models converge faster and stronger.
- **MAE pre-training**: Self-supervised masked autoencoder pre-training on keypoint sequences gave +4.5 Top-1 points on the d=128 model and better coverage of rare signs
- **Knowledge distillation**: A student (d=128) trained against the d=256 teacher reached 49.5% — useful for cases where model size matters
- **ONNX export**: Inference runs via ONNX Runtime on `CPUExecutionProvider` at ~1.5ms per call (median, 200 runs, M2 MacBook)
- **72.8% Top-1 on 1,896 classes** matches or beats GPU video models (VideoMAE ~65%, I3D ~60%) while running on a laptop webcam

---

## How to Run

### Setup

```bash
git clone https://github.com/gyuszix/signFormer.git
cd signFormer
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Download pre-extracted keypoints

Training data is hosted on HuggingFace (~2.7 GB):

```bash
python3 - << 'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="khuranaradhika/asl-realtime-keypoints",
    repo_type="dataset",
    local_dir="data/processed")
EOF
```

### Run the real-time demo

```bash
./scripts/run_demo.sh
```

Prompts you to select a model and confidence threshold, exports to ONNX, and opens a webcam window with live predictions.

### Train

```bash
# Best model (d=256, no augmentation)
python3 -m src.train --vocab 1896 --epochs 150 --combined --workers 0 --no-augment --d_model 256 --n_layers 4

# Compact model (d=128)
python3 -m src.train --vocab 1896 --epochs 150 --combined --workers 0 --no-augment

# Baselines
python3 -m src.train --model lstm --vocab 1896 --epochs 150 --combined --workers 0
python3 -m src.train --model cnn  --vocab 1896 --epochs 150 --combined --workers 0
```

### Export to ONNX

```bash
python3 -m src.export \
  --checkpoint models/checkpoints/transformer_d256_l4_v1896_noaug_combined_best.pt \
  --vocab 1896
```

### Run the web server locally

```bash
uvicorn src.server:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

---

## References

- [WLASL Dataset](https://github.com/dxli94/WLASL) — Li et al., WACV 2020
- [ASL Citizen](https://huggingface.co/datasets/google/asl-citizen) — Desai et al., 2023
- [Aslense](https://huggingface.co/datasets/akasheroor/American-Sign-Language-Dataset) — Mittha, 2025
- [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) — Google, 2023
- [SPOTER](https://github.com/matyasbohacek/spoter) — Bohácek & Hrúz, WACV 2022
