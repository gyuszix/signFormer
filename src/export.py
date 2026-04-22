"""
src/export.py

Export a trained checkpoint to ONNX for CPU/web deployment.
Auto-detects CE (SignClassifier) vs CTC (SignTransformer) from the checkpoint.

Usage:
    python3 -m src.export --checkpoint models/checkpoints/transformer_d128_l3_v300_combined_best.pt
"""

import argparse
import time
import numpy as np
import torch
from pathlib import Path

from src.model import (build_student_model, build_student_classifier,
                       build_cnn_baseline, build_lstm_baseline, make_padding_mask)


def export_to_onnx(checkpoint_path: str, output_path: str = "models/sign_model.onnx",
                   vocab_size: int = 300):
    device = torch.device("cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Detect model type from saved args
    saved_args  = ckpt.get("args", {})
    loss_type   = saved_args.get("loss", "ctc")   # old checkpoints default to ctc
    model_type  = saved_args.get("model", "transformer")

    d_model  = saved_args.get("d_model",  128)
    n_layers = saved_args.get("n_layers", 3)

    if model_type == "cnn":
        model = build_cnn_baseline(n_classes=vocab_size)
    elif model_type == "lstm":
        model = build_lstm_baseline(n_classes=vocab_size)
    elif loss_type == "ce":
        model = build_student_classifier(n_classes=vocab_size,
                                         d_model=d_model, n_layers=n_layers)
    else:
        model = build_student_model(n_classes=vocab_size)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model type: {model.__class__.__name__} (loss={loss_type}, d_model={d_model}, n_layers={n_layers})")
    print(f"Loaded checkpoint: Top-1 = {ckpt.get('top1', 'N/A'):.3f}")

    dummy_kpts = torch.randn(1, 80, 126)
    dummy_mask = torch.zeros(1, 80, dtype=torch.bool)

    # CE outputs (B, C) — CTC outputs (T, B, C+1)
    if loss_type == "ce" or model_type in ("cnn_ce", "lstm_ce"):
        output_names  = ["logits"]
        dynamic_axes  = {
            "keypoints":    {0: "batch", 1: "seq_len"},
            "padding_mask": {0: "batch", 1: "seq_len"},
            "logits":       {0: "batch"},
        }
    else:
        output_names  = ["log_probs"]
        dynamic_axes  = {
            "keypoints":    {0: "batch", 1: "seq_len"},
            "padding_mask": {0: "batch", 1: "seq_len"},
            "log_probs":    {0: "seq_len", 1: "batch"},
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_kpts, dummy_mask),
        output_path,
        input_names  = ["keypoints", "padding_mask"],
        output_names = output_names,
        dynamic_axes = dynamic_axes,
        opset_version       = 17,
        do_constant_folding = True,
    )

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"Exported to {output_path} ({size_mb:.1f} MB)")

    benchmark_latency(output_path, output_name=output_names[0])


def benchmark_latency(onnx_path: str, n_runs: int = 200,
                      seq_len: int = 80, output_name: str = "logits"):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    kpts = np.random.randn(1, seq_len, 126).astype(np.float32)
    mask = np.zeros((1, seq_len), dtype=bool)

    for _ in range(10):
        sess.run([output_name], {"keypoints": kpts, "padding_mask": mask})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run([output_name], {"keypoints": kpts, "padding_mask": mask})
        times.append((time.perf_counter() - t0) * 1000)

    times = sorted(times)
    print(f"\nLatency over {n_runs} runs (seq_len={seq_len}):")
    print(f"  Mean:   {np.mean(times):.1f} ms")
    print(f"  Median: {np.median(times):.1f} ms")
    print(f"  p95:    {np.percentile(times, 95):.1f} ms")
    print(f"  p99:    {np.percentile(times, 99):.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output",     type=str, default="models/sign_model.onnx")
    parser.add_argument("--vocab",      type=int, default=300)
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output, args.vocab)
