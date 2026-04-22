"""
src/distill.py

Knowledge distillation — train student to mimic teacher soft labels.

Usage:
    # Using large model (d=256, l=4) as teacher
    python3 -m src.distill \
        --teacher models/checkpoints/transformer_d256_l4_v1896_combined_best.pt \
        --vocab 1896 --epochs 100 --alpha 0.5 --tau 4.0

    # Stronger distillation (higher tau = softer teacher)
    python3 -m src.distill \
        --teacher models/checkpoints/transformer_d256_l4_v1896_combined_best.pt \
        --vocab 1896 --epochs 100 --alpha 0.7 --tau 6.0
"""

import json
import argparse
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import CHECKPOINT_DIR
from src.dataloader import get_dataloader
from src.model import (build_student_classifier, build_teacher_model,
                       make_padding_mask)
from src.evaluate import evaluate


def distillation_loss(student_logits, teacher_logits, labels,
                      alpha=0.5, tau=4.0):
    """
    Combined distillation + CE loss.

    Args:
        student_logits: (B, C) raw student logits
        teacher_logits: (B, C) raw teacher logits
        labels:         (B,) ground truth indices
        alpha:          weight on distillation loss (1-alpha on CE)
        tau:            temperature — higher = softer teacher distribution

    Returns:
        total loss scalar
    """
    # Soft distillation loss (KL divergence)
    soft_targets = F.softmax(teacher_logits / tau, dim=-1)
    soft_student  = F.log_softmax(student_logits / tau, dim=-1)
    kl_loss = F.kl_div(soft_student, soft_targets,
                        reduction="batchmean") * (tau ** 2)

    # Hard CE loss with label smoothing
    ce_loss = F.cross_entropy(student_logits, labels, label_smoothing=0.1)

    return alpha * kl_loss + (1 - alpha) * ce_loss


def main(args):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load teacher ──────────────────────────────────────────────────
    print(f"\nLoading teacher from {args.teacher}")
    ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    # Auto-detect if teacher is large or small model
    if saved_args.get("teacher", False):
        teacher = build_teacher_model(n_classes=args.vocab)
        print("Teacher type: large teacher model")
    else:
        t_d     = saved_args.get("d_model", 128)
        t_l     = saved_args.get("n_layers", 3)
        teacher = build_student_classifier(
            n_classes=args.vocab, input_dim=126,
            d_model=t_d, n_layers=t_l)
        print(f"Teacher type: student classifier (d={t_d}, l={t_l})")

    teacher.load_state_dict(ckpt["model_state"])
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False  # freeze teacher

    print(f"Teacher Top-1: {ckpt.get('top1', 0.0):.3f}")
    print(f"Teacher params: {teacher.count_parameters():,}")

    # ── Build student ─────────────────────────────────────────────────
    student = build_student_classifier(
        n_classes=args.vocab, input_dim=126,
        d_model=args.d_model, n_layers=args.n_layers,
        dropout=args.dropout)
    student = student.to(device)
    print(f"\nStudent params: {student.count_parameters():,}")
    print(f"Distillation: alpha={args.alpha}, tau={args.tau}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader = get_dataloader(
        "train", vocab_size=args.vocab,
        batch_size=args.batch_size,
        num_workers=args.workers,
        combined=True)
    val_loader = get_dataloader(
        "val", vocab_size=args.vocab,
        batch_size=args.batch_size,
        num_workers=args.workers,
        combined=False)

    # ── Training setup ────────────────────────────────────────────────
    optimizer = optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    run_name  = (f"distill_s{args.d_model}l{args.n_layers}"
                 f"_a{int(args.alpha*10)}_t{int(args.tau)}"
                 f"_v{args.vocab}_combined")
    ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    best_top1 = 0.0
    history   = []

    print(f"\nSaving to: {ckpt_path}")
    print(f"Starting distillation for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        student.train()
        total_loss = 0.0
        n_batches  = 0
        t0 = time.time()

        for batch in tqdm(train_loader, desc="Distilling", leave=False):
            kpts    = batch["keypoints"].to(device)
            labels  = batch["label"].to(device).squeeze(1)
            in_lens = batch["input_length"].to(device)
            mask    = make_padding_mask(
                in_lens, max_len=kpts.size(1)).to(device)

            # Student forward
            s_logits = student(kpts, src_key_padding_mask=mask)

            # Teacher forward — no gradients
            with torch.no_grad():
                t_logits = teacher(kpts, src_key_padding_mask=mask)

            loss = distillation_loss(
                s_logits, t_logits, labels,
                alpha=args.alpha, tau=args.tau)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        top1, top5 = evaluate(
            student, val_loader, device, args.vocab, loss_type="ce")
        elapsed = time.time() - t0

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Top-1: {top1:.3f} | Top-5: {top5:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed:.1f}s")

        history.append({
            "epoch": epoch, "loss": train_loss,
            "top1": top1, "top5": top5})

        if top1 > best_top1:
            best_top1 = top1
            torch.save({
                "epoch":       epoch,
                "model_state": student.state_dict(),
                "top1":        top1,
                "top5":        top5,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  ✓ New best: {top1:.3f} → {ckpt_path}")

    # Save history
    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDistillation complete.")
    print(f"Best Top-1: {best_top1:.3f}")
    print(f"Checkpoint: {ckpt_path}")

    # Per-class breakdown
    print("\n=== Per-class accuracy breakdown ===")
    _, _, class_stats = evaluate(
        student, val_loader, device, args.vocab,
        per_class=True, loss_type="ce")

    vocab_path = Path("data/processed/vocab.json")
    idx2word   = {}
    if vocab_path.exists():
        with open(vocab_path) as f:
            idx2word = {v: k for k, v in json.load(f).items()}

    per_class_results = []
    for label_idx, (correct, total_c) in sorted(class_stats.items()):
        acc  = correct / max(total_c, 1)
        word = idx2word.get(label_idx, str(label_idx))
        per_class_results.append({
            "label_idx": label_idx,
            "word":      word,
            "correct":   correct,
            "total":     total_c,
            "accuracy":  round(acc, 4),
        })

    sorted_by_acc = sorted(per_class_results, key=lambda x: -x["accuracy"])
    print("Top-10 classes:")
    for r in sorted_by_acc[:10]:
        print(f"  {r['word']:25s} {r['correct']}/{r['total']}  "
              f"({r['accuracy']*100:.0f}%)")
    print("Bottom-10 classes:")
    for r in sorted_by_acc[-10:]:
        print(f"  {r['word']:25s} {r['correct']}/{r['total']}  "
              f"({r['accuracy']*100:.0f}%)")

    zero_acc = sum(1 for r in per_class_results if r["accuracy"] == 0)
    print(f"\nClasses with 0% accuracy: {zero_acc}/{len(per_class_results)}")

    with open(results_dir / f"{run_name}_per_class.json", "w") as f:
        json.dump(per_class_results, f, indent=2)
    print(f"Per-class saved → results/metrics/{run_name}_per_class.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knowledge distillation for ASL sign classifier")
    parser.add_argument("--teacher",    type=str, required=True,
                        help="Path to teacher checkpoint .pt file")
    parser.add_argument("--vocab",      type=int, default=1896)
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--alpha",      type=float, default=0.5,
                        help="Weight on distillation loss (0=pure CE, 1=pure KD)")
    parser.add_argument("--tau",        type=float, default=4.0,
                        help="Temperature — higher = softer teacher distribution")
    parser.add_argument("--d_model",    type=int, default=128,
                        help="Student model dimension (default 128)")
    parser.add_argument("--n_layers",   type=int, default=3,
                        help="Student number of layers (default 3)")
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers",    type=int,
                        default=0 if platform.system() == "Darwin" else 4)
    args = parser.parse_args()
    main(args)