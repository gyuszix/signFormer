"""
src/train.py

Training loop for SignTransformer with CTC loss.
Includes checkpointing, learning rate scheduling, and logging.

Usage:
    # BiLSTM baseline
    python src/train.py --model lstm --vocab 300 --epochs 50 --combined

    # Transformer student (~672K params)
    python src/train.py --model transformer --vocab 300 --epochs 50 --combined

    # Transformer teacher (GPU recommended)
    python src/train.py --model transformer --vocab 300 --epochs 100 --teacher --combined
"""

import json
import argparse
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import CHECKPOINT_DIR
from src.dataloader import get_dataloader
from src.model import (build_student_model, build_student_classifier,
                       build_teacher_model, build_lstm_baseline, build_cnn_baseline,
                       make_padding_mask)


def train_one_epoch(model, loader, optimizer, criterion, device, vocab_size, loss_type="ce"):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        kpts    = batch["keypoints"].to(device)
        labels  = batch["label"].to(device).squeeze(1)
        in_lens = batch["input_length"].to(device)

        mask   = make_padding_mask(in_lens, max_len=kpts.size(1)).to(device)
        output = model(kpts, src_key_padding_mask=mask)

        if loss_type == "ce":
            # output: (B, C) — direct cross-entropy
            loss = criterion(output, labels)
        else:
            # output: (T, B, C+1) — CTC
            lb_lens = batch["label_length"].to(device)
            loss    = criterion(output, labels, in_lens, lb_lens)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


from src.evaluate import evaluate


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = get_dataloader("train", vocab_size=args.vocab,
                                  batch_size=args.batch_size, num_workers=args.workers,
                                  augment=not args.no_augment, combined=args.combined)
    val_loader   = get_dataloader("val",   vocab_size=args.vocab,
                                  batch_size=args.batch_size, num_workers=args.workers,
                                  combined=False)  # val always uses WLASL-only manifest

    aug_tag      = "_noaug" if args.no_augment else ""
    combined_tag = "_combined" if args.combined else ""

    loss_tag = "_ctc" if args.loss == "ctc" else ""

    if args.model == "cnn":
        model    = build_cnn_baseline(n_classes=args.vocab, loss=args.loss)
        run_name = f"cnn_d128_l4_v{args.vocab}{aug_tag}{loss_tag}{combined_tag}"
    elif args.model == "lstm":
        model = build_lstm_baseline(n_classes=args.vocab, loss=args.loss)
        run_name = f"lstm_h128_l2_v{args.vocab}{aug_tag}{loss_tag}{combined_tag}"
    elif args.teacher:
        model    = build_teacher_model(n_classes=args.vocab)
        run_name = f"transformer_teacher_v{args.vocab}{aug_tag}{loss_tag}{combined_tag}"
    elif args.loss == "ce":
        model    = build_student_classifier(n_classes=args.vocab, input_dim=126,
                                            d_model=args.d_model, n_layers=args.n_layers,
                                            dropout=args.dropout)
        run_name = f"transformer_d{args.d_model}_l{args.n_layers}_v{args.vocab}{aug_tag}{combined_tag}"
    else:
        model    = build_student_model(n_classes=args.vocab)
        run_name = f"transformer_d{args.d_model}_l{args.n_layers}_v{args.vocab}{aug_tag}_ctc{combined_tag}"

    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CTCLoss(blank=args.vocab, reduction="mean", zero_infinity=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_top1 = 0.0
    history   = []

    # Protect against overwriting a better existing checkpoint.
    # If a checkpoint already exists for this run_name, initialize best_top1
    # from it so a fresh run only saves if it genuinely beats the prior best.
    ckpt_path = CHECKPOINT_DIR / f"{run_name}_best.pt"
    if ckpt_path.exists():
        try:
            existing = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            best_top1 = existing.get("top1", 0.0)
            print(f"Existing checkpoint found: Top-1 = {best_top1:.3f} "
                  f"(epoch {existing.get('epoch', '?')}) — will only overwrite if beaten.")
        except Exception as e:
            print(f"Could not read existing checkpoint: {e}")

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device, args.vocab, loss_type=args.loss)
        top1, top5 = evaluate(model, val_loader, device, args.vocab, loss_type=args.loss)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Top-1: {top1:.3f} | Top-5: {top5:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed:.1f}s")

        history.append({"epoch": epoch, "loss": train_loss, "top1": top1, "top5": top5})

        if top1 > best_top1:
            best_top1 = top1
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "top1":        top1,
                "top5":        top5,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  ✓ New best: {top1:.3f} → saved to {ckpt_path}")

    results_dir = Path("results/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best Top-1: {best_top1:.3f}")

    # Per-class accuracy breakdown (final epoch)
    print("\n=== Per-class accuracy breakdown ===")
    _, _, class_stats = evaluate(model, val_loader, device, args.vocab,
                                  per_class=True, loss_type=args.loss)

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

    # Print top-10 and bottom-10
    sorted_by_acc = sorted(per_class_results, key=lambda x: -x["accuracy"])
    print("Top-10 classes:")
    for r in sorted_by_acc[:10]:
        print(f"  {r['word']:20s} {r['correct']}/{r['total']}  ({r['accuracy']*100:.0f}%)")
    print("Bottom-10 classes:")
    for r in sorted_by_acc[-10:]:
        print(f"  {r['word']:20s} {r['correct']}/{r['total']}  ({r['accuracy']*100:.0f}%)")

    zero_acc = sum(1 for r in per_class_results if r["accuracy"] == 0)
    print(f"\nClasses with 0% accuracy: {zero_acc}/{len(per_class_results)}")

    with open(results_dir / f"{run_name}_per_class.json", "w") as f:
        json.dump(per_class_results, f, indent=2)
    print(f"Per-class results saved → results/metrics/{run_name}_per_class.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SignTransformer on WLASL")
    default_workers = 0 if platform.system() == "Darwin" else 4
    parser.add_argument("--vocab",      type=int,   default=300)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--d_model",    type=int,   default=128)
    parser.add_argument("--n_layers",   type=int,   default=3)
    parser.add_argument("--workers",    type=int,   default=default_workers)
    parser.add_argument("--model",      type=str,   default="transformer",
                        choices=["transformer", "lstm", "cnn"],
                        help="Model architecture: transformer (default), lstm, or cnn (baselines)")
    parser.add_argument("--teacher",    action="store_true",
                        help="Use larger teacher transformer (ignored when --model lstm)")
    parser.add_argument("--loss",       type=str, default="ce",
                        choices=["ce", "ctc"],
                        help="Loss function: ce (cross-entropy, default) or ctc (for future continuous signing)")
    parser.add_argument("--dropout",    type=float, default=0.1,
                        help="Dropout rate (default 0.1; try 0.3 for stronger regularization)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable training augmentations (use for EXP-001 baseline)")
    parser.add_argument("--combined",   action="store_true",
                        help="Use combined WLASL+ASL Citizen manifest for training")
    args = parser.parse_args()
    main(args)
