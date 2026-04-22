"""
src/evaluate.py

Evaluation logic shared by train.py and standalone eval runs.
Person 3 (Jian) adds WER computation here when the CTC decoder is extended.
"""

import torch
from tqdm import tqdm

from src.decode import greedy_decode
from src.model import make_padding_mask


@torch.no_grad()
def evaluate(model, loader, device, vocab_size: int,
             per_class: bool = False, loss_type: str = "ce") -> tuple:
    """
    Run top-1 / top-5 evaluation on a dataloader.

    Args:
        per_class:  if True, also returns a dict {label_idx: (correct, total)}
        loss_type:  "ce" (SignClassifier, output (B,C)) or "ctc" (SignTransformer, output (T,B,C+1))

    Returns:
        (top1, top5)              — per_class=False
        (top1, top5, class_stats) — per_class=True
    """
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total        = 0
    class_stats  = {}

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        kpts    = batch["keypoints"].to(device)
        labels  = batch["label"].to(device).squeeze(1)
        in_lens = batch["input_length"].to(device)

        mask   = make_padding_mask(in_lens, max_len=kpts.size(1)).to(device)
        output = model(kpts, src_key_padding_mask=mask)

        B = labels.size(0)

        if loss_type == "ce":
            # output: (B, C) — direct classification
            preds_top1 = output.argmax(dim=1)                   # (B,)
            preds_top5 = output.topk(min(5, output.size(1)), dim=1).indices  # (B, 5)

            for i in range(B):
                label_idx = labels[i].item()
                if label_idx not in class_stats:
                    class_stats[label_idx] = [0, 0]
                class_stats[label_idx][1] += 1
                total += 1

                if preds_top1[i].item() == label_idx:
                    correct_top1 += 1
                    class_stats[label_idx][0] += 1
                if label_idx in preds_top5[i].tolist():
                    correct_top5 += 1

        else:
            # output: (T, B, C+1) — CTC
            preds_seq = output.argmax(dim=-1).permute(1, 0)  # (B, T)
            preds     = greedy_decode(preds_seq, blank=vocab_size)

            for i, pred_seq in enumerate(preds):
                label_idx = labels[i].item()
                if label_idx not in class_stats:
                    class_stats[label_idx] = [0, 0]
                class_stats[label_idx][1] += 1
                total += 1

                if len(pred_seq) == 0:
                    continue
                if pred_seq[0] == label_idx:
                    correct_top1 += 1
                    class_stats[label_idx][0] += 1

                T_i      = in_lens[i].item()
                avg_prob = output[:T_i, i, :].mean(dim=0)
                top5     = avg_prob.topk(5).indices.tolist()
                if label_idx in top5:
                    correct_top5 += 1

    top1 = correct_top1 / max(total, 1)
    top5 = correct_top5 / max(total, 1)

    if per_class:
        return top1, top5, class_stats
    return top1, top5
