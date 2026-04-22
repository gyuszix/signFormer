"""
src/decode.py

CTC decoding utilities shared by training, evaluation, and the live demo.
Person 3 (Jian) adds beam search here.
"""

import numpy as np
import torch


def greedy_decode(preds_seq: torch.Tensor, blank: int) -> list:
    """
    CTC greedy decoding over a batch of argmax predictions.

    Args:
        preds_seq: (B, T) int tensor of argmax token indices
        blank:     blank token index

    Returns:
        list of B lists, each containing decoded token indices for one sample
    """
    results = []
    for b in range(preds_seq.size(0)):
        seq     = preds_seq[b].tolist()
        decoded = []
        prev    = None
        for token in seq:
            if token != blank and token != prev:
                decoded.append(token)
            prev = token
        results.append(decoded)
    return results


def greedy_decode_sequence(log_probs: np.ndarray, blank: int) -> list:
    """
    CTC greedy decoding for a single sequence (numpy, used by demo).

    Args:
        log_probs: (T, C) log-probability array
        blank:     blank token index

    Returns:
        list of decoded token indices
    """
    preds  = log_probs.argmax(axis=-1)  # (T,)
    result = []
    prev   = None
    for token in preds:
        if token != blank and token != prev:
            result.append(int(token))
        prev = token
    return result
