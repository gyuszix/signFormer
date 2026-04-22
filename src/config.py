"""
src/config.py

Shared constants and paths used across the project.
All other modules import from here — no more scattered magic numbers.
"""

from pathlib import Path

# ── Keypoint dimensions ────────────────────────────────────────────────────────
KEYPOINT_DIM  = 126     # 21 left-hand landmarks × 3 + 21 right-hand landmarks × 3
HOLISTIC_DIM  = 225     # hands (126) + full 33-point pose (99) — for next model
MAX_SEQ_LEN   = 150     # frames — clips longer than this are truncated at training time

# Upper-body pose landmark indices used in holistic mode (from MediaPipe 33-pt model)
# Pairs that must be swapped on horizontal flip: (L_idx, R_idx)
POSE_FLIP_PAIRS = [(11,12),(13,14),(15,16),(17,18),(19,20),(21,22),(23,24),(25,26),(27,28),(29,30),(31,32)]

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_RAW_DIR   = Path("data/raw/wlasl")
DATA_PROC_DIR  = Path("data/processed")
CHECKPOINT_DIR = Path("models/checkpoints")

# ── Demo ───────────────────────────────────────────────────────────────────────
WINDOW_FRAMES = 60      # frames to buffer before running inference
SMOOTH_WINDOW = 5       # smooth predictions over this many inference calls
