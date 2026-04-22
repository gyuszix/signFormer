"""
article/generate_figures.py

Generates all figures for the Medium article on Real-Time ASL Recognition.
Run from the project root:
    python3 article/generate_figures.py

Outputs to article/figures/
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────
TEAL      = "#0D9488"
TEAL_LIGHT= "#CCFBF1"
SLATE_900 = "#0F172A"
SLATE_700 = "#334155"
SLATE_500 = "#64748B"
SLATE_200 = "#E2E8F0"
SLATE_100 = "#F1F5F9"
BLUE      = "#2563EB"
AMBER     = "#D97706"
RED       = "#DC2626"
WHITE     = "#FFFFFF"

OUT = Path("article/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Inter", "Helvetica Neue", "Arial"],
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.grid":        True,
    "axes.grid.axis":   "x",
    "grid.color":       SLATE_200,
    "grid.linewidth":   0.8,
    "text.color":       SLATE_700,
    "axes.labelcolor":  SLATE_700,
    "xtick.color":      SLATE_500,
    "ytick.color":      SLATE_500,
    "figure.facecolor": WHITE,
    "axes.facecolor":   WHITE,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "savefig.facecolor":WHITE,
})

# ── Fig 1: Ablation bar chart ──────────────────────────────────────────────
def fig_ablation():
    models = [
        "BiLSTM",
        "1D CNN",
        "Transformer\n(d=128, aug)",
        "Transformer\n(d=256, aug)",
        "Distilled\n(d=128)",
        "Transformer\nd=128 + MAE",
        "Transformer\nd=256, no aug",
    ]
    top1 = [31.5, 38.4, 40.1, 44.0, 49.5, 71.5, 72.8]
    colors = [SLATE_500]*5 + [TEAL, TEAL]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(models, top1, color=colors, height=0.55, zorder=3)

    for bar, val in zip(bars, top1):
        ax.text(val + 0.8, bar.get_y() + bar.get_height()/2,
                f"{val}%", va="center", fontsize=10,
                color=SLATE_700, fontweight="500")

    ax.axvline(0.05, color=SLATE_200, linewidth=1, linestyle="--", label="Random baseline (0.05%)")
    ax.set_xlim(0, 85)
    ax.set_xlabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_title("Ablation: All Models on 1,896-Class ASL", fontsize=13,
                 fontweight="600", color=SLATE_900, pad=14)

    legend_patches = [
        mpatches.Patch(color=TEAL,     label="Best models (no aug)"),
        mpatches.Patch(color=SLATE_500,label="Baseline / augmented"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", frameon=False, fontsize=9)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "01_ablation.png")
    plt.close()
    print("✓ 01_ablation.png")


# ── Fig 2: Augmentation comparison ────────────────────────────────────────
def fig_augmentation():
    metrics   = ["Top-1 Accuracy (%)", "Classes at 100%", "Classes at 0%"]
    aug_vals  = [44.0, 266,  465]
    noaug_vals= [72.8, 844,  125]
    # Normalise to same scale for grouped display — use 3 separate subplots
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))

    pairs = list(zip(metrics, aug_vals, noaug_vals))
    for ax, (label, av, nv) in zip(axes, pairs):
        bars = ax.bar(["With Aug", "No Aug"], [av, nv],
                      color=[SLATE_500, TEAL], width=0.45, zorder=3)
        for bar, val in zip(bars, [av, nv]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(av,nv)*0.02,
                    str(val), ha="center", va="bottom", fontsize=11, fontweight="600",
                    color=SLATE_700)
        ax.set_title(label, fontsize=10, fontweight="600", color=SLATE_900, pad=10)
        ax.set_ylim(0, max(av, nv) * 1.2)
        ax.tick_params(length=0)
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    fig.suptitle("Augmentation vs. No Augmentation (d=256 Transformer)",
                 fontsize=13, fontweight="600", color=SLATE_900, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "02_augmentation.png")
    plt.close()
    print("✓ 02_augmentation.png")


# ── Fig 3: Accuracy distribution ──────────────────────────────────────────
def fig_accuracy_distribution():
    # d=256 no-aug model (72.8% Top-1) — distribution from ablation analysis
    # 844 at 100%, 0 at 1-19% (precision gap), 700 at 60-99%, 227 at 20-59%, 125 at 0%
    buckets = {"0%": 125, "1–19%": 0, "20–59%": 227, "60–99%": 700, "100%": 844}

    labels = list(buckets.keys())
    counts = list(buckets.values())
    colors = ["#DC2626", "#F59E0B", "#94A3B8", "#5EEAD4", "#0D9488"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, counts, color=colors, width=0.55, zorder=3)

    for bar, val in zip(bars, counts):
        if val == 0:
            continue
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 12,
                str(val), ha="center", va="bottom", fontsize=11,
                fontweight="600", color=SLATE_700)

    ax.set_ylabel("Number of Signs", fontsize=11)
    ax.set_title("Per-Class Accuracy Distribution — 1,896 ASL Signs\n"
                 "transformer_d256, no aug · 72.8% Top-1",
                 fontsize=12, fontweight="600", color=SLATE_900, pad=12)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.tick_params(length=0)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "03_accuracy_distribution.png")
    plt.close()
    print("✓ 03_accuracy_distribution.png")


# ── Fig 4: SOTA comparison ────────────────────────────────────────────────
def fig_sota():
    models   = ["I3D", "SPOTER", "VideoMAE",
                "Ours (d=128 + MAE)", "Ours (d=256, no aug)"]
    top1     = [60,    60,       65,       71.5,              72.8]
    hardware = ["GPU", "GPU",    "GPU",    "CPU",             "CPU"]
    colors   = [SLATE_500 if h == "GPU" else TEAL for h in hardware]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(models, top1, color=colors, height=0.5, zorder=3)

    for bar, val, hw in zip(bars, top1, hardware):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val}%  [{hw}]", va="center", fontsize=10,
                color=SLATE_700, fontweight="500")

    ax.set_xlim(0, 90)
    ax.set_xlabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_title("SOTA Comparison on ASL Word Recognition",
                 fontsize=13, fontweight="600", color=SLATE_900, pad=14)

    legend_patches = [
        mpatches.Patch(color=TEAL,     label="CPU (ours)"),
        mpatches.Patch(color=SLATE_500,label="GPU required"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", frameon=False, fontsize=9)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(50, 85)
    fig.tight_layout()
    fig.savefig(OUT / "04_sota_comparison.png")
    plt.close()
    print("✓ 04_sota_comparison.png")


# ── Fig 5: Pipeline diagram ───────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")

    steps = [
        ("Webcam\n30fps", "6.2M\npixels/frame", SLATE_500,  1.0),
        ("MediaPipe\nLandmarks", "~8ms\nper frame",           TEAL,      3.2),
        ("60-Frame\nBuffer", "2-second\nwindow",              BLUE,      5.4),
        ("ONNX\nTransformer", "~0.6ms\ninference",           TEAL,      7.6),
        ("Prediction\nTop-3", "1,896\nclasses",               SLATE_700, 9.8),
    ]

    box_w, box_h = 1.7, 1.4
    for label, sub, color, x in steps:
        fancy = FancyBboxPatch((x - box_w/2, 1.3), box_w, box_h,
                                boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor="none", alpha=0.9)
        ax.add_patch(fancy)
        ax.text(x, 2.05 + 0.07, label, ha="center", va="center",
                fontsize=10, fontweight="700", color=WHITE,
                multialignment="center")
        ax.text(x, 1.52, sub, ha="center", va="center",
                fontsize=8, color=WHITE, alpha=0.85, multialignment="center")

    # Arrows between boxes
    arrow_xs = [(1.85, 2.35), (4.05, 4.55), (6.25, 6.75), (8.45, 8.95)]
    for x1, x2 in arrow_xs:
        ax.annotate("", xy=(x2, 2.0), xytext=(x1, 2.0),
                    arrowprops=dict(arrowstyle="->", color=SLATE_500,
                                   lw=1.8, mutation_scale=16))

    # End-to-end latency label
    ax.annotate("", xy=(10.65, 0.7), xytext=(0.15, 0.7),
                arrowprops=dict(arrowstyle="<->", color=SLATE_200, lw=1.5))
    ax.text(5.4, 0.42, "< 25ms end-to-end · CPU only · 3MB model · 0 network requests",
            ha="center", va="center", fontsize=9, color=SLATE_500, style="italic")

    ax.set_title("Real-Time ASL Inference Pipeline",
                 fontsize=13, fontweight="600", color=SLATE_900, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "05_pipeline.png")
    plt.close()
    print("✓ 05_pipeline.png")


# ── Fig 6: Keypoint compression ───────────────────────────────────────────
def fig_keypoint_compression():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Left: pixel grid
    n = 18
    size = 0.12
    for i in range(n):
        for j in range(n):
            c = plt.cm.Reds(0.3 + 0.5 * np.random.rand())
            rect = patches.Rectangle((0.3 + j*size, 0.8 + i*size),
                                      size*0.9, size*0.9,
                                      facecolor=c, edgecolor="none")
            ax.add_patch(rect)

    ax.text(1.35, 0.35, "6,220,800 pixels/frame", ha="center",
            fontsize=11, fontweight="700", color=RED)
    ax.text(1.35, 0.10, "GPU required · Privacy risk", ha="center",
            fontsize=9, color=SLATE_500)

    # Arrow
    ax.annotate("", xy=(6.5, 2.0), xytext=(3.4, 2.0),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=3,
                               mutation_scale=22))
    ax.text(4.95, 2.4, "49,371×\nreduction", ha="center", fontsize=11,
            fontweight="700", color=TEAL, multialignment="center")
    ax.text(4.95, 1.45, "MediaPipe\nHandLandmarker", ha="center", fontsize=8.5,
            color=SLATE_500, multialignment="center")

    # Right: keypoint dots on a hand outline
    # Simplified landmark positions (normalised)
    landmarks = [
        (0.50, 0.10),  # wrist
        (0.30, 0.35), (0.25, 0.55), (0.22, 0.70), (0.20, 0.82),  # thumb
        (0.40, 0.38), (0.38, 0.62), (0.37, 0.78), (0.36, 0.88),  # index
        (0.50, 0.36), (0.50, 0.62), (0.50, 0.79), (0.50, 0.90),  # middle
        (0.60, 0.38), (0.61, 0.62), (0.62, 0.78), (0.62, 0.88),  # ring
        (0.70, 0.42), (0.73, 0.60), (0.74, 0.72), (0.74, 0.82),  # pinky
    ]
    ox, oy, scale = 7.0, 0.7, 2.5
    for lx, ly in landmarks:
        ax.plot(ox + lx*scale, oy + ly*scale, 'o',
                color=TEAL, markersize=7, zorder=5)
    # Connect bones (simplified)
    bones = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
             (0,17),(17,18),(18,19),(19,20)]
    for a, b in bones:
        ax.plot([ox + landmarks[a][0]*scale, ox + landmarks[b][0]*scale],
                [oy + landmarks[a][1]*scale, oy + landmarks[b][1]*scale],
                '-', color=TEAL, linewidth=1.2, alpha=0.5, zorder=4)

    ax.text(8.3, 0.35, "126 floats/frame", ha="center",
            fontsize=11, fontweight="700", color=TEAL)
    ax.text(8.3, 0.10, "CPU-ready · Private by design", ha="center",
            fontsize=9, color=SLATE_500)

    ax.set_title("From Pixels to Keypoints: 49,371× Data Reduction",
                 fontsize=13, fontweight="600", color=SLATE_900, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "06_keypoint_compression.png")
    plt.close()
    print("✓ 06_keypoint_compression.png")


# ── Fig 7: MAE pre-training diagram ───────────────────────────────────────
def fig_mae():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    n_frames = 10
    frame_w, frame_h = 0.7, 1.0
    y_frames = 2.8
    masked = {2, 4, 7}  # which frames are masked

    # Draw input frames
    for i in range(n_frames):
        x = 0.4 + i * 0.85
        color = SLATE_200 if i in masked else TEAL
        alpha = 1.0
        rect = FancyBboxPatch((x, y_frames), frame_w, frame_h,
                               boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor="none", alpha=0.9)
        ax.add_patch(rect)
        if i in masked:
            ax.text(x + frame_w/2, y_frames + frame_h/2, "?",
                    ha="center", va="center", fontsize=16,
                    color=SLATE_500, fontweight="bold")
        else:
            ax.text(x + frame_w/2, y_frames + frame_h/2, "...",
                    ha="center", va="center", fontsize=9, color=WHITE)

    ax.text(4.65, y_frames + frame_h + 0.25, "Input: 30% of frames masked",
            ha="center", fontsize=10, color=SLATE_700, style="italic")

    # Encoder box
    enc_x = 3.5
    enc = FancyBboxPatch((enc_x, 1.0), 2.3, 0.9,
                          boxstyle="round,pad=0.1",
                          facecolor=SLATE_900, edgecolor="none")
    ax.add_patch(enc)
    ax.text(enc_x + 1.15, 1.45, "Transformer Encoder", ha="center",
            va="center", fontsize=10, fontweight="700", color=WHITE)

    # Arrow down to encoder
    ax.annotate("", xy=(4.65, 1.9), xytext=(4.65, 2.75),
                arrowprops=dict(arrowstyle="->", color=SLATE_500, lw=1.5))

    # Reconstruction output
    y_out = -0.05
    for i in range(n_frames):
        x = 0.4 + i * 0.85
        is_reconstructed = i in masked
        color = AMBER if is_reconstructed else SLATE_100
        border = AMBER if is_reconstructed else SLATE_200
        rect = FancyBboxPatch((x, 0.1), frame_w, 0.75,
                               boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor=border,
                               linewidth=1.5 if is_reconstructed else 0.5,
                               alpha=0.9)
        ax.add_patch(rect)
        if is_reconstructed:
            ax.text(x + frame_w/2, 0.1 + 0.375, "OK",
                    ha="center", va="center", fontsize=8,
                    color=WHITE, fontweight="bold")

    ax.annotate("", xy=(4.65, 1.0), xytext=(4.65, 0.87),
                arrowprops=dict(arrowstyle="->", color=SLATE_500, lw=1.5))
    ax.text(4.65, -0.3, "Reconstruction: predict masked keypoints (MSE loss)",
            ha="center", fontsize=10, color=SLATE_700, style="italic")

    # Transfer arrow
    ax.annotate("", xy=(9.5, 2.4), xytext=(5.9, 1.45),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.5,
                               mutation_scale=18,
                               connectionstyle="arc3,rad=-0.3"))
    ax.text(8.3, 2.1, "Transfer encoder\nweights", ha="center",
            fontsize=9.5, color=TEAL, fontweight="600", multialignment="center")

    # Downstream classifier
    clf = FancyBboxPatch((9.0, 2.5), 2.5, 0.9,
                          boxstyle="round,pad=0.1",
                          facecolor=TEAL, edgecolor="none")
    ax.add_patch(clf)
    ax.text(10.25, 2.95, "Sign Classifier\n(fine-tuned)", ha="center",
            va="center", fontsize=9.5, fontweight="700", color=WHITE,
            multialignment="center")

    ax.set_title("MAE Pre-Training: Learn Hand Geometry Before Seeing Labels",
                 fontsize=13, fontweight="600", color=SLATE_900, pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "07_mae_pretraining.png")
    plt.close()
    print("✓ 07_mae_pretraining.png")


# ── Fig 8: Training curves ────────────────────────────────────────────────
def fig_training_curves():
    files = {
        "d=128, aug (150 ep)":    ("results/metrics/transformer_d128_l3_v1896_combined_history.json",    SLATE_500),
        "CNN (50 ep)":            ("results/metrics/cnn_d128_l4_v1896_combined_history.json",            SLATE_200.replace("E2","A8")),
        "LSTM (50 ep)":           ("results/metrics/lstm_h128_l2_v1896_combined_history.json",           AMBER),
        "d=128, no aug (50 ep)":  ("results/metrics/transformer_d128_l3_v1896_noaug_combined_history.json", TEAL),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for label, (path, color) in files.items():
        with open(path) as f:
            hist = json.load(f)
        epochs = [h["epoch"] for h in hist]
        top1   = [h["top1"] * 100 for h in hist]
        loss   = [h["loss"] for h in hist]
        lw = 2.5 if "no aug" in label else 1.5
        ax1.plot(epochs, top1, color=color, linewidth=lw, label=label)
        ax2.plot(epochs, loss, color=color, linewidth=lw, label=label)

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax1.set_title("Validation Top-1 Accuracy", fontsize=12,
                  fontweight="600", color=SLATE_900)
    ax1.legend(frameon=False, fontsize=9)
    ax1.tick_params(length=0)

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Training Loss", fontsize=11)
    ax2.set_title("Training Loss", fontsize=12,
                  fontweight="600", color=SLATE_900)
    ax2.legend(frameon=False, fontsize=9)
    ax2.tick_params(length=0)

    fig.suptitle("Training Curves Across Models", fontsize=13,
                 fontweight="600", color=SLATE_900)
    fig.tight_layout()
    fig.savefig(OUT / "08_training_curves.png")
    plt.close()
    print("✓ 08_training_curves.png")


# ── Fig 9: Restricted vocabulary accuracy curve ───────────────────────────
def fig_restricted_vocab():
    # From evaluation_of_final_model_training.txt (d=256 aug model)
    top_n    = [100,  200,  300,  500,   1000,  1500,  1896]
    accuracy = [100.0, 100.0, 96.1, 84.0, 68.4,  55.7,  44.0]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(top_n, accuracy, color=TEAL, linewidth=2.5, marker="o",
            markersize=7, markerfacecolor=WHITE, markeredgewidth=2,
            markeredgecolor=TEAL, zorder=5)

    # Annotate key points
    for n, a in zip(top_n, accuracy):
        offset = 4 if n != 1000 else -8
        ax.annotate(f"{a}%", xy=(n, a), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="600", color=SLATE_700)

    # Shade the "everyday ASL" zone
    ax.axvspan(0, 500, alpha=0.07, color=TEAL, zorder=0)
    ax.text(250, 10, "Everyday ASL\n(top 500 signs)", ha="center", va="bottom",
            fontsize=8.5, color=TEAL, style="italic")

    ax.set_xlabel("Vocabulary size (Top N most common signs)", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs. Vocabulary Size\n"
                 "Restricting to the N most common signs boosts real-world usability",
                 fontsize=12, fontweight="600", color=SLATE_900, pad=12)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 115)
    ax.tick_params(length=0)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "09_restricted_vocab.png")
    plt.close()
    print("ok 09_restricted_vocab.png")


# ── Fig 10: Demo tier breakdown ───────────────────────────────────────────
def fig_demo_tiers():
    # From evaluation_of_final_model_training.txt
    # Tiers are mutually exclusive for the aug model; noaug model would be better
    tiers  = ["Tier 1\n(70%+, 3+ val)", "Tier 2\n(60%+, 2+ val)",
              "Other\n(<60%)", "Avoid\n(0% acc)"]
    counts = [245, 528, 658, 465]
    colors = [TEAL, "#5EEAD4", SLATE_500, RED]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: donut chart
    wedges, texts, autotexts = ax1.pie(
        counts, labels=None, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.55, edgecolor=WHITE, linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("600")
        at.set_color(WHITE)
    ax1.set_title("1,896 Signs by Demo Usability", fontsize=12,
                  fontweight="600", color=SLATE_900, pad=12)
    legend_patches = [mpatches.Patch(color=c, label=f"{l.replace(chr(10),' ')} — {n}")
                      for c, l, n in zip(colors, tiers, counts)]
    ax1.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.15),
               ncol=2, frameon=False, fontsize=9)

    # Right: horizontal bars showing what "safe to demo" means
    demo_labels = ["Tier 1 alone", "Tier 1 + Tier 2", "All 1,896 classes"]
    demo_acc    = [88.2,  74.8,  44.0]   # approximate weighted acc for each subset
    demo_counts = [245,   773,   1896]
    bar_colors  = [TEAL, "#5EEAD4", SLATE_500]

    bars = ax2.barh(demo_labels, demo_acc, color=bar_colors, height=0.45, zorder=3)
    for bar, val, cnt in zip(bars, demo_acc, demo_counts):
        ax2.text(val + 0.8, bar.get_y() + bar.get_height()/2,
                 f"{val}%  ({cnt} signs)", va="center", fontsize=10, color=SLATE_700)
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("Average Accuracy (%)", fontsize=11)
    ax2.set_title("Accuracy by Demo Subset", fontsize=12,
                  fontweight="600", color=SLATE_900, pad=12)
    ax2.tick_params(length=0)
    ax2.grid(axis="x")
    ax2.set_axisbelow(True)

    fig.suptitle("How to Deploy Responsibly: Tier the Vocabulary",
                 fontsize=13, fontweight="600", color=SLATE_900, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "10_demo_tiers.png")
    plt.close()
    print("ok 10_demo_tiers.png")


# ── Fig 11: Distillation training curve ──────────────────────────────────
def fig_distillation_curve():
    import re
    # Parse the last/best run (100 epochs, best Top-1=0.495)
    runs = []
    current_run = []
    with open("results/epoch_loss_and_top_k_results.txt") as f:
        for line in f:
            m = re.match(r"Epoch (\d+)/\d+ \| Loss: ([\d.]+) \| Top-1: ([\d.]+) \| Top-5: ([\d.]+)", line)
            if m:
                ep = int(m.group(1))
                if ep == 1 and current_run:
                    runs.append(current_run)
                    current_run = []
                current_run.append((ep, float(m.group(2)), float(m.group(3)), float(m.group(4))))
    if current_run:
        runs.append(current_run)
    last_run = runs[-1]  # 100-epoch run, best Top-1=0.495

    epochs = [r[0] for r in last_run]
    top1   = [r[2] * 100 for r in last_run]
    top5   = [r[3] * 100 for r in last_run]
    loss   = [r[1] for r in last_run]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # Top-1 / Top-5 accuracy
    ax1.plot(epochs, top1, color=TEAL, linewidth=2.5, label="Student Top-1 (distilled)")
    ax1.plot(epochs, top5, color=TEAL, linewidth=1.5, linestyle="--", alpha=0.6,
             label="Student Top-5")
    ax1.axhline(72.8, color=SLATE_700, linewidth=1.5, linestyle=":",
                label="Teacher Top-1 (72.8%)")
    ax1.axhline(49.5, color=RED, linewidth=1, linestyle="--", alpha=0.7,
                label="Student best (49.5%)")
    ax1.annotate("Best: 49.5%\n(ep. 81)", xy=(81, 49.5), xytext=(60, 55),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
                 fontsize=9, color=RED)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("Knowledge Distillation: Student Accuracy", fontsize=12,
                  fontweight="600", color=SLATE_900)
    ax1.legend(frameon=False, fontsize=8.5)
    ax1.tick_params(length=0)
    ax1.set_ylim(0, 85)

    # Loss
    ax2.plot(epochs, loss, color=AMBER, linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Training Loss (CE + KL)", fontsize=11)
    ax2.set_title("Distillation Loss", fontsize=12,
                  fontweight="600", color=SLATE_900)
    ax2.tick_params(length=0)

    fig.suptitle("Knowledge Distillation: alpha=0.7, tau=6 (Student d=128 from Teacher d=256)",
                 fontsize=12, fontweight="600", color=SLATE_900)
    fig.tight_layout()
    fig.savefig(OUT / "11_distillation_curve.png")
    plt.close()
    print("ok 11_distillation_curve.png")


# ── Run all ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...\n")
    fig_ablation()
    fig_augmentation()
    fig_accuracy_distribution()
    fig_sota()
    fig_pipeline()
    fig_keypoint_compression()
    fig_mae()
    fig_training_curves()
    fig_restricted_vocab()
    fig_demo_tiers()
    fig_distillation_curve()
    print(f"\nAll figures saved to {OUT}/")
