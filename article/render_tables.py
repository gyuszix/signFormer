"""
article/render_tables.py

Renders all markdown tables in the Medium article as styled PNG images.
Run from the project root: python3 article/render_tables.py
Outputs to article/figures/
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Palette (matches generate_figures.py) ─────────────────────────────────
TEAL       = "#0D9488"
TEAL_LIGHT = "#CCFBF1"
TEAL_MID   = "#99F6E4"
SLATE_900  = "#0F172A"
SLATE_700  = "#334155"
SLATE_500  = "#64748B"
SLATE_200  = "#E2E8F0"
SLATE_100  = "#F8FAFC"
WHITE      = "#FFFFFF"
AMBER      = "#D97706"
AMBER_LIGHT= "#FEF3C7"
RED        = "#DC2626"
RED_LIGHT  = "#FEE2E2"

OUT = Path("article/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial"],
    "figure.facecolor": WHITE,
})


def _clean(s):
    """Strip markdown bold/code markers."""
    return s.replace("**", "").replace("`", "").strip()


def render_table(
    headers, rows, filename,
    title=None,
    highlight_rows=None,      # list of 0-based data-row indices → teal
    warn_rows=None,           # list of 0-based data-row indices → amber
    bad_rows=None,            # list of 0-based data-row indices → red tint
    col_widths=None,          # relative proportions, e.g. [2,1,1,3]
    col_align=None,           # 'l' or 'c' per column, default all 'l'
    figsize=None,
    row_height=0.42,
    header_height=0.52,
    font_size=9.5,
    title_font_size=11.5,
):
    n_cols = len(headers)
    n_rows = len(rows)

    # ── Figure sizing ──────────────────────────────────────────────────────
    title_h  = 0.46 if title else 0.05
    pad      = 0.30
    fig_h    = title_h + header_height + n_rows * row_height + pad
    fig_w    = figsize[0] if figsize else min(14, max(8, n_cols * 2.1))
    if figsize:
        fig_w, fig_h = figsize

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=WHITE)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=WHITE)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # ── Column x-positions (with 4% left/right margin) ────────────────────
    margin_x = fig_w * 0.025
    usable_w = fig_w - 2 * margin_x
    if col_widths is None:
        col_widths = [1] * n_cols
    total = sum(col_widths)
    col_px = [margin_x + usable_w * sum(col_widths[:j]) / total
              for j in range(n_cols)]
    col_px.append(margin_x + usable_w)           # right edge of last col

    # ── Y positions (top → bottom) ─────────────────────────────────────────
    y_top   = fig_h - pad / 2
    y_title = y_top - title_h / 2
    y_hdr_t = y_top - title_h
    y_hdr_b = y_hdr_t - header_height

    # ── Title ──────────────────────────────────────────────────────────────
    if title:
        ax.text(margin_x, y_title, title,
                ha="left", va="center",
                fontsize=title_font_size, fontweight="600", color=SLATE_900)

    # ── Header row ─────────────────────────────────────────────────────────
    for j in range(n_cols):
        x0 = col_px[j]
        x1 = col_px[j + 1]
        rect = mpatches.FancyBboxPatch(
            (x0, y_hdr_b), x1 - x0, header_height,
            boxstyle="square,pad=0",
            facecolor=SLATE_900, edgecolor=WHITE, linewidth=0.8,
            zorder=2,
        )
        ax.add_patch(rect)
        align = (col_align[j] if col_align and j < len(col_align) else "l")
        tx = x0 + (x1 - x0) / 2 if align == "c" else x0 + 0.12
        ha = "center" if align == "c" else "left"
        ax.text(tx, (y_hdr_t + y_hdr_b) / 2, _clean(headers[j]),
                ha=ha, va="center",
                fontsize=font_size, fontweight="bold", color=WHITE, zorder=3)

    # ── Data rows ──────────────────────────────────────────────────────────
    for i, row in enumerate(rows):
        y_row_t = y_hdr_b - i * row_height
        y_row_b = y_row_t - row_height

        is_hl   = highlight_rows and i in highlight_rows
        is_warn = warn_rows       and i in warn_rows
        is_bad  = bad_rows        and i in bad_rows

        if is_hl:
            row_bg = TEAL_LIGHT
        elif is_warn:
            row_bg = AMBER_LIGHT
        elif is_bad:
            row_bg = RED_LIGHT
        else:
            row_bg = WHITE if i % 2 == 0 else SLATE_100

        for j, cell in enumerate(row):
            x0 = col_px[j]
            x1 = col_px[j + 1]
            rect = mpatches.FancyBboxPatch(
                (x0, y_row_b), x1 - x0, row_height,
                boxstyle="square,pad=0",
                facecolor=row_bg, edgecolor=SLATE_200, linewidth=0.4,
                zorder=2,
            )
            ax.add_patch(rect)
            align = (col_align[j] if col_align and j < len(col_align) else "l")
            tx = x0 + (x1 - x0) / 2 if align == "c" else x0 + 0.12
            ha = "center" if align == "c" else "left"
            fw = "bold" if is_hl else "normal"
            ax.text(tx, (y_row_t + y_row_b) / 2, _clean(cell),
                    ha=ha, va="center",
                    fontsize=font_size, fontweight=fw,
                    color=SLATE_900 if is_hl else SLATE_700,
                    zorder=3)

    fig.savefig(OUT / filename, dpi=200, bbox_inches="tight", facecolor=WHITE)
    plt.close()
    print(f"ok  {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# Table definitions
# ══════════════════════════════════════════════════════════════════════════════

def t01_dataset():
    render_table(
        headers=["Dataset", "Source", "Videos / Clips", "Classes", "Challenge"],
        rows=[
            ["WLASL",        "Academic (2020)",        "6,845 recovered of 21,083", "2,000", "70% of hosting links dead"],
            ["ASL Citizen",  "Google / HuggingFace",   "1,542 pre-extracted",       "~300",  "None — clean"],
            ["ASLense",      "Custom collection",       "48,797 clips",              "2,208", "108k videos, 2 GB disk budget"],
            ["Combined",     "",                        "52,998 train / 5,376 val",  "1,896", "Stratified split"],
        ],
        filename="t01_dataset.png",
        title="Dataset Aggregation",
        highlight_rows=[3],
        col_widths=[1.4, 1.8, 2.0, 0.9, 2.4],
        figsize=(13, 3.5),
    )

def t02_vocab_tradeoff():
    render_table(
        headers=["Vocab", "Avg Samples / Class", "Learnable?", "Coverage"],
        rows=[
            ["300",         "9.6",   "Barely",  "Limited"],
            ["1,896",       "28.0",  "Yes",     "Good"],
            ["2,591",       "20.7",  "Marginal","Better"],
            ["Full WLASL",  "3.4",   "No",      "—"],
        ],
        filename="t02_vocab_tradeoff.png",
        title="Vocabulary Size vs. Learnability",
        highlight_rows=[1],
        warn_rows=[2],
        bad_rows=[3],
        col_widths=[1.2, 1.6, 1.0, 1.0],
        col_align=["l", "c", "c", "c"],
        figsize=(8, 3.2),
    )

def t03_attention_friend():
    render_table(
        headers=["Frame Range", "What's Happening", "Attention Level"],
        rows=[
            ["1–7",   "Setup — hands approach",       "Low"],
            ["8–12",  "Hand approach",                 "High"],
            ["13–21", "Hands traveling toward each other", "Low"],
            ["22–26", "Fingers interlock",             "Highest"],
            ["27–30", "Release motion",                "Medium"],
        ],
        filename="t03_attention_friend.png",
        title='Attention Weights: "FRIEND"',
        highlight_rows=[3],
        col_widths=[1.0, 2.5, 1.2],
        figsize=(8, 3.4),
    )

def t04_attention_airplane():
    render_table(
        headers=["Frame Range", "What's Happening", "Attention Level"],
        rows=[
            ["1–2",   "Start position",                        "Low"],
            ["3–6",   "Arm begins extending",                  "Medium"],
            ["7–9",   "Arm moving outward",                    "Low"],
            ["10–18", "Full arm extension with spread hand",   "Highest"],
            ["19–23", "Hold at extension",                     "Low"],
            ["24–28", "Hand shape refinement",                 "High"],
        ],
        filename="t04_attention_airplane.png",
        title='Attention Weights: "AIRPLANE"',
        highlight_rows=[3],
        col_widths=[1.0, 2.8, 1.2],
        figsize=(8, 3.8),
    )

def t05_architecture():
    render_table(
        headers=["Step", "What it does"],
        rows=[
            ["Input",               "(B, T, 126) — batch x 150 frames x 126 keypoints per frame"],
            ["Linear projection",   "126 -> d_model  |  lifts raw coords into feature space"],
            ["Positional encoding", "Sinusoidal  |  injects frame order so model knows when each keypoint occurs"],
            ["Transformer x 3",     "4 attention heads · FFN 2xd_model · pre-norm layer"],
            ["Masked mean pool",    "Averages only non-padded frames — excludes padding noise"],
            ["Classifier",          "d_model -> 1,896 class logits"],
        ],
        filename="t05_architecture.png",
        title="Model Architecture Steps",
        col_widths=[1.5, 4.5],
        figsize=(12, 3.8),
    )

def t06_experiments():
    render_table(
        headers=["Experiment", "What Changed", "Top-1", "Lesson"],
        rows=[
            ["EXP-003",   "CTC loss",                   "6.7%",  "Wrong loss — CTC is for sequence output, not single labels"],
            ["EXP-003CE", "Cross-entropy loss",          "40.8%", "Correct loss is foundational — nothing else works without it"],
            ["EXP-004",   "d=256 + augmentation",        "44.0%", "Scale helps marginally; augmentation is the bottleneck"],
            ["EXP-004B",  "Knowledge distillation",      "49.5%", "Teacher soft labels improve calibration but don't close the gap"],
            ["EXP-005",   "MAE pre-training + no aug",   "71.5%", "Self-supervised pre-training + clean geometry: +27pt jump"],
            ["EXP-006",   "d=256, no augmentation",      "72.8%", "Best: increased capacity + no geometry corruption"],
        ],
        filename="t06_experiments.png",
        title="Five Models. Five Lessons.",
        highlight_rows=[4, 5],
        bad_rows=[0],
        col_widths=[1.2, 2.0, 0.9, 4.0],
        figsize=(13, 4.0),
    )

def t07_training_setup():
    render_table(
        headers=["Setting", "Value", "Why"],
        rows=[
            ["Optimizer",       "AdamW",         "Decoupled weight decay; better on sparse class gradients than Adam"],
            ["Learning rate",   "3e-4",          "Standard transformer LR; cosine schedule decays it smoothly"],
            ["Schedule",        "Cosine annealing", "Avoids late-stage loss plateaus; smooth LR decay to zero"],
            ["Gradient clip",   "1.0",           "Prevents attention layer gradient explosions during early training"],
            ["Label smoothing", "0.1",           "Prevents overconfidence on ambiguous signs at 1,896 classes"],
            ["Sequence length", "T = 150",       "Covers the longest sign; padding mask excludes empty frames"],
            ["Sampler",         "1/class_count", "Every sign gets equal batch exposure regardless of clip count"],
        ],
        filename="t07_training_setup.png",
        title="Training Hyperparameters",
        col_widths=[1.4, 1.6, 4.5],
        figsize=(13, 4.2),
    )

def t08_augmentation():
    render_table(
        headers=["Metric", "With Augmentation", "No Augmentation"],
        rows=[
            ["Top-1 Accuracy (d=256)", "44.0%", "72.8%"],
            ["Top-5 Accuracy",         "60.2%", "90.0%"],
            ["Classes at 100%",        "266",   "844"],
            ["Classes at 0%",          "465",   "125"],
        ],
        filename="t08_augmentation.png",
        title="Augmentation vs. No Augmentation (d=256 Transformer)",
        highlight_rows=[0],
        col_widths=[2.0, 1.6, 1.6],
        col_align=["l", "c", "c"],
        figsize=(8, 3.2),
    )

def t09_mae():
    render_table(
        headers=["Metric", "d=128 + MAE, no aug", "d=256, no aug"],
        rows=[
            ["Top-1 Accuracy",              "71.5%",   "72.8%"],
            ["Top-5 Accuracy",              "89.5%",   "90.0%"],
            ["Zero-accuracy classes",       "113",     "125"],
            ["Classes recovered vs. baseline", "+23",  "—"],
            ["Recovered class avg accuracy","~10-15%", "0%"],
        ],
        filename="t09_mae.png",
        title="MAE Pre-Training vs. Best Model",
        highlight_rows=[2, 3],
        col_widths=[2.2, 1.8, 1.6],
        col_align=["l", "c", "c"],
        figsize=(8, 3.5),
    )

def t10_ablation():
    render_table(
        headers=["Model", "Top-1", "Top-5", "Params", "Zero-Acc Classes"],
        rows=[
            ["BiLSTM",                       "31.5%", "53.1%", "1.1M",  "749"],
            ["1D CNN",                       "38.4%", "61.1%", "656K",  "612"],
            ["Transformer d=128 + aug",      "40.1%", "58.9%", "658K",  "~500"],
            ["Transformer d=256 + aug",      "44.0%", "60.2%", "2.6M",  "465"],
            ["Distilled (d=128)",            "49.5%", "71.3%", "658K",  "374"],
            ["Transformer d=128 + MAE, no aug", "71.5%", "89.5%", "658K", "113"],
            ["Transformer d=256, no aug",    "72.8%", "90.0%", "2.6M",  "125"],
        ],
        filename="t10_ablation.png",
        title="Full Ablation: Every Model, Every Variable",
        highlight_rows=[5, 6],
        col_widths=[2.8, 0.9, 0.9, 0.9, 1.5],
        col_align=["l", "c", "c", "c", "c"],
        figsize=(11, 4.4),
    )

def t11_accuracy_dist():
    render_table(
        headers=["Accuracy Range", "Number of Signs"],
        rows=[
            ["100%",    "844"],
            ["60–99%",  "~700"],
            ["20–59%",  "~227"],
            ["0%",      "125"],
        ],
        filename="t11_accuracy_dist.png",
        title="Per-Class Accuracy Distribution — d=256, no aug (72.8% Top-1)",
        highlight_rows=[0],
        bad_rows=[3],
        col_widths=[1.5, 1.2],
        col_align=["l", "c"],
        figsize=(7, 3.0),
    )

def t12_sota():
    render_table(
        headers=["Model", "Top-1", "Hardware", "Input"],
        rows=[
            ["I3D",                        "~60%",  "GPU", "Raw video"],
            ["VideoMAE",                   "~65%",  "GPU", "Raw video"],
            ["SPOTER",                     "~60%",  "GPU", "Keypoints"],
            ["Ours — d=128 + MAE, no aug", "71.5%", "CPU", "Keypoints"],
            ["Ours — d=256, no aug",       "72.8%", "CPU", "Keypoints"],
        ],
        filename="t12_sota.png",
        title="State-of-the-Art Comparison on ASL Word Recognition",
        highlight_rows=[3, 4],
        col_widths=[2.4, 0.9, 0.9, 1.2],
        col_align=["l", "c", "c", "c"],
        figsize=(9, 3.5),
    )

def t13_failures():
    render_table(
        headers=["Confused Pair", "Why"],
        rows=[
            ["MOTHER / FATHER",   "Same handshape, different face position (chin vs. forehead)"],
            ["WEEK / NEXT-WEEK",  "Same motion pattern, one temporally shifted"],
            ["HELP / ASSIST",     "Near-identical hand configuration"],
            ["APPLE / ONION",     "Both involve a twist at the cheek"],
        ],
        filename="t13_failures.png",
        title="Systematic Sign Confusions",
        col_widths=[1.8, 4.5],
        figsize=(10, 3.0),
    )


# ── Run all ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Rendering tables...\n")
    t01_dataset()
    t02_vocab_tradeoff()
    t03_attention_friend()
    t04_attention_airplane()
    t05_architecture()
    t06_experiments()
    t07_training_setup()
    t08_augmentation()
    t09_mae()
    t10_ablation()
    t11_accuracy_dist()
    t12_sota()
    t13_failures()
    print(f"\nAll tables saved to {OUT}/")
