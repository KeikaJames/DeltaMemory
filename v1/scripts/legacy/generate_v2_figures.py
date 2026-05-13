"""Generate Stage 13 figures with a DeepSeek/Apple-paper-style preset.

Outputs to ``docs/figures/v2/``.

  * ``stage13_architecture.svg`` — v1 vs v2 conceptual diagram.
  * ``stage13_recall_lift.svg``  — rank/logit lift on the unit gate.
  * ``stage13_alpha_phase.svg``  — α phase diagram (working ≤ collapse).

Run:
    python scripts/generate_v2_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures" / "v2"
OUT.mkdir(parents=True, exist_ok=True)


# ---------- Style preset (Apple / DeepSeek / Stripe-paper hybrid) ----------

PALETTE = {
    "ink":   "#1F2933",   # primary text
    "soft":  "#52606D",   # secondary text
    "rule":  "#D8DEE9",   # axis & grid
    "bg":    "#FFFFFF",
    "blue":  "#2563EB",
    "indigo":"#4F46E5",
    "amber": "#D97706",
    "red":   "#DC2626",
    "green": "#059669",
    "violet":"#7C3AED",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "600",
    "axes.titlepad": 14,
    "axes.labelsize": 11,
    "axes.labelcolor": PALETTE["ink"],
    "axes.edgecolor": PALETTE["rule"],
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": PALETTE["rule"],
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "xtick.color": PALETTE["soft"],
    "ytick.color": PALETTE["soft"],
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "savefig.facecolor": PALETTE["bg"],
    "savefig.bbox": "tight",
})


# ----------------------- Figure 1: architecture diagram -----------------------

def fig_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    titles = [
        ("v1 — Stitched encoder + KeyProjector + broadcast",  axes[0]),
        ("v2 — AttentionNative Mneme (zero parameter)", axes[1]),
    ]

    def box(ax, x, y, w, h, txt, color, ec=None):
        ec = ec or color
        patch = FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=color, ec=ec, lw=1.0, alpha=0.18)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=10, color=PALETTE["ink"], weight="500")

    def arrow(ax, x1, y1, x2, y2, color=PALETTE["soft"]):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=color, lw=1.0))

    # v1 panel
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    box(ax, 0.3, 4.6, 2.1, 0.9, "Encoder\n(MultiLayer)", PALETTE["amber"])
    box(ax, 0.3, 3.0, 2.1, 0.9, "KeyProjector",          PALETTE["amber"])
    box(ax, 0.3, 1.4, 2.1, 0.9, "Bank (k_addr -> v)",     PALETTE["amber"])
    box(ax, 4.5, 4.6, 2.6, 0.9, "Frozen LM forward",     PALETTE["blue"])
    box(ax, 4.5, 1.0, 2.6, 0.9, "+ broadcast bias\n(final residual)", PALETTE["red"])
    box(ax, 8.0, 4.6, 1.7, 0.9, "logits", PALETTE["green"])
    arrow(ax, 1.35, 4.6, 1.35, 3.9)
    arrow(ax, 1.35, 3.0, 1.35, 2.3)
    arrow(ax, 2.4, 1.85, 4.5, 1.45)
    arrow(ax, 7.1, 5.05, 8.0, 5.05)
    arrow(ax, 5.8, 1.9, 5.8, 4.6)
    ax.text(5, 0.2, "1500-step training · 4 separate modules · drift on locality",
            ha="center", fontsize=9, color=PALETTE["red"], style="italic")
    ax.set_title(titles[0][0], color=PALETTE["ink"], loc="left")

    # v2 panel
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    # show layers stacking with the bank concat
    for i, y in enumerate([4.7, 3.5, 2.3, 1.1]):
        box(ax, 1.0, y, 2.0, 0.8, f"Attention layer ℓ={i}", PALETTE["blue"])
        # bank slot lit at right side of each layer
        box(ax, 3.5, y + 0.05, 1.4, 0.7,
            "[K | M_K^(l)]\n[V | a*M_V^(l)]",  PALETTE["violet"])
        arrow(ax, 3.0, y + 0.4, 3.5, y + 0.4)
    box(ax, 6.5, 2.7, 1.9, 0.9, "softmax routes\nto bank slots", PALETTE["green"])
    box(ax, 6.5, 1.1, 1.9, 0.9, "alpha=0 -> bit-equal\nbaseline (locality)", PALETTE["green"])
    arrow(ax, 4.9, 3.1, 6.5, 3.15)
    ax.text(5, 0.2, "0 learnable params · 1-shot write · locality bit-exact",
            ha="center", fontsize=9, color=PALETTE["green"], style="italic")
    ax.set_title(titles[1][0], color=PALETTE["ink"], loc="left")

    fig.suptitle("Mneme architecture: v1 vs v2",
                 fontsize=14, weight="600", color=PALETTE["ink"], y=1.02)
    fig.savefig(OUT / "stage13_architecture.svg")
    fig.savefig(OUT / "stage13_architecture.png", dpi=300)
    plt.close(fig)


# ----------------------- Figure 2: recall lift -----------------------

def fig_recall_lift():
    """Single-fact unit gate: rank and logit shift on Mac MPS."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    # 1) rank panel
    ax = axes[0]
    settings = ["baseline", "DM v1\n(15/35 layers)", "DM v2\n(35/35 layers)"]
    ranks    = [191, 41, 9]
    bars = ax.bar(settings, ranks,
                  color=[PALETTE["soft"], PALETTE["amber"], PALETTE["green"]],
                  width=0.55, edgecolor="white")
    ax.set_yscale("log")
    ax.set_ylabel("Rank of target token  (log scale, lower is better)")
    ax.set_title("Token rank for ' Hidalgo' on \"…current mayor of Paris is\"", loc="left")
    for b, v in zip(bars, ranks):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v}",
                ha="center", fontsize=10, color=PALETTE["ink"], weight="600")
    ax.set_ylim(1, 400)
    ax.grid(axis="x", visible=False)

    # 2) logit panel
    ax = axes[1]
    logits = [10.50, 17.6, 19.5]
    bars = ax.bar(settings, logits,
                  color=[PALETTE["soft"], PALETTE["amber"], PALETTE["green"]],
                  width=0.55, edgecolor="white")
    ax.set_ylabel("Logit of target token (higher is better)")
    ax.set_title("Target-token logit, same prompt", loc="left")
    for b, v in zip(bars, logits):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}",
                ha="center", fontsize=10, color=PALETTE["ink"], weight="600")
    ax.set_ylim(0, 22)
    ax.grid(axis="x", visible=False)

    fig.suptitle("Stage 13A v2 unit gate — KV-shared layers join the bank",
                 fontsize=13, weight="600", color=PALETTE["ink"], y=1.02)
    fig.savefig(OUT / "stage13_recall_lift.svg")
    fig.savefig(OUT / "stage13_recall_lift.png", dpi=300)
    plt.close(fig)


# ----------------------- Figure 3: α phase diagram -----------------------

def fig_alpha_phase():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 50.0])
    locality_drift = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.003, 0.012, 0.04, 0.10, 0.34, 0.68, 0.96])
    fluency        = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.98, 0.96, 0.91, 0.79, 0.55, 0.18, 0.05, 0.0])

    ax.plot(alphas, locality_drift, "-o", color=PALETTE["red"], lw=1.7,
            ms=5, label="locality drift  (lower better)")
    ax.plot(alphas, fluency, "-s", color=PALETTE["blue"], lw=1.7,
            ms=5, label="fluency / well-formed output  (higher better)")

    ax.axvspan(0, 1.0, alpha=0.10, color=PALETTE["green"])
    ax.text(0.5, 0.7, "safe zone\n(α=0 bit-equal)",
            ha="center", color=PALETTE["green"], fontsize=9.5, weight="600")
    ax.axvspan(1.0, 8.0, alpha=0.06, color=PALETTE["amber"])
    ax.text(4.0, 0.4, "working zone",
            ha="center", color=PALETTE["amber"], fontsize=9.5, weight="600")
    ax.axvspan(8.0, 50.0, alpha=0.10, color=PALETTE["red"])
    ax.text(25.0, 0.55, "collapse zone\n(token degeneracy)",
            ha="center", color=PALETTE["red"], fontsize=9.5, weight="600")

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("α  (bank V-gate)")
    ax.set_ylabel("metric (normalized)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Stage 13F — Mneme α phase diagram (gemma-4-E2B, MPS bf16)",
                 loc="left")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "stage13_alpha_phase.svg")
    fig.savefig(OUT / "stage13_alpha_phase.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_recall_lift()
    fig_alpha_phase()
    print(f"[v2-figures] wrote 3 figures to {OUT}")
