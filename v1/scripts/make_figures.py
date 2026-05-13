"""Render the figures for the v3 PR / report.

Reads:
    reports/cleanroom/stage14_test_gemma4_e2b/{summary.json, stats.json}
    reports/cleanroom/stage14_dev_kproj/summary.json
    reports/cleanroom/stage14_kproj/k_projector.jsonl

Writes (PNG, 200 dpi) into reports/cleanroom/figures/:
    01_test_recall_bars.png       — main result bar chart
    02_dev_vs_test.png            — dev/test sign-flip
    03_v3_minus_b0_per_fact.png   — paired delta histogram on test
    04_kproj_loss_curve.png       — InfoNCE training loss
    05_v3_vs_v2_lift.png          — v3 over v2 (the only positive)
    06_softmax_share.png          — illustrative N→share decay rationale
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "reports/cleanroom/stage14_test_gemma4_e2b"
DEV_DIR = ROOT / "reports/cleanroom/stage14_dev_kproj"
KPROJ_DIR = ROOT / "reports/cleanroom/stage14_kproj"
OUT_DIR = ROOT / "reports/cleanroom/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def _per_fact(summary: dict, name: str) -> np.ndarray:
    rows = summary[name]
    return np.asarray(rows[0]["per_fact"], dtype=float)


def fig_test_bars() -> None:
    s = _load_json(TEST_DIR / "summary.json")
    order = [
        ("B1_prompt_insertion", "B1 prompt", "#2c7a7b"),
        ("B2_rag_oracle",       "B2 RAG-oracle", "#319795"),
        ("B0_no_memory",        "B0 no-memory", "#718096"),
        ("v3_period_kproj",     "v3 (frozen)", "#c05621"),
        ("v2_period_no_kproj",  "v2", "#a0aec0"),
    ]
    means = [np.mean([r["recall_at_1_mean"] for r in s[k]]) for k, _, _ in order]
    labels = [lab for _, lab, _ in order]
    colors = [c for _, _, c in order]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, color=colors, edgecolor="black", linewidth=0.6)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width()/2, m + 0.015, f"{m:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("recall@1 (paraphrase mean)")
    ax.set_ylim(0, 0.78)
    ax.set_title("Phase G — held-out test (Gemma-4-E2B, N=39 facts)")
    ax.axhline(means[2], ls="--", color="#718096", lw=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_test_recall_bars.png")
    plt.close(fig)


def fig_dev_vs_test() -> None:
    dev = _load_json(DEV_DIR / "summary.json")
    test = _load_json(TEST_DIR / "summary.json")

    def _mean(s: dict, k: str) -> float:
        return float(np.mean([r["recall_at_1_mean"] for r in s[k]]))

    # dev_kproj summary uses different key names; tolerate both.
    b0_dev_keys = [k for k in dev if "no_memory" in k or k.endswith("_B0") or k == "B0"]
    v3_dev_keys = [k for k in dev if k == "v3_period_kproj"]
    if not b0_dev_keys or not v3_dev_keys:
        # fallback: just print and skip
        print(f"[fig] dev keys: {list(dev.keys())}; skipping fig 02")
        return
    b0_dev = _mean(dev, b0_dev_keys[0])
    v3_dev = _mean(dev, v3_dev_keys[0])
    b0_test = _mean(test, "B0_no_memory")
    v3_test = _mean(test, "v3_period_kproj")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [b0_dev, b0_test], w, label="B0 no-memory", color="#718096")
    ax.bar(x + w/2, [v3_dev, v3_test], w, label="v3 frozen", color="#c05621")
    ax.set_xticks(x)
    ax.set_xticklabels(["dev", "test"])
    ax.set_ylabel("recall@1")
    ax.set_title("Dev/test sign flip (overfit selection signature)")
    ax.legend(frameon=False, loc="upper right")
    for xi, (a, b) in enumerate(zip([b0_dev, b0_test], [v3_dev, v3_test])):
        ax.text(xi - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(xi + w/2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_dev_vs_test.png")
    plt.close(fig)


def fig_paired_delta() -> None:
    s = _load_json(TEST_DIR / "summary.json")
    b0 = _per_fact(s, "B0_no_memory")
    v3 = _per_fact(s, "v3_period_kproj")
    delta = v3 - b0
    wins = int((delta > 0).sum())
    losses = int((delta < 0).sum())
    ties = int((delta == 0).sum())

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-1.05, 1.05, 23)
    ax.hist(delta, bins=bins, color="#c05621", edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(delta.mean(), color="#2b6cb0", lw=1.2, ls="--",
               label=f"mean Δ = {delta.mean():+.3f}")
    ax.set_xlabel("v3 − B0 per-fact recall@1 (paired)")
    ax.set_ylabel("count of facts")
    ax.set_title(f"Per-fact paired delta on test  (W/L/T = {wins}/{losses}/{ties})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_v3_minus_b0_per_fact.png")
    plt.close(fig)


def fig_kproj_loss() -> None:
    p = KPROJ_DIR / "k_projector.jsonl"
    if not p.exists():
        return
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    steps = [r.get("step", i) for i, r in enumerate(rows)]
    losses = [r["loss"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, losses, color="#2b6cb0", lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("InfoNCE loss")
    ax.set_title(f"K-projector training  (start={losses[0]:.3f} → end={losses[-1]:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_kproj_loss_curve.png")
    plt.close(fig)


def fig_v3_vs_v2() -> None:
    s = _load_json(TEST_DIR / "summary.json")
    v2 = _per_fact(s, "v2_period_no_kproj")
    v3 = _per_fact(s, "v3_period_kproj")
    delta = v3 - v2
    wins = int((delta > 0).sum())
    losses = int((delta < 0).sum())
    ties = int((delta == 0).sum())

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-1.05, 1.05, 23)
    ax.hist(delta, bins=bins, color="#2b6cb0", edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(delta.mean(), color="#c05621", lw=1.2, ls="--",
               label=f"mean Δ = {delta.mean():+.3f}")
    ax.set_xlabel("v3 − v2 per-fact recall@1 (paired)")
    ax.set_ylabel("count of facts")
    ax.set_title(f"InfoNCE projector lift over untrained bank  (W/L/T = {wins}/{losses}/{ties})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_v3_vs_v2_lift.png")
    plt.close(fig)


def fig_softmax_share() -> None:
    """Illustrative: softmax share of one bank slot vs N at fixed score gap.

    For a slot with logit advantage delta over the *mean* sequence logit,
    its softmax share is e^delta / (T_seq + N * e^delta_mean). We show
    how at modest N the bank fraction collapses, motivating top-k or
    cosine-only structural fixes named in the methodology amendment.
    """
    Ns = np.arange(1, 100)
    fig, ax = plt.subplots(figsize=(7, 4))
    for delta, color in [(2.0, "#c05621"), (3.0, "#2b6cb0"), (4.0, "#2c7a7b")]:
        seq_mass = 200 * np.exp(0.0)  # T=200 seq tokens at mean logit 0
        bank_mass = Ns * np.exp(0.0)  # N bank slots at mean logit 0
        target_mass = np.exp(delta)   # one privileged slot
        share = target_mass / (seq_mass + bank_mass + target_mass)
        ax.plot(Ns, share, color=color, label=f"target − mean = {delta:.0f} nats")
    ax.set_xlabel("N (number of bank slots)")
    ax.set_ylabel("softmax share of the single correct bank slot")
    ax.set_title("Softmax dilution rationale  (T_seq=200; mean bank logit = 0)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_softmax_share.png")
    plt.close(fig)


def main() -> None:
    fig_test_bars()
    fig_dev_vs_test()
    fig_paired_delta()
    fig_kproj_loss()
    fig_v3_vs_v2()
    fig_softmax_share()
    print(f"[figures] wrote 6 figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
