"""W.2 LOPI dissect — aggregate & figure generator.

Reads cells.jsonl (or cells.jsonl.gz) produced by run.py and:
  1. Computes summary statistics per (model, arm, alpha)
  2. Generates 5 SVG figures
  3. Prints Q1/Q2/Q3 verdict table to stdout

Usage
-----
    python experiments/W2_lopi_dissect/aggregate.py \\
        --cells experiments/W2_lopi_dissect/cells.jsonl \\
        --out   experiments/W2_lopi_dissect/

Figures produced
----------------
fig1_arm_vs_alpha.svg   — mean_drift vs α, 5 arm lines, faceted by model
fig2_marginal_M_perp.svg — (A1 − A0) drift vs α
fig3_marginal_gauss.svg  — (A2 − A1) drift vs α
fig4_marginal_deriv.svg  — (A3 − A2) drift vs α
fig5_gamma_t_hist.svg    — γ_t distribution faceted by α (A3 arm only)
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Load cells


def load_cells(path: Path) -> list[dict]:
    open_fn = gzip.open if str(path).endswith(".gz") else open
    cells = []
    with open_fn(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                cells.append(json.loads(line))
    return cells


# ---------------------------------------------------------------------------
# Aggregation helpers


def mean_drift_table(cells: list[dict]) -> dict:
    """Return {(model, arm, alpha) -> list of drift values}."""
    table: dict[tuple, list] = {}
    for c in cells:
        key = (c["model"], c["arm"], float(c["alpha"]))
        table.setdefault(key, []).append(float(c["drift"]))
    return table


def mean_drift_df(cells: list[dict]):
    """Return a pandas DataFrame with columns: model, arm, alpha, mean_drift, std_drift, n."""
    import pandas as pd

    rows = []
    table = mean_drift_table(cells)
    for (model, arm, alpha), drifts in table.items():
        rows.append({
            "model": model,
            "arm": arm,
            "alpha": alpha,
            "mean_drift": float(np.mean(drifts)),
            "std_drift": float(np.std(drifts)),
            "n": len(drifts),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "arm", "alpha"]).reset_index(drop=True)
    return df


def gamma_t_values(cells: list[dict], arm: str = "A3") -> dict:
    """Return {alpha -> list of lopi_gamma_t_mean values} for a given arm."""
    result: dict[float, list] = {}
    for c in cells:
        if c.get("arm") != arm:
            continue
        val = c.get("lopi_gamma_t_mean")
        if val is None:
            continue
        alpha = float(c["alpha"])
        result.setdefault(alpha, []).append(float(val))
    return result


# ---------------------------------------------------------------------------
# Figure helpers (matplotlib)


def _get_model_short(name: str) -> str:
    if "gpt2-medium" in name:
        return "GPT-2-M"
    if "Qwen2.5-0.5B" in name:
        return "Qwen2.5-0.5B"
    if "Qwen2.5-1.5B" in name:
        return "Qwen2.5-1.5B"
    if "Qwen3" in name:
        return name.split("/")[-1].replace("-Instruct-2507", "")
    if "Qwen" in name:
        return name.split("/")[-1]
    if "llama" in name.lower() or "Llama" in name:
        return "Llama-3.2-1B"
    return name.split("/")[-1][:12]


PAPER_COLORS = {
    "blue": "#4169E1",
    "orange": "#F4A261",
    "green": "#2A9D8F",
    "purple": "#8E6BBE",
    "red": "#E76F51",
    "gray": "#7A7F87",
    "light_grid": "#D8DEE9",
    "text": "#1F2937",
}


ARM_STYLES: dict[str, dict] = {
    "A0": {"color": PAPER_COLORS["gray"], "linestyle": "-",  "marker": "o", "label": "A0 no LOPI"},
    "A1": {"color": PAPER_COLORS["red"], "linestyle": "-", "marker": "s", "label": "A1 M_perp"},
    "A2": {"color": PAPER_COLORS["orange"], "linestyle": "-", "marker": "^", "label": "A2 M_perp+Gauss"},
    "A3": {"color": PAPER_COLORS["green"], "linestyle": "-",  "marker": "D", "label": "A3 full"},
    "A4": {"color": PAPER_COLORS["blue"], "linestyle": "-", "marker": "v", "label": "A4 Gauss+gamma"},
}


def _apply_paper_style(ax, *, ylabel: str | None = None) -> None:
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=PAPER_COLORS["light_grid"], linewidth=0.8, alpha=0.7)
    ax.grid(True, axis="x", color=PAPER_COLORS["light_grid"], linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAB2BD")
    ax.spines["bottom"].set_color("#AAB2BD")
    ax.tick_params(colors=PAPER_COLORS["text"], labelsize=9)
    ax.set_xlabel("alpha (log2 scale)", fontsize=10, color=PAPER_COLORS["text"])
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10, color=PAPER_COLORS["text"])


def fig1_arm_vs_alpha(df, out_dir: Path):
    """5 arm lines of mean_drift vs α, faceted by model (one subplot per model)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(df["model"].unique())
    arms = ["A0", "A1", "A2", "A3", "A4"]
    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 3.8), sharey=False)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        for arm in arms:
            sub = df[(df["model"] == model) & (df["arm"] == arm)].sort_values("alpha")
            if sub.empty:
                continue
            style = ARM_STYLES.get(arm, {})
            ax.plot(sub["alpha"], sub["mean_drift"],
                    color=style.get("color", "black"),
                    linestyle=style.get("linestyle", "-"),
                    marker=style.get("marker", "o"),
                    label=style.get("label", arm),
                    linewidth=2.2, markersize=4.8, markeredgewidth=0)

        ax.set_xscale("log", base=2)
        _apply_paper_style(ax, ylabel="mean drift (nats)")
        ax.set_title(_get_model_short(model), fontsize=11, color=PAPER_COLORS["text"], pad=8)
        ax.axhline(0, color="#111827", linewidth=0.8, linestyle=":")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.03), fontsize=9, frameon=False)
    fig.suptitle("W.2 LOPI ablation: neutral drift by alpha", fontsize=13,
                 color=PAPER_COLORS["text"], y=1.12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = out_dir / "fig1_arm_vs_alpha.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[agg] saved {out}")


def fig2_marginal_M_perp(df, out_dir: Path):
    """(A1 − A0) vs α per model."""
    _marginal_figure(df, "A1", "A0", "A1 − A0 (M⊥ effect)", "marginal M⊥ (nats)",
                     out_dir / "fig2_marginal_M_perp.svg")


def fig3_marginal_gauss(df, out_dir: Path):
    """(A2 − A1) vs α per model."""
    _marginal_figure(df, "A2", "A1", "A2 − A1 (Gaussian effect)", "marginal Gaussian (nats)",
                     out_dir / "fig3_marginal_gauss.svg")


def fig4_marginal_deriv(df, out_dir: Path):
    """(A3 − A2) vs α per model."""
    _marginal_figure(df, "A3", "A2", "A3 − A2 (γ_t effect)", "marginal γ_t (nats)",
                     out_dir / "fig4_marginal_deriv.svg")


def _marginal_figure(df, arm_num: str, arm_den: str, title: str, ylabel: str, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(df["model"].unique())
    alphas = sorted(df["alpha"].unique())
    palette = [PAPER_COLORS["blue"], PAPER_COLORS["orange"], PAPER_COLORS["green"],
               PAPER_COLORS["purple"], PAPER_COLORS["red"]]

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.0))

    for i, model in enumerate(models):
        num_sub = df[(df["model"] == model) & (df["arm"] == arm_num)].set_index("alpha")
        den_sub = df[(df["model"] == model) & (df["arm"] == arm_den)].set_index("alpha")
        alphas_here = sorted(set(num_sub.index) & set(den_sub.index))
        if not alphas_here:
            continue
        deltas = [float(num_sub.loc[a, "mean_drift"]) - float(den_sub.loc[a, "mean_drift"])
                  for a in alphas_here]
        ax.plot(alphas_here, deltas, color=palette[i % len(palette)],
                marker="o", label=_get_model_short(model), linewidth=2.4,
                markersize=5, markeredgewidth=0)

    ax.set_xscale("log", base=2)
    _apply_paper_style(ax, ylabel=ylabel)
    ax.set_title(title, fontsize=12, color=PAPER_COLORS["text"], pad=10)
    ax.axhline(0, color="#111827", linewidth=0.9, linestyle="--")
    ax.legend(fontsize=9, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[agg] saved {out}")


def fig5_gamma_t_hist(cells: list[dict], out_dir: Path):
    """γ_t distribution over tokens, faceted by α (A3 arm only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_dict = gamma_t_values(cells, arm="A3")
    if not gt_dict:
        print("[agg] no lopi_gamma_t data for fig5 (A3 arm may not have run yet)")
        return

    alphas = sorted(gt_dict.keys())
    fig, axes = plt.subplots(1, len(alphas), figsize=(2.7 * len(alphas), 3.4), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        vals = gt_dict[alpha]
        med = float(np.median(vals))
        ax.hist(vals, bins=np.linspace(0, 1.05, 22), color=PAPER_COLORS["green"],
                edgecolor="white", linewidth=0.8, alpha=0.85)
        ax.set_title(f"alpha={alpha:.1f}", fontsize=10, color=PAPER_COLORS["text"])
        ax.set_xlabel("gamma_t mean", fontsize=9)
        ax.axvline(med, color=PAPER_COLORS["red"], linewidth=1.6,
                   linestyle="--", label=f"p50={med:.3f}")
        _apply_paper_style(ax)
        ax.legend(fontsize=7, frameon=False)

    axes[0].set_ylabel("count")
    fig.suptitle("W.2 derivative gate distribution (A3, all models/seeds/prompts)",
                 fontsize=12, color=PAPER_COLORS["text"], y=1.03)
    fig.tight_layout()
    out = out_dir / "fig5_gamma_t_hist.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[agg] saved {out}")


# ---------------------------------------------------------------------------
# Verdict computation


def compute_verdicts(df) -> dict[str, str]:
    """Compute Q1/Q2/Q3 pass/fail verdicts from aggregate data."""
    verdicts: dict[str, str] = {}

    # Q1: M⊥ utility — A1 vs A0 at α ≥ 2
    try:
        a0 = df[df["arm"] == "A0"].set_index(["model", "alpha"])
        a1 = df[df["arm"] == "A1"].set_index(["model", "alpha"])
        high_alpha = [a for a in df["alpha"].unique() if a >= 2]
        model_means = []
        for model in df["model"].unique():
            model_diffs = []
            for alpha in high_alpha:
                try:
                    d0 = float(a0.loc[(model, alpha), "mean_drift"])
                    d1 = float(a1.loc[(model, alpha), "mean_drift"])
                    model_diffs.append(d1 - d0)
                except KeyError:
                    pass
            if model_diffs:
                model_means.append(float(np.mean(model_diffs)))
        if model_means:
            n_pass = sum(1 for d in model_means if d <= 0)
            mean_q1 = float(np.mean(model_means))
            verdicts["Q1"] = (
                f"PASS ({n_pass}/{len(model_means)} models; A1−A0={mean_q1:+.3f} nats at α≥2)"
                if n_pass == len(model_means) else
                f"FAIL ({n_pass}/{len(model_means)} models; A1−A0={mean_q1:+.3f} nats at α≥2)"
            )
        else:
            verdicts["Q1"] = "INSUFFICIENT_DATA"
    except Exception as e:
        verdicts["Q1"] = f"ERROR: {e}"

    # Q2: Gaussian gate — A2 vs A1
    try:
        a2 = df[df["arm"] == "A2"].set_index(["model", "alpha"])
        model_means = []
        for model in df["model"].unique():
            model_diffs = []
            for alpha in [a for a in df["alpha"].unique() if a >= 2]:
                try:
                    d1 = float(a1.loc[(model, alpha), "mean_drift"])
                    d2 = float(a2.loc[(model, alpha), "mean_drift"])
                    model_diffs.append(d2 - d1)
                except KeyError:
                    pass
            if model_diffs:
                model_means.append(float(np.mean(model_diffs)))
        if model_means:
            n_pass = sum(1 for d in model_means if d < -0.05)
            mean_q2 = float(np.mean(model_means))
            verdicts["Q2"] = (
                f"PASS ({n_pass}/{len(model_means)} models; A2−A1={mean_q2:+.3f} nats)"
                if n_pass == len(model_means) else
                f"FAIL ({n_pass}/{len(model_means)} models; A2−A1={mean_q2:+.3f} nats)"
            )
        else:
            verdicts["Q2"] = "INSUFFICIENT_DATA"
    except Exception as e:
        verdicts["Q2"] = f"ERROR: {e}"

    # Q3: Derivative gate — A3 vs A2
    try:
        a3 = df[df["arm"] == "A3"].set_index(["model", "alpha"])
        model_means = []
        for model in df["model"].unique():
            model_diffs = []
            for alpha in [a for a in df["alpha"].unique() if a >= 1]:
                try:
                    d2 = float(a2.loc[(model, alpha), "mean_drift"])
                    d3 = float(a3.loc[(model, alpha), "mean_drift"])
                    model_diffs.append(d3 - d2)
                except KeyError:
                    pass
            if model_diffs:
                model_means.append(float(np.mean(model_diffs)))
        if model_means:
            n_pass = sum(1 for d in model_means if d < -0.05)
            mean_q3 = float(np.mean(model_means))
            verdicts["Q3"] = (
                f"PASS ({n_pass}/{len(model_means)} models; A3−A2={mean_q3:+.3f} nats)"
                if n_pass == len(model_means) else
                f"FAIL ({n_pass}/{len(model_means)} models; A3−A2={mean_q3:+.3f} nats; gate may be pinned)"
            )
        else:
            verdicts["Q3"] = "INSUFFICIENT_DATA"
    except Exception as e:
        verdicts["Q3"] = f"ERROR: {e}"

    return verdicts


# ---------------------------------------------------------------------------
# Main


def main():
    ap = argparse.ArgumentParser(description="W.2 aggregate & figures")
    ap.add_argument("--cells", default="experiments/W2_lopi_dissect/cells.jsonl")
    ap.add_argument("--out", default="/tmp/deltamemory/W2_lopi_dissect/")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cells_path = Path(args.cells)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cells_path.exists():
        # Try .gz variant
        gz_path = cells_path.with_suffix(".jsonl.gz")
        if gz_path.exists():
            cells_path = gz_path
        else:
            print(f"[agg] cells file not found: {cells_path}", file=sys.stderr)
            sys.exit(1)

    print(f"[agg] loading {cells_path}...", flush=True)
    cells = load_cells(cells_path)
    print(f"[agg] {len(cells)} cells loaded", flush=True)

    if not cells:
        print("[agg] no cells — nothing to aggregate")
        return

    import pandas as pd
    df = mean_drift_df(cells)
    print(f"[agg] unique (model, arm, alpha) combos: {len(df)}")
    print(df.to_string(index=False))

    # Save aggregate CSV
    csv_out = out_dir / "aggregate.csv"
    df.to_csv(csv_out, index=False)
    print(f"[agg] saved {csv_out}")

    verdicts = compute_verdicts(df)
    print("\n=== Q1/Q2/Q3 VERDICTS ===")
    for q, v in verdicts.items():
        print(f"  {q}: {v}")

    # Save verdicts
    verdicts_out = out_dir / "verdicts.json"
    with open(verdicts_out, "w") as f:
        json.dump(verdicts, f, indent=2)
    print(f"[agg] saved {verdicts_out}")

    if not args.no_figures:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("[agg] matplotlib not found — skipping figures")
            return

        fig1_arm_vs_alpha(df, out_dir)
        fig2_marginal_M_perp(df, out_dir)
        fig3_marginal_gauss(df, out_dir)
        fig4_marginal_deriv(df, out_dir)
        fig5_gamma_t_hist(cells, out_dir)
        print("[agg] all figures done")


if __name__ == "__main__":
    main()
