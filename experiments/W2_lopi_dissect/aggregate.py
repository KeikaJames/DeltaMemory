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
    if "Qwen" in name:
        return "Qwen2.5-0.5B"
    if "llama" in name.lower() or "Llama" in name:
        return "Llama-3.2-1B"
    return name.split("/")[-1][:12]


ARM_STYLES: dict[str, dict] = {
    "A0": {"color": "#888888", "linestyle": "-",  "marker": "o", "label": "A0 (no LOPI)"},
    "A1": {"color": "#e74c3c", "linestyle": "--", "marker": "s", "label": "A1 (M⊥ only)"},
    "A2": {"color": "#e67e22", "linestyle": "-.", "marker": "^", "label": "A2 (M⊥+Gauss)"},
    "A3": {"color": "#2ecc71", "linestyle": "-",  "marker": "D", "label": "A3 (full LOPI)"},
    "A4": {"color": "#3498db", "linestyle": "--", "marker": "v", "label": "A4 (Gauss+γ)"},
}


def fig1_arm_vs_alpha(df, out_dir: Path):
    """5 arm lines of mean_drift vs α, faceted by model (one subplot per model)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(df["model"].unique())
    arms = ["A0", "A1", "A2", "A3", "A4"]
    alphas = sorted(df["alpha"].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), sharey=False)
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
                    linewidth=1.5, markersize=5)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("α (log₂ scale)")
        ax.set_ylabel("mean drift (nats)")
        ax.set_title(_get_model_short(model))
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.05), fontsize=9)
    fig.suptitle("W.2 LOPI Dissect — mean drift vs α (all arms)", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
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
    palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22"]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    for i, model in enumerate(models):
        num_sub = df[(df["model"] == model) & (df["arm"] == arm_num)].set_index("alpha")
        den_sub = df[(df["model"] == model) & (df["arm"] == arm_den)].set_index("alpha")
        alphas_here = sorted(set(num_sub.index) & set(den_sub.index))
        if not alphas_here:
            continue
        deltas = [float(num_sub.loc[a, "mean_drift"]) - float(den_sub.loc[a, "mean_drift"])
                  for a in alphas_here]
        ax.plot(alphas_here, deltas, color=palette[i % len(palette)],
                marker="o", label=_get_model_short(model), linewidth=1.5, markersize=5)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("α (log₂ scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
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
    fig, axes = plt.subplots(1, len(alphas), figsize=(3 * len(alphas), 3.5), sharey=True)
    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        vals = gt_dict[alpha]
        ax.hist(vals, bins=20, color="#2ecc71", edgecolor="white", alpha=0.8)
        ax.set_title(f"α={alpha:.1f}")
        ax.set_xlabel("γ_t mean")
        ax.axvline(float(np.median(vals)), color="red", linewidth=1.5,
                   linestyle="--", label=f"p50={np.median(vals):.3f}")
        ax.legend(fontsize=7)

    axes[0].set_ylabel("count")
    fig.suptitle("W.2 — lopi_gamma_t distribution (A3 arm, all models/seeds/prompts)")
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
        diffs_q1 = []
        for model in df["model"].unique():
            for alpha in high_alpha:
                try:
                    d0 = float(a0.loc[(model, alpha), "mean_drift"])
                    d1 = float(a1.loc[(model, alpha), "mean_drift"])
                    diffs_q1.append(d1 - d0)
                except KeyError:
                    pass
        if diffs_q1:
            mean_q1 = float(np.mean(diffs_q1))
            # PASS if A1 ≤ A0 on average at high α
            verdicts["Q1"] = f"PASS (A1−A0={mean_q1:+.3f} nats at α≥2)" if mean_q1 <= 0 else \
                             f"FAIL (A1−A0={mean_q1:+.3f} nats at α≥2; M⊥ increases drift)"
        else:
            verdicts["Q1"] = "INSUFFICIENT_DATA"
    except Exception as e:
        verdicts["Q1"] = f"ERROR: {e}"

    # Q2: Gaussian gate — A2 vs A1
    try:
        a2 = df[df["arm"] == "A2"].set_index(["model", "alpha"])
        diffs_q2 = []
        for model in df["model"].unique():
            for alpha in [a for a in df["alpha"].unique() if a >= 2]:
                try:
                    d1 = float(a1.loc[(model, alpha), "mean_drift"])
                    d2 = float(a2.loc[(model, alpha), "mean_drift"])
                    diffs_q2.append(d2 - d1)
                except KeyError:
                    pass
        if diffs_q2:
            mean_q2 = float(np.mean(diffs_q2))
            verdicts["Q2"] = f"PASS (A2−A1={mean_q2:+.3f} nats; Gaussian reduces drift)" if mean_q2 < -0.05 else \
                             f"FAIL (A2−A1={mean_q2:+.3f} nats; Gaussian effect marginal)"
        else:
            verdicts["Q2"] = "INSUFFICIENT_DATA"
    except Exception as e:
        verdicts["Q2"] = f"ERROR: {e}"

    # Q3: Derivative gate — A3 vs A2
    try:
        a3 = df[df["arm"] == "A3"].set_index(["model", "alpha"])
        diffs_q3 = []
        for model in df["model"].unique():
            for alpha in [a for a in df["alpha"].unique() if a >= 1]:
                try:
                    d2 = float(a2.loc[(model, alpha), "mean_drift"])
                    d3 = float(a3.loc[(model, alpha), "mean_drift"])
                    diffs_q3.append(d3 - d2)
                except KeyError:
                    pass
        if diffs_q3:
            mean_q3 = float(np.mean(diffs_q3))
            verdicts["Q3"] = f"PASS (A3−A2={mean_q3:+.3f} nats; derivative gate fires)" if mean_q3 < -0.05 else \
                             f"FAIL (A3−A2={mean_q3:+.3f} nats; gate pinned at 1.0)"
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
    ap.add_argument("--out", default="experiments/W2_lopi_dissect/")
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
