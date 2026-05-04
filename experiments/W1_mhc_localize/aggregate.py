"""W.1 aggregate — reads cells.jsonl, produces summary.csv, 4 SVG plots, verdict.md.

Usage:
    python -m experiments.W1_mhc_localize.aggregate \
        --input experiments/W1_mhc_localize/cells.jsonl \
        --outdir experiments/W1_mhc_localize/figures
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_cells(path: Path) -> list[dict]:
    cells = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                cells.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return cells


def _safe_mean(vals):
    vals = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def aggregate(cells: list[dict]) -> "pd.DataFrame":
    import pandas as pd
    df = pd.DataFrame(cells)
    for col in ["mean_drift", "bank_col_sum_p99", "attn_entropy_bank_mean",
                "m_perp_energy_ratio_mean", "residual_norm_p50"]:
        if col not in df.columns:
            df[col] = float("nan")
    return df


# ---------------------------------------------------------------------------
# PASS/PARTIAL/FAIL verdict logic
# ---------------------------------------------------------------------------

PASS_THRESHOLD = 0.5   # nats: mean_drift(shield=on, V-scale=on, α≥1) ≤ 0.5

def compute_verdict(df) -> dict:
    """Compute per-model PASS/FAIL and experiment-level verdict."""
    import pandas as pd

    models = df["model"].unique()
    results = {}

    for model in models:
        m = df[
            (df["model"] == model) &
            (df["shield"] == True) &          # noqa: E712
            (df["v_scale"] == True) &          # noqa: E712
            (df["alpha"] >= 1.0)
        ]
        if len(m) == 0:
            results[model] = {"status": "no_data", "mean_drift": float("nan")}
            continue
        avg_drift = _safe_mean(m["mean_drift"].tolist())
        status = "PASS" if avg_drift <= PASS_THRESHOLD else "FAIL"
        results[model] = {"status": status, "mean_drift": avg_drift}

    pass_count = sum(1 for v in results.values() if v["status"] == "PASS")
    total = len(results)
    if total == 0:
        verdict = "NO_DATA"
    elif pass_count >= 5:
        verdict = "PASS"
    elif pass_count >= 3:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {"model_results": results, "pass_count": pass_count, "total": total,
            "verdict": verdict, "threshold": PASS_THRESHOLD}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _marginal_plot(df, x_var: str, outdir: Path, models=None):
    """Holding others fixed (marginalised), plot mean_drift vs x_var, one line per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if models is None:
        models = sorted(df["model"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    x_vals = sorted(df[x_var].unique())

    for model in models:
        sub = df[df["model"] == model]
        ys = []
        for xv in x_vals:
            rows = sub[sub[x_var] == xv]["mean_drift"]
            ys.append(_safe_mean(rows.tolist()))
        ax.plot(x_vals, ys, marker="o", label=model.split("/")[-1])

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(PASS_THRESHOLD, color="red", linestyle=":", linewidth=1.0,
               alpha=0.8, label=f"PASS threshold ({PASS_THRESHOLD} nats)")
    ax.set_xlabel(x_var)
    ax.set_ylabel("mean_drift (nats)")
    ax.set_title(f"mean_drift vs {x_var} (others marginalised)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fname = outdir / f"holding_others_fixed_{x_var}.svg"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def make_plots(df, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    for x_var in ["alpha", "shield", "kappa", "v_scale"]:
        if x_var in df.columns:
            _marginal_plot(df, x_var, outdir)
        else:
            print(f"  Skipping plot for {x_var}: column not found")


# ---------------------------------------------------------------------------
# Write verdict.md
# ---------------------------------------------------------------------------

def write_verdict(verdict_info: dict, df, outdir: Path):
    lines = ["# W.1 Verdict\n"]
    lines.append(f"**Overall verdict: {verdict_info['verdict']}**  \n")
    lines.append(f"PASS threshold: mean_drift(shield=on, V-scale=on, α≥1) ≤ {verdict_info['threshold']} nats  \n")
    lines.append(f"Models passing: {verdict_info['pass_count']}/{verdict_info['total']}\n\n")
    lines.append("## Per-model results\n\n")
    lines.append("| Model | Status | mean_drift (shield+V-scale, α≥1) |\n")
    lines.append("|-------|--------|-----------------------------------|\n")
    for model, r in verdict_info["model_results"].items():
        drift_str = f"{r['mean_drift']:.4f}" if not math.isnan(r['mean_drift']) else "N/A"
        lines.append(f"| {model} | **{r['status']}** | {drift_str} |\n")

    lines.append("\n## DH1: Shield truncates col-sums?\n\n")
    if "bank_col_sum_p99" in df.columns:
        shon = df[df["shield"] == True]["bank_col_sum_p99"].dropna()  # noqa: E712
        shoff = df[df["shield"] == False]["bank_col_sum_p99"].dropna()  # noqa: E712
        if len(shon) and len(shoff):
            lines.append(f"bank_col_sum_p99 shield=ON: {shon.mean():.4f}  \n")
            lines.append(f"bank_col_sum_p99 shield=OFF: {shoff.mean():.4f}  \n")
            if shon.mean() < shoff.mean():
                lines.append("→ DH1 **CONFIRMED**: shield reduces col-sum p99\n\n")
            else:
                lines.append("→ DH1 **NOT CONFIRMED**: col-sums not reduced by shield\n\n")
    else:
        lines.append("(no bank_col_sum_p99 data)\n\n")

    # Write to file
    outpath = outdir / "verdict.md"
    with open(outpath, "w") as fh:
        fh.writelines(lines)
    print(f"  Saved {outpath}")
    return verdict_info["verdict"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="W.1 aggregate")
    parser.add_argument("--input", default="experiments/W1_mhc_localize/cells.jsonl")
    parser.add_argument("--outdir", default="experiments/W1_mhc_localize/figures")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    cells = load_cells(input_path)
    print(f"Loaded {len(cells)} cell records from {input_path}")

    if not cells:
        print("No data to aggregate.")
        return

    df = aggregate(cells)
    print(f"DataFrame shape: {df.shape}")

    # Save summary CSV
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir.parent / "summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Saved summary CSV: {summary_path}")

    # Compute and write verdict
    verdict_info = compute_verdict(df)
    verdict_str = write_verdict(verdict_info, df, outdir)
    print(f"\nVerdict: {verdict_str}")
    print(f"Pass: {verdict_info['pass_count']}/{verdict_info['total']}")

    # Generate plots
    try:
        make_plots(df, outdir)
    except ImportError as e:
        print(f"  Could not generate plots (matplotlib missing?): {e}")

    return verdict_str


if __name__ == "__main__":
    main()
