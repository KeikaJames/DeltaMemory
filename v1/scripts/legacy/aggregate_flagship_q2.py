"""Phase Q4/Q5 — Aggregate Q2 sweep data with bootstrap CIs, Wilcoxon, and figures.

Reads per-model JSONs from ``reports/cleanroom/flagship_v32/Q2/``, computes
paired bootstrap 95% CIs and Wilcoxon signed-rank tests on shield ON vs OFF,
and emits (a) ``AGGREGATE.json``, (b) ``REPORT.md``, (c) SVG figures.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _bootstrap_ci(values, n_resample: int = 10000):
    """Bootstrap 95% CI of paired differences via simple numpy-free resample."""
    import random
    random.seed(42)
    n = len(values)
    means = []
    for _ in range(n_resample):
        sample = [values[random.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resample)]
    hi = means[int(0.975 * n_resample)]
    mean = sum(values) / n
    return {"mean": mean, "ci95_low": lo, "ci95_high": hi}


def _wilcoxon_paired(off_vals, on_vals) -> float:
    """Wilcoxon signed-rank paired test. Returns two-sided p-value (approximate)."""
    n = len(off_vals)
    diffs = [off_vals[i] - on_vals[i] for i in range(n)]
    # Remove zeros
    nonzero = [(abs(d), d > 0) for d in diffs if d != 0]
    if len(nonzero) < 3:
        return 1.0  # too few non-zero diffs
    # Rank absolute differences
    sorted_pairs = sorted(nonzero, key=lambda x: x[0])
    ranks = []
    i = 0
    while i < len(sorted_pairs):
        j = i
        while j < len(sorted_pairs) and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-indexed
        for _ in range(j - i):
            ranks.append(avg_rank)
        i = j
    # Sum of positive ranks
    W_plus = sum(r for (r, (_, pos)) in zip(ranks, sorted_pairs) if pos)
    W_minus = sum(r for (r, (_, pos)) in zip(ranks, sorted_pairs) if not pos)
    W = min(W_plus, W_minus)
    # Normal approximation
    N = len(ranks)
    mu = N * (N + 1) / 4
    sigma = math.sqrt(N * (N + 1) * (2 * N + 1) / 24)
    z = (W - mu) / max(sigma, 1e-9)
    # Two-sided p from normal approximation
    p = 2 * (1 - _norm_cdf(abs(z)))
    return max(p, 1e-10)


def _norm_cdf(x):
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _holm_bonferroni(p_values, m: int = 5) -> list[bool]:
    """Holm-Bonferroni correction. Returns [significant, ...] per hypothesis."""
    n = len(p_values)
    indexed = list(enumerate(p_values))
    indexed.sort(key=lambda x: x[1])
    rejected = [False] * n
    for k, (idx, p) in enumerate(indexed):
        threshold = 0.01 / (m - k)
        if p <= threshold:
            rejected[idx] = True
        else:
            break
    return rejected


def load_q2_data(q2_dir: Path) -> list[dict]:
    """Load all Q2 cell results from per-model JSON files."""
    cells = []
    for json_path in sorted(q2_dir.glob("*.json")):
        if json_path.name in ("AGGREGATE.json", "meta.json"):
            continue
        with open(json_path) as f:
            model_cells = json.load(f)
        if isinstance(model_cells, list):
            cells.extend(model_cells)
        elif isinstance(model_cells, dict) and "model" in model_cells:
            cells.append(model_cells)
    return cells


def compute_stats(cells: list[dict]) -> dict:
    """Aggregate Q2 cells into per-(model,α,shield) stats with bootstrap CIs."""
    # Group by (model, alpha, shield)
    from collections import defaultdict
    groups = defaultdict(list)
    for c in cells:
        key = (c["model"], c["alpha"], c["shield"])
        groups[key].append(c)

    rows = []
    for (model, alpha, shield), group in sorted(groups.items()):
        lifts = [g["mean_lift"] for g in group]
        drifts = [g["nll_drift"] for g in group]
        lift_ci = _bootstrap_ci(lifts)
        drift_ci = _bootstrap_ci(drifts)
        rows.append(dict(
            model=model, alpha=alpha, shield=shield,
            n_seeds=len(group),
            mean_lift=sum(lifts)/len(lifts),
            lift_ci_low=lift_ci["ci95_low"],
            lift_ci_high=lift_ci["ci95_high"],
            mean_drift=sum(drifts)/len(drifts),
            drift_ci_low=drift_ci["ci95_low"],
            drift_ci_high=drift_ci["ci95_high"],
        ))
    return rows


def check_hypotheses(rows: list[dict]) -> dict:
    """Evaluate H1 and H2 per model from aggregated rows."""
    from collections import defaultdict
    models = sorted(set(r["model"] for r in rows))
    alphas = sorted(set(r["alpha"] for r in rows))

    # Build lookup: (model, alpha, shield) -> row
    lut = {}
    for r in rows:
        lut[(r["model"], r["alpha"], r["shield"])] = r

    h1_by_model = {}
    h2_by_model = {}
    for model in models:
        shield_on_drifts = []
        shield_on_lifts = []
        shield_off_vals = []
        shield_on_vals_lift = []
        for alpha in alphas:
            on = lut.get((model, alpha, True))
            off = lut.get((model, alpha, False))
            if on:
                shield_on_drifts.append(abs(on["mean_drift"]))
                shield_on_lifts.append(on["mean_lift"])
            if on and off:
                shield_off_vals.append(off["mean_drift"])
                shield_on_vals_lift.append(on["mean_lift"])

        # H1: drift ≤ 0.5 on ≥ 5/7 α
        h1_pass = sum(1 for d in shield_on_drifts if d <= 0.5) >= 5
        h1_by_model[model] = dict(
            passed=h1_pass,
            n_pass=sum(1 for d in shield_on_drifts if d <= 0.5),
            n_total=len(shield_on_drifts),
            max_drift=max(shield_on_drifts) if shield_on_drifts else None,
        )

        # H2: lift > 0 on ≥ 5/7 α
        h2_pass = sum(1 for l in shield_on_lifts if l > 0) >= 5
        h2_by_model[model] = dict(
            passed=h2_pass,
            n_pass=sum(1 for l in shield_on_lifts if l > 0),
            n_total=len(shield_on_lifts),
        )

    # Wilcoxon: shield ON vs OFF drift per α
    wilcoxon = {}
    for alpha in alphas:
        off_d = []
        on_d = []
        for model in models:
            off = lut.get((model, alpha, False))
            on = lut.get((model, alpha, True))
            if off and on:
                off_d.append(off["mean_drift"])
                on_d.append(on["mean_drift"])
        if len(off_d) >= 3:
            p = _wilcoxon_paired(off_d, on_d)
            wilcoxon[f"alpha_{alpha}"] = dict(
                p_value=round(p, 6),
                shield_better=p < 0.05 and (sum(on_d)/len(on_d) < sum(off_d)/len(off_d)),
            )

    return dict(
        h1=h1_by_model,
        h2=h2_by_model,
        wilcoxon=wilcoxon,
    )


def render_report(models, rows, hypotheses, out_path: Path):
    """Write AGGREGATE.md with tables and hypothesis verdicts."""
    lines = []
    lines.append("# Phase Q2 Aggregate — mHC Shield α-Safety Sweep")
    lines.append("")
    lines.append(f"**Models**: {len(models)}  |  **Date**: 2026-05-04")
    lines.append("")

    # Per-model table
    for model in sorted(models):
        short = model.replace("/", "_")
        lines.append(f"## {model}")
        lines.append("")
        lines.append("| α | Shield OFF lift | Shield OFF drift | Shield ON lift | Shield ON drift |")
        lines.append("|---:|---:|---:|---:|---:|")
        model_rows = [r for r in rows if r["model"] == model]
        alphas = sorted(set(r["alpha"] for r in model_rows))
        for alpha in alphas:
            off = [r for r in model_rows if r["alpha"] == alpha and not r["shield"]]
            on = [r for r in model_rows if r["alpha"] == alpha and r["shield"]]
            off_l = f"{off[0]['mean_lift']:+.3f}" if off else "—"
            off_d = f"{off[0]['mean_drift']:+.3f}" if off else "—"
            on_l = f"{on[0]['mean_lift']:+.3f}" if on else "—"
            on_d = f"{on[0]['mean_drift']:+.3f}" if on else "—"
            lines.append(f"| {alpha:.2f} | {off_l} | {off_d} | {on_l} | {on_d} |")
        lines.append("")

    # Hypotheses
    lines.append("## Hypothesis Verdicts")
    lines.append("")
    lines.append("### H1: shield ON drift ≤ 0.5 nats")
    lines.append("")
    for model in sorted(models):
        h = hypotheses["h1"].get(model, {})
        tag = "✅ PASS" if h.get("passed") else "❌ FAIL"
        md = h.get("max_drift")
        max_drift_str = f"{md:.3f}" if md is not None else "N/A"
        lines.append(f"- **{model}**: {tag}  ({h.get('n_pass',0)}/{h.get('n_total',0)} α pass, max drift={max_drift_str})")

    lines.append("")
    lines.append("### H2: shield ON lift > 0")
    lines.append("")
    for model in sorted(models):
        h = hypotheses["h2"].get(model, {})
        tag = "✅ PASS" if h.get("passed") else "❌ FAIL"
        lines.append(f"- **{model}**: {tag}  ({h.get('n_pass',0)}/{h.get('n_total',0)} α pass)")

    lines.append("")
    lines.append("### Wilcoxon Signed-Rank: shield ON vs OFF drift")
    lines.append("")
    for key, v in sorted(hypotheses.get("wilcoxon", {}).items()):
        better = "shield ON better" if v["shield_better"] else "no significant difference"
        lines.append(f"- {key}: p={v['p_value']:.4f}  ({better})")

    lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    q2_dir = Path("reports/cleanroom/flagship_v32/Q2")
    cells = load_q2_data(q2_dir)
    if not cells:
        print("No Q2 cells found. Run Q2 sweep first.")
        return

    print(f"[aggregate] {len(cells)} cells loaded")
    models = sorted(set(c["model"] for c in cells))
    print(f"[aggregate] models: {models}")

    rows = compute_stats(cells)
    hypotheses = check_hypotheses(rows)

    out_dir = q2_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write AGGREGATE.json
    with open(out_dir / "AGGREGATE.json", "w") as f:
        json.dump(dict(rows=rows, hypotheses=hypotheses), f, indent=2)
    print(f"[aggregate] wrote {out_dir}/AGGREGATE.json")

    # Write AGGREGATE.md
    render_report(models, rows, hypotheses, out_dir / "AGGREGATE.md")
    print(f"[aggregate] wrote {out_dir}/AGGREGATE.md")

    # Generate SVG figure
    render_svg(models, rows, out_dir / "fig_q2_alpha_lift_drift.svg")
    print(f"[aggregate] wrote {out_dir}/fig_q2_alpha_lift_drift.svg")


def render_svg(models, rows, out_path: Path):
    """Generate the primary Q2 lift-vs-drift SVG as multi-model small multiples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {
        False: "#E76F51",  # shield off
        True: "#4169E1",   # shield on
    }
    labels = {
        False: "Shield OFF",
        True: "Shield ON",
    }

    def model_short(name: str) -> str:
        if "gemma-4" in name:
            return "Gemma-4-E2B"
        if "Qwen3" in name:
            return "Qwen3-4B"
        if "GLM-4" in name:
            return "GLM-4-9B"
        if "DeepSeek" in name:
            return "DeepSeek-R1-Distill-Qwen-32B"
        return name.split("/")[-1]

    models_sorted = sorted(models)
    ncols = 2
    nrows = max(1, math.ceil(len(models_sorted) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 4.0 * nrows), squeeze=False)

    for ax, model in zip(axes.flat, models_sorted):
        model_rows = [r for r in rows if r["model"] == model]
        alphas = sorted(set(r["alpha"] for r in model_rows))
        for shield in (False, True):
            lifts = []
            drifts = []
            used_alphas = []
            for alpha in alphas:
                row = [r for r in model_rows if r["alpha"] == alpha and r["shield"] is shield]
                if not row:
                    continue
                lifts.append(row[0]["mean_lift"])
                drifts.append(row[0]["mean_drift"])
                used_alphas.append(alpha)
            if not lifts:
                continue
            ax.plot(
                lifts, drifts,
                color=palette[shield],
                marker="o",
                linewidth=2.4,
                markersize=5,
                markeredgewidth=0,
                label=labels[shield],
            )
            for idx in (0, len(used_alphas) - 1):
                ax.annotate(
                    f"a={used_alphas[idx]:.2g}",
                    (lifts[idx], drifts[idx]),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8,
                    color=palette[shield],
                )

        ax.axhline(0.5, color="#E76F51", linewidth=1.0, linestyle="--", alpha=0.75)
        ax.text(0.99, 0.5, "0.5 drift cap", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="#E76F51")
        ax.axhline(0.0, color="#111827", linewidth=0.8, linestyle=":", alpha=0.65)
        ax.set_title(model_short(model), fontsize=11, pad=8, color="#1F2937")
        ax.set_xlabel("counter-prior lift (nats)", fontsize=10, color="#1F2937")
        ax.set_ylabel("neutral NLL drift (nats)", fontsize=10, color="#1F2937")
        ax.grid(True, axis="y", color="#D8DEE9", linewidth=0.8, alpha=0.7)
        ax.grid(True, axis="x", color="#D8DEE9", linewidth=0.5, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#AAB2BD")
        ax.spines["bottom"].set_color("#AAB2BD")
        ax.tick_params(labelsize=9, colors="#1F2937")

    for ax in axes.flat[len(models_sorted):]:
        ax.axis("off")

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.01), fontsize=10)
    fig.suptitle("Phase Q2 mHC shield: lift vs drift across alpha",
                 fontsize=14, color="#1F2937", y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
