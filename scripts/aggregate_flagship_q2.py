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
    """Generate a simple lift-vs-drift SVG for the primary Q2 figure."""
    model_rows = [r for r in rows if r["model"] == sorted(models)[0]]
    alphas = sorted(set(r["alpha"] for r in model_rows))

    # Collect data series
    off_lift = []
    off_drift = []
    on_lift = []
    on_drift = []
    for alpha in alphas:
        off = [r for r in model_rows if r["alpha"] == alpha and not r["shield"]]
        on = [r for r in model_rows if r["alpha"] == alpha and r["shield"]]
        off_lift.append(off[0]["mean_lift"] if off else 0)
        off_drift.append(off[0]["mean_drift"] if off else 0)
        on_lift.append(on[0]["mean_lift"] if on else 0)
        on_drift.append(on[0]["mean_drift"] if on else 0)

    W, H = 640, 400
    mx, my = 40, H - 50
    gw, gh = W - 80, H - 120

    # Scale
    all_l = off_lift + on_lift
    all_d = off_drift + on_drift
    l_min, l_max = min(all_l), max(all_l) * 1.15
    d_min, d_max = min(min(all_d), 0), max(max(all_d), 0.6)

    def x(v): return mx + (v - l_min) / max(l_max - l_min, 1) * gw
    def y(v): return my - (v - d_min) / max(d_max - d_min, 1) * gh

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
    svg.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    svg.append(f'<text x="{W/2}" y="22" text-anchor="middle" font-family="system-ui" font-size="14" fill="#1a1a2e">mHC Shield: Counter-Prior Lift vs NLL Drift — Gemma-4-E2B</text>')

    # Axes
    svg.append(f'<line x1="{mx}" y1="{my}" x2="{mx+gw}" y2="{my}" stroke="#ccc" stroke-width="1"/>')
    svg.append(f'<line x1="{mx}" y1="{my}" x2="{mx}" y2="{my-gh}" stroke="#ccc" stroke-width="1"/>')

    # Drift threshold line (H1: 0.5 nats)
    y05 = y(0.5)
    svg.append(f'<line x1="{mx}" y1="{y05}" x2="{mx+gw}" y2="{y05}" stroke="#FF3B30" stroke-width="1" stroke-dasharray="4,4"/>')
    svg.append(f'<text x="{mx+gw+4}" y="{y05+4}" font-family="system-ui" font-size="10" fill="#FF3B30">0.5 nats</text>')

    # Data points + connecting lines (shield OFF = red, shield ON = blue)
    for label, lifts, drifts, color in [
        ("Shield OFF", off_lift, off_drift, "#FF3B30"),
        ("Shield ON", on_lift, on_drift, "#007AFF"),
    ]:
        pts = [(x(l), y(d)) for l, d in zip(lifts, drifts)]
        # Connecting line
        path = " ".join(f"L{px},{py}" for px, py in pts)
        svg.append(f'<path d="M{pts[0][0]},{pts[0][1]} {path}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.5"/>')
        # Points
        for i, (px, py) in enumerate(pts):
            svg.append(f'<circle cx="{px}" cy="{py}" r="4" fill="{color}"/>')
            if i == 0 or i == len(pts) - 1:
                svg.append(f'<text x="{px+6}" y="{py-4}" font-family="system-ui" font-size="9" fill="{color}">α={alphas[i]:.2f}</text>')

    # Legend
    svg.append(f'<circle cx="{mx+10}" cy="{my-gh+12}" r="4" fill="#007AFF"/>')
    svg.append(f'<text x="{mx+20}" y="{my-gh+16}" font-family="system-ui" font-size="11" fill="#333">Shield ON (V2 column cap)</text>')
    svg.append(f'<circle cx="{mx+10}" cy="{my-gh+30}" r="4" fill="#FF3B30"/>')
    svg.append(f'<text x="{mx+20}" y="{my-gh+34}" font-family="system-ui" font-size="11" fill="#333">Shield OFF</text>')

    # Axis labels
    svg.append(f'<text x="{W/2}" y="{H-8}" text-anchor="middle" font-family="system-ui" font-size="11" fill="#666">Counter-Prior Lift (nats)</text>')
    svg.append(f'<text x="14" y="{H/2}" text-anchor="middle" font-family="system-ui" font-size="10" fill="#666" transform="rotate(-90,14,{H/2})">NLL Drift (nats)</text>')

    svg.append('</svg>')
    out_path.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
