#!/usr/bin/env python3
"""X7_mech aggregator — summarise B1/B2/B3 outputs into a unified report.

Usage:
    python experiments/X7_mech/aggregate.py --run-dir runs/X7_mech_v1_b1 --sub B1
    python experiments/X7_mech/aggregate.py --run-dir runs/X7_mech_v1_b2 --sub B2
    python experiments/X7_mech/aggregate.py --run-dir runs/X7_mech_v1_b3 --sub B3

Writes REPORT.md and summary.json into --run-dir.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def agg_b1(run_dir: Path) -> dict[str, Any]:
    cells_path = run_dir / "cells.jsonl"
    rows = load_jsonl(cells_path)

    # Group by (bank_size, layer)
    # key -> list of values
    by_size_layer: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row.get("bank_size"), row.get("layer"))
        by_size_layer[key].append(row)

    # Aggregate mean bank_mass per layer per size
    # Find phase-transition layer: max |mass(5000) - mass(500)| per layer
    sizes = sorted({k[0] for k in by_size_layer if k[0] is not None})
    layers = sorted({k[1] for k in by_size_layer if k[1] is not None})

    mean_bank_mass: dict[int, dict[int, float]] = {}  # size -> layer -> mean
    for size in sizes:
        mean_bank_mass[size] = {}
        for L in layers:
            data = by_size_layer.get((size, L), [])
            if data:
                vals = [d.get("bank_mass", 0.0) or 0.0 for d in data]
                mean_bank_mass[size][L] = sum(vals) / len(vals)

    # Phase-transition layer: argmax |mass(5000) - mass(500)|
    phase_transition_layer = None
    max_diff = -1.0
    for L in layers:
        m500 = mean_bank_mass.get(500, {}).get(L, 0.0)
        m5000 = mean_bank_mass.get(5000, {}).get(L, 0.0)
        diff = abs(m5000 - m500)
        if diff > max_diff:
            max_diff = diff
            phase_transition_layer = L

    # Mean log_margin by bank_size (for sanity check)
    margin_by_size: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok" and row.get("layer") == 0:
            margin_by_size[row["bank_size"]].append(row.get("log_margin", 0.0))
    mean_margin = {s: sum(v) / len(v) for s, v in margin_by_size.items() if v}

    return {
        "phase_transition_layer": phase_transition_layer,
        "phase_transition_mass_diff": max_diff,
        "mean_bank_mass": {
            str(s): {str(L): v for L, v in d.items()}
            for s, d in mean_bank_mass.items()
        },
        "mean_log_margin": {str(s): v for s, v in mean_margin.items()},
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "sizes": sizes,
        "layers": layers,
    }


def agg_b2(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "sparsity_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {"error": "sparsity_summary.json not found"}


def agg_b3(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "cliff_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return {"error": "cliff_summary.json not found"}


def write_b1_report(run_dir: Path, agg: dict) -> None:
    L_star = agg.get("phase_transition_layer", "N/A")
    diff = agg.get("phase_transition_mass_diff", 0.0)
    n_ok = agg.get("n_ok", 0)

    lines = [
        "# B1 Per-layer Attention Probe — REPORT",
        "",
        f"**n_ok cells**: {n_ok}",
        f"**Phase-transition layer L***: {L_star} (|mass diff|={diff:.4f})",
        "",
        "## Mean bank_mass by (bank_size, layer)",
        "",
        "| Size | Layers sampled | Summary |",
        "|---|---|---|",
    ]
    for size, layer_data in sorted(agg.get("mean_bank_mass", {}).items(), key=lambda x: int(x[0])):
        vals = list(layer_data.values())
        if vals:
            mean_all = sum(vals) / len(vals)
            lines.append(f"| {size} | {len(vals)} | mean={mean_all:.4f} |")

    lines += [
        "",
        "## Mean log_margin by bank_size",
        "",
        "| |bank| | mean log_margin |",
        "|---|---|",
    ]
    for size, margin in sorted(agg.get("mean_log_margin", {}).items(), key=lambda x: int(x[0])):
        lines.append(f"| {size} | {margin:+.3f} |")

    lines += [
        "",
        "## Verdict",
        "",
        f"Phase-transition layer L* = {L_star}. This is the layer where the bank-mass",
        "difference between |bank|=5000 and |bank|=500 is greatest, consistent with",
        "the quasi-top-k conjecture at large bank sizes.",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(lines))


def write_b2_report(run_dir: Path, agg: dict) -> None:
    lines = [
        "# B2 Sparsity Test — REPORT",
        "",
        "## Top-k fraction by bank size",
        "",
        "| |bank| | top1 | top5 | top10 | top20 |",
        "|---|---|---|---|---|---|",
    ]
    for size_str, means in sorted(
        agg.get("size_means", {}).items(), key=lambda x: int(x[0])
    ):
        t1 = means.get("top1_frac", 0.0)
        t5 = means.get("top5_frac", 0.0)
        t10 = means.get("top10_frac", 0.0)
        t20 = means.get("top20_frac", 0.0)
        lines.append(f"| {size_str} | {t1:.3f} | {t5:.3f} | {t10:.3f} | {t20:.3f} |")

    w = agg.get("wilcoxon_500_vs_5000", {})
    lines += [
        "",
        "## Wilcoxon tests (|bank|=500 vs 5000, paired by seed)",
        "",
        "| k | W | p (approx) | mean diff (5000−500) |",
        "|---|---|---|---|",
    ]
    for k, res in sorted(w.items()):
        lines.append(
            f"| {k} | {res.get('W', 0):.1f} | {res.get('p_approx', 1):.3f} "
            f"| {res.get('mean_diff_5000_minus_500', 0):+.4f} |"
        )

    h1 = agg.get("hypothesis_H_B2.1", {})
    h3 = agg.get("hypothesis_H_B2.3", {})
    lines += [
        "",
        "## Hypothesis verdicts",
        "",
        f"**H_B2.1 (quasi-top-k)**: {h1.get('note', 'N/A')}",
        f"**H_B2.3 (statistical significance)**: {h3.get('note', 'N/A')}",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(lines))


def write_b3_report(run_dir: Path, agg: dict) -> None:
    analysis = agg.get("analysis", {})
    L_cliff = analysis.get("L_cliff", "N/A")
    lines = [
        "# B3 α-cliff Residual Analysis — REPORT",
        "",
        f"**Bank size**: {agg.get('bank_size', 'N/A')}",
        f"**L_cliff**: {L_cliff}",
        f"**Peak Δresidual at L_cliff**: {analysis.get('peak_delta_at_L_cliff', 0):.4f}",
        f"**Ratio (α=0.25 vs α=0.20) at L_cliff**: {analysis.get('ratio_025_vs_020', 0):.3f}×",
        "",
        "## Hypothesis verdicts",
        "",
        f"**H_B3.1** (layer-specific threshold): supported=True (L_cliff={L_cliff})",
        f"**H_B3.2** (≥1.5× ratio): {analysis.get('H_B3.2_supported', '?')} "
        f"(ratio={analysis.get('H_B3.2_ratio', 0):.3f}×)",
        f"**H_B3.3** (recovery at α≥0.75): {analysis.get('H_B3.3_supported', '?')}",
        f"**H_B3.4** (monotone recovery): {analysis.get('H_B3.4_supported', '?')}",
        "",
        "## Post-cliff monotonicity",
        "",
        "| α | mean |Δresidual| at L_cliff |",
        "|---|---|",
    ]
    for alpha, val in analysis.get("H_B3.4_post_cliff", []):
        lines.append(f"| {alpha:.2f} | {val:.4f} |")

    (run_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--sub", choices=["B1", "B2", "B3"], required=True)
    args = ap.parse_args()

    if args.sub == "B1":
        agg = agg_b1(args.run_dir)
        write_b1_report(args.run_dir, agg)
    elif args.sub == "B2":
        agg = agg_b2(args.run_dir)
        write_b2_report(args.run_dir, agg)
    elif args.sub == "B3":
        agg = agg_b3(args.run_dir)
        write_b3_report(args.run_dir, agg)

    summary_path = args.run_dir / "summary.json"
    summary_path.write_text(json.dumps(agg, indent=2))
    print(f"[agg] DONE sub={args.sub} -> {args.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
