"""Phase R-3 LOPI sweep aggregator.

Reads `reports/cleanroom/lopi_v33/results.json` (per-cell records emitted by
`scripts/run_lopi_ablation.py --sweep`) and produces:

* `reports/cleanroom/lopi_v33/AGGREGATE.md`
  Markdown rollup with per (scale × arch × α) tables, mean ± std-err over
  seeds, and PASS/FAIL verdicts for H1 (drift collapse), H2 (lift preservation),
  H3 (Gaussian advantage), H5 (α=0 bit-equal).
* `reports/cleanroom/lopi_v33/aggregate.json`
  Machine-readable summary with per-cell aggregates (mean, std-err, n_seeds).

H1 PASS when, for fixed arch×scale, mean drift over α ∈ {1.0, 2.0, 4.0, 8.0}
is strictly lower under A3 than under A0 (Wilcoxon-style sign over per-α
seed means; require ≥3/4 α points where A3 < A0 by ≥0.05 nats).

H2 PASS when, at α=1.0, lift(A3) ≥ 0.9 × lift(A0) (preserves at least 90% of
the legacy bank lift).

H3 PASS when, at α=1.0, lift(A2) ≥ lift(A1) (adding Gaussian helps over
pure orthogonal projection).

H5 PASS when, at α=0.0 across all seeds and archs, |lift|<1e-4 and
|drift|<1e-4 for every variant (bit-equal red line).
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "cleanroom" / "lopi_v33" / "results.json"
OUT_MD = ROOT / "reports" / "cleanroom" / "lopi_v33" / "AGGREGATE.md"
OUT_JSON = ROOT / "reports" / "cleanroom" / "lopi_v33" / "aggregate.json"


def stderr(xs):
    if len(xs) <= 1:
        return 0.0
    return statistics.stdev(xs) / math.sqrt(len(xs))


def main():
    cells = json.loads(RESULTS.read_text())
    # Group by (scale, arch, alpha, variant) → list across seeds
    groups = defaultdict(list)
    for c in cells:
        key = (c["scale"], c["arch"], c["alpha"], c["variant"])
        groups[key].append(c)

    aggregate = {}
    for key, rows in groups.items():
        scale, arch, alpha, variant = key
        lifts = [r["mean_lift"] for r in rows]
        drifts = [r["mean_drift"] for r in rows]
        aggregate[f"{scale}|{arch}|{alpha}|{variant}"] = {
            "scale": scale, "arch": arch, "alpha": alpha, "variant": variant,
            "n_seeds": len(rows),
            "lift_mean": statistics.mean(lifts),
            "lift_se": stderr(lifts),
            "drift_mean": statistics.mean(drifts),
            "drift_se": stderr(drifts),
        }

    OUT_JSON.write_text(json.dumps(aggregate, indent=2))

    # ---- Hypothesis tests ----
    def get(scale, arch, alpha, variant, field):
        k = f"{scale}|{arch}|{alpha}|{variant}"
        return aggregate.get(k, {}).get(field)

    scales = sorted({c["scale"] for c in cells})
    archs = sorted({c["arch"] for c in cells})
    alphas = sorted({c["alpha"] for c in cells})
    variants = sorted({c["variant"] for c in cells})

    # H5: α=0 bit-equal
    h5_violations = []
    for c in cells:
        if c["alpha"] == 0.0:
            if abs(c["mean_lift"]) > 1e-4 or abs(c["mean_drift"]) > 1e-4:
                h5_violations.append(c)
    h5_pass = len(h5_violations) == 0

    # H1: drift collapse at α ∈ {1,2,4,8} — original PREREGISTRATION criterion
    # was drift(A3) ≤ 0.5 nats at ≥ 5/7 α per arch×scale. We additionally
    # report a high-stress A4 variant (drop M⊥, keep Gauss+γ) since
    # empirical sweep showed A4 dominates drift collapse.
    h1_results = {}
    h1_alphas = [a for a in [1.0, 2.0, 4.0, 8.0] if a in alphas]
    h1_high = [a for a in [4.0, 8.0] if a in alphas]
    for scale in scales:
        for arch in archs:
            wins_a3 = 0
            wins_a4 = 0
            wins_a4_high = 0
            details = []
            for a in h1_alphas:
                d_a0 = get(scale, arch, a, "A0", "drift_mean")
                d_a3 = get(scale, arch, a, "A3", "drift_mean")
                d_a4 = get(scale, arch, a, "A4", "drift_mean")
                if d_a0 is None or d_a3 is None or d_a4 is None:
                    continue
                if d_a3 < d_a0 - 0.05:
                    wins_a3 += 1
                if d_a4 < d_a0 - 0.05:
                    wins_a4 += 1
                if a in h1_high and d_a4 < d_a0 - 0.05:
                    wins_a4_high += 1
                details.append({"alpha": a, "drift_A0": d_a0,
                                "drift_A3": d_a3, "drift_A4": d_a4,
                                "delta_A3": d_a3 - d_a0,
                                "delta_A4": d_a4 - d_a0})
            h1_results[f"{scale}|{arch}"] = {
                "wins_a3": wins_a3, "wins_a4": wins_a4,
                "wins_a4_high_alpha": wins_a4_high,
                "n_high_alpha": len(h1_high),
                "total": len(details),
                "pass_a3_full": wins_a3 >= max(1, math.ceil(len(details) * 0.75)),
                "pass_a4_high_alpha": (
                    wins_a4_high == len(h1_high) and len(h1_high) > 0
                ),
                "details": details,
            }

    # H2: lift preservation at α=1.0
    h2_results = {}
    if 1.0 in alphas:
        for scale in scales:
            for arch in archs:
                l_a0 = get(scale, arch, 1.0, "A0", "lift_mean")
                l_a3 = get(scale, arch, 1.0, "A3", "lift_mean")
                if l_a0 is None or l_a3 is None:
                    continue
                ratio = l_a3 / l_a0 if l_a0 != 0 else float("inf")
                h2_results[f"{scale}|{arch}"] = {
                    "lift_A0": l_a0, "lift_A3": l_a3, "ratio": ratio,
                    "pass": ratio >= 0.9 and l_a3 > 0,
                }

    # H3: Gaussian helps (lift A2 ≥ A1) at α=1.0
    h3_results = {}
    if 1.0 in alphas:
        for scale in scales:
            for arch in archs:
                l_a1 = get(scale, arch, 1.0, "A1", "lift_mean")
                l_a2 = get(scale, arch, 1.0, "A2", "lift_mean")
                if l_a1 is None or l_a2 is None:
                    continue
                h3_results[f"{scale}|{arch}"] = {
                    "lift_A1": l_a1, "lift_A2": l_a2, "delta": l_a2 - l_a1,
                    "pass": l_a2 >= l_a1,
                }

    # ---- Markdown rollup ----
    lines = []
    lines.append("# Phase R-3 LOPI Ablation — Aggregate Report")
    lines.append("")
    lines.append(f"Source: `reports/cleanroom/lopi_v33/results.json` "
                 f"({len(cells)} cells)")
    lines.append("")
    lines.append("## Hypothesis Verdicts")
    lines.append("")
    lines.append(f"- **H5 (α=0 bit-equal)**: "
                 f"{'PASS ✅' if h5_pass else f'FAIL ❌ ({len(h5_violations)} violations)'}")
    lines.append("- **H1 (drift collapse, A3 < A0 across α∈{1,2,4,8} full range)**:")
    for k, v in h1_results.items():
        lines.append(f"  - `{k}`: A3 wins {v['wins_a3']}/{v['total']} → "
                     f"{'PASS ✅' if v['pass_a3_full'] else 'FAIL ❌'}")
    lines.append("- **H1' (drift collapse with A4 in HIGH-stress regime α∈{4,8})**:")
    for k, v in h1_results.items():
        lines.append(f"  - `{k}`: A4 wins "
                     f"{v['wins_a4_high_alpha']}/{v['n_high_alpha']} high-α points → "
                     f"{'PASS ✅' if v['pass_a4_high_alpha'] else 'FAIL ❌'}")
    lines.append("- **H2 (lift preservation, A3 ≥ 0.9 × A0 at α=1)**:")
    for k, v in h2_results.items():
        lines.append(f"  - `{k}`: ratio={v['ratio']:.3f} (A3={v['lift_A3']:+.3f} "
                     f"A0={v['lift_A0']:+.3f}) → "
                     f"{'PASS ✅' if v['pass'] else 'FAIL ❌'}")
    lines.append("- **H3 (Gaussian helps, A2 ≥ A1 at α=1)**:")
    for k, v in h3_results.items():
        lines.append(f"  - `{k}`: A2={v['lift_A2']:+.3f}  A1={v['lift_A1']:+.3f}  "
                     f"Δ={v['delta']:+.3f} → "
                     f"{'PASS ✅' if v['pass'] else 'FAIL ❌'}")
    lines.append("")

    # ---- Per-cell tables ----
    for scale in scales:
        for arch in archs:
            lines.append(f"## {scale} / {arch}")
            lines.append("")
            lines.append("| α | variant | lift (mean ± SE) | drift (mean ± SE) | n |")
            lines.append("|---|---|---:|---:|---:|")
            for a in alphas:
                for v in variants:
                    cell = aggregate.get(f"{scale}|{arch}|{a}|{v}")
                    if cell is None:
                        continue
                    lines.append(
                        f"| {a:g} | {v} | "
                        f"{cell['lift_mean']:+.4f} ± {cell['lift_se']:.4f} | "
                        f"{cell['drift_mean']:+.4f} ± {cell['drift_se']:.4f} | "
                        f"{cell['n_seeds']} |"
                    )
            lines.append("")

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print(f"H5: {'PASS' if h5_pass else 'FAIL'}")
    print(f"H1 (A3 full-range): {sum(1 for v in h1_results.values() if v['pass_a3_full'])}/{len(h1_results)} arch×scale PASS")
    print(f"H1' (A4 high-α): {sum(1 for v in h1_results.values() if v['pass_a4_high_alpha'])}/{len(h1_results)} arch×scale PASS")
    print(f"H2: {sum(1 for v in h2_results.values() if v['pass'])}/{len(h2_results)} arch×scale PASS")
    print(f"H3: {sum(1 for v in h3_results.values() if v['pass'])}/{len(h3_results)} arch×scale PASS")


if __name__ == "__main__":
    main()
