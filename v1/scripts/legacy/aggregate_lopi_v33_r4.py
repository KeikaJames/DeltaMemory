"""Phase R-4 cross-arch LOPI sweep aggregator.

Reads per-model JSONs from ``reports/cleanroom/lopi_v33/R4_xarch/`` and
emits a paired comparison table: for each (model, alpha) we report
shield-off / shield-only / lopi-only / shield+lopi cells with both
counter-prior lift and neutral-prompt drift (mean per-token NLL).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_cells(out_dir: Path):
    cells = []
    for p in sorted(out_dir.glob("*.json")):
        if p.name in {"AGGREGATE.json", "meta.json"}:
            continue
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            cells.extend(data)
    return cells


def fmt(x):
    return f"{x:+.3f}" if x is not None else "  —  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir",
                    default="reports/cleanroom/lopi_v33/R4_xarch")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    cells = load_cells(in_dir)

    # group by (model, alpha)
    by_key = {}
    models = []
    for c in cells:
        m = c["model"]
        if m not in models:
            models.append(m)
        key = (m, c["alpha"])
        by_key.setdefault(key, {})[(c["shield"], c["lopi"])] = c

    lines = ["# R-4 cross-arch LOPI sweep — paired aggregate",
             "",
             "Each row = one (model, α) cell.  Columns = (shield, lopi) configurations.",
             "Lift = mean log-prob lift on FALSE_FACTS (higher = bank pushes counter-prior);",
             "drift = mean per-token NLL change on NEUTRAL_PROMPTS (lower magnitude = safer).",
             ""]

    for m in models:
        lines.append(f"## {m}")
        lines.append("")
        lines.append("| α | sh=OFF lopi=OFF lift / drift | sh=OFF lopi=ON lift / drift "
                     "| sh=ON  lopi=OFF lift / drift | sh=ON  lopi=ON lift / drift |")
        lines.append("|---:|---:|---:|---:|---:|")
        alphas = sorted({a for (mm, a) in by_key if mm == m})
        for a in alphas:
            row = by_key.get((m, a), {})
            cells4 = []
            for sh, lp in [(False, False), (False, True), (True, False), (True, True)]:
                c = row.get((sh, lp))
                if c is None:
                    cells4.append("—")
                else:
                    cells4.append(f"{fmt(c['mean_lift'])} / {fmt(c['nll_drift'])}")
            lines.append(f"| {a} | " + " | ".join(cells4) + " |")
        lines.append("")

        # paired drift comparison: lopi-on vs lopi-off at each (shield, alpha)
        lines.append(f"### {m} — drift ratio LOPI/no-LOPI (paired by shield, α)")
        lines.append("")
        lines.append("| α | shield=OFF | shield=ON |")
        lines.append("|---:|---:|---:|")
        for a in alphas:
            if a == 0.0:
                continue
            row = by_key.get((m, a), {})
            ratios = []
            for sh in (False, True):
                off = row.get((sh, False))
                on = row.get((sh, True))
                if off and on and abs(off["nll_drift"]) > 1e-6:
                    r = on["nll_drift"] / off["nll_drift"]
                    ratios.append(f"{r:+.2f}")
                else:
                    ratios.append("—")
            lines.append(f"| {a} | " + " | ".join(ratios) + " |")
        lines.append("")

    # L2 verdict per model: cell-wise paired drift reduction
    lines.append("## L2 verdict — per-model drift reduction (LOPI ON vs OFF, paired)")
    lines.append("")
    lines.append("| model | n_pairs | n_LOPI_reduces_drift | mean_drift_LOPI_OFF | "
                 "mean_drift_LOPI_ON | abs_reduction_pp | L2_strict (≤0.5 nats LOPI ON) |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    for m in models:
        pairs = []
        for (mm, a), row in by_key.items():
            if mm != m or a == 0.0:
                continue
            for sh in (False, True):
                off = row.get((sh, False))
                on = row.get((sh, True))
                if off and on:
                    pairs.append((off["nll_drift"], on["nll_drift"]))
        n = len(pairs)
        if n == 0:
            continue
        n_red = sum(1 for o, on in pairs if abs(on) < abs(o))
        mean_off = sum(abs(o) for o, _ in pairs) / n
        mean_on = sum(abs(on) for _, on in pairs) / n
        red_pp = mean_off - mean_on
        all_lopi_on_below_half = all(abs(on) <= 0.5 for _, on in pairs)
        verdict = "✅" if all_lopi_on_below_half else "❌"
        lines.append(f"| {m} | {n} | {n_red} | {mean_off:.3f} | "
                     f"{mean_on:.3f} | {red_pp:+.3f} | {verdict} |")
    lines.append("")

    out_md = in_dir / "AGGREGATE.md"
    out_md.write_text("\n".join(lines))
    out_json = in_dir / "AGGREGATE.json"
    out_json.write_text(json.dumps(cells, indent=2))
    print(f"[r4-agg] {len(cells)} cells -> {out_md} + {out_json}")


if __name__ == "__main__":
    main()
