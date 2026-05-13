"""W.5 MoE aggregator.

Inputs
------
``experiments/W5_moe/cells.jsonl`` (or ``cells_smoke.jsonl`` with ``--smoke``).

Outputs
-------
* ``aggregate.csv`` — one row per (alpha, cap_mode, seed) with mean drift +
  bootstrap 95% CI.
* ``pivot.csv`` — cap_mode × alpha pivot of mean drift over seeds.
* ``REPORT.md`` — skeleton report; PASS criterion is per-expert beats global
  by ≥30% mean drift reduction at every α ≥ 1.

Statistical tests
-----------------
* Bootstrap 95% CI per cell aggregate (1000 resamples).
* Paired Wilcoxon (per-expert vs global) at each α, prompts as pairs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "W5_moe"


def _bootstrap_ci(xs: list[float], iters: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    import random

    rng = random.Random(0)
    n = len(xs)
    means = []
    for _ in range(iters):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(iters * alpha / 2)]
    hi = means[int(iters * (1 - alpha / 2))]
    return lo, hi


def _wilcoxon_paired(a: list[float], b: list[float]) -> float:
    """Paired Wilcoxon signed-rank test → two-sided p-value (normal approx)."""
    if len(a) != len(b) or len(a) < 6:
        return float("nan")
    diffs = [x - y for x, y in zip(a, b) if (x - y) != 0.0]
    n = len(diffs)
    if n < 6:
        return float("nan")
    abs_diffs = sorted([(abs(d), 1 if d > 0 else -1) for d in diffs])
    # Rank with average ties.
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_diffs[j + 1][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    w_pos = sum(r for r, (_, s) in zip(ranks, abs_diffs) if s > 0)
    w_neg = sum(r for r, (_, s) in zip(ranks, abs_diffs) if s < 0)
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return float("nan")
    z = (w - mu) / sigma
    # two-sided p ≈ 2 * Φ(-|z|)
    p = math.erfc(abs(z) / math.sqrt(2))
    return p


def load_cells(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def aggregate(rows: list[dict]) -> tuple[list[dict], dict, dict]:
    """Return (per-cell aggregate rows, pivot table, paired-wilcoxon by alpha)."""
    by_key: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        key = (round(r["alpha"], 6), r["cap_mode"], r["seed"])
        by_key[key].append(r["drift"])

    agg = []
    for (alpha, cap_mode, seed), drifts in sorted(by_key.items()):
        mean = statistics.fmean(drifts)
        lo, hi = _bootstrap_ci(drifts)
        agg.append({
            "alpha": alpha,
            "cap_mode": cap_mode,
            "seed": seed,
            "n": len(drifts),
            "mean_drift": mean,
            "ci_lo": lo,
            "ci_hi": hi,
        })

    # Pivot: rows=cap_mode, cols=alpha, value=mean over seeds.
    pivot: dict[str, dict[float, float]] = defaultdict(dict)
    by_pair: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        by_pair[(round(r["alpha"], 6), r["cap_mode"])].append(r["drift"])
    for (alpha, cap_mode), drifts in by_pair.items():
        pivot[cap_mode][alpha] = statistics.fmean(drifts)

    # Paired Wilcoxon: per-expert vs global at each alpha (prompts paired).
    by_pe: dict[float, list[tuple[str, int, float]]] = defaultdict(list)
    by_g: dict[float, list[tuple[str, int, float]]] = defaultdict(list)
    for r in rows:
        if r["cap_mode"] == "per_expert":
            by_pe[round(r["alpha"], 6)].append((r["prompt_id"], r["seed"], r["drift"]))
        elif r["cap_mode"] == "global":
            by_g[round(r["alpha"], 6)].append((r["prompt_id"], r["seed"], r["drift"]))

    wilcox: dict[float, float] = {}
    for alpha in sorted(by_pe.keys() | by_g.keys()):
        # match by (prompt_id, seed)
        pe_idx = {(p, s): d for p, s, d in by_pe.get(alpha, [])}
        g_idx = {(p, s): d for p, s, d in by_g.get(alpha, [])}
        common = sorted(pe_idx.keys() & g_idx.keys())
        if not common:
            continue
        a = [pe_idx[k] for k in common]
        b = [g_idx[k] for k in common]
        wilcox[alpha] = _wilcoxon_paired(a, b)

    return agg, dict(pivot), wilcox


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_pivot(pivot: dict[str, dict[float, float]], path: Path) -> None:
    alphas = sorted({a for d in pivot.values() for a in d})
    cap_modes = sorted(pivot.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cap_mode"] + [str(a) for a in alphas])
        for cm in cap_modes:
            w.writerow([cm] + [f"{pivot[cm].get(a, float('nan')):.5f}" for a in alphas])


def write_report(
    pivot: dict[str, dict[float, float]],
    wilcox: dict[float, float],
    n_cells: int,
    path: Path,
) -> None:
    lines = ["# W.5 MoE per-expert column-cap — REPORT", ""]
    lines.append(f"* Cells: {n_cells}")
    alphas = sorted({a for d in pivot.values() for a in d})
    lines.append("")
    lines.append("## Pivot: cap_mode × α (mean drift)")
    lines.append("")
    lines.append("| cap_mode | " + " | ".join(f"α={a}" for a in alphas) + " |")
    lines.append("|" + "---|" * (1 + len(alphas)))
    for cm in sorted(pivot.keys()):
        row = [cm] + [f"{pivot[cm].get(a, float('nan')):+.4f}" for a in alphas]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Per-expert vs global — paired Wilcoxon p-values")
    lines.append("")
    lines.append("| α | p (per_expert vs global) |")
    lines.append("|---|---|")
    for a in sorted(wilcox.keys()):
        lines.append(f"| {a} | {wilcox[a]:.4g} |")
    lines.append("")
    # PASS verdict.
    lines.append("## PASS criterion (W.5.6)")
    lines.append("")
    lines.append(
        "Per-expert cap reduces mean drift by ≥30% relative to global cap at every α ≥ 1.0."
    )
    lines.append("")
    pe = pivot.get("per_expert", {})
    g = pivot.get("global", {})
    fail = []
    for a in alphas:
        if a < 1.0:
            continue
        if a not in pe or a not in g:
            continue
        if g[a] <= 0:
            continue
        red = 1.0 - pe[a] / g[a]
        lines.append(f"* α={a}: drift_pe={pe[a]:+.4f}, drift_global={g[a]:+.4f}, reduction={red*100:+.1f}%")
        if red < 0.30:
            fail.append(a)
    verdict = "PASS" if not fail else f"FAIL (insufficient reduction at α∈{fail})"
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cells_path = OUT_DIR / ("cells_smoke.jsonl" if args.smoke else "cells.jsonl")
    if not cells_path.exists():
        print(f"[skip] no cells at {cells_path}")
        return 0

    rows = load_cells(cells_path)
    agg, pivot, wilcox = aggregate(rows)
    write_csv(agg, OUT_DIR / ("aggregate_smoke.csv" if args.smoke else "aggregate.csv"))
    write_pivot(pivot, OUT_DIR / ("pivot_smoke.csv" if args.smoke else "pivot.csv"))
    write_report(
        pivot, wilcox, len(rows),
        OUT_DIR / ("REPORT_smoke.md" if args.smoke else "REPORT.md"),
    )
    print(f"[done] aggregated {len(rows)} cells; report at {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
