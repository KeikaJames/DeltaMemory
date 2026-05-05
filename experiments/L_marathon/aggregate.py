"""L Marathon aggregator — summary statistics and H_L verdict.

Reads ``cells.jsonl`` produced by ``run.py`` and emits:
- ``summary.json`` — H_L paired Wilcoxon verdicts per model.
- ``flat_table.csv`` — (model, seed, turn, recall, residual_norm, mem_rss) for plotting.

Statistics
----------
* H_L: paired Wilcoxon signed-rank, two-sided, ``zero_method='wilcox'``,
  on ``nll_target_new(turn=2000) - nll_target_new(turn=1)`` paired by seed.
  Per model, n=3 (small-N caveat documented).
* Effect size: median paired diff with 95% bootstrap CI (B=1000, seed=0).

Author: BIRI GA, 2026-05-10.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Loading

def load_cells(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_non_aborted(cells: list[dict]) -> list[dict]:
    return [c for c in cells if c.get("abort_reason") is None]


# ---------------------------------------------------------------------------
# Wilcoxon + bootstrap

def _wilcoxon_two_sided(diffs: list[float]) -> tuple[float, float]:
    """Return (statistic, p_value) for a paired Wilcoxon signed-rank test
    with ``zero_method='wilcox'`` (drop zero pairs)."""
    if not diffs:
        return float("nan"), float("nan")
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return float("nan"), float("nan")
    nz = [d for d in diffs if d != 0.0]
    if len(nz) < 1:
        return float("nan"), 1.0
    try:
        stat, p = wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def _bootstrap_median_ci(
    diffs: list[float],
    B: int = 1000,
    seed: int = 0,
    conf: float = 0.95,
) -> tuple[float, float, float]:
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    medians = []
    n = len(arr)
    for _ in range(B):
        sample = arr[rng.integers(0, n, size=n)]
        medians.append(float(np.median(sample)))
    lo = float(np.percentile(medians, (1 - conf) / 2 * 100))
    hi = float(np.percentile(medians, (1 + conf) / 2 * 100))
    return float(np.median(arr)), lo, hi


# ---------------------------------------------------------------------------
# H_L per model

def aggregate_h_l_per_model(cells: list[dict], model: str, method: str) -> dict:
    """Paired test: nll_target_new(turn=2000) - nll_target_new(turn=1) per seed."""
    # Extract turn=1 and turn=2000 rows for this (model, method)
    turn1_map: dict[int, float] = {}  # seed -> nll_target_new
    turn2k_map: dict[int, float] = {}
    
    for c in cells:
        if c["model"] != model or c["method"] != method:
            continue
        seed = int(c["seed"])
        turn = int(c["turn"])
        nll = c.get("nll_target_new")
        if nll is None or nll != nll:  # skip NaN
            continue
        if turn == 1:
            turn1_map[seed] = float(nll)
        elif turn == 2000:
            turn2k_map[seed] = float(nll)
    
    # Pair by seed
    diffs = []
    for seed in turn1_map:
        if seed in turn2k_map:
            diffs.append(turn2k_map[seed] - turn1_map[seed])
    
    stat, p = _wilcoxon_two_sided(diffs)
    med, lo, hi = _bootstrap_median_ci(diffs)
    
    # H_L: recall(turn=2000) >= 0.5 * recall(turn=1)
    # i.e., nll_target_new(turn=2000) <= 2 * nll_target_new(turn=1)
    # directional success: median_diff < nll_turn1  (i.e., not 2x worse)
    # For simplicity, we just report the paired diff and p-value.
    # The actual H_L condition is that the median diff is not too large.
    # We'll flag success if p < 0.05 and median_diff < threshold.
    
    # Threshold: turn1_median * 1.0 (i.e., nll didn't double on average)
    turn1_vals = [turn1_map[s] for s in turn1_map if s in turn2k_map]
    turn1_median = float(np.median(turn1_vals)) if turn1_vals else float("nan")
    
    h_l_pass = (
        p == p and p < 0.05 and
        med == med and turn1_median == turn1_median and
        med < turn1_median  # nll didn't increase by more than turn1 median
    )
    
    return {
        "model": model,
        "method": method,
        "n_pairs": len(diffs),
        "wilcoxon_stat": stat,
        "p_value": p,
        "median_diff": med,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "turn1_median": turn1_median,
        "h_l_pass": bool(h_l_pass),
    }


# ---------------------------------------------------------------------------
# Flat table for plotting

def build_flat_table(cells: list[dict]) -> list[dict]:
    """Return (model, seed, turn, nll_target_new, residual_norm_mu, mem_rss_mb)."""
    table = []
    for c in cells:
        table.append({
            "model": c["model"],
            "method": c["method"],
            "seed": c["seed"],
            "turn": c["turn"],
            "nll_target_new": c.get("nll_target_new"),
            "residual_norm_mu": c.get("residual_norm_mu"),
            "mem_rss_mb": c.get("mem_rss_mb"),
        })
    return table


# ---------------------------------------------------------------------------
# Main

def aggregate(cells: list[dict]) -> tuple[dict, list[dict]]:
    real = filter_non_aborted(cells)
    
    models = sorted(set(c["model"] for c in real))
    methods = sorted(set(c["method"] for c in real))
    
    h_l_results: list[dict] = []
    for model in models:
        for method in methods:
            h_l = aggregate_h_l_per_model(real, model, method)
            h_l_results.append(h_l)
    
    abort_count = sum(1 for c in cells if c.get("abort_reason") is not None)
    
    summary = {
        "n_cells_total": len(cells),
        "n_cells_non_aborted": len(real),
        "n_cells_aborted": abort_count,
        "models": models,
        "methods": methods,
        "h_l_results": h_l_results,
        "n_h_l_pass": sum(1 for h in h_l_results if h["h_l_pass"]),
    }
    
    flat_table = build_flat_table(real)
    return summary, flat_table


def main() -> None:
    ap = argparse.ArgumentParser(description="L Marathon aggregator")
    ap.add_argument("--cells", default="experiments/L_marathon/cells.jsonl")
    ap.add_argument("--out", default="experiments/L_marathon/")
    args = ap.parse_args()
    
    cells_path = Path(args.cells)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not cells_path.exists():
        print(f"[agg] cells file not found: {cells_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[agg] loading {cells_path}", flush=True)
    cells = load_cells(cells_path)
    print(f"[agg] {len(cells)} cells loaded", flush=True)
    
    summary, flat_table = aggregate(cells)
    
    sum_path = out_dir / "summary.json"
    table_path = out_dir / "flat_table.csv"
    
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    
    with open(table_path, "w", newline="") as f:
        if flat_table:
            writer = csv.DictWriter(f, fieldnames=flat_table[0].keys())
            writer.writeheader()
            writer.writerows(flat_table)
    
    print(f"[agg] -> {sum_path}", flush=True)
    print(f"[agg] -> {table_path}", flush=True)
    print(f"[agg] H_L: {summary['n_h_l_pass']}/{len(summary['h_l_results'])} "
          f"(model, method) pairs pass (p<0.05, recall stable)")


if __name__ == "__main__":
    main()
