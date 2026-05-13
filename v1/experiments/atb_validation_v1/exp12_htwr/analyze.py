"""Exp12 HTWR analyze — six-way verdict ladder + bootstrap CI.

Reads `results.jsonl` from a Phase α / Tier 0 run dir, computes per-variant
margin statistics with bootstrap 95% CI, top-fact-id retrieval accuracy
breakdown, and routes to a verdict:

- PASS_STRONG       : oracle_correct CI lower > max(controls).mean AND
                      delta_vs_base > 0
- PASS_DIRECTIONAL  : oracle_correct.mean > base AND > all controls,
                      but CI overlap
- RETRIEVAL_ONLY    : oracle_correct > oracle_random/shuffled, but
                      oracle_sign_flip ≥ oracle_correct (sign confused)
- INJECTION_ONLY    : oracle_correct ≈ oracle_shuffled, but both beat base
                      (any injected memory helps)
- STEERING_ONLY     : oracle_random or sign_flip beats oracle_correct
- FAIL_DEEP         : oracle_correct ≤ base
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load(jsonl_path: Path) -> list[dict]:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _bootstrap_ci(values: list[float], B: int = 2000, alpha: float = 0.05,
                  seed: int = 0xB007) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    boots = rng.choice(arr, size=(B, arr.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def summarize(rows: list[dict]) -> dict[str, Any]:
    by_var: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_var[r["variant"]].append(r)
    out: dict[str, Any] = {}
    for var, rs in by_var.items():
        margins = [float(r["margin"]) for r in rs if isinstance(r.get("margin"), (int, float)) and not math.isnan(float(r["margin"]))]
        recalls = [bool(r.get("recall_at_1")) for r in rs]
        retr_acc = [bool(r["htwr_retrieval_correct"]) for r in rs
                    if r.get("htwr_retrieval_correct") is not None]
        top_scores = [float(r["htwr_top_score"]) for r in rs
                      if isinstance(r.get("htwr_top_score"), (int, float))
                      and not math.isnan(float(r["htwr_top_score"]))]
        ci_lo, ci_hi = _bootstrap_ci(margins)
        out[var] = {
            "n": len(rs),
            "n_margin": len(margins),
            "mean_margin": statistics.fmean(margins) if margins else float("nan"),
            "median_margin": statistics.median(margins) if margins else float("nan"),
            "ci95_margin_lo": ci_lo,
            "ci95_margin_hi": ci_hi,
            "mean_recall_at_1": statistics.fmean([1.0 if v else 0.0 for v in recalls]) if recalls else float("nan"),
            "retrieval_accuracy": statistics.fmean([1.0 if v else 0.0 for v in retr_acc]) if retr_acc else None,
            "mean_top_score": statistics.fmean(top_scores) if top_scores else float("nan"),
        }
    return out


def verdict_t0(stats: dict[str, dict]) -> dict[str, Any]:
    """Apply six-way ladder to a T0 summary."""
    def m(v: str) -> float:
        return stats.get(v, {}).get("mean_margin", float("nan"))
    def lo(v: str) -> float:
        return stats.get(v, {}).get("ci95_margin_lo", float("nan"))

    base = m("base_model")
    correct = m("oracle_correct")
    random_m = m("oracle_random")
    shuffled = m("oracle_shuffled")
    sign_flip = m("oracle_sign_flip")
    correct_lo = lo("oracle_correct")
    controls = [x for x in (random_m, shuffled, sign_flip) if not math.isnan(x)]
    strongest_control = max(controls) if controls else float("-inf")

    # FAIL_DEEP: oracle can't even beat base.
    if not math.isnan(base) and not math.isnan(correct) and correct <= base:
        verdict = "FAIL_DEEP"
    # STEERING_ONLY: a control matches or beats correct.
    elif controls and strongest_control >= correct:
        verdict = "STEERING_ONLY"
    # RETRIEVAL_ONLY: random/shuffled lose, but sign_flip wins (sign confused).
    elif (not math.isnan(sign_flip) and sign_flip >= correct
          and (math.isnan(random_m) or correct > random_m)):
        verdict = "RETRIEVAL_ONLY"
    # INJECTION_ONLY: shuffled ~ correct (both well above base), random worse.
    elif (not math.isnan(shuffled) and abs(correct - shuffled) < 0.1
          and not math.isnan(base) and (correct - base) > 0.5):
        verdict = "INJECTION_ONLY"
    elif (not math.isnan(correct_lo) and correct_lo > strongest_control
          and (math.isnan(base) or correct > base)):
        verdict = "PASS_STRONG"
    elif (correct > strongest_control and (math.isnan(base) or correct > base)):
        verdict = "PASS_DIRECTIONAL"
    else:
        verdict = "FAIL_DEEP"

    delta_vs_base = (correct - base) if not math.isnan(base) else float("nan")
    gap_vs_strongest = (correct - strongest_control) if controls else float("nan")
    return {
        "verdict": verdict,
        "base_margin": base,
        "oracle_correct_margin": correct,
        "oracle_correct_ci_lo": correct_lo,
        "oracle_random_margin": random_m,
        "oracle_shuffled_margin": shuffled,
        "oracle_sign_flip_margin": sign_flip,
        "delta_vs_base": delta_vs_base,
        "gap_vs_strongest_control": gap_vs_strongest,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    results = run_dir / "results.jsonl"
    if not results.exists():
        # Walk one level (eta subdirs).
        sub = sorted([d for d in run_dir.iterdir() if d.is_dir() and (d / "results.jsonl").exists()])
        all_v = {}
        for d in sub:
            rows = _load(d / "results.jsonl")
            stats = summarize(rows)
            vd = verdict_t0(stats)
            all_v[d.name] = {"stats": stats, "verdict": vd}
            (d / "analysis.json").write_text(json.dumps({"stats": stats, "verdict": vd}, indent=2))
            (d / "verdict.txt").write_text(vd["verdict"] + "\n")
        (run_dir / "phase_alpha_analysis.json").write_text(json.dumps(all_v, indent=2))
        # Pick best (highest correct margin) for top-level verdict.
        best_key = max(all_v, key=lambda k: all_v[k]["verdict"].get("oracle_correct_margin", -1e9))
        (run_dir / "verdict.txt").write_text(all_v[best_key]["verdict"]["verdict"] + "\n")
        print(json.dumps({k: v["verdict"] for k, v in all_v.items()}, indent=2))
        return
    rows = _load(results)
    stats = summarize(rows)
    vd = verdict_t0(stats)
    (run_dir / "analysis.json").write_text(json.dumps({"stats": stats, "verdict": vd}, indent=2))
    (run_dir / "verdict.txt").write_text(vd["verdict"] + "\n")
    print(json.dumps(vd, indent=2))


if __name__ == "__main__":
    main()
