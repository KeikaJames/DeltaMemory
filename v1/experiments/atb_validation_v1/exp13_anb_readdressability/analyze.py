"""Exp13 analyzer — bootstrap paired diffs + verdict ladder.

Reads ``rows.jsonl`` (and skips ``variant == 'skipped'`` rows), aggregates by
(site, query_kind, variant), computes per-row paired differences of
correct_in_bank vs each control, runs paired bootstrap CIs, and writes
``summary.json``, ``analysis.json``, and ``VERDICT.txt``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

CONTROLS = ("shuffle_layer", "shuffle_V", "random_K_only")
# Controls whose difference from correct_in_bank must be positive to claim
# addressability.  ``shuffle_V`` is excluded: V does not enter QK scoring,
# so its diff is algebraically 0 and would always block a positive verdict.
DECISIVE_CONTROLS = ("shuffle_layer", "random_K_only")


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_paired_diff(a, b, n_resamples=2000, seed=0xCAFE, alpha=0.05):
    """(mean_diff, lo, hi) for paired a - b. Drops pairs with NaN."""
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None
             and not math.isnan(x) and not math.isnan(y)]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    diffs = [x - y for x, y in pairs]
    n = len(diffs)
    md = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_resamples):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int(alpha / 2 * n_resamples)]
    hi = boots[int((1 - alpha / 2) * n_resamples) - 1]
    return md, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    in_dir = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    cells_path = in_dir / "cells.jsonl"
    if not cells_path.exists():
        cells_path = in_dir / "rows.jsonl"  # backward compat
    with cells_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("variant") == "skipped":
                continue
            rows.append(r)

    # Group by (site, query_kind, variant) -> list of dicts.
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["site"], r["query_kind"], r["variant"])
        grouped[key].append(r)

    # Per-cell summary.
    summary: dict[str, dict] = {}
    cells = sorted({(s, q) for (s, q, _) in grouped.keys()})
    for site, qk in cells:
        cell = {}
        for variant in ("correct_in_bank",) + CONTROLS:
            recs = grouped.get((site, qk, variant), [])
            cell[variant] = {
                "n": len(recs),
                "mean_score_gap": mean([r.get("score_gap") for r in recs]),
                "mean_correct_score": mean([r.get("correct_score") for r in recs]),
                "mean_best_other_score": mean([r.get("best_other_score") for r in recs]),
                "mean_recall_at_1": mean([r.get("recall_at_1") for r in recs]),
                "mean_recall_at_5": mean([r.get("recall_at_5") for r in recs]),
                "mean_correct_rank": mean([r.get("correct_rank") for r in recs]),
            }
        # Hard negatives are only on correct_in_bank.
        cor = grouped.get((site, qk, "correct_in_bank"), [])
        cell["hard_negatives"] = {
            "mean_hn_same_subject": mean([r.get("hn_same_subject") for r in cor]),
            "mean_hn_same_relation": mean([r.get("hn_same_relation") for r in cor]),
            "mean_hn_same_object": mean([r.get("hn_same_object") for r in cor]),
        }
        summary[f"{site}/{qk}"] = cell
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Paired bootstrap: correct vs each control on per-(seed, fact_id) pairs.
    analysis: dict[str, dict] = {}
    verdict_signal_per_cell = []
    for site, qk in cells:
        cor_rows = grouped.get((site, qk, "correct_in_bank"), [])
        cor_by_key = {(r["seed"], r["fact_id"]): r for r in cor_rows}
        cell_analysis = {}
        directional_against_all = True
        strong_against_all = True
        for variant in CONTROLS:
            ctl_rows = grouped.get((site, qk, variant), [])
            a, b = [], []
            for r in ctl_rows:
                k = (r["seed"], r["fact_id"])
                if k not in cor_by_key:
                    continue
                a.append(cor_by_key[k].get("score_gap"))
                b.append(r.get("score_gap"))
            md, lo, hi = bootstrap_paired_diff(a, b)
            cell_analysis[f"score_gap_diff_vs_{variant}"] = {
                "n_pairs": len([1 for x, y in zip(a, b)
                                if x is not None and y is not None]),
                "mean_diff": md, "ci_lo": lo, "ci_hi": hi,
            }
            if variant in DECISIVE_CONTROLS:
                if not (isinstance(md, float) and md > 0):
                    directional_against_all = False
                if not (isinstance(lo, float) and lo > 0):
                    strong_against_all = False
        cell_analysis["recall_at_5"] = (
            summary[f"{site}/{qk}"]["correct_in_bank"]["mean_recall_at_5"])
        cell_analysis["directional_against_all_controls"] = directional_against_all
        cell_analysis["strong_against_all_controls"] = strong_against_all
        analysis[f"{site}/{qk}"] = cell_analysis
        verdict_signal_per_cell.append((site, qk, cell_analysis))

    # Verdict for Exp13 alone (per PREREG).
    best_strong = False
    best_dir = False
    best_recall = 0.0
    for site, qk, ca in verdict_signal_per_cell:
        r5 = ca.get("recall_at_5") or 0.0
        if (isinstance(r5, float) and r5 >= 0.50 and
                ca["strong_against_all_controls"]):
            best_strong = True
        if (isinstance(r5, float) and r5 >= 0.30 and
                ca["directional_against_all_controls"]):
            best_dir = True
        if isinstance(r5, float):
            best_recall = max(best_recall, r5)

    if best_strong:
        verdict = "ADDRESSABILITY_STRONG"
    elif best_dir:
        verdict = "ADDRESSABILITY_DIRECTIONAL"
    elif best_recall >= 0.15:
        verdict = "ADDRESSABILITY_WEAK"
    else:
        verdict = "FAIL"
    analysis["_verdict"] = verdict
    analysis["_best_recall_at_5_over_cells"] = best_recall
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (out_dir / "VERDICT.txt").write_text(verdict + "\n")
    print(f"[exp13] verdict = {verdict}; best recall@5 = {best_recall:.3f}")


if __name__ == "__main__":
    main()
