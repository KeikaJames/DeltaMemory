"""Exp14 analyzer — oracle KV injection causality verdict.

Aggregates cells.jsonl by (alpha, variant), computes paired bootstrap CIs for
the contrast `margin(oracle_correct_KV) - margin(best_control)`, and emits a
verdict per the Exp13 PREREG ladder.

Verdicts (this stage only):
  * ORACLE_PASS:        ∃α such that mean(correct) > all controls AND
                        paired bootstrap 95% CI lower bound > 0.
  * ORACLE_DIRECTIONAL: ∃α such that mean(correct) > all controls but CI crosses 0.
  * ORACLE_FAIL:        For every α, mean(correct) ≤ max(controls).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

CONTROLS = ("oracle_random_KV", "oracle_shuffled_layer_KV", "oracle_KcorrectVrandom")
TARGET = "oracle_correct_KV"
B_BOOT = 2000


def bootstrap_ci(deltas, B=B_BOOT, alpha=0.05, seed=12345):
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = []
    for _ in range(B):
        s = sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int(alpha / 2 * B)]
    hi = means[int((1 - alpha / 2) * B) - 1]
    return (statistics.fmean(deltas), lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    rows_path = in_dir / "cells.jsonl"
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines() if ln.strip()]

    # Index: (seed, fact_id, alpha, variant) -> margin
    idx = {}
    base_idx = {}  # (seed, fact_id) -> base margin
    for r in rows:
        if r.get("variant") == "skipped" or "margin" not in r:
            continue
        key = (r["seed"], r["fact_id"], float(r["alpha"]), r["variant"])
        idx[key] = float(r["margin"])
        if r["variant"] == "base":
            base_idx[(r["seed"], r["fact_id"])] = float(r["margin"])

    alphas = sorted({float(r["alpha"]) for r in rows if r.get("variant") not in (None, "skipped", "base")})
    seeds = sorted({r["seed"] for r in rows})
    fids = sorted({r["fact_id"] for r in rows})

    per_alpha = {}
    overall_pass = False
    overall_directional = False
    for a in alphas:
        # Per-variant means.
        per_var = {}
        for v in (TARGET,) + CONTROLS:
            ms = [idx[(s, f, a, v)] for s in seeds for f in fids
                  if (s, f, a, v) in idx]
            per_var[v] = ms
        # Paired deltas vs each control; we report `correct - max(controls)`
        # per row (the worst-case contrast).
        paired_pts = []
        for s in seeds:
            for f in fids:
                if (s, f, a, TARGET) not in idx:
                    continue
                m_target = idx[(s, f, a, TARGET)]
                ctrl_vals = []
                for v in CONTROLS:
                    if (s, f, a, v) in idx:
                        ctrl_vals.append(idx[(s, f, a, v)])
                if not ctrl_vals:
                    continue
                paired_pts.append(m_target - max(ctrl_vals))
        mean_d, lo, hi = bootstrap_ci(paired_pts)

        # Per-control individual contrast for diagnostics.
        per_control_ci = {}
        for v in CONTROLS:
            d = []
            for s in seeds:
                for f in fids:
                    if (s, f, a, TARGET) in idx and (s, f, a, v) in idx:
                        d.append(idx[(s, f, a, TARGET)] - idx[(s, f, a, v)])
            per_control_ci[v] = bootstrap_ci(d)

        # delta vs base
        d_base = []
        for s in seeds:
            for f in fids:
                if (s, f, a, TARGET) in idx and (s, f) in base_idx:
                    d_base.append(idx[(s, f, a, TARGET)] - base_idx[(s, f)])
        base_ci = bootstrap_ci(d_base)

        mean_var = {v: (statistics.fmean(ms) if ms else float("nan")) for v, ms in per_var.items()}
        beats_all = mean_var[TARGET] > max(mean_var[v] for v in CONTROLS)
        ci_lo_pos = lo > 0
        if beats_all and ci_lo_pos:
            overall_pass = True
        elif beats_all:
            overall_directional = True

        per_alpha[str(a)] = {
            "mean_margin": mean_var,
            "n_pairs": len(paired_pts),
            "delta_vs_worst_control": {"mean": mean_d, "ci_lo": lo, "ci_hi": hi},
            "delta_vs_each_control": {
                v: {"mean": m, "ci_lo": l, "ci_hi": h}
                for v, (m, l, h) in per_control_ci.items()
            },
            "delta_vs_base": {"mean": base_ci[0], "ci_lo": base_ci[1], "ci_hi": base_ci[2]},
            "beats_all_controls": beats_all,
            "ci_lo_above_zero": ci_lo_pos,
        }

    if overall_pass:
        verdict = "ORACLE_PASS"
    elif overall_directional:
        verdict = "ORACLE_DIRECTIONAL"
    else:
        verdict = "ORACLE_FAIL"

    analysis = {
        "experiment": "exp14_oracle_addressed",
        "n_seeds": len(seeds),
        "n_facts": len(fids),
        "alphas": alphas,
        "per_alpha": per_alpha,
        "verdict": verdict,
        "notes": (
            "Decisive controls = oracle_random_KV, oracle_shuffled_layer_KV, "
            "oracle_KcorrectVrandom. Paired bootstrap B=2000 on (correct - "
            "max(controls)) deltas. PREREG: if every alpha yields "
            "correct <= max(controls), Exp15/Exp18 are skipped."
        ),
    }
    (in_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (in_dir / "VERDICT.txt").write_text(verdict + "\n")
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
