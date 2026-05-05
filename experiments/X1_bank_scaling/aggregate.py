#!/usr/bin/env python3
"""X.1 aggregate: dilution curves per arm; H_X1.* verdicts."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.cells.read_text().splitlines() if l]
    by = defaultdict(list)  # (arm, alpha, N) -> [log_margin]
    for r in rows:
        if r.get("status") != "ok":
            continue
        by[(r["arm"], r["alpha"], r["N"])].append(r["log_margin"])

    curves = {}  # arm -> {N: {mean, ci, n}}
    for (arm, alpha, N), margins in by.items():
        if alpha != 1.0:
            continue
        d = curves.setdefault(arm, {})
        d[N] = {
            "mean": statistics.mean(margins),
            "stdev": statistics.pstdev(margins) if len(margins) > 1 else 0.0,
            "n": len(margins),
            "samples": margins,
        }

    # H_X1.1: monotone decay in `none` arm, ≥2× drop by N=100 vs N=1.
    h1 = {"verdict": "n/a"}
    if "none" in curves:
        c = curves["none"]
        if 1 in c and 100 in c:
            ratio = c[1]["mean"] / max(c[100]["mean"], 1e-9)
            h1 = {"verdict": "supported" if ratio >= 2.0 else "not_supported",
                  "n1_mean": c[1]["mean"], "n100_mean": c[100]["mean"],
                  "ratio_n1_over_n100": ratio}

    # H_X1.2: bank_topk=4 keeps log_margin within 20% of N=1 across all N.
    h2 = {"verdict": "n/a"}
    if "topk_4" in curves:
        c = curves["topk_4"]
        ref = c.get(1, {}).get("mean")
        if ref is not None:
            devs = [(N, abs(c[N]["mean"] - ref) / max(abs(ref), 1e-9))
                    for N in sorted(c)]
            max_dev = max(d for _, d in devs)
            h2 = {"verdict": "supported" if max_dev <= 0.20 else "not_supported",
                  "max_rel_dev": max_dev, "ref_n1_mean": ref, "devs": devs}

    # H_X1.3: bank_separate_softmax keeps log_margin within 10% across all N.
    h3 = {"verdict": "n/a"}
    if "separate_softmax" in curves:
        c = curves["separate_softmax"]
        ref = c.get(1, {}).get("mean")
        if ref is not None:
            devs = [(N, abs(c[N]["mean"] - ref) / max(abs(ref), 1e-9))
                    for N in sorted(c)]
            max_dev = max(d for _, d in devs)
            h3 = {"verdict": "supported" if max_dev <= 0.10 else "not_supported",
                  "max_rel_dev": max_dev, "ref_n1_mean": ref, "devs": devs}

    # Latency by N (all alphas, all arms).
    lat_by_N = defaultdict(list)
    for r in rows:
        if r.get("status") == "ok" and "read_latency_ms" in r:
            lat_by_N[r["N"]].append(r["read_latency_ms"])
    lat_summary = {N: {"median_ms": statistics.median(v),
                       "n_samples": len(v)} for N, v in lat_by_N.items()}

    out = {
        "n_cells": len(rows),
        "curves": {k: {N: {kk: vv for kk, vv in d.items() if kk != "samples"}
                       for N, d in v.items()}
                   for k, v in curves.items()},
        "H_X1_1_dilution_exists": h1,
        "H_X1_2_topk_restores": h2,
        "H_X1_3_separate_softmax_restores": h3,
        "latency_by_N_ms": lat_summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[X1][aggregate] -> {args.out}")
    print(f"  H_X1.1 dilution: {h1.get('verdict')} "
          f"(ratio={h1.get('ratio_n1_over_n100', 'n/a')})")
    print(f"  H_X1.2 top-k:   {h2.get('verdict')} "
          f"(max_rel_dev={h2.get('max_rel_dev', 'n/a')})")
    print(f"  H_X1.3 sep-sm:  {h3.get('verdict')} "
          f"(max_rel_dev={h3.get('max_rel_dev', 'n/a')})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
