#!/usr/bin/env python3
"""Aggregate W.6 CounterFact cells → recall/drift table for the paper.

Reports per (method, alpha, seed) and per (method, alpha) over seeds:
  - n_real_cells (excludes sentinel-dropped rows)
  - mean nll_new, nll_true
  - margin = nll_true - nll_new   (positive ⇒ model prefers target_new)
  - recall@1 proxy = fraction with margin > 0
  - mean kl_unrel    (drift on unrelated windows)

Also prints a summary block ready to drop into Section 8 of the paper.

Usage:
  python scripts/aggregate_w6_counterfact.py runs/W6_counterfact_1k_qwen3_4B
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def load(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("method") == "__any__":
                continue  # sentinel for dropped prompts
            if r.get("relation_template_missing"):
                continue
            if r.get("method_unsupported"):
                continue
            rows.append(r)
    return rows


def safe_mean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return mean(xs) if xs else float("nan")


def safe_stdev(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return stdev(xs) if len(xs) > 1 else 0.0


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    cells_path = run_dir / "cells.jsonl"
    if not cells_path.exists():
        print(f"missing {cells_path}", file=sys.stderr)
        sys.exit(2)
    rows = load(cells_path)
    print(f"# loaded {len(rows)} real cells from {cells_path}")

    # group by (method, alpha, seed) for per-seed stats
    groups = defaultdict(list)
    for r in rows:
        key = (r["method"], float(r["alpha"]), int(r["seed"]))
        groups[key].append(r)

    print()
    print("## per-(method, alpha, seed)")
    print(f"{'method':<14} {'alpha':>6} {'seed':>4} {'n':>5} "
          f"{'nll_new':>9} {'nll_true':>9} {'margin':>9} {'recall@1':>9} {'kl_unrel':>9}")
    print("-" * 80)
    for key in sorted(groups):
        cells = groups[key]
        nll_new = [c.get("nll_new") for c in cells]
        nll_true = [c.get("nll_true") for c in cells]
        margins = [(t - n) for t, n in zip(nll_true, nll_new)
                   if t is not None and n is not None
                   and not math.isnan(t) and not math.isnan(n)]
        recall = sum(1 for m in margins if m > 0) / len(margins) if margins else float("nan")
        kl = [c.get("kl_unrel") for c in cells]
        method, a, s = key
        print(f"{method:<14} {a:6.2f} {s:4d} {len(cells):5d} "
              f"{safe_mean(nll_new):9.3f} {safe_mean(nll_true):9.3f} "
              f"{safe_mean(margins):9.3f} {recall:9.3f} {safe_mean(kl):9.4f}")

    # group by (method, alpha) for cross-seed paper table
    by_ma = defaultdict(list)
    for (method, a, s), cells in groups.items():
        margins_seed = []
        for c in cells:
            t, n = c.get("nll_true"), c.get("nll_new")
            if t is None or n is None or math.isnan(t) or math.isnan(n):
                continue
            margins_seed.append(t - n)
        if margins_seed:
            recall_seed = sum(1 for m in margins_seed if m > 0) / len(margins_seed)
            kl_seed = safe_mean([c.get("kl_unrel") for c in cells])
            margin_seed = mean(margins_seed)
            by_ma[(method, a)].append((recall_seed, margin_seed, kl_seed))

    print()
    print("## CROSS-SEED PAPER TABLE  (method, alpha) → recall@1 ± σ, margin ± σ, kl_unrel ± σ")
    print(f"{'method':<14} {'alpha':>6} {'seeds':>6} "
          f"{'recall@1':>16} {'margin (nats)':>20} {'kl_unrel':>16}")
    print("-" * 90)
    for key in sorted(by_ma):
        rs, ms, ks = zip(*by_ma[key])
        method, a = key
        rmean, rstd = mean(rs), safe_stdev(list(rs))
        mmean, mstd = mean(ms), safe_stdev(list(ms))
        kmean, kstd = safe_mean(list(ks)), safe_stdev(list(ks))
        print(f"{method:<14} {a:6.2f} {len(rs):6d} "
              f"{rmean:8.4f} ± {rstd:5.4f}  "
              f"{mmean:9.4f} ± {mstd:7.4f}  "
              f"{kmean:7.4f} ± {kstd:6.4f}")

    # write JSON summary
    out = {
        "n_real_cells": len(rows),
        "groups": {
            f"{m}|alpha={a}": {
                "seeds": len(by_ma[(m, a)]),
                "recall_at_1_mean": mean([r for r, _, _ in by_ma[(m, a)]]),
                "recall_at_1_std": safe_stdev([r for r, _, _ in by_ma[(m, a)]]),
                "margin_mean": mean([mg for _, mg, _ in by_ma[(m, a)]]),
                "margin_std": safe_stdev([mg for _, mg, _ in by_ma[(m, a)]]),
                "kl_unrel_mean": safe_mean([k for _, _, k in by_ma[(m, a)]]),
                "kl_unrel_std": safe_stdev([k for _, _, k in by_ma[(m, a)]]),
            }
            for m, a in by_ma
        },
    }
    out_path = run_dir / "aggregate_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"# wrote {out_path}")


if __name__ == "__main__":
    main()
