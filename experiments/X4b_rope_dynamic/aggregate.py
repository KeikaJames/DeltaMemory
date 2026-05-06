#!/usr/bin/env python3
"""X.4b aggregate: per-(model, fact, alpha) margin variance across P."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    rows = []
    with args.cells.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Group margins by (model, alpha, fact_id, seed) across P.
    by_group = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok" or "score_margin" not in r:
            continue
        key = (r["model"], r["alpha"], r["fact_id"], r["seed"])
        by_group[key].append((r["P"], r["score_margin"]))

    # Per-group stability metric: max-min margin and stdev across P.
    summaries = []
    for (model, alpha, fid, seed), pts in by_group.items():
        margins = [m for _, m in pts]
        if len(margins) < 2:
            continue
        spread = max(margins) - min(margins)
        std = statistics.pstdev(margins) if len(margins) > 1 else 0.0
        rel = std / (abs(statistics.mean(margins)) + 1e-9)
        summaries.append({
            "model": model, "alpha": alpha, "fact_id": fid, "seed": seed,
            "n_positions": len(margins),
            "margin_min": min(margins), "margin_max": max(margins),
            "margin_spread": spread, "margin_stdev": std,
            "margin_rel_stdev": rel,
            "positions": [int(p) for p, _ in pts],
        })

    # Red-line aggregate.
    redline_rows = [r for r in rows if r.get("alpha") == 0.0
                    and "redline_ok" in r]
    redline_pass = sum(1 for r in redline_rows if r["redline_ok"])
    redline_total = len(redline_rows)
    redline_max_diff = max(
        (r.get("redline_max_abs_diff", 0.0) for r in redline_rows),
        default=0.0,
    )

    # H_X4b.1 verdict per (model, alpha=1) cohort.
    alpha1 = [s for s in summaries if s["alpha"] == 1.0]
    h1_violations = [s for s in alpha1 if s["margin_rel_stdev"] > 0.10]

    out = {
        "n_cells": len(rows),
        "n_groups": len(summaries),
        "redline": {
            "pass": redline_pass, "total": redline_total,
            "max_abs_diff": redline_max_diff,
            "ok_all": redline_pass == redline_total and redline_total > 0,
        },
        "H_X4b_1_alpha1_groups": len(alpha1),
        "H_X4b_1_violations_count": len(h1_violations),
        "H_X4b_1_pass_rate": (
            (len(alpha1) - len(h1_violations)) / len(alpha1)
            if alpha1 else None
        ),
        "violations": h1_violations,
        "per_group": summaries,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[X4b][aggregate] {len(rows)} cells -> {args.out}")
    print(f"  redline: {redline_pass}/{redline_total} pass "
          f"(max_diff={redline_max_diff:.2e})")
    if alpha1:
        print(f"  H_X4b.1: {len(h1_violations)}/{len(alpha1)} violations "
              f"(rel_stdev > 0.10)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
