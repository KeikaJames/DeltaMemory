#!/usr/bin/env python3
"""X.7 aggregate: capacity x policy curves; H_X7.* verdicts.

Reads cells.jsonl produced by run.py and emits summary.json with:
- per (N, capacity, policy) mean log_margin + std + n + target_resident_rate
- per (N, capacity) LRU-vs-FIFO discriminator (paired diff over seeds)
- H_X7.0 redline: at capacity == 0, behavior matches v0.4 (proxy: target
  always resident, alpha=0 margin == -5 constant)
- H_X7.3 verdict: at capacity = K > 0, recall@1 of target beats the
  no-capacity baseline (capacity=0 unbounded) once N > K — i.e. is
  the LRU-policy version of capacity sweep BETTER, EQUAL, or WORSE
  than just letting the bank grow unbounded?
"""
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
    by = defaultdict(list)  # (N, cap, pol, alpha) -> [margins]
    resident_by = defaultdict(list)  # (N, cap, pol, alpha) -> [bool]
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["N"], r["capacity"], r["policy"], r["alpha"])
        by[key].append(float(r["log_margin"]))
        resident_by[key].append(bool(r.get("target_resident", False)))

    cells = {}  # "N|cap|pol|alpha" -> {mean, stdev, n, resident_rate}
    for (N, cap, pol, alpha), margins in by.items():
        residents = resident_by[(N, cap, pol, alpha)]
        cells[f"{N}|{cap}|{pol}|{alpha}"] = {
            "N": N, "capacity": cap, "policy": pol, "alpha": alpha,
            "mean": statistics.mean(margins),
            "stdev": statistics.pstdev(margins) if len(margins) > 1 else 0.0,
            "n": len(margins),
            "resident_rate": sum(residents) / max(len(residents), 1),
            "samples": margins,
        }

    # H_X7.0 redline: at capacity == 0, alpha == 0, margin should be a
    # constant (≈ the dataset's target_canonical advantage). Concrete
    # value: -5.0 in our X.1 fact pack.
    h0 = {"verdict": "n/a"}
    redline_margins = []
    for k, c in cells.items():
        if c["capacity"] == 0 and c["alpha"] == 0.0:
            redline_margins.extend(c["samples"])
    if redline_margins:
        constant = all(abs(m - redline_margins[0]) < 1e-3 for m in redline_margins)
        h0 = {"verdict": "supported" if constant else "not_supported",
              "redline_value": redline_margins[0],
              "all_equal": constant,
              "n_observations": len(redline_margins)}

    # H_X7.3: at capacity > 0, LRU should preserve target better than
    # FIFO once N > capacity. Compute per (N, cap) the (LRU - FIFO) margin
    # diff at alpha=1 averaged over seeds.
    h3_diffs = {}  # "N|cap" -> {lru_mean, fifo_mean, diff, beats_fifo}
    for N in sorted({c["N"] for c in cells.values()}):
        for cap in sorted({c["capacity"] for c in cells.values()}):
            if cap == 0:
                continue
            lru_key = f"{N}|{cap}|lru|1.0"
            fifo_key = f"{N}|{cap}|fifo|1.0"
            if lru_key in cells and fifo_key in cells:
                lru_m = cells[lru_key]["mean"]
                fifo_m = cells[fifo_key]["mean"]
                h3_diffs[f"{N}|{cap}"] = {
                    "N": N, "capacity": cap,
                    "lru_mean": lru_m, "fifo_mean": fifo_m,
                    "lru_resident": cells[lru_key]["resident_rate"],
                    "fifo_resident": cells[fifo_key]["resident_rate"],
                    "diff_lru_minus_fifo": lru_m - fifo_m,
                    "lru_beats_fifo": lru_m > fifo_m,
                }

    # H_X7.3 final verdict: LRU strictly beats FIFO in MORE THAN HALF the
    # (N, cap) cells where N > cap (the regime where eviction actually
    # bites).
    bite_cells = [v for v in h3_diffs.values() if v["N"] > v["capacity"]]
    if bite_cells:
        wins = sum(1 for v in bite_cells if v["lru_beats_fifo"])
        h3 = {
            "verdict": ("supported" if wins / len(bite_cells) > 0.5
                        else "not_supported"),
            "wins": wins, "total": len(bite_cells),
            "win_rate": wins / len(bite_cells),
            "mean_diff_lru_minus_fifo": statistics.mean(
                v["diff_lru_minus_fifo"] for v in bite_cells),
        }
    else:
        h3 = {"verdict": "n/a", "reason": "no eviction-biting cells"}

    # H_X7 vs unbounded: does capacity-capped LRU ever match unbounded?
    h_match = {}
    for N in sorted({c["N"] for c in cells.values()}):
        unb = cells.get(f"{N}|0|lru|1.0")
        if unb is None:
            continue
        unb_m = unb["mean"]
        for cap in sorted({c["capacity"] for c in cells.values()}):
            if cap == 0:
                continue
            lru = cells.get(f"{N}|{cap}|lru|1.0")
            if lru is None:
                continue
            h_match[f"{N}|{cap}"] = {
                "N": N, "capacity": cap,
                "unbounded_mean": unb_m,
                "lru_capped_mean": lru["mean"],
                "diff": lru["mean"] - unb_m,
                "lru_capped_matches": abs(lru["mean"] - unb_m) < 0.5,
            }

    summary = {
        "n_cells": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "cells": cells,
        "H_X7_0_redline": h0,
        "H_X7_3_lru_beats_fifo": h3,
        "H_X7_3_per_cell": h3_diffs,
        "lru_capped_vs_unbounded": h_match,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"[X7][aggregate] -> {args.out}", flush=True)
    print(f"  H_X7.0 redline:        {h0.get('verdict')}", flush=True)
    print(f"  H_X7.3 LRU>FIFO:       {h3.get('verdict')} "
          f"(wins {h3.get('wins')}/{h3.get('total')})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
