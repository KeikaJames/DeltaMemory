#!/usr/bin/env python3
"""X.2 aggregator: H_X2.0/1/2/3 verdicts from cells.jsonl."""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def load_cells(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def mean(xs):
    return st.fmean(xs) if xs else float("nan")


def stdev(xs):
    return st.pstdev(xs) if len(xs) > 1 else 0.0


def aggregate(cells: list[dict]) -> dict:
    ok = [c for c in cells if c.get("status") == "ok"]
    by_cond = defaultdict(list)
    for c in ok:
        key = (c["order"], c["N"], c["capacity"], c["policy"], c["alpha"])
        by_cond[key].append(c)

    per_cond = {}
    for key, rows in by_cond.items():
        order, N, cap, pol, a = key
        ms = [r["log_margin_AB"] for r in rows]
        winners = [r["winner"] for r in rows]
        per_cond[f"{order}|N={N}|cap={cap}|{pol}|a={a}"] = {
            "order": order, "N": N, "capacity": cap, "policy": pol,
            "alpha": a, "n": len(rows),
            "mean_log_margin_AB": mean(ms),
            "std_log_margin_AB": stdev(ms),
            "winner_A_rate": sum(1 for w in winners if w == "A") / len(winners),
            "mean_score_A": mean([r["score_A"] for r in rows]),
            "mean_score_B": mean([r["score_B"] for r in rows]),
            "mean_score_canon": mean([r["score_canon"] for r in rows]),
            "mean_target_A_resident": mean(
                [float(r["target_A_resident"]) for r in rows]
            ),
            "mean_target_B_resident": mean(
                [float(r["target_B_resident"]) for r in rows]
            ),
        }

    # H_X2.0: alpha=0 redline — across all alpha=0 cells, log_margin_AB
    # should be invariant up to abstol=1e-3.
    a0 = [c for c in ok if c["alpha"] == 0.0]
    a0_ms = [c["log_margin_AB"] for c in a0]
    h0 = {
        "n_cells": len(a0),
        "min_log_margin_AB": min(a0_ms) if a0_ms else None,
        "max_log_margin_AB": max(a0_ms) if a0_ms else None,
        "spread": (max(a0_ms) - min(a0_ms)) if a0_ms else None,
        "supported": (
            (max(a0_ms) - min(a0_ms) < 1e-3) if len(a0_ms) > 1 else None
        ),
    }

    # H_X2.1: recency wins at cap=0, N=0, alpha=1.
    # A_first => last writer is B => winner should be B.
    # B_first => last writer is A => winner should be A.
    rec = [c for c in ok if c["capacity"] == 0 and c["N"] == 0
           and c["alpha"] == 1.0]
    correct = 0
    total = 0
    for r in rec:
        total += 1
        expected = "B" if r["order"] == "A_first" else "A"
        if r["winner"] == expected:
            correct += 1
    h1 = {
        "n_cells": total,
        "recency_winner_rate": correct / total if total else None,
        "supported": (correct / total >= 0.8) if total else None,
    }

    # H_X2.2: LRU distance sensitivity. At cap=64, N=1000, alpha=1, lru:
    # mean(target_A_resident | A_first) < mean(target_A_resident | B_first) - 0.2.
    def res_mean(order):
        rows = [c for c in ok if c["order"] == order
                and c["capacity"] == 64 and c["N"] == 1000
                and c["alpha"] == 1.0 and c["policy"] == "lru"]
        if not rows:
            return None, 0
        return mean([float(r["target_A_resident"]) for r in rows]), len(rows)
    a_res, na = res_mean("A_first")
    b_res, nb = res_mean("B_first")
    if a_res is None or b_res is None:
        h2 = {"n_A": na, "n_B": nb, "supported": None}
    else:
        h2 = {
            "n_A": na, "n_B": nb,
            "A_first_target_A_resident": a_res,
            "B_first_target_A_resident": b_res,
            "diff": b_res - a_res,
            "supported": (b_res - a_res >= 0.2),
        }

    # H_X2.3: FIFO rigidity. order=A_first, policy=fifo, N>cap, alpha>0:
    # target_A_resident should be False in all cells.
    fifo_rigid = [c for c in ok if c["order"] == "A_first"
                  and c["policy"] == "fifo" and c["capacity"] > 0
                  and c["N"] > c["capacity"] and c["alpha"] > 0]
    n_resident = sum(1 for c in fifo_rigid if c["target_A_resident"])
    h3 = {
        "n_cells": len(fifo_rigid),
        "n_target_A_still_resident": n_resident,
        "supported": (
            (n_resident == 0) if fifo_rigid else None
        ),
    }

    return {
        "n_cells_total": len(cells),
        "n_cells_ok": len(ok),
        "verdicts": {
            "H_X2.0_alpha0_redline": h0,
            "H_X2.1_recency_wins": h1,
            "H_X2.2_lru_distance_sensitivity": h2,
            "H_X2.3_fifo_rigidity": h3,
        },
        "per_condition": per_cond,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cells = load_cells(args.cells)
    summary = aggregate(cells)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"[X2][aggregate] -> {args.out}")
    v = summary["verdicts"]
    for k, h in v.items():
        print(f"  {k}: supported={h.get('supported')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
