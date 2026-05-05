"""Minimal W.7 aggregator.

Reads ``cells.jsonl`` (default: alongside this script) and groups successful
rows by ``(model, method, alpha, length)``.  Computes mean ``nll_target``
and ``top1_match_frac`` plus a 1000-sample bootstrap 95% CI on the mean
NLL.  Emits ``length_curve.json`` and a tiny ``REPORT.md`` skeleton.

Heavy stats (Wilcoxon + Holm per PREREG section 5) live in a follow-up;
this script's job is to be runnable on smoke output and on partial cell
files without depending on the full grid.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _bootstrap_ci(xs: list[float], B: int = 1000, seed: int = 0
                  ) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(xs)
    means: list[float] = []
    for _ in range(B):
        s = sum(xs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * B)]
    hi = means[int(0.975 * B)]
    return lo, hi


def _is_real(row: dict) -> bool:
    return (
        row.get("status") == "ok"
        and not row.get("method_unsupported", False)
        and row.get("alpha", -1) != -1
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=str(
        ROOT / "experiments" / "W7_longctx" / "cells.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "experiments" / "W7_longctx"))
    args = ap.parse_args()

    cells_path = Path(args.cells)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in _load_jsonl(cells_path) if _is_real(r)]
    print(f"[W7-agg] loaded {len(rows)} real rows from {cells_path}")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["method"], float(r["alpha"]), int(r["length"]))
        groups[key].append(r)

    curve = []
    for (model, method, alpha, length), grp in sorted(groups.items()):
        nlls = [float(g["nll_target"]) for g in grp
                if g.get("nll_target") is not None]
        t1s = [float(g["top1_match_frac"]) for g in grp
               if g.get("top1_match_frac") is not None]
        drifts = [float(g["drift"]) for g in grp
                  if g.get("drift") is not None]
        ci_lo, ci_hi = _bootstrap_ci(nlls)
        curve.append({
            "model": model,
            "method": method,
            "alpha": alpha,
            "length": length,
            "n": len(grp),
            "mean_nll_target": (sum(nlls) / len(nlls)) if nlls else None,
            "nll_target_ci95": [ci_lo, ci_hi],
            "mean_top1_match_frac": (sum(t1s) / len(t1s)) if t1s else None,
            "mean_drift": (sum(drifts) / len(drifts)) if drifts else None,
            "redline_violations": sum(
                1 for g in grp if g.get("redline_violation")),
        })

    out_curve = out_dir / "length_curve.json"
    with open(out_curve, "w") as f:
        json.dump({"groups": curve}, f, indent=2)
    print(f"[W7-agg] wrote {out_curve}")

    # Minimal REPORT.md skeleton — full narrative pending the full grid.
    rep = out_dir / "REPORT.md"
    if not rep.exists():
        rep.write_text(
            "# W.7 Long-Context Degradation — REPORT (skeleton)\n\n"
            "Status: smoke / partial.  Full grid pending GB10 launch.\n\n"
            "## Length curve\n\nSee `length_curve.json`.\n\n"
            "## Hypotheses\n\n- H7a (drift bound): pending full grid.\n"
            "- H7b (rank preservation): pending full grid.\n"
            "- H7c (alpha=0 bit-equality): see redline_violations column "
            "in `length_curve.json`.\n"
        )
        print(f"[W7-agg] wrote skeleton {rep}")


if __name__ == "__main__":
    main()
