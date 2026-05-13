#!/usr/bin/env python3
"""X.7-NL aggregator → summary.json with per-hypothesis verdicts.

Reads `cells.jsonl` from a single run dir, partitions rows by sub-experiment,
and writes summary.json + a minimal REPORT.md skeleton.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


def load_cells(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def is_monotone_decreasing(xs: list[float], tol: float = 1e-3) -> bool:
    return all(b <= a + tol for a, b in zip(xs, xs[1:]))


def is_monotone_increasing(xs: list[float], tol: float = 1e-3) -> bool:
    return all(b >= a - tol for a, b in zip(xs, xs[1:]))


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def verdict_A(rows_A: list[dict]) -> dict[str, Any]:
    """H_X7N.A1/2/3 — bank-size scaling."""
    out: dict[str, Any] = {}
    if not rows_A:
        return {"H_X7N.A1": "inconclusive",
                "H_X7N.A2": "inconclusive",
                "H_X7N.A3": "inconclusive",
                "n": 0}
    by_size: dict[int, list[dict]] = {}
    for r in rows_A:
        if r.get("status") != "ok":
            continue
        by_size.setdefault(int(r["bank_size"]), []).append(r)
    sizes = sorted(by_size)
    bank_ent = [statistics.mean(r.get("attn_entropy_bank_mean", float("nan"))
                                for r in by_size[s])
                for s in sizes]
    bank_col = [statistics.mean(r.get("bank_col_mean", float("nan"))
                                for r in by_size[s])
                for s in sizes]
    resid = [statistics.mean(r.get("residual_norm_mean", float("nan"))
                             for r in by_size[s])
             for s in sizes]

    def _drop_nan(xs):
        return [x for x in xs if not (isinstance(x, float) and math.isnan(x))]

    bank_col_clean = _drop_nan(bank_col)
    resid_clean = _drop_nan(resid)

    # A1: mean attn weight on injected tokens monotone-decreasing in |bank|.
    if len(bank_col_clean) >= 3:
        out["H_X7N.A1"] = "supported" if is_monotone_decreasing(
            bank_col_clean) else "not_supported"
    else:
        out["H_X7N.A1"] = "inconclusive"

    # A2: bank-attention entropy plateaus at large |bank|.
    if len(bank_ent) >= 2 and not math.isnan(bank_ent[-1]):
        delta_last = abs(bank_ent[-1] - bank_ent[-2])
        out["H_X7N.A2"] = "supported" if delta_last < 0.05 else "not_supported"
    else:
        out["H_X7N.A2"] = "inconclusive"

    # A3: residual norm drift <5% across sweep.
    if len(resid_clean) >= 2:
        rng = max(resid_clean) - min(resid_clean)
        rel = rng / max(1e-9, statistics.mean(resid_clean))
        out["H_X7N.A3"] = "supported" if rel < 0.05 else "not_supported"
        out["A3_rel_drift"] = rel
    else:
        out["H_X7N.A3"] = "inconclusive"

    out["sizes"] = sizes
    out["bank_col_mean_per_size"] = bank_col
    out["bank_entropy_per_size"] = bank_ent
    out["residual_norm_per_size"] = resid
    out["n"] = sum(len(v) for v in by_size.values())
    return out


def verdict_B(rows_B: list[dict]) -> dict[str, Any]:
    """H_X7N.B1/2/3 — alpha sweep."""
    out: dict[str, Any] = {}
    if not rows_B:
        return {"H_X7N.B1": "inconclusive",
                "H_X7N.B2": "inconclusive",
                "H_X7N.B3": "inconclusive",
                "n": 0}
    by_alpha: dict[float, list[dict]] = {}
    for r in rows_B:
        if r.get("status") != "ok":
            continue
        by_alpha.setdefault(round(float(r["alpha"]), 4), []).append(r)
    alphas = sorted(by_alpha)
    margin = [statistics.mean(r["log_margin"] for r in by_alpha[a])
              for a in alphas]
    resid = [statistics.mean(r.get("residual_norm_mean", float("nan"))
                             for r in by_alpha[a])
             for a in alphas]

    # B1: monotone-increasing on [0,1].
    sub_low = [(a, m) for a, m in zip(alphas, margin) if a <= 1.0 + 1e-6]
    if len(sub_low) >= 3:
        ms = [m for _, m in sub_low]
        out["H_X7N.B1"] = "supported" if is_monotone_increasing(
            ms, tol=0.1) else "not_supported"
    else:
        out["H_X7N.B1"] = "inconclusive"

    # B2: saturation/inversion above α=1.
    sub_high = [(a, m) for a, m in zip(alphas, margin) if a >= 1.0]
    if len(sub_high) >= 3:
        ms = [m for _, m in sub_high]
        peak = max(ms)
        end = ms[-1]
        out["H_X7N.B2"] = "supported" if (peak - end) > 0.5 or \
            not is_monotone_increasing(ms, tol=0.1) else "not_supported"
    else:
        out["H_X7N.B2"] = "inconclusive"

    # B3: residual super-linear above α=1.5.
    sub_top = [(a, r) for a, r in zip(alphas, resid)
               if a >= 1.5 and not math.isnan(r)]
    sub_one = [(a, r) for a, r in zip(alphas, resid)
               if abs(a - 1.0) < 0.05 and not math.isnan(r)]
    if sub_top and sub_one:
        ratio = sub_top[-1][1] / max(1e-9, sub_one[0][1])
        out["H_X7N.B3"] = "supported" if ratio > 1.5 else "not_supported"
        out["B3_resid_ratio_2_over_1"] = ratio
    else:
        out["H_X7N.B3"] = "inconclusive"

    out["alphas"] = alphas
    out["margin_per_alpha"] = margin
    out["residual_per_alpha"] = resid
    out["n"] = sum(len(v) for v in by_alpha.values())
    return out


def verdict_C(rows_C: list[dict]) -> dict[str, Any]:
    """H_X7N.C1/2 — multi-turn alternation."""
    out: dict[str, Any] = {}
    if not rows_C:
        return {"H_X7N.C1": "inconclusive",
                "H_X7N.C2": "inconclusive",
                "n": 0}
    ok = [r for r in rows_C if r.get("status") == "ok"]
    if not ok:
        return {"H_X7N.C1": "inconclusive",
                "H_X7N.C2": "inconclusive", "n": 0}

    # Aggregate by turn (mean across seeds), look for sign-alternation in
    # log_margin (margin is positive when target_new (A) wins).
    by_turn: dict[int, list[float]] = {}
    by_turn_resid: dict[int, list[float]] = {}
    for r in ok:
        t = int(r["turn"])
        by_turn.setdefault(t, []).append(float(r["log_margin"]))
        rn = r.get("residual_norm_mean")
        if rn is not None and not math.isnan(float(rn)):
            by_turn_resid.setdefault(t, []).append(float(rn))
    turns = sorted(by_turn)
    margins = [statistics.mean(by_turn[t]) for t in turns]

    # C1: sign alternates with most-recent write. Even turns wrote A,
    # odd turns wrote A'. So even-turn margins > 0 and odd-turn < 0.
    even_pos = [m for t, m in zip(turns, margins) if t % 2 == 0]
    odd_neg = [m for t, m in zip(turns, margins) if t % 2 == 1]
    if even_pos and odd_neg:
        even_avg = statistics.mean(even_pos)
        odd_avg = statistics.mean(odd_neg)
        out["H_X7N.C1"] = ("supported"
                           if (even_avg - odd_avg) > 0.5
                           else "not_supported")
        out["C1_even_avg"] = even_avg
        out["C1_odd_avg"] = odd_avg
    else:
        out["H_X7N.C1"] = "inconclusive"

    # C2: per-turn ‖residual‖₂ change is bounded by a constant ε.
    if len(by_turn_resid) >= 3:
        rs = [statistics.mean(by_turn_resid[t])
              for t in sorted(by_turn_resid)]
        diffs = [abs(b - a) for a, b in zip(rs, rs[1:])]
        eps = max(diffs) if diffs else 0.0
        rel = eps / max(1e-9, statistics.mean(rs))
        out["H_X7N.C2"] = "supported" if rel < 0.10 else "not_supported"
        out["C2_max_rel_step"] = rel
    else:
        out["H_X7N.C2"] = "inconclusive"

    out["n"] = len(ok)
    out["turns"] = turns
    out["margin_per_turn"] = margins
    return out


def verdict_D(rows_all: list[dict]) -> dict[str, Any]:
    """H_X7N.D1/D2 — SCAR signal correlation (post-hoc)."""
    out: dict[str, Any] = {}
    ok = [r for r in rows_all if r.get("status") == "ok"]
    proj = [r.get("scar_proj_mass_mean") for r in ok]
    drift = [r.get("scar_alpha_drift_mean") for r in ok]
    margin = [r.get("log_margin") for r in ok]

    def _filter(xs, ys):
        out_x, out_y = [], []
        for a, b in zip(xs, ys):
            if a is None or b is None:
                continue
            if isinstance(a, float) and (math.isnan(a) or math.isinf(a)):
                continue
            if isinstance(b, float) and (math.isnan(b) or math.isinf(b)):
                continue
            out_x.append(float(a))
            out_y.append(float(b))
        return out_x, out_y

    px, py = _filter(proj, margin)
    if len(px) >= 8:
        r1 = pearson(px, py)
        out["D1_pearson_proj_margin"] = r1
        out["H_X7N.D1"] = "supported" if r1 > 0.5 else "not_supported"
    else:
        out["H_X7N.D1"] = "inconclusive"

    dx, dy = _filter(drift, margin)
    if len(dx) >= 8:
        r2 = pearson(dx, dy)
        out["D2_pearson_drift_margin"] = r2
        # D2 is a tail-behaviour claim; we only flag inconclusive vs anomalous.
        out["H_X7N.D2"] = "inconclusive"  # requires saturation arm in B
    else:
        out["H_X7N.D2"] = "inconclusive"

    out["n"] = len(ok)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()

    cells = load_cells(args.run_dir / "cells.jsonl")
    if not cells:
        print("[X7NL.agg] no cells found", file=sys.stderr)
        return 1

    rows_A = [r for r in cells if r.get("sub") == "A"]
    rows_B = [r for r in cells if r.get("sub") == "B"]
    rows_C = [r for r in cells if r.get("sub") == "C"]

    vA = verdict_A(rows_A)
    vB = verdict_B(rows_B)
    vC = verdict_C(rows_C)
    vD = verdict_D(cells)

    verdicts = {}
    for src in (vA, vB, vC, vD):
        for k, v in src.items():
            if k.startswith("H_X7N."):
                verdicts[k] = v

    n_ok = sum(1 for r in cells if r.get("status") == "ok")
    arch = cells[0].get("model", "?") if cells else "?"
    summary = {
        "verdicts": verdicts,
        "n_cells": len(cells),
        "n_ok": n_ok,
        "arch": arch,
        "subA": vA, "subB": vB, "subC": vC, "subD": vD,
        "headline": _headline(verdicts),
    }
    out_path = args.run_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[X7NL.agg] wrote {out_path} (n_cells={len(cells)} n_ok={n_ok})")

    rep = args.run_dir / "REPORT.md"
    if not rep.exists():
        lines = [
            f"# X.7-NL Report — {arch}",
            "",
            "Auto-generated skeleton; replace with narrative when run is complete.",
            "",
            "## Verdicts",
            "",
        ]
        for k in sorted(verdicts):
            lines.append(f"- **{k}**: {verdicts[k]}")
        lines += ["", f"**Headline**: {summary['headline']}", ""]
        rep.write_text("\n".join(lines))

    return 0


def _headline(verdicts: dict[str, str]) -> str:
    sup = sum(1 for v in verdicts.values() if v == "supported")
    nsup = sum(1 for v in verdicts.values() if v == "not_supported")
    inc = sum(1 for v in verdicts.values() if v == "inconclusive")
    return f"{sup} supported / {nsup} not_supported / {inc} inconclusive"


if __name__ == "__main__":
    sys.exit(main())
