"""W.6 counter-prior aggregator.

Reads ``cells.jsonl`` (or ``cells_smoke.jsonl``) produced by ``run.py`` and
emits ``summary.json`` (H6a + H6b verdicts) and ``pareto.json`` (H6c
frontier per model).

Statistics
----------
* H6a: paired Wilcoxon signed-rank, two-sided, ``zero_method='wilcox'``,
  on ``nll_new(M_winner) - nll_new(none)`` paired by ``(seed, prompt_id)``;
  Holm-Bonferroni across 5 models x 7 alphas = 35 comparisons; threshold
  0.01.
* H6b: same test on ``nll_new(M_winner) - nll_true(M_winner)``; same
  Holm family.
* Effect size: median paired diff with 95% bootstrap CI, B=1000, seed=0.
* H6c: per-model Pareto frontier of ``(alpha, median nll_new, median
  kl_unrel)``; the alpha that minimises ``nll_new`` and the matching
  ``kl_unrel`` are reported, with a strict per-model threshold of
  ``kl_unrel < 0.5`` nats.

Sentinel rows where ``method_unsupported`` or ``relation_template_missing``
are dropped before any statistic is computed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Loading


def load_cells(path: Path) -> list[dict]:
    open_fn = gzip.open if str(path).endswith(".gz") else open
    rows: list[dict] = []
    with open_fn(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_real(cells: list[dict]) -> list[dict]:
    return [
        c for c in cells
        if not c.get("method_unsupported", False)
        and not c.get("relation_template_missing", False)
    ]


# ---------------------------------------------------------------------------
# Wilcoxon + Holm + bootstrap


def _wilcoxon_two_sided(diffs: list[float]) -> tuple[float, float]:
    """Return (statistic, p_value) for a paired Wilcoxon signed-rank test
    with ``zero_method='wilcox'`` (drop zero pairs)."""
    if not diffs:
        return float("nan"), float("nan")
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return float("nan"), float("nan")
    nz = [d for d in diffs if d != 0.0]
    if len(nz) < 1:
        return float("nan"), 1.0
    try:
        stat, p = wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def _holm_bonferroni(pvals: list[float], alpha: float) -> list[bool]:
    """Holm-Bonferroni step-down.  Returns ``reject`` booleans aligned with
    ``pvals``."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (pvals[i] if pvals[i] == pvals[i]
                                             else float("inf")))
    reject = [False] * n
    for rank, idx in enumerate(order):
        p = pvals[idx]
        if p != p:  # NaN
            continue
        thresh = alpha / (n - rank)
        if p <= thresh:
            reject[idx] = True
        else:
            break
    return reject


def _bootstrap_median_ci(
    diffs: list[float],
    B: int = 1000,
    seed: int = 0,
    conf: float = 0.95,
) -> tuple[float, float, float]:
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    medians = []
    n = len(arr)
    for _ in range(B):
        sample = arr[rng.integers(0, n, size=n)]
        medians.append(float(np.median(sample)))
    lo = float(np.percentile(medians, (1 - conf) / 2 * 100))
    hi = float(np.percentile(medians, (1 + conf) / 2 * 100))
    return float(np.median(arr)), lo, hi


# ---------------------------------------------------------------------------
# Per-(model, alpha) paired tables


def _pair_diffs_h6a(real: list[dict], model: str, alpha: float,
                     M_winner: str) -> list[float]:
    """nll_new(M_winner) - nll_new(none), paired by (seed, prompt_id)."""
    none_map: dict[tuple, float] = {}
    win_map: dict[tuple, float] = {}
    for c in real:
        if c["model"] != model or float(c["alpha"]) != float(alpha):
            continue
        key = (int(c["seed"]), c["prompt_id"])
        if c["method"] == "none":
            none_map[key] = float(c["nll_new"])
        elif c["method"] == M_winner:
            win_map[key] = float(c["nll_new"])
    diffs = []
    for k, v in win_map.items():
        if k in none_map and v == v and none_map[k] == none_map[k]:
            diffs.append(v - none_map[k])
    return diffs


def _pair_diffs_h6b(real: list[dict], model: str, alpha: float,
                     M_winner: str) -> list[float]:
    """nll_new(M_winner) - nll_true(M_winner), paired by (seed, prompt_id)."""
    diffs = []
    for c in real:
        if c["model"] != model or float(c["alpha"]) != float(alpha):
            continue
        if c["method"] != M_winner:
            continue
        nn, nt = c.get("nll_new"), c.get("nll_true")
        if nn is None or nt is None:
            continue
        if nn != nn or nt != nt:
            continue
        diffs.append(float(nn) - float(nt))
    return diffs


# ---------------------------------------------------------------------------
# H6c — Pareto frontier


def pareto_frontier(real: list[dict], M_winner: str) -> dict:
    """Per-model frontier of (alpha, median nll_new, median kl_unrel)."""
    out: dict[str, Any] = {"M_winner": M_winner, "models": {}}
    models = sorted(set(c["model"] for c in real))
    for model in models:
        rows = [c for c in real if c["model"] == model and c["method"] == M_winner]
        alphas = sorted(set(float(c["alpha"]) for c in rows))
        frontier = []
        for a in alphas:
            sub = [c for c in rows if float(c["alpha"]) == a]
            nlls = [float(c["nll_new"]) for c in sub
                    if c.get("nll_new") is not None
                    and c["nll_new"] == c["nll_new"]]
            kls = [float(c["kl_unrel"]) for c in sub
                   if c.get("kl_unrel") is not None
                   and c["kl_unrel"] == c["kl_unrel"]]
            if not nlls:
                continue
            frontier.append({
                "alpha": a,
                "median_nll_new": float(np.median(nlls)),
                "median_kl_unrel": float(np.median(kls)) if kls else float("nan"),
                "n": len(sub),
            })
        if not frontier:
            out["models"][model] = {
                "frontier": [],
                "best_alpha": None,
                "best_kl_unrel": None,
                "passes_h6c": False,
            }
            continue

        best = min(frontier, key=lambda r: r["median_nll_new"])
        passes = (best["median_kl_unrel"] == best["median_kl_unrel"]
                  and best["median_kl_unrel"] < 0.5)
        out["models"][model] = {
            "frontier": frontier,
            "best_alpha": best["alpha"],
            "best_median_nll_new": best["median_nll_new"],
            "best_kl_unrel": best["median_kl_unrel"],
            "passes_h6c": bool(passes),
        }
    return out


# ---------------------------------------------------------------------------
# Main


def aggregate(cells: list[dict], M_winner: str) -> tuple[dict, dict]:
    real = filter_real(cells)

    models = sorted(set(c["model"] for c in real))
    alphas = sorted(set(float(c["alpha"]) for c in real if float(c["alpha"]) >= 0))

    # H6a / H6b — collect (model, alpha) cells.
    h6a_cells: list[dict] = []
    h6b_cells: list[dict] = []
    for model in models:
        for alpha in alphas:
            d_a = _pair_diffs_h6a(real, model, alpha, M_winner)
            stat_a, p_a = _wilcoxon_two_sided(d_a)
            med_a, lo_a, hi_a = _bootstrap_median_ci(d_a)
            h6a_cells.append({
                "model": model, "alpha": alpha, "n_pairs": len(d_a),
                "wilcoxon_stat": stat_a, "p_raw": p_a,
                "median_diff": med_a, "ci95_lo": lo_a, "ci95_hi": hi_a,
            })
            d_b = _pair_diffs_h6b(real, model, alpha, M_winner)
            stat_b, p_b = _wilcoxon_two_sided(d_b)
            med_b, lo_b, hi_b = _bootstrap_median_ci(d_b)
            h6b_cells.append({
                "model": model, "alpha": alpha, "n_pairs": len(d_b),
                "wilcoxon_stat": stat_b, "p_raw": p_b,
                "median_diff": med_b, "ci95_lo": lo_b, "ci95_hi": hi_b,
            })

    # Holm-Bonferroni per family.
    p_a = [c["p_raw"] for c in h6a_cells]
    rej_a = _holm_bonferroni(p_a, alpha=0.01)
    for c, r in zip(h6a_cells, rej_a):
        c["holm_reject"] = bool(r)

    p_b = [c["p_raw"] for c in h6b_cells]
    rej_b = _holm_bonferroni(p_b, alpha=0.01)
    for c, r in zip(h6b_cells, rej_b):
        c["holm_reject"] = bool(r)

    # Counts.
    redline_count = sum(1 for c in cells if c.get("redline_violation"))
    unsupported_count = sum(1 for c in cells if c.get("method_unsupported"))
    template_missing = sum(1 for c in cells if c.get("relation_template_missing"))

    summary = {
        "M_winner": M_winner,
        "n_cells_total": len(cells),
        "n_cells_real": len(real),
        "n_cells_redline_violation": redline_count,
        "n_cells_method_unsupported": unsupported_count,
        "n_cells_relation_template_missing": template_missing,
        "models": models,
        "alphas": alphas,
        "h6a": {
            "name": "median_p [nll_new(M_winner) - nll_new(none)] < 0",
            "family_size": len(h6a_cells),
            "holm_threshold": 0.01,
            "cells": h6a_cells,
            "n_reject": sum(1 for c in h6a_cells if c["holm_reject"]),
        },
        "h6b": {
            "name": "median_p [nll_new(M_winner) - nll_true(M_winner)] < 0",
            "family_size": len(h6b_cells),
            "holm_threshold": 0.01,
            "cells": h6b_cells,
            "n_reject": sum(1 for c in h6b_cells if c["holm_reject"]),
        },
    }

    pareto = pareto_frontier(real, M_winner)
    return summary, pareto


def _resolve_m_winner(cells_path: Path) -> str:
    env = cells_path.parent / "env.json"
    if env.exists():
        try:
            with open(env) as f:
                e = json.load(f)
            return e.get("method_winner", "caa")
        except Exception:
            pass
    return "caa"


def main() -> None:
    ap = argparse.ArgumentParser(description="W.6 counter-prior aggregator")
    ap.add_argument("--cells", default="experiments/W6_counter_prior/cells.jsonl")
    ap.add_argument("--out", default="experiments/W6_counter_prior/")
    ap.add_argument("--m-winner", default=None,
                    help="Override M_winner (default: read from env.json).")
    args = ap.parse_args()

    cells_path = Path(args.cells)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cells_path.exists():
        gz = cells_path.with_suffix(".jsonl.gz")
        if gz.exists():
            cells_path = gz
        else:
            print(f"[agg] cells file not found: {cells_path}", file=sys.stderr)
            sys.exit(1)

    print(f"[agg] loading {cells_path}", flush=True)
    cells = load_cells(cells_path)
    print(f"[agg] {len(cells)} cells loaded", flush=True)

    M_winner = args.m_winner or _resolve_m_winner(cells_path)
    print(f"[agg] M_winner={M_winner}", flush=True)

    summary, pareto = aggregate(cells, M_winner)

    sum_path = out_dir / "summary.json"
    par_path = out_dir / "pareto.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    with open(par_path, "w") as f:
        json.dump(pareto, f, indent=2, sort_keys=True)

    print(f"[agg] -> {sum_path}", flush=True)
    print(f"[agg] -> {par_path}", flush=True)
    print(f"[agg] H6a: {summary['h6a']['n_reject']}/"
          f"{summary['h6a']['family_size']} cells reject (Holm 0.01)")
    print(f"[agg] H6b: {summary['h6b']['n_reject']}/"
          f"{summary['h6b']['family_size']} cells reject (Holm 0.01)")
    h6c_passes = sum(1 for v in pareto["models"].values()
                     if v.get("passes_h6c"))
    print(f"[agg] H6c: {h6c_passes}/{len(pareto['models'])} models "
          f"pass kl_unrel<0.5 at best alpha")


if __name__ == "__main__":
    main()
