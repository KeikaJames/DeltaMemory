"""X.7-NL/D — SCAR signal correlation analysis (post-hoc, no new run).

Reads runs/X7NL_full_v1_gemma4_31B/cells.jsonl and computes Pearson + Spearman
correlations between SCAR-style proxies and recall outcomes.

Proxy mapping (documented in REPORT):
- proj         ≈ bank_col_mean
- ortho        ≈ attn_entropy_bank_mean - attn_entropy_native_mean
- alpha_drift  ≈ residual_norm_mean - residual_norm_mean(α=0 ref) within (sub,seed,bank_size)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rank(xs), _rank(ys))


def add_alpha_drift(rows: list[dict]) -> None:
    """Compute alpha_drift proxy = residual_norm_mean - reference at α=0
    within the same (sub, seed, bank_size) cohort. If no α=0 cell exists in
    cohort, fall back to within-cohort mean.
    """
    from collections import defaultdict
    cohorts: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cohorts[(r["sub"], r["seed"], r.get("bank_size", -1))].append(r)
    for cohort_rows in cohorts.values():
        ref_rows = [c for c in cohort_rows if abs(c.get("alpha", 1.0)) < 1e-6]
        if ref_rows:
            ref = ref_rows[0]["residual_norm_mean"]
        else:
            vals = [c["residual_norm_mean"] for c in cohort_rows
                    if c.get("residual_norm_mean") is not None]
            ref = mean(vals) if vals else 0.0
        for c in cohort_rows:
            c["alpha_drift"] = (c.get("residual_norm_mean") or 0.0) - ref


def correlations(rows: list[dict], outcome_key: str = "log_margin") -> dict:
    """Return Pearson + Spearman of each proxy vs outcome, overall and per-sub."""
    proxies = {
        "proj": lambda r: r.get("bank_col_mean"),
        "ortho": lambda r: (
            (r.get("attn_entropy_bank_mean") or 0.0)
            - (r.get("attn_entropy_native_mean") or 0.0)
        ),
        "alpha_drift": lambda r: r.get("alpha_drift"),
    }
    out: dict = {"overall": {}, "by_sub": {}}

    def block(rs: list[dict]) -> dict:
        d = {}
        for name, f in proxies.items():
            xs = [f(r) for r in rs if f(r) is not None and r.get(outcome_key) is not None]
            ys = [r[outcome_key] for r in rs if f(r) is not None and r.get(outcome_key) is not None]
            d[name] = {
                "n": len(xs),
                "pearson": _pearson(xs, ys),
                "spearman": _spearman(xs, ys),
            }
        return d

    out["overall"] = block(rows)
    for s in sorted({r["sub"] for r in rows}):
        out["by_sub"][s] = block([r for r in rows if r["sub"] == s])
    return out


def write_report(corr: dict, run_dir: Path) -> Path:
    path = run_dir / "SCAR_CORRELATION.md"
    lines = [
        "# X.7-NL/D — SCAR Signal Correlation (gemma-4-31B-it)",
        "",
        "## Method",
        "",
        "Post-hoc correlation analysis on `runs/X7NL_full_v1_gemma4_31B/cells.jsonl`",
        "(N=291). SCAR proxies (since explicit proj/ortho/alpha-drift were not",
        "logged in this run):",
        "",
        "- **proj**         ≈ `bank_col_mean` (mean attention mass on bank columns)",
        "- **ortho**        ≈ `attn_entropy_bank_mean − attn_entropy_native_mean`",
        "- **alpha_drift**  ≈ `residual_norm_mean − residual_norm_mean(α=0)` per (sub,seed,bank_size)",
        "",
        "Outcome: `log_margin = score_new − score_canonical`.",
        "",
        "## Overall correlations",
        "",
        "| proxy | n | pearson | spearman |",
        "| --- | ---: | ---: | ---: |",
    ]
    for k, v in corr["overall"].items():
        lines.append(
            f"| {k} | {v['n']} | {v['pearson']:+.3f} | {v['spearman']:+.3f} |"
        )
    lines += ["", "## Per-sub correlations", ""]
    for s, sub_corr in corr["by_sub"].items():
        lines += [f"### sub = {s}", "",
                  "| proxy | n | pearson | spearman |",
                  "| --- | ---: | ---: | ---: |"]
        for k, v in sub_corr.items():
            lines.append(
                f"| {k} | {v['n']} | {v['pearson']:+.3f} | {v['spearman']:+.3f} |"
            )
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- Proxies are *correlated with* but not identical to the prereg'd",
        "  SCAR signals. A future X.7-NL re-run should record proj/ortho/alpha-drift",
        "  directly via `deltamemory.memory.scar_injector` hooks.",
        "- alpha_drift uses a within-cohort baseline; cohorts without an α=0",
        "  reference (sub A bank-scaling fixes α=1.0) fall back to cohort mean.",
        "- |Pearson| > 0.3 with N≥30 is treated as *suggestive*; |Pearson| > 0.5",
        "  is *strong*. We do not infer causation.",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        default="runs/X7NL_full_v1_gemma4_31B",
        help="Path to X.7-NL run directory (must contain cells.jsonl)",
    )
    args = ap.parse_args()
    run = Path(args.run_dir)
    rows = _load(run / "cells.jsonl")
    add_alpha_drift(rows)
    corr = correlations(rows)
    out = write_report(corr, run)
    print(json.dumps(corr, indent=2, default=str))
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
