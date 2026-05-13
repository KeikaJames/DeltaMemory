"""Q.1 — Hegel generation runner: aggregate cells.jsonl into summary.json + REPORT.md.

Reads ``cells.jsonl`` and ``transcripts/`` from the experiment output directory.
Computes:
  - per-(model, alpha) hit-rate of contains_counterfact
  - per-(model, alpha) hit-rate of contains_canonical
  - 95% bootstrap CI (1000 resamples, percentile method)

Writes ``summary.json`` and ``REPORT.md`` in the same directory.

Refuses to run if cells.jsonl is missing.
Emits ``partial: true`` in summary if any expected cell has status != 'ok'.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

PREREG_VERSION = "Q1.v1"
BOOTSTRAP_N = 1000
CI_LEVEL = 0.95


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(values: list[float], n: int = BOOTSTRAP_N, ci: float = CI_LEVEL) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(42)
    means = []
    k = len(values)
    for _ in range(n):
        sample = [rng.choice(values) for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    lo_idx = int((1 - ci) / 2 * n)
    hi_idx = int((1 + ci) / 2 * n)
    hi_idx = min(hi_idx, n - 1)
    return (means[lo_idx], means[hi_idx])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_cells(cells_path: Path) -> list[dict]:
    rows = []
    with cells_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate(out_dir: Path) -> dict:
    cells_path = out_dir / "cells.jsonl"
    if not cells_path.exists():
        print(f"[aggregate] ERROR: {cells_path} not found.", file=sys.stderr)
        sys.exit(1)

    rows = load_cells(cells_path)
    if not rows:
        print("[aggregate] ERROR: cells.jsonl is empty.", file=sys.stderr)
        sys.exit(1)

    total = len(rows)
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    partial = len(ok_rows) < total

    # Group by (model, alpha)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in ok_rows:
        key = (r["model"], r["alpha"])
        groups[key].append(r)

    results = []
    for (model, alpha), cells in sorted(groups.items()):
        cf_vals = [1.0 if c.get("contains_counterfact") else 0.0 for c in cells]
        can_vals = [1.0 if c.get("contains_canonical") else 0.0 for c in cells]

        cf_mean = sum(cf_vals) / len(cf_vals) if cf_vals else float("nan")
        can_mean = sum(can_vals) / len(can_vals) if can_vals else float("nan")

        cf_ci = bootstrap_ci(cf_vals)
        can_ci = bootstrap_ci(can_vals)

        # Red-line check
        redline_rows = [c for c in cells if abs(c.get("alpha", 1.0)) < 1e-9]
        redline_all_ok = all(c.get("redline_ok", True) for c in redline_rows) if redline_rows else None

        results.append({
            "model": model,
            "alpha": alpha,
            "n_cells": len(cells),
            "counterfact_hit_rate": cf_mean,
            "counterfact_ci95_lo": cf_ci[0],
            "counterfact_ci95_hi": cf_ci[1],
            "canonical_hit_rate": can_mean,
            "canonical_ci95_lo": can_ci[0],
            "canonical_ci95_hi": can_ci[1],
            "redline_ok": redline_all_ok,
        })

    # H_Q1 check: at alpha=1.0, hit-rate > 0.50 per model?
    h_q1_verdicts = {}
    for r in results:
        if abs(r["alpha"] - 1.0) < 1e-9:
            h_q1_verdicts[r["model"]] = {
                "hit_rate": r["counterfact_hit_rate"],
                "pass": r["counterfact_hit_rate"] > 0.50,
            }

    summary = {
        "prereg_version": PREREG_VERSION,
        "total_cells": total,
        "ok_cells": len(ok_rows),
        "partial": partial,
        "n_observed": len(ok_rows),
        "n_expected": total,
        "results": results,
        "h_q1_verdicts": h_q1_verdicts,
    }
    return summary


def write_report(out_dir: Path, summary: dict) -> None:
    lines = [
        "# Q.1 Hegel Counterfactual Generation — Report",
        "",
        f"**prereg_version:** {summary['prereg_version']}  ",
        f"**cells:** {summary['ok_cells']} / {summary['total_cells']} ok  ",
        f"**partial:** {summary['partial']}  ",
        "",
        "## Hit-rate table",
        "",
        "| Model | α | CF hit-rate | 95% CI | Canon hit-rate | 95% CI | Redline |",
        "|-------|---|-------------|--------|----------------|--------|---------|",
    ]
    for r in summary["results"]:
        cf = f"{r['counterfact_hit_rate']:.3f}"
        cf_ci = f"[{r['counterfact_ci95_lo']:.3f}, {r['counterfact_ci95_hi']:.3f}]"
        can = f"{r['canonical_hit_rate']:.3f}"
        can_ci = f"[{r['canonical_ci95_lo']:.3f}, {r['canonical_ci95_hi']:.3f}]"
        rl = str(r["redline_ok"]) if r["redline_ok"] is not None else "n/a"
        lines.append(
            f"| {r['model']} | {r['alpha']} | {cf} | {cf_ci} | {can} | {can_ci} | {rl} |"
        )

    lines += [
        "",
        "## H_Q1 verdicts (α = 1.0)",
        "",
    ]
    for model, v in summary["h_q1_verdicts"].items():
        verdict = "**PASS**" if v["pass"] else "FAIL"
        lines.append(f"- `{model}`: hit-rate = {v['hit_rate']:.3f} → {verdict}")

    lines += [
        "",
        "## Deviations log",
        "",
        "*(fill in any deviations from PREREG.md here)*",
        "",
    ]

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Q.1 aggregate cells.jsonl → summary.json + REPORT.md")
    p.add_argument("--out", required=True, help="Experiment output directory containing cells.jsonl")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    summary = aggregate(out_dir)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[aggregate] Wrote {summary_path}", flush=True)

    write_report(out_dir, summary)
    print(f"[aggregate] Wrote {out_dir / 'REPORT.md'}", flush=True)

    for model, v in summary["h_q1_verdicts"].items():
        verdict = "PASS" if v["pass"] else "FAIL"
        print(f"[aggregate] H_Q1 {model} α=1.0: {v['hit_rate']:.3f} → {verdict}")


if __name__ == "__main__":
    main()
