"""Exp11 RSM post-processing."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _safe_mean(xs: list[float]) -> float:
    xs = [float(x) for x in xs if x is not None and not math.isnan(float(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _bootstrap_ci(vals: list[float], n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    vals = [float(v) for v in vals if v is not None and not math.isnan(float(v))]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(vals)
    mean = sum(vals) / n
    boots = []
    for _ in range(n_boot):
        boots.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return mean, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def _summarize_results(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_variant[row.get("variant", "unknown")].append(row)
    out = []
    for variant, vrows in by_variant.items():
        margins = [r.get("margin") for r in vrows if r.get("margin") is not None]
        mean, lo, hi = _bootstrap_ci(margins)
        out.append({
            "variant": variant,
            "n": len(vrows),
            "mean_margin": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "recall_at_1": _safe_mean([float(bool(r.get("recall_at_1"))) for r in vrows]),
            "activation_rate": _safe_mean([
                r.get("rsm_activation_rate") for r in vrows
                if r.get("rsm_activation_rate") is not None
            ]),
            "max_score": _safe_mean([
                r.get("rsm_max_score") for r in vrows
                if r.get("rsm_max_score") is not None
            ]),
            "top_memory_hit": _safe_mean([
                float(bool(r.get("rsm_top_memory_hit"))) for r in vrows
                if "rsm_top_memory_hit" in r
            ]),
        })
    return sorted(out, key=lambda r: r["variant"])


def _gap(summary: list[dict[str, Any]]) -> float:
    by_v = {r["variant"]: r for r in summary}
    correct = by_v.get("correct_memory", {}).get("mean_margin", float("-inf"))
    controls = [
        by_v.get("random_memory", {}).get("mean_margin"),
        by_v.get("shuffled_layers", {}).get("mean_margin"),
        by_v.get("gate_off", {}).get("mean_margin"),
    ]
    controls = [c for c in controls if c is not None]
    return correct - max(controls) if controls else float("nan")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze Exp11 RSM runs")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_configs = []
    for phase in ("phase_a", "phase_b"):
        phase_dir = run_dir / phase
        if not phase_dir.exists():
            continue
        phase_rows = []
        for cfg_dir in sorted(p for p in phase_dir.iterdir() if p.is_dir()):
            summary = _summarize_results(cfg_dir / "results.jsonl")
            if not summary:
                continue
            _write_csv(summary, out_dir / f"{phase}_{cfg_dir.name}_summary.csv")
            row = {"phase": phase, "config": cfg_dir.name, "gap": _gap(summary)}
            for s in summary:
                row[s["variant"]] = s["mean_margin"]
            phase_rows.append(row)
            all_configs.append(row)
        _write_csv(phase_rows, out_dir / f"{phase}_comparison.csv")

    best = sorted(all_configs, key=lambda r: r.get("gap", float("-inf")), reverse=True)
    (out_dir / "analysis.json").write_text(json.dumps({"configs": all_configs, "best": best[:3]}, indent=2))
    if best:
        top = best[0]
        verdict = (
            "PASS_DIRECTIONAL"
            if top.get("gap", float("-inf")) > 0 and top.get("correct_memory", -999) > top.get("base_model", 999)
            else "STABILIZER_ONLY"
            if top.get("correct_memory", -999) > top.get("base_model", 999)
            else "FAIL"
        )
    else:
        verdict = "FAIL"
    (out_dir / "README.md").write_text(
        "# Exp11 RSM Analysis\n\n"
        f"**Verdict:** {verdict}\n\n"
        "See `phase_a_comparison.csv`, `phase_b_comparison.csv`, and per-config summaries.\n"
    )
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
