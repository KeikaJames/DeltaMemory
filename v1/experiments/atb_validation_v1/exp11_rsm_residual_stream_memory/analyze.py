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
            "mean_score": _safe_mean([
                r.get("rsm_mean_score") for r in vrows
                if r.get("rsm_mean_score") is not None
            ]),
            "min_score": _safe_mean([
                r.get("rsm_min_score") for r in vrows
                if r.get("rsm_min_score") is not None
            ]),
            "score_std": _safe_mean([
                r.get("rsm_score_std") for r in vrows
                if r.get("rsm_score_std") is not None
            ]),
            "top_minus_mean": _safe_mean([
                r.get("rsm_top_score_minus_mean") for r in vrows
                if r.get("rsm_top_score_minus_mean") is not None
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
        by_v.get("gate_uniform", {}).get("mean_margin"),
    ]
    controls = [c for c in controls if c is not None]
    return correct - max(controls) if controls else float("nan")


_CONTROL_NAMES = ("random_memory", "shuffled_layers", "gate_off", "gate_uniform")


def _verdict_from_summary(summary: list[dict[str, Any]]) -> str:
    """Verdict using CI separation when available.

    Ladder:
      PASS_STRONG       : correct.ci_lo > max(controls.mean) and correct > base.
      PASS_DIRECTIONAL  : correct.mean > all controls.mean and correct > base.
      STABILIZER_ONLY   : correct > base and correct > random_memory.
      FAIL              : otherwise.
    """
    by_v = {r["variant"]: r for r in summary}
    correct = by_v.get("correct_memory", {})
    base_mean = by_v.get("base_model", {}).get("mean_margin", float("inf"))
    random_mean = by_v.get("random_memory", {}).get("mean_margin", float("inf"))
    if not correct:
        return "FAIL"
    correct_mean = correct.get("mean_margin", float("-inf"))
    correct_lo = correct.get("ci_lo", correct_mean)
    if math.isnan(correct_lo):
        correct_lo = correct_mean
    control_means: list[float] = []
    for name in _CONTROL_NAMES:
        m = by_v.get(name, {}).get("mean_margin")
        if m is not None and not math.isnan(m):
            control_means.append(m)
    if not control_means:
        return "FAIL"
    max_control_mean = max(control_means)
    if correct_mean <= base_mean:
        return "FAIL"
    if correct_lo > max_control_mean:
        return "PASS_STRONG"
    if correct_mean > max_control_mean:
        return "PASS_DIRECTIONAL"
    if correct_mean > random_mean:
        return "STABILIZER_ONLY"
    return "FAIL"


def _phase_b_verdict(top: dict[str, float]) -> str:
    """Back-compat point-estimate verdict used by older callers/tests."""
    correct = top.get("correct_memory", float("-inf"))
    base = top.get("base_model", float("inf"))
    random_memory = top.get("random_memory", float("inf"))
    gap = top.get("gap", float("-inf"))
    if gap > 0 and correct > base and correct > random_memory:
        return "PASS_DIRECTIONAL"
    if correct > base and correct > random_memory:
        return "STABILIZER_ONLY"
    return "FAIL"


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
    phase_b_configs = []
    config_summaries: dict[str, list[dict[str, Any]]] = {}
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
            row = {
                "phase": phase,
                "config": cfg_dir.name,
                "gap": _gap(summary),
                "verdict": _verdict_from_summary(summary),
            }
            for s in summary:
                row[s["variant"]] = s["mean_margin"]
            phase_rows.append(row)
            all_configs.append(row)
            config_summaries[f"{phase}/{cfg_dir.name}"] = summary
            if phase == "phase_b":
                phase_b_configs.append(row)
        _write_csv(phase_rows, out_dir / f"{phase}_comparison.csv")

    best = sorted(all_configs, key=lambda r: r.get("gap", float("-inf")), reverse=True)
    verdict_pool = phase_b_configs or all_configs
    verdict_best = sorted(verdict_pool, key=lambda r: r.get("gap", float("-inf")), reverse=True)
    if verdict_best:
        top_row = verdict_best[0]
        top_key = f"{top_row['phase']}/{top_row['config']}"
        verdict = _verdict_from_summary(config_summaries.get(top_key, []))
    else:
        verdict = "FAIL"
    (out_dir / "analysis.json").write_text(json.dumps({
        "configs": all_configs,
        "best": best[:3],
        "verdict_best": verdict_best[:1],
        "verdict": verdict,
    }, indent=2))
    (out_dir / "README.md").write_text(
        "# Exp11 RSM Analysis\n\n"
        f"**Verdict:** {verdict}\n\n"
        "Verdict ladder uses bootstrap CI on `margin`:\n"
        "- **PASS_STRONG**: correct.ci_lo > max(controls.mean) and correct > base.\n"
        "- **PASS_DIRECTIONAL**: correct.mean > all controls and correct > base.\n"
        "- **STABILIZER_ONLY**: correct > base and correct > random_memory.\n"
        "- **FAIL**: otherwise.\n\n"
        "Controls considered: random_memory, shuffled_layers, gate_off, gate_uniform.\n\n"
        "See `phase_a_comparison.csv`, `phase_b_comparison.csv`, and per-config summaries.\n"
    )
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
