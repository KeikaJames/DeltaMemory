"""Exp10 analysis: gap metric and JS-drift comparison across arms.

Usage:
  python analyze.py --run-dir exp10_runs/run_20260509 [--out-dir .]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _mean(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def _summarize_phase(phase_dir: Path) -> list[dict]:
    """Summarize each arm sub-directory in a phase dir."""
    summaries = []
    for arm_dir in sorted(phase_dir.iterdir()):
        results_path = arm_dir / "results.jsonl"
        if not results_path.exists():
            continue
        rows = _load_jsonl(results_path)
        arm_name = arm_dir.name

        by_variant: dict[str, list[float]] = defaultdict(list)
        drift_by_arm: list[float] = []
        lopi_enabled = None

        for r in rows:
            vname = r.get("variant", "")
            margin = r.get("margin")
            if margin is not None:
                by_variant[vname].append(float(margin))
            js = r.get("js_drift")
            if js is not None:
                drift_by_arm.append(float(js))
            if lopi_enabled is None:
                lopi_enabled = r.get("lopi_enabled", False)

        mean_margins = {v: _mean(ms) for v, ms in by_variant.items()}
        correct_m = mean_margins.get("correct_bank", float("-inf"))
        controls = {v: m for v, m in mean_margins.items()
                    if v in ("random_kv", "random_K_correct_V",
                             "shuffled_bank", "correct_K_random_V")}
        max_control = max(controls.values(), default=float("inf"))
        gap = correct_m - max_control
        mean_drift = _mean(drift_by_arm)

        summaries.append({
            "arm": arm_name,
            "correct_bank": correct_m,
            "random_kv": mean_margins.get("random_kv", float("nan")),
            "shuffled_bank": mean_margins.get("shuffled_bank", float("nan")),
            "correct_K_random_V": mean_margins.get("correct_K_random_V", float("nan")),
            "random_K_correct_V": mean_margins.get("random_K_correct_V", float("nan")),
            "gap": gap,
            "mean_js_drift": mean_drift,
            "lopi_enabled": lopi_enabled,
            "n_rows": len(rows),
        })

    return summaries


def _print_table(summaries: list[dict], title: str) -> None:
    print(f"\n=== {title} ===")
    header = ["arm", "correct_bank", "random_kv", "gap", "js_drift", "LOPI"]
    print(f"  {'arm':<30}  {'correct':>8}  {'rnd_kv':>8}  {'gap':>8}  {'drift':>8}  {'LOPI'}")
    print("  " + "-" * 75)
    for s in sorted(summaries, key=lambda x: x.get("gap", float("-inf")), reverse=True):
        print(f"  {s['arm']:<30}  {s['correct_bank']:>8.4f}  {s['random_kv']:>8.4f}"
              f"  {s['gap']:>8.4f}  {s['mean_js_drift']:>8.4f}  {s.get('lopi_enabled', '?')!s:>5}")


def _save_csv(summaries: list[dict], path: Path) -> None:
    if not summaries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    print(f"Saved: {path}")


def _analyze_qualitative(run_dir: Path) -> None:
    """Summarize qualitative hit rates."""
    qual_dirs = list(run_dir.glob("*/qual_results.jsonl")) + \
                list(run_dir.glob("qual_results.jsonl"))
    for qpath in qual_dirs:
        rows = _load_jsonl(qpath)
        by_config: dict[str, dict] = defaultdict(lambda: {
            "n": 0, "tnew_hit": 0, "ttrue_hit": 0,
            "off_topic_leak": 0, "prompt_echo": 0,
        })
        for r in rows:
            cid = r.get("config_id", "?")
            by_config[cid]["n"] += 1
            by_config[cid]["tnew_hit"] += int(r.get("target_new_hit", False))
            by_config[cid]["ttrue_hit"] += int(r.get("target_true_hit", False))
            by_config[cid]["off_topic_leak"] += int(r.get("off_topic_leak", False))
            by_config[cid]["prompt_echo"] += int(r.get("prompt_echo", False))

        print(f"\n=== Qualitative: {qpath.parent.name} ===")
        print(f"  {'config':<25}  {'n':>4}  {'tnew%':>6}  {'ttrue%':>6}  "
              f"{'offleak%':>8}  {'echo%':>6}")
        print("  " + "-" * 60)
        for cid in ["no_bank", "alpha0", "raw", "mhc_dynlopi", "mhc_dynlopi_beta"]:
            s = by_config.get(cid)
            if s is None:
                continue
            n = s["n"] or 1
            print(f"  {cid:<25}  {s['n']:>4}  "
                  f"{100*s['tnew_hit']/n:>6.1f}  "
                  f"{100*s['ttrue_hit']/n:>6.1f}  "
                  f"{100*s['off_topic_leak']/n:>8.1f}  "
                  f"{100*s['prompt_echo']/n:>6.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Exp10 analysis")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "analysis"

    # Phase A
    phase_a_dir = run_dir / "phase_a"
    if phase_a_dir.exists():
        sums_a = _summarize_phase(phase_a_dir)
        _print_table(sums_a, "Phase A — Smoke")
        _save_csv(sums_a, out_dir / "phase_a_summary.csv")

    # Phase B
    phase_b_dir = run_dir / "phase_b"
    if phase_b_dir.exists():
        sums_b = _summarize_phase(phase_b_dir)
        _print_table(sums_b, "Phase B — Confirm")
        _save_csv(sums_b, out_dir / "phase_b_summary.csv")

        # Print verdict.
        verdict_path = run_dir / "verdict.txt"
        if verdict_path.exists():
            verdict = verdict_path.read_text().strip()
            print(f"\n>>> Phase B Verdict: {verdict} <<<")

    # Qualitative
    _analyze_qualitative(run_dir)


if __name__ == "__main__":
    main()
