"""Authenticity contract checker.

Enforces docs/authenticity.md against every experiments/<phase>/ that
has produced a result artifact.

Usage::

    python tools/check_authenticity.py                 # standard
    python tools/check_authenticity.py --strict        # fail grandfathered dirs too
    python tools/check_authenticity.py --paths A B     # check only listed paths
    python tools/check_authenticity.py --bit-equality  # also enforce alpha=0 witness

Exit codes:
    0 - all checks passed
    1 - one or more violations
    2 - usage error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_ENV_FIELDS = (
    "commit",
    "dirty",
    "dirty_diff_sha1",
    "prereg_version",
    "dataset_sha1",
    "torch",
    "transformers",
    "python",
    "device",
    "dtype",
    "started_at",
    "host",
)

RESULT_FILE_GLOBS = (
    "cells.jsonl",
    "cells_smoke.jsonl",
    "cells_*.jsonl",
    "summary.json",
    "pareto.json",
    "verdicts.json",
)

GRANDFATHERED_DIRS = frozenset(
    {
        "experiments/W4_caa_baseline",
        "experiments/W6_counter_prior",
        "experiments/A1_rope_audit",
        "experiments/A3_layer_weighting",
        "experiments/A4_gate_metric",
        "experiments/A5_profiler_n100",
        "experiments/W1_mhc_localize",
        "experiments/W2_lopi_dissect",
        "experiments/W3_decision",
        "experiments/W5_moe",
        "experiments/W7_longctx",
        "experiments/W8_multifact",
        "experiments/W9_multiturn",
        "experiments/W10_vs_baselines",
        "experiments/W12_full_ablation",
        "experiments/W13_synthetic",
        "experiments/W14_benchmark",
        "experiments/W_T3_6_ecor_e2e",
        "experiments/W_T3_6_ecor_op",
        "experiments/W_T3_6_ulopi_profiler",
        "experiments/W_T3_lopi_fixes",
        "experiments/X3_caa_smoke",
        "experiments/scar_smoke",
        "experiments/bench",
        "experiments/datasets",
    }
)


class Violation(Exception):
    pass


def _result_files(d: Path) -> list[Path]:
    out: list[Path] = []
    for pattern in RESULT_FILE_GLOBS:
        out.extend(sorted(d.glob(pattern)))
    return out


def _has_results(d: Path) -> bool:
    return bool(_result_files(d))


def _check_env_json(d: Path) -> list[str]:
    env_path = d / "env.json"
    if not env_path.exists():
        return [f"{d.relative_to(REPO_ROOT)}: missing env.json"]
    try:
        env = json.loads(env_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{d.relative_to(REPO_ROOT)}: env.json not valid JSON: {exc}"]
    missing = [f for f in REQUIRED_ENV_FIELDS if f not in env]
    if missing:
        return [
            f"{d.relative_to(REPO_ROOT)}: env.json missing fields: {sorted(missing)}"
        ]
    return []


def _check_aggregate_has_cells(d: Path) -> list[str]:
    has_summary = (d / "summary.json").exists()
    has_cells = bool(list(d.glob("cells*.jsonl")))
    if has_summary and not has_cells:
        return [
            f"{d.relative_to(REPO_ROOT)}: summary.json present but no cells*.jsonl alongside (aggregate-only forbidden)"
        ]
    return []


def _check_bit_equality_witness(d: Path) -> list[str]:
    """For each cells file: every (model, method) with alpha>0 rows must have an alpha=0 witness with drift==0.0."""
    violations: list[str] = []
    for cf in d.glob("cells*.jsonl"):
        seen: dict[tuple[str, str], dict[str, bool]] = {}
        try:
            for line in cf.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                model = row.get("model")
                method = row.get("method")
                alpha = row.get("alpha")
                if model is None or method is None or alpha is None:
                    continue
                key = (str(model), str(method))
                slot = seen.setdefault(key, {"alpha0_zero_drift": False, "has_alpha_pos": False})
                if float(alpha) == 0.0:
                    drift = row.get("drift")
                    if drift is not None and float(drift) == 0.0:
                        slot["alpha0_zero_drift"] = True
                else:
                    slot["has_alpha_pos"] = True
        except (json.JSONDecodeError, ValueError) as exc:
            violations.append(f"{cf.relative_to(REPO_ROOT)}: parse error: {exc}")
            continue
        for (model, method), slot in seen.items():
            if slot["has_alpha_pos"] and not slot["alpha0_zero_drift"]:
                violations.append(
                    f"{cf.relative_to(REPO_ROOT)}: ({model},{method}) has alpha>0 rows but no alpha=0 drift==0 witness"
                )
    return violations


def walk_experiments(roots: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for sub in sorted(root.rglob("*")):
            if sub.is_dir() and _has_results(sub):
                out.append(sub)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Mneme authenticity contract checker")
    p.add_argument("--strict", action="store_true", help="check grandfathered dirs too")
    p.add_argument("--bit-equality", action="store_true", help="also enforce alpha=0 witness")
    p.add_argument(
        "--paths",
        nargs="*",
        help="check only these dirs (relative to repo root)",
    )
    args = p.parse_args(argv)

    if args.paths:
        targets = [REPO_ROOT / pth for pth in args.paths]
    else:
        targets = walk_experiments([REPO_ROOT / "experiments"])

    violations: list[str] = []
    skipped = 0
    for d in targets:
        rel = str(d.relative_to(REPO_ROOT))
        is_grandfathered = any(
            rel == g or rel.startswith(g + "/") for g in GRANDFATHERED_DIRS
        )
        if is_grandfathered and not args.strict:
            skipped += 1
            continue
        violations.extend(_check_env_json(d))
        violations.extend(_check_aggregate_has_cells(d))
        if args.bit_equality:
            violations.extend(_check_bit_equality_witness(d))

    if violations:
        print("AUTHENTICITY CHECK: FAIL")
        for v in violations:
            print(f"  - {v}")
        print(f"\nchecked {len(targets)} dir(s), {skipped} grandfathered, {len(violations)} violation(s)")
        return 1

    print(
        f"AUTHENTICITY CHECK: PASS ({len(targets)} dir(s) checked, "
        f"{skipped} grandfathered)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
