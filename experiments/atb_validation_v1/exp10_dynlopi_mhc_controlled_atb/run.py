"""Exp10 — Dynamic LOPI + mHC Controlled ATB.

Tests whether the full ATB control stack — Dynamic LOPI v3.4, mHC bank shielding,
and residual beta gating — can convert raw external-KV injection from a non-linear
alpha-sensitive perturbation into a controlled memory readout.

Five arms are tested in a two-phase protocol:

  A0_raw_atb          — raw baseline (no mHC, no LOPI)
  A1_mhc_only         — Exp8 exact control (mHC κ=0.25, no LOPI)
  A2_dynlopi_only     — LOPI-only ablation (no mHC)
  A3_mhc_dynlopi      — mHC + Dynamic LOPI
  A4_mhc_dynlopi_beta — mHC + Dynamic LOPI + beta=0.05

Phase A  → smoke grid: n=50, 3 alphas × 5 arms × 3 variants.
            Selects top-2 arms (by gap) and best alpha for Phase B.

Phase B  → confirm: n=200, all 5 variants, top-2 arms + A0 + A1 (deduplicated).

Run end-to-end:
  python run.py \\
    --model /path/to/Qwen3-4B-Instruct-2507 \\
    --out exp10_runs/run_$(date +%Y%m%d_%H%M%S)

See PREREG.md for pre-registration and pass/fail criteria.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.atb_validation_v1._lib import Variant
from experiments.atb_validation_v1._lib import multi_bank_runner
from experiments.atb_validation_v1._lib.aggregator import aggregate, append_to_global
from experiments.atb_validation_v1._lib.manifest import write_manifest

EXPERIMENT = "exp10_dynlopi_mhc_controlled_atb"

_ALL_VARIANTS = ["correct_bank", "shuffled_bank", "random_kv",
                 "correct_K_random_V", "random_K_correct_V"]
_SMOKE_VARIANTS = ["correct_bank", "random_kv", "random_K_correct_V"]

# Fixed LOPI v3.4 retained form config flags.
_LOPI_CFG = dict(
    lopi_enabled=True,
    lopi_orthogonal=False,
    lopi_gaussian=True,
    lopi_derivative=True,
    lopi_profile_mode="auto",
)


# ---------------------------------------------------------------------------
# Arm definitions

def _build_arms(alpha: float, kappa: float = 0.25) -> dict[str, list[Variant]]:
    """Return a dict mapping arm_name → list of Variant objects for that arm.

    The same arm definition is used for every variant name / perturbation.
    Callers filter by variant name.
    """
    common = dict(
        method="anb",
        alpha=alpha,
        bank_key_mode="pre_rope",
        value_scale_mode="auto_rms_cap",
    )
    perturbations: dict[str, dict] = {
        "correct_bank": {},
        "shuffled_bank": {"bank_perturbation": "shuffled"},
        "random_kv": {"bank_perturbation": "random_kv"},
        "correct_K_random_V": {"bank_perturbation": "random_V_only"},
        "random_K_correct_V": {"bank_perturbation": "random_K_only"},
    }

    arms: dict[str, dict] = {
        "A0_raw_atb": dict(
            mhc_shield=False, mhc_kappa=kappa,
            bank_merge_beta=1.0, **{k: False for k in _LOPI_CFG},
            description=f"Raw ATB baseline alpha={alpha}.",
        ),
        "A1_mhc_only": dict(
            mhc_shield=True, mhc_kappa=kappa,
            bank_merge_beta=1.0, **{k: False for k in _LOPI_CFG},
            description=f"mHC-only (Exp8 control) kappa={kappa} alpha={alpha}.",
        ),
        "A2_dynlopi_only": dict(
            mhc_shield=False, mhc_kappa=kappa,
            bank_merge_beta=1.0, **_LOPI_CFG,
            description=f"Dynamic LOPI only (no mHC) alpha={alpha}.",
        ),
        "A3_mhc_dynlopi": dict(
            mhc_shield=True, mhc_kappa=kappa,
            bank_merge_beta=1.0, **_LOPI_CFG,
            description=f"mHC + Dynamic LOPI kappa={kappa} alpha={alpha}.",
        ),
        "A4_mhc_dynlopi_beta": dict(
            mhc_shield=True, mhc_kappa=kappa,
            bank_merge_beta=0.05, **_LOPI_CFG,
            description=f"mHC + LOPI + beta=0.05 kappa={kappa} alpha={alpha}.",
        ),
    }

    result: dict[str, list[Variant]] = {}
    for arm_name, arm_cfg in arms.items():
        variants = []
        for vname, pert in perturbations.items():
            variants.append(Variant(name=vname, **common, **arm_cfg, **pert))
        result[arm_name] = variants
    return result


# ---------------------------------------------------------------------------
# Multi-GPU dispatch helpers (identical pattern to exp9)

def _worker_run_one(kwargs: dict) -> tuple[str, str]:
    tag = kwargs.pop("_tag")
    result_path = multi_bank_runner.run(**kwargs)
    return tag, str(result_path)


def _dispatch(items: list[dict], devices: list[str]) -> list[tuple[str, str]]:
    if len(devices) == 1:
        results = []
        for item in items:
            item = dict(item)
            item["device"] = devices[0]
            results.append(_worker_run_one(item))
        return results
    ctx = multiprocessing.get_context("spawn")
    pools = [
        concurrent.futures.ProcessPoolExecutor(
            max_workers=1, max_tasks_per_child=1, mp_context=ctx
        )
        for _ in devices
    ]
    try:
        futures: list[concurrent.futures.Future] = []
        for i, item in enumerate(items):
            item = dict(item)
            item["device"] = devices[i % len(devices)]
            futures.append(pools[i % len(devices)].submit(_worker_run_one, item))
        return [f.result() for f in concurrent.futures.as_completed(futures)]
    finally:
        for p in pools:
            p.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Scoring helpers

def _sha1_of_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def _read_variant_margins(out_dir: Path) -> dict[str, float]:
    csv_path = out_dir / "summary.csv"
    if not csv_path.exists():
        return {}
    margins: dict[str, float] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            v = row.get("variant", "")
            try:
                margins[v] = float(row["mean_margin"])
            except (KeyError, ValueError):
                pass
    return margins


def _gap_score(out_dir: Path) -> float:
    """correct_bank_margin − max(random_kv, random_K_correct_V)."""
    margins = _read_variant_margins(out_dir)
    correct = margins.get("correct_bank", float("-inf"))
    controls = [margins.get("random_kv", float("inf")),
                margins.get("random_K_correct_V", float("inf"))]
    controls = [c for c in controls if c != float("inf")]
    return correct - max(controls) if controls else correct


def _write_manifest(out_dir: Path, cf_path: Path, model: str, dtype: str,
                    seeds: list[int], variants: list[Variant],
                    bank_size: int, extra: dict) -> None:
    write_manifest(
        out_dir,
        experiment=EXPERIMENT,
        repo_root=ROOT,
        dataset_path=cf_path,
        dataset_sha1=_sha1_of_file(cf_path),
        model=model,
        dtype=dtype,
        attention_impl="eager",
        seeds=seeds,
        variants=[v.to_dict() for v in variants],
        write_template="Fact: {subject} {phrase} {target_new}.",
        read_template="prompt.format(subject)",
        enabled_modules=["AttnNativeBank", "mHC-shield", "Dynamic-LOPI-v3.4", "residual-beta-gate"],
        disabled_modules=["SCAR", "CAA"],
        extra={"bank_key_mode": "pre_rope", "bank_size": bank_size,
               "exp8_baseline": "experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope",
               "exp9_baseline": "experiments/atb_validation_v1/exp9_residual_gated_mhc_pre_rope",
               **extra},
    )


# ---------------------------------------------------------------------------
# Phase A — smoke grid

def _run_phase_a(args, cf_path: Path, seeds: list[int],
                 devices: list[str]) -> tuple[list[str], float]:
    """Run arm × alpha smoke.  Returns (top2_arm_names, best_alpha)."""
    print("\n=== Phase A: arm × alpha smoke (n=50, 3 alphas × 5 arms × 3 variants) ===")
    base_out = Path(args.out) / "phase_a"
    work_items: list[dict] = []
    item_meta: list[dict] = []

    alphas: list[float] = [float(a) for a in args.alpha_grid.split(",")]
    arm_order = ["A0_raw_atb", "A1_mhc_only", "A2_dynlopi_only", "A3_mhc_dynlopi",
                 "A4_mhc_dynlopi_beta"]

    for alpha in alphas:
        arms = _build_arms(alpha=alpha, kappa=args.kappa)
        for arm_name in arm_order:
            all_variants = arms[arm_name]
            smoke_variants = [v for v in all_variants if v.name in _SMOKE_VARIANTS]
            tag = f"{arm_name}_alpha_{alpha:.2f}".replace(".", "_")
            out_dir = base_out / tag
            _write_manifest(out_dir, cf_path, args.model, args.dtype, seeds,
                            smoke_variants, args.bank_size,
                            {"phase": "A", "arm": arm_name, "alpha": alpha,
                             "n_prompts": args.n_prompts_smoke})
            work_items.append({
                "_tag": tag,
                "model_name": args.model,
                "dtype": args.dtype,
                "counterfact_path": cf_path,
                "variants": smoke_variants,
                "seeds": seeds,
                "out_dir": out_dir,
                "bank_size": args.bank_size,
                "n_prompts": args.n_prompts_smoke,
            })
            item_meta.append({"arm": arm_name, "alpha": alpha,
                               "out_dir": out_dir, "tag": tag})

    tag_to_path = {tag: Path(p) for tag, p in _dispatch(work_items, devices)}

    # Score each (arm, alpha) cell.
    scores: list[dict] = []
    for meta in item_meta:
        tag, out_dir = meta["tag"], meta["out_dir"]
        arm_name, alpha = meta["arm"], meta["alpha"]
        aggregate(tag_to_path[tag], experiment=f"{EXPERIMENT}_a_{tag}",
                  model=args.model, dataset=cf_path.name, out_dir=out_dir)
        gap = _gap_score(out_dir)
        correct_m = _read_variant_margins(out_dir).get("correct_bank", float("nan"))
        print(f"  {arm_name} α={alpha:.2f}: correct={correct_m:.4f}  gap={gap:.4f}")
        scores.append({"arm": arm_name, "alpha": alpha, "gap": gap,
                       "out_dir": str(out_dir)})

    # For each arm, pick best alpha.
    from collections import defaultdict
    best_per_arm: dict[str, dict] = {}
    for s in scores:
        arm = s["arm"]
        if arm not in best_per_arm or s["gap"] > best_per_arm[arm]["gap"]:
            best_per_arm[arm] = s

    arm_ranking = sorted(best_per_arm.values(), key=lambda x: x["gap"], reverse=True)
    top2 = arm_ranking[:2]
    top2_names = [d["arm"] for d in top2]

    # Best alpha = best alpha of the top-1 arm.
    best_alpha = top2[0]["alpha"] if top2 else alphas[0]

    selection = {"top2_arms": top2, "best_alpha": best_alpha}
    sel_path = Path(args.out) / "phase_a_selection.json"
    sel_path.write_text(json.dumps(selection, indent=2))
    print(f"\nTop-2 arms from Phase A: {top2_names}  best_alpha={best_alpha}")
    return top2_names, best_alpha


# ---------------------------------------------------------------------------
# Phase B — confirm

def _run_phase_b(args, cf_path: Path, seeds: list[int],
                 top2_arms: list[str], best_alpha: float,
                 devices: list[str]) -> None:
    """Run top2 + A0 + A1 with all 5 variants, n=200, best_alpha."""
    # Build arm set: top2 + mandatory anchors, deduplicated, order preserved.
    mandatory = ["A0_raw_atb", "A1_mhc_only"]
    phase_b_arms: list[str] = list(dict.fromkeys(top2_arms + mandatory))
    print(f"\n=== Phase B: confirm (n=200, arms={phase_b_arms}, alpha={best_alpha}) ===")
    base_out = Path(args.out) / "phase_b"
    work_items: list[dict] = []
    item_meta: list[dict] = []

    arms = _build_arms(alpha=best_alpha, kappa=args.kappa)
    for arm_name in phase_b_arms:
        all_variants = arms[arm_name]
        tag = f"{arm_name}"
        out_dir = base_out / tag
        _write_manifest(out_dir, cf_path, args.model, args.dtype, seeds,
                        all_variants, args.bank_size,
                        {"phase": "B", "arm": arm_name, "alpha": best_alpha,
                         "n_prompts": args.n_prompts_confirm})
        work_items.append({
            "_tag": tag,
            "model_name": args.model,
            "dtype": args.dtype,
            "counterfact_path": cf_path,
            "variants": all_variants,
            "seeds": seeds,
            "out_dir": out_dir,
            "bank_size": args.bank_size,
            "n_prompts": args.n_prompts_confirm,
        })
        item_meta.append({"arm": arm_name, "out_dir": out_dir, "tag": tag})

    tag_to_path = {tag: Path(p) for tag, p in _dispatch(work_items, devices)}

    results: list[dict] = []
    for meta in item_meta:
        tag, out_dir = meta["tag"], meta["out_dir"]
        arm_name = meta["arm"]
        aggregate(tag_to_path[tag], experiment=f"{EXPERIMENT}_b_{tag}",
                  model=args.model, dataset=cf_path.name, out_dir=out_dir)
        gap = _gap_score(out_dir)
        margins = _read_variant_margins(out_dir)
        correct_m = margins.get("correct_bank", float("nan"))
        rnd_m = margins.get("random_kv", float("nan"))
        print(f"  {arm_name}: correct={correct_m:.4f}  random_kv={rnd_m:.4f}  gap={gap:.4f}")
        results.append({"arm": arm_name, "gap": gap, "correct_bank": correct_m,
                        "random_kv": rnd_m})

    summary_path = Path(args.out) / "phase_b_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    # Determine verdict.
    arm_gaps = {r["arm"]: r["gap"] for r in results}
    a0_gap = arm_gaps.get("A0_raw_atb", float("-inf"))
    a1_gap = arm_gaps.get("A1_mhc_only", float("-inf"))
    controlled_gaps = {k: v for k, v in arm_gaps.items()
                       if k not in ("A0_raw_atb", "A1_mhc_only")}
    best_controlled = max(controlled_gaps.values(), default=float("-inf"))

    if best_controlled > a0_gap and best_controlled > a1_gap:
        verdict = "PASS_STRONG"
    elif best_controlled > a1_gap:
        verdict = "PASS_STABILIZER"
    else:
        verdict = "FAIL"

    print(f"\nPhase B verdict: {verdict}")
    verdict_path = Path(args.out) / "verdict.txt"
    verdict_path.write_text(f"{verdict}\n")

    # Per-arm summaries already appended to SUMMARY.csv via aggregate() in the loop above.


# ---------------------------------------------------------------------------
# CLI

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp10: Dynamic LOPI + mHC ATB")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda",
                   help="Single device, or comma-separated list for multi-GPU.")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--bank-size", type=int, default=200)
    p.add_argument("--kappa", type=float, default=0.25)
    p.add_argument("--alpha-grid", default="0.05,0.10,0.20",
                   help="Comma-separated alpha values for Phase A sweep.")
    p.add_argument("--n-prompts-smoke", type=int, default=50)
    p.add_argument("--n-prompts-confirm", type=int, default=200)
    p.add_argument("--cf-path",
                   default="experiments/datasets/counterfact_1k.jsonl")
    p.add_argument("--phase", default="AB",
                   help="Which phases to run: A, B, or AB (default: AB).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cf_path = ROOT / args.cf_path
    seeds = [int(s) for s in args.seeds.split(",")]
    devices = [d.strip() for d in args.device.split(",") if d.strip()]

    Path(args.out).mkdir(parents=True, exist_ok=True)

    top2_arms: list[str] = []
    best_alpha: float = float(args.alpha_grid.split(",")[0])

    if "A" in args.phase.upper():
        top2_arms, best_alpha = _run_phase_a(args, cf_path, seeds, devices)

    # Allow Phase B to be restarted from a prior Phase A selection.
    if "B" in args.phase.upper():
        if not top2_arms:
            sel_path = Path(args.out) / "phase_a_selection.json"
            if sel_path.exists():
                sel = json.loads(sel_path.read_text())
                top2_arms = [d["arm"] for d in sel.get("top2_arms", [])]
                best_alpha = float(sel.get("best_alpha", best_alpha))
            else:
                print("WARNING: Phase A selection not found. Using A0+A1 only.")
                top2_arms = []
        _run_phase_b(args, cf_path, seeds, top2_arms, best_alpha, devices)

    print("\nExp10 complete.")


if __name__ == "__main__":
    main()
