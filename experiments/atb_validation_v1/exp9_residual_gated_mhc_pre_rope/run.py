"""Exp9 — Residual-Gated mHC AttnNativeBank.

Tests whether legacy residual-style output gating restores correct-bank selectivity
in AttnNativeBank (ATB), after Exp7/Exp8 showed V-path dominance with raw merged-softmax.

Two modes are compared at a beta grid {0.05, 0.10, 0.20, 0.50, 1.00}:
  merged_beta_mhc : keep merged-softmax structure + beta * out_bank + mHC
  sep_beta_mhc    : separate-softmax + beta * out_bank + mHC

Phases (all run sequentially, script does NOT stop between phases)
------------------------------------------------------------------
Phase A1 → out/phase_a1/mode_MODE_beta_BETA/
  smoke: n=100, 3 key variants × 5 betas × 2 modes = 9000 cells
  select top 2 (mode, beta) configs by A1 gap metric.

Phase A2 → out/phase_a2/config_N/
  smoke: n=100, all 5 variants × top 2 configs = 3000 cells
  select 1 best config for Phase B/C.

Phase B → out/phase_b/
  full: n=807, all 5 variants, best config = 12105 cells.

Phase C → out/phase_c/alpha_X_XX/
  stress: n=200, alphas [0.10, 0.20, 0.50, 1.00], 3 variants = 7200 cells.

Run end-to-end (recommended):
  python run.py \\
    --model /path/to/Qwen3-4B-Instruct-2507 \\
    --out exp9_runs/run_$(date +%Y%m%d_%H%M%S)

Restart from saved best config (after A1+A2 done):
  python run.py ... --phase B --best-config <path/to/best_config.json>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.atb_validation_v1._lib import Variant, sha1_of_file
from experiments.atb_validation_v1._lib import multi_bank_runner
from experiments.atb_validation_v1._lib.aggregator import aggregate, append_to_global
from experiments.atb_validation_v1._lib.manifest import write_manifest

EXPERIMENT = "exp9_residual_gated_mhc_pre_rope"

_BETAS = [0.05, 0.10, 0.20, 0.50, 1.00]
_MODES = ["merged_beta_mhc", "sep_beta_mhc"]
_ALL_VARIANTS = ["correct_bank", "shuffled_bank", "random_kv",
                 "correct_K_random_V", "random_K_correct_V"]
_A1_VARIANTS = ["correct_bank", "random_kv", "random_K_correct_V"]
_STRESS_VARIANTS = ["correct_bank", "random_kv", "random_K_correct_V"]
_PHASE_C_ALPHAS = [0.10, 0.20, 0.50, 1.00]


# ---------------------------------------------------------------------------
# Variant factory

def make_variants(
    alpha: float,
    kappa: float,
    beta: float,
    mode: str,
    names: list[str] | None = None,
) -> list[Variant]:
    """Build variant list for a given (alpha, kappa, beta, mode) config."""
    if mode == "merged_beta_mhc":
        bank_separate_softmax = False
    elif mode == "sep_beta_mhc":
        bank_separate_softmax = True
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    base = dict(
        method="anb",
        alpha=alpha,
        bank_key_mode="pre_rope",
        value_scale_mode="auto_rms_cap",
        mhc_shield=True,
        mhc_kappa=kappa,
        bank_separate_softmax=bank_separate_softmax,
        bank_merge_beta=beta,
    )
    pool = [
        Variant(name="correct_bank", **base,
                description=f"ATB pre_rope mode={mode} beta={beta} kappa={kappa}."),
        Variant(name="shuffled_bank", **base, bank_perturbation="shuffled",
                description=f"Correct K, V rows permuted; mode={mode} beta={beta}."),
        Variant(name="random_kv", **base, bank_perturbation="random_kv",
                description=f"K+V Gaussian RMS-matched; mode={mode} beta={beta}."),
        Variant(name="correct_K_random_V", **base, bank_perturbation="random_V_only",
                description=f"Correct K, random V; mode={mode} beta={beta}."),
        Variant(name="random_K_correct_V", **base, bank_perturbation="random_K_only",
                description=f"Random K, correct V; mode={mode} beta={beta}."),
    ]
    if names is None:
        return pool
    return [v for v in pool if v.name in names]


# ---------------------------------------------------------------------------
# Shared helpers

def _write_manifest(out_dir: Path, cf_path: Path, model: str, dtype: str,
                    seeds: list[int], variants: list[Variant],
                    bank_size: int, extra: dict) -> None:
    write_manifest(
        out_dir,
        experiment=EXPERIMENT,
        repo_root=ROOT,
        dataset_path=cf_path,
        dataset_sha1=sha1_of_file(cf_path),
        model=model,
        dtype=dtype,
        attention_impl="eager",
        seeds=seeds,
        variants=[v.to_dict() for v in variants],
        write_template="Fact: {subject} {phrase} {target_new}.",
        read_template="prompt.format(subject)",
        enabled_modules=["AttnNativeBank", "mHC-shield", "residual-beta-gate"],
        disabled_modules=["SCAR", "CAA", "LOPI-skip-list"],
        extra={
            "bank_key_mode": "pre_rope",
            "bank_size": bank_size,
            "exp8_baseline": "experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope",
            **extra,
        },
    )


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


def _gap_score(out_dir: Path, control_names: list[str] | None = None) -> float:
    """correct_bank_margin - max_control_margin. Higher is better."""
    margins = _read_variant_margins(out_dir)
    correct = margins.get("correct_bank", float("-inf"))
    if control_names is None:
        control_names = [v for v in margins if v != "correct_bank"]
    controls = [margins[v] for v in control_names if v in margins]
    if not controls:
        return correct
    return correct - max(controls)


# ---------------------------------------------------------------------------
# Phase A1

def _run_phase_a1(args, cf_path: Path, seeds: list[int]) -> list[dict]:
    """Beta × mode grid smoke with 3 key variants, n=100. Returns scored configs."""
    print("\n=== Phase A1: beta × mode grid smoke (n=100, 3 variants) ===")
    base_out = Path(args.out) / "phase_a1"
    results: list[dict] = []

    for mode in _MODES:
        for beta in _BETAS:
            tag = f"mode_{mode}_beta_{beta:.2f}".replace(".", "_")
            out_dir = base_out / tag
            variants = make_variants(alpha=args.alpha, kappa=args.kappa,
                                     beta=beta, mode=mode, names=_A1_VARIANTS)
            _write_manifest(out_dir, cf_path, args.model, args.dtype, seeds,
                            variants, args.bank_size,
                            {"phase": "A1", "mode": mode, "beta": beta,
                             "kappa": args.kappa, "n_prompts": args.n_prompts_smoke})
            res = multi_bank_runner.run(
                model_name=args.model,
                dtype=args.dtype,
                device=args.device,
                counterfact_path=cf_path,
                variants=variants,
                seeds=seeds,
                out_dir=out_dir,
                bank_size=args.bank_size,
                n_prompts=args.n_prompts_smoke,
            )
            aggregate(res, experiment=f"{EXPERIMENT}_a1_{tag}",
                      model=args.model, dataset=cf_path.name, out_dir=out_dir)
            # A1 gap uses only the 3 key control variants
            score = _gap_score(out_dir, control_names=["random_kv", "random_K_correct_V"])
            margins = _read_variant_margins(out_dir)
            correct_m = margins.get("correct_bank", float("nan"))
            print(f"  mode={mode} beta={beta}: correct={correct_m:.4f}  A1-gap={score:.4f}")
            results.append({"mode": mode, "beta": beta, "gap_a1": score,
                            "out_dir": str(out_dir)})

    # Select top 2 by A1 gap score.
    results.sort(key=lambda x: x["gap_a1"], reverse=True)
    top2 = results[:2]
    selection_path = Path(args.out) / "phase_a1_selection.json"
    selection_path.write_text(json.dumps(top2, indent=2))
    print(f"\nTop 2 configs from A1:")
    for cfg in top2:
        print(f"  mode={cfg['mode']} beta={cfg['beta']:.2f} gap={cfg['gap_a1']:.4f}")
    return top2


# ---------------------------------------------------------------------------
# Phase A2

def _run_phase_a2(args, cf_path: Path, seeds: list[int],
                  top2_configs: list[dict]) -> dict:
    """Full 5-variant smoke for top 2 A1 configs. Returns best config."""
    print("\n=== Phase A2: full controls smoke (n=100, all 5 variants) ===")
    base_out = Path(args.out) / "phase_a2"
    results: list[dict] = []

    for i, cfg in enumerate(top2_configs):
        mode = cfg["mode"]
        beta = cfg["beta"]
        tag = f"config_{i}_mode_{mode}_beta_{beta:.2f}".replace(".", "_")
        out_dir = base_out / tag
        variants = make_variants(alpha=args.alpha, kappa=args.kappa,
                                 beta=beta, mode=mode)
        _write_manifest(out_dir, cf_path, args.model, args.dtype, seeds,
                        variants, args.bank_size,
                        {"phase": "A2", "mode": mode, "beta": beta,
                         "kappa": args.kappa, "n_prompts": args.n_prompts_smoke})
        res = multi_bank_runner.run(
            model_name=args.model,
            dtype=args.dtype,
            device=args.device,
            counterfact_path=cf_path,
            variants=variants,
            seeds=seeds,
            out_dir=out_dir,
            bank_size=args.bank_size,
            n_prompts=args.n_prompts_smoke,
        )
        aggregate(res, experiment=f"{EXPERIMENT}_a2_{tag}",
                  model=args.model, dataset=cf_path.name, out_dir=out_dir)
        score = _gap_score(out_dir)  # all 5 variants
        margins = _read_variant_margins(out_dir)
        correct_m = margins.get("correct_bank", float("nan"))
        print(f"  config {i} (mode={mode} beta={beta}): correct={correct_m:.4f} A2-gap={score:.4f}")
        results.append({"mode": mode, "beta": beta, "gap_a2": score,
                        "out_dir": str(out_dir)})

    results.sort(key=lambda x: x["gap_a2"], reverse=True)
    best = results[0]
    best_path = Path(args.out) / "best_config.json"
    best_path.write_text(json.dumps(best, indent=2))
    print(f"\nBest config for Phase B/C: mode={best['mode']} beta={best['beta']:.2f} "
          f"gap={best['gap_a2']:.4f}")
    return best


# ---------------------------------------------------------------------------
# Phase B

def _run_phase_b(args, cf_path: Path, seeds: list[int], best: dict) -> None:
    """Full 807-prompt validation with best (mode, beta)."""
    mode, beta = best["mode"], best["beta"]
    print(f"\n=== Phase B: full validation (n=807, mode={mode}, beta={beta}) ===")
    out_dir = Path(args.out) / "phase_b"
    variants = make_variants(alpha=args.alpha, kappa=args.kappa,
                             beta=beta, mode=mode)
    _write_manifest(out_dir, cf_path, args.model, args.dtype, seeds,
                    variants, args.bank_size,
                    {"phase": "B", "mode": mode, "beta": beta,
                     "kappa": args.kappa, "n_prompts": None})
    res = multi_bank_runner.run(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        counterfact_path=cf_path,
        variants=variants,
        seeds=seeds,
        out_dir=out_dir,
        bank_size=args.bank_size,
        n_prompts=None,
    )
    summary = aggregate(res, experiment=f"{EXPERIMENT}_phase_b",
                        model=args.model, dataset=cf_path.name, out_dir=out_dir)
    append_to_global(summary, ROOT / "experiments" / "atb_validation_v1" / "SUMMARY.csv")
    gap = _gap_score(out_dir)
    print(f"Phase B gap = {gap:.4f}  summary -> {out_dir / 'summary.csv'}")


# ---------------------------------------------------------------------------
# Phase C

def _run_phase_c(args, cf_path: Path, seeds: list[int], best: dict) -> None:
    """High-alpha stress: alphas 0.10/0.20/0.50/1.00, 3 variants, n=200."""
    mode, beta = best["mode"], best["beta"]
    print(f"\n=== Phase C: alpha stress (mode={mode}, beta={beta}) ===")
    base_out = Path(args.out) / "phase_c"

    for alpha_c in _PHASE_C_ALPHAS:
        alpha_tag = f"alpha_{alpha_c:.2f}".replace(".", "_")
        out_dir = base_out / alpha_tag
        variants = make_variants(alpha=alpha_c, kappa=args.kappa,
                                 beta=beta, mode=mode, names=_STRESS_VARIANTS)
        _write_manifest(out_dir, cf_path, args.model, args.dtype, seeds,
                        variants, args.bank_size,
                        {"phase": "C", "mode": mode, "beta": beta,
                         "kappa": args.kappa, "alpha": alpha_c, "n_prompts": 200})
        res = multi_bank_runner.run(
            model_name=args.model,
            dtype=args.dtype,
            device=args.device,
            counterfact_path=cf_path,
            variants=variants,
            seeds=seeds,
            out_dir=out_dir,
            bank_size=args.bank_size,
            n_prompts=200,
        )
        summary = aggregate(res, experiment=f"{EXPERIMENT}_phase_c_{alpha_tag}",
                            model=args.model, dataset=cf_path.name, out_dir=out_dir)
        append_to_global(summary, ROOT / "experiments" / "atb_validation_v1" / "SUMMARY.csv")
        gap = _gap_score(out_dir, control_names=["random_kv", "random_K_correct_V"])
        print(f"  alpha={alpha_c}: gap={gap:.4f}")


# ---------------------------------------------------------------------------
# CLI

def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Primary alpha (A1, A2, B). Phase C sweeps its own alphas.")
    ap.add_argument("--kappa", type=float, default=0.25,
                    help="mHC kappa (fixed from Exp8 Phase A).")
    ap.add_argument("--bank-size", type=int, default=200)
    ap.add_argument("--n-prompts-smoke", type=int, default=100,
                    help="Prompts per config in Phase A1/A2.")
    ap.add_argument("--out", required=True,
                    help="Root output directory. Phase subdirs created automatically.")
    ap.add_argument("--counterfact",
                    default=str(ROOT / "experiments" / "datasets" /
                                "counterfact_1k.jsonl"))
    ap.add_argument("--phase",
                    choices=["A1", "A2", "B", "C", "all"], default="all",
                    help="Phase to run. 'all' runs A1 → A2 → B → C sequentially.")
    ap.add_argument("--best-config", default=None,
                    help="Path to best_config.json (skips A1/A2 for phases B/C).")
    args = ap.parse_args()

    cf_path = Path(args.counterfact)
    seeds = [int(s) for s in args.seeds.split(",")]
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Determine best config if provided externally.
    best_config: dict | None = None
    if args.best_config:
        best_config = json.loads(Path(args.best_config).read_text())
    else:
        best_path = Path(args.out) / "best_config.json"
        if best_path.exists():
            best_config = json.loads(best_path.read_text())
            print(f"Loaded best_config from {best_path}")

    run_all = args.phase == "all"

    # --- Phase A1 ---
    if run_all or args.phase == "A1":
        top2 = _run_phase_a1(args, cf_path, seeds)
    else:
        a1_path = Path(args.out) / "phase_a1_selection.json"
        if a1_path.exists():
            top2 = json.loads(a1_path.read_text())
        else:
            raise RuntimeError(
                "Phase A1 selection not found. Run with --phase all or --phase A1 first."
            )

    # --- Phase A2 ---
    if run_all or args.phase == "A2":
        best_config = _run_phase_a2(args, cf_path, seeds, top2)

    if best_config is None:
        raise RuntimeError(
            "best_config not available. Run A1+A2 first or supply --best-config."
        )

    # --- Phase B ---
    if run_all or args.phase == "B":
        _run_phase_b(args, cf_path, seeds, best_config)

    # --- Phase C ---
    if run_all or args.phase == "C":
        _run_phase_c(args, cf_path, seeds, best_config)

    print("\nExp9 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
