"""Exp8 — mHC-Smoothed Pre-RoPE Negative Controls.

Repeats Exp7 (Qwen3-4B, pre_rope, bank_size=200, 5 variants, seeds 0-1-2)
with the mHC spectral shield enabled at tunable kappa.

Phases
------
Phase A — smoke (n=200): sweep kappas [1.0, 0.5, 0.25] at alpha=0.05.
Phase B — full  (n=807): best-kappa from Phase A.
Phase C — alpha stress: best-kappa, alphas [0.10, 0.20, 0.50, 1.00], n=200,
          3 key variants (correct_bank, random_kv, random_K_correct_V).

Each phase writes to its own sub-directory under --out.
Phase A → out/phase_a/kappa_X.XX/
Phase B → out/phase_b/
Phase C → out/phase_c/alpha_X.XX/

Run the full pipeline end-to-end:
  python run.py --model /path/to/Qwen3-4B-Instruct-2507 --out exp8_runs/run_001

Or run a single phase manually:
  python run.py ... --phase a --kappa 0.5   # specific kappa
  python run.py ... --phase b --kappa 0.5   # Phase B with forced kappa
  python run.py ... --phase c --kappa 0.5   # Phase C with forced kappa
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.atb_validation_v1._lib import Variant, sha1_of_file
from experiments.atb_validation_v1._lib import multi_bank_runner
from experiments.atb_validation_v1._lib.aggregator import aggregate, append_to_global
from experiments.atb_validation_v1._lib.manifest import write_manifest

EXPERIMENT = "exp8_mhc_smoothed_pre_rope"

_PHASE_A_KAPPAS = [1.0, 0.5, 0.25]
_ALL_VARIANTS = ["correct_bank", "shuffled_bank", "random_kv",
                 "correct_K_random_V", "random_K_correct_V"]
_STRESS_VARIANTS = ["correct_bank", "random_kv", "random_K_correct_V"]
_PHASE_C_ALPHAS = [0.10, 0.20, 0.50, 1.00]


def make_variants(alpha: float, kappa: float,
                  names: list[str] | None = None) -> list[Variant]:
    base = dict(
        method="anb",
        alpha=alpha,
        bank_key_mode="pre_rope",
        value_scale_mode="auto_rms_cap",
        mhc_shield=True,
        mhc_kappa=kappa,
    )
    pool = [
        Variant(name="correct_bank", **base,
                description=f"Production ATB pre_rope + mHC kappa={kappa}."),
        Variant(name="shuffled_bank", **base, bank_perturbation="shuffled",
                description="Correct K (pre_rope), V rows permuted across facts; mHC on."),
        Variant(name="random_kv", **base, bank_perturbation="random_kv",
                description="K and V both Gaussian RMS-matched; mHC on."),
        Variant(name="correct_K_random_V", **base, bank_perturbation="random_V_only",
                description="Correct K, random V; mHC on."),
        Variant(name="random_K_correct_V", **base, bank_perturbation="random_K_only",
                description="Random K, correct V; mHC on."),
    ]
    if names is None:
        return pool
    return [v for v in pool if v.name in names]


def _write_manifest_for(out_dir: Path, cf_path: Path, model: str, dtype: str,
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
        enabled_modules=["AttnNativeBank", "mHC-shield"],
        disabled_modules=["SCAR", "CAA", "LOPI-skip-list"],
        extra={
            "bank_key_mode": "pre_rope",
            "bank_size": bank_size,
            "exp7_baseline": "experiments/atb_validation_v1/exp7_non_gemma_pre_rope",
            **extra,
        },
    )


def _run_phase_a(args, cf_path: Path, seeds: list[int]) -> float:
    """Run Phase A: kappa sweep at n=200. Return best kappa for Phase B/C."""
    print("\n=== Phase A: kappa sweep (smoke, n=200) ===")
    base_out = Path(args.out) / "phase_a"
    best_kappa = _PHASE_A_KAPPAS[0]
    best_margin = float("-inf")

    for kappa in _PHASE_A_KAPPAS:
        kappa_tag = f"kappa_{kappa:.2f}".replace(".", "_")
        out_dir = base_out / kappa_tag
        variants = make_variants(alpha=args.alpha, kappa=kappa)
        _write_manifest_for(out_dir, cf_path, args.model, args.dtype, seeds,
                            variants, args.bank_size,
                            {"phase": "A", "kappa": kappa, "n_prompts": 200})
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
        summary = aggregate(res, experiment=f"{EXPERIMENT}_phase_a_{kappa_tag}",
                            model=args.model, dataset=cf_path.name,
                            out_dir=out_dir)
        append_to_global(summary, ROOT / "experiments" / "atb_validation_v1" /
                         "SUMMARY.csv")
        # Score kappa as correct_bank_margin - max_control_margin (gap metric).
        score = _score_kappa(out_dir)
        correct_margin = _read_variant_margins(out_dir).get("correct_bank", float("-inf"))
        print(f"  kappa={kappa}: correct_bank={correct_margin:.4f}  gap={score:.4f}")
        if score > best_margin:
            best_margin = score
            best_kappa = kappa

    print(f"Best kappa from Phase A: {best_kappa} (gap={best_margin:.4f})")
    # Persist best kappa for downstream phases.
    best_path = Path(args.out) / "best_kappa.json"
    best_path.write_text(json.dumps({"best_kappa": best_kappa,
                                     "best_gap_score": best_margin}))
    return best_kappa


def _run_phase_b(args, cf_path: Path, seeds: list[int], kappa: float) -> None:
    """Run Phase B: full 807-prompt run with best kappa."""
    print(f"\n=== Phase B: full run (n=807, kappa={kappa}) ===")
    out_dir = Path(args.out) / "phase_b"
    variants = make_variants(alpha=args.alpha, kappa=kappa)
    _write_manifest_for(out_dir, cf_path, args.model, args.dtype, seeds,
                        variants, args.bank_size,
                        {"phase": "B", "kappa": kappa, "n_prompts": None})
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
                        model=args.model, dataset=cf_path.name,
                        out_dir=out_dir)
    append_to_global(summary, ROOT / "experiments" / "atb_validation_v1" /
                     "SUMMARY.csv")
    print(f"Phase B summary -> {summary}")


def _run_phase_c(args, cf_path: Path, seeds: list[int], kappa: float) -> None:
    """Run Phase C: high-alpha stress with best kappa."""
    print(f"\n=== Phase C: alpha stress (kappa={kappa}) ===")
    base_out = Path(args.out) / "phase_c"
    for alpha_c in _PHASE_C_ALPHAS:
        alpha_tag = f"alpha_{alpha_c:.2f}".replace(".", "_")
        out_dir = base_out / alpha_tag
        variants = make_variants(alpha=alpha_c, kappa=kappa,
                                 names=_STRESS_VARIANTS)
        _write_manifest_for(out_dir, cf_path, args.model, args.dtype, seeds,
                            variants, args.bank_size,
                            {"phase": "C", "kappa": kappa,
                             "alpha": alpha_c, "n_prompts": 200})
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
                            model=args.model, dataset=cf_path.name,
                            out_dir=out_dir)
        append_to_global(summary, ROOT / "experiments" / "atb_validation_v1" /
                         "SUMMARY.csv")
        print(f"  alpha={alpha_c}: summary -> {summary}")


def _read_variant_margins(out_dir: Path) -> dict[str, float]:
    """Read mean_margin per variant from summary.csv."""
    import csv
    csv_path = out_dir / "summary.csv"
    if not csv_path.exists():
        return {}
    margins: dict[str, float] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row.get("variant", "")
            try:
                margins[v] = float(row["mean_margin"])
            except (KeyError, ValueError):
                pass
    return margins


def _score_kappa(out_dir: Path) -> float:
    """Score a kappa run as: correct_bank_margin - max_control_margin.

    Positive score means correct_bank is best; score=0 means tied for first;
    negative means some control beats correct_bank.
    Maximising this is better than maximising correct_bank alone because it
    filters out kappas where all variants improve uniformly.
    """
    margins = _read_variant_margins(out_dir)
    correct = margins.get("correct_bank", float("-inf"))
    controls = [v for v in margins if v != "correct_bank"]
    if not controls:
        return correct
    max_control = max(margins[v] for v in controls)
    return correct - max_control


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Primary alpha (Phase A + B). Phase C sweeps its own alphas.")
    ap.add_argument("--bank-size", type=int, default=200)
    ap.add_argument("--out", required=True,
                    help="Root output directory. Subdirs per phase are created automatically.")
    ap.add_argument("--counterfact",
                    default=str(ROOT / "experiments" / "datasets" /
                                "counterfact_1k.jsonl"))
    ap.add_argument("--phase", choices=["a", "b", "c", "all"], default="all",
                    help="Which phase to run. 'all' runs A → B → C sequentially.")
    ap.add_argument("--kappa", type=float, default=None,
                    help="Force a specific kappa (skips Phase A selection for phases B/C).")
    args = ap.parse_args()

    cf_path = Path(args.counterfact)
    seeds = [int(s) for s in args.seeds.split(",")]
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.phase == "a" or args.phase == "all":
        best_kappa = _run_phase_a(args, cf_path, seeds)
    elif args.kappa is not None:
        best_kappa = args.kappa
    else:
        # Try reading from previous Phase A run.
        best_path = Path(args.out) / "best_kappa.json"
        if best_path.exists():
            best_kappa = json.loads(best_path.read_text())["best_kappa"]
            print(f"Loaded best_kappa={best_kappa} from {best_path}")
        else:
            raise RuntimeError(
                "No Phase A results found. Run with --phase all or --kappa <value>."
            )

    if args.phase == "b" or args.phase == "all":
        _run_phase_b(args, cf_path, seeds, best_kappa)

    if args.phase == "c" or args.phase == "all":
        _run_phase_c(args, cf_path, seeds, best_kappa)

    print("\nExp8 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
