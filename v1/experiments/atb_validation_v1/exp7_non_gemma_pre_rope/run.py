"""Exp 7 — Non-Gemma Pre-RoPE Negative Controls.

Tests whether pre-RoPE AttnNativeBank achieves correct K/V binding on a model
without Gemma-4 native V-norm.

Model: Qwen3-4B-Instruct-2507 (no native V-norm; standard GQA).
bank_key_mode: pre_rope (position-invariant K-addressing).
bank_size: 200 (1 target + 199 distractors; makes shuffled a valid control).

5 variants:
  * correct_bank          — production ATB (pre_rope K + V)
  * shuffled_bank         — K/V rows permuted across facts
  * random_kv             — Gaussian K & V, per-layer RMS-matched
  * correct_K_random_V    — correct pre_rope K, random V
  * random_K_correct_V    — random K, correct V

Success criterion: correct_bank has the highest mean_margin AND better/comparable
target_rank compared to all controls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.atb_validation_v1._lib import Variant, sha1_of_file
from experiments.atb_validation_v1._lib import multi_bank_runner
from experiments.atb_validation_v1._lib.aggregator import aggregate, append_to_global
from experiments.atb_validation_v1._lib.manifest import write_manifest

EXPERIMENT = "exp7_non_gemma_pre_rope"


def make_variants(alpha: float = 0.05) -> list[Variant]:
    base = dict(
        method="anb",
        alpha=alpha,
        bank_key_mode="pre_rope",
        value_scale_mode="auto_rms_cap",
    )
    return [
        Variant(name="correct_bank", **base,
                description="Production ATB with pre_rope; reference."),
        Variant(name="shuffled_bank", **base, bank_perturbation="shuffled",
                description="K/V rows permuted across facts (bank_size=200, valid control)."),
        Variant(name="random_kv", **base, bank_perturbation="random_kv",
                description="K and V both Gaussian, RMS-matched per layer."),
        Variant(name="correct_K_random_V", **base, bank_perturbation="random_V_only",
                description="Correct K (pre_rope), random V."),
        Variant(name="random_K_correct_V", **base, bank_perturbation="random_K_only",
                description="Random K, correct V (pre_rope)."),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--bank-size", type=int, default=200)
    ap.add_argument("--n-prompts", type=int, default=None,
                    help="Truncate eligible prompts (smoke/dry-run only).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--counterfact",
                    default=str(ROOT / "experiments" / "datasets" /
                                "counterfact_1k.jsonl"))
    args = ap.parse_args()

    cf_path = Path(args.counterfact)
    out_dir = Path(args.out)
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = make_variants(alpha=args.alpha)

    write_manifest(
        out_dir,
        experiment=EXPERIMENT,
        repo_root=ROOT,
        dataset_path=cf_path,
        dataset_sha1=sha1_of_file(cf_path),
        model=args.model,
        dtype=args.dtype,
        attention_impl="eager",
        seeds=seeds,
        variants=[v.to_dict() for v in variants],
        write_template="Fact: {subject} {phrase} {target_new}.",
        read_template="prompt.format(subject)",
        enabled_modules=["AttnNativeBank"],
        disabled_modules=["SCAR", "CAA", "LOPI-skip-list"],
        extra={
            "bank_key_mode": "pre_rope",
            "bank_size": args.bank_size,
            "motivation": (
                "Exp 6 (pre_rope, Gemma-4-31B): V-norm makes injection near-no-op. "
                "Exp 6b (post_rope, Gemma-4-31B): post-RoPE K is position-specific. "
                "Exp 7 uses Qwen3-4B (no V-norm) + pre_rope + bank_size=200 "
                "(shuffled is a valid 200-row permutation)."
            ),
            "perturbation_protocol": (
                "Random K/V draws use "
                "torch.Generator('cpu').manual_seed(0xC0FFEE ^ seed); "
                "per-layer RMS-matched to true bank."
            ),
            "distractor_protocol": (
                "For each (seed, prompt): sample bank_size-1 distractors using "
                "random.Random(seed ^ hash(prompt_id) & 0xFFFFFFFF); "
                "base_bank written once and cloned per variant."
            ),
            "n_prompts": args.n_prompts,
        },
    )

    res = multi_bank_runner.run(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        counterfact_path=cf_path,
        variants=variants,
        seeds=seeds,
        out_dir=out_dir,
        bank_size=args.bank_size,
        n_prompts=args.n_prompts,
    )

    summary = aggregate(res, experiment=EXPERIMENT,
                        model=args.model, dataset=cf_path.name,
                        out_dir=out_dir)
    append_to_global(summary, ROOT / "experiments" / "atb_validation_v1" /
                     "SUMMARY.csv")
    print(f"summary -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
