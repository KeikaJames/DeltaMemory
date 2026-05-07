"""Exp 6b — post-RoPE Negative Controls.

Rerun of Exp 6 with bank_key_mode="post_rope" instead of "pre_rope".
Motivation: pre_rope is a near-no-op on Gemma-4-31B (native v_norm makes
auto_rms_cap produce scale≈1.0).  post_rope is the only mode confirmed to
produce positive margin on Gemma-4-31B (Exp 1: +0.088 mean margin).

5 variants on identical CF prompts:
  * correct_bank          — production ATB
  * shuffled_bank         — fact↔target labels permuted within bank
  * random_kv             — Gaussian K & V, per-layer RMS-matched
  * correct_K_random_V    — correct K, random V
  * random_K_correct_V    — random K, correct V
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.atb_validation_v1._lib import Variant, sha1_of_file
from experiments.atb_validation_v1._lib import cf_runner
from experiments.atb_validation_v1._lib.aggregator import aggregate, append_to_global
from experiments.atb_validation_v1._lib.manifest import write_manifest


def make_variants(alpha: float = 1.0) -> list[Variant]:
    # post_rope: confirmed to produce positive margin on Gemma-4-31B (Exp 1)
    base = dict(method="anb", alpha=alpha, bank_key_mode="post_rope",
                value_scale_mode="auto_rms_cap")
    return [
        Variant(name="correct_bank", **base,
                description="Production ATB with post_rope; reference."),
        Variant(name="shuffled_bank", **base, bank_perturbation="shuffled",
                description="Fact↔target labels permuted within bank."),
        Variant(name="random_kv", **base, bank_perturbation="random_kv",
                description="K and V both Gaussian, RMS-matched per layer."),
        Variant(name="correct_K_random_V", **base,
                bank_perturbation="random_V_only",
                description="Correct K (post_rope), random V."),
        Variant(name="random_K_correct_V", **base,
                bank_perturbation="random_K_only",
                description="Random K, correct V (post_rope)."),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alpha", type=float, default=1.0)
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
        experiment="exp6b_post_rope_negative_controls",
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
            "bank_key_mode": "post_rope",
            "motivation": (
                "Exp 6 used pre_rope which is near-no-op on Gemma-4-31B. "
                "This rerun uses post_rope, the only mode with confirmed "
                "positive margin on Gemma-4-31B (Exp 1: +0.088 mean)."
            ),
            "perturbation_protocol": (
                "Random K/V draws use "
                "torch.Generator('cpu').manual_seed(0xC0FFEE); "
                "per-layer RMS-matched to true bank."
            ),
            "n_prompts": args.n_prompts,
        },
    )
    res = cf_runner.run(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        counterfact_path=cf_path,
        variants=variants,
        seeds=seeds,
        out_dir=out_dir,
        n_prompts=args.n_prompts,
    )
    summary = aggregate(res, experiment="exp6b_post_rope_negative_controls",
                        model=args.model, dataset=cf_path.name,
                        out_dir=out_dir)
    append_to_global(summary, ROOT / "experiments" / "atb_validation_v1" /
                     "SUMMARY.csv")
    print(f"summary -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
