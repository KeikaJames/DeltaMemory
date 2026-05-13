"""Exp 1 — Core Ablation.

5 variants on the same dataset+model:
  * no_bank
  * post_rope_bank
  * pre_rope_bank_only   (V-scale=none, SCAR=off)
  * pre_rope_vscale      (V-scale=auto_rms_cap, SCAR=off)
  * full_attnnativebank  (production default; SCAR explicitly off)
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

VARIANTS = [
    Variant(name="no_bank", method="none", alpha=0.0,
            description="Unpatched baseline."),
    Variant(name="post_rope_bank", method="anb", alpha=1.0,
            bank_key_mode="post_rope", value_scale_mode="auto_rms_cap",
            description="ATB with post-RoPE K capture and q_post scoring."),
    Variant(name="pre_rope_bank_only", method="anb", alpha=1.0,
            bank_key_mode="pre_rope", value_scale_mode="none",
            description="Pre-RoPE bank, no V-scale, SCAR off."),
    Variant(name="pre_rope_vscale", method="anb", alpha=1.0,
            bank_key_mode="pre_rope", value_scale_mode="auto_rms_cap",
            description="Pre-RoPE bank + auto_rms_cap V-scale, SCAR off."),
    Variant(name="full_attnnativebank", method="anb", alpha=1.0,
            bank_key_mode="pre_rope", value_scale_mode="auto_rms_cap",
            description="Production ATB default, SCAR explicitly OFF."),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-prompts", type=int, default=None,
                    help="dry-run subset size; default = all eligible")
    ap.add_argument("--out", required=True)
    ap.add_argument("--counterfact",
                    default=str(ROOT / "experiments" / "datasets" /
                                "counterfact_1k.jsonl"))
    args = ap.parse_args()

    cf_path = Path(args.counterfact)
    out_dir = Path(args.out)
    seeds = [int(s) for s in args.seeds.split(",")]

    write_manifest(
        out_dir,
        experiment="exp1_core_ablation",
        repo_root=ROOT,
        dataset_path=cf_path,
        dataset_sha1=sha1_of_file(cf_path),
        model=args.model,
        dtype=args.dtype,
        attention_impl="eager",
        seeds=seeds,
        variants=[v.to_dict() for v in VARIANTS],
        write_template="Fact: {subject} {phrase} {target_new}.",
        read_template="prompt.format(subject)",
        enabled_modules=["AttnNativeBank"],
        disabled_modules=["SCAR", "CAA", "LOPI-skip-list"],
        extra={"n_prompts": args.n_prompts},
    )

    res = cf_runner.run(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        counterfact_path=cf_path,
        variants=VARIANTS,
        seeds=seeds,
        out_dir=out_dir,
        n_prompts=args.n_prompts,
    )
    summary = aggregate(res, experiment="exp1_core_ablation",
                        model=args.model, dataset=cf_path.name,
                        out_dir=out_dir)
    append_to_global(summary, ROOT / "experiments" / "atb_validation_v1" /
                     "SUMMARY.csv")
    print(f"summary -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
