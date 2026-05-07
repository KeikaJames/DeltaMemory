"""Exp 4 — CounterFact-1k Main Result + Protocol.

Three rows:
  * none α=0
  * AttnNativeBank α=0
  * AttnNativeBank α=1

Full ~807-prompt CF-1k (after W.6 filter), 3 seeds. McNemar on recall@1.
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
    Variant(name="none_alpha0", method="none", alpha=0.0,
            description="No bank, baseline reference."),
    Variant(name="AttnNativeBank_alpha0", method="anb", alpha=0.0,
            bank_key_mode="pre_rope", value_scale_mode="auto_rms_cap",
            description="ATB installed but α=0 (bit-equal redline)."),
    Variant(name="AttnNativeBank_alpha1", method="anb", alpha=1.0,
            bank_key_mode="pre_rope", value_scale_mode="auto_rms_cap",
            description="ATB at production α=1 (main result)."),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-prompts", type=int, default=None)
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
        experiment="exp4_cf1k_main",
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
        extra={
            "filter_rule": ("paraphrase_prompts present and target tokenizable "
                             "with distinct ≥3-alpha-token heads"),
            "sampled_size": 1000,
            "drift_protocol": "100 fixed neutral Wikitext-2 prompts",
            "stat_tests": ["paired bootstrap CI on Δrecall and Δmargin",
                            "McNemar χ² on recall@1"],
        },
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
    summary = aggregate(res, experiment="exp4_cf1k_main",
                        model=args.model, dataset=cf_path.name,
                        out_dir=out_dir)
    append_to_global(summary, ROOT / "experiments" / "atb_validation_v1" /
                     "SUMMARY.csv")
    print(f"summary -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
