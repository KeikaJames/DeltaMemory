"""Exp 3 — α=0 bit-equality probe (single model entry).

Run separately for each model in {Gemma-4-E2B, Qwen3.6-4B, GLM-4-9B}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.atb_validation_v1._lib import sha1_of_file
from experiments.atb_validation_v1._lib import bit_equal_probe
from experiments.atb_validation_v1._lib.aggregator import aggregate, append_to_global
from experiments.atb_validation_v1._lib.manifest import write_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--n-facts", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--counterfact",
                    default=str(ROOT / "experiments" / "datasets" /
                                "counterfact_1k.jsonl"))
    args = ap.parse_args()

    cf_path = Path(args.counterfact)
    out_dir = Path(args.out)

    write_manifest(
        out_dir,
        experiment="exp3_bit_equal",
        repo_root=ROOT,
        dataset_path=cf_path,
        dataset_sha1=sha1_of_file(cf_path),
        model=args.model,
        dtype=args.dtype,
        attention_impl="eager",
        seeds=[args.seed],
        variants=[{"name": "anb_alpha0", "method": "anb", "alpha": 0.0,
                   "bank_key_mode": "pre_rope",
                   "value_scale_mode": "auto_rms_cap"}],
        write_template="Fact: {subject} {phrase} {target_new}.",
        read_template="100 fixed neutral Wikitext-2 sentences",
        enabled_modules=["AttnNativeBank"],
        disabled_modules=["SCAR", "CAA", "LOPI-skip-list"],
        extra={"n_prompts": args.n_prompts, "n_facts": args.n_facts,
               "comparison": "torch.equal of next-token logits"},
    )
    res = bit_equal_probe.run(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        counterfact_path=cf_path,
        out_dir=out_dir,
        n_prompts=args.n_prompts,
        n_facts=args.n_facts,
        seed=args.seed,
    )
    summary = aggregate(res, experiment="exp3_bit_equal",
                        model=args.model, dataset="neutral_100",
                        out_dir=out_dir)
    append_to_global(summary, ROOT / "experiments" / "atb_validation_v1" /
                     "SUMMARY.csv")
    print(f"summary -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
