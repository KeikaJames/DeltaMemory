"""Exp 5 — α dense sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.atb_validation_v1._lib import sha1_of_file
from experiments.atb_validation_v1._lib import alpha_sweep
from experiments.atb_validation_v1._lib.aggregator import aggregate, append_to_global
from experiments.atb_validation_v1._lib.manifest import write_manifest

DEFAULT_ALPHAS = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                  0.40, 0.50, 0.75, 1.00, 1.50, 2.00]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Spec pins Gemma-4-31B-it for the paper.")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--bank-size", type=int, default=200)
    ap.add_argument("--alphas",
                    default=",".join(f"{a:.2f}" for a in DEFAULT_ALPHAS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--counterfact",
                    default=str(ROOT / "experiments" / "datasets" /
                                "counterfact_1k.jsonl"))
    args = ap.parse_args()

    cf_path = Path(args.counterfact)
    out_dir = Path(args.out)
    seeds = [int(s) for s in args.seeds.split(",")]
    alphas = [float(a) for a in args.alphas.split(",")]

    write_manifest(
        out_dir,
        experiment="exp5_alpha_sweep",
        repo_root=ROOT,
        dataset_path=cf_path,
        dataset_sha1=sha1_of_file(cf_path),
        model=args.model,
        dtype=args.dtype,
        attention_impl="eager",
        seeds=seeds,
        variants=[{"alpha": a, "method": "anb",
                   "bank_key_mode": "pre_rope",
                   "value_scale_mode": "auto_rms_cap"} for a in alphas],
        write_template="Fact: {subject} {phrase} {target_new}.",
        read_template="prompt.format(subject) [target row only]",
        enabled_modules=["AttnNativeBank"],
        disabled_modules=["SCAR", "CAA", "LOPI-skip-list"],
        extra={"alphas": alphas, "bank_size": args.bank_size,
               "target_index": 0},
    )
    res = alpha_sweep.run(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        counterfact_path=cf_path,
        alphas=alphas,
        seeds=seeds,
        bank_size=args.bank_size,
        out_dir=out_dir,
    )
    summary = aggregate(res, experiment="exp5_alpha_sweep",
                        model=args.model, dataset=cf_path.name,
                        out_dir=out_dir)
    append_to_global(summary, ROOT / "experiments" / "atb_validation_v1" /
                     "SUMMARY.csv")
    print(f"summary -> {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
