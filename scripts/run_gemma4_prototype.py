#!/usr/bin/env python3
"""Run the first practical Gemma4 RCV-HC attention-memory prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rcvhc.engine.prototype import PrototypeRunConfig, run_attention_memory_prototype, write_prototype_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--input", default="examples/delta_attention_demo.txt")
    parser.add_argument("--store", default="runs/gemma4_attention_store")
    parser.add_argument("--report-dir", default="reports/cleanroom")
    parser.add_argument("--question", default="What is the secret code for unit XJQ-482?")
    parser.add_argument("--answer", default="tulip-91")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--memory-dim", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--layers", default="max_exposed")
    parser.add_argument("--alpha-scale", type=float, default=0.2)
    parser.add_argument("--gate-bias", type=float, default=-1.0)
    args = parser.parse_args()

    cfg = PrototypeRunConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        input_path=args.input,
        store_path=args.store,
        report_dir=args.report_dir,
        question=args.question,
        answer=args.answer,
        block_size=args.block_size,
        memory_dim=args.memory_dim,
        top_k=args.top_k,
        layers=args.layers,
        alpha_scale=args.alpha_scale,
        gate_bias=args.gate_bias,
    )
    summary = run_attention_memory_prototype(cfg)
    paths = write_prototype_report(summary, args.report_dir)
    print(json.dumps(paths | {"signal_status": summary["diagnosis"]["signal_status"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
